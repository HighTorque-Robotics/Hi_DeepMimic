import torch
from mimic_real.agents.vec_env import VecEnv
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.scene import InteractiveScene
from isaaclab.assets.articulation import Articulation
from mimic_real.utils.env_utils.scene import SceneCfg
import numpy as np
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp.events import randomize_rigid_body_material, randomize_rigid_body_mass, reset_joints_by_scale, reset_root_state_uniform, push_by_setting_velocity

from isaaclab.managers import RewardManager
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
from .base_env_config import BaseEnvCfg
from mimic_real.envs.motion_loader.motion_loader import MotionLoader

from mimic_real.utils.env_utils.marker import define_sphere_markers, define_cylinder_markers, visualize_cylinder

class BaseEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, headless):
        self.cfg: BaseEnvCfg
        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(
                gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        
        self.local_root_marker_name = "waist_link" # TODO write into config
        self.use_local_capture_points = False # TODO write into config
        
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        self.motion_loader = MotionLoader(self.cfg.motion_data.motion_file_path, self.robot.joint_names, self.robot.body_names, device=self.device, add_static_frame=False)
        print('motion_file_path==============',self.cfg.motion_data.motion_file_path)

        self.capture_points_body_names = self.motion_loader.capture_points_link_names # capture_points的link的名称
        self.capture_points_body_asset_cfg = SceneEntityCfg("robot", body_names=self.capture_points_body_names, preserve_order = True) # preserve_order important here
        self.capture_points_body_asset_cfg.resolve(self.scene)
        self.capture_points_body_ids = self.capture_points_body_asset_cfg.body_ids

        self.capture_points_root_body_names = [self.local_root_marker_name]
        self.capture_points_root_body_asset_cfg = SceneEntityCfg("robot", body_names=self.capture_points_root_body_names, preserve_order = True)
        self.capture_points_root_body_asset_cfg.resolve(self.scene)
        self.capture_points_root_body_ids = self.capture_points_root_body_asset_cfg.body_ids

        self.base_link_body_asset_cfg = SceneEntityCfg("robot", body_names=self.cfg.robot.base_link_body_names, preserve_order = True)
        self.base_link_body_asset_cfg.resolve(self.scene)
        self.base_link_body_ids = self.base_link_body_asset_cfg.body_ids

        print(self.robot.joint_names)
        print(self.robot.body_names)
        # print(self.capture_points_body_names)

        if not self.headless:
            self.sphere_marker = define_sphere_markers(radius=0.02)
            # self.cylinder_marker = define_cylinder_markers()

        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.apply_domain_random_at_start(env_ids)
        self.reset(env_ids)

    def init_buffers(self):
        self.extras = {}
        if self.cfg.motion_data.cycle_motion:
            self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        else:
            self.max_episode_length_s = self.motion_loader.record_time
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        print("max_episode_length:", self.max_episode_length)

        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.normalization.action_scale
        self.action_buffer = DelayBuffer(self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device)
        self.action_buffer.compute((torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)))
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(low=self.cfg.domain_rand.action_delay.params["min_delay"], high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1, size=(self.num_envs,), dtype=torch.int, device=self.device,)
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.terminate.terminate_contacts_body_names)
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
  
        self.gravity_vec = torch.zeros((self.num_envs, 3)).to(self.device)
        self.gravity_vec[:, 2] = -1.0
        self.gravity_vec_feet = torch.zeros((self.num_envs, 2, 3)).to(self.device)
        self.gravity_vec_feet[:, :, 2] = -1.0
        self.forward_vec_feet = torch.zeros((self.num_envs, 2, 3)).to(self.device)
        self.forward_vec_feet[:, :, 0] = 1.0 

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype = torch.float)
        self.debug_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.termination_buf = torch.zeros(self.num_envs, dtype = torch.bool, device = self.device, requires_grad=False)
        self.init_obs_buffer()

    
    def apply_domain_random_at_start(self, env_ids):
        if self.cfg.domain_rand.randomize_robot_friction.enable:
            self.cfg.domain_rand.randomize_robot_friction.params["asset_cfg"] = (self.robot_cfg)
            rand_rb_material = randomize_rigid_body_material(self.cfg.domain_rand.randomize_robot_friction, self)
            rand_rb_material(
                env=self,
                env_ids=env_ids,
                static_friction_range=self.cfg.domain_rand.randomize_robot_friction.params["static_friction_range"],
                dynamic_friction_range=self.cfg.domain_rand.randomize_robot_friction.params["dynamic_friction_range"],
                restitution_range=self.cfg.domain_rand.randomize_robot_friction.params["restitution_range"],
                num_buckets=self.cfg.domain_rand.randomize_robot_friction.params["num_buckets"],
                asset_cfg=self.robot_cfg
            )

        if self.cfg.domain_rand.add_rigid_body_mass.enable:
            robot_cfg = SceneEntityCfg(name="robot", body_names=self.cfg.domain_rand.add_rigid_body_mass.params["body_names"])
            robot_cfg.resolve(self.scene)
            randomize_rigid_body_mass(
                env=self,
                env_ids=env_ids,
                asset_cfg=robot_cfg,
                mass_distribution_params=self.cfg.domain_rand.add_rigid_body_mass.params["mass_distribution_params"],
                operation=self.cfg.domain_rand.add_rigid_body_mass.params["operation"],
            )


    def compute_current_observations(self):
        phase2pi = (self.phase * 2.0 * 3.1415926).unsqueeze(-1)

        desired_joint_pos = self.motion_loader.get_dof_pos_batch(self.phase)

        base_quat = self.robot.data.body_quat_w[:, self.base_link_body_ids, :].squeeze()
        base_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.base_link_body_ids, :].squeeze()
        base_lin_vel_b = math_utils.quat_rotate_inverse(base_quat, base_lin_vel_w)
        base_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.base_link_body_ids, :].squeeze()
        base_ang_vel_b = math_utils.quat_rotate_inverse(base_quat, base_ang_vel_w)
        base_projected_gravity = math_utils.quat_rotate_inverse(base_quat, self.gravity_vec)
        joint_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel = self.robot.data.joint_vel - self.robot.data.default_joint_vel
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        joint_pos_error = desired_joint_pos - self.robot.data.joint_pos

        # if self.cfg.motion_data.cycle_motion:
        #     pass
        # else:
        #     desired_capture_points_pos = self.motion_loader.get_capture_points_batch(self.phase)
        # desired_capture_points_pos = desired_capture_points_pos.view(self.num_envs, -1)
        # capture_points_pos = self.robot.data.body_pos_w[:, self.capture_points_body_ids, :] - self.scene.env_origins.unsqueeze(1)
        # capture_points_pos = capture_points_pos.view(self.num_envs, -1)
        # desired_capture_points_error = desired_capture_points_pos \
        #                              - capture_points_pos
        if self.use_local_capture_points:
            desired_capture_points_error = self.local_capture_points_error().view(self.num_envs, -1)
        else:
            desired_capture_points_error = self.global_capture_points_error().view(self.num_envs, -1)

        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        last_action = self.last_action 
        # self.action_buffer._circular_buffer.buffer[:, -1, :] # TODO 哪一个是正确的？
        current_actor_obs = torch.cat([
            base_ang_vel_b * self.obs_scales.ang_vel,
            base_projected_gravity * self.obs_scales.projected_gravity,
            joint_pos * self.obs_scales.joint_pos,
            joint_vel * self.obs_scales.joint_vel,
            last_action * self.obs_scales.actions,
            # base_quat,
            torch.sin(phase2pi),
            torch.cos(phase2pi),
            torch.sin(2 * phase2pi),
            torch.cos(2 * phase2pi),
            torch.sin(4 * phase2pi),
            torch.cos(4 * phase2pi),
            # self.debug_buf.unsqueeze(-1)
        ], dim=-1)
        # print(base_projected_gravity[0])
        current_critic_obs = torch.cat([
            base_lin_vel_b * self.obs_scales.lin_vel,
            base_ang_vel_b * self.obs_scales.ang_vel,
            base_projected_gravity * self.obs_scales.projected_gravity,
            joint_pos * self.obs_scales.joint_pos,
            joint_vel * self.obs_scales.joint_vel,
            last_action * self.obs_scales.actions,
            base_quat,
            torch.sin(phase2pi),
            torch.cos(phase2pi),
            torch.sin(2 * phase2pi),
            torch.cos(2 * phase2pi),
            torch.sin(4 * phase2pi),
            torch.cos(4 * phase2pi),
            joint_pos_error * self.obs_scales.joint_pos_error,
            desired_capture_points_error * self.obs_scales.capture_points_error, # TODO it is world frame now
            feet_contact,
        ], dim=-1)
        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)
        return actor_obs, critic_obs

    def reset(self, env_ids): # 不是纯随机reset state，在轨迹上某一点随机
        if len(env_ids) == 0:
            return
        self.extras["log"] = dict()

        self.scene.reset(env_ids)
        # ---------------- reset episode length ---------------------
        self.episode_length_buf[env_ids] = 0
        # self.episode_length_buf[env_ids] = torch.randint_like(self.episode_length_buf[env_ids], low = 0, high = int(self.max_episode_length.item())) # TODO add curriculum
        self.phase[env_ids] = (self.episode_length_buf[env_ids] / self.max_episode_length)
        
        # ---------------- reset joint state ------------------------
        ref_joint_pos = self.motion_loader.get_dof_pos_batch(self.phase[env_ids])
        ref_joint_pos += math_utils.sample_uniform(*self.cfg.domain_rand.reset_robot_joints.params["position_range"], ref_joint_pos.shape, self.device)
        if self.cfg.motion_data.use_dof_vel_data:
            # ref_joint_vel = self.robot.data.default_joint_vel[env_ids] # TODO reset joint vel
            ref_joint_vel = self.motion_loader.get_dof_vel_batch(self.phase[env_ids])
        else:
            ref_joint_vel = self.robot.data.default_joint_vel[env_ids]

        ref_joint_vel += math_utils.sample_uniform(*self.cfg.domain_rand.reset_robot_joints.params["velocity_range"], ref_joint_vel.shape, self.device)
        self.robot.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, env_ids=env_ids) 
        # ---------------- reset root state -------------------------
        # positions = root_states[:, 0:3] + self.scene.env_origins[env_ids]
        # orientations = root_states[:, 3:7]
        # velocities = root_states[:, 7:13]
        ref_positions = (self.motion_loader.get_root_trans_batch(self.phase[env_ids]) + self.scene.env_origins[env_ids]).to(torch.float32)
        ref_orientations = self.motion_loader.get_root_rot_batch(self.phase[env_ids]).to(torch.float32)
        pose_range = self.cfg.domain_rand.reset_robot_base.params["pose_range"]
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        positions_delta = rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        ref_positions += positions_delta
        # import ipdb; ipdb.set_trace();
        ref_orientations = math_utils.quat_mul(ref_orientations, orientations_delta)

        if self.cfg.motion_data.use_body_vel_data:
            # root_states = self.robot.data.default_root_state[env_ids].clone()
            # ref_velocities = root_states[:, 7:13]  # TODO reset body velocity
            root_states = self.robot.data.default_root_state[env_ids].clone()
            ref_velocities = root_states[:, 7:13] 
            ref_velocities[:, 0:3] = self.motion_loader.get_root_vel_batch(self.phase[env_ids])
            ref_velocities[:, 3:6] = self.motion_loader.get_root_omega_batch(self.phase[env_ids])
        else:
            root_states = self.robot.data.default_root_state[env_ids].clone()
            ref_velocities = root_states[:, 7:13] 
        velocity_range = self.cfg.domain_rand.reset_robot_base.params["velocity_range"]
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        ref_velocities += rand_samples

        self.robot.write_root_pose_to_sim(torch.cat([ref_positions, ref_orientations], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(ref_velocities, env_ids=env_ids)
        # -----------------------------------------------------------

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras['log'].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        # ----------------- reset observation action buffer --------- 
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.last_action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.action_buffer.reset(env_ids)
        # ----------------- -----------------------------------------
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        self.pre_step_callback(actions)
        debug = False
        if not debug:
            delayed_actions = self.action_buffer.compute(actions)
            cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
            processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos

            for _ in range(self.cfg.sim.decimation):
                self.robot.set_joint_position_target(processed_actions + self.motion_loader.get_dof_pos_batch(self.phase)) 
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
        else:
            joint_pos = self.motion_loader.get_dof_pos_batch(self.phase)
            self.robot.write_joint_state_to_sim(joint_pos, \
                                        self.robot.data.default_joint_vel)
            positions = self.robot.data.default_root_state[:, 0:3] + self.scene.env_origins
            orientations = self.robot.data.default_root_state[:, 3:7]
            velocities = self.robot.data.default_root_state[:, 7:13]

            positions = self.motion_loader.get_root_trans_batch(self.phase) + self.scene.env_origins
            orientations = self.motion_loader.get_root_rot_batch(self.phase)

            self.robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1)) #, env_ids=env_ids)
            self.robot.write_root_velocity_to_sim(velocities)#, env_ids=env_ids) # 设定 root state
            self.scene.update(dt = self.step_dt)

        if not self.headless:
            if self.use_local_capture_points:
                desired_capture_points = self.motion_loader.get_local_capture_points_batch(self.phase) \
                    + self.robot.data.body_pos_w[:, self.capture_points_root_body_ids, :]
            else:
                desired_capture_points = self.motion_loader.get_capture_points_batch(self.phase) + self.scene.env_origins.unsqueeze(1)
            
            marker_locations = desired_capture_points
            marker_indices = torch.zeros((desired_capture_points.shape[0], desired_capture_points.shape[1]))
            marker_indices[:, self.capture_points_body_names.index("head_link")] = 1 # 头红色
            marker_indices[:, self.capture_points_body_names.index("left_hand_link")] = 2
            marker_indices[:, self.capture_points_body_names.index("right_hand_link")] = 2 #手蓝色
            self.sphere_marker.visualize(marker_locations.view(-1, 3), marker_indices=marker_indices.view(-1)) 
            # capture_points = self.robot.data.body_pos_w[:, self.capture_points_body_ids, :][0] [显示距离]
            # visualize_cylinder(self.cylinder_marker, 
            #                 capture_points.view(-1, 3), 
            #                 desired_capture_points.view(-1, 3),  
            #                 device = self.device,
            #                 arrow_thickness= 0.07)
            self.sim.render()

        self.episode_length_buf += 1
        self.phase = (self.episode_length_buf / self.max_episode_length)

        reward_buf = self.reward_manager.compute(self.step_dt)
        self.post_step_callback(actions)

        self.reset_buf, self.time_out_buf = self.check_reset()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def show_motion(self):
        joint_pos = self.motion_loader.get_dof_pos_batch(self.phase)
        self.robot.write_joint_state_to_sim(joint_pos, \
                                       self.robot.data.default_joint_vel)
        # self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, \
        #                                self.robot.data.default_joint_vel) # , env_ids=env_ids) #设定关节角
        positions = self.robot.data.default_root_state[:, 0:3] + self.scene.env_origins
        orientations = self.robot.data.default_root_state[:, 3:7]
        velocities = self.robot.data.default_root_state[:, 7:13]

        positions = self.motion_loader.get_root_trans_batch(self.phase) + self.scene.env_origins
        orientations = self.motion_loader.get_root_rot_batch(self.phase)

        self.robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1)) #, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(velocities)#, env_ids=env_ids) # 设定 root state
        self.scene.update(dt = self.step_dt)

        # marker_locations = torch.zeros((2, 3))
        # marker_locations[0][2] = 1.0
        # marker_locations[1][0] = 1.0
        # marker_locations[1][1] = 1.0
        # marker_indices = torch.zeros(2)

        desired_capture_points = self.motion_loader.get_capture_points_batch(self.phase)
        # desired_capture_points = self.motion_loader.get_body_pos_batch(self.phase)
        marker_locations = desired_capture_points
        marker_indices = torch.zeros(desired_capture_points.shape[1])
        marker_indices[self.capture_points_body_names.index("head_link")] = 1 # 头红色
        marker_indices[self.capture_points_body_names.index("left_hand_link")] = 2
        marker_indices[self.capture_points_body_names.index("right_hand_link")] = 2 #手蓝色

        capture_points = self.robot.data.body_pos_w[:, self.capture_points_body_ids, :][0] 
        # marker_locations = capture_points
        # marker_indices = torch.zeros(capture_points.shape[0])
        # marker_indices[self.capture_points_body_names.index("head_link")] = 1 # 头红色
        # marker_indices[self.capture_points_body_names.index("left_hand_link")] = 2
        # marker_indices[self.capture_points_body_names.index("right_hand_link")] = 2 #手蓝色

        self.sphere_marker.visualize(marker_locations.view(-1, 3), marker_indices=marker_indices)  # translation orientations scales marker_indices
        # visualize_cylinder(self.cylinder_marker, 
        #                 capture_points.view(-1, 3), 
        #                 desired_capture_points.view(-1, 3),  
        #                 device = self.device,
        #                 arrow_thickness= 0.07)
        self.sim.render()

        self.episode_length_buf += 1
        self.phase = (self.episode_length_buf / self.max_episode_length)
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[time_out_buf] = 0

    def post_step_callback(self, action):
        self.last_action[:] = action[:]
        if self.cfg.domain_rand.push_robot.enable:
            env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.push_robot.push_interval_s / self.step_dt) == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) != 0:
                push_by_setting_velocity(env=self, env_ids=env_ids, velocity_range=self.cfg.domain_rand.push_robot.params["velocity_range"], asset_cfg=self.robot_cfg,)


    def pre_step_callback(self, action):
        self.action[:] = action[:]

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.termination_buf = torch.zeros(self.num_envs, dtype = torch.bool, device = self.device, requires_grad=False)

        if self.cfg.terminate.terminate_contacts:
            reset_buf |= torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1,), dim=1,)[0] > 1.0, dim=1) # termination when contact
        if self.cfg.terminate.terminate_capture_points_far:
            if self.use_local_capture_points:
                reset_buf |= (self.local_capture_points_error_sum() > self.cfg.terminate.capture_points_distance_threshold)
            else:
                reset_buf |= (self.global_capture_points_error_sum() > self.cfg.terminate.capture_points_distance_threshold)
        if self.cfg.terminate.terminate_dof_pos_limit:
            pass
        if self.cfg.terminate.terminate_dof_vel_limit:
            pass
        if self.cfg.terminate.terminate_non_flat_contact:
            pass

        self.termination_buf[:] = reset_buf[:]
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            print("actor observation size:", actor_obs.shape[0])
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_level = self.cfg.noise.noise_level

            noise_vec[:3] = noise_level * noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_level * noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6: 6 + self.num_actions] = noise_level * noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[6 + self.num_actions: 6 + 2 * self.num_actions] = noise_level * noise_scales.joint_vel * self.obs_scales.joint_vel
            self.noise_scale_vec = noise_vec

        self.actor_obs_buffer = CircularBuffer(max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device)
        self.critic_obs_buffer = CircularBuffer(max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device)

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras
    
    def global_capture_points_error(self):
        desired_capture_points_pos = self.motion_loader.get_capture_points_batch(self.phase)
        desired_capture_points_pos = desired_capture_points_pos.view(self.num_envs, -1)

        capture_points_pos = self.robot.data.body_pos_w[:, self.capture_points_body_ids, :] - self.scene.env_origins.unsqueeze(1)
        capture_points_pos = capture_points_pos.view(self.num_envs, -1)
        desired_capture_points_error = desired_capture_points_pos \
                                        - capture_points_pos
        return desired_capture_points_error

    def global_capture_points_error_sum(self):
        desired_capture_points_error = self.global_capture_points_error()
        error_sum = torch.sum(torch.square(desired_capture_points_error), dim=1) 
        return error_sum

    def local_capture_points_error(self):
        
        desired_capture_points_pos_local = self.motion_loader.get_local_capture_points_batch(self.phase, self.local_root_marker_name)
        desired_capture_points_pos_local = desired_capture_points_pos_local.view(self.num_envs, -1)

        capture_points_pos_local = self.robot.data.body_pos_w[:, self.capture_points_body_ids, :] \
                                 - self.robot.data.body_pos_w[:, self.capture_points_root_body_ids, :]
        capture_points_pos_local = capture_points_pos_local.view(self.num_envs, -1)

        desired_capture_points_error = desired_capture_points_pos_local \
                                        - capture_points_pos_local
        return desired_capture_points_error
    
    def local_capture_points_error_sum(self):
        desired_capture_points_error = self.local_capture_points_error()
        error_sum = torch.sum(torch.square(desired_capture_points_error), dim=1)
        return error_sum
    
    def noise_curriculum(self):
        pass # TODO

    def domain_rand_curriculum(self):
        pass # TODO



    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep
            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
