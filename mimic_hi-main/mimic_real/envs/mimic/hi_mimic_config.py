from .base_env_config import (  # noqa:F401
    BaseEnvCfg, BaseAgentCfg, BaseSceneCfg, RobotCfg,
    RewardCfg, PhysxCfg, SimCfg, MLPPolicyCfg, AlgorithmCfg
)
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from mimic_real.assets.usd.high_torque import HI_CFG
from mimic_real.data import WALK_MOTION_DATA_DIR, SIDE_FLIP_MOTION_DATA_DIR, RUN_MOTION_DATA_DIR, WAVING_MOTION_DATA_DIR, BOXING_MOTION_DATA_DIR, PUSHUP_MOTION_DATA_DIR, MOTION_DATA_DIR

from .hi_mimic_rewards import *
import torch

# 这两一定要按顺序来
masked_dof_pos_names = ['l_hip_pitch_joint', 'r_hip_pitch_joint', 'waist_joint', 'l_hip_roll_joint', 'r_hip_roll_joint', 'l_shoulder_pitch_joint', 'r_shoulder_pitch_joint', 'l_thigh_joint', 'r_thigh_joint', 'l_shoulder_roll_joint', 'r_shoulder_roll_joint', 'l_calf_joint', 'r_calf_joint', 'l_upper_arm_joint', 'r_upper_arm_joint', 'l_elbow_joint', 'r_elbow_joint', 'l_wrist_joint', 'r_wrist_joint']
masked_capture_points_names = ['waist_link', 'l_hip_pitch_link', 'l_calf_link', 'l_ankle_roll_link', 'r_hip_pitch_link', 'r_calf_link', 'r_ankle_roll_link', 'l_shoulder_pitch_link', 'l_elbow_link', 'left_hand_link', 'r_shoulder_pitch_link', 'r_elbow_link', 'right_hand_link', 'head_link']

@configclass
class HIRewardCfg(RewardCfg):
    # ---------- Task ----------------------------------------
    # keep_balance = RewTerm(func = keep_balance, weight=1.0)
    tracking_dof_pos = RewTerm(func = tracking_dof_pos, weight=2.0, params = {"std": 0.5})
    tracking_capture_points = RewTerm(func = tracking_capture_points, weight=1.0, params = {"std": 0.5}) # 0.5
    # tracking_masked_dof_pos = RewTerm(func=tracking_masked_dof_pos, \
    #                                   weight = 2.0, 
    #                                   params = {"std": 1.2, # 0.5
    #                                             "masked_ids": get_indices(masked_dof_pos_names, all_dof_pos_names)})
    
    # # tracking_masked_dof_vel = RewTerm(func=tracking_masked_dof_vel, \
    # #                                   weight = 0.5, 
    # #                                   params = {"std": 0.5,\
    # #                                             "masked_ids": get_indices(masked_dof_pos_names, all_dof_pos_names)})

    # tracking_masked_capture_points = RewTerm(func=tracking_masked_capture_points, 
    #                                          weight = 1.0, 
    #                                          params = {"std": 0.2, # 0.5  
    #                                                    "masked_ids": get_indices(masked_capture_points_names, all_capture_points_names)})
    # ----------- Regularization -----------------------
    joint_torques_l2 = RewTerm(func = mdp.joint_torques_l2, weight = -1e-6)
    action_rate_l2 = RewTerm(func = action_rate_l2, weight = -0.3)  # action_rate竟然如此重要 0.4
    # flat_feet_force = RewTerm(func=flat_feet_force, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),\
    #                                                                     "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"]) }) 
    # feet_horizontal = RewTerm(func=mdp.feet_horizontal_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),\
    #                                                                             "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"]) }) 
    # feet_heading = RewTerm(func=mdp.feet_heading_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])}) # -5.0
    # ------------ penalty -------------------------------
    # joint_pos_limit = RewTerm(func = mdp.joint_pos_limits, weight = -10.0) # Penalize joint positions if they cross the soft limits
    # joint_vel_limit = RewTerm(func = mdp.joint_vel_limits, weight = -5.0, params = {"soft_ratio": 0.95}) # Penalize joint positions if they cross the soft limits
    # torque_limits = RewTerm(func = mdp.applied_torque_limits, weight = -0.1)
    # termination = RewTerm(func = termination, weight = -100.0)
@configclass
class HIMimicEnvCfg(BaseEnvCfg):
    reward = HIRewardCfg()
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HI_CFG
        # self.motion_data.motion_file_path = MOTION_DATA_DIR + "/hi/crawl.json"
        # self.motion_data.motion_file_path = MOTION_DATA_DIR + "/hi/waving.json"
        self.motion_data.motion_file_path = MOTION_DATA_DIR + "/hi/pushup.json"


        self.motion_data.use_dof_vel_data = False
        self.motion_data.use_body_vel_data = False

        self.robot.actor_obs_history_length = 10
        self.robot.critic_obs_history_length = 10
        self.robot.feet_body_names = [".*ankle_roll_link"]
        self.robot.base_link_body_names = ["base_link"]

        self.terminate.terminate_contacts = False
        self.terminate.terminate_capture_points_far = False
        self.terminate.terminate_contacts_body_names = ["base_link", ".*shoulder_pitch_link", ".*upper_arm_link", ".*elbow_link"]
        self.terminate.capture_points_distance_threshold = 0.5

        self.normalization.obs_scales.lin_vel = 1.0
        self.normalization.obs_scales.ang_vel = 1.0
        self.normalization.obs_scales.projected_gravity = 1.0
        self.normalization.obs_scales.joint_pos = 1.0
        self.normalization.obs_scales.joint_vel = 1.0
        self.normalization.obs_scales.actions = 1.0
        self.normalization.obs_scales.joint_pos_error = 1.0 
        self.normalization.obs_scales.capture_points_error = 1.0

        # noise ----------------------------------
        self.noise.add_noise = True
        self.noise.noise_scales.ang_vel = 0.2
        self.noise.noise_scales.projected_gravity = 0.05
        self.noise.noise_scales.joint_pos = 0.02
        self.noise.noise_scales.joint_vel = 1.5

        # reset ----------------------------------
        self.domain_rand.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.domain_rand.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.domain_rand.reset_robot_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.domain_rand.reset_robot_base.params["pose_range"]["z"] = (0.02, 0.02)
        self.domain_rand.reset_robot_base.params["velocity_range"]["x"] = (0.0, 0.0)

        # robot property -------------------------
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {"max_delay": 5, "min_delay": 0} 
        self.domain_rand.randomize_robot_friction.enable = True

        self.domain_rand.add_rigid_body_mass.enable = True
        self.domain_rand.add_rigid_body_mass.params["body_names"] = "base_link"
        self.domain_rand.add_rigid_body_mass.params["mass_distribution_params"] = [-1.0, 1.0]
        # TODO: dof_offset、

        # disturbance ---------------------------
        self.domain_rand.push_robot.enable = True
        self.domain_rand.push_robot.push_interval_s = 1.0
        self.domain_rand.push_robot.params["velocity_range"]["x"] = (-0.5, 0.5)
        self.domain_rand.push_robot.params["velocity_range"]["y"] = (-0.5, 0.5)


@configclass
class HIMimicAgentCfg(BaseAgentCfg):
    experiment_name = "hi_mimic"
    wandb_project = "hi_mimic"
    logger = "tensorboard"

    policy = MLPPolicyCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu"
    )

    algorithm: AlgorithmCfg = AlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )