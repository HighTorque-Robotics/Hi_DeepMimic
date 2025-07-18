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
    # 大幅增加平衡奖励权重
    keep_balance = RewTerm(func = keep_balance, weight=2.0)  # 从0.5增加到2.0
    
    # 降低动作跟踪要求，增加容忍度
    tracking_dof_pos = RewTerm(func = tracking_dof_pos, weight=1.5, params = {"std": 1.2})  # 降低权重，增加std
    tracking_capture_points = RewTerm(func = tracking_capture_points, weight=0.8, params = {"std": 1.2})  # 增加std
    
    # 使用mask的关键部位跟踪，降低要求
    tracking_masked_dof_pos = RewTerm(func=tracking_masked_dof_pos, \
                                      weight = 1.0,  # 降低权重
                                      params = {"std": 1.5, # 进一步增加容忍度
                                                "masked_ids": get_indices(masked_dof_pos_names, all_dof_pos_names)})
    
    # 关键捕获点的跟踪，降低要求
    tracking_masked_capture_points = RewTerm(func=tracking_masked_capture_points, 
                                             weight = 0.5,  # 降低权重
                                             params = {"std": 1.2,   
                                                       "masked_ids": get_indices(masked_capture_points_names, all_capture_points_names)})
    
    # ----------- Regularization -----------------------
    joint_torques_l2 = RewTerm(func = mdp.joint_torques_l2, weight = -2e-6)  # 进一步减小
    action_rate_l2 = RewTerm(func = action_rate_l2, weight = -0.1)  # 减小权重，允许更多动作变化
    
    # 足部约束（保持稳定性）
    flat_feet_force = RewTerm(func=flat_feet_force, weight=-0.02, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),\
                                                                         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"]) }) 
    
    # 足部水平约束
    feet_horizontal = RewTerm(func=feet_horizontal_l2, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),\
                                                                                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"]) }) 
    
    # 足部方向约束
    feet_heading = RewTerm(func=feet_heading_l2, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])})
    
    # ------------ penalty -------------------------------
    # 关节位置限制（减小惩罚）
    joint_pos_limit = RewTerm(func = mdp.joint_pos_limits, weight = -2.0) # 进一步减小
    
    # 关节速度限制（减小惩罚，增加容忍度）
    joint_vel_limit = RewTerm(func = mdp.joint_vel_limits, weight = -1.0, params = {"soft_ratio": 0.9})  # 降低要求
    
    # 扭矩限制
    torque_limits = RewTerm(func = mdp.applied_torque_limits, weight = -0.02)
    
    # 终止条件惩罚（大幅减小）
    termination = RewTerm(func = termination, weight = -20.0)  # 从-50减少到-20
@configclass
class HIMimicEnvCfg(BaseEnvCfg):
    reward = HIRewardCfg()
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HI_CFG
        self.motion_data.motion_file_path = MOTION_DATA_DIR + "/hi/waving.json"

        self.motion_data.use_dof_vel_data = False
        self.motion_data.use_body_vel_data = False

        self.robot.actor_obs_history_length = 10
        self.robot.critic_obs_history_length = 10
        self.robot.feet_body_names = [".*ankle_roll_link"]
        self.robot.base_link_body_names = ["base_link"]

        # 终止条件优化（更宽松）
        self.terminate.terminate_contacts = False  # 暂时禁用接触终止，提高稳定性
        self.terminate.terminate_capture_points_far = True  # 启用距离终止
        self.terminate.terminate_contacts_body_names = ["base_link"]  # 只监测基座接触
        self.terminate.capture_points_distance_threshold = 1.2  # 进一步增加容忍度

        # 观察量标准化
        self.normalization.obs_scales.lin_vel = 1.0
        self.normalization.obs_scales.ang_vel = 1.0
        self.normalization.obs_scales.projected_gravity = 1.0
        self.normalization.obs_scales.joint_pos = 1.0
        self.normalization.obs_scales.joint_vel = 0.1  # 减小关节速度的权重
        self.normalization.obs_scales.actions = 1.0
        self.normalization.obs_scales.joint_pos_error = 1.0 
        self.normalization.obs_scales.capture_points_error = 1.0

        # 噪声配置优化
        self.noise.add_noise = True
        self.noise.noise_scales.ang_vel = 0.1  # 减小角速度噪声
        self.noise.noise_scales.projected_gravity = 0.02  # 减小重力噪声
        self.noise.noise_scales.joint_pos = 0.01  # 减小关节位置噪声
        self.noise.noise_scales.joint_vel = 0.8  # 减小关节速度噪声

        # 域随机化优化
        self.domain_rand.reset_robot_joints.params["position_range"] = (-0.05, 0.05)  # 减小初始位置随机性
        self.domain_rand.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)  # 保持初始速度为0
        self.domain_rand.reset_robot_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.domain_rand.reset_robot_base.params["pose_range"]["y"] = (0.0, 0.0)  # 添加y方向约束
        self.domain_rand.reset_robot_base.params["pose_range"]["z"] = (0.01, 0.03)  # 减小z方向变化
        self.domain_rand.reset_robot_base.params["pose_range"]["roll"] = (-0.05, 0.05)  # 添加roll约束
        self.domain_rand.reset_robot_base.params["pose_range"]["pitch"] = (-0.05, 0.05)  # 添加pitch约束
        self.domain_rand.reset_robot_base.params["pose_range"]["yaw"] = (-0.1, 0.1)  # 添加yaw约束
        
        # 摩擦力随机化优化
        self.domain_rand.randomize_robot_friction.params["static_friction_range"] = [0.8, 1.2]  # 增加摩擦力范围
        self.domain_rand.randomize_robot_friction.params["dynamic_friction_range"] = [0.6, 1.0]  # 增加动态摩擦力

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
    
    # 增加训练步数和保存频率
    num_steps_per_env: int = 32  # 增加步数
    max_iterations: int = 10000  # 调整迭代次数
    save_interval: int = 200  # 增加保存频率

    policy = MLPPolicyCfg(
        class_name="ActorCritic",
        init_noise_std=0.8,  # 减小初始噪声
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu"
    )

    algorithm: AlgorithmCfg = AlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,  # 降低价值损失权重，避免过度约束
        use_clipped_value_loss=True,
        clip_param=0.3,  # 增加clip参数，允许更大的策略更新
        entropy_coef=0.02,  # 进一步增加熵系数，鼓励探索
        num_learning_epochs=6,  # 适中的学习轮数
        num_mini_batches=4,  # 减少批量数量，提高稳定性
        learning_rate=3e-4,  # 进一步降低学习率
        schedule="adaptive",
        gamma=0.99,  # 适中的折扣因子
        lam=0.95,  # 适中的GAE参数
        desired_kl=0.01,  # 适中的KL散度目标
        max_grad_norm=1.0,  # 适中的梯度裁剪
    )