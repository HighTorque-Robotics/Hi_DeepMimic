from mimic_real.envs.base.clean_env_config import (  # noqa:F401
    BaseEnvCfg, BaseAgentCfg, BaseSceneCfg, RobotCfg,
    RewardCfg, PhysxCfg, SimCfg, MLPPolicyCfg, AlgorithmCfg
)
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from mimic_real.assets.usd.high_torque import HI_CFG
from mimic_real.data import WALK_MOTION_DATA_DIR, SIDE_FLIP_MOTION_DATA_DIR, RUN_MOTION_DATA_DIR

import mimic_real.envs.hi_clean.hi_mdp as mdp
import torch

def get_indices(child_list, parent_list):
    indices = []
    for element in child_list:
        try:
            index = parent_list.index(element)
            indices.append(index)
        except ValueError:
            indices.append(-1)
    return indices

all_dof_pos_names = ['l_hip_pitch_joint', 'r_hip_pitch_joint', 'waist_joint', 'l_hip_roll_joint', 'r_hip_roll_joint', 'l_shoulder_pitch_joint', 'r_shoulder_pitch_joint', 'l_thigh_joint', 'r_thigh_joint', 'l_shoulder_roll_joint', 'r_shoulder_roll_joint', 'l_calf_joint', 'r_calf_joint', 'l_upper_arm_joint', 'r_upper_arm_joint', 'l_ankle_pitch_joint', 'r_ankle_pitch_joint', 'l_elbow_joint', 'r_elbow_joint', 'l_ankle_roll_joint', 'r_ankle_roll_joint', 'l_wrist_joint', 'r_wrist_joint']
all_capture_points_names = ['waist_link', 'l_hip_pitch_link', 'l_calf_link', 'l_ankle_roll_link', 'r_hip_pitch_link', 'r_calf_link', 'r_ankle_roll_link', 'l_shoulder_pitch_link', 'l_elbow_link', 'left_hand_link', 'r_shoulder_pitch_link', 'r_elbow_link', 'right_hand_link', 'head_link']
masked_dof_pos_names = ['l_hip_pitch_joint', 'r_hip_pitch_joint', 'waist_joint', 'l_hip_roll_joint', 'r_hip_roll_joint', 'l_shoulder_pitch_joint', 'r_shoulder_pitch_joint', 'l_thigh_joint', 'r_thigh_joint', 'l_shoulder_roll_joint', 'r_shoulder_roll_joint', 'l_calf_joint', 'r_calf_joint', 'l_upper_arm_joint', 'r_upper_arm_joint', 'l_elbow_joint', 'r_elbow_joint', 'l_wrist_joint', 'r_wrist_joint']
masked_capture_points_names = ['waist_link', 'l_hip_pitch_link', 'l_calf_link', 'l_ankle_roll_link', 'r_hip_pitch_link', 'r_calf_link', 'r_ankle_roll_link', 'l_shoulder_pitch_link', 'l_elbow_link', 'left_hand_link', 'r_shoulder_pitch_link', 'r_elbow_link', 'right_hand_link', 'head_link']

@configclass
class HIRewardCfg(RewardCfg):
    keep_balance = RewTerm(func=mdp.keep_balance, weight=0.5)
    tracking_dof_pos = RewTerm(func=mdp.tracking_dof_pos, weight=2.0, params = {"std": 0.5}) # 0.5
    tracking_capture_points = RewTerm(func=mdp.tracking_capture_points, weight=1.0, params = {"std": 0.5}) # 0.5

    # tracking_masked_dof_pos = RewTerm(func=mdp.tracking_masked_dof_pos, \
    #                                   weight=2.0, 
    #                                   params = {"std": 0.5,\
    #                                             "masked_ids": get_indices(masked_dof_pos_names, all_dof_pos_names)})
    # tracking_masked_capture_points = RewTerm(func=mdp.tracking_masked_capture_points, 
    #                                          weight=1.0, 
    #                                          params = {"std": 0.5, 
    #                                                    "masked_ids": get_indices(masked_capture_points_names, all_capture_points_names)})
    
    # flat_feet_force = RewTerm(func=mdp.flat_feet_force, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),\
    #                                                                     "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"]) }) 
    

    # feet_horizontal = RewTerm(func=mdp.feet_horizontal_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),\
    #                                                                             "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"]) }) 
    # feet_heading = RewTerm(func=mdp.feet_heading_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])}) # -5.0

    joint_pos_limit=RewTerm(func=mdp.joint_pos_limits, weight = -0.2)
    action_rate_l2 = RewTerm(func = mdp.action_rate_l2, weight = -0.1) 

@configclass
class HICleanEnvCfg(BaseEnvCfg):
    reward = HIRewardCfg()
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HI_CFG
        self.motion_file_path = WALK_MOTION_DATA_DIR
        self.scene.terrain_type = "plane"

        self.robot.action_scale = 0.25
        self.robot.terminate_contacts_body_names = ["base_link", ".*shoulder_pitch_link", ".*upper_arm_link", ".*elbow_link"]
        self.robot.feet_body_names = [".*ankle_roll_link"]

@configclass
class HICleanAgentCfg(BaseAgentCfg):
    experiment_name = "hi_clean"
    wandb_project = "hi_clean"
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



