from dataclasses import MISSING
import math
from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class RewardCfg:
    pass

@configclass
class BaseSceneCfg:
    seed: int = 42
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = "plane"
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5

@configclass
class TerminateCfg:
    terminate_contacts:bool = True
    terminate_contacts_body_names: list = []

    terminate_capture_points_far:bool = True
    capture_points_distance_threshold: float = 1.0

    terminate_dof_pos_limit: bool = True
    terminate_dof_vel_limit: bool = True
    terminate_non_flat_contact: bool = True

@configclass
class MotionDataCfg:
    motion_file_path: str = MISSING
    cycle_motion: bool = False
    use_dof_vel_data: bool = False
    use_body_vel_data: bool = False


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    base_link_body_names: list = []
    feet_body_names: list = []

@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    joint_pos_error: float = 1.0
    capture_points_error: float = 1.0
    sin_motion_phase: float = 1.0
    cos_motion_phase: float = 1.0
    sin2_motion_phase: float = 1.0
    cos2_motion_phase: float = 1.0
    motion_phase: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0

    action_scale: float = 0.25
    clip_actions: float = 100.0


@configclass
class NoiseScalesCfg:
    lin_vel: float = 0.0 # actor 不会用到
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    actions: float = 0.0
    sin_motion_phase: float = 0.0
    cos_motion_phase: float = 0.0
    sin2_motion_phase: float = 0.0
    cos2_motion_phase: float = 0.0
    motion_phase: float = 0.0


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_level: float = 1.0 # 没有用到，可能是课程？
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class ResetRobotJointsCfg:
    params: dict = {"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)}


@configclass
class ResetRobotBaseCfg:
    params: dict = {
        "pose_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }


@configclass
class RandomizeRobotFrictionCfg:
    enable: bool = True
    params: dict = {
        "static_friction_range": [0.6, 1.0],
        "dynamic_friction_range": [0.4, 0.8],
        "restitution_range": [0.0, 0.005],
        "num_buckets": 64,
    }


@configclass
class AddRigidBodyMassCfg:
    enable: bool = True
    params: dict = {
        "body_names": MISSING,
        "mass_distribution_params": (-5.0, 5.0),
        "operation": "add",
    }


@configclass
class PushRobotCfg:
    enable: bool = True
    push_interval_s: float = 15.0
    params: dict = {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}


@configclass
class DomainRandCfg:
    reset_robot_joints: ResetRobotJointsCfg = ResetRobotJointsCfg()
    reset_robot_base: ResetRobotBaseCfg = ResetRobotBaseCfg()
    randomize_robot_friction: RandomizeRobotFrictionCfg = RandomizeRobotFrictionCfg()
    add_rigid_body_mass: AddRigidBodyMassCfg = AddRigidBodyMassCfg()
    push_robot: PushRobotCfg = PushRobotCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    physx: PhysxCfg = PhysxCfg()


@configclass
class MLPPolicyCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "elu"

@configclass
class AlgorithmCfg:
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
