from dataclasses import MISSING
import math
from isaaclab.utils import configclass
from  .base_config import *


@configclass
class BaseEnvCfg:
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg()
    motion_data: MotionDataCfg = MotionDataCfg()
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        base_link_body_names = ["base_link"],
        feet_body_names=[".*ankle_roll_link"],
    )
    terminate: TerminateCfg = TerminateCfg()
    reward = RewardCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg()
    domain_rand: DomainRandCfg = DomainRandCfg(
        reset_robot_joints=ResetRobotJointsCfg(
            params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
        ),
        reset_robot_base=ResetRobotBaseCfg(
            params={
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
        ),
        randomize_robot_friction=RandomizeRobotFrictionCfg(
            enable=True,
            params={
                "static_friction_range": [0.6, 1.0],
                "dynamic_friction_range": [0.4, 0.8],
                "restitution_range": [0.0, 0.005],
                "num_buckets": 64,
            }
        ),
        push_robot=PushRobotCfg(
            enable=False,
            push_interval_s=15.0,
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}

        ),
        action_delay=ActionDelayCfg(
            enable=True,
            params={"max_delay": 5, "min_delay": 0}
        ),
    )
    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=4,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15
        )
    )

    def __post_init__(self):
        pass


@configclass
class BaseAgentCfg:
    resume: bool = False
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    experiment_name: str = MISSING
    empirical_normalization: bool = False
    device: str = "cuda:0"
    run_name: str = ""
    logger: str = "tensorboard"
    wandb_project: str = MISSING
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    policy: MLPPolicyCfg = MLPPolicyCfg(
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

    def __post_init__(self):
        pass
