import os
import torch
from mimic_real.utils import task_registry

import argparse
from isaaclab.app import AppLauncher
from mimic_real.agents.on_policy_runner import OnPolicyRunner

import mimic_real.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from mimic_real.envs import *  # noqa:F401, F403

def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, _ = task_registry.get_cfgs(env_class_name)
    env_cfg.noise.add_noise = False
    env_cfg.scene.num_envs = 1

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    # obs, _ = env.get_observations()
    while simulation_app.is_running():
        env.show_motion()

if __name__ == '__main__':
    play()
    simulation_app.close()
