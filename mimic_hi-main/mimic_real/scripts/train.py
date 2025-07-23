from mimic_real.utils import task_registry

import argparse

from isaaclab.app import AppLauncher
from mimic_real.agents.on_policy_runner import OnPolicyRunner
# local imports
import mimic_real.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from isaaclab.utils.io import dump_yaml

from mimic_real.envs import *  # noqa:F401, F403
from mimic_real.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
import os
from datetime import datetime
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_cfg.device = args_cli.device
    agent_cfg.device = args_cli.device

    print("simulation device:", env_cfg.device)
    print("algorithm device:", agent_cfg.device)
    
    env_class = task_registry.get_task_class(env_class_name)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.scene.seed = args_cli.seed
    env = env_class(env_cfg, args_cli.headless)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # agent_cfg.resume = True
    # agent_cfg.load_run = "2025-05-23_14-10-04"
    # agent_cfg.load_checkpoint = "model_2200.pt"
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    from mimic_real.utils.save_file import copy_py_files
    source_folder = '/home/sunteng/lab_ws/mimic_real/mimic_real/envs/mimic'
    destination_folder = log_dir + "/py"
    copy_py_files(source_folder, destination_folder)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    train()
    simulation_app.close()
