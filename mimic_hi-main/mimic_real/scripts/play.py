import os
import torch
from mimic_real.utils import task_registry

import argparse

from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner
# local imports
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

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

from mimic_real.envs import *  # noqa:F401, F403
from mimic_real.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)


    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.load_run = "2025-07-23_10-48-55"
    # agent_cfg.load_run = "0server"
    agent_cfg.load_checkpoint = "model_3800.pt"
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, )
    print('resume_path=============',resume_path)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy_723_pushup.pt")
    export_policy_as_onnx(runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy_723_pushup.onnx")

    if not args_cli.headless:
        from mimic_real.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    while simulation_app.is_running():
        with torch.inference_mode():
            # import ipdb; ipdb.set_trace();
            actions = policy(obs)
            # import ipdb; ipdb.set_trace();
            # actions = torch.zeros_like(actions)
            # actions = env.motion_loader.get_dof_pos_batch(env.phase)
            # print(actions[0])
            obs, _, _, _ = env.step(actions)

if __name__ == '__main__':
    play()
    simulation_app.close()
