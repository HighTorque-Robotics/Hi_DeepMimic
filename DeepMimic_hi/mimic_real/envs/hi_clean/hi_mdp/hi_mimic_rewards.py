from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from mimic_real.envs import BaseEnv

def keep_balance(env: BaseEnv):
    return torch.ones(
        env.num_envs, dtype = torch.float, device = env.device, requires_grad=False
    )

def tracking_dof_pos(env: BaseEnv, std: float):
    desired_joint_pos = env.motion_loader.get_dof_pos_batch(env.phase)
    joint_pos_error = desired_joint_pos - env.robot.data.joint_pos
    error_sum = torch.sum(torch.square(joint_pos_error), dim=1) 
    return torch.exp(-error_sum / std**2)

def tracking_capture_points(env: BaseEnv, std: float):
    desired_capture_points_pos = env.motion_loader.get_capture_points_batch(env.phase)
    desired_capture_points_pos = desired_capture_points_pos.view(env.num_envs, -1)

    capture_points_pos = env.robot.data.body_pos_w[:, env.capture_points_body_ids, :] - env.scene.env_origins.unsqueeze(1)
    capture_points_pos = capture_points_pos.view(env.num_envs, -1)
    desired_capture_points_error = desired_capture_points_pos \
                                    - capture_points_pos

    error_sum = torch.sum(torch.square(desired_capture_points_error), dim=1) 
    return torch.exp(-error_sum / std**2)

def tracking_masked_dof_pos(env: BaseEnv, std: float, masked_ids):
    desired_joint_pos = env.motion_loader.get_dof_pos_batch(env.phase)
    joint_pos_error = desired_joint_pos - env.robot.data.joint_pos
    error_sum = torch.sum(torch.square(joint_pos_error[:, masked_ids]), dim=1) 
    return torch.exp(-error_sum / std**2)

def tracking_masked_capture_points(env: BaseEnv, std: float, masked_ids):
    desired_capture_points_pos = env.motion_loader.get_capture_points_batch(env.phase)
    desired_capture_points_pos = desired_capture_points_pos.view(env.num_envs, -1)

    capture_points_pos = env.robot.data.body_pos_w[:, env.capture_points_body_ids, :] - env.scene.env_origins.unsqueeze(1)
    capture_points_pos = capture_points_pos.view(env.num_envs, -1)
    desired_capture_points_error = desired_capture_points_pos \
                                    - capture_points_pos

    error_sum = torch.sum(torch.square(desired_capture_points_error[:, masked_ids]), dim=1) 
    return torch.exp(-error_sum / std**2)

def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    reward = torch.sum(torch.square(env.action - env.last_action), dim=-1)
    reward = torch.clip(reward, 0, 5.0)
    return reward

def flat_feet_force(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]

    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    feet_projected = math_utils.quat_rotate_inverse(feet_quat, env.gravity_vec_feet)
    nonflat_force_loss = torch.sum(torch.square(feet_projected[:, :, 0:2]), dim=2) * contacts_force

    reward = torch.sum(nonflat_force_loss, dim=-1)
    reward = torch.clip(reward, 0, 2.0)
    return reward

def feet_heading_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]

    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    feet_forward = math_utils.quat_apply(feet_quat, env.forward_vec_feet)
    feet_heading = torch.atan2(feet_forward[:, :, 1], feet_forward[:, :, 0])

    heading_error = torch.square(math_utils.wrap_to_pi(asset.data.heading_w - feet_heading[:, 0]))\
                    + torch.square(math_utils.wrap_to_pi(asset.data.heading_w - feet_heading[:, 1]))
    
    return heading_error

def feet_horizontal_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0 

    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    feet_projected = math_utils.quat_rotate_inverse(feet_quat, env.gravity_vec_feet)
    feet_projected_loss = torch.sum(torch.square(feet_projected[:, :, 0:2]), dim=2)# * contacts

    reward = torch.sum(feet_projected_loss, dim=-1)
    return reward


# foot poistion
# foot air time?