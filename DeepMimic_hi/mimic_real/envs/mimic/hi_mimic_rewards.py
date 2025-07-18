from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from mimic_real.envs import BaseEnv


def get_indices(child_list, parent_list):
    indices = []
    for element in child_list:
        try:
            index = parent_list.index(element)
            indices.append(index)
        except ValueError:
            indices.append(-1)
    return indices

def keep_balance(env: BaseEnv):
    """
    计算机器人的平衡奖励，专门针对挥手动作优化
    """
    # 获取机器人基座的位置和姿态
    base_pos = env.robot.data.root_pos_w
    base_quat = env.robot.data.root_quat_w
    
    # 1. 重心高度奖励 - 保持适当的站立高度，容忍度更大
    target_height = 0.45  # 目标高度
    height_error = torch.abs(base_pos[:, 2] - target_height)
    height_reward = torch.exp(-height_error / 0.15)  # 增加容忍度
    
    # 2. 身体姿态奖励 - 保持基本直立，但允许小幅摆动
    # 计算机器人的roll和pitch角度，使用更稳定的计算方法
    projected_gravity = env.robot.data.projected_gravity_b
    roll = torch.atan2(projected_gravity[:, 1], projected_gravity[:, 2])
    pitch = torch.atan2(-projected_gravity[:, 0], 
                       torch.sqrt(projected_gravity[:, 1]**2 + projected_gravity[:, 2]**2))
    
    # 挥手时允许更大的姿态变化
    orientation_error = torch.abs(roll) + torch.abs(pitch)
    orientation_reward = torch.exp(-orientation_error / 0.5)  # 增加容忍度
    
    # 3. 足部接触奖励 - 确保双脚都有良好的接触
    contact_forces = env.contact_sensor.data.net_forces_w_history
    feet_contact = torch.norm(contact_forces[:, -1, env.feet_cfg.body_ids, :], dim=-1)
    
    # 检查双脚接触情况
    left_foot_contact = (feet_contact[:, 0] > 5.0).float()   # 左脚
    right_foot_contact = (feet_contact[:, 1] > 5.0).float()  # 右脚
    both_feet_contact = left_foot_contact * right_foot_contact
    
    # 鼓励双脚都接触地面
    contact_reward = both_feet_contact * 0.8 + (left_foot_contact + right_foot_contact) * 0.1
    
    # 4. 基座稳定性 - 减少过度的线性和角度运动
    base_vel = env.robot.data.root_lin_vel_w
    base_ang_vel = env.robot.data.root_ang_vel_w
    
    # 对于挥手动作，允许适度的基座运动
    lin_vel_penalty = torch.exp(-torch.norm(base_vel[:, :2], dim=-1) / 0.3)  # x,y方向
    z_vel_penalty = torch.exp(-torch.abs(base_vel[:, 2]) / 0.2)  # z方向更严格
    ang_vel_penalty = torch.exp(-torch.norm(base_ang_vel, dim=-1) / 0.8)  # 角速度
    
    velocity_reward = lin_vel_penalty * 0.4 + z_vel_penalty * 0.4 + ang_vel_penalty * 0.2
    
    # 5. 重心稳定性 - 确保重心在支撑面内
    # 简化的重心稳定性检查
    base_xy_pos = base_pos[:, :2]  # x,y位置
    com_stability = torch.exp(-torch.norm(base_xy_pos, dim=-1) / 0.2)
    
    # 组合所有奖励，重新调整权重
    balance_reward = height_reward * 0.25 + orientation_reward * 0.25 + \
                     contact_reward * 0.3 + velocity_reward * 0.15 + \
                     com_stability * 0.05
    
    return balance_reward

def tracking_dof_pos(env: BaseEnv, std: float):
    desired_joint_pos = env.motion_loader.get_dof_pos_batch(env.phase)
    joint_pos_error = desired_joint_pos - env.robot.data.joint_pos
    error_sum = torch.sum(torch.square(joint_pos_error), dim=1) 
    return torch.exp(-error_sum / std**2)

def tracking_capture_points(env: BaseEnv, std: float):
    # desired_capture_points_pos = env.motion_loader.get_capture_points_batch(env.phase)
    # desired_capture_points_pos = desired_capture_points_pos.view(env.num_envs, -1)

    # capture_points_pos = env.robot.data.body_pos_w[:, env.capture_points_body_ids, :] - env.scene.env_origins.unsqueeze(1)
    # capture_points_pos = capture_points_pos.view(env.num_envs, -1)
    # desired_capture_points_error = desired_capture_points_pos \
    #                                 - capture_points_pos

    # error_sum = torch.sum(torch.square(desired_capture_points_error), dim=1) 
    if env.use_local_capture_points:
        error_sum = env.local_capture_points_error_sum()
    else:
        error_sum = env.global_capture_points_error_sum()
    return torch.exp(-error_sum / std**2)

def tracking_masked_dof_pos(env: BaseEnv, std: float, masked_ids):
    desired_joint_pos = env.motion_loader.get_dof_pos_batch(env.phase)
    joint_pos_error = desired_joint_pos - env.robot.data.joint_pos
    error_sum = torch.sum(torch.square(joint_pos_error[:, masked_ids]), dim=1) 
    return torch.exp(-error_sum / std**2)

def tracking_masked_dof_vel(env: BaseEnv, std: float, masked_ids):
    desired_joint_pos = env.motion_loader.get_dof_vel_batch(env.phase)
    joint_pos_error = desired_joint_pos - env.robot.data.joint_pos
    error_sum = torch.sum(torch.square(joint_pos_error[:, masked_ids]), dim=1) 
    return torch.exp(-error_sum / std**2)

def tracking_masked_capture_points(env: BaseEnv, std: float, masked_ids):
    # desired_capture_points_pos = env.motion_loader.get_capture_points_batch(env.phase)
    # desired_capture_points_pos = desired_capture_points_pos.view(env.num_envs, -1)
    # capture_points_pos = env.robot.data.body_pos_w[:, env.capture_points_body_ids, :] - env.scene.env_origins.unsqueeze(1)
    # capture_points_pos = capture_points_pos.view(env.num_envs, -1)
    # desired_capture_points_error = desired_capture_points_pos \
    #                                 - capture_points_pos
    # error_sum = torch.sum(torch.square(desired_capture_points_error[:, masked_ids]), dim=1)
    if env.use_local_capture_points:
        desired_capture_points_error = env.local_capture_points_error()
    else:
        desired_capture_points_error = env.global_capture_points_error()
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

def termination(env: BaseEnv) -> torch.Tensor:
    return env.termination_buf

# foot poistion
# foot air time?


all_dof_pos_names = ['l_hip_pitch_joint', 
                     'r_hip_pitch_joint', 
                     'waist_joint', 
                     'l_hip_roll_joint', 
                     'r_hip_roll_joint', 
                     'l_shoulder_pitch_joint', 
                     'r_shoulder_pitch_joint', 
                     'l_thigh_joint', 
                     'r_thigh_joint', 
                     'l_shoulder_roll_joint', 
                     'r_shoulder_roll_joint', 
                     'l_calf_joint', 
                     'r_calf_joint', 
                     'l_upper_arm_joint', 
                     'r_upper_arm_joint', 
                     'l_ankle_pitch_joint', 
                     'r_ankle_pitch_joint', 
                     'l_elbow_joint',
                     'r_elbow_joint', 
                     'l_ankle_roll_joint', 
                     'r_ankle_roll_joint', 
                     'l_wrist_joint', 
                     'r_wrist_joint']
all_capture_points_names = ['waist_link', 
                            'l_hip_pitch_link', 
                            'l_calf_link', 
                            'l_ankle_roll_link', 
                            'r_hip_pitch_link', 
                            'r_calf_link', 
                            'r_ankle_roll_link', 
                            'l_shoulder_pitch_link', 
                            'l_elbow_link', 
                            'left_hand_link', 
                            'r_shoulder_pitch_link', 
                            'r_elbow_link', 
                            'right_hand_link', 
                            'head_link']

