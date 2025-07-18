from smpl_skeleton.smpl_wrapper import SMPL_Parser
from smpl_skeleton.smpl_wrapper import SMPL_BONE_ORDER_NAMES
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from utils import *
from param import robot_config

import pytorch_kinematics as pk
chain = pk.build_chain_from_urdf(open(robot_config.urdf_file_path, mode="rb").read())
joint_names = chain.get_joint_parameter_names()
link_names = chain.get_link_names()

q = torch.zeros(len(joint_names))
q[joint_names.index("left_elbow_joint")] = np.pi / 2.0
q[joint_names.index("right_elbow_joint")] = np.pi / 2.0
q[joint_names.index("left_shoulder_roll_joint")] = np.pi / 2.0
q[joint_names.index("right_shoulder_roll_joint")] = -np.pi / 2.0

tg = chain.forward_kinematics(q)
positions = []
for i in range(len(link_names)):
    m = tg[link_names[i]].get_matrix()
    pos = np.array(m[0, :3, 3])
    if link_names[i] == "head_link":
        print(pos)
    positions.append(pos)
 
positions = np.array(positions)
rotation = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
positions = rotation.apply(positions)

print("robot_link和smpl_joint对应关系")
for i in range(len(robot_config.link_pick)):
    print(robot_config.link_pick[i], "||", robot_config.smpl_joint_pick[i])
link_pick_idx = [link_names.index(j) for j in robot_config.link_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in robot_config.smpl_joint_pick]

robot_config.default_J[0][:, 1] += 0.22
robot_config.default_J[0][:, 2] -= 0.015
robot_config.default_J[0][:, 1] *= 0.5
robot_config.default_J[0][:, 0] *= 1.0
robot_config.default_J[0][smpl_joint_pick_idx, :] = torch.tensor(positions[link_pick_idx, :]).float()
robot_config.default_J[0][SMPL_BONE_ORDER_NAMES.index("L_Wrist"), 0] -= 0.18
robot_config.default_J[0][SMPL_BONE_ORDER_NAMES.index("R_Wrist"), 0] += 0.18
robot_config.default_J[0][SMPL_BONE_ORDER_NAMES.index("L_Wrist"), 1] += 0.07
robot_config.default_J[0][SMPL_BONE_ORDER_NAMES.index("R_Wrist"), 1] += 0.07
robot_config.default_J[0][SMPL_BONE_ORDER_NAMES.index("L_Wrist"), 2] += 0.07
robot_config.default_J[0][SMPL_BONE_ORDER_NAMES.index("R_Wrist"), 2] += 0.07


print(robot_config.default_J)

smpl_parser_n = SMPL_Parser(default_joint_pos = robot_config.default_J)
pose_aa_stand = np.zeros((1, 72)) # SMPL模型的位姿
# rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec() # root旋转, 因为SMPL初始姿态是躺着的
# pose_aa_stand[:, :3] = rotvec # root旋转
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3) # 转成关节表示SMPL模型共有24个关节
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Hip')] = sRot.from_euler("xyz", [-np.pi / 2, 0, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Hip')] = sRot.from_euler("xyz", [-np.pi / 2, 0, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Knee')] = sRot.from_euler("xyz", [np.pi / 2, 0, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Knee')] = sRot.from_euler("xyz", [np.pi / 2, 0, 0],  degrees = False).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))
trans = torch.zeros([1, 3]) 
smpl_joints = smpl_parser_n.get_joints_verts(pose_aa_stand, trans)

toe_ori_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in robot_config.toe_ori_joint_pick]

xyzws = get_toe_rotation(smpl_joints[:, toe_ori_joint_pick_idx].detach().cpu().numpy())
left_toe = get_coord_mesh(origin = smpl_joints[0, SMPL_BONE_ORDER_NAMES.index('L_Toe')].detach().cpu().numpy().squeeze(), xyzw = xyzws[0][0], size = 0.07)
right_toe = get_coord_mesh(origin = smpl_joints[0, SMPL_BONE_ORDER_NAMES.index('R_Toe')].detach().cpu().numpy().squeeze(), xyzw = xyzws[0][1], size = 0.07)

sleleton_list = get_sleleton_o3d(smpl_joints[0].detach().cpu().numpy().squeeze(), smpl_parser_n.parents)
robot_point_list = get_positions_o3d(positions[link_pick_idx, :], color = [0, 1, 0], radius=0.012)
smpl_target_list = get_positions_o3d(smpl_joints[0, smpl_joint_pick_idx].detach().cpu().numpy().squeeze(), color = [0, 0, 1], radius=0.012)
geometry = sleleton_list + robot_point_list + smpl_target_list
geometry.append(left_toe)
geometry.append(right_toe)

o3d.visualization.draw_geometries(geometry)