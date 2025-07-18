# DONT USE IT ANYMORE
import os
import sys
sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
import joblib
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import urdfpy
from utils import *

# from smpl_skeleton.smpl_wrapper import SMPL_BONE_ORDER_NAMES
# from smpl_skeleton.smpl_wrapper import SMPL_Parser

from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
# 非常不好用
class robot_config:
    urdf_file_path = 'resources/robots/hi/urdf/hi_new.urdf'
    link_pick = ['l_hip_pitch_link', "l_calf_link", "l_ankle_roll_link", 
                'r_hip_pitch_link', 'r_calf_link', 'r_ankle_roll_link', \
                "l_shoulder_pitch_link", "l_elbow_link", "left_hand_link", 
                "r_shoulder_pitch_link", "r_elbow_link", "right_hand_link", 
                "left_toe", "right_toe", 
                "head_link"]
    smpl_joint_pick = ["L_Hip", "L_Knee", "L_Ankle",  
                        "R_Hip", "R_Knee", "R_Ankle", 
                        "L_Shoulder", "L_Elbow", "L_Hand",
                        "R_Shoulder", "R_Elbow", "R_Hand",
                        "L_Toe", "R_Toe", 
                        "Head"]

robot = urdfpy.URDF.load(robot_config.urdf_file_path)
# links = robot.links
# link_names = []
# for link in links:
#     # print(f"连杆: {link.name}")
#     link_names.append(link.name)
# # print("-----------------------------------------")
joints = robot.joints
joint_names = []
for joint in joints:
    # print(f"关节: {joint.name} (类型: {joint.joint_type})")
    if (joint.joint_type != "fixed"):
        joint_names.append(joint.name)

q = list(np.zeros(len(joint_names)))
cfg = dict(zip(joint_names, q))
cfg["l_shoulder_roll_joint"] = np.pi / 2.0
cfg["r_shoulder_roll_joint"] = -np.pi / 2.0

# cfg["l_elbow_joint"] = -np.pi / 2.0
# cfg["r_elbow_joint"] = -np.pi / 2.0

# cfg["l_hip_pitch_joint"] = -np.pi / 2.0
# cfg["r_hip_pitch_joint"] = -np.pi / 2.0
# cfg["l_calf_joint"] = np.pi / 2.0
# cfg["r_calf_joint"] = np.pi / 2.0

print(cfg)

link_fk = robot.link_fk(cfg)  # 字典: {link_name: 4x4变换矩阵}
positions = []
link_names = []
for link, T in link_fk.items():
    position = T[:3, 3]
    positions.append(position) 
    link_names.append(link.name) 
    # rotation = T[:3, :3]
    # print(f"Link: {link.name}")
    # print(f"Position: {position}")
positions = np.array(positions)

print("robot_link和smpl_joint对应关系")
for i in range(len(robot_config.link_pick)):
    print(robot_config.link_pick[i], "||", robot_config.smpl_joint_pick[i])
link_pick_idx = [link_names.index(j) for j in robot_config.link_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in robot_config.smpl_joint_pick]
print(SMPL_BONE_ORDER_NAMES)
toe_smpl_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in ["L_Toe", "R_Toe"]]
foot_smpl_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in ["L_Ankle", "R_Ankle"]]
# vis_positions_o3d(positions[link_pick_idx, :])
# vis_positions_target_o3d(positions[link_pick_idx, :], positions[link_names.index("l_elbow_link"), :])
#### Preparing fitting varialbes##################
device = torch.device("cpu")
#### default pause for robot#######################
target_link_pos = torch.from_numpy(positions[link_pick_idx, :]).to(device)
###### prepare SMPL #############################
pose_aa_stand = np.zeros((1, 72)) # SMPL模型的位姿
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec() # root旋转, 因为SMPL初始姿态是躺着的
pose_aa_stand[:, :3] = rotvec # root旋转
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3) # 转成关节表示SMPL模型共有24个关节
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2],  degrees = False).as_rotvec() # 欧拉角到旋转向量
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2],  degrees = False).as_rotvec() 
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0],  degrees = False).as_rotvec()

# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Hip')] = sRot.from_euler("xyz", [-np.pi / 2, 0, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Hip')] = sRot.from_euler("xyz", [-np.pi / 2, 0, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Knee')] = sRot.from_euler("xyz", [np.pi / 2, 0, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Knee')] = sRot.from_euler("xyz", [np.pi / 2, 0, 0],  degrees = False).as_rotvec()

pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))
smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

###### Shape fitting
trans = torch.zeros([1, 3]) # root位置
beta = torch.zeros([1, 10]) # 形状参数
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans) # SMPL forward kinmatics得到verts(mesh顶点)和joint（24个关节点位置）
offset = joints[:, 0] - trans
root_trans_offset = trans + offset

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True) # 形状参数
scale = Variable(torch.ones([1]).to(device), requires_grad=True) # 缩放参数
optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)

target_link_pos[:, 1] *= 1.0
for iteration in range(2000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    smpl_joints_pos = (joints - joints[:, 0]) * scale
    # smpl_joints_pos = joints * scale
    diff = target_link_pos - smpl_joints_pos[:, smpl_joint_pick_idx]
    # diff2 = smpl_joints_pos[:, toe_smpl_pick_idx, 1] - smpl_joints_pos[:, foot_smpl_pick_idx, 1]

    loss_g = diff.norm(dim = -1).mean() # + diff2.norm(dim = -1).mean()
    loss_r =  torch.square(scale - 0.6) # + shape_new.norm(dim = -1).mean() * 1e-5 
    loss = loss_g + 0.2 * loss_r

    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000)
    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()

print(scale)
print(shape_new)
os.makedirs("data/hi", exist_ok=True)
joblib.dump((shape_new.detach(), scale), "data/hi/shape_optimized_v1.pkl") # V2 has hip jointsrea
print(f"shape fitted and saved to data/hi/shape_optimized_v1.pkl")


## visulization ################################################
trans = torch.zeros([1, 3]) # root位置
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans) # SMPL forward kinmatics得到verts(可能和关节有关)和joint（关节点）
faces = smpl_parser_n.faces
# vis_positions_o3d(joints[0].detach().cpu().numpy().squeeze())
vis_sleleton_o3d(joints[0].detach().cpu().numpy().squeeze(), smpl_parser_n.parents)

verts = verts[0].detach().cpu().numpy().squeeze()
faces = faces.astype(np.int32)
# vis_mesh_o3d(verts, faces)
joints = joints[:, smpl_joint_pick_idx]
joints = joints[0].detach().cpu().numpy().squeeze()
# vis_mesh_o3d_with_joints(verts, faces, joints, 1.0)
