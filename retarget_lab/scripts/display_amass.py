import torch
from smpl_skeleton.smpl_wrapper import SMPL_Parser
from smpl_skeleton.smpl_wrapper import SMPL_BONE_ORDER_NAMES
import joblib
import numpy as np
from utils import *
from param import robot_config

# amass_data = load_amass_data("data/AMASS/AMASS_Complete/CMU/CMU/45/45_01_poses.npz") # 行走
# amass_data = load_amass_data("data/AMASS/AMASS_Complete/ACCAD/Female1Gestures_c3d/D6- CartWheel_poses.npz") # 行走

# amass_data = load_amass_data("data/AMASS/AMASS_Complete/ACCAD/Male2MartialArtsPunches_c3d/E14 - body cross right_poses.npz") # 打拳
# amass_data = load_amass_data("data/AMASS/AMASS_Complete/CMU/CMU/10/10_05_poses.npz") # 踢球

# amass_data = load_amass_data("data/AMASS/AMASS_Complete/ACCAD/Female1General_c3d/A8 - crouch to lie_poses.npz")
amass_data = load_amass_data("data/from_video/waving.npz")

load_frame_rate = 30
default_joint_pos = robot_config.robot_J
scale = robot_config.robot_scale
slow_rate = 1.0
# default_joint_pos = None
# scale = 1.0

device = "cpu"


smpl_parser_n = SMPL_Parser(default_joint_pos = default_joint_pos)
smpl_parser_n.to(device)

skip = int(amass_data['fps'] // load_frame_rate)
print("data frame per second:", amass_data['fps'])
trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device) * scale
N = trans.shape[0]
pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis = -1)).float().to(device)
# pose_aa_walk[:, :3] = torch.from_numpy(sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()).float().to(device) * pose_aa_walk.cpu().numpy()[:, :3]

joints = smpl_parser_n.get_joints_verts(pose_aa_walk, trans)

toe_ori_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in robot_config.toe_ori_joint_pick]
toe_xyzws = get_toe_rotation(joints[:, toe_ori_joint_pick_idx].detach().cpu().numpy())

# 创建可视化器
vis = o3d.visualization.Visualizer()
vis.create_window()

coordinate_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.]) # 加上坐标系 size = 0.2
line_ground = create_line_checkerboard_ground(5, 20)
vis.add_geometry(line_ground)
vis.add_geometry(coordinate_frame_mesh)

skeleton_items = get_sleleton_o3d(joints[0].detach().cpu().numpy().squeeze(), smpl_parser_n.parents)
for item in skeleton_items:
    vis.add_geometry(item)

left_toe = get_coord_mesh(origin = joints[0, SMPL_BONE_ORDER_NAMES.index('L_Toe')].detach().cpu().numpy().squeeze(), xyzw = toe_xyzws[0][0], size = 0.07)
right_toe = get_coord_mesh(origin = joints[0, SMPL_BONE_ORDER_NAMES.index('R_Toe')].detach().cpu().numpy().squeeze(), xyzw = toe_xyzws[0][1], size = 0.07)
vis.add_geometry(left_toe)
vis.add_geometry(right_toe)

for i in range(joints.shape[0]):
    new_skeleton_items = get_sleleton_o3d(joints[i].detach().cpu().numpy().squeeze(), smpl_parser_n.parents)
    for j in range(len(skeleton_items)):
        if isinstance(skeleton_items[j], o3d.geometry.TriangleMesh):
            skeleton_items[j].vertices = new_skeleton_items[j].vertices
            skeleton_items[j].triangles = new_skeleton_items[j].triangles
        elif isinstance(item, o3d.geometry.LineSet):
            skeleton_items[j].points = new_skeleton_items[j].points
        vis.update_geometry(skeleton_items[j])

    new_left_toe = get_coord_mesh(origin = joints[i, SMPL_BONE_ORDER_NAMES.index('L_Toe')].detach().cpu().numpy().squeeze(), xyzw = toe_xyzws[i][0], size = 0.07)
    new_right_toe = get_coord_mesh(origin = joints[i, SMPL_BONE_ORDER_NAMES.index('R_Toe')].detach().cpu().numpy().squeeze(), xyzw = toe_xyzws[i][1], size = 0.07)
    left_toe.vertices = new_left_toe.vertices
    left_toe.triangles = new_left_toe.triangles
    right_toe.vertices = new_right_toe.vertices
    right_toe.triangles = new_right_toe.triangles
    vis.update_geometry(left_toe)
    vis.update_geometry(right_toe)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(slow_rate / float(load_frame_rate))
vis.destroy_window()

smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in robot_config.smpl_joint_pick]
filename = input("输入要保存的文件名（无需后缀）：")
data_dump = {
        "joints": joints[:, smpl_joint_pick_idx].squeeze().cpu().detach().numpy(),
        "fps": load_frame_rate, 
        "robot_name": robot_config.robot_name,
        "link_name": robot_config.link_pick,
        "toe_xyzws": toe_xyzws
        }

output_dir = f"output/{robot_config.robot_name}"
os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时出错
joblib.dump(data_dump, os.path.join(output_dir, f"{filename}.pkl"))