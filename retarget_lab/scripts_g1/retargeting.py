import time
import viser

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks
import yourdfpy
from pyroki.collision import HalfSpace, RobotCollision, Sphere

import jaxlie
import jax.numpy as jnp
import numpy as onp
import os
import joblib
import json

class robot_config:
    robot_name = "unitree_g1"
    assets_path = "assets/urdf/g1_description/g1_29dof_rev_1_0.urdf"
    mesh_dir = "assets/urdf/g1_description/meshes"
    target_link_names = ['pelvis',  'left_hip_pitch_link', "left_knee_link", "left_ankle_roll_link",  \
                                'right_hip_pitch_link', 'right_knee_link', 'right_ankle_roll_link', \
                                "left_shoulder_pitch_link", "left_elbow_link", "left_rubber_hand", \
                                "right_shoulder_pitch_link", "right_elbow_link", "right_rubber_hand", "human_head_link"]
    foot_names = ['l1', 'l2', 'l3', 'l4',
                  'r1', 'r2', 'r3', 'r4']
    toe_names = ['left_toe', 'right_toe']

    base_link_names = ['base_link']
    
    pos_weight = onp.ones((len(target_link_names), 3)) * 5.0
    ori_weight = onp.zeros((len(target_link_names), 3))

def main():
    # load data
    file_name = input("载入预处理文件名称(需后缀):")
    file_path = os.path.join("output/{}".format(robot_config.robot_name), file_name)
    try:
        loaded_data = joblib.load(file_path)
        print(f"数据加载成功，类型: {type(loaded_data)}")
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except Exception as e:
        print(f"错误：加载文件时出错 - {e}")

    # 加载预处理数据
    robot_config.target_link_names = loaded_data["link_name"]
    # print(robot_config.target_link_names)
    fps = loaded_data["fps"]
    target_positions = loaded_data["joints"] # [time, body, xyz]
    toe_wxyzs = loaded_data["toe_xyzws"][:, :, [3, 0, 1, 2]] # time, body, wxyz

    # 目标位置和末端姿态的目标和权重
    min_foot_height = onp.min(target_positions[:, :, 2])
    target_positions[:, :, 2] -= (min_foot_height) #将整体高度降低

    target_wxyzs = onp.zeros((target_positions.shape[0], target_positions.shape[1], 4))
    target_wxyzs[:, :, 0] = 1.0
    target_wxyzs[:, [robot_config.target_link_names.index(robot_config.toe_names[0]), \
                     robot_config.target_link_names.index(robot_config.toe_names[1])]] = toe_wxyzs

    robot_config.pos_weight = onp.ones((len(robot_config.target_link_names), 3)) * 5.0
    robot_config.ori_weight = onp.zeros((len(robot_config.target_link_names), 3))
    robot_config.ori_weight[robot_config.target_link_names.index(robot_config.toe_names[0]), :] = 1.0
    robot_config.ori_weight[robot_config.target_link_names.index(robot_config.toe_names[1]), :] = 1.0

##############################################################################################
    # Create robot.
    urdf = yourdfpy.URDF.load(robot_config.assets_path, mesh_dir=robot_config.mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)
    print(robot.links.names)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)

    base_frame = server.scene.add_frame("/base", show_axes=False, axes_length= 0.1, axes_radius=0.001)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    cfg = onp.array(robot.joint_var_cls(0).default_factory())
    
    prev_foot_pos = onp.zeros((len(robot_config.foot_names), 3))
    prev_foot_wxyz = onp.zeros((len(robot_config.foot_names), 4))
    prev_foot_wxyz[:, 0] = 1.0

    ############################
    # data to save
    root_trans_to_save = []
    root_wxyz_to_save = []
    target_link_pos_to_save = []
    dof_pos_to_save = []
    import re
    pattern = "fixed"
    data_joint_names = [x for x in robot.joints.names if not re.search(pattern, x)]
    print(data_joint_names)
    ############################
    idx = 0
    # while True: # 
    while idx != target_positions.shape[0]:
        if idx == target_positions.shape[0]:
            idx = 0
        start_time = time.time()

        base_pos, base_wxyz, cfg = pks.solve_retargeting (
            robot=robot,
            coll=robot_coll,

            target_link_names = robot_config.target_link_names,
            foot_link_names = robot_config.foot_names,

            target_positions = target_positions[idx],
            target_wxyzs = target_wxyzs[idx],
            prev_pos=base_frame.position,
            prev_wxyz=base_frame.wxyz,
            prev_foot_pos = prev_foot_pos,
            prev_foot_wxyz = prev_foot_wxyz,
            prev_cfg = cfg,

            link_pos_weight=robot_config.pos_weight,
            link_ori_weight=robot_config.ori_weight,
        )
        
        # 显示 keypoints-----------------------------------
        keypoints_pos, keypoints_ori\
            = pks.fk_my(robot,
                base_pos,
                base_wxyz,
                cfg,
                robot_config.target_link_names)
        # server.scene.add_batched_axes("/marker", 
        #                         axes_length= 0.1, 
        #                         axes_radius=0.001,
        #                         batched_positions = keypoints_pos,
        #                         batched_wxyzs = keypoints_ori)

        # 显示脚-------------------------------------------
        # feet_pos, feet_ori\
        #     = pks.fk_my(robot,
        #         base_pos,
        #         base_wxyz,
        #         cfg,
        #         robot_config.foot_names)
        # server.scene.add_batched_axes("/marker", 
        #                         axes_length= 0.1, 
        #                         axes_radius=0.001,
        #                         batched_positions = feet_pos,
        #                         batched_wxyzs = feet_ori)

        # 显示目标-----------------------------------------
        # server.scene.add_batched_axes("/marker", 
        #                 axes_length= 0.1, 
        #                 axes_radius=0.001,
        #                 batched_positions = target_positions[idx],
        #                 batched_wxyzs = target_wxyzs[idx])

        if idx == 0:
            prev_foot_pos, prev_foot_wxyz \
                = pks.fk_my(robot,
                            base_pos,
                            base_wxyz,
                            cfg,
                            robot_config.foot_names)
        
        # base_link_pos, base_link_wxyz \
        # = pks.fk_my(robot,
        #             base_pos,
        #             base_wxyz,
        #             cfg,
        #             robot_config.base_link_names)
        # server.scene.add_batched_axes("/marker", 
        #                         axes_length= 0.1, 
        #                         axes_radius=0.001,
        #                         batched_positions = base_link_pos,
        #                         batched_wxyzs = base_link_wxyz)

        

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        
        # Update visualizer.
        urdf_vis.update_cfg(cfg)
        base_frame.position = onp.array(base_pos)
        base_frame.wxyz = onp.array(base_wxyz)

        urdf_vis.update_cfg(cfg)
        base_frame.position = onp.array(base_pos)
        base_frame.wxyz = onp.array(base_wxyz)

        # urdf_vis.update_cfg(onp.zeros_like(cfg))
        # base_frame.position = onp.zeros_like(base_pos)
        # tmp = onp.zeros_like(base_wxyz)
        # tmp[0] = 1.0
        # base_frame.wxyz = tmp

        time.sleep(1.0 / fps)
        idx += 1

        root_trans_to_save.append(base_pos)
        root_wxyz_to_save.append(base_wxyz)
        target_link_pos_to_save.append(keypoints_pos)
        dof_pos_to_save.append(cfg)

    
    data_out = {
        "fps": fps,
        "target_link_names": robot_config.target_link_names,
        "data_joint_names": data_joint_names,
        "root_trans": onp.array(root_trans_to_save).tolist(),
        "root_wxyz": onp.array(root_wxyz_to_save).tolist(),
        "target_link_pos": onp.array(target_link_pos_to_save).tolist(),
        "dof_pos": onp.array(dof_pos_to_save).tolist()
    }
    file_name = input("输出文件名称(需后缀):")
    file_path = os.path.join(f"output/hightorque_hi", file_name)
    with open(file_path, 'w') as f:
        json.dump(data_out, f, indent=2)

if __name__ == "__main__":
    main()


