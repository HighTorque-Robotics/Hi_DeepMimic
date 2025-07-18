"""Mobile IK

Same as 01_basic_ik.py, but with a mobile base!
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks
import yourdfpy

import jaxlie
import jax.numpy as jnp
import numpy as onp

def main():
    """Main function for IK with a mobile base.
    The base is fixed along the xy plane, and is biased towards being at the origin.
    """

    urdf = yourdfpy.URDF.load("assets/urdf/hi/urdf/hi_23dof_250401_rl.urdf", mesh_dir="assets/urdf/hi/meshes")

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False, axes_length= 0.1, axes_radius=0.001)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_l_hand = server.scene.add_transform_controls(
        "/left_hand", scale=0.2, position=(0.083, 0.262, 0.0), wxyz=(1, 0, 0, 0), active_axes= [True, True, False]
    )
    ik_l_ankle = server.scene.add_transform_controls(
        "/left_foot", scale=0.2, position=(-0.530, 0.099, 0.089), wxyz=(0.707, 0.0, 0.707, 0.0),
    )
    ik_base = server.scene.add_transform_controls(
        "/fuck", scale=0.2, position=(0.138, 0.0, 0.318), wxyz=(0.707, 0.0, 0.707, 0.0), active_axes= [True, False, True]
    )

    target_link_names = ["left_hand_link", "right_hand_link", "l_ankle_roll_link", "r_ankle_roll_link", "base_link"]

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    cfg = onp.array(robot.joint_var_cls(0).default_factory())

    marker_link_names = ['waist_link',  'l_hip_pitch_link', "l_calf_link", "l_ankle_roll_link",  \
                                'r_hip_pitch_link', 'r_calf_link', 'r_ankle_roll_link', \
                                "l_shoulder_pitch_link", "l_elbow_link", "left_hand_link", \
                                "r_shoulder_pitch_link", "r_elbow_link", "right_hand_link", "head_link"]
    marker_indices = []
    for link_name in marker_link_names:
        marker_indices.append(robot.links.names.index(link_name))

    data_dump = {}
    root_trans_offset = []
    root_rot = []
    capture_points = []
    dof = []
    import re
    pattern = "fixed"
    mjcf_joint_names = [x for x in robot.joints.names if not re.search(pattern, x)]


    frame_time = 0.0
    frame_duration = 1.0 / 30.0

    motion_frequency = 0.8
    while True:#frame_time < 3.0 / motion_frequency:
        # Solve IK.
        start_time = time.time()
        # start_time = frame_time

        # base_position = ik_base.position
        base_position = onp.array([0.138, 0.0, 0.17 + 0.148 * 0.5 * (onp.sin(2 * 3.1415926 * motion_frequency * start_time) + 1)])

        ik_r_hand_pos = onp.array([ik_l_hand.position[0], -ik_l_hand.position[1], ik_l_hand.position[2]])
        ik_r_ankle_pos = onp.array([ik_l_ankle.position[0], -ik_l_ankle.position[1], ik_l_ankle.position[2]])
        # import ipdb; ipdb.set_trace();
        target_positions=onp.array([ik_l_hand.position, ik_r_hand_pos,
                                    ik_l_ankle.position, ik_r_ankle_pos,
                                    base_position])
        target_wxyzs = onp.array([ik_l_hand.wxyz, ik_l_hand.wxyz, \
                                    ik_l_ankle.wxyz, ik_l_ankle.wxyz,
                                    ik_base.wxyz])


        base_pos, base_wxyz, cfg = pks.solve_ik_with_base_multi(
            robot=robot,
            target_link_names=target_link_names,
            target_positions = target_positions,
            target_wxyzs = target_wxyzs,
            fix_base_position=(False, True, False),  # Only free along xy plane.
            fix_base_orientation=(True, False, True),  # Free along z-axis rotation.
            prev_pos=base_frame.position,
            prev_wxyz=base_frame.wxyz,
            prev_cfg=cfg,
        )

        T_world_root = jaxlie.SE3(
            jnp.concatenate([jnp.array(base_wxyz), jnp.array(base_pos)], axis=-1) 
        )
        T_base_marker = jaxlie.SE3(robot.forward_kinematics(cfg)[marker_indices, :])
        T_world_marker = T_world_root @ T_base_marker

        # import ipdb; ipdb.set_trace();
        marker_frame = server.scene.add_batched_axes("/marker", axes_length= 0.1, axes_radius=0.001,\
                                                      batched_positions= onp.array(T_world_marker.translation()), \
                                                      batched_wxyzs = onp.array(T_world_marker.rotation().wxyz))

        # print("base_pos:", base_pos)
        # print("base_wxyz:", base_wxyz)
        # print("cfg:", cfg)

        # print("ik_hand", ik_l_hand.position)

        root_trans_offset.append(base_pos)
        root_rot.append(onp.array([base_wxyz[1], base_wxyz[2], base_wxyz[3], base_wxyz[0]]))
        capture_points.append(onp.array(T_world_marker.translation()))
        dof.append(onp.array(cfg))
        


        # Update timing handle.

        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        
        # Update visualizer.
        urdf_vis.update_cfg(cfg)
        base_frame.position = onp.array(base_pos)
        base_frame.wxyz = onp.array(base_wxyz)

        # update time 
        frame_time += frame_duration

    data_dump["default"] = {
        "root_trans_offset": onp.array(root_trans_offset),
        "capture_points": onp.array(capture_points),
        "dof": onp.array(dof), 
        "root_rot": onp.array(root_rot),
        "fps": 30, 
    }
    data_dump["marker_link_names"] = marker_link_names
    data_dump["mjcf_joint_names"] = mjcf_joint_names
    import joblib
    joblib.dump(data_dump, "output/test.pkl")


if __name__ == "__main__":
    main()


