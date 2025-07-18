"""Mobile IK

Same as 01_basic_ik.py, but with a mobile base!
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks
import yourdfpy


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
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_l_hand = server.scene.add_transform_controls(
        "/left_hand", scale=0.2, position=(0.147, 0.187, 0.), wxyz=(1, 0, 0, 0), active_axes= [True, True, False]
    )
    ik_l_ankle = server.scene.add_transform_controls(
        "/left_foot", scale=0.2, position=(-0.530, 0.099, 0.089), wxyz=(0.707, 0.0, 0.707, 0.0),
    )
    ik_base = server.scene.add_transform_controls(
        "/fuck", scale=0.2, position=(0.138, 0.0, 0.318), wxyz=(0.707, 0.0, 0.707, 0.0), active_axes= [True, False, True]
    ) # 名称用base还不行

    target_link_names = ["left_hand_link", "right_hand_link", "l_ankle_roll_link", "r_ankle_roll_link", "base_link"]

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    cfg = np.array(robot.joint_var_cls(0).default_factory())

    while True:
        # Solve IK.
        start_time = time.time()
        
        # base_position = ik_base.position
        base_position = np.array([0.138, 0.0, 0.15 + 0.168 * 0.5 * (np.sin(2 * 3.14 * 0.5 * start_time) + 1)])

        ik_r_hand_pos = np.array([ik_l_hand.position[0], -ik_l_hand.position[1], ik_l_hand.position[2]])
        ik_r_ankle_pos = np.array([ik_l_ankle.position[0], -ik_l_ankle.position[1], ik_l_ankle.position[2]])
        # import ipdb; ipdb.set_trace();
        target_positions=np.array([ik_l_hand.position, ik_r_hand_pos,
                                    ik_l_ankle.position, ik_r_ankle_pos,
                                    base_position])
        target_wxyzs = np.array([ik_l_hand.wxyz, ik_l_hand.wxyz, \
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

        # print("base_pos:", base_pos)
        # print("base_wxyz:", base_wxyz)
        # print("cfg:", cfg)

        # print("ik_hand", ik_l_hand.position)
        # print("ik_l_ankle", ik_l_ankle.position)
        # print("ik_base", ik_base.position)

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        # Update visualizer.
        urdf_vis.update_cfg(cfg)
        base_frame.position = np.array(base_pos)
        base_frame.wxyz = np.array(base_wxyz)


if __name__ == "__main__":
    main()


