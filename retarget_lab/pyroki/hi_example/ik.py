"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import yourdfpy

import pyroki_snippets as pks


def main():
    """Main function for basic IK."""

    # urdf = load_robot_description("panda_description")
    urdf = yourdfpy.URDF.load("assets/urdf/hi/urdf/hi_23dof_250401_rl.urdf", mesh_dir="assets/urdf/hi/meshes")

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    target_link_names = ["left_hand_link", "right_hand_link", "l_ankle_roll_link", "r_ankle_roll_link", "base_link"]
    ik_target0 = server.scene.add_transform_controls(
        "/ik_target0", scale=0.2, position=(0.0, 0.3, 0.0), wxyz=(1, 0, 0, 0), active_axes = [True, True, False]
    )
    ik_target1 = server.scene.add_transform_controls(
        "/ik_target1", scale=0.2, position=(0.0, -0.3, 0.0), wxyz=(1, 0, 0, 0)
    )
    ik_target2 = server.scene.add_transform_controls(
        "/ik_target2", scale=0.2, position=(0.0, 0.3, -0.3), wxyz=(1, 0, 0, 0)
    )
    ik_target3 = server.scene.add_transform_controls(
        "/ik_target3", scale=0.2, position=(0.0, -0.3, -0.3), wxyz=(1, 0, 0, 0)
    )
    ik_target4 = server.scene.add_transform_controls(
        "/ik_target4", scale=0.2, position=(0.0, 0.0, 0.0), wxyz=(1, 0, 0, 0)
    )


    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik_with_multiple_targets(
            robot=robot,
            target_link_names=target_link_names,
            target_positions=np.array([ik_target0.position, ik_target1.position, ik_target2.position, ik_target3.position, ik_target4.position]),
            target_wxyzs=np.array([ik_target0.wxyz, ik_target1.wxyz, ik_target2.wxyz, ik_target3.wxyz, ik_target4.wxyz]),
        )


        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
