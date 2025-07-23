import torch
import json
import numpy as np

import isaaclab.utils.math as math_utils
def compute_angular_velocity(r: torch.Tensor, time_delta: float):
    # r : xyzw
    r_wxyz = r[:, [3, 0, 1, 2]]

    diff_quat_data = torch.zeros_like(r_wxyz).to(r_wxyz)
    diff_quat_data[:, 0] = 0.0

    diff_quat_data[:-1, :] = math_utils.quat_mul(r_wxyz[1:, :], math_utils.quat_inv(r_wxyz[:-1, :]))
    diff_angle = math_utils.axis_angle_from_quat(diff_quat_data)
    angular_velocity = diff_angle / time_delta
    return angular_velocity  

class MotionLoader:
    def __init__(self, motion_file_path, simulator_joint_names, device = "cuda:0", add_static_frame: bool = False):
        self.device = device
        with open(motion_file_path, 'r') as f:
            loaded_data = json.load(f)
        self.root_trans = torch.tensor(loaded_data["root_trans"]).to(self.device)
        self.root_rot = torch.tensor(loaded_data["root_wxyz"]).to(self.device)
        # self.root_rot[:] = self.root_rot[:, [3, 0, 1, 2]] # xyzw->wxyz
        self.capture_points = torch.tensor(loaded_data["target_link_pos"]).to(self.device)
        self.dof_pos = torch.tensor(loaded_data["dof_pos"]).to(self.device)
        self.fps = loaded_data["fps"]
        print("mocap frame rate: ", self.fps)
        if add_static_frame:
            self._add_static_frame(0.5)
        self.frame_num = self.root_trans.shape[0]
        print("mocap frame number: ", self.frame_num)
        print("mocap record time: {:.2f} [s]".format(self.frame_num * (1.0 / float(self.fps))))
        self.record_time = float(self.frame_num) / float(self.fps)

        self.capture_points_link_names = loaded_data["target_link_names"]
        self.data_joint_names = loaded_data["data_joint_names"]
        self._trans_dof_pos(simulator_joint_names)
        self._smoothen_data(0.5)
        self._compute_velocity()

    def _smoothen_data(self, alpha): # alpha (0, 1)
        for i in range(self.capture_points.shape[0] - 1):
            self.capture_points[i + 1] = (1 - alpha) * self.capture_points[i + 1] + alpha * self.capture_points[i]
            # self.dof_pos[i + 1] = (1 - alpha) * self.dof_pos[i + 1] + alpha * self.dof_pos[i]

    # body_rotation: link的角位置 TODO

    def _compute_velocity(self):
        # body_angular_velocity: body角速度 TODO
        self.capture_points_vel = torch.zeros_like(self.capture_points)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.root_vel = torch.zeros_like(self.root_trans)
        self.root_ang_vel = torch.zeros_like(self.root_vel)
        for i in range(1, self.capture_points.shape[0] - 1):
            self.capture_points_vel[i] = (self.capture_points[i + 1] - self.capture_points[i - 1]) * self.fps / 2.0
            self.dof_vel[i] = (self.dof_pos[i + 1] - self.dof_pos[i - 1]) * self.fps / 2.0 # 中点微分
            self.root_vel[i] = (self.root_trans[i + 1] - self.root_trans[i - 1]) * self.fps / 2.0 # 中点微分
        self.root_ang_vel = compute_angular_velocity(self.root_rot, self.fps) # 前向微分

    def _trans_dof_pos(self, simulator_joint_names):
        joint_index = [] # 仿真器顺序的关节索引

        for joint_name in simulator_joint_names:
            joint_index.append(self.data_joint_names.index(joint_name))
        self.dof_pos[:] = self.dof_pos[:, joint_index]

    def _add_static_frame(self, static_time_second):
        add_frame_number = int(static_time_second * self.fps)
        root_trans_start = self.root_trans[0].repeat(add_frame_number, 1)
        root_rot_start = self.root_rot[0].repeat(add_frame_number, 1)
        # pose_aa_start = self.pose_aa[0].repeat(add_frame_number, 1, 1)
        capture_points_start = self.capture_points[0].repeat(add_frame_number, 1, 1)
        dof_pos_start = self.dof_pos[0].repeat(add_frame_number, 1)

        root_trans_end = self.root_trans[-1].repeat(add_frame_number, 1)
        root_rot_end = self.root_rot[-1].repeat(add_frame_number, 1)
        # pose_aa_end = self.pose_aa[-1].repeat(add_frame_number, 1, 1)
        capture_points_end = self.capture_points[-1].repeat(add_frame_number, 1, 1)
        dof_pos_end = self.dof_pos[-1].repeat(add_frame_number, 1)
        

        self.root_trans = torch.cat([root_trans_start, self.root_trans, root_trans_end], dim=0)
        self.root_rot = torch.cat([root_rot_start, self.root_rot, root_rot_end], dim=0)
        # self.pose_aa = torch.cat([pose_aa_start, self.pose_aa, pose_aa_end], dim=0)
        self.capture_points = torch.cat([capture_points_start, self.capture_points, capture_points_end], dim=0)
        self.dof_pos = torch.cat([dof_pos_start, self.dof_pos, dof_pos_end], dim=0)
        
    def get_dof_pos(self):
        return self.dof_pos
    
    # def get_pose_aa(self):
    #     return self.pose_aa
    
    def get_capture_points(self):
        return self.capture_points

    def get_dof_pos_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.dof_pos[frame_ids]

    # def get_pose_aa_batch(self, phase: torch.Tensor):
    #     assert len(phase.shape) == 1
    #     frame_ids = self.get_frame_ids_batch(phase)
    #     return self.pose_aa[frame_ids]
    
    def get_capture_points_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.capture_points[frame_ids]
    
    def get_local_capture_points_batch(self, phase: torch.Tensor, root_marker_name: str = "waist_link"):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        global_capture_points_batch = self.capture_points[frame_ids]
        root_marker_index = self.capture_points_link_names.index(root_marker_name)
        local_capture_points_batch = global_capture_points_batch \
            - global_capture_points_batch[:, root_marker_index: root_marker_index + 1, :]
        return local_capture_points_batch
    
    def get_root_trans_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.root_trans[frame_ids]
    
    def get_root_rot_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.root_rot[frame_ids]
    
    def get_root_vel_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.root_vel[frame_ids]
    
    def get_root_omega_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.root_ang_vel[frame_ids]

    def get_body_vel_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.capture_points_vel[frame_ids]

    def get_dof_vel_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.dof_vel[frame_ids]
    
    def get_frame_ids_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        self.frame_ids = (phase * (self.frame_num - 1)).to(torch.long)
        return self.frame_ids
    
    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1
    
    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.record_time
        n = self.frame_num
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)
