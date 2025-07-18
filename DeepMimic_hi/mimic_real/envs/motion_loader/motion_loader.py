import torch
import json
import numpy as np
import isaaclab.utils.math as math_utils
import pytorch_kinematics as pk

def compute_angular_velocity(r_wxyz: torch.Tensor, fps: float):
    time_delta = 1.0 / fps
    diff_quat_data = torch.zeros_like(r_wxyz).to(r_wxyz.device)
    diff_quat_data[:, 0] = 1.0
    diff_quat_data[:-1, :] = math_utils.quat_mul(r_wxyz[1:, :], math_utils.quat_inv(r_wxyz[:-1, :]))
    diff_angle = math_utils.axis_angle_from_quat(diff_quat_data)
    angular_velocity = diff_angle / time_delta
    
    return angular_velocity  

def lerp(val0, val1, blend): # 线性差值
    return (1.0 - blend) * val0 + blend * val1

def quaternion_lerp(q0, q1, fraction):
    qt = (1.0 - fraction) * q0 + fraction * q1
    qt = qt / torch.norm(qt)
    return qt

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

class URDFFK:
	def __init__(self, urdf_file_path, device):
		self.device = device
		self.chain = pk.build_chain_from_urdf(
			open(urdf_file_path, 
				mode="rb").read()).to(device = self.device)
		self.joint_names = self.chain.get_joint_parameter_names()
		self.link_names = self.chain.get_link_names()

	def fk(self, root_trans: torch.Tensor, root_wxyz: torch.Tensor, dof_pos: torch.Tensor, dof_pos_names: list):
		q = torch.zeros(dof_pos.shape[0], len(self.joint_names)).to(self.device)
		for i in range(len(dof_pos_names)):
			q[:, self.joint_names.index(dof_pos_names[i])] = dof_pos[:, i]
		tg = self.chain.forward_kinematics(q)
		positions = []
		wxyzs = []
		for i in range(len(self.link_names)):
			m = tg[self.link_names[i]].get_matrix()
			pos = m[:, :3, 3]
			positions.append(pos)
			wxyz = pk.matrix_to_quaternion(m[:, :3, :3])
			wxyzs.append(wxyz)
		positions = torch.stack(positions, dim=1)
		wxyzs = torch.stack(wxyzs, dim=1)

		root_wxyz_full = root_wxyz.unsqueeze(1).repeat(1, wxyzs.shape[1], 1)
		root_trans_full = root_trans.unsqueeze(1).repeat(1, positions.shape[1], 1)
		positions = quat_apply(root_wxyz_full, positions) + root_trans_full
		wxyzs = quat_mul(root_wxyz_full, wxyzs)
		return positions, wxyzs


class MotionLoader:
    def __init__(self, motion_file_path, 
                 simulator_joint_names, 
                 simulator_body_link_names = None, 
                 device = "cuda:0", 
                 add_static_frame: bool = False):
        self.device = device
        with open(motion_file_path, 'r') as f:
            loaded_data = json.load(f)
        self.root_trans = torch.tensor(loaded_data["root_trans"]).to(self.device)
        self.root_rot = torch.tensor(loaded_data["root_wxyz"]).to(self.device) # wxyz
        self.dof_pos = torch.tensor(loaded_data["dof_pos"]).to(self.device)
        self.fps = loaded_data["fps"]
        self.data_joint_names = loaded_data["data_joint_names"]
        if add_static_frame:
            self._add_static_frame(0.5)
        self._trans_dof_pos(simulator_joint_names)
        assert self.root_trans.shape[0] == self.root_rot.shape[0]
        assert self.root_trans.shape[0] == self.dof_pos.shape[0]
        self.frame_num = self.root_trans.shape[0]        
        self.record_time = self.frame_num / self.fps
        self.sync_fps()
        self._smoothen_data(0.0) 

        self.capture_points_link_names = loaded_data["target_link_names"]
        urdf_fk = URDFFK("mimic_real/assets/urdf/hi/urdf/hi_23dof_250401_rl.urdf", device=self.device)
        
        positions, wxyzs = urdf_fk.fk(self.root_trans, 
                                      self.root_rot, 
                                      self.dof_pos, 
                                      simulator_joint_names) # 此时关节顺序已经转换 
        
        target_link_index = []
        for name in self.capture_points_link_names:
            target_link_index.append(urdf_fk.link_names.index(name))  
        self.capture_points = positions[:, target_link_index]
        
        if simulator_body_link_names is not None:
            body_link_index = []
            for name in simulator_body_link_names:
                body_link_index.append(urdf_fk.link_names.index(name))
            self.body_link_pos = positions[:, body_link_index]
            self.body_link_wxyzs = wxyzs[:, body_link_index]
        self._compute_velocity()
            
    def _add_static_frame(self, static_time_second):
        add_frame_number = int(static_time_second * self.fps)
        
        root_trans_start = self.root_trans[0].repeat(add_frame_number, 1)
        root_rot_start = self.root_rot[0].repeat(add_frame_number, 1)
        dof_pos_start = self.dof_pos[0].repeat(add_frame_number, 1)

        root_trans_end = self.root_trans[-1].repeat(add_frame_number, 1)
        root_rot_end = self.root_rot[-1].repeat(add_frame_number, 1)
        dof_pos_end = self.dof_pos[-1].repeat(add_frame_number, 1)
        

        self.root_trans = torch.cat([root_trans_start, self.root_trans, root_trans_end], dim=0)
        self.root_rot = torch.cat([root_rot_start, self.root_rot, root_rot_end], dim=0)
        self.dof_pos = torch.cat([dof_pos_start, self.dof_pos, dof_pos_end], dim=0)
        
    def _trans_dof_pos(self, simulator_joint_names):
        joint_index = [] # 仿真器顺序的关节索引
        for joint_name in simulator_joint_names:
            joint_index.append(self.data_joint_names.index(joint_name))
        self.dof_pos[:] = self.dof_pos[:, joint_index]
 
    def get_frame_at_time(self, time, frame_num: int, fps):
        assert time >= 0.0
        assert time * fps < frame_num
        idx_low = int(np.floor(time * fps))
        idx_high = int(np.ceil(time * fps))
        blend = float(time * fps) - float(idx_low)
        return idx_low, idx_high, blend

    def sync_fps(self, desired_fps = 50):
        old_fps = self.fps
        old_dt = 1.0 / old_fps
        old_len = self.root_trans.shape[0]
        old_time = (old_len - 1) * old_dt
        
        self.time = old_time
        self.fps = desired_fps
        self.dt = 1.0 / desired_fps
        self.frame_num = int(np.floor(self.time / self.dt))
        self.record_time = (self.frame_num - 1) * self.dt
        
        old_root_trans = self.root_trans
        old_root_rot = self.root_rot
        old_dof_pos = self.dof_pos

        self.root_trans = torch.zeros(self.frame_num, 3).to(self.device)
        self.root_rot = torch.zeros(self.frame_num, 4).to(self.device)
        self.dof_pos = torch.zeros(self.frame_num, old_dof_pos.shape[1]).to(self.device)
        
        for i in range(self.frame_num):
            time = float(i) / float(self.fps)
            old_idx_low, old_idx_high, blend = self.get_frame_at_time(time, old_len, old_fps)
            self.dof_pos[i] = lerp(old_dof_pos[old_idx_low],
                                         old_dof_pos[old_idx_high],
                                         blend)
            self.root_trans[i] = lerp(old_root_trans[old_idx_low],
                                            old_root_trans[old_idx_high],
                                            blend)
            # self.root_rot[i] = old_root_rot[old_idx_low]
            self.root_rot[i] = quaternion_lerp(old_root_rot[old_idx_low],
                                                     old_root_rot[old_idx_high],
                                                     blend)

    def _smoothen_data(self, alpha): # alpha (0, 1)
        for i in range(self.frame_num - 1):
            self.root_trans[i + 1] = lerp(self.root_trans[i], self.root_trans[i + 1], 1 - alpha)
            self.dof_pos[i + 1] = lerp(self.dof_pos[i], self.dof_pos[i + 1], 1 - alpha)
            self.root_rot[i + 1] = quaternion_lerp(self.root_rot[i], self.root_rot[i + 1], 1 - alpha)

    def _compute_velocity(self):
        self.capture_points_vel = torch.zeros_like(self.capture_points)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.root_vel = torch.zeros_like(self.root_trans)
        self.root_ang_vel = torch.zeros_like(self.root_vel)
        
        self.body_link_vel = torch.zeros_like(self.body_link_pos)
        self.body_link_ang_vel = torch.zeros_like(self.body_link_vel)
        
        for i in range(1, self.capture_points.shape[0] - 1):
            self.capture_points_vel[i] = (self.capture_points[i + 1] - self.capture_points[i - 1]) * self.fps / 2.0 # 中点微分
            self.dof_vel[i] = (self.dof_pos[i + 1] - self.dof_pos[i - 1]) * self.fps / 2.0 # 中点微分
            self.root_vel[i] = (self.root_trans[i + 1] - self.root_trans[i - 1]) * self.fps / 2.0 # 中点微分
            self.body_link_vel[i] = (self.body_link_pos[i + 1] - self.body_link_pos[i - 1]) * self.fps / 2.0 # 中点微分
            
        self.root_ang_vel = compute_angular_velocity(self.root_rot, self.fps) # 前向微分 没什么用，已经包含在内了
        self.body_link_ang_vel = compute_angular_velocity(self.body_link_wxyzs, self.fps) # 前向微分

    
    def get_frame_ids_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        self.frame_ids = (phase * (self.frame_num - 1)).to(torch.long)
        return self.frame_ids
    
#### dof相关 --------------------------------------------
    def get_dof_pos_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.dof_pos[frame_ids]

    def get_dof_vel_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.dof_vel[frame_ids]
    
#### 根节点link相关 ---------------------------------------      
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
#### body_link相关 ---------------------------------------
    def get_body_pos_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.body_link_pos[frame_ids]
    
    def get_body_wxyz_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.body_link_wxyzs[frame_ids]
    
    def get_body_vel_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.body_link_vel[frame_ids]
    
    def get_body_omega_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.body_link_ang_vel[frame_ids]
#### 跟踪点相关 ---------------------------------------------
    def get_capture_points_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.capture_points[frame_ids]
    
    def get_capture_points_vel_batch(self, phase: torch.Tensor):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        return self.capture_points_vel[frame_ids]

    def get_local_capture_points_batch(self, phase: torch.Tensor, root_marker_name: str = "waist_link"):
        assert len(phase.shape) == 1
        frame_ids = self.get_frame_ids_batch(phase)
        global_capture_points_batch = self.capture_points[frame_ids]
        root_marker_index = self.capture_points_link_names.index(root_marker_name)
        local_capture_points_batch = global_capture_points_batch \
            - global_capture_points_batch[:, root_marker_index: root_marker_index + 1, :]
        return local_capture_points_batch
#### ------------------------------------------------------
if __name__ == "__main__":
    urdf_file_path = "mimic_real/data/hi/crawl.json"
    simulator_joint_names = ['l_hip_pitch_joint', 'r_hip_pitch_joint', 'waist_joint', 'l_hip_roll_joint', 'r_hip_roll_joint', 'l_shoulder_pitch_joint', 'r_shoulder_pitch_joint', 'l_thigh_joint', 'r_thigh_joint', 'l_shoulder_roll_joint', 'r_shoulder_roll_joint', 'l_calf_joint', 'r_calf_joint', 'l_upper_arm_joint', 'r_upper_arm_joint', 'l_ankle_pitch_joint', 'r_ankle_pitch_joint', 'l_elbow_joint', 'r_elbow_joint', 'l_ankle_roll_joint', 'r_ankle_roll_joint', 'l_wrist_joint', 'r_wrist_joint']
    loader = MotionLoader(urdf_file_path, 
                          simulator_joint_names 
                          )