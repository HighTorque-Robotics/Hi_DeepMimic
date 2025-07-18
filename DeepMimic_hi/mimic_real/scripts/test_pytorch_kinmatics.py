import pytorch_kinematics as pk
import numpy as np
import torch
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
	def __init__(self, urdf_file_path):
		self.chain = pk.build_chain_from_urdf(
			open(urdf_file_path, 
				mode="rb").read())
		self.joint_names = self.chain.get_joint_parameter_names()
		self.link_names = self.chain.get_link_names()

	def fk(self, root_trans: torch.Tensor, root_wxyz: torch.Tensor, dof_pos: torch.Tensor, dof_pos_names: list):
		q = torch.zeros(dof_pos.shape[0], len(self.joint_names))
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


root_trans = [[0.6558521389961243,
      -0.5619995594024658,
      0.15271830558776855], [
      0.6599105000495911,
      -0.5650776028633118,
      0.15879790484905243
    ]]
root_wxyz = [[0.27662932872772217,
      -0.5565862655639648,
      0.266539990901947,
      0.7366437911987305],
	[
      0.2728867530822754,
      -0.572624146938324,
      0.27015209197998047,
      0.7243284583091736
    ]]
dof_pos = [[
      0.05120428279042244,
      -0.10396604984998703,
      -0.03658497333526611,
      -0.08024732768535614,
      0.7228206396102905,
      -0.277447372674942,
      -0.048934850841760635,
      -0.12113244086503983,
      0.011332527734339237,
      0.021476471796631813,
      0.7252943515777588,
      -0.2742130756378174,
      0.054143026471138,
      -0.3329581320285797,
      0.23019346594810486,
      0.07732829451560974,
      -0.3803337514400482,
      0.07762348651885986,
      -0.028605595231056213,
      -0.3785759210586548,
      0.02192317321896553,
      -0.2521342933177948,
      0.05922307074069977
	],
    [
      0.06948072463274002,
      -0.1085662990808487,
      -0.05259205400943756,
      -0.13726431131362915,
      0.5811499357223511,
      -0.29251354932785034,
      -0.07426959276199341,
      -0.12597660720348358,
      0.02200092375278473,
      0.06383006274700165,
      0.5831080675125122,
      -0.29882803559303284,
      0.0799097791314125,
      -0.4114428162574768,
      0.30625826120376587,
      0.0792817771434784,
      -0.6909247040748596,
      0.12126591056585312,
      0.014453906565904617,
      -0.5132234692573547,
      -4.522909875959158e-05,
      -0.4900011420249939,
      0.09720286726951599
    ]
	]
target_link_names = [
    "l_hip_pitch_link",
    "l_calf_link",
    "l_ankle_roll_link",
    "r_hip_pitch_link",
    "r_calf_link",
    "r_ankle_roll_link",
    "l_shoulder_pitch_link",
    "l_elbow_link",
    "left_hand_link",
    "r_shoulder_pitch_link",
    "r_elbow_link",
    "right_hand_link",
    "left_toe",
    "right_toe",
    "head_link"
]
dof_pos_names = ["waist_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "l_wrist_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
    "r_wrist_joint"]

urdf_fk = URDFFK("mimic_real/assets/urdf/hi/urdf/hi_23dof_250401_rl.urdf")
target_index = []
for name in target_link_names:
    target_index.append(urdf_fk.link_names.index(name))


# (N_batch, N_link, 3)
# (N_batch, N_link, 4)
root_trans = torch.Tensor(root_trans)
# root_trans = root_trans.repeat(7, 1)
root_wxyz = torch.Tensor(root_wxyz)
# root_wxyz = root_wxyz.repeat(7, 1)
dof_pos = torch.Tensor(dof_pos)
# dof_pos = dof_pos.repeat(7, 1)

positions, wxyzs = urdf_fk.fk(root_trans, root_wxyz, dof_pos, dof_pos_names) 

print("target positions:")
print(positions[:, target_index])
# print("target rotations:")
# print(wxyzs[:, target_index])
# positions = np.array(positions)
# print(positions.shape)
# rotation = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
# positions = rotation.apply(positions)
