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
q[joint_names.index("l_shoulder_roll_joint")] = np.pi / 2.0
q[joint_names.index("r_shoulder_roll_joint")] = -np.pi / 2.0

tg = chain.forward_kinematics(q)
positions = []
for i in range(len(link_names)):
    m = tg[link_names[i]].get_matrix()
    # import ipdb; ipdb.set_trace();
    pos = np.array(m[0, :3, 3])
    positions.append(pos) 
positions = np.array(positions)
rotation = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
positions = rotation.apply(positions)
print(positions)