import pytorch_kinematics as pk
from param import robot_config
import torch
import numpy as np

chain = pk.build_chain_from_urdf(open(robot_config.urdf_file_path, mode="rb").read())
joint_names = chain.get_joint_parameter_names()
link_names = chain.get_link_names()
print(len(joint_names))
th = torch.zeros(len(joint_names))
tg = chain.forward_kinematics(th)["l_shoulder_roll_link"]
m = tg.get_matrix()
pos = np.array(m[0, :3, 3])
import ipdb; ipdb.set_trace();
