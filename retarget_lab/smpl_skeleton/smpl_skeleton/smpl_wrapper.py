# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import torch
import numpy as np
import os.path as osp
from .body_models import SMPL as SMPL
SMPL_BONE_ORDER_NAMES = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']
SMPL_EE_NAMES = ["L_Ankle", "R_Ankle", "L_Wrist", "R_Wrist", "Head"]
JOINST_TO_USE = np.array([
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    37,
])


class SMPL_Parser(SMPL):
    def __init__(self, create_transl=False, *args, **kwargs):
        super(SMPL_Parser, self).__init__(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.joint_names = SMPL_BONE_ORDER_NAMES

        self.joint_axes = {x: np.identity(3) for x in self.joint_names}
        self.joint_dofs = {x: ["x", "y", "z"] for x in self.joint_names}
        self.joint_range = {x: np.hstack([np.ones([3, 1]) * -np.pi, np.ones([3, 1]) * np.pi]) for x in self.joint_names}
        self.joint_range["L_Elbow"] *= 4
        self.joint_range["R_Elbow"] *= 4
        self.joint_range["L_Shoulder"] *= 4
        self.joint_range["R_Shoulder"] *= 4

        # self.contype = {1: self.joint_names}
        # self.conaffinity = {1: self.joint_names}

        self.zero_pose = torch.zeros(1, 72).float()

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL_Parser, self).forward(*args, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_trans=None):
        """
        Pose should be batch_size x 72
        """
        if pose.shape[1] != 72:
            pose = pose.reshape(-1, 72)
        pose = pose.float()
        smpl_output = self.forward(
            transl=th_trans,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3],
        )
        joints = smpl_output["joints"][:, :24]
        # joints = smpl_output.joints[:,JOINST_TO_USE]
        return joints
