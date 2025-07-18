# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import torch
import numpy as np
import os.path as osp
from smplx import SMPL as _SMPL
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


class SMPL_Parser(_SMPL):

    def __init__(self, create_transl=False, *args, **kwargs):
        """SMPL model constructor
        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """
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

        self.contype = {1: self.joint_names}
        self.conaffinity = {1: self.joint_names}

        # self.contype = {
        #     3: ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee','R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Neck', 'Head','L_Thorax',  'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax',  'R_Elbow', 'R_Wrist', 'R_Hand'],
        #     1: ['Chest', "L_Shoulder", "R_Shoulder"]
        #     }

        # self.conaffinity = {
        #     1: ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee','R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Neck', 'Head','L_Thorax',  'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax',  'R_Elbow', 'R_Wrist', 'R_Hand'],
        #     3: ['Chest', "L_Shoulder", "R_Shoulder"]
        # }

        self.zero_pose = torch.zeros(1, 72).float()

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL_Parser, self).forward(*args, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Pose should be batch_size x 72
        """
        if pose.shape[1] != 72:
            pose = pose.reshape(-1, 72)

        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

            if th_betas.shape[-1] == 16:
                th_betas = th_betas[:, :10]

        batch_size = pose.shape[0]

        smpl_output = self.forward(
            betas=th_betas,
            transl=th_trans,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3],
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints[:, :24]
        # joints = smpl_output.joints[:,JOINST_TO_USE]
        return vertices, joints

    def get_offsets(self, v_template=None,   zero_pose=None, betas=torch.zeros(1, 10).float()):
        with torch.no_grad():
            if zero_pose is None:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)
            else:
                verts, Jtr = self.get_joints_verts(zero_pose, th_betas=betas)
            verts_np = verts.detach().cpu().numpy()
            jts_np = Jtr.detach().cpu().numpy()
            parents = self.parents.cpu().numpy()
            offsets_smpl = [np.array([0, 0, 0])]
            for i in range(1, len(parents)):
                p_id = parents[i]
                p3d = jts_np[0, p_id]
                curr_3d = jts_np[0, i]
                offset_curr = curr_3d - p3d
                offsets_smpl.append(offset_curr)
            offsets_smpl = np.array(offsets_smpl)
            joint_names = self.joint_names
            joint_pos = Jtr[0].numpy()
            smpl_joint_parents = self.parents.cpu().numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in enumerate(smpl_joint_parents)}
            parents_dict = {joint_names[i]: joint_names[parents[i]] for i in range(len(joint_names))}
            channels = ["z", "y", "x"]
            skin_weights = self.lbs_weights.numpy()
            return (verts[0], jts_np[0], skin_weights, self.joint_names, joint_offsets, parents_dict, channels, self.joint_range)
