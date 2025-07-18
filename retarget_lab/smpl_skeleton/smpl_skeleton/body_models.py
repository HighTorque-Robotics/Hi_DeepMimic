#  -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import NewType, Optional, Dict, Union
import os
import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn as nn

from .lbs import lbs
Tensor = NewType("Tensor", torch.Tensor)


class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
        self,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        default_joint_pos: Tensor = None,
        **kwargs,
    ) -> None:
        """SMPL model constructor

        Parameters
        ----------        
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
        """
        super(SMPL, self).__init__()
        self.dtype = dtype
        self.joint_mapper = joint_mapper
        self.default_joint_pos = default_joint_pos
        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3],
                                                    dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient,
                                                         dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter("global_orient", global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose, dtype=dtype)
            self.register_parameter(
                "body_pose", nn.Parameter(default_body_pose,
                                          requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                "transl", nn.Parameter(default_transl, requires_grad=True))

        self.parents = torch.tensor([-1,  0,  0,  0,  
                                     1,  2,  3,  4,  
                                     5,  6,  7,  8, 
                                     9,  9,  9, 12, 
                                     13, 14, 16, 17, 
                                     18, 19, 20, 21])

    def name(self) -> str:
        return "SMPL"

    def forward(
        self,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        pose2rot: bool = True,
        **kwargs,
    ):
        """Forward pass for the SMPL model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3
            If given, ignore the member variable and use it as the global
            rotation of the body. Useful if someone wishes to predicts this
            with an external model. (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        batch_size = max(
            global_orient.shape[0] if not global_orient is None else 1,
            body_pose.shape[0] if not body_pose is None else 1,
        )

        global_orient = (global_orient if global_orient is not None else
                         match_dim(self.global_orient, batch_size))
        body_pose = (body_pose if body_pose is not None else match_dim(
            self.body_pose, batch_size))

        apply_trans = transl is not None or hasattr(self, "transl")

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(global_orient.shape[0], body_pose.shape[0])


        joints = lbs(
            full_pose,
            self.parents,
            pose2rot=pose2rot,
            default_joint_pos=self.default_joint_pos
        )

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)

        return {"global_orient": global_orient, 
                "body_pose": body_pose, 
                "joints": joints}


def match_dim(x, batch_size):
    if x.shape[0] == batch_size:
        return x
    elif x.shape[0] == 1:
        return x.repeat((batch_size, 1))
    else:
        import ipdb

        ipdb.set_trace()
        raise NotImplementedError()
