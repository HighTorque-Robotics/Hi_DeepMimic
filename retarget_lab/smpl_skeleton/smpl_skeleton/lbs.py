# -*- coding: utf-8 -*-

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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import Tuple, NewType
import numpy as np

import torch
import torch.nn.functional as F
Tensor = NewType("Tensor", torch.Tensor)

DEFALUT_J = torch.tensor([[[-1.7951e-03, -2.2333e-01,  2.8219e-02],
         [ 6.7725e-02, -3.1474e-01,  2.1404e-02],
         [-6.9466e-02, -3.1386e-01,  2.3899e-02],
         [-4.3279e-03, -1.1437e-01,  1.5228e-03],
         [ 1.0200e-01, -6.8994e-01,  1.6908e-02],
         [-1.0776e-01, -6.9642e-01,  1.5049e-02],
         [ 1.1591e-03,  2.0810e-02,  2.6153e-03],
         [ 8.8406e-02, -1.0879e+00, -2.6785e-02],
         [-9.1982e-02, -1.0948e+00, -2.7263e-02],
         [ 2.6161e-03,  7.3732e-02,  2.8040e-02],
         [ 1.1476e-01, -1.1437e+00,  9.2503e-02],
         [-1.1735e-01, -1.1430e+00,  9.6085e-02],
         [-1.6228e-04,  2.8760e-01, -1.4817e-02],
         [ 8.1461e-02,  1.9548e-01, -6.0498e-03],
         [-7.9143e-02,  1.9257e-01, -1.0575e-02],
         [ 4.9896e-03,  3.5257e-01,  3.6532e-02],
         [ 1.7244e-01,  2.2595e-01, -1.4918e-02],
         [-1.7516e-01,  2.2512e-01, -1.9719e-02],
         [ 4.3205e-01,  2.1318e-01, -4.2374e-02],
         [-4.2890e-01,  2.1179e-01, -4.1119e-02],
         [ 6.8128e-01,  2.2216e-01, -4.3545e-02],
         [-6.8420e-01,  2.1956e-01, -4.6679e-02],
         [ 7.6533e-01,  2.1400e-01, -5.8491e-02],
         [-7.6882e-01,  2.1344e-01, -5.6994e-02]]])

def lbs(
    pose: Tensor,
    parents: Tensor,
    pose2rot: bool = True,
    default_joint_pos: Tensor = None
) -> Tensor:
    
    batch_size = pose.shape[0]
    dtype = pose.dtype

    # Get the joints
    # NxJx3 array
    if default_joint_pos is None:    
        J = DEFALUT_J
    else:
        J = default_joint_pos
    J = J.repeat(pose.shape[0], 1, 1)
    # 3. Add pose blend shapes
    # N x J x 3 x 3
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
        # (N x P) x (P, V * 3) -> N x V x 3
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)
    
    # import ipdb; ipdb.set_trace();
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    return J_transformed


def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms
