import numpy as np
import os
import sys
import torch

import time
import numpy as np
import os
import joblib

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    Vector3dVector = o3d.utility.Vector3dVector
    Vector3iVector = o3d.utility.Vector3iVector
    Vector2iVector = o3d.utility.Vector2iVector
    TriangleMesh = o3d.geometry.TriangleMesh
except Exception as e:
    print(e)
    print("run pip install open3d for vis.")
    o3d = None

def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if colors is not None:
        colors = np.array(colors)
        # mesh.vertex_colors = Vector3dVector(colors)
        mesh.paint_uniform_color(colors)
    else:
        r_c = np.random.random(3)
        mesh.paint_uniform_color(r_c)
    return mesh

def vis_mesh_o3d(vertices, faces):
    mesh = create_mesh(vertices, faces)
    # min_y = -mesh.get_min_bound()[1]
    # mesh.translate([0, min_y, 0])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]) # 加上坐标系

    o3d.visualization.draw_geometries([coordinate_frame, mesh])

def vis_mesh_o3d_with_joints(vertices, faces, joints, scale = 1.0):
    meshes = []
    mesh = create_mesh(vertices, faces)
    meshes.append(mesh)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]) # 加上坐标系
    meshes.append(coordinate_frame)
    for i in range(joints.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        color = [0.8, 0.2, 0.2] 
        sphere.paint_uniform_color(color)
        sphere.translate(joints[i])        
        meshes.append(sphere)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2 * scale, origin=joints[i]) # 加上坐标系
        meshes.append(coordinate_frame)

    o3d.visualization.draw_geometries(meshes)

def vis_positions_o3d(positions):
    meshes = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]) # 加上坐标系
    meshes.append(coordinate_frame)
    for i in range(positions.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        color = [0.8, 0.2, 0.2] 
        sphere.paint_uniform_color(color)
        sphere.translate(positions[i])        
        meshes.append(sphere)
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=joints[i]) # 加上坐标系
        # meshes.append(coordinate_frame)
    o3d.visualization.draw_geometries(meshes)

def get_positions_o3d(positions, color = [0.8, 0.2, 0.2], radius = 0.01):
    meshes = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]) # 加上坐标系
    meshes.append(coordinate_frame)
    for i in range(positions.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius) 
        sphere.paint_uniform_color(color)
        sphere.translate(positions[i])        
        meshes.append(sphere)
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=joints[i]) # 加上坐标系
        # meshes.append(coordinate_frame)
    return meshes

def vis_positions_target_o3d(positions, position):
    meshes = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]) # 加上坐标系
    meshes.append(coordinate_frame)
    for i in range(positions.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        color = [0.8, 0.2, 0.2] 
        sphere.paint_uniform_color(color)
        sphere.translate(positions[i])        
        meshes.append(sphere)
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=joints[i]) # 加上坐标系
        # meshes.append(coordinate_frame)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    color = [0.0, 0.0, 1.0] 
    sphere.paint_uniform_color(color)
    sphere.translate(position)        
    meshes.append(sphere)

    o3d.visualization.draw_geometries(meshes)


def vis_sleleton_o3d(positions, parents):
    assert positions.shape[0] == parents.shape[0]
    meshes = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]) # 加上坐标系
    meshes.append(coordinate_frame)
    lines = []
    for i in range(positions.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        color = [0.8, 0.2, 0.2] 
        sphere.paint_uniform_color(color)
        sphere.translate(positions[i])
        meshes.append(sphere)

        if (parents[i] != -1):
            lines.append([i, parents[i]])

    points = np.array(positions)
    lines = np.array(lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    colors = np.zeros((lines.shape[0], 3))
    colors[:, 0] = 1.0
    colors[:, 1] = 0.0
    colors[:, 2] = 0.0

    # colors = np.array([[1, 0, 0]])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    meshes.append(line_set)

    o3d.visualization.draw_geometries(meshes)


def get_sleleton_o3d(positions, parents):
    assert positions.shape[0] == parents.shape[0]
    meshes = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]) # 加上坐标系
    meshes.append(coordinate_frame)
    lines = []
    for i in range(positions.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        color = [0.8, 0.2, 0.2] 
        sphere.paint_uniform_color(color)
        sphere.translate(positions[i])
        meshes.append(sphere)

        if (parents[i] != -1):
            lines.append([i, parents[i]])

    points = np.array(positions)
    lines = np.array(lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    colors = np.zeros((lines.shape[0], 3))
    colors[:, 0] = 1.0
    colors[:, 1] = 0.0
    colors[:, 2] = 0.0

    # colors = np.array([[1, 0, 0]])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    meshes.append(line_set)

    return meshes

    # o3d.visualization.draw_geometries(meshes)


def create_line_checkerboard_ground(size, num_cells):
    # 计算每个方格的大小
    cell_size = size / num_cells

    # 初始化顶点和线条索引列表
    vertices = []
    lines = []

    # 生成水平方向的线
    for i in range(num_cells + 1):
        y = -size / 2 + i * cell_size
        start_point = [-size / 2, y, 0]
        end_point = [size / 2, y, 0]
        vertices.extend([start_point, end_point])
        line_index = len(vertices) - 2
        lines.append([line_index, line_index + 1])

    # 生成垂直方向的线
    for j in range(num_cells + 1):
        x = -size / 2 + j * cell_size
        start_point = [x, -size / 2, 0]
        end_point = [x, size / 2, 0]
        vertices.extend([start_point, end_point])
        line_index = len(vertices) - 2
        lines.append([line_index, line_index + 1])

    # 创建线条集
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(vertices))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))

    return line_set

def get_coord_mesh(origin = [0, 0, 0], xyzw = [0, 0, 0, 1], size = 0.5):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin = origin
    )
    rot = Rotation.from_quat(xyzw)
    R = rot.as_matrix()
    coord_frame.rotate(R, center = origin)
    return coord_frame

#################################################################################################
def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
#################################################################################################
from scipy.spatial.transform import Rotation

def rotate_vector_around_axis(v, k, theta):
    """
    v: 待旋转
    k: 旋转轴
    theta: 旋转角度
    """
    k = k / np.linalg.norm(k)
    v_rot = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
    return v_rot


def get_single_toe_rotation(l_toe, l_ankle, l_knee, theta = -20 * np.pi / 180):
    """
    输入的position格式:
    [link, 3]
    link顺序: l_toe, l_ankle, l_knee,
    输出batch格式: [4]
    link顺序: l_toe
    """
    v1 = l_toe - l_ankle
    v2 = l_knee - l_ankle
    ny = np.cross(v2, v1)
    ny = ny / np.linalg.norm(ny)
    
    nx = rotate_vector_around_axis(v1, ny, theta)
    nx = nx / np.linalg.norm(nx)

    R = np.column_stack((nx, ny, np.cross(nx, ny)))
    rot = Rotation.from_matrix(R)
    xyzw = rot.as_quat()  # 格式为 [x, y, z, w]
    return xyzw

def get_toe_rotation(positions: np.array, theta = -20 * np.pi / 180):
    """
    输入的position格式:
    [batch, link, 3]
    link顺序: l_toe, l_ankle, l_knee, r_toe, r_ankle, r_knee, 
    输出batch格式: [batch, link, 4]
    link顺序: l_toe, r_toe
    """
    xyzws = []
    for i in range(positions.shape[0]):
        l_toe = positions[i, 0, :]
        l_ankle = positions[i, 1, :]
        l_knee = positions[i, 2, :]

        r_toe = positions[i, 3, :]
        r_ankle = positions[i, 4, :]
        r_knee = positions[i, 5, :]
        left_xyzw = get_single_toe_rotation(l_toe, l_ankle, l_knee, theta)
        right_xyzw = get_single_toe_rotation(r_toe, r_ankle, r_knee, theta)
        xyzw = np.array([left_xyzw, right_xyzw])
        xyzws.append(xyzw)
    return np.array(xyzws)


