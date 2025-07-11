import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


def define_sphere_markers(radius: float = 0.5) -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "green_sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "red_sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "blue_sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

from isaaclab.utils import math as math_utils

def define_cylinder_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=0.1,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)



def visualize_cylinder(visualizer: VisualizationMarkers, 
                    start_points: torch.Tensor, 
                    end_points: torch.Tensor, 
                    device,
                    arrow_thickness: float = 0.1):
    trans = (start_points + end_points) / 2
    direction = end_points - start_points

    default_direction = torch.zeros_like(direction)
    default_direction[:, 2] = 1.0
    
    normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True)  # arrow-direction
    axis = torch.cross(default_direction, normalized_direction, dim=-1)
    dot_prod_ = torch.sum(default_direction * normalized_direction, dim=-1)
    angle = torch.acos(torch.clamp(dot_prod_, -1.0, 1.0))
    quat = math_utils.quat_from_angle_axis(
        angle,
        axis,
    )
    # compute the scale to match the length and the sharp edge thickness.
    scales = torch.ones(start_points.shape[0], 3, device=device)
    scales[:, 0] = arrow_thickness
    scales[:, 1] = arrow_thickness
    scales[:, 2] = torch.norm(direction, dim=-1)
    visualizer.visualize(
        translations=trans,
        orientations=quat,
        scales=scales,
    )






