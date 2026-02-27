"""This visualization is to see the trajectory of the object in the camera frame, in the right rotation."""

import pathlib

import numpy as np
import pinocchio as pin
from robomeshcat import Object, Scene

from object_following_ocp.data.data_loader import DataLoader
from object_following_ocp.robot.robot_loader import load_reduced_panda

if __name__ == "__main__":
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
    )
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json"
    )

    dataloader = DataLoader(
        object_trajectory_path=object_traj_path, scales_path=scale_path
    )

    object_trajectory_in_camera_frame = dataloader.to_trajectory_SE3()

    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    vdata = vmodel.createData()

    scene = Scene()
    object_info = dataloader.object_info

    # -----------------------------
    # Add object to scene (using paths from parser)
    # -----------------------------

    o = Object.create_mesh(
        path_to_mesh=object_info.mesh_path,
        name="robot/movable_obj",
        texture=object_info.texture_path,
        scale=object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    scene.add_object(o)

    elev_angle_deg = 25
    default_rot = pin.exp3(np.array([0, 0, np.deg2rad(90)])) @ pin.exp3(
        np.array([-np.pi / 2 - np.deg2rad(elev_angle_deg), 0, 0])
    )

    SE3_rot = pin.SE3(default_rot, np.array([0, 0, 0]))

    for k, pose_data in enumerate(object_trajectory_in_camera_frame):
        color = [0.0, 1.0, 0.0] if k == 0 else [0.5, 0.5, 0.5]

        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"target_{k}", color=color)
        )
        scene[f"target_{k}"].pos[:] = (SE3_rot * pose_data).translation

        o.pose = (SE3_rot * pose_data).homogeneous
        input()
