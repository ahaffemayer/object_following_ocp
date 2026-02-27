"""This visualization is to see the trajectory of the object in the camera frame."""

import pathlib

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

    for k, pose_data in enumerate(object_trajectory_in_camera_frame):
        color = [0.0, 1.0, 0.0] if k == 0 else [0.5, 0.5, 0.5]

        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"target_{k}", color=color)
        )
        scene[f"target_{k}"].pos[:] = pose_data.translation

        o.pose = pose_data.homogeneous
        input()
