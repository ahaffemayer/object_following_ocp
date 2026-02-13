"""Visualization of the grasp trajectory in the robot frame (if the transformation of the camera to robot's world is identity)
"""

import pathlib

from robomeshcat import Object, Robot, Scene

from object_following_ocp.data_loader import DataLoader
from object_following_ocp.robot_loader import load_reduced_panda

if __name__ == "__main__":
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json")
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json")

    dataloader = DataLoader(object_trajectory_path=object_traj_path,
                            scales_path=scale_path,
                            load_grasps=True)

    object_trajectory_in_camera_frame = dataloader.to_trajectory_SE3()

    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    vdata = vmodel.createData()

    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )
    scene = Scene()
    scene.add_robot(robot=robot)
    object_info = dataloader.object_info
    print(object_info)
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

    best_grasp = dataloader.best_grasp_SE3
    print(best_grasp)

    for k, pose_data in enumerate(object_trajectory_in_camera_frame):
        color = [0.0, 1.0, 0.0] if k == 0 else [0.5, 0.5, 0.5]

        scene.add_object(
            Object.create_sphere(
                radius=0.01, name=f"target_{k}", color=color)
        )

        grasp_pose = pose_data * best_grasp
        scene[f"target_{k}"].pos[:] = grasp_pose.translation

        o.pose = pose_data.homogeneous
        input()
