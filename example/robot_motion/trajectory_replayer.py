"""
Replay OCP trajectories from a saved trajectory file.

Usage:
    python replay_ocp.py <mesh_id>
"""

import pathlib
import pickle
import sys
import time

import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data_loader import ConfigLoader, DataLoader
from object_following_ocp.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.robot_loader import load_reduced_panda

OUTPUT_DIR = pathlib.Path("/mnt/user-data/outputs")


def main():
    # -------------------------
    # Load file
    # -------------------------
    if len(sys.argv) < 2:
        print("Usage: python replay_ocp.py <mesh_id>")
        print("\nAvailable files:")
        for f in sorted(OUTPUT_DIR.glob("*_kept_trajectories.pkl")):
            print(f"  {f.name}")
        sys.exit(1)

    mesh_id = sys.argv[1]
    filepath = OUTPUT_DIR / f"{mesh_id}_kept_trajectories.pkl"

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    trajectories = data["trajectories"]
    print(f"Loaded {len(trajectories)} trajectories for '{mesh_id}'")

    config_path = pathlib.Path(data["config_path"])
    robot_config = ConfigLoader.load(config_path)

    # -------------------------
    # Setup robot and scene
    # -------------------------
    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()

    scene = Scene()
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vmodel.createData(),
    )
    scene.add_robot(robot=robot)

    dataloader = DataLoader(
        object_trajectory_path=pathlib.Path(data["object_traj_path"]),
        scales_path=pathlib.Path(data["scale_path"]),
        load_grasps=True,
    )

    obj = Object.create_mesh(
        path_to_mesh=dataloader.object_info.mesh_path,
        name="robot/movable_obj",
        texture=dataloader.object_info.texture_path,
        scale=dataloader.object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    scene.add_object(obj)
    scene.camera_pos = [3, 0, 1]
    scene.camera_zoom = 2

    # -------------------------
    # Replay each trajectory
    # -------------------------
    for idx, traj in enumerate(trajectories):
        if "joint_trajectory_ocp" not in traj:
            print(
                f"\nTrajectory {idx + 1}/{len(trajectories)}: no OCP solution, skipping."
            )
            continue

        ocp_states = traj["joint_trajectory_ocp"]
        object_poses = traj["object_trajectory_poses"]
        tcp_poses = traj["tcp_trajectory_poses"]
        cam = traj["camera_translation"]

        print(f"\n{'=' * 50}")
        print(f"Trajectory {idx + 1}/{len(trajectories)}")
        print(f"Camera: {cam}")
        print(f"Frames: {len(ocp_states)}")
        print(f"{'=' * 50}")
        input("Press Enter to start...")

        grasp_config = GraspTransformConfig.from_robot_config(
            robot_config=robot_config,
            camera_translation=np.array(cam),
            grasp_correction_angle_deg=data["grasp_correction_angle_deg"],
            elevation_angle_deg=data["elevation_angle_deg"],
        )
        chain = GraspTransformChain(grasp_config)

        for k, xs in enumerate(ocp_states):
            q = np.array(xs[: rmodel.nq])
            robot[:] = q
            pin.framesForwardKinematics(rmodel, rdata, q)
            tcp_se3 = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
            object_se3 = chain.compute_object_pose_from_tcp(
                tcp_se3, dataloader.best_grasp_SE3
            )
            if k < len(object_poses):
                obj.pose = object_se3.homogeneous
                # obj.pose = np.array(object_poses[k])
                time.sleep(0.01)
            # input(
            #     f"  frame {k + 1}/{len(ocp_states)} — Enter to continue, Ctrl+C to stop"
            # )
        render = input("Render this trajectory? y/n")
        if "y" in render:
            with scene.video_recording(filename=f"{mesh_id}.mp4", fps=25):
                for k, xs in enumerate(ocp_states):
                    q = np.array(xs[: rmodel.nq])
                    robot[:] = q
                    pin.framesForwardKinematics(rmodel, rdata, q)
                    tcp_se3 = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
                    object_se3 = chain.compute_object_pose_from_tcp(
                        tcp_se3, dataloader.best_grasp_SE3
                    )
                    obj.pose = object_se3.homogeneous
                    scene.render()


if __name__ == "__main__":
    main()
