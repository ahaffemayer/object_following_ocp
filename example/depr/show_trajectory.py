from pathlib import Path

import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.dataclass import ConfigLoader
from object_following_ocp.robot_loader import load_reduced_panda, self_collision_pairs
from object_following_ocp.trajectory_parser import JSONTrajectoryParser

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load robot configuration from YAML
    robot_config = ConfigLoader.load(
        Path(__file__).parent / "robot_config.yml")

    # Load trajectory data from JSON
    json_path = Path(
        "/workspaces/object_following_ocp/ressources/json/jug.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json")

    # Parser automatically resolves mesh paths and applies smoothing
    traj_parser = JSONTrajectoryParser(
        json_path, smooth_depth=True, smooth_k=2.0)

    # Show available objects
    print("Available objects:", traj_parser.get_available_objects())

    # Get object information (auto-selects first available object)
    object_info = traj_parser.get_object_info(texture_name="material_0.png")

    print(f"Object mesh: {object_info.mesh_path}")
    print(f"Object texture: {object_info.texture_path}")
    print(f"Object scale: {object_info.scale}")

    # Get trajectory poses (auto-selects first available object)
    trajectory = traj_parser.get_poses_for_object()

    print(f"Number of poses: {len(trajectory)}")

    # -----------------------------
    # Load robot and collision model
    # -----------------------------
    rmodel, cmodel, vmodel = load_reduced_panda()

    for cp in self_collision_pairs:
        if cmodel.existGeometryName(cp[0]) and cmodel.existGeometryName(cp[1]):
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId(cp[0]),
                    cmodel.getGeometryId(cp[1]),
                )
            )

    rdata = rmodel.createData()
    cdata = cmodel.createData()
    vdata = vmodel.createData()

    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )

    # -----------------------------
    # Scene
    # -----------------------------
    scene = Scene()
    scene.add_robot(robot)

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

    # -----------------------------
    # Visualize trajectory
    # -----------------------------
    for k, pose_data in enumerate(trajectory):
        color = [0.0, 1.0, 0.0] if k == 0 else [0.5, 0.5, 0.5]

        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"target_{k}", color=color)
        )

        # Create homogeneous transformation matrix from R and t
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = pose_data.R
        pose_matrix[:3, 3] = pose_data.t

        scene[f"target_{k}"].pos[:] = pose_data.t
        o.pose = pose_matrix

        print(f"Pose {k}: im_id={pose_data.im_id}, score={pose_data.score:.3f}")
        input("Press Enter to continue to the next target...")
