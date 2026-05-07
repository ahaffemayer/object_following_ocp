"""
Main script for camera position grid search and trajectory review.

Workflow:
1. Compute reference camera that centres the trajectory at the robot
2. Grid search ± half-extents around that reference
3. For each position, test IK on first pose
4. If successful, solve IK for entire trajectory
5. Store all valid trajectories
6. For each valid trajectory, solve OCP, animate, and auto-save
"""

import argparse
import pathlib

import numpy as np
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data.data_loader import ConfigLoader, DataLoader
from object_following_ocp.geom.camera_grid_search import CameraPositionGridSearch
from object_following_ocp.robot.robot_loader import load_reduced_panda
from object_following_ocp.visualizer.trajectory_reviewer import TrajectoryReviewer
from object_following_ocp.visualizer.trajectory_visualizer import TrajectoryVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera position grid search and trajectory review."
    )

    # --- Required paths ---
    parser.add_argument(
        "--object-traj",
        type=pathlib.Path,
        required=True,
        help="Path to the object trajectory JSON file.",
    )
    parser.add_argument(
        "--config-path",
        type=pathlib.Path,
        required=True,
        help="Path to the OCP config YAML file.",
    )

    # --- Optional paths ---
    parser.add_argument(
        "--scale-path",
        type=pathlib.Path,
        default=None,
        help="Path to the grasps/scales JSON file (optional; if omitted, scale is read from the trajectory JSON).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/user-data/outputs"),
        help="Directory where results are saved (default: /mnt/user-data/outputs).",
    )

    # --- Grid search parameters ---
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.7],
        metavar=("X", "Y", "Z"),
        help="Target world-frame position for the trajectory average (default: 0 0 0.7).",
    )
    parser.add_argument(
        "--half-extents",
        type=float,
        nargs=3,
        default=[0.2, 0.2, 0.2],
        metavar=("DX", "DY", "DZ"),
        help="Half-extents of the grid search per axis (default: 0.2 0.2 0.2).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs=3,
        default=[3, 3, 3],
        metavar=("NX", "NY", "NZ"),
        help="Number of grid samples per axis (default: 3 3 3).",
    )

    # --- Trajectory selection ---
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of best trajectories to keep and visualise (default: 3).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Derive trajectory name from the input file stem for output naming
    traj_name = args.object_traj.stem

    args.output_dir.mkdir(parents=True, exist_ok=True)

    TARGET_POSITION = np.array(args.target_position)
    HALF_EXTENTS = tuple(args.half_extents)
    STEPS = tuple(args.steps)
    N = args.top_n

    # ========================================================================
    # SETUP
    # ========================================================================

    print("=" * 70)
    print("CAMERA POSITION GRID SEARCH & TRAJECTORY REVIEW")
    print("=" * 70)

    dataloader = DataLoader(
        object_trajectory_path=args.object_traj,
        scales_path=args.scale_path,
        load_grasps=True,
    )
    robot_config = ConfigLoader.load(args.config_path)

    print(f"\nLoaded object: {dataloader.object_info.mesh_id}")
    print(f"Trajectory length: {len(dataloader.poses)} poses")
    print(
        f"Best grasp: {dataloader.best_grasp.name if dataloader.has_grasps else 'None'}"
    )
    print(f"Gripper depth: {robot_config.gripper_depth} m")

    mesh_id = dataloader.object_info.mesh_id
    grid_search_results_path = args.output_dir / f"{traj_name}_grid_search_results.pkl"
    kept_trajectories_path = args.output_dir / f"{traj_name}_kept_trajectories.pkl"

    print(f"\nOutput: {grid_search_results_path.name}, {kept_trajectories_path.name}")

    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    vdata = vmodel.createData()

    scene = Scene()
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )
    scene.add_robot(robot=robot)

    movable_object = Object.create_mesh(
        path_to_mesh=dataloader.object_info.mesh_path,
        name="robot/movable_obj",
        texture=dataloader.object_info.texture_path,
        scale=dataloader.object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    scene.add_object(movable_object)

    # ========================================================================
    # STEP 1: Grid Search (auto-centred)
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 1: Grid Search")
    print("=" * 70)

    grid_search = CameraPositionGridSearch(
        dataloader=dataloader,
        robot_config=robot_config,
        object_traj_path=args.object_traj,
        scale_path=args.scale_path,
        config_path=args.config_path,
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
        verbose=True,
    )

    valid_trajectories = grid_search.run_grid_search(
        target_position=TARGET_POSITION,
        half_extents=HALF_EXTENTS,
        steps=STEPS,
    )

    summary = grid_search.get_summary()
    print("\nGrid Search Summary:")
    print(f"  Valid trajectories found: {summary['num_valid']}")
    if summary["num_valid"] > 0:
        print(f"  Best IK success rate:  {summary['best_success_rate']:.1f}%")
        print(f"  Worst IK success rate: {summary['worst_success_rate']:.1f}%")
        print(f"  Mean IK success rate:  {summary['mean_success_rate']:.1f}%")

    grid_search.save_results(grid_search_results_path)

    if len(valid_trajectories) == 0:
        print(
            "\n⚠ No valid trajectories found. Try adjusting half_extents or target_position."
        )
        exit(0)

    # ========================================================================
    # STEP 2: OCP + Visualization (auto-save all)
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: OCP Solving & Visualization")
    print("=" * 70)

    visualizer = TrajectoryVisualizer(
        scene=scene,
        robot=robot,
        movable_object=movable_object,
    )

    reviewer = TrajectoryReviewer(
        dataloader=dataloader,
        robot_config=robot_config,
        visualizer=visualizer,
        rmodel=rmodel,
        cmodel=cmodel,
        object_traj_path=args.object_traj,
        scale_path=args.scale_path,
        config_path=args.config_path,
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
    )

    kept_trajectories = reviewer.review_all_candidates(valid_trajectories, top_n=N)

    if kept_trajectories:
        reviewer.save_kept_trajectories(kept_trajectories_path)
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for idx, traj in enumerate(kept_trajectories):
            print(
                f"  [{idx + 1}] camera={traj.camera_translation}  "
                f"IK={traj.ik_success_rate:.1f}%  cost={traj.tracking_cost:.6f}"
            )

    # ========================================================================
    # COMPLETE
    # ========================================================================

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Grid search: {grid_search_results_path}")
    if kept_trajectories:
        print(f"  Trajectories: {kept_trajectories_path}")
    print("=" * 70)
