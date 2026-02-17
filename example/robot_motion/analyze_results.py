"""
Utility script to analyze and load saved grid search results.
"""

import pathlib
import pickle

import numpy as np


def load_and_analyze_grid_search(filepath: pathlib.Path):
    """
    Load and analyze grid search results.

    Args:
        filepath: Path to grid search results file
    """
    print("=" * 70)
    print("GRID SEARCH RESULTS ANALYSIS")
    print("=" * 70)

    with open(filepath, "rb") as f:
        results = pickle.load(f)

    # Display metadata
    print(f"\nMesh ID: {results['mesh_id']}")
    print(f"Object trajectory: {results['object_traj_path']}")
    print(f"Scales file: {results['scale_path']}")
    print(f"Config file: {results['config_path']}")
    print(f"Mesh file: {results['mesh_path']}")
    if results["texture_path"]:
        print(f"Texture file: {results['texture_path']}")
    print(f"Object scale: {results['object_scale']}")
    if results["best_grasp_name"]:
        print(f"Best grasp: {results['best_grasp_name']}")
    print(f"Grasp correction angle: {results['grasp_correction_angle_deg']}°")
    print(f"Elevation angle: {results['elevation_angle_deg']}°")

    summary = results["summary"]
    trajectories = results["trajectories"]

    print(f"\nTotal valid trajectories: {summary['num_valid']}")

    if summary["num_valid"] == 0:
        print("No valid trajectories found.")
        return

    print("\nIK Success Rates:")
    print(f"  Best:  {summary['best_success_rate']:.1f}%")
    print(f"  Worst: {summary['worst_success_rate']:.1f}%")
    print(f"  Mean:  {summary['mean_success_rate']:.1f}%")

    print("\nCamera Positions:")
    for idx, pos in enumerate(summary["camera_positions"]):
        print(f"  {idx + 1}. [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")

    # Analyze camera position distribution
    camera_positions = np.array(summary["camera_positions"])
    print("\nCamera Position Statistics:")
    print(
        f"  X: min={camera_positions[:, 0].min():.2f}, "
        f"max={camera_positions[:, 0].max():.2f}, "
        f"mean={camera_positions[:, 0].mean():.2f}"
    )
    print(
        f"  Y: min={camera_positions[:, 1].min():.2f}, "
        f"max={camera_positions[:, 1].max():.2f}, "
        f"mean={camera_positions[:, 1].mean():.2f}"
    )
    print(
        f"  Z: min={camera_positions[:, 2].min():.2f}, "
        f"max={camera_positions[:, 2].max():.2f}, "
        f"mean={camera_positions[:, 2].mean():.2f}"
    )

    print("\n" + "=" * 70)


def load_and_analyze_kept_trajectories(filepath: pathlib.Path):
    """
    Load and analyze kept trajectories.

    Args:
        filepath: Path to kept trajectories file
    """
    print("=" * 70)
    print("KEPT TRAJECTORIES ANALYSIS")
    print("=" * 70)

    with open(filepath, "rb") as f:
        results = pickle.load(f)

    # Display metadata
    print(f"\nMesh ID: {results['mesh_id']}")
    print(f"Object trajectory: {results['object_traj_path']}")
    print(f"Scales file: {results['scale_path']}")
    print(f"Config file: {results['config_path']}")
    print(f"Mesh file: {results['mesh_path']}")
    if results["texture_path"]:
        print(f"Texture file: {results['texture_path']}")
    print(f"Object scale: {results['object_scale']}")
    if results["best_grasp_name"]:
        print(f"Best grasp: {results['best_grasp_name']}")
    print(f"Grasp correction angle: {results['grasp_correction_angle_deg']}°")
    print(f"Elevation angle: {results['elevation_angle_deg']}°")

    num_kept = results["num_kept"]
    trajectories = results["trajectories"]

    print(f"\nTotal kept trajectories: {num_kept}")

    if num_kept == 0:
        print("No trajectories were kept.")
        return

    for idx, traj in enumerate(trajectories):
        print(f"\n--- Trajectory {idx + 1} ---")
        print(f"Camera position: {traj['camera_translation']}")
        print(f"IK success rate: {traj['ik_success_rate']:.1f}%")
        print(f"Number of waypoints: {len(traj['object_trajectory_poses'])}")
        print(f"IK trajectory length: {len(traj['joint_trajectory_ik'])}")
        print(f"OCP trajectory length: {len(traj['joint_trajectory_ocp'])}")
        if traj["user_notes"]:
            print(f"Notes: {traj['user_notes']}")

    # Statistics
    ik_success_rates = [t["ik_success_rate"] for t in trajectories]

    print("\n--- Statistics ---")
    print("IK Success Rates:")
    print(f"  Mean: {np.mean(ik_success_rates):.1f}%")
    print(f"  Best: {np.max(ik_success_rates):.1f}%")
    print(f"  Worst: {np.min(ik_success_rates):.1f}%")

    print("\n" + "=" * 70)


def compare_trajectories(
    grid_search_path: pathlib.Path,
    kept_trajectories_path: pathlib.Path,
):
    """
    Compare grid search results with kept trajectories.

    Args:
        grid_search_path: Path to grid search results
        kept_trajectories_path: Path to kept trajectories
    """
    print("=" * 70)
    print("COMPARISON: GRID SEARCH vs KEPT TRAJECTORIES")
    print("=" * 70)

    with open(grid_search_path, "rb") as f:
        grid_results = pickle.load(f)

    with open(kept_trajectories_path, "rb") as f:
        kept_results = pickle.load(f)

    num_found = grid_results["summary"]["num_valid"]
    num_kept = kept_results["num_kept"]

    print(f"\nTrajectories found in grid search: {num_found}")
    print(f"Trajectories kept after review: {num_kept}")
    print(f"Keep rate: {100 * num_kept / num_found:.1f}%")

    if num_kept > 0:
        # Find which camera positions were kept
        kept_positions = [t["camera_translation"] for t in kept_results["trajectories"]]

        print("\nKept camera positions:")
        for idx, pos in enumerate(kept_positions):
            print(f"  {idx + 1}. {pos}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    output_dir = pathlib.Path("/mnt/user-data/outputs")
    grid_search_path = (
        output_dir
        / "/mnt/user-data/outputs/cbb0cdd9bbcc4fdfa2e16db1db4cda61_grid_search_results.pkl"
    )
    kept_trajectories_path = (
        output_dir / "cbb0cdd9bbcc4fdfa2e16db1db4cda61_kept_trajectories.pkl"
    )

    print("\n" + "=" * 70)
    print("RESULTS ANALYZER")
    print("=" * 70)

    # Check which files exist
    if grid_search_path.exists():
        print(f"\n✓ Found grid search results: {grid_search_path}")
        load_and_analyze_grid_search(grid_search_path)
    else:
        print(f"\n✗ Grid search results not found: {grid_search_path}")

    if kept_trajectories_path.exists():
        print(f"\n✓ Found kept trajectories: {kept_trajectories_path}")
        load_and_analyze_kept_trajectories(kept_trajectories_path)
    else:
        print(f"\n✗ Kept trajectories not found: {kept_trajectories_path}")

    if grid_search_path.exists() and kept_trajectories_path.exists():
        print()
        compare_trajectories(grid_search_path, kept_trajectories_path)
