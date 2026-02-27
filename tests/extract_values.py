"""
Helper script to extract expected values from real files for unit tests.

Run this script from your workspace to get the expected values,
then copy them into test_dataloader.py's test_real_files_expected_values() test.

Usage:
    python extract_test_values.py
"""

from object_following_ocp.data.data_loader import DataLoader
import pathlib
import sys

# Add your project to path if needed
sys.path.insert(0, "/workspaces/object_following_ocp")


def main():
    print("=" * 80)
    print("EXTRACTING EXPECTED VALUES FOR UNIT TESTS")
    print("=" * 80)

    # File paths
    grasp_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/filtered_grasps/"
        "0d0d1c59b0474d2ea92ce2e172c9f56a_filtered.yml"
    )
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/"
        "bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
    )
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json"
    )

    print(f"\nLoading from:")
    print(f"  Trajectory: {object_traj_path.name}")
    print(f"  Grasp:      {grasp_path.name}")
    print(f"  Scales:     {scale_path.name}")
    print()

    # Load data
    loader = DataLoader(
        object_trajectory_path=object_traj_path,
        grasp_poses_SE3_path=grasp_path,
        scales_path=scale_path,
    )

    print("=" * 80)
    print("COPY THESE VALUES INTO test_real_files_expected_values():")
    print("=" * 80)
    print()
    print("# Object info")
    print(f"assert loader.object_id == {loader.object_id}")
    print(f'assert loader.object_info.mesh_id == "{loader.object_info.mesh_id}"')
    print(
        f"assert loader.object_info.scale == pytest.approx({loader.object_info.scale})"
    )
    print(
        f"assert loader.object_info.score == pytest.approx({loader.object_info.score})"
    )
    print()

    print("# Counts")
    print(f"assert len(loader.poses) == {len(loader.poses)}")
    print(f"assert len(loader.grasp_poses) == {len(loader.grasp_poses)}")
    print()

    if len(loader.poses) > 0:
        print("# First pose")
        t = loader.poses[0].t
        print(f"expected_t_0 = np.array([{t[0]}, {t[1]}, {t[2]}])")
        print(f"np.testing.assert_array_almost_equal(loader.poses[0].t, expected_t_0)")
        print(f"assert loader.poses[0].score == pytest.approx({loader.poses[0].score})")
        print(f"assert loader.poses[0].im_id == {loader.poses[0].im_id}")
        print(f"assert loader.poses[0].object_id == {loader.poses[0].object_id}")
        print()

        # First pose rotation matrix (sample check)
        R = loader.poses[0].R
        print(f"# First pose rotation (sample check of first row)")
        print(f"expected_R_0_row0 = np.array([{R[0, 0]}, {R[0, 1]}, {R[0, 2]}])")
        print(
            f"np.testing.assert_array_almost_equal(loader.poses[0].R[0], expected_R_0_row0)"
        )
        print()

    if len(loader.grasp_poses) > 0:
        best = loader.best_grasp
        print("# Best grasp")
        print(f'assert loader.best_grasp.name == "{best.name}"')
        print(
            f"assert loader.best_grasp.confidence == pytest.approx({best.confidence})"
        )
        pos = best.position
        print(f"expected_grasp_pos = np.array([{pos[0]}, {pos[1]}, {pos[2]}])")
        print(
            f"np.testing.assert_array_almost_equal(loader.best_grasp.position, expected_grasp_pos)"
        )
        print()

        # Orientation
        ori = best.orientation
        print(
            f"expected_grasp_ori = np.array([{ori[0]}, {ori[1]}, {ori[2]}, {ori[3]}])"
        )
        print(
            f"np.testing.assert_array_almost_equal(loader.best_grasp.orientation, expected_grasp_ori)"
        )
        print()

    # Trajectory conversion
    print("# Trajectory conversion")
    trajectory = loader.to_trajectory_SE3()
    print(f"trajectory = loader.to_trajectory_SE3()")
    print(f"assert len(trajectory) == {len(trajectory)}")
    print(f"assert isinstance(trajectory[0].translation, np.ndarray)")
    print()

    # Best grasp SE3
    print("# Best grasp SE3")
    best_se3 = loader.best_grasp_SE3
    print(f"best_se3 = loader.best_grasp_SE3")
    print(
        f"np.testing.assert_array_almost_equal(best_se3.translation, expected_grasp_pos)"
    )
    print()

    # Mesh paths
    print("# Mesh paths (check construction is correct)")
    print(
        f'assert loader.object_info.mesh_path.name == "{loader.object_info.mesh_path.name}"'
    )
    print(
        f'assert loader.object_info.texture_path.name == "{loader.object_info.texture_path.name}"'
    )
    print(
        "# Note: mesh files may be in a different location, so we just check the filename"
    )
    print()

    print("=" * 80)
    print("Additional information for manual verification:")
    print("=" * 80)
    print(f"\nMesh path:    {loader.object_info.mesh_path}")
    print(f"Texture path: {loader.object_info.texture_path}")
    print(f"Mesh exists:  {loader.object_info.mesh_path.exists()}")
    print(f"Texture exists: {loader.object_info.texture_path.exists()}")
    print(f"\nNote: Mesh paths use .parent.parent logic:")
    print(f"  trajectory.json.parent.parent / 'meshes' / mesh_id / mesh_id.obj")
    print(
        f"\nNumber of grasps with confidence > 0.98: {sum(1 for g in loader.grasp_poses if g.confidence > 0.98)}"
    )
    print(
        f"Lowest confidence grasp: {min(g.confidence for g in loader.grasp_poses):.4f}"
    )
    print(
        f"Highest confidence grasp: {max(g.confidence for g in loader.grasp_poses):.4f}"
    )
    print(f"\nFirst 3 pose times: {[p.time for p in loader.poses[:3]]}")
    print(f"Last 3 pose times: {[p.time for p in loader.poses[-3:]]}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you're running this from within your workspace environment")
        print("and that all required packages are installed.")
        import traceback

        traceback.print_exc()
