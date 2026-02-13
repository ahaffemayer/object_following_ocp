"""
Tests for trajectory iteration functionality
"""
import numpy as np
import pinocchio as pin
import pytest


def test_trajectory_se3_iteration():
    """Test that TrajectorySE3 is properly iterable"""
    from object_following_ocp.trajectories import TrajectorySE3

    # Create test poses
    poses = [
        pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
        pin.SE3(np.eye(3), np.array([2.0, 0.0, 0.0])),
        pin.SE3(np.eye(3), np.array([3.0, 0.0, 0.0])),
    ]

    trajectory = TrajectorySE3(poses)

    # Test for loop
    collected_poses = []
    for pose in trajectory:
        collected_poses.append(pose)

    assert len(collected_poses) == 3
    np.testing.assert_array_almost_equal(
        collected_poses[0].translation, [1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(
        collected_poses[1].translation, [2.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(
        collected_poses[2].translation, [3.0, 0.0, 0.0])


def test_trajectory_se3_enumerate():
    """Test that TrajectorySE3 works with enumerate()"""
    from object_following_ocp.trajectories import TrajectorySE3

    poses = [
        pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
        pin.SE3(np.eye(3), np.array([2.0, 0.0, 0.0])),
    ]

    trajectory = TrajectorySE3(poses)

    # Test enumerate
    for i, pose in enumerate(trajectory):
        expected = float(i + 1)
        np.testing.assert_array_almost_equal(
            pose.translation, [expected, 0.0, 0.0])


def test_trajectory_se3_list_conversion():
    """Test that TrajectorySE3 can be converted to list"""
    from object_following_ocp.trajectories import TrajectorySE3

    poses = [
        pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
        pin.SE3(np.eye(3), np.array([2.0, 0.0, 0.0])),
    ]

    trajectory = TrajectorySE3(poses)

    # Test list conversion
    pose_list = list(trajectory)
    assert len(pose_list) == 2
    assert isinstance(pose_list[0], pin.SE3)


def test_trajectory_config_space_iteration():
    """Test that TrajectoryInConfigurationSpace is properly iterable"""
    from object_following_ocp.trajectories import TrajectoryInConfigurationSpace

    configs = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.2, 0.4, 0.6]),
    ]

    trajectory = TrajectoryInConfigurationSpace(configs)

    # Test for loop
    collected_configs = []
    for config in trajectory:
        collected_configs.append(config)

    assert len(collected_configs) == 3
    np.testing.assert_array_almost_equal(collected_configs[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(collected_configs[1], [0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(collected_configs[2], [0.2, 0.4, 0.6])


def test_trajectory_config_space_enumerate():
    """Test that TrajectoryInConfigurationSpace works with enumerate()"""
    from object_following_ocp.trajectories import TrajectoryInConfigurationSpace

    configs = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.2, 0.3]),
    ]

    trajectory = TrajectoryInConfigurationSpace(configs)

    # Test enumerate
    for i, config in enumerate(trajectory):
        expected = np.array([i * 0.1, i * 0.2, i * 0.3])
        np.testing.assert_array_almost_equal(config, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
