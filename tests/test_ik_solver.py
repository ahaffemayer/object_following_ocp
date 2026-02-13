"""
Pytest unit tests for RobotIKSolver class with Pinocchio SE3 support.
Tests IK solving, FK verification, batch processing, and edge cases.

Run with: pytest test_ik_solver_pytest.py -v
"""

import numpy as np
import pinocchio as pin
import pytest

from object_following_ocp.ik_curobo import RobotIKSolver


@pytest.fixture(scope="module")
def solver():
    """Fixture providing a reusable solver instance."""
    return RobotIKSolver(
        robot_name="franka",
        num_seeds=20,
        position_threshold=0.005,
        rotation_threshold=0.05,
    )


@pytest.fixture
def valid_pose_posquat(solver):
    """Fixture providing a valid pose as (position, quaternion)."""
    q_sample = solver.ik_solver.sample_configs(1)
    position, quaternion = solver.forward_kinematics(q_sample[0].cpu().numpy())
    return position, quaternion


@pytest.fixture
def valid_pose_se3(solver):
    """Fixture providing a valid pose as SE3."""
    q_sample = solver.ik_solver.sample_configs(1)
    se3 = solver.forward_kinematics(q_sample[0].cpu().numpy(), return_se3=True)
    return se3


class TestInitialization:
    """Test solver initialization."""

    def test_initialization(self, solver):
        """Test that solver initializes correctly."""
        assert solver.ik_solver is not None
        assert solver.robot_name == "franka"
        assert solver.tensor_args is not None

    def test_invalid_robot_name(self):
        """Test initialization with invalid robot name."""
        with pytest.raises((FileNotFoundError, KeyError, Exception)):
            RobotIKSolver(robot_name="nonexistent_robot")


class TestSE3Conversion:
    """Test SE3 conversion utilities."""

    def test_se3_to_pos_quat(self):
        """Test converting SE3 to position and quaternion."""
        # Create a simple SE3
        translation = np.array([1.0, 2.0, 3.0])
        rotation = np.eye(3)
        se3 = pin.SE3(rotation, translation)

        position, quaternion = RobotIKSolver.se3_to_pos_quat(se3)

        # Check position
        np.testing.assert_array_almost_equal(position, translation)

        # Check quaternion (identity rotation = [1, 0, 0, 0])
        assert quaternion[0] == pytest.approx(1.0, abs=1e-6)  # w component
        assert np.allclose(quaternion[1:], [0, 0, 0], atol=1e-6)

    def test_pos_quat_to_se3(self):
        """Test converting position and quaternion to SE3."""
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation

        se3 = RobotIKSolver.pos_quat_to_se3(position, quaternion)

        # Check translation
        np.testing.assert_array_almost_equal(se3.translation, position)

        # Check rotation (should be identity)
        np.testing.assert_array_almost_equal(se3.rotation, np.eye(3))

    def test_round_trip_conversion(self):
        """Test that SE3 -> pos/quat -> SE3 preserves the transformation."""
        # Create an arbitrary SE3
        translation = np.array([0.5, -0.3, 0.8])
        angle = np.pi / 4
        axis = np.array([0, 0, 1])
        rotation = pin.AngleAxis(angle, axis).toRotationMatrix()
        original_se3 = pin.SE3(rotation, translation)

        # Convert to pos/quat and back
        pos, quat = RobotIKSolver.se3_to_pos_quat(original_se3)
        recovered_se3 = RobotIKSolver.pos_quat_to_se3(pos, quat)

        # Check they're equal
        np.testing.assert_array_almost_equal(
            recovered_se3.translation, original_se3.translation
        )
        np.testing.assert_array_almost_equal(
            recovered_se3.rotation, original_se3.rotation
        )


class TestForwardKinematics:
    """Test forward kinematics functionality."""

    def test_forward_kinematics_pos_quat(self, solver):
        """Test FK returning position and quaternion."""
        q_sample = solver.ik_solver.sample_configs(1)
        q_numpy = q_sample[0].cpu().numpy()

        position, quaternion = solver.forward_kinematics(q_numpy, return_se3=False)

        assert position.shape == (3,)
        assert quaternion.shape == (4,)
        assert isinstance(position, np.ndarray)
        assert isinstance(quaternion, np.ndarray)

        # Check quaternion is normalized
        quat_norm = np.linalg.norm(quaternion)
        assert np.isclose(quat_norm, 1.0, atol=1e-5)

    def test_forward_kinematics_se3(self, solver):
        """Test FK returning SE3."""
        q_sample = solver.ik_solver.sample_configs(1)
        q_numpy = q_sample[0].cpu().numpy()

        se3 = solver.forward_kinematics(q_numpy, return_se3=True)

        assert isinstance(se3, pin.SE3)
        assert se3.translation.shape == (3,)
        assert se3.rotation.shape == (3, 3)

    def test_fk_consistency(self, solver):
        """Test that FK returns consistent results in both formats."""
        q_sample = solver.ik_solver.sample_configs(1)
        q_numpy = q_sample[0].cpu().numpy()

        # Get both formats
        pos, quat = solver.forward_kinematics(q_numpy, return_se3=False)
        se3 = solver.forward_kinematics(q_numpy, return_se3=True)

        # They should represent the same transform
        np.testing.assert_array_almost_equal(pos, se3.translation)

        # Convert SE3 back to quat and compare
        _, quat_from_se3 = RobotIKSolver.se3_to_pos_quat(se3)
        # Account for quaternion double cover (q and -q represent same rotation)
        assert np.allclose(quat, quat_from_se3, atol=1e-5) or np.allclose(
            quat, -quat_from_se3, atol=1e-5
        )


class TestSinglePoseSolving:
    """Test IK solving for single poses."""

    def test_solve_with_tuple(self, solver, valid_pose_posquat):
        """Test IK solving with (position, quaternion) tuple."""
        position, quaternion = valid_pose_posquat

        solution, info = solver.solve((position, quaternion))

        assert solution is not None
        assert info["success"]
        assert solution.ndim == 1
        assert solution.shape[0] == 7  # Franka has 7 joints
        assert info["position_error"] < 0.01
        assert info["rotation_error"] < 0.1

    def test_solve_with_se3(self, solver, valid_pose_se3):
        """Test IK solving with SE3 object."""
        solution, info = solver.solve(valid_pose_se3)

        assert solution is not None
        assert info["success"]
        assert solution.ndim == 1
        assert solution.shape[0] == 7
        assert info["position_error"] < 0.01
        assert info["rotation_error"] < 0.1

    def test_solve_se3_vs_posquat_consistency(self, solver):
        """Test that solving with SE3 and pos/quat gives similar results."""
        # Get a test pose
        q_sample = solver.ik_solver.sample_configs(1)
        pos, quat = solver.forward_kinematics(q_sample[0].cpu().numpy())
        se3 = solver.forward_kinematics(q_sample[0].cpu().numpy(), return_se3=True)

        # Solve with both methods
        sol_tuple, _ = solver.solve((pos, quat))
        sol_se3, _ = solver.solve(se3)

        if sol_tuple is not None and sol_se3 is not None:
            # Both solutions should produce similar end-effector poses
            fk_tuple = solver.forward_kinematics(sol_tuple, return_se3=True)
            fk_se3 = solver.forward_kinematics(sol_se3, return_se3=True)

            pos_diff = np.linalg.norm(fk_tuple.translation - fk_se3.translation)
            assert pos_diff < 0.01

    def test_solve_verify_with_fk(self, solver, valid_pose_se3):
        """Test that IK solution is correct by verifying with FK."""
        solution, info = solver.solve(valid_pose_se3)

        if solution is not None:
            # Verify with FK
            fk_se3 = solver.forward_kinematics(solution, return_se3=True)

            # Check position matches
            pos_error = np.linalg.norm(fk_se3.translation - valid_pose_se3.translation)
            assert pos_error < 0.01

    def test_return_all_solutions(self, solver, valid_pose_se3):
        """Test returning all valid solutions."""
        solution, info = solver.solve(valid_pose_se3, return_all_solutions=True)

        if solution is not None:
            assert "all_solutions" in info
            all_solutions = info["all_solutions"]

            assert len(all_solutions) > 0
            assert solution.ndim == 1
            assert all_solutions.ndim == 2

            np.testing.assert_array_almost_equal(all_solutions[0], solution, decimal=5)

    def test_impossible_pose(self, solver):
        """Test solver behavior with unreachable pose."""
        # Create unreachable SE3
        far_translation = np.array([10.0, 10.0, 10.0])
        se3 = pin.SE3(np.eye(3), far_translation)

        solution, info = solver.solve(se3)

        assert solution is None
        assert not info["success"]
        assert "position_error" in info
        assert "rotation_error" in info


class TestBatchSolving:
    """Test batch IK solving."""

    def test_solve_batch_with_se3_list(self, solver):
        """Test batch solving with list of SE3 objects."""
        # Generate multiple SE3 poses
        q_samples = solver.ik_solver.sample_configs(10)
        se3_list = []
        for i in range(10):
            se3 = solver.forward_kinematics(q_samples[i].cpu().numpy(), return_se3=True)
            se3_list.append(se3)

        # Solve batch
        solutions, info = solver.solve_batch(se3_list)

        assert solutions.shape[0] == 10
        assert len(info["success"]) == 10
        assert info["success_rate"] > 0.5
        assert info["mean_position_error"] < 0.01
        assert info["mean_rotation_error"] < 0.1

    def test_solve_batch_with_arrays(self, solver):
        """Test batch solving with position/quaternion arrays."""
        # Generate multiple poses
        q_samples = solver.ik_solver.sample_configs(10)
        positions = []
        quaternions = []
        for i in range(10):
            pos, quat = solver.forward_kinematics(q_samples[i].cpu().numpy())
            positions.append(pos)
            quaternions.append(quat)

        positions = np.array(positions)
        quaternions = np.array(quaternions)

        # Solve batch
        solutions, info = solver.solve_batch((positions, quaternions))

        assert solutions.shape[0] == 10
        assert info["success_rate"] > 0.5

    def test_batch_empty_input(self, solver):
        """Test batch solver with empty arrays."""
        positions = np.array([]).reshape(0, 3)
        quaternions = np.array([]).reshape(0, 4)

        with pytest.raises(ValueError):
            solver.solve_batch((positions, quaternions))


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_position_shape(self, solver):
        """Test with incorrect position shape."""
        position = np.array([1.0, 2.0])  # Should be 3D
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Position must have shape"):
            solver.solve((position, quaternion))

    def test_invalid_quaternion_shape(self, solver):
        """Test with incorrect quaternion shape."""
        position = np.array([0.5, 0.0, 0.5])
        quaternion = np.array([1.0, 0.0, 0.0])  # Should be 4D

        with pytest.raises(ValueError, match="Quaternion must have shape"):
            solver.solve((position, quaternion))


class TestPerformance:
    """Performance and stress tests."""

    def test_solve_speed_se3(self, solver, valid_pose_se3):
        """Test that solve time is reasonable with SE3 input."""
        solution, info = solver.solve(valid_pose_se3)

        assert info["solve_time"] < 1.0

    def test_batch_solve_speed(self, solver):
        """Test batch solving performance."""
        batch_size = 100

        # Generate SE3 poses
        q_samples = solver.ik_solver.sample_configs(batch_size)
        se3_list = []
        for i in range(batch_size):
            se3 = solver.forward_kinematics(q_samples[i].cpu().numpy(), return_se3=True)
            se3_list.append(se3)

        # Solve batch
        solutions, info = solver.solve_batch(se3_list)

        poses_per_second = batch_size / info["solve_time"]
        print(f"\nBatch performance: {poses_per_second:.1f} poses/second")

        assert poses_per_second > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
