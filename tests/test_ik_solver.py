"""
Pytest unit tests for RobotIKSolver class.
Tests IK solving, FK verification, batch processing, and edge cases.

Run with: pytest test_ik_solver.py -v
"""

import numpy as np
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
def valid_pose(solver):
    """Fixture providing a valid pose from FK."""
    q_sample = solver.ik_solver.sample_configs(1)
    position, quaternion = solver.forward_kinematics(q_sample[0].cpu().numpy())
    return position, quaternion


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


class TestForwardKinematics:
    """Test forward kinematics functionality."""

    def test_forward_kinematics(self, solver):
        """Test forward kinematics computation."""
        # Sample a random valid configuration
        q_sample = solver.ik_solver.sample_configs(1)
        q_numpy = q_sample[0].cpu().numpy()

        # Compute FK
        position, quaternion = solver.forward_kinematics(q_numpy)

        # Check output shapes and types
        assert position.shape == (3,)
        assert quaternion.shape == (4,)
        assert isinstance(position, np.ndarray)
        assert isinstance(quaternion, np.ndarray)

        # Check quaternion is normalized
        quat_norm = np.linalg.norm(quaternion)
        assert np.isclose(quat_norm, 1.0, atol=1e-5)


class TestSinglePoseSolving:
    """Test IK solving for single poses."""

    def test_solve_single_pose(self, solver, valid_pose):
        """Test IK solving for a single pose."""
        position, quaternion = valid_pose

        # Solve IK
        solution, info = solver.solve(position, quaternion)

        # Check that solution was found
        assert solution is not None, "IK solution should be found for valid pose"
        assert info["success"], "Success flag should be True"

        # Check solution is 1D with correct number of joints
        assert solution.ndim == 1, "Solution should be 1D array"
        assert solution.shape[0] == 7, "Franka should have 7 joints"

        # Check errors are within threshold
        assert info["position_error"] < 0.01, "Position error too large"
        assert info["rotation_error"] < 0.1, "Rotation error too large"

        # Check timing info exists
        assert "solve_time" in info
        assert info["solve_time"] > 0

    def test_solve_verify_with_fk(self, solver, valid_pose):
        """Test that IK solution is correct by verifying with FK."""
        target_pos, target_quat = valid_pose

        # Solve IK
        solution, info = solver.solve(target_pos, target_quat)

        if solution is not None:
            # Verify solution with FK
            fk_pos, fk_quat = solver.forward_kinematics(solution)

            # Check position matches
            pos_error = np.linalg.norm(fk_pos - target_pos)
            assert pos_error < 0.01, "FK position doesn't match target"

            # Check quaternion orientation (account for double cover)
            quat_dot = np.abs(np.dot(fk_quat, target_quat))
            assert quat_dot > 0.99, "FK orientation doesn't match target"

    def test_solve_with_list_input(self, solver, valid_pose):
        """Test that solver accepts list inputs as well as numpy arrays."""
        position, quaternion = valid_pose

        # Convert to lists
        position_list = position.tolist()
        quaternion_list = quaternion.tolist()

        # Solve with list input
        solution, info = solver.solve(position_list, quaternion_list)

        # Should still work
        assert solution is not None
        assert info["success"]

    def test_return_all_solutions(self, solver, valid_pose):
        """Test returning all valid solutions."""
        position, quaternion = valid_pose

        # Solve with return_all_solutions=True
        solution, info = solver.solve(position, quaternion, return_all_solutions=True)

        if solution is not None:
            assert "all_solutions" in info
            all_solutions = info["all_solutions"]

            # Should have at least one solution
            assert len(all_solutions) > 0

            # solution should be 1D, all_solutions should be 2D
            assert solution.ndim == 1, "Single solution should be 1D"
            assert all_solutions.ndim == 2, "All solutions should be 2D"

            # First solution from all_solutions should match the returned solution
            np.testing.assert_array_almost_equal(all_solutions[0], solution, decimal=5)

    def test_impossible_pose(self, solver):
        """Test solver behavior with unreachable pose."""
        # Create a pose that's likely out of reach
        position = np.array([10.0, 10.0, 10.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # Try to solve
        solution, info = solver.solve(position, quaternion)

        # Should fail gracefully
        assert solution is None, "Should return None for impossible pose"
        assert not info["success"], "Success should be False"
        assert "position_error" in info
        assert "rotation_error" in info

    def test_identity_quaternion(self, solver):
        """Test with identity quaternion (no rotation)."""
        # Get a pose with identity rotation
        q_sample = solver.ik_solver.sample_configs(1)
        position, _ = solver.forward_kinematics(q_sample[0].cpu().numpy())

        # Use identity quaternion
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # Should still solve
        solution, info = solver.solve(position, quaternion)

        # May or may not succeed depending on reachability
        assert "success" in info
        assert isinstance(info["success"], bool)

    def test_multiple_solves_same_pose(self, solver, valid_pose):
        """Test that solving the same pose multiple times is consistent."""
        position, quaternion = valid_pose

        # Solve multiple times
        solutions = []
        for _ in range(3):
            solution, info = solver.solve(position, quaternion)
            if solution is not None:
                solutions.append(solution)

        # Should get solutions
        assert len(solutions) > 0, "Should find solution at least once"

        # All solutions should produce similar FK results
        if len(solutions) > 1:
            fk_positions = [solver.forward_kinematics(s)[0] for s in solutions]

            # All positions should be close to target
            for fk_pos in fk_positions:
                pos_error = np.linalg.norm(fk_pos - position)
                assert pos_error < 0.01


class TestBatchSolving:
    """Test batch IK solving."""

    def test_solve_batch(self, solver):
        """Test batch IK solving."""
        batch_size = 10

        # Generate multiple valid poses
        q_samples = solver.ik_solver.sample_configs(batch_size)

        positions = []
        quaternions = []
        for i in range(batch_size):
            pos, quat = solver.forward_kinematics(q_samples[i].cpu().numpy())
            positions.append(pos)
            quaternions.append(quat)

        positions = np.array(positions)
        quaternions = np.array(quaternions)

        # Solve batch
        solutions, info = solver.solve_batch(positions, quaternions)

        # Check output shapes
        assert solutions.shape[0] == batch_size
        assert len(info["success"]) == batch_size

        # Check success rate
        assert info["success_rate"] > 0.5, "Success rate too low"

        # Check info contains required fields
        assert "solve_time" in info
        assert "mean_position_error" in info
        assert "mean_rotation_error" in info
        assert "position_errors" in info
        assert "rotation_errors" in info

        # Check errors are reasonable
        assert info["mean_position_error"] < 0.01
        assert info["mean_rotation_error"] < 0.1

    def test_batch_empty_input(self, solver):
        """Test batch solver with empty arrays."""
        positions = np.array([]).reshape(0, 3)
        quaternions = np.array([]).reshape(0, 4)

        # Should raise ValueError for empty input
        with pytest.raises(ValueError):
            solver.solve_batch(positions, quaternions)


class TestDifferentConfigurations:
    """Test solver with different configurations."""

    def test_different_num_seeds(self, valid_pose):
        """Test solver with different number of seeds."""
        # Create solver with fewer seeds
        solver_few_seeds = RobotIKSolver(robot_name="franka", num_seeds=5)

        position, quaternion = valid_pose

        # Should still work
        solution, info = solver_few_seeds.solve(position, quaternion)
        assert "success" in info

    def test_stricter_thresholds(self, valid_pose):
        """Test solver with stricter error thresholds."""
        solver_strict = RobotIKSolver(
            robot_name="franka",
            position_threshold=0.001,
            rotation_threshold=0.01,
        )

        position, quaternion = valid_pose

        # Solve
        solution, info = solver_strict.solve(position, quaternion)

        # If successful, errors should be within stricter thresholds
        if solution is not None:
            assert info["position_error"] < 0.001
            assert info["rotation_error"] < 0.01

    def test_quaternion_normalization(self, solver, valid_pose):
        """Test that unnormalized quaternions are handled."""
        position, quaternion = valid_pose

        # Create unnormalized quaternion
        quaternion_unnorm = quaternion * 2.0

        # Should still work (CuRobo likely normalizes internally)
        solution, info = solver.solve(position, quaternion_unnorm)

        # Should either succeed or fail gracefully
        assert "success" in info


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_position_shape(self):
        """Test with incorrect position shape."""
        solver = RobotIKSolver(robot_name="franka")

        # Wrong shape
        position = np.array([1.0, 2.0])  # Should be 3D
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # Should raise ValueError
        with pytest.raises(ValueError, match="Position must have shape"):
            solver.solve(position, quaternion)

    def test_invalid_quaternion_shape(self):
        """Test with incorrect quaternion shape."""
        solver = RobotIKSolver(robot_name="franka")

        position = np.array([0.5, 0.0, 0.5])
        quaternion = np.array([1.0, 0.0, 0.0])  # Should be 4D

        # Should raise ValueError
        with pytest.raises(ValueError, match="Quaternion must have shape"):
            solver.solve(position, quaternion)


class TestPerformance:
    """Performance and stress tests."""

    def test_solve_speed(self, solver, valid_pose):
        """Test that solve time is reasonable."""
        position, quaternion = valid_pose

        # Solve and check timing
        solution, info = solver.solve(position, quaternion)

        # Should solve in reasonable time (< 1 second for single pose)
        assert info["solve_time"] < 1.0, "Single solve took too long"

    def test_batch_solve_speed(self, solver):
        """Test batch solving performance."""
        batch_size = 100

        # Generate valid poses
        q_samples = solver.ik_solver.sample_configs(batch_size)
        positions = []
        quaternions = []
        for i in range(batch_size):
            pos, quat = solver.forward_kinematics(q_samples[i].cpu().numpy())
            positions.append(pos)
            quaternions.append(quat)

        positions = np.array(positions)
        quaternions = np.array(quaternions)

        # Solve batch
        solutions, info = solver.solve_batch(positions, quaternions)

        # Check throughput
        poses_per_second = batch_size / info["solve_time"]
        print(f"\nBatch performance: {poses_per_second:.1f} poses/second")

        # Should be reasonably fast (at least 10 poses/sec on GPU)
        assert poses_per_second > 10, "Batch solving too slow"

    def test_consecutive_solves(self, solver):
        """Test multiple consecutive solves don't degrade performance."""
        # Get valid poses
        q_samples = solver.ik_solver.sample_configs(10)

        solve_times = []
        for i in range(10):
            pos, quat = solver.forward_kinematics(q_samples[i].cpu().numpy())
            _, info = solver.solve(pos, quat)
            solve_times.append(info["solve_time"])

        # Later solves shouldn't be significantly slower
        avg_early = np.mean(solve_times[:3])
        avg_late = np.mean(solve_times[-3:])

        # Allow some variance but shouldn't increase by more than 2x
        assert avg_late < avg_early * 2.0, "Performance degraded"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
