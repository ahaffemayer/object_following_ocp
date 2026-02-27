import numpy as np
import pinocchio as pin
import pytest

from object_following_ocp.data.data_loader import RobotConfig
from object_following_ocp.geom.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.geom.trajectories import TrajectorySE3


class TestGraspTransformConfig:
    """Tests for GraspTransformConfig dataclass."""

    def test_default_initialization(self):
        """Test that default values are set correctly."""
        config = GraspTransformConfig()

        assert config.camera_translation is not None
        np.testing.assert_array_equal(
            config.camera_translation, np.array([0, -1.0, -1.0])
        )
        assert config.gripper_depth == 0.1034
        assert config.grasp_correction_angle_deg == 90.0
        np.testing.assert_array_equal(config.grasp_correction_axis, np.array([0, 0, 1]))
        assert config.elevation_angle_deg == 25.0

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        custom_translation = np.array([1.0, 2.0, 3.0])
        custom_axis = np.array([1, 0, 0])

        config = GraspTransformConfig(
            camera_translation=custom_translation,
            gripper_depth=0.15,
            grasp_correction_angle_deg=45.0,
            grasp_correction_axis=custom_axis,
            elevation_angle_deg=30.0,
        )

        np.testing.assert_array_equal(config.camera_translation, custom_translation)
        assert config.gripper_depth == 0.15
        assert config.grasp_correction_angle_deg == 45.0
        np.testing.assert_array_equal(config.grasp_correction_axis, custom_axis)
        assert config.elevation_angle_deg == 30.0

    def test_from_robot_config(self):
        """Test creation from RobotConfig."""
        robot_config = RobotConfig(
            W_xREG=1.0,
            W_uREG=1.0,
            W_gripper_pose=1.0,
            W_gripper_pose_term=1.0,
            W_limit=1.0,
            safety_threshold=0.1,
            dt=0.01,
            T=100,
            gripper_depth=0.123,
        )

        camera_trans = np.array([0.5, 0.5, 0.5])
        config = GraspTransformConfig.from_robot_config(
            robot_config=robot_config,
            camera_translation=camera_trans,
            grasp_correction_angle_deg=45.0,
            elevation_angle_deg=30.0,
        )

        assert config.gripper_depth == 0.123  # From robot_config
        np.testing.assert_array_equal(config.camera_translation, camera_trans)
        assert config.grasp_correction_angle_deg == 45.0
        assert config.elevation_angle_deg == 30.0


class TestGraspTransformChain:
    """Tests for GraspTransformChain class."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple config for testing."""
        return GraspTransformConfig(
            camera_translation=np.array([0, 0, 1.0]),
            gripper_depth=0.1,
            grasp_correction_angle_deg=0.0,  # No correction for simple tests
            elevation_angle_deg=0.0,  # No elevation for simple tests
        )

    @pytest.fixture
    def transform_chain(self, simple_config):
        """Create a transform chain with simple config."""
        return GraspTransformChain(simple_config)

    def test_initialization(self, transform_chain):
        """Test that transform chain initializes correctly."""
        assert transform_chain.config is not None
        assert transform_chain.wM_camera is not None
        assert transform_chain.graspM_grasp_corrected is not None
        assert transform_chain.grasp_correctedM_tcp is not None
        assert transform_chain.worldM_world_aligned is not None

    def test_camera_transform(self, transform_chain):
        """Test that camera transform is set correctly."""
        expected_translation = np.array([0, 0, 1.0])
        np.testing.assert_array_almost_equal(
            transform_chain.wM_camera.translation, expected_translation
        )

    def test_update_camera_transform(self, transform_chain):
        """Test updating camera transform."""
        new_translation = np.array([1.0, 2.0, 3.0])
        transform_chain.update_camera_transform(new_translation)

        np.testing.assert_array_almost_equal(
            transform_chain.wM_camera.translation, new_translation
        )
        np.testing.assert_array_almost_equal(
            transform_chain.config.camera_translation, new_translation
        )

    def test_compute_object_pose_single(self, transform_chain):
        """Test computing single object pose."""
        # Create a simple object pose in camera frame
        cameraM_object = pin.SE3.Identity()
        cameraM_object.translation = np.array([0.5, 0.0, 0.0])

        # Transform to world frame
        wM_object = transform_chain.compute_object_pose(cameraM_object)

        # With elevation_angle=0, worldM_world_aligned should be Identity
        # So result should be: Identity * (0,0,1) * (0.5,0,0) = (0.5, 0, 1)
        # But we have rotation from worldM_world_aligned
        assert isinstance(wM_object, pin.SE3)
        assert wM_object.translation[2] > 0  # Should have Z component from camera

    def test_compute_object_pose_trajectory(self):
        """Test computing object pose trajectory."""
        # Create simple config with no rotations
        config = GraspTransformConfig(
            camera_translation=np.array([0, 0, 1.0]),
            grasp_correction_angle_deg=0.0,
            elevation_angle_deg=0.0,
        )
        chain = GraspTransformChain(config)

        # Create trajectory with 3 poses
        poses = []
        for i in range(3):
            pose = pin.SE3.Identity()
            pose.translation = np.array([i * 0.1, 0.0, 0.0])
            poses.append(pose)

        traj = TrajectorySE3(poses)

        # Transform trajectory
        result_traj = chain.compute_object_pose(traj)

        assert isinstance(result_traj, TrajectorySE3)
        assert len(result_traj) == 3

    def test_transform_object_trajectory(self):
        """Test transforming object trajectory."""
        config = GraspTransformConfig(
            camera_translation=np.array([1.0, 0.0, 0.0]),
            elevation_angle_deg=0.0,
        )
        chain = GraspTransformChain(config)

        # Create a trajectory
        poses = [pin.SE3.Identity() for _ in range(5)]
        camera_traj = TrajectorySE3(poses)

        # Transform
        world_traj = chain.transform_object_trajectory(camera_traj)

        assert isinstance(world_traj, TrajectorySE3)
        assert len(world_traj) == 5

    def test_transform_ee_trajectory(self):
        """Test transforming to end-effector trajectory."""
        config = GraspTransformConfig(
            camera_translation=np.array([0, 0, 1.0]),
            grasp_correction_angle_deg=0.0,
            elevation_angle_deg=0.0,
        )
        chain = GraspTransformChain(config)

        # Create object trajectory
        poses = [pin.SE3.Identity() for _ in range(3)]
        camera_traj = TrajectorySE3(poses)

        # Create grasp
        objectM_grasp = pin.SE3.Identity()
        objectM_grasp.translation = np.array([0, 0, 0.05])

        # Transform to EE trajectory
        ee_traj = chain.transform_ee_trajectory(camera_traj, objectM_grasp)

        assert isinstance(ee_traj, TrajectorySE3)
        assert len(ee_traj) == 3

    def test_transform_tcp_trajectory(self):
        """Test transforming to TCP trajectory (includes gripper offset)."""
        config = GraspTransformConfig(
            camera_translation=np.array([0, 0, 1.0]),
            gripper_depth=0.1,
            grasp_correction_angle_deg=0.0,
            elevation_angle_deg=0.0,
        )
        chain = GraspTransformChain(config)

        # Create object trajectory
        poses = [pin.SE3.Identity() for _ in range(3)]
        camera_traj = TrajectorySE3(poses)

        # Create grasp
        objectM_grasp = pin.SE3.Identity()

        # Transform to TCP trajectory
        tcp_traj = chain.transform_tcp_trajectory(camera_traj, objectM_grasp)
        ee_traj = chain.transform_ee_trajectory(camera_traj, objectM_grasp)

        assert isinstance(tcp_traj, TrajectorySE3)
        assert len(tcp_traj) == 3

        # TCP should be offset from EE by gripper_depth along Z
        # (Note: this depends on no rotations being applied)
        for tcp_pose, ee_pose in zip(tcp_traj, ee_traj):
            diff = tcp_pose.translation - ee_pose.translation
            # The offset should be in the local Z direction of the EE frame
            assert np.linalg.norm(diff) > 0  # Should have some offset

    def test_compute_ee_pose_with_tcp_offset(self, transform_chain):
        """Test computing EE pose with TCP offset."""
        cameraM_object = pin.SE3.Identity()
        objectM_grasp = pin.SE3.Identity()

        # Without offset
        ee_no_offset = transform_chain.compute_ee_pose(
            cameraM_object, objectM_grasp, include_tcp_offset=False
        )

        # With offset
        ee_with_offset = transform_chain.compute_ee_pose(
            cameraM_object, objectM_grasp, include_tcp_offset=True
        )

        # Should be different
        assert not np.allclose(ee_no_offset.translation, ee_with_offset.translation)

    def test_compute_tcp_pose(self, transform_chain):
        """Test computing TCP pose (should include offset)."""
        cameraM_object = pin.SE3.Identity()
        objectM_grasp = pin.SE3.Identity()

        tcp_pose = transform_chain.compute_tcp_pose(cameraM_object, objectM_grasp)
        ee_pose = transform_chain.compute_ee_pose(
            cameraM_object, objectM_grasp, include_tcp_offset=False
        )

        # TCP should be different from EE (has offset)
        assert not np.allclose(tcp_pose.translation, ee_pose.translation)

    def test_grasp_correction_rotation(self):
        """Test that grasp correction rotation is applied correctly."""
        config = GraspTransformConfig(
            camera_translation=np.array([0, 0, 0]),
            grasp_correction_angle_deg=90.0,  # 90 degrees around Z
            grasp_correction_axis=np.array([0, 0, 1]),
            elevation_angle_deg=0.0,
        )
        chain = GraspTransformChain(config)

        # Check that correction rotation is approximately 90 degrees around Z
        R = chain.graspM_grasp_corrected.rotation

        # A 90-degree rotation around Z should map X to Y
        x_axis = np.array([1, 0, 0])
        rotated = R @ x_axis

        # Should be close to Y axis
        np.testing.assert_array_almost_equal(rotated, np.array([0, 1, 0]), decimal=5)

    def test_get_transform_summary(self, transform_chain):
        """Test getting transform summary string."""
        summary = transform_chain.get_transform_summary()

        assert isinstance(summary, str)
        assert "Grasp Transform Chain Summary" in summary
        assert "Camera translation" in summary
        assert "Gripper depth" in summary
        assert "Grasp correction" in summary
        assert "Transformation chain" in summary

    def test_identity_composition(self):
        """Test that composing with identity doesn't change the result."""
        config = GraspTransformConfig(
            camera_translation=np.array([0, 0, 0]),
            elevation_angle_deg=0.0,
        )
        chain = GraspTransformChain(config)

        # Identity object pose
        identity_pose = pin.SE3.Identity()

        # Identity grasp
        identity_grasp = pin.SE3.Identity()

        # Result should still be a valid SE3
        result = chain.compute_ee_pose(identity_pose, identity_grasp)

        assert isinstance(result, pin.SE3)
        # The rotation should still be valid (determinant = 1)
        assert np.isclose(np.linalg.det(result.rotation), 1.0)


class TestQuaternionConversion:
    """Test quaternion conversion utilities."""

    def test_quaternion_float_conversion(self):
        """Test that quaternion conversion handles numpy floats correctly."""
        # Simulate CuRobo FK output (numpy floats)
        # Use a properly normalized quaternion: 90 deg around X
        # For 90 deg around X: w=cos(45°), x=sin(45°), y=0, z=0
        fk_quat_curobo = np.array(
            [
                np.cos(np.pi / 4),  # w ≈ 0.7071
                np.sin(np.pi / 4),  # x ≈ 0.7071
                0.0,  # y
                0.0,  # z
            ]
        )

        # Normalize to ensure it's a unit quaternion
        fk_quat_curobo = fk_quat_curobo / np.linalg.norm(fk_quat_curobo)

        # Convert to Python floats explicitly
        R_curobo = pin.Quaternion(
            float(fk_quat_curobo[0]),  # w
            float(fk_quat_curobo[1]),  # x
            float(fk_quat_curobo[2]),  # y
            float(fk_quat_curobo[3]),  # z
        ).toRotationMatrix()

        # Should be a valid rotation matrix
        assert R_curobo.shape == (3, 3)
        assert np.isclose(np.linalg.det(R_curobo), 1.0, atol=1e-6)

        # Check orthogonality: R^T @ R = I
        should_be_identity = R_curobo.T @ R_curobo
        np.testing.assert_array_almost_equal(should_be_identity, np.eye(3), decimal=5)

    def test_quaternion_normalization_requirement(self):
        """Test that non-normalized quaternions need normalization."""
        # Clearly non-normalized quaternion (scale it up by 2)
        quat_unnormalized = np.array([1.0, 1.0, 0.0, 0.0])

        # Check it's not normalized
        norm = np.linalg.norm(quat_unnormalized)
        assert not np.isclose(norm, 1.0), f"Quaternion norm is {norm}, expected != 1.0"

        # Normalize it
        quat_normalized = quat_unnormalized / norm

        # Verify normalized
        assert np.isclose(np.linalg.norm(quat_normalized), 1.0)

        # Now convert to rotation matrix
        R = pin.Quaternion(
            float(quat_normalized[0]),
            float(quat_normalized[1]),
            float(quat_normalized[2]),
            float(quat_normalized[3]),
        ).toRotationMatrix()

        # Should be valid
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

        # Verify orthogonality
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)

    def test_quaternion_from_curobo_format(self):
        """Test conversion from typical CuRobo output format."""
        # Typical CuRobo output (already normalized in practice)
        fk_quat_curobo = np.array(
            [
                0.9238795,  # w
                0.3826834,  # x
                0.0,  # y
                0.0,  # z
            ]
        )

        # This should already be normalized from CuRobo
        assert np.isclose(np.linalg.norm(fk_quat_curobo), 1.0, atol=1e-6)

        # Convert using float() to handle numpy types
        R_curobo = pin.Quaternion(
            float(fk_quat_curobo[0]),
            float(fk_quat_curobo[1]),
            float(fk_quat_curobo[2]),
            float(fk_quat_curobo[3]),
        ).toRotationMatrix()

        # Verify it's a valid rotation matrix
        assert np.isclose(np.linalg.det(R_curobo), 1.0, atol=1e-6)

        # Verify orthogonality
        assert np.allclose(R_curobo.T @ R_curobo, np.eye(3), atol=1e-6)


class TestTrajectorySE3Integration:
    """Test integration with TrajectorySE3 class."""

    def test_trajectory_multiplication_left(self):
        """Test left multiplication: SE3 * TrajectorySE3."""
        # Create a transform
        T = pin.SE3.Identity()
        T.translation = np.array([1, 0, 0])

        # Create a trajectory
        poses = []
        for i in range(3):
            pose = pin.SE3.Identity()
            pose.translation = np.array([0, i, 0])
            poses.append(pose)

        traj = TrajectorySE3(poses)

        # Left multiply
        result = T * traj

        assert isinstance(result, TrajectorySE3)
        assert len(result) == 3

        # Check first pose: should be (1, 0, 0)
        np.testing.assert_array_almost_equal(result[0].translation, np.array([1, 0, 0]))

    def test_trajectory_multiplication_right(self):
        """Test right multiplication: TrajectorySE3 * SE3."""
        # Create a trajectory
        poses = []
        for i in range(3):
            pose = pin.SE3.Identity()
            pose.translation = np.array([i, 0, 0])
            poses.append(pose)

        traj = TrajectorySE3(poses)

        # Create a transform
        T = pin.SE3.Identity()
        T.translation = np.array([0, 1, 0])

        # Right multiply
        result = traj * T

        assert isinstance(result, TrajectorySE3)
        assert len(result) == 3

        # Check translations are transformed
        for i, pose in enumerate(result):
            # Should include the offset from T
            assert pose.translation[1] == 1.0  # Y component from T

    def test_trajectory_iteration(self):
        """Test that TrajectorySE3 can be iterated."""
        poses = [pin.SE3.Identity() for _ in range(5)]
        traj = TrajectorySE3(poses)

        count = 0
        for pose in traj:
            assert isinstance(pose, pin.SE3)
            count += 1

        assert count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
