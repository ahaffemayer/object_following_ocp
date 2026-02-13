"""
Unit tests for DataLoader and related classes.

To run: pytest test_dataloader.py -v
"""

import json
import pathlib

import numpy as np
import pinocchio as pin
import pytest
import yaml

from object_following_ocp.data_loader import (
    DataLoader,
    GraspPose,
    ObjectInfo,
    PoseData,
)
from object_following_ocp.trajectories import TrajectorySE3


class TestPoseData:
    """Tests for PoseData dataclass"""

    def test_pose_data_creation(self):
        """Test creating a PoseData object"""
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])

        pose = PoseData(
            im_id=0,
            object_id=2,
            score=0.5,
            R=R,
            t=t,
            bbox_visib=[100, 200, 300, 400],
            time=1.0,
        )

        assert pose.im_id == 0
        assert pose.object_id == 2
        assert pose.score == 0.5
        np.testing.assert_array_equal(pose.R, R)
        np.testing.assert_array_equal(pose.t, t)
        assert pose.bbox_visib == [100, 200, 300, 400]
        assert pose.time == 1.0

    def test_pose_to_SE3(self):
        """Test conversion from PoseData to SE3"""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t = np.array([1.0, 2.0, 3.0])

        pose = PoseData(
            im_id=0,
            object_id=2,
            score=0.5,
            R=R,
            t=t,
            bbox_visib=[100, 200, 300, 400],
            time=1.0,
        )

        se3 = pose.to_SE3()

        assert isinstance(se3, pin.SE3)
        np.testing.assert_array_almost_equal(se3.rotation, R)
        np.testing.assert_array_almost_equal(se3.translation, t)


class TestGraspPose:
    """Tests for GraspPose dataclass"""

    def test_grasp_pose_creation(self):
        """Test creating a GraspPose object"""
        position = np.array([0.1, 0.2, 0.3])
        orientation = np.array([0.7071, 0.7071, 0.0, 0.0])  # [w, x, y, z]

        grasp = GraspPose(
            name="grasp_0", confidence=0.95, position=position, orientation=orientation
        )

        assert grasp.name == "grasp_0"
        assert grasp.confidence == 0.95
        np.testing.assert_array_equal(grasp.position, position)
        np.testing.assert_array_equal(grasp.orientation, orientation)

    def test_grasp_to_SE3(self):
        """Test conversion from GraspPose to SE3"""
        position = np.array([0.1, 0.2, 0.3])
        # Identity quaternion [w, x, y, z]
        orientation = np.array([1.0, 0.0, 0.0, 0.0])

        grasp = GraspPose(
            name="grasp_0", confidence=0.95, position=position, orientation=orientation
        )

        se3 = grasp.to_SE3()

        assert isinstance(se3, pin.SE3)
        np.testing.assert_array_almost_equal(se3.translation, position)
        # Identity quaternion should give identity rotation
        np.testing.assert_array_almost_equal(se3.rotation, np.eye(3), decimal=5)

    def test_grasp_quaternion_conversion(self):
        """Test that quaternion conversion follows correct convention"""
        position = np.array([0.0, 0.0, 0.0])
        # 90 degree rotation around z-axis: [cos(45°), 0, 0, sin(45°)]
        orientation = np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865476])

        grasp = GraspPose(
            name="grasp_test",
            confidence=0.9,
            position=position,
            orientation=orientation,
        )

        se3 = grasp.to_SE3()

        # Expected rotation matrix for 90° around z
        expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        np.testing.assert_array_almost_equal(se3.rotation, expected_R, decimal=5)


class TestObjectInfo:
    """Tests for ObjectInfo dataclass"""

    def test_object_info_creation(self):
        """Test creating an ObjectInfo object"""
        mesh_path = pathlib.Path("/path/to/mesh.obj")
        texture_path = pathlib.Path("/path/to/texture.png")

        obj_info = ObjectInfo(
            mesh_id="abc123",
            score=0.75,
            scale=0.1,
            mesh_path=mesh_path,
            texture_path=texture_path,
        )

        assert obj_info.mesh_id == "abc123"
        assert obj_info.score == 0.75
        assert obj_info.scale == 0.1
        assert obj_info.mesh_path == mesh_path
        assert obj_info.texture_path == texture_path

    def test_object_info_optional_texture(self):
        """Test ObjectInfo with optional texture path"""
        mesh_path = pathlib.Path("/path/to/mesh.obj")

        obj_info = ObjectInfo(
            mesh_id="abc123", score=0.75, scale=0.1, mesh_path=mesh_path
        )

        assert obj_info.texture_path is None


class TestDataLoader:
    """Tests for DataLoader class"""

    @pytest.fixture
    def temp_files(self, tmp_path):
        """Create temporary test files"""
        # Create object trajectory JSON
        object_data = {
            "objects": {
                "2": {
                    "mesh": "0d0d1c59b0474d2ea92ce2e172c9f56a",
                    "score": 0.7291240692138672,
                    "scale": 0.07171772395566117,
                }
            },
            "poses": [
                {
                    "im_id": 0,
                    "object_id": 2,
                    "score": 0.38671875,
                    "R": [
                        [
                            -0.9735720171552442,
                            0.22036993567234633,
                            -0.059955140430447834,
                        ],
                        [-0.10804320588553487, -0.675714976133025, -0.7292022604816316],
                        [-0.2012068415194789, -0.7034531700698429, 0.6816666666666664],
                    ],
                    "t": [
                        -0.22166704223417696,
                        -0.0755651959446513,
                        0.8099392606970864,
                    ],
                    "bbox_visib": [268, 82, 522, 310],
                    "time": 0.0,
                },
                {
                    "im_id": 1,
                    "object_id": 2,
                    "score": 0.39,
                    "R": [
                        [-0.97, 0.22, -0.06],
                        [-0.11, -0.68, -0.73],
                        [-0.20, -0.70, 0.68],
                    ],
                    "t": [-0.21, -0.08, 0.81],
                    "bbox_visib": [270, 85, 520, 308],
                    "time": 0.033,
                },
            ],
        }

        # Create grasp poses YAML
        grasp_data = {
            "format": "isaac_grasp",
            "format_version": 1.0,
            "grasps": {
                "grasp_0": {
                    "confidence": 0.9850150942802429,
                    "position": [
                        0.09796400394090717,
                        0.04605246219166667,
                        0.0335818800359627,
                    ],
                    "orientation": {
                        "w": -0.02462670080231983,
                        "xyz": [
                            0.8126014170473901,
                            0.007884706018450371,
                            -0.582245905121856,
                        ],
                    },
                },
                "grasp_1": {
                    "confidence": 0.9798907041549683,
                    "position": [
                        -0.08693484936109479,
                        -0.1273814078854665,
                        -0.033933113214407326,
                    ],
                    "orientation": {
                        "w": -0.13208252084043554,
                        "xyz": [
                            0.5461539955528465,
                            0.18659771640219328,
                            0.8058854217961657,
                        ],
                    },
                },
            },
        }

        # Create scales JSON
        scales_data = [
            {
                "Name": "0d0d1c59b0474d2ea92ce2e172c9f56a",
                "scale": 0.09,
                "scale_from_dataset": 0.136,
            },
            {"Name": "other_mesh_id", "scale": 0.08, "scale_from_dataset": 0.162},
        ]

        # Write files
        traj_path = tmp_path / "trajectory.json"
        grasp_path = tmp_path / "grasps.yaml"
        scales_path = tmp_path / "scales.json"

        with open(traj_path, "w") as f:
            json.dump(object_data, f)

        with open(grasp_path, "w") as f:
            yaml.dump(grasp_data, f)

        with open(scales_path, "w") as f:
            json.dump(scales_data, f)

        # Create mesh directory structure
        # With .parent.parent: traj_path.parent.parent / "meshes"
        # traj_path = tmp_path/trajectory.json
        # .parent = tmp_path
        # .parent.parent = tmp_path's parent
        mesh_base_dir = (
            traj_path.parent.parent / "meshes" / "0d0d1c59b0474d2ea92ce2e172c9f56a"
        )
        mesh_base_dir.mkdir(parents=True, exist_ok=True)
        (mesh_base_dir / "0d0d1c59b0474d2ea92ce2e172c9f56a.obj").touch()
        (mesh_base_dir / "material_0.png").touch()

        return {
            "trajectory": traj_path,
            "grasp": grasp_path,
            "scales": scales_path,
            "tmp_path": tmp_path,
        }

    def test_dataloader_initialization(self, temp_files):
        """Test DataLoader loads files correctly"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            grasp_poses_SE3_path=temp_files["grasp"],
            scales_path=temp_files["scales"],
        )

        # Check object info
        assert loader.object_id == 2
        assert loader.object_info.mesh_id == "0d0d1c59b0474d2ea92ce2e172c9f56a"
        assert loader.object_info.score == pytest.approx(0.7291240692138672)
        # Scale should be overridden by scales file
        assert loader.object_info.scale == pytest.approx(0.09)

        # Check poses
        assert len(loader.poses) == 2
        assert loader.poses[0].im_id == 0
        assert loader.poses[0].object_id == 2
        assert loader.poses[1].im_id == 1

        # Check grasp poses
        assert len(loader.grasp_poses) == 2
        assert loader.grasp_poses[0].name == "grasp_0"
        assert loader.grasp_poses[0].confidence == pytest.approx(0.9850150942802429)
        assert loader.grasp_poses[1].name == "grasp_1"

    def test_scale_override(self, temp_files):
        """Test that scales file overrides object trajectory scale"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            grasp_poses_SE3_path=temp_files["grasp"],
            scales_path=temp_files["scales"],
        )

        # Scale from scales.json (0.09) should override trajectory (0.0717...)
        assert loader.object_info.scale == pytest.approx(0.09)
        assert loader.object_info.scale != pytest.approx(0.07171772395566117)

    def test_mesh_paths_construction(self, temp_files):
        """Test that mesh and texture paths are constructed correctly"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            grasp_poses_SE3_path=temp_files["grasp"],
            scales_path=temp_files["scales"],
        )

        # With .parent.parent:
        # temp_files["trajectory"] = tmp_path/trajectory.json
        # .parent = tmp_path
        # .parent.parent = tmp_path's parent
        # Then /meshes/ is added
        expected_mesh = (
            temp_files["trajectory"].parent.parent
            / "meshes"
            / "0d0d1c59b0474d2ea92ce2e172c9f56a"
            / "0d0d1c59b0474d2ea92ce2e172c9f56a.obj"
        )
        expected_texture = (
            temp_files["trajectory"].parent.parent
            / "meshes"
            / "0d0d1c59b0474d2ea92ce2e172c9f56a"
            / "material_0.png"
        )

        assert loader.object_info.mesh_path == expected_mesh
        assert loader.object_info.texture_path == expected_texture

    def test_to_trajectory_SE3(self, temp_files):
        """Test conversion to TrajectorySE3"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            grasp_poses_SE3_path=temp_files["grasp"],
            scales_path=temp_files["scales"],
        )

        trajectory = loader.to_trajectory_SE3()

        assert isinstance(trajectory, TrajectorySE3)
        assert len(trajectory) == 2

        # Check first pose
        se3_0 = trajectory[0]
        assert isinstance(se3_0, pin.SE3)
        expected_t_0 = np.array(
            [-0.22166704223417696, -0.0755651959446513, 0.8099392606970864]
        )
        np.testing.assert_array_almost_equal(se3_0.translation, expected_t_0)

    def test_best_grasp_property(self, temp_files):
        """Test best_grasp property returns highest confidence grasp"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            grasp_poses_SE3_path=temp_files["grasp"],
            scales_path=temp_files["scales"],
        )

        best = loader.best_grasp

        assert best.name == "grasp_0"
        assert best.confidence == pytest.approx(0.9850150942802429)

    def test_best_grasp_SE3_property(self, temp_files):
        """Test best_grasp_SE3 property returns SE3 object"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            grasp_poses_SE3_path=temp_files["grasp"],
            scales_path=temp_files["scales"],
        )

        best_se3 = loader.best_grasp_SE3

        assert isinstance(best_se3, pin.SE3)
        expected_position = np.array(
            [0.09796400394090717, 0.04605246219166667, 0.0335818800359627]
        )
        np.testing.assert_array_almost_equal(best_se3.translation, expected_position)

    def test_invalid_object_count(self, tmp_path):
        """Test that DataLoader raises error with multiple objects in file"""
        # Create file with multiple objects
        multi_object_data = {
            "objects": {
                "1": {"mesh": "mesh1", "score": 0.5, "scale": 0.1},
                "2": {"mesh": "mesh2", "score": 0.6, "scale": 0.2},
            },
            "poses": [],
        }

        traj_path = tmp_path / "multi_traj.json"
        with open(traj_path, "w") as f:
            json.dump(multi_object_data, f)

        grasp_path = tmp_path / "grasp.yaml"
        with open(grasp_path, "w") as f:
            yaml.dump({"grasps": {}}, f)

        scales_path = tmp_path / "scales.json"
        with open(scales_path, "w") as f:
            json.dump([], f)

        # Create mesh directory using .parent.parent logic
        mesh_dir = traj_path.parent.parent / "meshes" / "mesh1"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="Expected 1 object per file"):
            DataLoader(
                object_trajectory_path=traj_path,
                scales_path=scales_path,
                grasp_poses_SE3_path=grasp_path,
            )

    def test_scale_fallback(self, tmp_path):
        """Test that object scale is used when not in scales file"""
        # Create data with mesh_id not in scales
        object_data = {
            "objects": {"1": {"mesh": "unknown_mesh_id", "score": 0.5, "scale": 0.123}},
            "poses": [],
        }

        scales_data = [
            {"Name": "other_mesh", "scale": 0.456, "scale_from_dataset": 0.789}
        ]

        traj_path = tmp_path / "traj.json"
        grasp_path = tmp_path / "grasp.yaml"
        scales_path = tmp_path / "scales.json"

        with open(traj_path, "w") as f:
            json.dump(object_data, f)

        with open(grasp_path, "w") as f:
            yaml.dump({"grasps": {}}, f)

        with open(scales_path, "w") as f:
            json.dump(scales_data, f)

        # Create mesh directory using .parent.parent logic
        mesh_dir = traj_path.parent.parent / "meshes" / "unknown_mesh_id"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        (mesh_dir / "unknown_mesh_id.obj").touch()
        (mesh_dir / "material_0.png").touch()

        loader = DataLoader(
            object_trajectory_path=traj_path,
            scales_path=scales_path,
            grasp_poses_SE3_path=grasp_path,
        )

        # Should use scale from object file
        assert loader.object_info.scale == pytest.approx(0.123)

    def test_pose_data_numpy_arrays(self, temp_files):
        """Test that R and t are properly converted to numpy arrays"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            grasp_poses_SE3_path=temp_files["grasp"],
            scales_path=temp_files["scales"],
        )

        pose = loader.poses[0]

        assert isinstance(pose.R, np.ndarray)
        assert pose.R.shape == (3, 3)
        assert isinstance(pose.t, np.ndarray)
        assert pose.t.shape == (3,)

    def test_grasp_pose_numpy_arrays(self, temp_files):
        """Test that position and orientation are properly converted to numpy arrays"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            grasp_poses_SE3_path=temp_files["grasp"],
            scales_path=temp_files["scales"],
        )

        grasp = loader.grasp_poses[0]

        assert isinstance(grasp.position, np.ndarray)
        assert grasp.position.shape == (3,)
        assert isinstance(grasp.orientation, np.ndarray)
        assert grasp.orientation.shape == (4,)

    def test_no_grasps_by_default(self, temp_files):
        """Test that grasps are not loaded by default"""
        loader = DataLoader(
            object_trajectory_path=temp_files["trajectory"],
            scales_path=temp_files["scales"],
            # Note: no grasp path, no load_grasps flag
        )

        assert not loader.has_grasps
        assert len(loader.grasp_poses) == 0
        assert loader.best_grasp is None
        assert loader.best_grasp_SE3 is None

    def test_auto_load_grasps(self, tmp_path):
        """Test auto-loading grasps based on mesh_id"""
        mesh_id = "test_mesh_123"

        # Create trajectory file
        object_data = {
            "objects": {"1": {"mesh": mesh_id, "score": 0.5, "scale": 0.1}},
            "poses": [],
        }

        # Create directory structure
        json_dir = tmp_path / "json"
        json_dir.mkdir()
        traj_path = json_dir / "traj.json"

        with open(traj_path, "w") as f:
            json.dump(object_data, f)

        # Create scales file
        scales_path = tmp_path / "scales.json"
        with open(scales_path, "w") as f:
            json.dump([], f)

        # Create grasps file in expected location
        grasps_dir = tmp_path / "filtered_grasps"
        grasps_dir.mkdir()
        grasp_path = grasps_dir / f"{mesh_id}_filtered.yml"

        grasp_data = {
            "grasps": {
                "grasp_0": {
                    "confidence": 0.95,
                    "position": [0.1, 0.2, 0.3],
                    "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
                }
            }
        }

        with open(grasp_path, "w") as f:
            yaml.dump(grasp_data, f)

        # Create mesh directory
        mesh_dir = tmp_path / "meshes" / mesh_id
        mesh_dir.mkdir(parents=True, exist_ok=True)
        (mesh_dir / f"{mesh_id}.obj").touch()
        (mesh_dir / "material_0.png").touch()

        # Test auto-loading
        loader = DataLoader(
            object_trajectory_path=traj_path, scales_path=scales_path, load_grasps=True
        )

        assert loader.has_grasps
        assert len(loader.grasp_poses) == 1
        assert loader.best_grasp is not None
        assert loader.best_grasp.confidence == pytest.approx(0.95)

    def test_auto_load_grasps_not_found(self, tmp_path):
        """Test that auto-loading fails gracefully when grasp file doesn't exist"""
        # Create trajectory file
        object_data = {
            "objects": {"1": {"mesh": "nonexistent_mesh", "score": 0.5, "scale": 0.1}},
            "poses": [],
        }

        json_dir = tmp_path / "json"
        json_dir.mkdir()
        traj_path = json_dir / "traj.json"

        with open(traj_path, "w") as f:
            json.dump(object_data, f)

        scales_path = tmp_path / "scales.json"
        with open(scales_path, "w") as f:
            json.dump([], f)

        # Create mesh directory
        mesh_dir = tmp_path / "meshes" / "nonexistent_mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        (mesh_dir / "nonexistent_mesh.obj").touch()
        (mesh_dir / "material_0.png").touch()

        # Should not raise, just not load grasps
        loader = DataLoader(
            object_trajectory_path=traj_path, scales_path=scales_path, load_grasps=True
        )

        assert not loader.has_grasps
        assert len(loader.grasp_poses) == 0


class TestDataLoaderWithRealFiles:
    """
    Tests using the actual files from your workspace.

    Fill in the expected values by running this script first:

    ```python
    from object_following_ocp.data_loader import DataLoader
    import pathlib

    grasp_path = pathlib.Path("ressources/filtered_grasps/0d0d1c59b0474d2ea92ce2e172c9f56a_filtered.yml")
    object_traj_path = pathlib.Path("ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json")
    scale_path = pathlib.Path("ressources/grasps_scales.json")

    loader = DataLoader(object_traj_path, grasp_path, scale_path)

    print(f"object_id: {loader.object_id}")
    print(f"mesh_id: {loader.object_info.mesh_id}")
    print(f"scale: {loader.object_info.scale}")
    print(f"score: {loader.object_info.score}")
    print(f"num_poses: {len(loader.poses)}")
    print(f"num_grasps: {len(loader.grasp_poses)}")
    print(f"first_pose_t: {loader.poses[0].t}")
    print(f"first_pose_score: {loader.poses[0].score}")
    print(f"best_grasp_name: {loader.best_grasp.name}")
    print(f"best_grasp_confidence: {loader.best_grasp.confidence}")
    print(f"best_grasp_position: {loader.best_grasp.position}")
    ```
    """

    @pytest.fixture
    def real_file_paths(self):
        """Real file paths from your workspace"""
        base = pathlib.Path("/workspaces/object_following_ocp")
        return {
            "grasp": base
            / "ressources/filtered_grasps/0d0d1c59b0474d2ea92ce2e172c9f56a_filtered.yml",
            "trajectory": base
            / "ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json",
            "scales": base / "ressources/grasps_scales.json",
        }

    @pytest.mark.skipif(
        not pathlib.Path("/workspaces/object_following_ocp").exists(),
        reason="Real files not available in this environment",
    )
    def test_real_files_load(self, real_file_paths):
        """Test that real files load without errors"""
        loader = DataLoader(
            object_trajectory_path=real_file_paths["trajectory"],
            grasp_poses_SE3_path=real_file_paths["grasp"],
            scales_path=real_file_paths["scales"],
        )

        # Basic sanity checks
        assert loader.object_id is not None
        assert loader.object_info.mesh_id == "cbb0cdd9bbcc4fdfa2e16db1db4cda61"
        assert len(loader.poses) > 0
        assert len(loader.grasp_poses) > 0

        # Test that we can convert to trajectory
        trajectory = loader.to_trajectory_SE3()
        assert len(trajectory) == len(loader.poses)

        # Test that we can get best grasp
        best = loader.best_grasp
        assert best.confidence > 0.0
        assert best.confidence <= 1.0

    @pytest.mark.skipif(
        not pathlib.Path("/workspaces/object_following_ocp").exists(),
        reason="Real files not available in this environment",
    )
    def test_real_files_expected_values(self, real_file_paths):
        """
        Test expected values from real files.

        TODO: Fill in these values after running the DataLoader manually (see class docstring)
        """
        loader = DataLoader(
            object_trajectory_path=real_file_paths["trajectory"],
            grasp_poses_SE3_path=real_file_paths["grasp"],
            scales_path=real_file_paths["scales"],
        )

        # Object info
        assert loader.object_id == 2
        assert loader.object_info.mesh_id == "cbb0cdd9bbcc4fdfa2e16db1db4cda61"
        assert loader.object_info.scale == pytest.approx(0.07)
        assert loader.object_info.score == pytest.approx(0.7291240692138672)

        # Counts
        assert len(loader.poses) == 99
        assert len(loader.grasp_poses) == 10

        # First pose
        expected_t_0 = np.array(
            [-0.22166704223417696, -0.0755651959446513, 0.8099392606970864]
        )
        np.testing.assert_array_almost_equal(loader.poses[0].t, expected_t_0)
        assert loader.poses[0].score == pytest.approx(0.38671875)
        assert loader.poses[0].im_id == 0
        assert loader.poses[0].object_id == 2

        # First pose rotation (sample check of first row)
        expected_R_0_row0 = np.array(
            [-0.9735720171552442, 0.22036993567234633, -0.059955140430447834]
        )
        np.testing.assert_array_almost_equal(loader.poses[0].R[0], expected_R_0_row0)

        # Best grasp
        assert loader.best_grasp.name == "grasp_2"
        assert loader.best_grasp.confidence == pytest.approx(0.9861612319946289)
        expected_grasp_pos = np.array(
            [-0.05379145939103063, 0.06558237853773982, -0.07833266153636968]
        )
        np.testing.assert_array_almost_equal(
            loader.best_grasp.position, expected_grasp_pos
        )

        expected_grasp_ori = np.array(
            [
                0.9513417916293845,
                -0.01512138067120747,
                0.30711066173445567,
                -0.02007936241552457,
            ]
        )
        np.testing.assert_array_almost_equal(
            loader.best_grasp.orientation, expected_grasp_ori
        )

        # Trajectory conversion
        trajectory = loader.to_trajectory_SE3()
        assert len(trajectory) == 99
        assert isinstance(trajectory[0].translation, np.ndarray)

        # Best grasp SE3
        best_se3 = loader.best_grasp_SE3
        np.testing.assert_array_almost_equal(best_se3.translation, expected_grasp_pos)

        # Mesh paths (check construction is correct)
        assert (
            loader.object_info.mesh_path.name == "cbb0cdd9bbcc4fdfa2e16db1db4cda61.obj"
        )
        assert loader.object_info.texture_path.name == "material_0.png"
        # Note: mesh files may be in a different location, so we just check the filename


class TestRobotConfig:
    """Tests for RobotConfig dataclass"""

    def test_robot_config_creation(self):
        """Test creating a RobotConfig object"""
        from object_following_ocp.data_loader import RobotConfig

        config = RobotConfig(
            W_xREG=0.001,
            W_uREG=0.001,
            W_gripper_pose=1000.0,
            W_gripper_pose_term=100.0,
            W_limit=100.0,
            safety_threshold=0.02,
            dt=0.01,
            gripper_depth=0.1034,
            T=15,
        )

        assert config.W_xREG == pytest.approx(0.001)
        assert config.W_uREG == pytest.approx(0.001)
        assert config.W_gripper_pose == pytest.approx(1000.0)
        assert config.W_gripper_pose_term == pytest.approx(100.0)
        assert config.W_limit == pytest.approx(100.0)
        assert config.safety_threshold == pytest.approx(0.02)
        assert config.dt == pytest.approx(0.01)
        assert config.gripper_depth == pytest.approx(0.1034)
        assert config.T == pytest.approx(15)

    def test_robot_config_attributes(self):
        """Test that RobotConfig has all expected attributes"""
        from object_following_ocp.data_loader import RobotConfig

        config = RobotConfig(
            W_xREG=1.0,
            W_uREG=2.0,
            W_gripper_pose=3.0,
            W_gripper_pose_term=4.0,
            W_limit=5.0,
            safety_threshold=6.0,
            dt=7.0,
            gripper_depth=8.0,
            T=15,
        )

        # Verify all attributes are accessible
        assert hasattr(config, "W_xREG")
        assert hasattr(config, "W_uREG")
        assert hasattr(config, "W_gripper_pose")
        assert hasattr(config, "W_gripper_pose_term")
        assert hasattr(config, "W_limit")
        assert hasattr(config, "safety_threshold")
        assert hasattr(config, "dt")
        assert hasattr(config, "gripper_depth")
        assert hasattr(config, "T")


class TestConfigLoader:
    """Tests for ConfigLoader class"""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config YAML file"""
        config_data = {
            "weights": {
                "W_xREG": 0.001,
                "W_uREG": 0.001,
                "W_gripper_pose": 1000.0,
                "W_gripper_pose_term": 100.0,
                "W_limit": 100.0,
            },
            "safety_threshold": 0.02,
            "dt": 0.01,
            "gripper_depth": 0.1034,
            "T": 15,
        }

        config_path = tmp_path / "test_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        return config_path

    def test_config_loader_load(self, temp_config_file):
        """Test ConfigLoader.load() method"""
        from object_following_ocp.data_loader import ConfigLoader

        config = ConfigLoader.load(str(temp_config_file))

        assert config.W_xREG == pytest.approx(0.001)
        assert config.W_uREG == pytest.approx(0.001)
        assert config.W_gripper_pose == pytest.approx(1000.0)
        assert config.W_gripper_pose_term == pytest.approx(100.0)
        assert config.W_limit == pytest.approx(100.0)
        assert config.safety_threshold == pytest.approx(0.02)
        assert config.dt == pytest.approx(0.01)
        assert config.gripper_depth == pytest.approx(0.1034)
        assert config.T == pytest.approx(15)

    def test_config_loader_returns_robot_config(self, temp_config_file):
        """Test that ConfigLoader returns a RobotConfig instance"""
        from object_following_ocp.data_loader import ConfigLoader, RobotConfig

        config = ConfigLoader.load(str(temp_config_file))

        assert isinstance(config, RobotConfig)

    def test_config_loader_with_different_values(self, tmp_path):
        """Test ConfigLoader with different config values"""
        from object_following_ocp.data_loader import ConfigLoader

        config_data = {
            "weights": {
                "W_xREG": 0.01,
                "W_uREG": 0.02,
                "W_gripper_pose": 500.0,
                "W_gripper_pose_term": 50.0,
                "W_limit": 200.0,
            },
            "safety_threshold": 0.05,
            "dt": 0.02,
            "gripper_depth": 0.15,
            "T": 10,
        }

        config_path = tmp_path / "custom_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = ConfigLoader.load(str(config_path))

        assert config.W_xREG == pytest.approx(0.01)
        assert config.W_uREG == pytest.approx(0.02)
        assert config.W_gripper_pose == pytest.approx(500.0)
        assert config.W_gripper_pose_term == pytest.approx(50.0)
        assert config.W_limit == pytest.approx(200.0)
        assert config.safety_threshold == pytest.approx(0.05)
        assert config.dt == pytest.approx(0.02)
        assert config.gripper_depth == pytest.approx(0.15)

    def test_config_loader_missing_file(self):
        """Test ConfigLoader with non-existent file"""
        from object_following_ocp.data_loader import ConfigLoader

        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("/nonexistent/path/config.yml")

    def test_config_loader_invalid_yaml(self, tmp_path):
        """Test ConfigLoader with invalid YAML"""
        from object_following_ocp.data_loader import ConfigLoader

        config_path = tmp_path / "invalid.yml"
        with open(config_path, "w") as f:
            f.write("{ invalid yaml content [")

        with pytest.raises(yaml.YAMLError):
            ConfigLoader.load(str(config_path))

    def test_config_loader_missing_weights_section(self, tmp_path):
        """Test ConfigLoader with missing weights section"""
        from object_following_ocp.data_loader import ConfigLoader

        config_data = {"safety_threshold": 0.02, "dt": 0.01, "gripper_depth": 0.1034}

        config_path = tmp_path / "no_weights.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(KeyError):
            ConfigLoader.load(str(config_path))

    @pytest.mark.skipif(
        not pathlib.Path("/workspaces/object_following_ocp").exists(),
        reason="Real config file not available in this environment",
    )
    def test_config_loader_real_file(self):
        """Test ConfigLoader with the real config file"""
        from object_following_ocp.data_loader import ConfigLoader

        config_path = pathlib.Path(
            "/workspaces/object_following_ocp/example/robot_motion/configs/ik_config.yml"
        )

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = ConfigLoader.load(str(config_path))

        # Test with expected values from the real file
        assert config.W_xREG == pytest.approx(0.001)
        assert config.W_uREG == pytest.approx(0.001)
        assert config.W_gripper_pose == pytest.approx(1000.0)
        assert config.W_gripper_pose_term == pytest.approx(100.0)
        assert config.W_limit == pytest.approx(100.0)
        assert config.safety_threshold == pytest.approx(0.02)
        assert config.dt == pytest.approx(0.01)
        assert config.gripper_depth == pytest.approx(0.1034)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
