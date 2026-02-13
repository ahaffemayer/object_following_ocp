"""
Unit tests for RobotConfig and ConfigLoader classes.

To run: pytest test_config_loader.py -v
"""
import pathlib

import pytest
import yaml

from object_following_ocp.data_loader import ConfigLoader, RobotConfig


class TestRobotConfig:
    """Tests for RobotConfig dataclass"""

    def test_robot_config_creation(self):
        """Test creating a RobotConfig object"""
        config = RobotConfig(
            W_xREG=0.001,
            W_uREG=0.001,
            W_gripper_pose=1000.0,
            W_gripper_pose_term=100.0,
            W_limit=100.0,
            safety_threshold=0.02,
            dt=0.01,
            gripper_depth=0.1034
        )

        assert config.W_xREG == pytest.approx(0.001)
        assert config.W_uREG == pytest.approx(0.001)
        assert config.W_gripper_pose == pytest.approx(1000.0)
        assert config.W_gripper_pose_term == pytest.approx(100.0)
        assert config.W_limit == pytest.approx(100.0)
        assert config.safety_threshold == pytest.approx(0.02)
        assert config.dt == pytest.approx(0.01)
        assert config.gripper_depth == pytest.approx(0.1034)

    def test_robot_config_attributes(self):
        """Test that RobotConfig has all expected attributes"""
        config = RobotConfig(
            W_xREG=1.0,
            W_uREG=2.0,
            W_gripper_pose=3.0,
            W_gripper_pose_term=4.0,
            W_limit=5.0,
            safety_threshold=6.0,
            dt=7.0,
            gripper_depth=8.0
        )

        # Check all attributes exist
        assert hasattr(config, 'W_xREG')
        assert hasattr(config, 'W_uREG')
        assert hasattr(config, 'W_gripper_pose')
        assert hasattr(config, 'W_gripper_pose_term')
        assert hasattr(config, 'W_limit')
        assert hasattr(config, 'safety_threshold')
        assert hasattr(config, 'dt')
        assert hasattr(config, 'gripper_depth')


class TestConfigLoader:
    """Tests for ConfigLoader class"""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file"""
        config_data = {
            "weights": {
                "W_xREG": 0.001,
                "W_uREG": 0.001,
                "W_gripper_pose": 1000.0,
                "W_gripper_pose_term": 100.0,
                "W_limit": 100.0
            },
            "safety_threshold": 0.02,
            "dt": 0.01,
            "gripper_depth": 0.1034
        }

        config_path = tmp_path / "test_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        return config_path

    def test_config_loader_load(self, temp_config_file):
        """Test that ConfigLoader.load() correctly loads a YAML file"""
        config = ConfigLoader.load(str(temp_config_file))

        assert isinstance(config, RobotConfig)
        assert config.W_xREG == pytest.approx(0.001)
        assert config.W_uREG == pytest.approx(0.001)
        assert config.W_gripper_pose == pytest.approx(1000.0)
        assert config.W_gripper_pose_term == pytest.approx(100.0)
        assert config.W_limit == pytest.approx(100.0)
        assert config.safety_threshold == pytest.approx(0.02)
        assert config.dt == pytest.approx(0.01)
        assert config.gripper_depth == pytest.approx(0.1034)

    def test_config_loader_with_different_values(self, tmp_path):
        """Test ConfigLoader with different weight values"""
        config_data = {
            "weights": {
                "W_xREG": 0.5,
                "W_uREG": 0.75,
                "W_gripper_pose": 500.0,
                "W_gripper_pose_term": 50.0,
                "W_limit": 25.0
            },
            "safety_threshold": 0.05,
            "dt": 0.02,
            "gripper_depth": 0.2
        }

        config_path = tmp_path / "custom_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = ConfigLoader.load(str(config_path))

        assert config.W_xREG == pytest.approx(0.5)
        assert config.W_uREG == pytest.approx(0.75)
        assert config.W_gripper_pose == pytest.approx(500.0)
        assert config.W_gripper_pose_term == pytest.approx(50.0)
        assert config.W_limit == pytest.approx(25.0)
        assert config.safety_threshold == pytest.approx(0.05)
        assert config.dt == pytest.approx(0.02)
        assert config.gripper_depth == pytest.approx(0.2)

    def test_config_loader_missing_weights(self, tmp_path):
        """Test that ConfigLoader raises error when weights section is missing"""
        config_data = {
            "safety_threshold": 0.02,
            "dt": 0.01,
            "gripper_depth": 0.1034
        }

        config_path = tmp_path / "bad_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(KeyError):
            ConfigLoader.load(str(config_path))

    def test_config_loader_missing_parameter(self, tmp_path):
        """Test that ConfigLoader raises error when a parameter is missing"""
        config_data = {
            "weights": {
                "W_xREG": 0.001,
                "W_uREG": 0.001,
                # Missing W_gripper_pose
                "W_gripper_pose_term": 100.0,
                "W_limit": 100.0
            },
            "safety_threshold": 0.02,
            "dt": 0.01,
            "gripper_depth": 0.1034
        }

        config_path = tmp_path / "incomplete_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(KeyError):
            ConfigLoader.load(str(config_path))

    def test_config_loader_file_not_found(self):
        """Test that ConfigLoader raises error for non-existent file"""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("nonexistent_file.yml")

    def test_config_loader_invalid_yaml(self, tmp_path):
        """Test that ConfigLoader raises error for invalid YAML"""
        config_path = tmp_path / "invalid.yml"
        with open(config_path, "w") as f:
            f.write("{ invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            ConfigLoader.load(str(config_path))

    def test_config_loader_with_extra_fields(self, tmp_path):
        """Test that ConfigLoader ignores extra fields in YAML"""
        config_data = {
            "weights": {
                "W_xREG": 0.001,
                "W_uREG": 0.001,
                "W_gripper_pose": 1000.0,
                "W_gripper_pose_term": 100.0,
                "W_limit": 100.0,
                "extra_weight": 999.0  # Extra field that should be ignored
            },
            "safety_threshold": 0.02,
            "dt": 0.01,
            "gripper_depth": 0.1034,
            "extra_param": "ignored"  # Extra field that should be ignored
        }

        config_path = tmp_path / "extra_fields_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Should load without error, ignoring extra fields
        config = ConfigLoader.load(str(config_path))

        assert config.W_xREG == pytest.approx(0.001)
        assert config.W_uREG == pytest.approx(0.001)
        assert not hasattr(config, 'extra_weight')
        assert not hasattr(config, 'extra_param')


class TestConfigLoaderWithRealFile:
    """Tests using the actual config file from your workspace"""

    @pytest.fixture
    def real_config_path(self):
        """Real config file path from your workspace"""
        return pathlib.Path(
            "/workspaces/object_following_ocp/example/robot_motion/configs/ik_config.yml"
        )

    @pytest.mark.skipif(
        not pathlib.Path("/workspaces/object_following_ocp").exists(),
        reason="Real files not available in this environment"
    )
    def test_real_config_load(self, real_config_path):
        """Test loading the actual config file"""
        if not real_config_path.exists():
            pytest.skip(f"Config file not found: {real_config_path}")

        config = ConfigLoader.load(str(real_config_path))

        # Verify it's a RobotConfig instance
        assert isinstance(config, RobotConfig)

        # Check expected values from the real file
        assert config.W_xREG == pytest.approx(0.001)
        assert config.W_uREG == pytest.approx(0.001)
        assert config.W_gripper_pose == pytest.approx(1000.0)
        assert config.W_gripper_pose_term == pytest.approx(100.0)
        assert config.W_limit == pytest.approx(100.0)
        assert config.safety_threshold == pytest.approx(0.02)
        assert config.dt == pytest.approx(0.01)
        assert config.gripper_depth == pytest.approx(0.1034)

    @pytest.mark.skipif(
        not pathlib.Path("/workspaces/object_following_ocp").exists(),
        reason="Real files not available in this environment"
    )
    def test_real_config_weight_types(self, real_config_path):
        """Test that all weights are numeric types"""
        if not real_config_path.exists():
            pytest.skip(f"Config file not found: {real_config_path}")

        config = ConfigLoader.load(str(real_config_path))

        # Check all weights are numbers
        assert isinstance(config.W_xREG, (int, float))
        assert isinstance(config.W_uREG, (int, float))
        assert isinstance(config.W_gripper_pose, (int, float))
        assert isinstance(config.W_gripper_pose_term, (int, float))
        assert isinstance(config.W_limit, (int, float))
        assert isinstance(config.safety_threshold, (int, float))
        assert isinstance(config.dt, (int, float))
        assert isinstance(config.gripper_depth, (int, float))

    @pytest.mark.skipif(
        not pathlib.Path("/workspaces/object_following_ocp").exists(),
        reason="Real files not available in this environment"
    )
    def test_real_config_positive_values(self, real_config_path):
        """Test that all config values are positive"""
        if not real_config_path.exists():
            pytest.skip(f"Config file not found: {real_config_path}")

        config = ConfigLoader.load(str(real_config_path))

        # All values should be positive
        assert config.W_xREG > 0
        assert config.W_uREG > 0
        assert config.W_gripper_pose > 0
        assert config.W_gripper_pose_term > 0
        assert config.W_limit > 0
        assert config.safety_threshold > 0
        assert config.dt > 0
        assert config.gripper_depth > 0


class TestConfigLoaderIntegration:
    """Integration tests for typical usage patterns"""

    def test_load_and_use_config(self, tmp_path):
        """Test typical workflow: load config and use values"""
        # Create a realistic config file
        config_data = {
            "weights": {
                "W_xREG": 0.001,
                "W_uREG": 0.001,
                "W_gripper_pose": 1000.0,
                "W_gripper_pose_term": 100.0,
                "W_limit": 100.0
            },
            "safety_threshold": 0.02,
            "dt": 0.01,
            "gripper_depth": 0.1034
        }

        config_path = tmp_path / "robot_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Load config
        config = ConfigLoader.load(str(config_path))

        # Use config values (simulating real usage)
        # For example, computing total weight
        total_weight = (
            config.W_xREG +
            config.W_uREG +
            config.W_gripper_pose +
            config.W_gripper_pose_term +
            config.W_limit
        )

        expected_total = 0.001 + 0.001 + 1000.0 + 100.0 + 100.0
        assert total_weight == pytest.approx(expected_total)

        # Check time step is reasonable
        assert 0.001 <= config.dt <= 0.1

        # Check gripper depth is reasonable (in meters, should be 10cm)
        assert 0.05 <= config.gripper_depth <= 0.2

    def test_multiple_configs(self, tmp_path):
        """Test loading multiple different config files"""
        # Create two different configs
        config1_data = {
            "weights": {
                "W_xREG": 0.1,
                "W_uREG": 0.1,
                "W_gripper_pose": 100.0,
                "W_gripper_pose_term": 10.0,
                "W_limit": 10.0
            },
            "safety_threshold": 0.01,
            "dt": 0.02,
            "gripper_depth": 0.1
        }

        config2_data = {
            "weights": {
                "W_xREG": 0.5,
                "W_uREG": 0.5,
                "W_gripper_pose": 500.0,
                "W_gripper_pose_term": 50.0,
                "W_limit": 50.0
            },
            "safety_threshold": 0.03,
            "dt": 0.005,
            "gripper_depth": 0.15
        }

        path1 = tmp_path / "config1.yml"
        path2 = tmp_path / "config2.yml"

        with open(path1, "w") as f:
            yaml.dump(config1_data, f)
        with open(path2, "w") as f:
            yaml.dump(config2_data, f)

        # Load both configs
        config1 = ConfigLoader.load(str(path1))
        config2 = ConfigLoader.load(str(path2))

        # Verify they're different
        assert config1.W_xREG != config2.W_xREG
        assert config1.W_gripper_pose != config2.W_gripper_pose
        assert config1.dt != config2.dt
        assert config1.gripper_depth != config2.gripper_depth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
