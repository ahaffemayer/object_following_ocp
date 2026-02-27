"""
Helper utilities to load and reuse saved trajectories.
"""

import pathlib
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin

from object_following_ocp.data.data_loader import ConfigLoader, DataLoader
from object_following_ocp.geom.trajectories import (
    TrajectoryInConfigurationSpace,
    TrajectorySE3,
)


class TrajectoryLoader:
    """
    Load saved trajectories and reconstruct all necessary objects.

    This allows you to:
    1. Load grid search or kept trajectories
    2. Reconstruct DataLoader, config, and robot from saved paths
    3. Access trajectories and metadata
    4. Replay or further process trajectories
    """

    def __init__(self, filepath: pathlib.Path):
        """
        Initialize loader with saved trajectory file.

        Args:
            filepath: Path to .pkl file (grid search or kept trajectories)
        """
        self.filepath = filepath
        with open(filepath, "rb") as f:
            self.data = pickle.load(f)

        self.mesh_id = self.data["mesh_id"]
        self.num_trajectories = self._get_num_trajectories()

    def _get_num_trajectories(self) -> int:
        """Get number of trajectories in file."""
        if "num_kept" in self.data:
            return self.data["num_kept"]
        else:
            return self.data["summary"]["num_valid"]

    def get_metadata(self) -> Dict:
        """
        Get metadata about the trajectories.

        Returns:
            Dictionary with mesh_id, paths, grasp info, etc.
        """
        return {
            "mesh_id": self.data["mesh_id"],
            "object_traj_path": self.data["object_traj_path"],
            "scale_path": self.data["scale_path"],
            "config_path": self.data["config_path"],
            "mesh_path": self.data["mesh_path"],
            "texture_path": self.data["texture_path"],
            "object_scale": self.data["object_scale"],
            "best_grasp_name": self.data["best_grasp_name"],
            "grasp_correction_angle_deg": self.data["grasp_correction_angle_deg"],
            "elevation_angle_deg": self.data["elevation_angle_deg"],
            "num_trajectories": self.num_trajectories,
        }

    def print_metadata(self):
        """Print metadata in human-readable format."""
        metadata = self.get_metadata()
        print("=" * 70)
        print(f"TRAJECTORY FILE: {self.filepath.name}")
        print("=" * 70)
        print(f"\nMesh ID: {metadata['mesh_id']}")
        print(f"Number of trajectories: {metadata['num_trajectories']}")
        print("\nPaths:")
        print(f"  Object trajectory: {metadata['object_traj_path']}")
        print(f"  Scales: {metadata['scale_path']}")
        print(f"  Config: {metadata['config_path']}")
        print(f"  Mesh: {metadata['mesh_path']}")
        if metadata["texture_path"]:
            print(f"  Texture: {metadata['texture_path']}")
        print(f"\nObject scale: {metadata['object_scale']}")
        if metadata["best_grasp_name"]:
            print(f"Best grasp: {metadata['best_grasp_name']}")
        print(f"Grasp correction: {metadata['grasp_correction_angle_deg']}°")
        print(f"Elevation: {metadata['elevation_angle_deg']}°")
        print("=" * 70)

    def load_dataloader_and_config(self) -> Tuple[DataLoader, any]:
        """
        Reconstruct DataLoader and RobotConfig from saved paths.

        Returns:
            Tuple of (DataLoader, RobotConfig)
        """
        dataloader = DataLoader(
            object_trajectory_path=pathlib.Path(self.data["object_traj_path"]),
            scales_path=pathlib.Path(self.data["scale_path"]),
            load_grasps=True,
        )

        robot_config = ConfigLoader.load(pathlib.Path(self.data["config_path"]))

        return dataloader, robot_config

    def get_trajectory(self, index: int) -> Dict:
        """
        Get a specific trajectory by index.

        Args:
            index: Trajectory index (0-based)

        Returns:
            Dictionary with trajectory data
        """
        if index < 0 or index >= self.num_trajectories:
            raise IndexError(
                f"Trajectory index {index} out of range [0, {self.num_trajectories})"
            )

        traj = self.data["trajectories"][index]

        # Reconstruct SE3 trajectories
        object_traj = TrajectorySE3(
            [pin.SE3(np.array(pose)) for pose in traj["object_trajectory_poses"]]
        )

        # Reconstruct joint trajectory
        # Grid search files use 'joint_configurations', kept files use 'joint_trajectory_ik'
        joint_key = (
            "joint_trajectory_ik"
            if "joint_trajectory_ik" in traj
            else "joint_configurations"
        )
        joint_configs = [np.array(q) for q in traj[joint_key]]
        joint_traj_ik = TrajectoryInConfigurationSpace(joint_configs)

        result = {
            "camera_translation": np.array(traj["camera_translation"]),
            "object_trajectory": object_traj,
            "joint_trajectory_ik": joint_traj_ik,
            "ik_success_rate": traj["ik_success_rate"],
        }

        # Add OCP data if this is a kept trajectory
        if "joint_trajectory_ocp" in traj:
            result["joint_trajectory_ocp"] = [
                np.array(xs) for xs in traj["joint_trajectory_ocp"]
            ]
            result["tcp_trajectory_poses"] = [
                np.array(pose) for pose in traj["tcp_trajectory_poses"]
            ]
            result["ocp_iterations"] = traj["ocp_iterations"]
            result["user_notes"] = traj.get("user_notes", "")

        # Add EE trajectory if available (grid search results)
        if "ee_trajectory_poses" in traj:
            ee_traj = TrajectorySE3(
                [pin.SE3(np.array(pose)) for pose in traj["ee_trajectory_poses"]]
            )
            result["ee_trajectory"] = ee_traj

        return result

    def get_all_trajectories(self) -> List[Dict]:
        """
        Get all trajectories.

        Returns:
            List of trajectory dictionaries
        """
        return [self.get_trajectory(i) for i in range(self.num_trajectories)]

    def get_camera_positions(self) -> np.ndarray:
        """
        Get all camera positions.

        Returns:
            Nx3 array of camera positions
        """
        positions = [
            np.array(traj["camera_translation"]) for traj in self.data["trajectories"]
        ]
        return np.array(positions)

    def get_best_trajectory(
        self, criterion: str = "ik_success_rate"
    ) -> Tuple[int, Dict]:
        """
        Get best trajectory based on criterion.

        Args:
            criterion: 'ik_success_rate' or 'ocp_iterations' (for kept trajectories)

        Returns:
            Tuple of (index, trajectory_dict)
        """
        if criterion == "ik_success_rate":
            rates = [traj["ik_success_rate"] for traj in self.data["trajectories"]]
            best_idx = np.argmax(rates)
        elif criterion == "ocp_iterations":
            if "ocp_iterations" not in self.data["trajectories"][0]:
                raise ValueError(
                    "OCP iterations not available (not a kept trajectories file)"
                )
            iters = [traj["ocp_iterations"] for traj in self.data["trajectories"]]
            best_idx = np.argmin(iters)  # Fewer iterations is better
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return best_idx, self.get_trajectory(best_idx)


def load_trajectory_file(filepath: pathlib.Path) -> TrajectoryLoader:
    """
    Convenience function to load trajectory file.

    Args:
        filepath: Path to .pkl file

    Returns:
        TrajectoryLoader instance
    """
    loader = TrajectoryLoader(filepath)
    loader.print_metadata()
    return loader


def find_trajectory_files(
    directory: pathlib.Path, mesh_id: Optional[str] = None
) -> Dict[str, List[pathlib.Path]]:
    """
    Find all trajectory files in directory.

    Args:
        directory: Directory to search
        mesh_id: Optional mesh_id to filter by

    Returns:
        Dictionary with 'grid_search' and 'kept' lists of paths
    """
    grid_search_files = []
    kept_files = []

    for file in directory.glob("*.pkl"):
        if mesh_id and not file.name.startswith(mesh_id):
            continue

        if "grid_search_results" in file.name:
            grid_search_files.append(file)
        elif "kept_trajectories" in file.name:
            kept_files.append(file)

    return {
        "grid_search": sorted(grid_search_files),
        "kept": sorted(kept_files),
    }


# Example usage
if __name__ == "__main__":
    # Example: Load and display trajectories
    output_dir = pathlib.Path("/mnt/user-data/outputs")

    # Find all trajectory files
    files = find_trajectory_files(output_dir)

    print("\n" + "=" * 70)
    print("AVAILABLE TRAJECTORY FILES")
    print("=" * 70)

    print(f"\nGrid Search Results ({len(files['grid_search'])} files):")
    for f in files["grid_search"]:
        print(f"  - {f.name}")

    print(f"\nKept Trajectories ({len(files['kept'])} files):")
    for f in files["kept"]:
        print(f"  - {f.name}")

    # Example: Load first kept trajectory file if available
    if files["kept"]:
        print("\n" + "=" * 70)
        print("LOADING FIRST KEPT TRAJECTORY FILE")
        print("=" * 70)

        loader = load_trajectory_file(files["kept"][0])

        # Get best trajectory
        best_idx, best_traj = loader.get_best_trajectory("ik_success_rate")
        print(f"\nBest trajectory (by IK success rate): #{best_idx + 1}")
        print(f"  Camera position: {best_traj['camera_translation']}")
        print(f"  IK success rate: {best_traj['ik_success_rate']:.1f}%")
        print(f"  OCP iterations: {best_traj['ocp_iterations']}")

        # Load DataLoader and config
        print("\nReconstructing DataLoader and config...")
        dataloader, robot_config = loader.load_dataloader_and_config()
        print(f"  Loaded object: {dataloader.object_info.mesh_id}")
        print(f"  Gripper depth: {robot_config.gripper_depth} m")
