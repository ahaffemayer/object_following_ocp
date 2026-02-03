#!/usr/bin/env python3
"""
Script to process JSON files, extract unique objects, and generate grasps for each.
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract objects from JSON files and generate grasps"
    )
    parser.add_argument(
        "--json_folder",
        type=str,
        required=True,
        help="Path to folder containing JSON files with object poses",
    )
    parser.add_argument(
        "--mesh_folder",
        type=str,
        required=True,
        help="Path to folder containing mesh files (default: ../meshes relative to json_folder)",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        required=True,
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./grasp_outputs",
        help="Folder to save output YML files",
    )
    parser.add_argument(
        "--demo_script",
        type=str,
        default="scripts/demo_object_mesh.py",
        help="Path to demo_object_mesh.py script",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=400,
        help="Number of grasps to generate per object",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=-1,
        help="Threshold for valid grasps",
    )

    return parser.parse_args()


def extract_unique_objects(json_folder):
    """
    Extract unique objects (mesh + scale combinations) from all JSON files.

    Returns:
        dict: {object_id: {"mesh": mesh_name, "scale": scale_value}}
    """
    unique_objects = {}
    json_files = list(Path(json_folder).glob("*.json"))

    print(f"Found {len(json_files)} JSON files in {json_folder}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract objects from the JSON
            if "objects" in data:
                for obj_id, obj_data in data["objects"].items():
                    mesh_name = obj_data.get("mesh")
                    scale = obj_data.get("scale")

                    if mesh_name and scale:
                        # Use object_id from the JSON as key
                        if obj_id not in unique_objects:
                            unique_objects[obj_id] = {
                                "mesh": mesh_name,
                                "scale": scale
                            }
                            print(
                                f"  Found object {obj_id}: mesh={mesh_name}, scale={scale:.6f}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"\nTotal unique objects found: {len(unique_objects)}")
    return unique_objects


def run_grasp_generation(mesh_path, scale, gripper_config, output_file, demo_script, num_grasps, grasp_threshold):
    """
    Run the grasp generation demo script for a single object.

    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        "python", demo_script,
        "--mesh_file", str(mesh_path),
        "--mesh_scale", str(scale),
        "--gripper_config", gripper_config,
        "--output_file", str(output_file),
        "--num_grasps", str(num_grasps),
        "--grasp_threshold", str(grasp_threshold),
        "--no-visualization"  # Disable visualization for batch processing
    ]

    try:
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def main():
    args = parse_args()

    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Extract unique objects
    print("=" * 80)
    print("STEP 1: Extracting unique objects from JSON files")
    print("=" * 80)
    unique_objects = extract_unique_objects(args.json_folder)

    if not unique_objects:
        print("No objects found! Exiting.")
        return

    # Process each unique object
    print("\n" + "=" * 80)
    print("STEP 2: Generating grasps for each object")
    print("=" * 80)

    mesh_folder = Path(args.mesh_folder)
    results = {}

    for obj_id, obj_data in unique_objects.items():
        mesh_name = obj_data["mesh"]
        scale = obj_data["scale"]

        # Construct mesh path: ../meshes/mesh_name/mesh_name.obj
        mesh_path = mesh_folder / mesh_name / f"{mesh_name}.obj"

        print(f"\nProcessing object {obj_id}:")
        print(f"  Mesh: {mesh_name}")
        print(f"  Scale: {scale}")
        print(f"  Path: {mesh_path}")

        if not mesh_path.exists():
            print(f"  ✗ Mesh file not found: {mesh_path}")
            results[obj_id] = {"status": "failed", "reason": "mesh_not_found"}
            continue

        # Output YML file for this object
        output_yml = output_folder / f"{obj_id}_{mesh_name}.yml"

        # Run grasp generation
        success = run_grasp_generation(
            mesh_path=mesh_path,
            scale=scale,
            gripper_config=args.gripper_config,
            output_file=output_yml,
            demo_script=args.demo_script,
            num_grasps=args.num_grasps,
            grasp_threshold=args.grasp_threshold
        )

        if success:
            results[obj_id] = {
                "status": "success",
                "output_file": str(output_yml),
                "mesh": mesh_name,
                "scale": scale
            }
        else:
            results[obj_id] = {
                "status": "failed",
                "reason": "grasp_generation_failed"
            }

    # Save processing summary
    summary_file = output_folder / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = len(results) - successful
    print(f"Total objects: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nProcessing summary saved to: {summary_file}")
    print(f"YML files saved to: {output_folder}")


if __name__ == "__main__":
    main()
