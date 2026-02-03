#!/usr/bin/env python3
"""
Script to consolidate individual YML grasp files into a single JSON database.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidate YML grasp files into a single JSON database"
    )
    parser.add_argument(
        "--yml_folder",
        type=str,
        required=True,
        help="Folder containing YML grasp files",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="grasps_database.json",
        help="Output JSON file for grasp database",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="",
        help="Optional processing summary JSON from step 1",
    )

    return parser.parse_args()


def load_yml_grasps(yml_file):
    """
    Load grasps from a YML file in Isaac Grasp format.

    Returns:
        list: List of grasp dictionaries with pose and score
    """
    try:
        with open(yml_file, 'r') as f:
            data = yaml.safe_load(f)

        grasps = []

        # Isaac grasp format typically has grasps as a list
        if isinstance(data, list):
            for grasp_data in data:
                # Extract pose (4x4 matrix) and score
                if 'pose' in grasp_data and 'score' in grasp_data:
                    grasps.append({
                        'pose': grasp_data['pose'],
                        'score': float(grasp_data['score'])
                    })
        elif isinstance(data, dict) and 'grasps' in data:
            # Alternative format
            for grasp_data in data['grasps']:
                if 'pose' in grasp_data and 'score' in grasp_data:
                    grasps.append({
                        'pose': grasp_data['pose'],
                        'score': float(grasp_data['score'])
                    })

        return grasps

    except Exception as e:
        print(f"Error loading {yml_file}: {e}")
        return []


def consolidate_grasps(yml_folder, summary_file=None):
    """
    Consolidate all YML files into a single grasp database.

    Returns:
        dict: {object_id: {"mesh": ..., "scale": ..., "grasps": [...]}}
    """
    yml_folder = Path(yml_folder)
    grasp_database = {}

    # Load summary if available
    object_metadata = {}
    if summary_file and Path(summary_file).exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            for obj_id, data in summary.items():
                if data.get("status") == "success":
                    object_metadata[obj_id] = {
                        "mesh": data.get("mesh"),
                        "scale": data.get("scale")
                    }

    # Process all YML files
    yml_files = list(yml_folder.glob("*.yml"))
    print(f"Found {len(yml_files)} YML files to process")

    for yml_file in yml_files:
        # Extract object_id from filename (format: {obj_id}_{mesh_name}.yml)
        filename = yml_file.stem
        parts = filename.split('_', 1)

        if len(parts) >= 1:
            obj_id = parts[0]

            print(f"\nProcessing {yml_file.name}...")
            grasps = load_yml_grasps(yml_file)

            if grasps:
                # Get metadata from summary or extract from filename
                if obj_id in object_metadata:
                    mesh_name = object_metadata[obj_id]["mesh"]
                    scale = object_metadata[obj_id]["scale"]
                else:
                    # Try to extract from filename
                    mesh_name = parts[1] if len(parts) > 1 else "unknown"
                    scale = None

                grasp_database[obj_id] = {
                    "mesh": mesh_name,
                    "scale": scale,
                    "num_grasps": len(grasps),
                    "grasps": grasps
                }

                print(f"  ✓ Added {len(grasps)} grasps for object {obj_id}")
                if grasps:
                    scores = [g['score'] for g in grasps]
                    print(
                        f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
            else:
                print(f"  ✗ No grasps found in {yml_file.name}")

    return grasp_database


def main():
    args = parse_args()

    print("=" * 80)
    print("Consolidating YML grasp files into JSON database")
    print("=" * 80)

    # Consolidate all grasps
    grasp_database = consolidate_grasps(
        yml_folder=args.yml_folder,
        summary_file=args.summary_file if args.summary_file else None
    )

    # Save to JSON
    output_path = Path(args.output_json)
    print(f"\n{'=' * 80}")
    print(f"Saving grasp database to: {output_path}")
    print(f"{'=' * 80}")

    with open(output_path, 'w') as f:
        json.dump(grasp_database, f, indent=2)

    # Print summary
    print(f"\nGrasp Database Summary:")
    print(f"  Total objects: {len(grasp_database)}")
    total_grasps = sum(obj_data['num_grasps']
                       for obj_data in grasp_database.values())
    print(f"  Total grasps: {total_grasps}")

    if grasp_database:
        print(f"\n  Objects in database:")
        for obj_id, obj_data in grasp_database.items():
            print(
                f"    {obj_id}: {obj_data['num_grasps']} grasps (mesh: {obj_data['mesh']}, scale: {obj_data['scale']})")

    print(f"\n✓ Done! Grasp database saved to {output_path}")


if __name__ == "__main__":
    main()
