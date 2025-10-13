#!/usr/bin/env python3
"""
Script to fix prediction file format and create annotated video
Usage: python3 create_visualization.py [prediction_file_path]
"""

import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.pose_video_visualizer import PoseVideoVisualizer


def fix_prediction_file(input_file: str, output_file: str):
    """Fix prediction file by adding missing required fields"""

    print(f"📁 Loading: {input_file}")

    # Load the original file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Check if file already has required fields
    required_fields = [
        "total_persons_detected",
        "avg_persons_per_frame",
        "avg_inference_time",
    ]
    missing_fields = [field for field in required_fields if field not in data]

    if not missing_fields:
        print("✅ File already has all required fields")
        return input_file

    print(f"🔧 Adding missing fields: {missing_fields}")

    # Calculate missing statistics
    frames = data["frames"]
    total_frames = len(frames)
    total_persons = sum(len(frame["persons"]) for frame in frames)
    total_inference_time = sum(frame["inference_time"] for frame in frames)

    # Add missing fields
    data["total_persons_detected"] = total_persons
    data["avg_persons_per_frame"] = (
        total_persons / total_frames if total_frames > 0 else 0.0
    )
    data["avg_inference_time"] = (
        total_inference_time / total_frames if total_frames > 0 else 0.0
    )

    # Save the fixed file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Fixed file saved to: {output_file}")
    print(f"📊 Statistics:")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Total persons detected: {total_persons}")
    print(f"   - Avg persons per frame: {data['avg_persons_per_frame']:.2f}")
    print(f"   - Avg inference time: {data['avg_inference_time']:.4f}s")
    print(f"   - Maneuver type: {data.get('maneuver_type', 'Unknown')}")
    print(f"   - Execution score: {data.get('execution_score', 'Unknown')}")
    if data.get("is_augmented", False):
        print(f"   - Augmentation: {data.get('augmentation_type', 'Unknown')} (Yes)")
    else:
        print(f"   - Augmentation: None")

    return output_file


def create_visualization(
    prediction_file: str,
    output_video: str,
    kpt_thr: float = 0.3,
    bbox_thr: float = 0.3,
    max_persons: int = 3,
):
    """Create annotated video from prediction file"""

    # Check if prediction file exists
    if not os.path.exists(prediction_file):
        print(f"❌ Error: Prediction file not found: {prediction_file}")
        return False

    print(f"🎥 Output video: {output_video}")
    print(
        f"🎛️  Settings: kpt_thr={kpt_thr}, bbox_thr={bbox_thr}, max_persons={max_persons}"
    )

    # Initialize the visualizer
    print("🔧 Initializing pose video visualizer...")
    visualizer = PoseVideoVisualizer()

    # Create visualization
    print("🎬 Creating annotated video...")
    success = visualizer.create_visualization_from_prediction_file(
        prediction_file_path=prediction_file,
        output_path=output_video,
        kpt_thr=kpt_thr,
        bbox_thr=bbox_thr,
        max_persons=max_persons,
    )

    if success:
        print("✅ Video created successfully!")
        print(f"📺 Output saved to: {output_video}")

        # Check file size
        if os.path.exists(output_video):
            file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
            print(f"📊 File size: {file_size:.2f} MB")

        return True
    else:
        print("❌ Failed to create video")
        return False


def main():
    # Check if file path provided as argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default to a common pattern - you can change this
        print("Usage: python3 create_visualization.py [prediction_file_path]")
        print("Or edit the script to set a default file path")

        # Example default - change this to your file
        input_file = input("Enter path to prediction file: ").strip()
        if not input_file:
            print("❌ No file path provided")
            return False

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        return False

    # Generate output file names
    input_path = Path(input_file)
    fixed_file = str(input_path.parent / f"{input_path.stem}_FIXED.json")
    output_video = str(input_path.parent / f"{input_path.stem}_visualization.mp4")

    print("🚀 Starting visualization process...")
    print("=" * 70)

    # Step 1: Fix the prediction file format
    prediction_file = fix_prediction_file(input_file, fixed_file)

    print("=" * 70)

    # Step 2: Create the visualization
    success = create_visualization(
        prediction_file=prediction_file,
        output_video=output_video,
        kpt_thr=0.3,  # Adjust these as needed
        bbox_thr=0.3,
        max_persons=3,
    )

    if success:
        print("=" * 70)
        print("🎉 Process completed successfully!")
        print(f"📁 Fixed prediction file: {fixed_file}")
        print(f"🎬 Annotated video: {output_video}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)








