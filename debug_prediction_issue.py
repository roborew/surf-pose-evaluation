#!/usr/bin/env python3
"""
Debug script to test prediction file generation and loading
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

from utils.prediction_file_format import PredictionFileHandler
from data_handling.data_loader import SurfingDataLoader


def test_maneuver_id_generation():
    """Test how maneuver IDs are generated"""
    print("🔍 Testing maneuver ID generation...")

    # Load configuration
    config_path = "configs/evaluation_config_macos.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize data loader
    data_loader = SurfingDataLoader(config)

    try:
        # Load a small number of maneuvers
        print("Loading maneuvers...")
        maneuvers = data_loader.load_maneuvers(max_clips=2, maneuvers_per_clip=1)

        if maneuvers:
            print(f"✅ Loaded {len(maneuvers)} maneuvers")
            for i, maneuver in enumerate(maneuvers[:3]):
                print(f"  Maneuver {i+1}:")
                print(f"    ID: '{maneuver.maneuver_id}'")
                print(f"    Type: '{maneuver.maneuver_type}'")
                print(f"    File: '{Path(maneuver.file_path).name}'")
                print(f"    Duration: {maneuver.duration:.1f}s")
                print()
        else:
            print("❌ No maneuvers loaded")

    except Exception as e:
        print(f"❌ Error loading maneuvers: {e}")
        import traceback

        traceback.print_exc()


def test_prediction_file_paths():
    """Test prediction file path generation"""
    print("🔍 Testing prediction file path generation...")

    # Test with different configurations
    configs = [
        {"base_path": "predictions"},
        {"base_path": "data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/PREDICTIONS"},
    ]

    for config in configs:
        print(f"\nTesting with config: {config}")
        handler = PredictionFileHandler(config["base_path"])

        # Test path generation
        test_model = "yolov8_pose"
        test_maneuver_id = "SONY_300_SESSION_060325_C0019_clip_1_maneuver_0"

        expected_path = handler.get_prediction_file_path(test_model, test_maneuver_id)
        print(f"  Expected path: {expected_path}")
        print(f"  Directory exists: {Path(expected_path).parent.exists()}")
        print(f"  File exists: {Path(expected_path).exists()}")


def test_prediction_generation_simple():
    """Test simple prediction file generation"""
    print("🔍 Testing simple prediction file generation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")

        # Create handler
        handler = PredictionFileHandler(temp_dir)

        # Create mock maneuver
        class MockClip:
            def __init__(self):
                self.width = 1920
                self.height = 1080

        class MockManeuver:
            def __init__(self):
                self.maneuver_id = "test_maneuver_001"
                self.maneuver_type = "popup"
                self.execution_score = 8.5
                self.start_time = 10.0
                self.end_time = 15.0
                self.start_frame = 250
                self.end_frame = 375
                self.total_frames = 125
                self.fps = 25.0
                self.file_path = "/path/to/test_video.mp4"
                self.clip = MockClip()

        maneuver = MockManeuver()

        # Create mock pose result
        import numpy as np

        pose_result = {
            "keypoints": np.random.rand(1, 17, 2) * 100,
            "scores": np.random.rand(1, 17) * 0.5 + 0.5,
            "bbox": np.array([[10, 10, 50, 80]]),
            "num_persons": 1,
            "metadata": {"model": "yolov8_pose", "inference_time": 0.05},
        }

        # Convert to standardized format
        from utils.prediction_file_format import (
            get_keypoint_format_for_model,
            get_keypoint_names_for_model,
        )

        model_name = "yolov8_pose"
        keypoint_format = get_keypoint_format_for_model(model_name)
        keypoint_names = get_keypoint_names_for_model(model_name)

        frame_pred = handler.convert_model_prediction_to_standard(
            model_result=pose_result,
            frame_id=0,
            absolute_frame_id=250,
            timestamp=10.0,
            keypoint_format=keypoint_format,
            keypoint_names=keypoint_names,
        )

        # Create maneuver prediction
        maneuver_prediction = handler.create_maneuver_prediction(
            maneuver=maneuver,
            model_name=model_name,
            model_config={"confidence_threshold": 0.5},
            keypoint_format=keypoint_format,
            keypoint_names=keypoint_names,
            frame_predictions=[frame_pred],
        )

        # Save prediction
        saved_path = handler.save_prediction_file(maneuver_prediction)
        print(f"✅ Saved prediction to: {saved_path}")

        # Test loading
        expected_path = handler.get_prediction_file_path(
            model_name, maneuver.maneuver_id
        )
        print(f"Expected path: {expected_path}")
        print(f"Saved path: {saved_path}")
        print(f"Paths match: {expected_path == saved_path}")

        if Path(expected_path).exists():
            print("✅ File exists at expected path")
            loaded = handler.load_prediction_file(expected_path)
            print(
                f"✅ Successfully loaded prediction for maneuver: {loaded.maneuver_id}"
            )
        else:
            print("❌ File does not exist at expected path")


def main():
    """Run debug tests"""
    print("🐛 Debugging Prediction File System")
    print("=" * 50)

    try:
        test_maneuver_id_generation()
        print()
        test_prediction_file_paths()
        print()
        test_prediction_generation_simple()

        print("\n✅ Debug tests completed")

    except Exception as e:
        print(f"\n❌ Debug test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
