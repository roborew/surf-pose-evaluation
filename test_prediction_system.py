#!/usr/bin/env python3
"""
Test script for the standardized prediction file system

This script tests the prediction file generation and loading functionality
to ensure it works correctly with all pose estimation models.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

from utils.prediction_file_format import (
    PredictionFileHandler,
    KeypointData,
    PersonPrediction,
    FramePrediction,
    ManeuverPrediction,
    get_keypoint_format_for_model,
    get_keypoint_names_for_model,
)


def create_mock_maneuver():
    """Create a mock maneuver object for testing"""

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

    return MockManeuver()


def create_mock_pose_result():
    """Create a mock pose estimation result"""
    # Simulate 2 people detected with 17 keypoints each
    keypoints = np.random.rand(2, 17, 2) * 100  # Random keypoints in 0-100 range
    scores = np.random.rand(2, 17) * 0.5 + 0.5  # Scores between 0.5-1.0
    bboxes = np.array([[10, 10, 50, 80], [60, 15, 90, 85]])  # Two bounding boxes

    return {
        "keypoints": keypoints,
        "scores": scores,
        "bbox": bboxes,
        "num_persons": 2,
        "metadata": {
            "model": "yolov8_pose",
            "inference_time": 0.05,
        },
    }


def test_keypoint_format_mapping():
    """Test keypoint format mapping for different models"""
    print("Testing keypoint format mapping...")

    models = ["yolov8_pose", "mediapipe", "blazepose", "mmpose", "pytorch_pose"]

    for model in models:
        format_name = get_keypoint_format_for_model(model)
        keypoint_names = get_keypoint_names_for_model(model)

        print(f"  {model}: {format_name} ({len(keypoint_names)} keypoints)")

        # Verify expected formats
        if model in ["yolov8_pose", "mmpose", "pytorch_pose"]:
            assert format_name == "coco_17"
            assert len(keypoint_names) == 17
        elif model == "mediapipe":
            assert format_name == "mediapipe_33"
            assert len(keypoint_names) == 33
        elif model == "blazepose":
            assert format_name == "blazepose_33"
            assert len(keypoint_names) == 33

    print("  ✅ Keypoint format mapping test passed")


def test_prediction_file_conversion():
    """Test conversion from model result to standardized format"""
    print("Testing prediction file conversion...")

    with tempfile.TemporaryDirectory() as temp_dir:
        handler = PredictionFileHandler(temp_dir)

        # Create mock data
        mock_pose_result = create_mock_pose_result()

        # Test conversion
        frame_prediction = handler.convert_model_prediction_to_standard(
            model_result=mock_pose_result,
            frame_id=0,
            absolute_frame_id=250,
            timestamp=10.0,
            keypoint_format="coco_17",
            keypoint_names=get_keypoint_names_for_model("yolov8_pose"),
        )

        # Verify conversion
        assert frame_prediction.frame_id == 0
        assert frame_prediction.absolute_frame_id == 250
        assert frame_prediction.timestamp == 10.0
        assert len(frame_prediction.persons) == 2

        # Check first person
        person = frame_prediction.persons[0]
        assert person.person_id == 0
        assert len(person.keypoints) == 17
        assert len(person.bbox) == 4
        assert person.detection_confidence > 0

        print("  ✅ Prediction file conversion test passed")


def test_maneuver_prediction_creation():
    """Test creation of complete maneuver prediction"""
    print("Testing maneuver prediction creation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        handler = PredictionFileHandler(temp_dir)

        # Create mock data
        mock_maneuver = create_mock_maneuver()
        model_name = "yolov8_pose"
        model_config = {"confidence_threshold": 0.5}
        keypoint_format = get_keypoint_format_for_model(model_name)
        keypoint_names = get_keypoint_names_for_model(model_name)

        # Create frame predictions
        frame_predictions = []
        for frame_idx in range(5):  # 5 frames
            mock_pose_result = create_mock_pose_result()
            frame_pred = handler.convert_model_prediction_to_standard(
                model_result=mock_pose_result,
                frame_id=frame_idx,
                absolute_frame_id=250 + frame_idx,
                timestamp=10.0 + frame_idx * 0.04,
                keypoint_format=keypoint_format,
                keypoint_names=keypoint_names,
            )
            frame_predictions.append(frame_pred)

        # Create maneuver prediction
        maneuver_prediction = handler.create_maneuver_prediction(
            maneuver=mock_maneuver,
            model_name=model_name,
            model_config=model_config,
            keypoint_format=keypoint_format,
            keypoint_names=keypoint_names,
            frame_predictions=frame_predictions,
        )

        # Verify maneuver prediction
        assert maneuver_prediction.maneuver_id == "test_maneuver_001"
        assert maneuver_prediction.model_name == model_name
        assert maneuver_prediction.keypoint_format == keypoint_format
        assert len(maneuver_prediction.frames) == 5
        assert maneuver_prediction.total_persons_detected > 0

        print("  ✅ Maneuver prediction creation test passed")


def test_prediction_file_save_load():
    """Test saving and loading prediction files"""
    print("Testing prediction file save/load...")

    with tempfile.TemporaryDirectory() as temp_dir:
        handler = PredictionFileHandler(temp_dir)

        # Create complete test data
        mock_maneuver = create_mock_maneuver()
        model_name = "yolov8_pose"
        model_config = {"confidence_threshold": 0.5}
        keypoint_format = get_keypoint_format_for_model(model_name)
        keypoint_names = get_keypoint_names_for_model(model_name)

        # Create frame predictions
        frame_predictions = []
        for frame_idx in range(3):
            mock_pose_result = create_mock_pose_result()
            frame_pred = handler.convert_model_prediction_to_standard(
                model_result=mock_pose_result,
                frame_id=frame_idx,
                absolute_frame_id=250 + frame_idx,
                timestamp=10.0 + frame_idx * 0.04,
                keypoint_format=keypoint_format,
                keypoint_names=keypoint_names,
            )
            frame_predictions.append(frame_pred)

        # Create and save maneuver prediction
        maneuver_prediction = handler.create_maneuver_prediction(
            maneuver=mock_maneuver,
            model_name=model_name,
            model_config=model_config,
            keypoint_format=keypoint_format,
            keypoint_names=keypoint_names,
            frame_predictions=frame_predictions,
        )

        # Save to file
        saved_path = handler.save_prediction_file(maneuver_prediction)
        assert Path(saved_path).exists()

        # Load from file
        loaded_prediction = handler.load_prediction_file(saved_path)

        # Verify loaded data
        assert loaded_prediction.maneuver_id == maneuver_prediction.maneuver_id
        assert loaded_prediction.model_name == maneuver_prediction.model_name
        assert loaded_prediction.keypoint_format == maneuver_prediction.keypoint_format
        assert len(loaded_prediction.frames) == len(maneuver_prediction.frames)

        # Verify frame data
        for orig_frame, loaded_frame in zip(
            maneuver_prediction.frames, loaded_prediction.frames
        ):
            assert orig_frame.frame_id == loaded_frame.frame_id
            assert orig_frame.timestamp == loaded_frame.timestamp
            assert len(orig_frame.persons) == len(loaded_frame.persons)

            for orig_person, loaded_person in zip(
                orig_frame.persons, loaded_frame.persons
            ):
                assert orig_person.person_id == loaded_person.person_id
                assert len(orig_person.keypoints) == len(loaded_person.keypoints)
                assert orig_person.bbox == loaded_person.bbox

        print("  ✅ Prediction file save/load test passed")


def test_prediction_file_listing():
    """Test listing prediction files"""
    print("Testing prediction file listing...")

    with tempfile.TemporaryDirectory() as temp_dir:
        handler = PredictionFileHandler(temp_dir)

        # Create multiple test files
        models = ["yolov8_pose", "mediapipe", "mmpose"]
        maneuver_types = ["popup", "cutback"]

        created_files = []
        for model in models:
            for maneuver_type in maneuver_types:
                # Create mock data
                mock_maneuver = create_mock_maneuver()
                mock_maneuver.maneuver_id = f"test_{maneuver_type}_{model}"
                mock_maneuver.maneuver_type = maneuver_type

                keypoint_format = get_keypoint_format_for_model(model)
                keypoint_names = get_keypoint_names_for_model(model)

                # Create minimal frame prediction
                mock_pose_result = create_mock_pose_result()
                frame_pred = handler.convert_model_prediction_to_standard(
                    model_result=mock_pose_result,
                    frame_id=0,
                    absolute_frame_id=250,
                    timestamp=10.0,
                    keypoint_format=keypoint_format,
                    keypoint_names=keypoint_names,
                )

                maneuver_prediction = handler.create_maneuver_prediction(
                    maneuver=mock_maneuver,
                    model_name=model,
                    model_config={},
                    keypoint_format=keypoint_format,
                    keypoint_names=keypoint_names,
                    frame_predictions=[frame_pred],
                )

                saved_path = handler.save_prediction_file(maneuver_prediction)
                created_files.append(saved_path)

        # Test listing all files
        all_files = handler.list_prediction_files()
        assert len(all_files) == len(created_files)

        # Test filtering by model
        yolo_files = handler.list_prediction_files(model_name="yolov8_pose")
        assert len(yolo_files) == 2  # 2 maneuver types

        # Test filtering by maneuver type
        popup_files = handler.list_prediction_files(maneuver_type="popup")
        assert len(popup_files) == 3  # 3 models

        print("  ✅ Prediction file listing test passed")


def main():
    """Run all tests"""
    print("🧪 Testing Standardized Prediction File System")
    print("=" * 50)

    try:
        test_keypoint_format_mapping()
        test_prediction_file_conversion()
        test_maneuver_prediction_creation()
        test_prediction_file_save_load()
        test_prediction_file_listing()

        print("\n✅ All tests passed!")
        print("The standardized prediction file system is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
