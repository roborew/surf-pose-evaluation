#!/usr/bin/env python3
"""
Test script to run a minimal evaluation and check prediction file generation
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from evaluate_pose_models import PoseEvaluator


def test_minimal_evaluation():
    """Test minimal evaluation with prediction generation"""
    print("🧪 Testing minimal evaluation with prediction generation")
    print("=" * 60)

    # Use macOS config
    config_path = "configs/evaluation_config_macos.yaml"

    try:
        # Initialize evaluator
        print("Initializing evaluator...")
        evaluator = PoseEvaluator(config_path)

        # Check available models
        available_models = evaluator.get_available_models()
        print(f"Available models: {available_models}")

        if not available_models:
            print("❌ No models available")
            return

        # Use the first available model
        model_name = available_models[0]
        print(f"Testing with model: {model_name}")

        # Load a minimal dataset
        print("Loading minimal dataset...")
        maneuvers = evaluator.data_loader.load_maneuvers(
            max_clips=1, maneuvers_per_clip=1
        )

        if not maneuvers:
            print("❌ No maneuvers loaded")
            return

        maneuver = maneuvers[0]
        print(f"Testing with maneuver: {maneuver.maneuver_id}")
        print(f"Maneuver type: {maneuver.maneuver_type}")
        print(f"Duration: {maneuver.duration:.1f}s")

        # Initialize model
        model_class = evaluator.model_registry[model_name]
        model = model_class(device=evaluator.device)

        # Test prediction generation
        print(f"\nTesting prediction generation...")
        print(
            f"Prediction handler available: {evaluator.prediction_handler is not None}"
        )

        if evaluator.prediction_handler:
            print(f"Prediction base path: {evaluator.prediction_handler.base_path}")

            # Check if directory exists and is writable
            pred_dir = Path(evaluator.prediction_handler.base_path)
            print(f"Prediction directory exists: {pred_dir.exists()}")
            if pred_dir.exists():
                print(f"Prediction directory writable: {os.access(pred_dir, os.W_OK)}")

        # Run evaluation on single maneuver
        print(f"\nRunning evaluation on single maneuver...")
        result = evaluator._process_video_maneuver(
            model, maneuver, model_name, generate_predictions=True
        )

        print(f"✅ Evaluation completed")
        print(f"Performance: {result['performance']['fps']:.1f} FPS")

        # Check if prediction file was created
        if evaluator.prediction_handler:
            expected_path = evaluator.prediction_handler.get_prediction_file_path(
                model_name, maneuver.maneuver_id
            )
            print(f"\nChecking for prediction file...")
            print(f"Expected path: {expected_path}")
            print(f"File exists: {Path(expected_path).exists()}")

            if Path(expected_path).exists():
                print(f"✅ Prediction file created successfully!")
                file_size = Path(expected_path).stat().st_size
                print(f"File size: {file_size} bytes")
            else:
                print(f"❌ Prediction file was not created")

                # List files in prediction directory
                pred_dir = Path(evaluator.prediction_handler.base_path)
                if pred_dir.exists():
                    files = list(pred_dir.glob("*.json"))
                    print(f"Files in prediction directory: {[f.name for f in files]}")

        # Test visualization loading
        print(f"\nTesting visualization loading...")
        evaluator._create_sample_visualizations(
            model, model_name, [maneuver], max_maneuvers=1
        )

        print(f"\n✅ Test completed successfully")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_minimal_evaluation()
