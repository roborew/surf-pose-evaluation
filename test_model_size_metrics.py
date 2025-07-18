#!/usr/bin/env python3
"""
Test script to verify model size metrics collection
"""

import yaml
import logging
from pathlib import Path
from utils.pose_evaluator import PoseEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_size_collection():
    """Test that model size metrics are being collected"""

    # Load a simple config
    config = {
        "data_source": {
            "base_data_path": "./data/SD_02_SURF_FOOTAGE_PREPT",
            "video_clips": {
                "h264_path": "03_CLIPPED/h264",
                "ffv1_path": "03_CLIPPED/ffv1",
                "input_format": "h264",
            },
            "annotations": {
                "labels_path": "04_ANNOTATED/surf-manoeuvre-labels",
                "sony_300_labels": "sony_300",
                "sony_70_labels": "sony_70",
            },
            "splits": {
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "random_seed": 42,
            },
        },
        "performance": {
            "device": "cpu",  # Use CPU for testing
        },
    }

    # Initialize evaluator
    evaluator = PoseEvaluator(config)

    # Test each model's performance metrics
    for model_name in evaluator.get_available_models():
        logger.info(f"Testing {model_name}...")

        try:
            # Load model config
            model_config_path = f"configs/model_configs/{model_name}.yaml"
            if Path(model_config_path).exists():
                with open(model_config_path, "r") as f:
                    model_config = yaml.safe_load(f)
            else:
                model_config = {}

            # Initialize model
            model_class = evaluator.model_registry[model_name]
            model = model_class(device=evaluator.device, **model_config)

            # Get performance metrics
            performance_metrics = model.get_performance_metrics()

            logger.info(f"  ✅ {model_name} performance metrics:")
            for key, value in performance_metrics.items():
                logger.info(f"    {key}: {value}")

            # Check if model_size_mb is present
            if "model_size_mb" in performance_metrics:
                logger.info(
                    f"  ✅ Model size: {performance_metrics['model_size_mb']}MB"
                )
            else:
                logger.warning(f"  ❌ Model size not found for {model_name}")

        except Exception as e:
            logger.error(f"  ❌ Error testing {model_name}: {e}")

    logger.info("Model size collection test completed!")


if __name__ == "__main__":
    test_model_size_collection()
