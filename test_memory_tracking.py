#!/usr/bin/env python3
"""
Test script to verify enhanced memory tracking
"""

import yaml
import logging
import numpy as np
from pathlib import Path
from utils.pose_evaluator import PoseEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_memory_tracking():
    """Test that memory tracking works across different devices"""

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
            "device": "auto",  # Let it auto-detect
        },
    }

    # Initialize evaluator
    evaluator = PoseEvaluator(config)
    logger.info(f"Using device: {evaluator.device}")

    # Test memory tracking with a simple model
    try:
        # Load MediaPipe (lightweight, good for testing)
        model_config_path = "configs/model_configs/mediapipe.yaml"
        if Path(model_config_path).exists():
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
        else:
            model_config = {}

        # Initialize model
        model_class = evaluator.model_registry["mediapipe"]
        model = model_class(device=evaluator.device, **model_config)

        # Create a dummy frame for testing
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test memory tracking
        logger.info("Testing memory tracking...")

        # Simulate the memory tracking logic
        memory_usage = []
        inference_times = []

        for i in range(5):  # Test 5 iterations
            # Simulate inference
            start_time = time.time()
            result = model.predict(dummy_frame)
            inference_time = time.time() - start_time

            inference_times.append(inference_time)

            # Enhanced memory tracking (same logic as in evaluator)
            if evaluator.device == "cuda":
                import torch

                memory_usage.append(torch.cuda.memory_allocated() / (1024**2))
            elif evaluator.device == "mps":
                import torch

                try:
                    mps_memory = torch.mps.current_allocated_memory() / (1024**2)
                    memory_usage.append(mps_memory)
                except Exception:
                    import psutil

                    process_memory = psutil.Process().memory_info().rss / (1024**2)
                    memory_usage.append(process_memory)
            else:
                import psutil

                process_memory = psutil.Process().memory_info().rss / (1024**2)
                memory_usage.append(process_memory)

        # Calculate metrics
        avg_memory = np.mean(memory_usage) if memory_usage else 0.0
        memory_std = np.std(memory_usage) if memory_usage else 0.0
        memory_peak_ratio = (
            max(memory_usage) / np.mean(memory_usage)
            if memory_usage and np.mean(memory_usage) > 0
            else 1.0
        )

        logger.info(f"✅ Memory tracking test results:")
        logger.info(f"  Memory samples: {memory_usage}")
        logger.info(f"  Average memory: {avg_memory:.2f}MB")
        logger.info(f"  Memory std: {memory_std:.2f}MB")
        logger.info(f"  Peak-to-avg ratio: {memory_peak_ratio:.2f}")
        logger.info(
            f"  Inference times: {[f'{t*1000:.1f}ms' for t in inference_times]}"
        )

    except Exception as e:
        logger.error(f"❌ Error testing memory tracking: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import time

    test_memory_tracking()
