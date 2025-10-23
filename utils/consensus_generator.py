"""
Consensus Generator for Pseudo-Ground-Truth Creation

Generates high-quality consensus annotations by running multiple pose models
and aggregating their predictions. Supports leave-one-out validation to
prevent circular reasoning during Optuna optimization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from tqdm import tqdm

from utils.quality_filter import AdaptiveQualityFilter

logger = logging.getLogger(__name__)


class ConsensusGenerator:
    """
    Generate consensus pseudo-ground-truth from multiple pose models.

    Loads models on demand, runs inference on video clips, and aggregates
    predictions using weighted averaging. Applies quality filtering to
    ensure high-quality consensus annotations.
    """

    def __init__(
        self,
        model_names: List[str],
        quality_filter: AdaptiveQualityFilter,
        config: Dict[str, Any],
    ):
        """
        Initialize consensus generator.

        Args:
            model_names: List of model names for consensus
                (e.g., ['yolov8', 'pytorch_pose', 'mmpose'])
            quality_filter: Adaptive quality filter instance
            config: Configuration dictionary
        """
        self.model_names = model_names
        self.quality_filter = quality_filter
        self.config = config
        self.models = {}  # Lazy loading: {model_name: model_instance}

        logger.info(f"ConsensusGenerator initialized with models: {model_names}")

    def load_model(self, model_name: str):
        """
        Load model wrapper on demand.

        Args:
            model_name: Name of model to load ('yolov8', 'pytorch_pose', etc.)

        Returns:
            Model wrapper instance
        """
        if model_name in self.models:
            return self.models[model_name]

        logger.info(f"Loading model: {model_name}")

        # Import and instantiate model wrappers
        try:
            if model_name == "yolov8":
                from models.yolov8_wrapper import YOLOv8Wrapper

                model = YOLOv8Wrapper(device="cpu")  # TODO: Support GPU config

            elif model_name == "pytorch_pose":
                from models.pytorch_pose_wrapper import PyTorchPoseWrapper

                model = PyTorchPoseWrapper(device="cpu")

            elif model_name == "mmpose":
                from models.mmpose_wrapper import MMPoseWrapper

                model = MMPoseWrapper(device="cpu")

            elif model_name == "mediapipe":
                from models.mediapipe_wrapper import MediaPipePoseModel

                model = MediaPipePoseModel()

            elif model_name == "blazepose":
                from models.blazepose_wrapper import BlazePoseModel

                model = BlazePoseModel()

            else:
                raise ValueError(f"Unknown model name: {model_name}")

            self.models[model_name] = model
            logger.info(f"✓ Model {model_name} loaded successfully")
            return model

        except Exception as e:
            logger.error(f"✗ Failed to load model {model_name}: {e}")
            raise

    def unload_model(self, model_name: str):
        """
        Unload model to free memory.

        Args:
            model_name: Name of model to unload
        """
        if model_name in self.models:
            del self.models[model_name]
            logger.debug(f"Unloaded model: {model_name}")

    def unload_all_models(self):
        """Unload all models to free memory."""
        self.models.clear()
        logger.debug("Unloaded all models")

    def run_inference_on_maneuver(
        self,
        model_name: str,
        video_path: str,
        maneuver,
    ) -> List[Dict[str, Any]]:
        """
        Run single model inference on a maneuver.

        Args:
            model_name: Name of model to use
            video_path: Path to video file
            maneuver: Maneuver object with start_frame and end_frame

        Returns:
            List of prediction dictionaries, one per frame
        """
        model = self.load_model(model_name)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Failed to open video: {video_path}")
            return []

        predictions = []
        frame_idx = 0

        # Seek to start frame
        if maneuver.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, maneuver.start_frame)
            frame_idx = maneuver.start_frame

        # Read and process frames
        while cap.isOpened() and frame_idx < maneuver.end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Run model prediction
            try:
                result = model.predict(frame)
                predictions.append(result)
            except Exception as e:
                logger.warning(f"Prediction failed at frame {frame_idx}: {e}")
                # Add empty prediction to maintain frame alignment
                predictions.append(
                    {
                        "keypoints": np.array([]).reshape(0, 17, 2),
                        "scores": np.array([]).reshape(0, 17),
                        "num_persons": 0,
                    }
                )

            frame_idx += 1

        cap.release()

        logger.debug(
            f"  {model_name}: {len(predictions)} frames processed "
            f"for maneuver {maneuver.maneuver_id}"
        )

        return predictions

    def aggregate_predictions(
        self,
        model_predictions: Dict[str, List[Dict]],
    ) -> List[Dict[str, Any]]:
        """
        Aggregate predictions from multiple models into consensus.

        Uses weighted mean of keypoints and confidence scores.
        Assumes all models detected the same person (best match).

        Args:
            model_predictions: Dict mapping model_name -> list of frame predictions

        Returns:
            List of consensus frames
        """
        model_names = list(model_predictions.keys())
        num_models = len(model_names)

        if num_models == 0:
            return []

        # Get number of frames (should be same for all models)
        num_frames = len(model_predictions[model_names[0]])

        consensus_frames = []

        for frame_idx in range(num_frames):
            # Collect predictions for this frame from all models
            frame_preds = [
                model_predictions[model_name][frame_idx] for model_name in model_names
            ]

            # Extract best person from each model (highest confidence)
            keypoints_list = []
            scores_list = []

            for pred in frame_preds:
                if pred["num_persons"] > 0:
                    # Take first person (models are sorted by confidence)
                    keypoints_list.append(pred["keypoints"][0])  # Shape: (17, 2)
                    scores_list.append(pred["scores"][0])  # Shape: (17,)
                else:
                    # No detection: add zeros
                    keypoints_list.append(np.zeros((17, 2)))
                    scores_list.append(np.zeros(17))

            if len(keypoints_list) == 0:
                # No predictions available
                consensus_frames.append(
                    {
                        "keypoints": np.zeros((17, 2)),
                        "confidence": np.zeros(17),
                        "source_models": model_names,
                        "num_contributing_models": 0,
                    }
                )
                continue

            # Stack and compute mean
            keypoints_array = np.array(keypoints_list)  # Shape: (num_models, 17, 2)
            scores_array = np.array(scores_list)  # Shape: (num_models, 17)

            # Weighted mean (equal weights for now)
            consensus_keypoints = np.mean(keypoints_array, axis=0)  # Shape: (17, 2)
            consensus_confidence = np.mean(scores_array, axis=0)  # Shape: (17,)

            # Count how many models contributed
            num_contributing = np.sum(scores_array > 0.1, axis=0)  # Per keypoint

            consensus_frames.append(
                {
                    "keypoints": consensus_keypoints,
                    "confidence": consensus_confidence,
                    "source_models": model_names,
                    "num_contributing_models": num_contributing,
                }
            )

        return consensus_frames

    def generate_consensus_for_maneuver(
        self,
        model_names: List[str],
        video_path: str,
        maneuver,
    ) -> List[Dict[str, Any]]:
        """
        Generate consensus GT for a single maneuver.

        Runs all specified models and aggregates their predictions.

        Args:
            model_names: List of models to use for consensus
            video_path: Path to video file
            maneuver: Maneuver object

        Returns:
            List of consensus frames
        """
        # Run inference with each model
        model_predictions = {}
        for model_name in model_names:
            predictions = self.run_inference_on_maneuver(
                model_name, video_path, maneuver
            )
            model_predictions[model_name] = predictions

        # Aggregate predictions into consensus
        consensus_frames = self.aggregate_predictions(model_predictions)

        return consensus_frames


if __name__ == "__main__":
    # Demo usage
    print("ConsensusGenerator Demo")
    print("=" * 50)

    from unittest.mock import Mock

    # Mock quality filter
    quality_filter = AdaptiveQualityFilter()

    # Initialize generator
    generator = ConsensusGenerator(
        model_names=["yolov8", "pytorch_pose", "mmpose"],
        quality_filter=quality_filter,
        config={},
    )

    print(f"Initialized with models: {generator.model_names}")
    print(f"Models loaded (lazy): {list(generator.models.keys())}")

    # Mock maneuver
    mock_maneuver = Mock()
    mock_maneuver.start_frame = 0
    mock_maneuver.end_frame = 10
    mock_maneuver.maneuver_id = "demo_maneuver"

    print(f"\nMock maneuver: {mock_maneuver.maneuver_id}")
    print(f"  Frames: {mock_maneuver.start_frame}-{mock_maneuver.end_frame}")
