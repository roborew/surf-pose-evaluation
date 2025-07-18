"""
COCO 2017 Pose Evaluation Module
Evaluates pose models against COCO ground truth annotations for true PCK scores
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
from PIL import Image
import requests
from io import BytesIO
import os
import time

from metrics.pose_metrics import PoseMetrics

logger = logging.getLogger(__name__)


class COCOPoseEvaluator:
    """Evaluate pose models on COCO 2017 validation dataset"""

    def __init__(
        self,
        coco_annotations_path: str,
        coco_images_path: Optional[str] = None,
        max_images: int = 100,
        download_images: bool = False,
    ):
        """Initialize COCO evaluator

        Args:
            coco_annotations_path: Path to COCO keypoint annotations JSON
            coco_images_path: Path to COCO images directory (optional if downloading)
            max_images: Maximum number of images to evaluate (for testing)
            download_images: Whether to download images from COCO URLs
        """
        self.coco_annotations_path = Path(coco_annotations_path)
        self.coco_images_path = Path(coco_images_path) if coco_images_path else None
        self.max_images = max_images
        self.download_images = download_images

        # Load COCO data
        self.coco_data = self._load_coco_annotations()
        self.pose_metrics = PoseMetrics()

        # COCO keypoint format (17 keypoints)
        self.coco_keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        logger.info(
            f"Initialized COCO evaluator with {len(self.coco_data['images'])} images"
        )

    def _load_coco_annotations(self) -> Dict[str, Any]:
        """Load COCO annotations from JSON file"""
        logger.info(f"Loading COCO annotations from {self.coco_annotations_path}")

        with open(self.coco_annotations_path, "r") as f:
            coco_data = json.load(f)

        logger.info(
            f"Loaded {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations"
        )
        return coco_data

    def evaluate_model(
        self, model_wrapper, model_name: str, subset_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate a pose model on COCO validation set

        Args:
            model_wrapper: Pose model wrapper instance
            model_name: Name of the model
            subset_size: Number of images to evaluate (None for max_images)

        Returns:
            Dictionary with COCO evaluation metrics
        """
        subset_size = subset_size or self.max_images

        logger.info(
            f"Evaluating {model_name} on COCO validation set (max {subset_size} images)"
        )

        # Get evaluation subset
        eval_data = self._prepare_evaluation_subset(subset_size)

        if not eval_data:
            logger.error("No evaluation data prepared")
            return {}

        predictions = []
        ground_truths = []
        performance_metrics = []

        for i, (image_data, gt_annotations) in enumerate(eval_data):
            logger.info(
                f"Processing image {i+1}/{len(eval_data)}: {image_data['file_name']}"
            )

            try:
                # Load image
                image = self._load_image(image_data)
                if image is None:
                    continue

                # Run pose estimation
                start_time = time.time()
                result = model_wrapper.predict(image)
                inference_time = time.time() - start_time

                # Convert result to evaluation format
                prediction = self._convert_prediction_to_eval_format(
                    result, image_data, inference_time
                )

                # Convert ground truth to evaluation format
                ground_truth = self._convert_ground_truth_to_eval_format(
                    gt_annotations, image_data
                )

                predictions.append(prediction)
                ground_truths.append(ground_truth)

                # Track performance
                performance_metrics.append(
                    {
                        "inference_time": inference_time,
                        "image_size": (image_data["width"], image_data["height"]),
                    }
                )

            except Exception as e:
                logger.error(f"Error processing image {image_data['file_name']}: {e}")
                continue

        if not predictions:
            logger.error("No successful predictions")
            return {}

        # Calculate COCO PCK metrics
        logger.info("Calculating COCO PCK metrics...")
        metrics = self._calculate_coco_metrics(predictions, ground_truths)

        # Add performance metrics
        perf_metrics = self._calculate_performance_metrics(performance_metrics)
        metrics.update(perf_metrics)

        # Add COCO prefix to distinguish from other metrics
        coco_metrics = {f"coco_{k}": v for k, v in metrics.items()}

        logger.info(f"COCO evaluation completed for {model_name}")
        return coco_metrics

    def _prepare_evaluation_subset(
        self, max_size: int
    ) -> List[Tuple[Dict, List[Dict]]]:
        """Prepare subset of COCO data for evaluation

        Args:
            max_size: Maximum number of images to include

        Returns:
            List of (image_data, annotations) tuples
        """
        # Filter images that have keypoint annotations
        images_with_keypoints = {}

        for annotation in self.coco_data["annotations"]:
            if (
                annotation["category_id"] == 1  # Person category
                and "keypoints" in annotation
                and annotation.get("num_keypoints", 0) > 5
            ):  # At least 5 visible keypoints

                image_id = annotation["image_id"]
                if image_id not in images_with_keypoints:
                    images_with_keypoints[image_id] = []
                images_with_keypoints[image_id].append(annotation)

        # Get image metadata
        image_metadata = {img["id"]: img for img in self.coco_data["images"]}

        # Prepare evaluation data
        eval_data = []
        for image_id, annotations in images_with_keypoints.items():
            if len(eval_data) >= max_size:
                break

            if image_id in image_metadata:
                image_data = image_metadata[image_id]
                eval_data.append((image_data, annotations))

        logger.info(f"Prepared {len(eval_data)} images for evaluation")
        return eval_data

    def _load_image(self, image_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load image from file or download from URL

        Args:
            image_data: COCO image metadata

        Returns:
            Image as numpy array or None if failed
        """
        file_name = image_data["file_name"]

        # Try to load from local path first
        if self.coco_images_path:
            image_path = self.coco_images_path / file_name
            if image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If download enabled and local file not found, try to download
        if self.download_images and "coco_url" in image_data:
            try:
                response = requests.get(image_data["coco_url"], timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    return np.array(image.convert("RGB"))
            except Exception as e:
                logger.warning(f"Failed to download image {file_name}: {e}")

        logger.warning(f"Could not load image {file_name}")
        return None

    def _convert_prediction_to_eval_format(
        self,
        prediction_result: Dict[str, Any],
        image_data: Dict[str, Any],
        inference_time: float,
    ) -> Dict[str, Any]:
        """Convert model prediction to evaluation format

        Args:
            prediction_result: Raw model prediction
            image_data: COCO image metadata
            inference_time: Time taken for inference

        Returns:
            Prediction in evaluation format
        """
        keypoints = []
        scores = []
        bboxes = []

        if "keypoints" in prediction_result:
            for person_kpts in prediction_result["keypoints"]:
                # Convert to COCO format if needed
                if len(person_kpts) >= 17:  # Ensure we have enough keypoints
                    keypoints.append(person_kpts[:17])  # Take first 17 keypoints

                    # Extract confidence scores if available
                    if len(person_kpts[0]) >= 3:  # Has confidence
                        person_scores = [
                            kpt[2] if len(kpt) >= 3 else 0.5 for kpt in person_kpts[:17]
                        ]
                    else:
                        person_scores = [0.5] * 17  # Default confidence
                    scores.append(person_scores)

        # Extract bounding boxes if available
        if "bbox" in prediction_result:
            bboxes = prediction_result["bbox"]

        return {
            "keypoints": np.array(keypoints) if keypoints else np.array([]),
            "scores": scores,
            "bbox": bboxes,
            "image_id": image_data["id"],
            "inference_time": inference_time,
            "num_persons": len(keypoints),
        }

    def _convert_ground_truth_to_eval_format(
        self, annotations: List[Dict[str, Any]], image_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert COCO annotations to evaluation format

        Args:
            annotations: List of COCO annotations for this image
            image_data: COCO image metadata

        Returns:
            Ground truth in evaluation format
        """
        keypoints = []
        visibility = []
        bboxes = []

        for annotation in annotations:
            if "keypoints" in annotation:
                # COCO keypoints format: [x1,y1,v1, x2,y2,v2, ...]
                coco_kpts = annotation["keypoints"]

                # Reshape to (17, 3) format
                person_kpts = np.array(coco_kpts).reshape(-1, 3)

                # Extract coordinates and visibility
                coords = person_kpts[:, :2]  # x, y coordinates
                vis = person_kpts[:, 2]  # visibility flags

                keypoints.append(coords)
                visibility.append(vis)

                # Extract bounding box
                if "bbox" in annotation:
                    bboxes.append(annotation["bbox"])

        return {
            "keypoints": np.array(keypoints) if keypoints else np.array([]),
            "visibility": visibility,
            "bbox": bboxes,
            "image_id": image_data["id"],
            "num_persons": len(keypoints),
        }

    def _calculate_coco_metrics(
        self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate COCO-specific metrics including multiple PCK thresholds

        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth annotations

        Returns:
            Dictionary with COCO metrics
        """
        metrics = {}

        # Calculate PCK at multiple thresholds
        pck_thresholds = [0.1, 0.2, 0.3, 0.5]

        for threshold in pck_thresholds:
            pck_metrics = self.pose_metrics.calculate_pck(
                predictions, ground_truths, threshold=threshold
            )

            # Rename metrics with threshold suffix
            for key, value in pck_metrics.items():
                if "pck" in key:
                    new_key = f"pck_{threshold}"
                    metrics[new_key] = value
                elif key == "mean_normalized_distance":
                    metrics[f"mean_distance_pck_{threshold}"] = value
                else:
                    metrics[key] = value

        # Calculate overall PCK error (using 0.2 threshold as standard)
        if "pck_0.2" in metrics:
            metrics["pck_error_mean"] = 1.0 - metrics["pck_0.2"]

        # Calculate detection metrics
        detection_metrics = self.pose_metrics.calculate_detection_metrics(
            predictions, ground_truths
        )
        metrics.update(detection_metrics)

        return metrics

    def _calculate_performance_metrics(
        self, performance_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate performance metrics from COCO evaluation

        Args:
            performance_data: List of performance measurements

        Returns:
            Dictionary with performance metrics
        """
        if not performance_data:
            return {}

        inference_times = [p["inference_time"] for p in performance_data]

        metrics = {
            "fps_mean": 1.0 / np.mean(inference_times) if inference_times else 0,
            "inference_time_ms": np.mean(inference_times) * 1000,
            "inference_time_std_ms": np.std(inference_times) * 1000,
            "min_inference_time_ms": np.min(inference_times) * 1000,
            "max_inference_time_ms": np.max(inference_times) * 1000,
            "total_images_processed": len(performance_data),
        }

        return metrics

    def get_coco_keypoint_mapping(self) -> Dict[str, int]:
        """Get mapping of COCO keypoint names to indices

        Returns:
            Dictionary mapping keypoint names to indices
        """
        return {name: idx for idx, name in enumerate(self.coco_keypoint_names)}

    def validate_model_compatibility(self, model_wrapper) -> bool:
        """Check if model wrapper is compatible with COCO evaluation

        Args:
            model_wrapper: Model wrapper to validate

        Returns:
            True if compatible, False otherwise
        """
        try:
            # Create a dummy image
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

            # Test prediction
            result = model_wrapper.predict(dummy_image)

            # Check if result has expected format
            if "keypoints" not in result:
                logger.error("Model prediction missing 'keypoints' field")
                return False

            if len(result["keypoints"]) > 0:
                person_kpts = result["keypoints"][0]
                if len(person_kpts) < 17:
                    logger.warning(
                        f"Model produces {len(person_kpts)} keypoints, need at least 17 for COCO"
                    )
                    return False

            logger.info("Model is compatible with COCO evaluation")
            return True

        except Exception as e:
            logger.error(f"Model compatibility check failed: {e}")
            return False
