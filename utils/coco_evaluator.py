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
        # Add comprehensive diagnostics
        model_name = prediction_result.get("metadata", {}).get("model", "unknown")
        logger.info(f"🔍 Converting {model_name} prediction for COCO evaluation")

        # Log the raw prediction structure
        logger.info(f"📊 Raw prediction keys: {list(prediction_result.keys())}")
        logger.info(
            f"📊 Prediction types: {[(k, type(v)) for k, v in prediction_result.items()]}"
        )

        if "keypoints" in prediction_result:
            kpts = prediction_result["keypoints"]
            logger.info(f"📊 Keypoints type: {type(kpts)}")
            logger.info(
                f"📊 Keypoints shape: {kpts.shape if hasattr(kpts, 'shape') else 'No shape'}"
            )
            logger.info(
                f"📊 Keypoints dtype: {kpts.dtype if hasattr(kpts, 'dtype') else 'No dtype'}"
            )
            logger.info(
                f"📊 Keypoints sample (first few values): {kpts.flat[:10] if hasattr(kpts, 'flat') else 'Cannot access'}"
            )

        if "scores" in prediction_result:
            scores = prediction_result["scores"]
            logger.info(f"📊 Scores type: {type(scores)}")
            logger.info(
                f"📊 Scores shape: {scores.shape if hasattr(scores, 'shape') else len(scores)}"
            )

        # Try the conversion and catch specific errors
        try:
            converted_result = self._convert_prediction_with_robust_handling(
                prediction_result, image_data, inference_time, model_name
            )
            logger.info(f"✅ Successfully converted {model_name} prediction")
            return converted_result

        except Exception as e:
            logger.error(f"❌ Failed to convert {model_name} prediction: {e}")
            logger.error(f"📊 Error type: {type(e).__name__}")
            import traceback

            logger.error(f"📊 Full traceback:\n{traceback.format_exc()}")

            # Return empty result to continue evaluation
            return {
                "keypoints": np.array([]),
                "scores": [],
                "bbox": [],
                "image_id": image_data["id"],
                "inference_time": inference_time,
                "num_persons": 0,
                "conversion_error": str(e),
                "model_name": model_name,
            }

    def _convert_prediction_with_robust_handling(
        self,
        prediction_result: Dict[str, Any],
        image_data: Dict[str, Any],
        inference_time: float,
        model_name: str,
    ) -> Dict[str, Any]:
        """Robust prediction conversion with model-specific handling"""

        # Model-specific conversion
        if model_name == "mmpose":
            return self._convert_mmpose_to_coco17(
                prediction_result, image_data, inference_time
            )
        elif model_name == "yolov8_pose":
            return self._convert_yolov8_to_coco17(
                prediction_result, image_data, inference_time
            )
        else:
            return self._convert_generic_to_coco17(
                prediction_result, image_data, inference_time
            )

    def _convert_mmpose_to_coco17(
        self,
        prediction_result: Dict[str, Any],
        image_data: Dict[str, Any],
        inference_time: float,
    ) -> Dict[str, Any]:
        """Convert MMPose prediction to COCO-17 format"""
        logger.info("🔧 Using MMPose-specific converter")

        keypoints = []
        scores = []
        bboxes = []

        if "keypoints" in prediction_result:
            raw_keypoints = prediction_result["keypoints"]
            raw_scores = prediction_result.get("scores", [])
            raw_bboxes = prediction_result.get("bbox", [])

            logger.info(f"📊 MMPose keypoints shape: {raw_keypoints.shape}")
            logger.info(
                f"📊 MMPose scores shape: {raw_scores.shape if hasattr(raw_scores, 'shape') else len(raw_scores)}"
            )

            # Handle different MMPose output formats
            if len(raw_keypoints.shape) == 3:  # (N, 17, 2) or (N, 17, 3)
                for person_idx in range(raw_keypoints.shape[0]):
                    person_kpts = raw_keypoints[person_idx]

                    # Ensure we have exactly 17 keypoints
                    if person_kpts.shape[0] >= 17:
                        # Take first 17 keypoints and only x,y coordinates
                        person_kpts_17 = person_kpts[:17, :2]
                        keypoints.append(person_kpts_17)

                        # Handle scores
                        if len(raw_scores) > person_idx and hasattr(
                            raw_scores[person_idx], "__len__"
                        ):
                            person_scores = (
                                raw_scores[person_idx][:17]
                                if len(raw_scores[person_idx]) >= 17
                                else [0.5] * 17
                            )
                        else:
                            person_scores = [0.5] * 17
                        scores.append(person_scores)

                        # Handle bboxes
                        if len(raw_bboxes) > person_idx:
                            bbox = raw_bboxes[person_idx]
                            if hasattr(bbox, "flatten"):
                                bbox = bbox.flatten()
                            bboxes.append(
                                bbox[:4] if len(bbox) >= 4 else [0, 0, 100, 100]
                            )
                        else:
                            bboxes.append([0, 0, 100, 100])

        return {
            "keypoints": np.array(keypoints) if keypoints else np.array([]),
            "scores": scores,
            "bbox": bboxes,
            "image_id": image_data["id"],
            "inference_time": inference_time,
            "num_persons": len(keypoints),
        }

    def _convert_yolov8_to_coco17(
        self,
        prediction_result: Dict[str, Any],
        image_data: Dict[str, Any],
        inference_time: float,
    ) -> Dict[str, Any]:
        """Convert YOLOv8 prediction to COCO-17 format"""
        logger.info("🔧 Using YOLOv8-specific converter")

        keypoints = []
        scores = []
        bboxes = []

        if "keypoints" in prediction_result:
            raw_keypoints = prediction_result["keypoints"]
            raw_scores = prediction_result.get("scores", [])
            raw_bboxes = prediction_result.get("bbox", [])

            logger.info(f"📊 YOLOv8 keypoints shape: {raw_keypoints.shape}")
            logger.info(
                f"📊 YOLOv8 scores shape: {raw_scores.shape if hasattr(raw_scores, 'shape') else len(raw_scores)}"
            )

            # Ensure numpy arrays (convert from tensors if needed)
            if hasattr(raw_keypoints, "cpu"):
                raw_keypoints = raw_keypoints.cpu().numpy()
            if hasattr(raw_scores, "cpu"):
                raw_scores = raw_scores.cpu().numpy()
            if hasattr(raw_bboxes, "cpu"):
                raw_bboxes = raw_bboxes.cpu().numpy()

            # Handle YOLOv8 format: (N, 17, 2) keypoints, (N, 17) scores
            if len(raw_keypoints.shape) == 3 and raw_keypoints.shape[1] == 17:
                for person_idx in range(raw_keypoints.shape[0]):
                    person_kpts = raw_keypoints[person_idx]  # (17, 2)
                    keypoints.append(person_kpts)

                    # Handle scores
                    if len(raw_scores.shape) == 2 and raw_scores.shape[1] == 17:
                        person_scores = raw_scores[person_idx]  # (17,)
                    else:
                        person_scores = [0.5] * 17
                    scores.append(person_scores)

                    # Handle bboxes
                    if len(raw_bboxes) > person_idx:
                        bbox = raw_bboxes[person_idx]
                        bboxes.append(bbox[:4] if len(bbox) >= 4 else [0, 0, 100, 100])
                    else:
                        bboxes.append([0, 0, 100, 100])

        return {
            "keypoints": np.array(keypoints) if keypoints else np.array([]),
            "scores": scores,
            "bbox": bboxes,
            "image_id": image_data["id"],
            "inference_time": inference_time,
            "num_persons": len(keypoints),
        }

    def _convert_generic_to_coco17(
        self,
        prediction_result: Dict[str, Any],
        image_data: Dict[str, Any],
        inference_time: float,
    ) -> Dict[str, Any]:
        """Generic conversion for other models (MediaPipe, BlazePose, PyTorch)"""
        model_name = prediction_result.get("metadata", {}).get("model", "unknown")
        logger.info(f"🔧 Using generic converter for {model_name}")

        keypoints = []
        scores = []
        bboxes = []

        if "keypoints" in prediction_result:
            raw_keypoints = prediction_result["keypoints"]
            raw_scores = prediction_result.get("scores", [])
            raw_bboxes = prediction_result.get("bbox", [])

            # Handle MediaPipe/BlazePose (33 keypoints) -> COCO-17 conversion
            if raw_keypoints.shape[1] == 33:  # MediaPipe/BlazePose format
                # Map 33 keypoints to 17 COCO keypoints
                coco_indices = [
                    0,
                    2,
                    5,
                    7,
                    8,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                ]

                for person_idx in range(raw_keypoints.shape[0]):
                    person_kpts_33 = raw_keypoints[person_idx]  # (33, 2/3)
                    person_kpts_17 = person_kpts_33[
                        coco_indices, :2
                    ]  # Take only x,y for COCO indices
                    keypoints.append(person_kpts_17)

                    # Map scores
                    if len(raw_scores) > person_idx:
                        person_scores_33 = raw_scores[person_idx]
                        person_scores_17 = person_scores_33[coco_indices]
                    else:
                        person_scores_17 = [0.5] * 17
                    scores.append(person_scores_17)

                    # Handle bboxes
                    if len(raw_bboxes) > person_idx:
                        bbox = raw_bboxes[person_idx]
                        bboxes.append(bbox[:4] if len(bbox) >= 4 else [0, 0, 100, 100])
                    else:
                        bboxes.append([0, 0, 100, 100])

            elif raw_keypoints.shape[1] == 17:  # Already COCO-17 format (PyTorch)
                for person_idx in range(raw_keypoints.shape[0]):
                    person_kpts = raw_keypoints[person_idx, :, :2]  # Take only x,y
                    keypoints.append(person_kpts)

                    # Handle scores
                    if len(raw_scores) > person_idx:
                        person_scores = raw_scores[person_idx][:17]
                    else:
                        person_scores = [0.5] * 17
                    scores.append(person_scores)

                    # Handle bboxes
                    if len(raw_bboxes) > person_idx:
                        bbox = raw_bboxes[person_idx]
                        bboxes.append(bbox[:4] if len(bbox) >= 4 else [0, 0, 100, 100])
                    else:
                        bboxes.append([0, 0, 100, 100])

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

            logger.info("🔍 Testing model compatibility with dummy image...")

            # Test prediction
            result = model_wrapper.predict(dummy_image)

            # Log the result structure for diagnostics
            model_name = result.get("metadata", {}).get("model", "unknown")
            logger.info(f"📊 Compatibility test for {model_name}")
            logger.info(f"📊 Result keys: {list(result.keys())}")
            logger.info(f"📊 Result types: {[(k, type(v)) for k, v in result.items()]}")

            # Check if result has expected format
            if "keypoints" not in result:
                logger.error(f"❌ {model_name} prediction missing 'keypoints' field")
                return False

            keypoints = result["keypoints"]
            logger.info(f"📊 Keypoints type: {type(keypoints)}")
            logger.info(
                f"📊 Keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'No shape'}"
            )

            # Check if we have any detections
            if hasattr(keypoints, "shape"):
                if len(keypoints.shape) == 0 or keypoints.shape[0] == 0:
                    logger.info(
                        f"ℹ️ {model_name} produced no detections on dummy image (this is normal)"
                    )
                    # No detections is OK for compatibility check
                    logger.info(
                        f"✅ {model_name} is compatible with COCO evaluation (format check passed)"
                    )
                    return True
                else:
                    # We have detections, check format
                    logger.info(
                        f"📊 {model_name} detected {keypoints.shape[0]} persons"
                    )
                    if len(keypoints.shape) >= 2:
                        logger.info(f"📊 Keypoints per person: {keypoints.shape[1]}")
                        if keypoints.shape[1] < 17 and keypoints.shape[1] != 33:
                            logger.warning(
                                f"⚠️ {model_name} produces {keypoints.shape[1]} keypoints, expected 17 (COCO) or 33 (MediaPipe/BlazePose)"
                            )
                            return False
            else:
                # Keypoints is not a numpy array, check if it's a list
                if isinstance(keypoints, list):
                    if len(keypoints) == 0:
                        logger.info(
                            f"ℹ️ {model_name} produced no detections on dummy image (this is normal)"
                        )
                        logger.info(
                            f"✅ {model_name} is compatible with COCO evaluation (format check passed)"
                        )
                        return True
                    else:
                        logger.info(
                            f"📊 {model_name} detected {len(keypoints)} persons (list format)"
                        )
                        if len(keypoints[0]) < 17:
                            logger.warning(
                                f"⚠️ {model_name} produces {len(keypoints[0])} keypoints, need at least 17 for COCO"
                            )
                            return False

            logger.info(f"✅ {model_name} is compatible with COCO evaluation")
            return True

        except Exception as e:
            model_name = getattr(model_wrapper, "model_name", "unknown")
            logger.error(f"❌ {model_name} compatibility check failed: {e}")
            import traceback

            logger.error(f"📊 Full traceback:\n{traceback.format_exc()}")
            return False
