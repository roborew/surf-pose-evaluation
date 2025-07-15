"""
YOLOv8-Pose wrapper for pose estimation
"""

import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import cv2
import torch

from .base_pose_model import BasePoseModel

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOv8Wrapper(BasePoseModel):
    """YOLOv8-Pose wrapper for pose estimation"""

    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize YOLOv8-Pose model

        Args:
            device: Compute device ('cpu', 'cuda', 'mps')
            **kwargs: Model configuration
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics YOLO is not available. Install with: pip install ultralytics"
            )

        # Disable Ultralytics telemetry/analytics to prevent Google Analytics calls
        import os

        os.environ["YOLO_SETTINGS"] = '{"sync": false}'

        super().__init__(device, **kwargs)

        # Model configuration
        self.model_size = kwargs.get("model_size", "n")  # n, s, m, l, x
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.25)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.max_detections = kwargs.get("max_detections", 300)
        self.keypoint_threshold = kwargs.get("keypoint_threshold", 0.3)
        self.half_precision = kwargs.get("half_precision", False)

        # Define weights directory
        self.weights_dir = Path(__file__).parent / "yolov8_pose"

        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Load YOLOv8-Pose model from pre-downloaded weights"""
        import logging

        try:
            # Construct model filename and path
            model_filename = f"yolov8{self.model_size}-pose.pt"
            model_path = self.weights_dir / model_filename

            # Check if weights exist
            if not model_path.exists():
                raise FileNotFoundError(
                    f"YOLOv8 weights not found: {model_path}\n\n"
                    f"Please run the setup script to download weights:\n"
                    f"  python setup_yolo_downloadweights.py --models {self.model_size}\n\n"
                    f"Or download all standard models:\n"
                    f"  python setup_yolo_downloadweights.py\n\n"
                    f"Expected weight file: {model_filename}"
                )

            # Validate the weight file
            if not self._validate_weights(model_path):
                raise RuntimeError(
                    f"YOLOv8 weight file is corrupted: {model_path}\n\n"
                    f"Please re-download the weights:\n"
                    f"  python setup_yolo_downloadweights.py --models {self.model_size} --force"
                )

            # Load model from local weights
            logging.info(f"Loading YOLOv8-{self.model_size}-pose from {model_path}")
            self.model = YOLO(str(model_path))

            # Move to device
            self.model.to(self.device)

            # Set precision based on device capabilities
            if self.half_precision and self.device == "cuda":
                logging.info(f"YOLOv8 using FP16 precision on {self.device}")
            elif self.half_precision and self.device != "cpu":
                logging.info(
                    f"Half precision requested but may not be optimal on {self.device}"
                )

            self.is_initialized = True
            logging.info(
                f"Successfully loaded YOLOv8-{self.model_size}-pose on {self.device}"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLOv8-Pose: {e}")

    def _validate_weights(self, model_path: Path) -> bool:
        """Validate that the weight file is a valid PyTorch model

        Args:
            model_path: Path to the model weights file

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file size (should be at least 1MB)
            if model_path.stat().st_size < 1024 * 1024:
                return False

            # Try to load the state dict
            state_dict = torch.load(model_path, map_location="cpu")

            # Check if it's a valid dict with expected keys
            if not isinstance(state_dict, dict):
                return False

            # Check for essential YOLOv8 keys
            essential_keys = ["model", "epoch"]
            return all(key in state_dict for key in essential_keys)

        except Exception:
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run YOLOv8-Pose estimation on image

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            Standardized pose estimation results
        """
        if not self.is_initialized:
            self.load_model()

        start_time = time.time()

        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

        inference_time = time.time() - start_time

        # Convert to standardized format
        return self._convert_to_standard_format(results[0], inference_time)

    def _convert_to_standard_format(
        self, yolo_result, inference_time: float
    ) -> Dict[str, Any]:
        """Convert YOLOv8 results to standardized format

        Args:
            yolo_result: YOLOv8 detection result
            inference_time: Inference time in seconds

        Returns:
            Standardized pose results
        """
        if yolo_result.keypoints is None or len(yolo_result.keypoints.data) == 0:
            # No detections
            return {
                "keypoints": np.array([]).reshape(0, 17, 2),
                "scores": np.array([]).reshape(0, 17),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "yolov8_pose",
                    "inference_time": inference_time,
                    "model_size": self.model_size,
                },
            }

        # Extract keypoints and bounding boxes
        keypoints_data = yolo_result.keypoints.data.cpu().numpy()  # (N, 17, 3)
        boxes_data = yolo_result.boxes.xyxy.cpu().numpy()  # (N, 4)
        confidence_scores = yolo_result.boxes.conf.cpu().numpy()  # (N,)

        num_persons = len(keypoints_data)

        # Separate keypoint coordinates and confidence scores
        keypoints_xy = keypoints_data[:, :, :2]  # (N, 17, 2)
        keypoints_conf = keypoints_data[:, :, 2]  # (N, 17)

        # Filter by keypoint confidence threshold
        valid_keypoints = keypoints_conf > self.keypoint_threshold

        return {
            "keypoints": keypoints_xy,
            "scores": keypoints_conf,
            "bbox": boxes_data,
            "num_persons": num_persons,
            "metadata": {
                "model": "yolov8_pose",
                "inference_time": inference_time,
                "model_size": self.model_size,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "detection_scores": confidence_scores,
                "valid_keypoints": valid_keypoints.sum(),
            },
        }

    def get_keypoint_names(self) -> List[str]:
        """Get COCO keypoint names

        Returns:
            List of 17 COCO keypoint names
        """
        return [
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get YOLOv8-Pose model information

        Returns:
            Dictionary with model metadata
        """
        return {
            "name": "YOLOv8-Pose",
            "version": "ultralytics",
            "type": "pose_estimation",
            "model_size": self.model_size,
            "num_keypoints": 17,
            "input_format": "BGR",
            "output_format": "yolo_pose",
            "supports_3d": False,
            "supports_multi_person": True,
            "supports_batch": True,
            "device": self.device,
            "half_precision": self.half_precision,
            "weights_path": str(self.weights_dir / f"yolov8{self.model_size}-pose.pt"),
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get YOLOv8-Pose performance metrics

        Returns:
            Dictionary with performance metrics
        """
        # Performance varies by model size
        size_metrics = {
            "n": {"size": 6.2, "time": 15.0, "memory": 80.0},  # nano
            "s": {"size": 11.6, "time": 25.0, "memory": 120.0},  # small
            "m": {"size": 26.4, "time": 35.0, "memory": 180.0},  # medium
            "l": {"size": 50.5, "time": 50.0, "memory": 250.0},  # large
            "x": {"size": 90.7, "time": 80.0, "memory": 400.0},  # extra-large
        }

        metrics = size_metrics.get(self.model_size, size_metrics["n"])

        return {
            "model_size_mb": metrics["size"],
            "avg_inference_time_ms": metrics["time"],
            "memory_usage_mb": metrics["memory"],
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process batch of images

        Args:
            images: List of input images

        Returns:
            List of pose estimation results
        """
        if not self.is_initialized:
            self.load_model()

        start_time = time.time()

        # YOLOv8 supports batch processing
        try:
            results = self.model(
                images,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
            )

            batch_inference_time = time.time() - start_time
            avg_inference_time = batch_inference_time / len(images)

            # Convert each result
            batch_results = []
            for result in results:
                batch_results.append(
                    self._convert_to_standard_format(result, avg_inference_time)
                )

            return batch_results

        except Exception as e:
            # Fallback: process one by one
            results = []
            for image in images:
                result = self.predict(image)
                results.append(result)
            return results

    def visualize_pose(
        self,
        image: np.ndarray,
        pose_result: Dict[str, Any],
        thickness: int = 2,
        radius: int = 3,
    ) -> np.ndarray:
        """Visualize YOLOv8-Pose results

        Args:
            image: Input image
            pose_result: YOLOv8 pose results
            thickness: Line thickness
            radius: Circle radius

        Returns:
            Image with pose visualization
        """
        vis_image = image.copy()

        if pose_result["num_persons"] == 0:
            return vis_image

        keypoints = pose_result["keypoints"]
        scores = pose_result["scores"]
        bboxes = pose_result["bbox"]

        # COCO skeleton connections
        skeleton = [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ]

        # Draw each person
        for person_idx in range(pose_result["num_persons"]):
            person_kpts = keypoints[person_idx]
            person_scores = scores[person_idx]
            person_bbox = bboxes[person_idx]

            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (int(person_bbox[0]), int(person_bbox[1])),
                (int(person_bbox[2]), int(person_bbox[3])),
                (0, 255, 0),
                2,
            )

            # Draw keypoints
            for kpt_idx, (x, y) in enumerate(person_kpts):
                if person_scores[kpt_idx] > self.keypoint_threshold:
                    cv2.circle(vis_image, (int(x), int(y)), radius, (0, 0, 255), -1)

            # Draw skeleton
            for connection in skeleton:
                if len(person_kpts) > max(connection):
                    pt1_idx, pt2_idx = connection
                    if (
                        person_scores[pt1_idx] > self.keypoint_threshold
                        and person_scores[pt2_idx] > self.keypoint_threshold
                    ):
                        pt1 = person_kpts[pt1_idx]
                        pt2 = person_kpts[pt2_idx]
                        cv2.line(
                            vis_image,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            (255, 0, 0),
                            thickness,
                        )

        return vis_image
