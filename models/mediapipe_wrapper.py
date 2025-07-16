"""
MediaPipe Pose estimation wrapper
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import mediapipe as mp
import logging

from .base_pose_model import BasePoseModel


class MediaPipeWrapper(BasePoseModel):
    """MediaPipe Pose estimation wrapper with robust initialization for macOS"""

    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize MediaPipe pose estimation

        Args:
            device: Compute device (MediaPipe runs on CPU)
            **kwargs: Additional configuration options
        """
        super().__init__(device=device, **kwargs)

        # Model configuration with surf-optimized settings
        self.model_complexity = kwargs.get("model_complexity", 1)  # Use balanced model
        self.enable_segmentation = kwargs.get("enable_segmentation", False)
        self.smooth_landmarks = kwargs.get("smooth_landmarks", True)
        self.min_detection_confidence = kwargs.get(
            "min_detection_confidence", 0.3
        )  # Lowered for surf footage
        self.min_tracking_confidence = kwargs.get(
            "min_tracking_confidence", 0.3
        )  # Lowered for surf footage

        # Surf-specific optimizations
        self.static_image_mode = kwargs.get(
            "static_image_mode", False
        )  # Better for video sequences

        # Initialize MediaPipe components
        self.mp_pose = None
        self.mp_drawing = None
        self.model = None

        # macOS compatibility: Force model_complexity=0 if higher values are problematic
        import platform

        if platform.system().lower() == "darwin":  # macOS
            if self.model_complexity > 0:
                logging.warning(
                    f"MediaPipe model_complexity={self.model_complexity} may be unstable on macOS, will try fallback to 0 if needed"
                )
                self.original_model_complexity = (
                    self.model_complexity
                )  # Store original for fallback

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.load_model()

    def load_model(self) -> None:
        """Load MediaPipe pose estimation model with optimized settings"""
        if self.is_initialized:
            return

        try:
            logging.info("Initializing MediaPipe Pose with surf-optimized settings...")
            import mediapipe as mp

            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils

            # Create pose model with surf-optimized settings
            self.model = self.mp_pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

            self.is_initialized = True
            logging.info(
                f"MediaPipe initialized successfully (complexity={self.model_complexity}, "
                f"detection_confidence={self.min_detection_confidence}, "
                f"tracking_confidence={self.min_tracking_confidence})"
            )

        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe: {e}")
            # Try fallback initialization
            if not self._try_fallback_initialization():
                raise RuntimeError(f"Failed to initialize MediaPipe pose model: {e}")

    def _create_pose_model(self):
        """Create MediaPipe Pose model with current configuration"""
        import platform

        # Try original configuration first
        try:
            return self.mp_pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception as e:
            # On macOS, if model_complexity > 0 fails, try with complexity=0
            if platform.system().lower() == "darwin" and self.model_complexity > 0:
                logging.warning(
                    f"MediaPipe model_complexity={self.model_complexity} failed on macOS: {e}"
                )
                logging.info(
                    "Falling back to model_complexity=0 for macOS compatibility"
                )
                self.model_complexity = 0  # Update to working value
                return self.mp_pose.Pose(
                    static_image_mode=self.static_image_mode,
                    model_complexity=0,  # Force to 0 for macOS
                    smooth_landmarks=self.smooth_landmarks,
                    enable_segmentation=self.enable_segmentation,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
            else:
                # Re-raise the exception if not macOS or complexity is already 0
                raise

    def _try_fallback_initialization(self) -> bool:
        """Try fallback initialization strategies for better macOS compatibility"""
        import os

        fallback_configs = [
            # Strategy 1: Ultra conservative - Lite model, static mode, high thresholds
            {
                "static_image_mode": True,
                "model_complexity": 0,  # Lite model
                "smooth_landmarks": False,
                "enable_segmentation": False,
                "min_detection_confidence": 0.8,
                "min_tracking_confidence": 0.8,
            },
            # Strategy 2: Simplest possible configuration with lower thresholds
            {
                "static_image_mode": True,
                "model_complexity": 0,
                "smooth_landmarks": False,
                "enable_segmentation": False,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            # Strategy 3: Default MediaPipe configuration (minimal parameters)
            {
                "static_image_mode": False,
                "model_complexity": 1,
            },
            # Strategy 4: Absolute minimal configuration
            {},
        ]

        for i, config in enumerate(fallback_configs):
            try:
                logging.info(f"Trying fallback initialization strategy {i+1}: {config}")

                # Set additional environment variables for each attempt
                os.environ["TF_DISABLE_MKL"] = "1" if i > 1 else "0"
                os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = (
                    "1"
                )

                # Update configuration
                for key, value in config.items():
                    setattr(self, key, value)

                # Try to create model with fallback config
                if config:  # If config is not empty
                    self.model = self.mp_pose.Pose(**config)
                else:  # Use default configuration
                    self.model = self.mp_pose.Pose()

                # Test the model with a dummy prediction to ensure it really works
                dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
                dummy_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
                test_results = self.model.process(dummy_rgb)

                self.is_initialized = True
                logging.info(f"Fallback strategy {i+1} successful and verified")
                return True

            except Exception as e:
                logging.warning(f"Fallback strategy {i+1} failed: {e}")
                # Clean up any partial initialization
                if hasattr(self, "model") and self.model:
                    try:
                        self.model.close()
                    except:
                        pass
                    self.model = None
                continue

        # If all fallback strategies fail, try one last desperate attempt with imported pose
        try:
            logging.info("Attempting emergency fallback with fresh import")
            import importlib

            importlib.reload(mp.solutions.pose)
            self.mp_pose = mp.solutions.pose

            self.model = self.mp_pose.Pose(
                static_image_mode=True, model_complexity=0, enable_segmentation=False
            )
            self.is_initialized = True
            logging.info("Emergency fallback successful")
            return True
        except Exception as e:
            logging.error(f"Emergency fallback also failed: {e}")
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run MediaPipe pose estimation on image

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            Standardized pose estimation results
        """
        if not self.is_initialized:
            self.load_model()

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run pose estimation
        start_time = time.time()
        try:
            results = self.model.process(rgb_image)
            inference_time = time.time() - start_time
        except Exception as e:
            logging.error(f"MediaPipe inference failed: {e}")
            inference_time = time.time() - start_time

            # Try fallback with model_complexity=0 if on macOS and complexity > 0
            import platform

            if platform.system().lower() == "darwin" and self.model_complexity > 0:
                try:
                    logging.warning(
                        f"Attempting fallback with model_complexity=0 due to inference failure"
                    )
                    # Close the failing model
                    if hasattr(self, "model") and self.model:
                        self.model.close()

                    # Reinitialize with complexity=0
                    self.model_complexity = 0
                    self.model = self._create_pose_model()

                    # Retry inference
                    start_time = time.time()  # Reset timer
                    results = self.model.process(rgb_image)
                    inference_time = time.time() - start_time

                    logging.info(f"Fallback to model_complexity=0 successful")

                except Exception as e2:
                    logging.error(f"Fallback inference also failed: {e2}")
                    # Return empty results on failure
                    return self._get_empty_results(image.shape, inference_time)
            else:
                # Return empty results on failure
                return self._get_empty_results(image.shape, inference_time)

        # Convert to standardized format
        return self._convert_to_standard_format(results, image.shape, inference_time)

    def _get_empty_results(
        self, image_shape: tuple, inference_time: float
    ) -> Dict[str, Any]:
        """Return empty results when inference fails"""
        return {
            "keypoints": np.array([]).reshape(0, 33, 3),
            "scores": np.array([]).reshape(0, 33),
            "bbox": np.array([]).reshape(0, 4),
            "num_persons": 0,
            "metadata": {
                "model": "mediapipe",
                "inference_time": inference_time,
                "model_complexity": self.model_complexity,
                "error": "inference_failed",
            },
        }

    def _convert_to_standard_format(
        self, mp_results, image_shape: tuple, inference_time: float
    ) -> Dict[str, Any]:
        """Convert MediaPipe results to standardized format

        Args:
            mp_results: MediaPipe pose results
            image_shape: Original image shape (H, W, C)
            inference_time: Inference time in seconds

        Returns:
            Standardized pose results
        """
        h, w = image_shape[:2]

        if mp_results.pose_landmarks is None:
            # No pose detected
            return {
                "keypoints": np.array([]).reshape(0, 33, 3),
                "scores": np.array([]).reshape(0, 33),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "mediapipe",
                    "inference_time": inference_time,
                    "model_complexity": self.model_complexity,
                },
            }

        # Extract landmarks
        landmarks = mp_results.pose_landmarks.landmark

        # Convert normalized coordinates to pixel coordinates
        keypoints = []
        scores = []

        for landmark in landmarks:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z  # Relative depth
            visibility = landmark.visibility

            keypoints.append([x, y, z])
            scores.append(visibility)

        keypoints = np.array(keypoints).reshape(1, 33, 3)  # MediaPipe has 33 keypoints
        scores = np.array(scores).reshape(1, 33)

        # Calculate bounding box from keypoints
        visible_points = keypoints[0][scores[0] > 0.5][:, :2]
        if len(visible_points) > 0:
            x_min, y_min = visible_points.min(axis=0)
            x_max, y_max = visible_points.max(axis=0)
            bbox = np.array([[x_min, y_min, x_max, y_max]])
        else:
            bbox = np.array([[0, 0, w, h]])

        return {
            "keypoints": keypoints,
            "scores": scores,
            "bbox": bbox,
            "num_persons": 1,
            "metadata": {
                "model": "mediapipe",
                "inference_time": inference_time,
                "model_complexity": self.model_complexity,
                "segmentation_available": mp_results.segmentation_mask is not None,
            },
        }

    def get_keypoint_names(self) -> List[str]:
        """Get MediaPipe keypoint names in order

        Returns:
            List of 33 MediaPipe keypoint names
        """
        return [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get MediaPipe model information

        Returns:
            Dictionary with model metadata
        """
        return {
            "name": "MediaPipe Pose",
            "version": mp.__version__,
            "type": "pose_estimation",
            "num_keypoints": 33,
            "model_complexity": self.model_complexity,
            "input_format": "RGB",
            "output_format": "normalized_landmarks",
            "supports_3d": True,
            "supports_segmentation": self.enable_segmentation,
            "device": "cpu",  # MediaPipe runs on CPU
        }

    def visualize_pose(
        self,
        image: np.ndarray,
        pose_result: Dict[str, Any],
        thickness: int = 2,
        radius: int = 3,
    ) -> np.ndarray:
        """Visualize MediaPipe pose results

        Args:
            image: Input image
            pose_result: MediaPipe pose results
            thickness: Line thickness
            radius: Circle radius

        Returns:
            Image with pose visualization
        """
        vis_image = image.copy()

        if pose_result["num_persons"] == 0:
            return vis_image

        # Convert to MediaPipe format for visualization
        keypoints = pose_result["keypoints"][0]  # First person
        scores = pose_result["scores"][0]

        # Create MediaPipe landmark list
        landmarks = []
        h, w = image.shape[:2]

        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            landmark = mp.solutions.pose.PoseLandmark
            x_norm = kpt[0] / w
            y_norm = kpt[1] / h
            z_norm = kpt[2]

            # Create landmark object
            lm = type("Landmark", (), {})()
            lm.x = x_norm
            lm.y = y_norm
            lm.z = z_norm
            lm.visibility = score
            landmarks.append(lm)

        # Create pose landmarks object
        pose_landmarks = type("PoseLandmarks", (), {})()
        pose_landmarks.landmark = landmarks

        # Draw landmarks and connections
        self.mp_drawing.draw_landmarks(
            vis_image,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        return vis_image

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get MediaPipe performance metrics

        Returns:
            Dictionary with performance metrics
        """
        # MediaPipe is lightweight
        model_size = {
            0: 3.0,  # Lite model
            1: 5.0,  # Full model
            2: 8.0,  # Heavy model
        }.get(self.model_complexity, 5.0)

        return {
            "model_size_mb": model_size,
            "avg_inference_time_ms": 15.0,  # Typical on CPU
            "memory_usage_mb": 50.0,
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process batch of images

        MediaPipe processes images individually for optimal performance

        Args:
            images: List of input images

        Returns:
            List of pose estimation results
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results

    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, "model") and self.model:
            self.model.close()
