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
        """Initialize MediaPipe pose model

        Args:
            device: Compute device (MediaPipe runs on CPU)
            **kwargs: Model configuration
        """
        super().__init__(device, **kwargs)

        # MediaPipe configuration
        self.static_image_mode = kwargs.get("static_image_mode", False)
        self.model_complexity = kwargs.get("model_complexity", 1)
        self.smooth_landmarks = kwargs.get("smooth_landmarks", True)
        self.enable_segmentation = kwargs.get("enable_segmentation", False)
        self.min_detection_confidence = kwargs.get("min_detection_confidence", 0.5)
        self.min_tracking_confidence = kwargs.get("min_tracking_confidence", 0.5)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.load_model()

    def load_model(self) -> None:
        """Load and initialize MediaPipe Pose model with CUDA-first approach"""
        import os
        import platform

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
        
        # CUDA-first: Enable GPU on Linux, CPU fallback on macOS for stability
        if platform.system().lower() == "darwin":  # macOS
            os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # CPU for macOS stability
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce logging
            logging.info("MediaPipe using CPU on macOS for stability")
        else:  # Linux/Windows - Enable GPU acceleration
            os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"  # Enable GPU
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth
            logging.info("MediaPipe enabling GPU acceleration for production")

        try:
            # First try with the configured parameters
            self.model = self._create_pose_model()
            self.is_initialized = True
            logging.info("MediaPipe Pose initialized successfully")
        except Exception as e:
            logging.warning(f"Initial MediaPipe initialization failed: {e}")
            # Try fallback initialization strategies
            if self._try_fallback_initialization():
                logging.info("MediaPipe Pose initialized with fallback configuration")
            else:
                raise RuntimeError(
                    f"Failed to initialize MediaPipe Pose after trying all fallback options: {e}"
                )

    def _create_pose_model(self):
        """Create MediaPipe Pose model with current configuration"""
        return self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

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
