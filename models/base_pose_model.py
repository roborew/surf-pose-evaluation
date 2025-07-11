"""
Abstract base class for pose estimation models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import cv2


class BasePoseModel(ABC):
    """Abstract base class for pose estimation models"""

    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize pose model

        Args:
            device: Compute device ('cpu', 'cuda', 'mps')
            **kwargs: Model-specific configuration
        """
        self.device = device
        self.model_config = kwargs
        self.model = None
        self.is_initialized = False

    @abstractmethod
    def load_model(self) -> None:
        """Load and initialize the pose estimation model"""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run pose estimation on a single image

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            Dictionary containing pose estimation results with standardized format:
            {
                'keypoints': np.ndarray,  # (N, K, 2) or (N, K, 3) for 2D/3D
                'scores': np.ndarray,     # (N, K) confidence scores
                'bbox': np.ndarray,       # (N, 4) bounding boxes if available
                'num_persons': int,       # Number of detected persons
                'metadata': dict          # Model-specific metadata
            }
        """
        pass

    @abstractmethod
    def get_keypoint_names(self) -> List[str]:
        """Get list of keypoint names in order

        Returns:
            List of keypoint names (e.g., ['nose', 'left_eye', ...])
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information

        Returns:
            Dictionary with model metadata (name, version, input_size, etc.)
        """
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # Default preprocessing - subclasses can override
        return image

    def postprocess_results(self, raw_results: Any) -> Dict[str, Any]:
        """Postprocess raw model results to standardized format

        Args:
            raw_results: Raw model output

        Returns:
            Standardized pose estimation results
        """
        # Default postprocessing - subclasses must override
        return raw_results

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Run pose estimation on batch of images

        Args:
            images: List of input images

        Returns:
            List of pose estimation results
        """
        # Default implementation - process one by one
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
        """Visualize pose estimation results on image

        Args:
            image: Input image
            pose_result: Pose estimation results
            thickness: Line thickness for skeleton
            radius: Circle radius for keypoints

        Returns:
            Image with pose visualization
        """
        vis_image = image.copy()

        if "keypoints" not in pose_result:
            return vis_image

        keypoints = pose_result["keypoints"]
        scores = pose_result.get("scores", None)

        # Draw keypoints and skeleton
        for person_idx in range(keypoints.shape[0]):
            person_kpts = keypoints[person_idx]
            person_scores = scores[person_idx] if scores is not None else None

            # Draw keypoints
            for kpt_idx, (x, y) in enumerate(person_kpts[:, :2]):
                if person_scores is None or person_scores[kpt_idx] > 0.3:
                    cv2.circle(vis_image, (int(x), int(y)), radius, (0, 255, 0), -1)

            # Draw skeleton (basic connections)
            self._draw_skeleton(vis_image, person_kpts, person_scores, thickness)

        return vis_image

    def _draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: Optional[np.ndarray] = None,
        thickness: int = 2,
    ):
        """Draw skeleton connections on image

        Args:
            image: Image to draw on
            keypoints: Keypoints array (K, 2) or (K, 3)
            scores: Confidence scores (K,)
            thickness: Line thickness
        """
        # Basic skeleton connections (COCO format)
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

        for connection in skeleton:
            if len(keypoints) > max(connection):
                pt1 = keypoints[connection[0]][:2]
                pt2 = keypoints[connection[1]][:2]

                # Check confidence scores if available
                if scores is not None:
                    if scores[connection[0]] < 0.3 or scores[connection[1]] < 0.3:
                        continue

                cv2.line(
                    image,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    (255, 0, 0),
                    thickness,
                )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get model performance metrics

        Returns:
            Dictionary with performance metrics (latency, memory, etc.)
        """
        return {
            "model_size_mb": 0.0,
            "avg_inference_time_ms": 0.0,
            "memory_usage_mb": 0.0,
        }

    def __str__(self) -> str:
        """String representation of model"""
        info = self.get_model_info()
        return f"{info.get('name', 'Unknown')} ({info.get('version', 'Unknown')})"

    def __repr__(self) -> str:
        """Detailed representation of model"""
        return f"{self.__class__.__name__}(device='{self.device}', config={self.model_config})"
