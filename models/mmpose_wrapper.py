"""
MMPose wrapper for pose estimation
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import torch

from .base_pose_model import BasePoseModel

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
    from mmdet.apis import inference_detector, init_detector

    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False


class MMPoseWrapper(BasePoseModel):
    """MMPose wrapper for pose estimation"""

    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize MMPose model

        Args:
            device: Compute device ('cpu', 'cuda')
            **kwargs: Model configuration
        """
        if not MMPOSE_AVAILABLE:
            raise ImportError(
                "MMPose is not available. Install with: pip install mmpose"
            )

        super().__init__(device, **kwargs)

        # Model configuration
        self.pose_config = kwargs.get(
            "pose_config", "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
        )
        self.pose_checkpoint = kwargs.get(
            "pose_checkpoint",
            "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth",
        )
        self.det_config = kwargs.get("det_config", "yolox_s_8x8_300e_coco.py")
        self.det_checkpoint = kwargs.get(
            "det_checkpoint", "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
        )

        # Detection settings
        self.det_cat_id = kwargs.get("det_cat_id", 1)  # person category
        self.bbox_thr = kwargs.get("bbox_thr", 0.3)
        self.kpt_thr = kwargs.get("kpt_thr", 0.3)

        self.pose_model = None
        self.det_model = None

        self.load_model()

    def load_model(self) -> None:
        """Load MMPose and detection models"""
        try:
            # Initialize detection model
            self.det_model = init_detector(
                self.det_config, self.det_checkpoint, device=self.device
            )

            # Initialize pose model
            self.pose_model = init_pose_model(
                self.pose_config, self.pose_checkpoint, device=self.device
            )

            self.is_initialized = True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize MMPose models: {e}")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run MMPose estimation on image

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            Standardized pose estimation results
        """
        if not self.is_initialized:
            self.load_model()

        start_time = time.time()

        # Run person detection
        det_results = inference_detector(self.det_model, image)

        # Extract person bboxes
        person_results = self._extract_person_bboxes(det_results)

        if len(person_results) == 0:
            # No persons detected
            inference_time = time.time() - start_time
            return {
                "keypoints": np.array([]).reshape(0, 17, 2),
                "scores": np.array([]).reshape(0, 17),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "mmpose",
                    "inference_time": inference_time,
                    "pose_config": self.pose_config,
                },
            }

        # Run pose estimation
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            image,
            person_results,
            bbox_thr=self.bbox_thr,
            format="xyxy",
            dataset=self.pose_model.cfg.test_dataloader.dataset.type,
            dataset_info=None,
            return_heatmap=False,
            outputs=None,
        )

        inference_time = time.time() - start_time

        # Convert to standardized format
        return self._convert_to_standard_format(
            pose_results, person_results, inference_time
        )

    def _extract_person_bboxes(self, det_results) -> List:
        """Extract person bounding boxes from detection results

        Args:
            det_results: Detection results from MMDetection

        Returns:
            List of person bounding boxes
        """
        person_results = []

        # Handle different MMDetection result formats
        if hasattr(det_results, "pred_instances"):
            # New format
            bboxes = det_results.pred_instances.bboxes.cpu().numpy()
            scores = det_results.pred_instances.scores.cpu().numpy()
            labels = det_results.pred_instances.labels.cpu().numpy()

            # Filter for person class (label 0 in COCO)
            person_mask = labels == 0
            person_bboxes = bboxes[person_mask]
            person_scores = scores[person_mask]

            for bbox, score in zip(person_bboxes, person_scores):
                if score > self.bbox_thr:
                    person_results.append({"bbox": bbox, "score": score})
        else:
            # Legacy format
            if isinstance(det_results, tuple):
                det_results = det_results[0]

            if len(det_results) > 0 and len(det_results[0]) > 0:
                # Person class is typically index 0
                person_dets = det_results[0]
                for det in person_dets:
                    if len(det) >= 5 and det[4] > self.bbox_thr:
                        person_results.append({"bbox": det[:4], "score": det[4]})

        return person_results

    def _convert_to_standard_format(
        self, pose_results: List, person_results: List, inference_time: float
    ) -> Dict[str, Any]:
        """Convert MMPose results to standardized format

        Args:
            pose_results: MMPose pose estimation results
            person_results: Person detection results
            inference_time: Inference time in seconds

        Returns:
            Standardized pose results
        """
        if not pose_results:
            return {
                "keypoints": np.array([]).reshape(0, 17, 2),
                "scores": np.array([]).reshape(0, 17),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "mmpose",
                    "inference_time": inference_time,
                    "pose_config": self.pose_config,
                },
            }

        num_persons = len(pose_results)
        num_keypoints = len(pose_results[0]["keypoints"])

        # Extract keypoints and scores
        all_keypoints = []
        all_scores = []
        all_bboxes = []

        for i, pose_result in enumerate(pose_results):
            keypoints = pose_result["keypoints"]  # Shape: (17, 3) for COCO format

            # Split into coordinates and scores
            kpts_xy = keypoints[:, :2]  # (17, 2)
            kpts_scores = keypoints[:, 2]  # (17,)

            all_keypoints.append(kpts_xy)
            all_scores.append(kpts_scores)

            # Get corresponding bbox
            if i < len(person_results):
                bbox = person_results[i]["bbox"]
                all_bboxes.append(bbox)
            else:
                # Fallback: calculate bbox from keypoints
                valid_kpts = kpts_xy[kpts_scores > self.kpt_thr]
                if len(valid_kpts) > 0:
                    x_min, y_min = valid_kpts.min(axis=0)
                    x_max, y_max = valid_kpts.max(axis=0)
                    all_bboxes.append([x_min, y_min, x_max, y_max])
                else:
                    all_bboxes.append([0, 0, 100, 100])  # Default bbox

        keypoints = np.array(all_keypoints)  # (N, 17, 2)
        scores = np.array(all_scores)  # (N, 17)
        bboxes = np.array(all_bboxes)  # (N, 4)

        return {
            "keypoints": keypoints,
            "scores": scores,
            "bbox": bboxes,
            "num_persons": num_persons,
            "metadata": {
                "model": "mmpose",
                "inference_time": inference_time,
                "pose_config": self.pose_config,
                "det_config": self.det_config,
                "bbox_threshold": self.bbox_thr,
                "keypoint_threshold": self.kpt_thr,
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
        """Get MMPose model information

        Returns:
            Dictionary with model metadata
        """
        return {
            "name": "MMPose",
            "version": "1.1.0",  # Update based on installed version
            "type": "pose_estimation",
            "num_keypoints": 17,
            "pose_config": self.pose_config,
            "det_config": self.det_config,
            "input_format": "BGR",
            "output_format": "coco_keypoints",
            "supports_3d": False,
            "supports_multi_person": True,
            "device": self.device,
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get MMPose performance metrics

        Returns:
            Dictionary with performance metrics
        """
        # Performance varies by model
        model_sizes = {"hrnet": 60.0, "resnet": 45.0, "mobilenet": 15.0}

        # Estimate based on config name
        model_size = 45.0  # Default
        for model_type, size in model_sizes.items():
            if model_type in self.pose_config.lower():
                model_size = size
                break

        return {
            "model_size_mb": model_size,
            "avg_inference_time_ms": 50.0,  # Typical on GPU
            "memory_usage_mb": 200.0,
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process batch of images

        Args:
            images: List of input images

        Returns:
            List of pose estimation results
        """
        # MMPose can handle batches more efficiently
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
