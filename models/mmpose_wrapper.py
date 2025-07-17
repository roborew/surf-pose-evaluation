"""
MMPose wrapper for pose estimation - Simplified approach based on working implementation
"""

import time
import torch
from typing import Dict, List, Any, Optional
import numpy as np
import cv2

from .base_pose_model import BasePoseModel

try:
    from mmpose.apis import MMPoseInferencer

    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False


class MMPoseWrapper(BasePoseModel):
    """MMPose wrapper using the simple, proven approach"""

    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize MMPose model

        Args:
            device: Compute device ('cpu', 'cuda', 'mps')
            **kwargs: Model configuration
        """
        if not MMPOSE_AVAILABLE:
            raise ImportError(
                "MMPose is not available. Install with: pip install mmpose"
            )

        super().__init__(device, **kwargs)

        # Configuration
        self.kpt_thr = kwargs.get("kpt_thr", 0.3)
        self.bbox_thr = kwargs.get("bbox_thr", 0.3)

        # Smart device selection with MPS override for MMPose/MMDetection compatibility
        self.requested_device = device
        if device == "cuda" and torch.cuda.is_available():
            self.mmpose_device = "cuda"
            print(f"🚀 MMPose will use CUDA (GPU acceleration)")
        elif device == "mps":
            # MMPose/MMDetection NMS operations don't support MPS yet
            # Fall back to CPU while logging the override
            self.mmpose_device = "cpu"
            print(
                "⚠️  MMPose: MPS requested but not supported (NMS operations incompatible)"
            )
            print("   → Using CPU instead. Other models will still use MPS.")
        else:
            self.mmpose_device = "cpu"
            if device not in ["cpu", "mps"]:
                print(f"⚠️  Requested device '{device}' not available, using CPU")

        self.inferencer = None
        self.load_model()

    def load_model(self) -> None:
        """Load MMPose inferencer with proper device and fallback"""
        try:
            # Primary: Use "human" preset with requested device
            print(f"Initializing MMPose with device: {self.mmpose_device}")
            self.inferencer = MMPoseInferencer(
                pose2d="human",
                det_model="rtmdet-m_640-8xb32_coco-person",  # Explicit detection
                device=self.mmpose_device,
            )
            self.is_initialized = True
            print(f"✅ MMPose initialized successfully on {self.mmpose_device}")
            return

        except Exception as primary_error:
            print(
                f"⚠️  MMPose initialization failed on {self.mmpose_device}: {primary_error}"
            )

            # Fallback: Try CPU if primary device failed
            if self.mmpose_device != "cpu":
                try:
                    print("🔄 Falling back to CPU...")
                    self.inferencer = MMPoseInferencer(
                        pose2d="human",
                        det_model="rtmdet-m_640-8xb32_coco-person",
                        device="cpu",
                    )
                    self.is_initialized = True
                    print("✅ MMPose initialized successfully on CPU (fallback)")
                    return

                except Exception as fallback_error:
                    print(f"❌ CPU fallback also failed: {fallback_error}")

            # Final error
            raise RuntimeError(
                f"Failed to initialize MMPose on both {self.mmpose_device} and CPU. "
                f"Primary error: {primary_error}"
            )

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

        try:
            # Use the simple inference approach that works
            results = list(self.inferencer(image))

            inference_time = time.time() - start_time

            # Convert to standardized format
            return self._convert_to_standard_format(results, inference_time)

        except Exception as e:
            inference_time = time.time() - start_time
            print(f"MMPose inference error: {e}")

            # Return empty result on error
            return {
                "keypoints": np.array([]).reshape(0, 17, 2),
                "scores": np.array([]).reshape(0, 17),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "mmpose",
                    "inference_time": inference_time,
                    "error": str(e),
                },
            }

    def _convert_to_standard_format(
        self, results: List, inference_time: float
    ) -> Dict[str, Any]:
        """Convert MMPose results to standardized format

        Args:
            results: MMPose inference results (list from generator)
            inference_time: Inference time in seconds

        Returns:
            Standardized pose results
        """
        try:
            if not results or len(results) == 0:
                return {
                    "keypoints": np.array([]).reshape(0, 17, 2),
                    "scores": np.array([]).reshape(0, 17),
                    "bbox": np.array([]).reshape(0, 4),
                    "num_persons": 0,
                    "metadata": {
                        "model": "mmpose",
                        "inference_time": inference_time,
                        "model_name": "human",
                    },
                }

            # Debug: Print the result structure to understand the format
            result = results[0] if isinstance(results, list) else results
            print(f"MMPose result type: {type(result)}")
            if hasattr(result, "__dict__"):
                print(f"MMPose result attributes: {list(result.__dict__.keys())}")
            elif isinstance(result, dict):
                print(f"MMPose result keys: {list(result.keys())}")

            # Try different ways to access the pose data
            keypoints = None
            keypoint_scores = None
            bboxes = None

            # Method 1: Direct access to pred_instances
            if hasattr(result, "pred_instances"):
                pred_instances = result.pred_instances
                if hasattr(pred_instances, "keypoints"):
                    keypoints = pred_instances.keypoints.cpu().numpy()
                    keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()
                    if hasattr(pred_instances, "bboxes"):
                        bboxes = pred_instances.bboxes.cpu().numpy()

            # Method 2: Dictionary access
            elif isinstance(result, dict):
                if "predictions" in result:
                    predictions = result["predictions"]
                    print(
                        f"Predictions type: {type(predictions)}, length: {len(predictions) if hasattr(predictions, '__len__') else 'N/A'}"
                    )

                    if isinstance(predictions, list) and len(predictions) > 0:
                        # predictions is a list of results, take the first one
                        prediction = predictions[0]
                        print(f"First prediction type: {type(prediction)}")

                        if hasattr(prediction, "pred_instances"):
                            pred_instances = prediction.pred_instances
                            keypoints = pred_instances.keypoints.cpu().numpy()
                            keypoint_scores = (
                                pred_instances.keypoint_scores.cpu().numpy()
                            )
                            if hasattr(pred_instances, "bboxes"):
                                bboxes = pred_instances.bboxes.cpu().numpy()
                        elif isinstance(prediction, dict):
                            # Handle case where prediction itself is a dict
                            print(
                                f"Prediction dict keys: {list(prediction.keys()) if isinstance(prediction, dict) else 'Not a dict'}"
                            )
                            if (
                                "keypoints" in prediction
                                and "keypoint_scores" in prediction
                            ):
                                keypoints = prediction["keypoints"]
                                keypoint_scores = prediction["keypoint_scores"]
                                bboxes = prediction.get(
                                    "bboxes", np.array([]).reshape(0, 4)
                                )

                                # Convert to numpy if they're tensors
                                if hasattr(keypoints, "cpu"):
                                    keypoints = keypoints.cpu().numpy()
                                if hasattr(keypoint_scores, "cpu"):
                                    keypoint_scores = keypoint_scores.cpu().numpy()
                                if hasattr(bboxes, "cpu"):
                                    bboxes = bboxes.cpu().numpy()
                        elif (
                            isinstance(prediction, list)
                            and len(prediction) > 0
                            and isinstance(prediction[0], dict)
                        ):
                            # Handle list of detection dictionaries - this is the correct format!
                            print(
                                f"Found list of {len(prediction)} detection dictionaries"
                            )
                            all_keypoints = []
                            all_scores = []
                            all_bboxes = []

                            for person_detection in prediction:
                                if (
                                    "keypoints" in person_detection
                                    and "keypoint_scores" in person_detection
                                ):
                                    kpts = person_detection["keypoints"]
                                    scores = person_detection["keypoint_scores"]
                                    bbox = person_detection.get("bbox", [0, 0, 0, 0])

                                    # Convert to numpy if they're tensors
                                    if hasattr(kpts, "cpu"):
                                        kpts = kpts.cpu().numpy()
                                    if hasattr(scores, "cpu"):
                                        scores = scores.cpu().numpy()
                                    if hasattr(bbox, "cpu"):
                                        bbox = bbox.cpu().numpy()
                                    elif isinstance(bbox, list):
                                        bbox = np.array(bbox)

                                    all_keypoints.append(kpts)
                                    all_scores.append(scores)
                                    all_bboxes.append(bbox)

                            if all_keypoints:
                                keypoints = np.array(all_keypoints)
                                keypoint_scores = np.array(all_scores)
                                bboxes = np.array(all_bboxes)
                                print(
                                    f"Successfully parsed {len(keypoints)} persons with keypoints shape {keypoints.shape}"
                                )
                        else:
                            # Fallback - unknown prediction format
                            print(f"Unknown prediction format: {type(prediction)}")
                            if hasattr(prediction, "__len__"):
                                print(f"Length: {len(prediction)}")
                            if (
                                isinstance(prediction, (list, tuple))
                                and len(prediction) > 0
                            ):
                                print(f"First element: {type(prediction[0])}")
                elif "instances" in result:
                    # Handle the format from your working code
                    instances = result["instances"]
                    return self._convert_from_instances_format(
                        instances, inference_time
                    )

            # Method 3: The result might be a direct list/array structure
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                # Handle cases where result is directly the pose data
                pose_data = (
                    result[0] if isinstance(result[0], (list, tuple)) else result
                )
                print(f"Direct pose data type: {type(pose_data)}")
                # This needs more investigation of the actual structure

            if keypoints is None:
                # If we couldn't extract keypoints, return empty result
                print("Could not extract keypoints from MMPose result")
                return {
                    "keypoints": np.array([]).reshape(0, 17, 2),
                    "scores": np.array([]).reshape(0, 17),
                    "bbox": np.array([]).reshape(0, 4),
                    "num_persons": 0,
                    "metadata": {
                        "model": "mmpose",
                        "inference_time": inference_time,
                        "model_name": "human",
                        "error": "Could not parse result structure",
                    },
                }

            # Set default bboxes if not extracted
            if bboxes is None:
                bboxes = np.array([]).reshape(0, 4)

            # Ensure COCO-17 format (17 keypoints)
            if keypoints.shape[1] != 17:
                # If different number of keypoints, pad or truncate to 17
                target_keypoints = np.zeros((keypoints.shape[0], 17, 2))
                target_scores = np.zeros((keypoints.shape[0], 17))

                min_kpts = min(keypoints.shape[1], 17)
                target_keypoints[:, :min_kpts] = keypoints[:, :min_kpts]
                target_scores[:, :min_kpts] = keypoint_scores[:, :min_kpts]

                keypoints = target_keypoints
                keypoint_scores = target_scores

            return {
                "keypoints": keypoints,
                "scores": keypoint_scores,
                "bbox": bboxes,
                "num_persons": len(keypoints),
                "metadata": {
                    "model": "mmpose",
                    "inference_time": inference_time,
                    "model_name": "human",
                },
            }

        except Exception as e:
            print(f"Error converting MMPose results: {e}")
            import traceback

            traceback.print_exc()
            return {
                "keypoints": np.array([]).reshape(0, 17, 2),
                "scores": np.array([]).reshape(0, 17),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "mmpose",
                    "inference_time": inference_time,
                    "error": f"Conversion error: {e}",
                },
            }

    def _convert_from_instances_format(
        self, instances: List, inference_time: float
    ) -> Dict[str, Any]:
        """Convert from the instances format used in your working code"""
        try:
            if not instances:
                return {
                    "keypoints": np.array([]).reshape(0, 17, 2),
                    "scores": np.array([]).reshape(0, 17),
                    "bbox": np.array([]).reshape(0, 4),
                    "num_persons": 0,
                    "metadata": {
                        "model": "mmpose",
                        "inference_time": inference_time,
                        "model_name": "human",
                    },
                }

            all_keypoints = []
            all_scores = []
            all_bboxes = []

            for person in instances:
                if isinstance(person, dict):
                    # Extract keypoints
                    keypoints = person.get("keypoints", [])
                    keypoint_scores = person.get("keypoint_scores", [])

                    if keypoints and keypoint_scores:
                        # Convert to numpy arrays
                        kpts = np.array(keypoints)
                        scores = np.array(keypoint_scores)

                        # Ensure we have 17 keypoints
                        if len(kpts) >= 17:
                            all_keypoints.append(kpts[:17])
                            all_scores.append(scores[:17])
                        else:
                            # Pad with zeros if less than 17
                            padded_kpts = np.zeros((17, 2))
                            padded_scores = np.zeros(17)
                            padded_kpts[: len(kpts)] = kpts
                            padded_scores[: len(scores)] = scores
                            all_keypoints.append(padded_kpts)
                            all_scores.append(padded_scores)

                    # Extract bbox
                    bbox = person.get("bbox", [[0, 0, 0, 0]])
                    if isinstance(bbox, list) and len(bbox) > 0:
                        all_bboxes.append(bbox[0][:4])
                    else:
                        all_bboxes.append([0, 0, 0, 0])

            if all_keypoints:
                keypoints = np.array(all_keypoints)
                scores = np.array(all_scores)
                bboxes = np.array(all_bboxes)
            else:
                keypoints = np.array([]).reshape(0, 17, 2)
                scores = np.array([]).reshape(0, 17)
                bboxes = np.array([]).reshape(0, 4)

            return {
                "keypoints": keypoints,
                "scores": scores,
                "bbox": bboxes,
                "num_persons": len(keypoints),
                "metadata": {
                    "model": "mmpose",
                    "inference_time": inference_time,
                    "model_name": "human",
                },
            }

        except Exception as e:
            print(f"Error converting instances format: {e}")
            return {
                "keypoints": np.array([]).reshape(0, 17, 2),
                "scores": np.array([]).reshape(0, 17),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "mmpose",
                    "inference_time": inference_time,
                    "error": f"Instance conversion error: {e}",
                },
            }

    def get_keypoint_names(self) -> List[str]:
        """Get COCO keypoint names"""
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
        """Get model information"""
        return {
            "name": "mmpose",
            "version": "1.1.0",
            "model_name": "human",
            "keypoints": 17,
            "format": "COCO",
            "device": self.mmpose_device,
            "bbox_threshold": self.bbox_thr,
            "keypoint_threshold": self.kpt_thr,
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get basic performance metrics"""
        return {
            "fps": getattr(self, "_last_fps", 0.0),
            "latency_ms": getattr(self, "_last_latency_ms", 0.0),
            "memory_mb": getattr(self, "_last_memory_mb", 0.0),
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process a batch of images"""
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
