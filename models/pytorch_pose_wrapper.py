"""
PyTorch-based pose estimation wrapper as MediaPipe alternative
Uses torchvision's KeypointRCNN models for robust Apple Silicon compatibility
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import torchvision.transforms as transforms

from .base_pose_model import BasePoseModel


class PyTorchPoseWrapper(BasePoseModel):
    """PyTorch-based pose estimation wrapper using torchvision KeypointRCNN"""

    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize PyTorch pose model

        Args:
            device: Compute device ('cpu', 'cuda', 'mps')
            **kwargs: Model configuration
        """
        super().__init__(device, **kwargs)

        # Model configuration
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.7)
        self.keypoint_threshold = kwargs.get("keypoint_threshold", 0.3)
        self.nms_threshold = kwargs.get("nms_threshold", 0.3)
        self.max_detections = kwargs.get("max_detections", 10)

        # Input preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Load PyTorch KeypointRCNN model with robust download handling"""
        import logging
        from pathlib import Path

        try:
            # Use weights parameter instead of deprecated pretrained
            from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

            weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1

            # Try loading with retry logic for corrupted downloads
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(
                            f"Retrying PyTorch model download (attempt {attempt + 1}/{max_retries})"
                        )
                        # Clear corrupted cache
                        self._clear_pytorch_cache()

                    # Load model with modern weights API
                    self.model = keypointrcnn_resnet50_fpn(weights=weights)
                    self.model.eval()

                    # Move to device (MPS support on Apple Silicon)
                    self.model.to(self.device)

                    # Test model with dummy input to ensure it works
                    dummy_input = torch.rand(1, 3, 480, 640).to(self.device)
                    with torch.no_grad():
                        test_output = self.model(dummy_input)
                        # Verify output structure
                        if len(test_output) > 0 and "keypoints" in test_output[0]:
                            break

                except Exception as e:
                    if (
                        "corrupted" in str(e).lower()
                        or "unexpected eof" in str(e).lower()
                    ):
                        print(
                            f"Download attempt {attempt + 1} failed with corruption: {e}"
                        )
                        if attempt == max_retries - 1:
                            # Last attempt - try manual download
                            return self._try_manual_download()
                        continue
                    else:
                        raise e

            self.is_initialized = True
            print(f"PyTorch KeypointRCNN loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyTorch pose model: {e}")

    def _clear_pytorch_cache(self):
        """Clear corrupted PyTorch model cache"""
        import torch
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        corrupted_files = ["keypointrcnn_resnet50_fpn_coco-fc266e95.pth"]

        for filename in corrupted_files:
            file_path = cache_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"Removed corrupted cache file: {filename}")
                except Exception as e:
                    print(f"Failed to remove {filename}: {e}")

    def _try_manual_download(self):
        """Try manual download as fallback"""
        import requests
        import torch
        from pathlib import Path

        try:
            print("Attempting manual download of KeypointRCNN model...")

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            cache_dir.mkdir(parents=True, exist_ok=True)

            model_filename = "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"
            model_path = cache_dir / model_filename

            # Download URL
            url = f"https://download.pytorch.org/models/{model_filename}"

            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if (
                            total_size > 0 and downloaded % (5 * 1024 * 1024) == 0
                        ):  # Log every 5MB
                            percent = (downloaded / total_size) * 100
                            print(
                                f"Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)"
                            )

            # Validate downloaded file
            try:
                torch.load(model_path, map_location="cpu")
                print("Manual download successful and validated")

                # Now load the model
                from torchvision.models.detection import (
                    KeypointRCNN_ResNet50_FPN_Weights,
                )

                self.model = keypointrcnn_resnet50_fpn(
                    weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
                )
                self.model.eval()
                self.model.to(self.device)

                self.is_initialized = True
                print(
                    f"PyTorch KeypointRCNN loaded successfully after manual download on {self.device}"
                )

            except Exception as e:
                model_path.unlink()  # Remove corrupted file
                raise RuntimeError(f"Manual download validation failed: {e}")

        except Exception as e:
            raise RuntimeError(f"Manual download failed: {e}")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run PyTorch pose estimation on image

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            Standardized pose estimation results
        """
        if not self.is_initialized:
            self.load_model()

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image
        img_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        start_time = time.time()

        try:
            with torch.no_grad():
                predictions = self.model(img_tensor)

            inference_time = time.time() - start_time

            # Convert to standardized format
            return self._convert_to_standard_format(
                predictions[0], image.shape, inference_time
            )

        except Exception as e:
            inference_time = time.time() - start_time
            print(f"PyTorch pose inference error: {e}")

            return {
                "keypoints": np.array([]).reshape(0, 17, 2),
                "scores": np.array([]).reshape(0, 17),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "pytorch_pose",
                    "inference_time": inference_time,
                    "error": str(e),
                },
            }

    def _convert_to_standard_format(
        self, prediction: Dict, image_shape: tuple, inference_time: float
    ) -> Dict[str, Any]:
        """Convert PyTorch KeypointRCNN results to standardized format"""

        try:
            # Filter detections by confidence
            scores = prediction["scores"].cpu().numpy()
            valid_detections = scores > self.confidence_threshold

            if not valid_detections.any():
                return {
                    "keypoints": np.array([]).reshape(0, 17, 2),
                    "scores": np.array([]).reshape(0, 17),
                    "bbox": np.array([]).reshape(0, 4),
                    "num_persons": 0,
                    "metadata": {
                        "model": "pytorch_pose",
                        "inference_time": inference_time,
                    },
                }

            # Extract valid detections
            boxes = prediction["boxes"][valid_detections].cpu().numpy()
            keypoints = prediction["keypoints"][valid_detections].cpu().numpy()
            keypoints_scores = (
                prediction["keypoints_scores"][valid_detections].cpu().numpy()
            )

            # Convert keypoints format (remove visibility dimension, keep only x,y)
            formatted_keypoints = keypoints[:, :, :2]  # (N, 17, 2)

            # Limit number of detections
            if len(formatted_keypoints) > self.max_detections:
                formatted_keypoints = formatted_keypoints[: self.max_detections]
                keypoints_scores = keypoints_scores[: self.max_detections]
                boxes = boxes[: self.max_detections]

            return {
                "keypoints": formatted_keypoints,
                "scores": keypoints_scores,
                "bbox": boxes,
                "num_persons": len(formatted_keypoints),
                "metadata": {
                    "model": "pytorch_pose",
                    "inference_time": inference_time,
                    "confidence_threshold": self.confidence_threshold,
                    "device": str(self.device),
                },
            }

        except Exception as e:
            print(f"Error converting PyTorch results: {e}")
            return {
                "keypoints": np.array([]).reshape(0, 17, 2),
                "scores": np.array([]).reshape(0, 17),
                "bbox": np.array([]).reshape(0, 4),
                "num_persons": 0,
                "metadata": {
                    "model": "pytorch_pose",
                    "inference_time": inference_time,
                    "error": f"Conversion error: {e}",
                },
            }

    def get_keypoint_names(self) -> List[str]:
        """Get COCO keypoint names (PyTorch uses COCO format)"""
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
            "name": "PyTorch KeypointRCNN",
            "version": torchvision.__version__,
            "backbone": "ResNet50 + FPN",
            "keypoints": 17,
            "format": "COCO",
            "device": str(self.device),
            "confidence_threshold": self.confidence_threshold,
            "keypoint_threshold": self.keypoint_threshold,
            "apple_silicon_compatible": True,
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return {
            "model_size_mb": 160.0,
            "avg_inference_time_ms": 100.0,  # Varies by device
            "memory_usage_mb": 200.0,
            "accuracy_coco_ap": 65.5,  # COCO AP score
        }
