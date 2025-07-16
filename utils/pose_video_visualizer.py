"""
Pose Video Visualization Utilities
Adapted from mmpose_estimation.py for evaluation framework integration
"""

import os
import cv2
import json
import time
import numpy as np
import shutil
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PoseVideoVisualizer:
    """High-quality pose video visualization with color-coded keypoints and skeleton"""

    def __init__(self, encoding_config: Optional[Dict[str, Any]] = None):
        """Initialize the pose video visualizer

        Args:
            encoding_config: Video encoding configuration
        """
        # Custom Color Scheme (adapted from mmpose_estimation.py)
        self.kpt_indices = {
            "face": [0, 1, 2, 3, 4],
            "torso": [5, 6, 11, 12],
            "left_arm": [5, 7, 9],
            "right_arm": [6, 8, 10],
            "left_leg": [11, 13, 15],
            "right_leg": [12, 14, 16],
        }

        # Video encoding configuration
        self.encoding_config = encoding_config or self._get_default_encoding_config()

        self.colors = {
            "right_limb": (0, 255, 0),  # Bright Green
            "left_limb": (0, 165, 255),  # Bright Orange
            "torso": (255, 255, 0),  # Bright Cyan
            "head_neck": (255, 0, 255),  # Bright Magenta
            "face": (0, 0, 255),  # Bright Red
            "bbox": (255, 191, 0),  # Light Blue
            "default": (255, 255, 255),  # White
        }

        # Skeleton connections with color mapping
        self.skeleton_definitions = [
            (15, 13, "left_limb"),  # Left ankle to knee
            (13, 11, "left_limb"),  # Left knee to hip
            (16, 14, "right_limb"),  # Right ankle to knee
            (14, 12, "right_limb"),  # Right knee to hip
            (11, 12, "torso"),  # Hip to hip
            (5, 11, "torso"),  # Left shoulder to hip
            (6, 12, "torso"),  # Right shoulder to hip
            (5, 6, "torso"),  # Shoulder to shoulder
            (5, 7, "left_limb"),  # Left shoulder to elbow
            (7, 9, "left_limb"),  # Left elbow to wrist
            (6, 8, "right_limb"),  # Right shoulder to elbow
            (8, 10, "right_limb"),  # Right elbow to wrist
            (1, 2, "face"),  # Face connections
            (0, 1, "face"),
            (0, 2, "face"),
            (1, 3, "face"),
            (2, 4, "face"),
            (3, 5, "head_neck"),  # Face to shoulders
            (4, 6, "head_neck"),
        ]

        # Drawing parameters
        self.link_thickness = 2
        self.kpt_radius = 3
        self.bbox_thickness = 2

        # Detect best available encoder
        self._best_encoder = self._detect_best_encoder()

    def _detect_best_encoder(self) -> Dict[str, str]:
        """Detect the best available hardware encoder"""
        import platform
        import subprocess

        best_encoder = {
            "h264": "libx264",  # Default fallback
            "h265": "libx265",  # Default fallback
            "hardware_type": "cpu",
        }

        try:
            # Check available encoders
            result = subprocess.run(
                ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                logger.warning(
                    "Could not detect available encoders, using CPU fallback"
                )
                return best_encoder

            encoders_output = result.stdout

            # Check for NVIDIA NVENC (best for RTX GPUs)
            if "h264_nvenc" in encoders_output:
                best_encoder.update(
                    {
                        "h264": "h264_nvenc",
                        "h265": "hevc_nvenc",
                        "hardware_type": "nvidia_nvenc",
                    }
                )
                logger.info("🚀 Detected NVIDIA NVENC encoder - using GPU acceleration")
                return best_encoder

            # Check for macOS VideoToolbox
            if platform.system() == "Darwin" and "h264_videotoolbox" in encoders_output:
                best_encoder.update(
                    {
                        "h264": "h264_videotoolbox",
                        "h265": "hevc_videotoolbox",
                        "hardware_type": "videotoolbox",
                    }
                )
                logger.info(
                    "🍎 Detected VideoToolbox encoder - using hardware acceleration"
                )
                return best_encoder

            # Check for Intel QuickSync
            if "h264_qsv" in encoders_output:
                best_encoder.update(
                    {
                        "h264": "h264_qsv",
                        "h265": "hevc_qsv",
                        "hardware_type": "intel_qsv",
                    }
                )
                logger.info(
                    "⚡ Detected Intel QuickSync encoder - using hardware acceleration"
                )
                return best_encoder

            # Check for AMD AMF
            if "h264_amf" in encoders_output:
                best_encoder.update(
                    {"h264": "h264_amf", "h265": "hevc_amf", "hardware_type": "amd_amf"}
                )
                logger.info("🔥 Detected AMD AMF encoder - using hardware acceleration")
                return best_encoder

            logger.info(
                "💻 No hardware encoders detected - using optimized CPU encoding"
            )

        except subprocess.TimeoutExpired:
            logger.warning("Encoder detection timed out, using CPU fallback")
        except Exception as e:
            logger.warning(f"Encoder detection failed: {e}, using CPU fallback")

        return best_encoder

    def _get_encoder_params(
        self, codec: str, hardware_type: str, config: Dict
    ) -> tuple:
        """Get encoder-specific quality and preset parameters"""
        quality_config = config.get("quality", {})
        crf = quality_config.get("crf", 18)
        preset = quality_config.get("preset", "medium")

        if hardware_type == "nvidia_nvenc":
            # NVENC parameters
            quality_params = ["-cq", str(crf)]  # Use -cq instead of -crf for NVENC
            if "nvenc" in codec:
                preset_params = ["-preset", "p4"]  # p1=fastest, p7=slowest, p4=balanced
            else:
                preset_params = ["-preset", preset]

        elif hardware_type == "videotoolbox":
            # VideoToolbox parameters
            quality_params = ["-q:v", str(crf)]  # Use -q:v for VideoToolbox
            preset_params = []  # VideoToolbox doesn't use presets

        elif hardware_type == "intel_qsv":
            # Intel QuickSync parameters
            quality_params = ["-global_quality", str(crf)]
            preset_params = [
                "-preset",
                "medium",
            ]  # QSV presets: veryslow, slower, slow, medium, fast, faster, veryfast

        elif hardware_type == "amd_amf":
            # AMD AMF parameters
            quality_params = ["-qp_i", str(crf), "-qp_p", str(crf)]
            preset_params = ["-quality", "balanced"]  # speed, balanced, quality

        else:
            # CPU encoding (libx264/libx265)
            quality_params = ["-crf", str(crf)]
            preset_params = ["-preset", preset]

        return quality_params, preset_params

    def _get_default_encoding_config(self) -> Dict[str, Any]:
        """Get default video encoding configuration with hardware acceleration"""
        return {
            "format": "h264",  # h264, h265, prores, ffv1, lossless
            "codec": "auto",  # auto, libx264, h264_nvenc, h264_videotoolbox
            "hardware_acceleration": True,  # Enable hardware acceleration when available
            "quality": {
                "crf": 18,  # Lower = better quality (0-51 for x264)
                "preset": "medium",  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
            },
            "pixel_format": "yuv420p",
            "audio": {
                "enabled": True,
                "codec": "copy",  # copy, aac, mp3
            },
            "container": "mp4",  # mp4, mkv, avi
        }

    def _get_encoding_settings(self) -> Dict[str, Any]:
        """Get FFmpeg encoding settings based on configuration"""
        config = self.encoding_config
        format_type = config.get("format", "h264").lower()
        use_hardware = config.get("hardware_acceleration", True)
        codec_override = config.get("codec", "auto")

        if format_type == "h264":
            # Determine codec to use
            if codec_override == "auto" and use_hardware:
                video_codec = self._best_encoder["h264"]
                hardware_type = self._best_encoder["hardware_type"]
            elif codec_override != "auto":
                video_codec = codec_override
                hardware_type = "manual"
            else:
                video_codec = "libx264"
                hardware_type = "cpu"

            # Get quality parameters based on encoder type
            quality_params, preset_params = self._get_encoder_params(
                video_codec, hardware_type, config
            )

            return {
                "video_codec": video_codec,
                "quality_params": quality_params,
                "preset_params": preset_params,
                "pixel_format": config.get("pixel_format", "yuv420p"),
                "container": config.get("container", "mp4"),
                "hardware_type": hardware_type,
            }
        elif format_type == "h265":
            return {
                "video_codec": "libx265",
                "quality_params": [
                    "-crf",
                    str(config.get("quality", {}).get("crf", 23)),
                ],
                "preset_params": [
                    "-preset",
                    config.get("quality", {}).get("preset", "medium"),
                ],
                "pixel_format": config.get("pixel_format", "yuv420p"),
                "container": config.get("container", "mp4"),
            }
        elif format_type == "prores":
            return {
                "video_codec": "prores_ks",
                "quality_params": ["-profile:v", "2"],  # ProRes 422
                "preset_params": [],
                "pixel_format": "yuv422p10le",
                "container": config.get("container", "mov"),
            }
        elif format_type == "ffv1":
            return {
                "video_codec": "ffv1",
                "quality_params": ["-level", "3"],  # FFV1 level 3
                "preset_params": [],
                "pixel_format": "yuv420p",
                "container": config.get("container", "mkv"),
            }
        elif format_type == "lossless":
            return {
                "video_codec": "libx264",
                "quality_params": ["-crf", "0"],  # Lossless
                "preset_params": ["-preset", "veryslow"],
                "pixel_format": "yuv444p",
                "container": config.get("container", "mkv"),
            }
        else:
            # Default to H.264
            return {
                "video_codec": "libx264",
                "quality_params": ["-crf", "18"],
                "preset_params": ["-preset", "medium"],
                "pixel_format": "yuv420p",
                "container": "mp4",
            }

    def get_keypoint_color(self, idx: int) -> Tuple[int, int, int]:
        """Get color for a specific keypoint index"""
        if idx in self.kpt_indices["left_arm"] or idx in self.kpt_indices["left_leg"]:
            return self.colors["left_limb"]
        if idx in self.kpt_indices["right_arm"] or idx in self.kpt_indices["right_leg"]:
            return self.colors["right_limb"]
        if idx in self.kpt_indices["torso"]:
            return self.colors["torso"]
        if idx in self.kpt_indices["face"]:
            return self.colors["face"]
        return self.colors["default"]

    def create_pose_visualization_video(
        self,
        video_path: str,
        pose_results: List[Dict[str, Any]],
        output_path: str,
        model_name: str,
        kpt_thr: float = 0.3,
        bbox_thr: float = 0.3,
        max_persons: int = 3,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        maneuver_start_frame: Optional[int] = None,
        maneuver_end_frame: Optional[int] = None,
    ) -> bool:
        """Create high-quality pose visualization video

        Args:
            video_path: Path to input video
            pose_results: List of pose results per frame
            output_path: Path for output video
            model_name: Name of the pose model
            kpt_thr: Keypoint confidence threshold
            bbox_thr: Bounding box confidence threshold
            max_persons: Maximum number of persons to visualize
            start_frame: Starting frame index (relative to video)
            end_frame: Ending frame index (relative to video)
            maneuver_start_frame: Starting frame of maneuver (absolute frame index)
            maneuver_end_frame: Ending frame of maneuver (absolute frame index)

        Returns:
            Success status
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Determine frame range to process
            if maneuver_start_frame is not None and maneuver_end_frame is not None:
                # Use maneuver frame range
                actual_start_frame = maneuver_start_frame
                actual_end_frame = maneuver_end_frame
                frames_to_process = actual_end_frame - actual_start_frame
                logger.info(
                    f"Creating maneuver visualization: {width}x{height} @ {fps:.2f} FPS, "
                    f"frames {actual_start_frame}-{actual_end_frame} ({frames_to_process} frames)"
                )
            else:
                # Use provided range or full video
                actual_start_frame = start_frame
                actual_end_frame = end_frame if end_frame is not None else total_frames
                frames_to_process = actual_end_frame - actual_start_frame
                logger.info(
                    f"Creating visualization: {width}x{height} @ {fps:.2f} FPS, "
                    f"frames {actual_start_frame}-{actual_end_frame} ({frames_to_process} frames)"
                )

            # Create temp directory for frames
            temp_dir = Path(output_path).parent / "temp_vis_frames"
            temp_dir.mkdir(exist_ok=True)

            # Skip to start frame if needed
            if actual_start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)

            # Process frames
            frame_idx = actual_start_frame
            pose_result_idx = 0  # Index into pose_results array
            annotated_frames = 0
            output_frame_idx = 0  # Index for output frame naming

            while frame_idx < actual_end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get pose results for this frame
                frame_annotated = False
                if pose_result_idx < len(pose_results):
                    pose_result = pose_results[pose_result_idx]
                    frame_annotated = self._draw_poses_on_frame(
                        frame, pose_result, kpt_thr, bbox_thr, max_persons
                    )

                # Add model name and frame info
                self._add_frame_info(
                    frame, model_name, output_frame_idx, frames_to_process
                )

                # Save frame
                frame_path = temp_dir / f"frame_{output_frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), frame)

                if frame_annotated:
                    annotated_frames += 1

                frame_idx += 1
                pose_result_idx += 1
                output_frame_idx += 1

                # Progress update
                if output_frame_idx % 50 == 0:
                    logger.info(
                        f"  Processed {output_frame_idx}/{frames_to_process} frames"
                    )

            cap.release()

            logger.info(
                f"Processed {output_frame_idx} frames, {annotated_frames} with annotations"
            )

            # Create video from frames
            success = self._create_video_from_frames(
                temp_dir, output_path, fps, video_path
            )

            # Cleanup
            shutil.rmtree(temp_dir)

            if success:
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(
                    f"Created visualization: {output_path} ({file_size:.2f} MB)"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to create pose visualization: {e}")
            return False

    def _draw_poses_on_frame(
        self,
        frame: np.ndarray,
        pose_result: Dict[str, Any],
        kpt_thr: float,
        bbox_thr: float,
        max_persons: int,
    ) -> bool:
        """Draw pose annotations on a single frame"""
        if pose_result.get("num_persons", 0) == 0:
            return False

        keypoints = pose_result.get("keypoints", [])
        scores = pose_result.get("scores", [])
        bboxes = pose_result.get("bbox", [])

        if len(keypoints) == 0:
            return False

        frame_annotated = False
        num_persons = min(len(keypoints), max_persons)

        for person_idx in range(num_persons):
            person_kpts = keypoints[person_idx] if person_idx < len(keypoints) else None
            person_scores = scores[person_idx] if person_idx < len(scores) else None
            person_bbox = bboxes[person_idx] if person_idx < len(bboxes) else None

            if person_kpts is None:
                continue

            # Draw bounding box
            if person_bbox is not None and len(person_bbox) >= 4:
                bbox_score = person_bbox[4] if len(person_bbox) > 4 else 1.0
                if bbox_score >= bbox_thr:
                    x1, y1, x2, y2 = [int(coord) for coord in person_bbox[:4]]
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        self.colors["bbox"],
                        self.bbox_thickness,
                    )
                    # Add person confidence score
                    score_text = f"P{person_idx+1}: {bbox_score:.2f}"
                    cv2.putText(
                        frame,
                        score_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.colors["bbox"],
                        1,
                    )
                    frame_annotated = True

            # Draw keypoints and skeleton
            if len(person_kpts) >= 17:  # COCO format
                confident_pts = {}

                # Draw keypoints
                for kpt_idx in range(min(17, len(person_kpts))):
                    if len(person_kpts[kpt_idx]) >= 2:
                        x, y = person_kpts[kpt_idx][:2]
                        kpt_score = (
                            person_scores[kpt_idx]
                            if (
                                person_scores is not None
                                and kpt_idx < len(person_scores)
                            )
                            else 1.0
                        )

                        if kpt_score >= kpt_thr:
                            color = self.get_keypoint_color(kpt_idx)
                            cv2.circle(
                                frame, (int(x), int(y)), self.kpt_radius, color, -1
                            )
                            confident_pts[kpt_idx] = (int(x), int(y))
                            frame_annotated = True

                # Draw skeleton
                for idx1, idx2, color_key in self.skeleton_definitions:
                    if idx1 in confident_pts and idx2 in confident_pts:
                        pt1 = confident_pts[idx1]
                        pt2 = confident_pts[idx2]
                        link_color = self.colors.get(color_key, self.colors["default"])
                        cv2.line(frame, pt1, pt2, link_color, self.link_thickness)
                        frame_annotated = True

        return frame_annotated

    def _add_frame_info(
        self,
        frame: np.ndarray,
        model_name: str,
        frame_idx: int,
        total_frames: int,
    ):
        """Add model name and frame information to frame"""
        height, width = frame.shape[:2]

        # Model name (top-left)
        cv2.putText(
            frame,
            f"Model: {model_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # Frame counter (top-right)
        frame_text = f"Frame: {frame_idx+1}/{total_frames}"
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(
            frame,
            frame_text,
            (width - text_size[0] - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Add legend (bottom-left)
        legend_y = height - 120
        legend_items = [
            ("Left Limbs", self.colors["left_limb"]),
            ("Right Limbs", self.colors["right_limb"]),
            ("Torso", self.colors["torso"]),
            ("Face", self.colors["face"]),
        ]

        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            cv2.circle(frame, (15, y_pos), 8, color, -1)
            cv2.putText(
                frame,
                label,
                (30, y_pos + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def _create_video_from_frames(
        self,
        frames_dir: Path,
        output_path: str,
        fps: float,
        original_video_path: str,
    ) -> bool:
        """Create video from frame images using FFmpeg"""
        try:
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Extract audio from original video
            temp_audio = frames_dir / "audio.aac"
            audio_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                original_video_path,
                "-vn",
                "-acodec",
                "copy",
                str(temp_audio),
            ]

            has_audio = False
            try:
                subprocess.run(
                    audio_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                has_audio = temp_audio.exists() and temp_audio.stat().st_size > 0
            except subprocess.CalledProcessError:
                logger.warning("Could not extract audio from original video")

            # Create video from frames
            frame_pattern = str(frames_dir / "frame_%06d.png")

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                frame_pattern,
            ]

            # Add audio if available
            if has_audio:
                cmd.extend(["-i", str(temp_audio)])

            # Get encoding settings
            encoding_settings = self._get_encoding_settings()

            # Log encoder being used
            codec = encoding_settings["video_codec"]
            hw_type = encoding_settings.get("hardware_type", "unknown")
            logger.info(f"🎬 Using encoder: {codec} ({hw_type})")

            # Video encoding settings
            cmd.extend(["-c:v", codec])
            cmd.extend(encoding_settings["quality_params"])
            cmd.extend(encoding_settings["preset_params"])
            cmd.extend(["-pix_fmt", encoding_settings["pixel_format"]])

            # Audio settings
            audio_config = self.encoding_config.get("audio", {})
            if has_audio and audio_config.get("enabled", True):
                audio_codec = audio_config.get("codec", "copy")
                if audio_codec == "copy":
                    cmd.extend(["-c:a", "copy", "-shortest"])
                else:
                    cmd.extend(["-c:a", audio_codec, "-shortest"])
            else:
                cmd.extend(["-an"])

            # Ensure output has correct extension
            container = encoding_settings["container"]
            if not output_path.endswith(f".{container}"):
                output_path = output_path.rsplit(".", 1)[0] + f".{container}"

            cmd.append(output_path)

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            return os.path.exists(output_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            return False

    def create_model_comparison_video(
        self,
        video_path: str,
        model_results: Dict[str, List[Dict[str, Any]]],
        output_path: str,
        kpt_thr: float = 0.3,
        bbox_thr: float = 0.3,
    ) -> bool:
        """Create side-by-side comparison video of multiple models

        Args:
            video_path: Path to input video
            model_results: Dict of model_name -> pose_results_per_frame
            output_path: Path for output comparison video
            kpt_thr: Keypoint confidence threshold
            bbox_thr: Bounding box confidence threshold

        Returns:
            Success status
        """
        try:
            if not model_results:
                logger.error("No model results provided for comparison")
                return False

            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate grid layout
            num_models = len(model_results)
            cols = min(2, num_models)
            rows = (num_models + cols - 1) // cols

            # Create comparison grid
            grid_width = width * cols
            grid_height = height * rows

            logger.info(
                f"Creating {num_models}-model comparison: {grid_width}x{grid_height}"
            )

            # Create temp directory
            temp_dir = Path(output_path).parent / "temp_comparison_frames"
            temp_dir.mkdir(exist_ok=True)

            frame_idx = 0
            model_names = list(model_results.keys())

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Create comparison grid
                grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

                for model_idx, model_name in enumerate(model_names):
                    # Calculate position in grid
                    row = model_idx // cols
                    col = model_idx % cols

                    # Get model results for this frame
                    model_pose_results = model_results[model_name]
                    pose_result = (
                        model_pose_results[frame_idx]
                        if frame_idx < len(model_pose_results)
                        else {}
                    )

                    # Create annotated frame
                    model_frame = frame.copy()
                    self._draw_poses_on_frame(
                        model_frame, pose_result, kpt_thr, bbox_thr, 2
                    )
                    self._add_frame_info(
                        model_frame, model_name, frame_idx, total_frames
                    )

                    # Place in grid
                    y_start = row * height
                    y_end = (row + 1) * height
                    x_start = col * width
                    x_end = (col + 1) * width

                    grid_frame[y_start:y_end, x_start:x_end] = model_frame

                # Save grid frame
                frame_path = temp_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), grid_frame)

                frame_idx += 1

                if frame_idx % 50 == 0:
                    logger.info(
                        f"  Processed {frame_idx}/{total_frames} comparison frames"
                    )

            cap.release()

            # Create comparison video
            success = self._create_video_from_frames(
                temp_dir, output_path, fps, video_path
            )

            # Cleanup
            shutil.rmtree(temp_dir)

            if success:
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(
                    f"Created comparison video: {output_path} ({file_size:.2f} MB)"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to create comparison video: {e}")
            return False
