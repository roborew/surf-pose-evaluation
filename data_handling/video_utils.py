"""
Video processing utilities for pose estimation evaluation
"""

import cv2
import numpy as np
from typing import List, Iterator, Optional, Tuple
import os
import logging


class VideoProcessor:
    """Utilities for processing video files"""

    def __init__(self):
        """Initialize video processor"""
        pass

    @staticmethod
    def load_video_frames(
        video_path: str, max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """Load frames from video file

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load (None for all)

        Returns:
            List of video frames as numpy arrays
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)
                frame_count += 1

                if max_frames is not None and frame_count >= max_frames:
                    break

        finally:
            cap.release()

        return frames

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video file information

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                / cap.get(cv2.CAP_PROP_FPS),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            }

            # Convert codec to string
            codec_int = info["codec"]
            if codec_int != 0:
                info["codec_str"] = "".join(
                    [chr((codec_int >> 8 * i) & 0xFF) for i in range(4)]
                )
            else:
                info["codec_str"] = "unknown"

        finally:
            cap.release()

        return info

    @staticmethod
    def frame_generator(
        video_path: str, start_frame: int = 0, end_frame: Optional[int] = None
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Generate frames from video file

        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all)

        Yields:
            Tuple of (frame_number, frame_array)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            # Skip to start frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_number = start_frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                yield frame_number, frame
                frame_number += 1

                if end_frame is not None and frame_number >= end_frame:
                    break

        finally:
            cap.release()

    @staticmethod
    def extract_frame_at_time(
        video_path: str, timestamp: float
    ) -> Optional[np.ndarray]:
        """Extract frame at specific timestamp

        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds

        Returns:
            Frame at timestamp or None if not found
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            # Set position to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            ret, frame = cap.read()
            if ret:
                return frame
            else:
                return None

        finally:
            cap.release()

    @staticmethod
    def save_frames_as_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> bool:
        """Save frames as video file

        Args:
            frames: List of video frames
            output_path: Output video file path
            fps: Frames per second
            codec: Video codec

        Returns:
            True if successful, False otherwise
        """
        if not frames:
            logging.error("No frames provided")
            return False

        # Get frame dimensions
        height, width, channels = frames[0].shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            logging.error(f"Could not create video writer for {output_path}")
            return False

        try:
            for frame in frames:
                writer.write(frame)
            return True

        except Exception as e:
            logging.error(f"Error writing video: {e}")
            return False

        finally:
            writer.release()

    @staticmethod
    def resize_frames(
        frames: List[np.ndarray], target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Resize frames to target size

        Args:
            frames: List of input frames
            target_size: Target size as (width, height)

        Returns:
            List of resized frames
        """
        resized_frames = []
        width, height = target_size

        for frame in frames:
            resized_frame = cv2.resize(frame, (width, height))
            resized_frames.append(resized_frame)

        return resized_frames

    @staticmethod
    def crop_frames(
        frames: List[np.ndarray], bbox: Tuple[int, int, int, int]
    ) -> List[np.ndarray]:
        """Crop frames to bounding box

        Args:
            frames: List of input frames
            bbox: Bounding box as (x, y, width, height)

        Returns:
            List of cropped frames
        """
        cropped_frames = []
        x, y, w, h = bbox

        for frame in frames:
            cropped_frame = frame[y : y + h, x : x + w]
            cropped_frames.append(cropped_frame)

        return cropped_frames

    @staticmethod
    def normalize_frames(
        frames: List[np.ndarray],
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """Normalize frames

        Args:
            frames: List of input frames
            mean: Mean values for normalization (default: ImageNet)
            std: Standard deviation values for normalization (default: ImageNet)

        Returns:
            List of normalized frames
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet means
        if std is None:
            std = [0.229, 0.224, 0.225]  # ImageNet stds

        normalized_frames = []

        for frame in frames:
            # Convert to float and normalize to [0, 1]
            norm_frame = frame.astype(np.float32) / 255.0

            # Apply mean and std normalization
            for i in range(3):
                norm_frame[:, :, i] = (norm_frame[:, :, i] - mean[i]) / std[i]

            normalized_frames.append(norm_frame)

        return normalized_frames

    @staticmethod
    def temporal_subsample(
        frames: List[np.ndarray], target_length: int, strategy: str = "uniform"
    ) -> List[np.ndarray]:
        """Subsample frames temporally

        Args:
            frames: List of input frames
            target_length: Target number of frames
            strategy: Subsampling strategy ('uniform', 'random', 'center')

        Returns:
            List of subsampled frames
        """
        if len(frames) <= target_length:
            return frames

        if strategy == "uniform":
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, target_length, dtype=int)

        elif strategy == "random":
            # Random sampling
            indices = np.random.choice(len(frames), target_length, replace=False)
            indices = np.sort(indices)

        elif strategy == "center":
            # Center sampling
            start = (len(frames) - target_length) // 2
            indices = range(start, start + target_length)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        return [frames[i] for i in indices]
