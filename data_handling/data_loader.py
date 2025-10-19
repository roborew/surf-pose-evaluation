"""
Data Loading System for Surfing Pose Estimation Evaluation

This module handles loading video clips and their corresponding annotations
from the surfing dataset, supporting both H264 and FFV1 video formats.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VideoClip:
    """Represents a video clip with metadata and annotations."""

    file_path: str
    video_id: str
    camera: str  # SONY_300, SONY_70, etc.
    session: str  # SESSION_060325, etc.
    duration: float
    fps: float
    width: int
    height: int
    format: str  # h264 or ffv1
    zoom_level: str  # "default", "wide", "full"
    base_clip_id: str  # Base clip without zoom suffix
    annotations: List[Dict] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return int(self.duration * self.fps)

    def get_frame_annotations(self, frame_idx: int) -> List[Dict]:
        """Get annotations for a specific frame based on timestamp."""
        timestamp = frame_idx / self.fps
        frame_annotations = []

        for annotation in self.annotations:
            if annotation["start"] <= timestamp <= annotation["end"]:
                frame_annotations.append(
                    {
                        "timestamp": timestamp,
                        "frame_idx": frame_idx,
                        "labels": annotation["labels"],
                        "maneuver_progress": (timestamp - annotation["start"])
                        / (annotation["end"] - annotation["start"]),
                    }
                )

        return frame_annotations

    def get_maneuvers(self) -> List["Maneuver"]:
        """Extract individual maneuvers from this clip."""
        maneuvers = []
        for i, annotation in enumerate(self.annotations):
            try:
                # Validate annotation structure
                if "labels" not in annotation:
                    logger.warning(
                        f"Annotation {i} in {self.video_id} missing 'labels' field, skipping"
                    )
                    continue

                labels = annotation["labels"]
                if not isinstance(labels, list):
                    logger.warning(
                        f"Annotation {i} in {self.video_id} has non-list labels: {labels}, skipping"
                    )
                    continue

                # Strict validation: must have exactly 2 labels
                if len(labels) != 2:
                    logger.warning(
                        f"Annotation {i} in {self.video_id} has incorrect number of labels (expected exactly 2, got {len(labels)}): {labels}, skipping"
                    )
                    continue

                # Extract execution score and maneuver type with validation
                execution_score_str = labels[0]
                maneuver_type = labels[1]

                # Check for empty, null, or whitespace-only values
                if (
                    not execution_score_str
                    or not isinstance(execution_score_str, str)
                    or not execution_score_str.strip()
                ):
                    logger.warning(
                        f"Annotation {i} in {self.video_id} has empty/invalid execution score: '{execution_score_str}', skipping"
                    )
                    continue

                if (
                    not maneuver_type
                    or not isinstance(maneuver_type, str)
                    or not maneuver_type.strip()
                ):
                    logger.warning(
                        f"Annotation {i} in {self.video_id} has empty/invalid maneuver type: '{maneuver_type}', skipping"
                    )
                    continue

                maneuver = Maneuver(
                    clip=self,
                    maneuver_id=f"{self.video_id}_maneuver_{i}",
                    maneuver_type=maneuver_type,  # Maneuver type (e.g., "Pop-up")
                    execution_score=execution_score_str,  # Execution score (e.g., "09")
                    start_time=annotation["start"],
                    end_time=annotation["end"],
                    start_frame=int(annotation["start"] * self.fps),
                    end_frame=int(annotation["end"] * self.fps),
                    annotation_data=annotation,
                )
                maneuvers.append(maneuver)

            except (IndexError, KeyError, TypeError) as e:
                logger.warning(
                    f"Error processing annotation {i} in {self.video_id}: {e}, skipping"
                )
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected error processing annotation {i} in {self.video_id}: {e}"
                )
                continue

        # Log summary of annotation processing
        total_annotations = len(self.annotations)
        valid_annotations = len(maneuvers)
        skipped_annotations = total_annotations - valid_annotations

        if skipped_annotations > 0:
            logger.info(
                f"Processed {self.video_id}: {valid_annotations}/{total_annotations} annotations valid, {skipped_annotations} skipped"
            )
        else:
            logger.debug(
                f"Processed {self.video_id}: {valid_annotations}/{total_annotations} annotations valid"
            )

        return maneuvers


@dataclass
class Maneuver:
    """Represents an individual surfing maneuver with its timecode and metadata."""

    clip: VideoClip
    maneuver_id: str
    maneuver_type: str  # "popup", "cutback", etc.
    execution_score: float
    start_time: float  # seconds
    end_time: float  # seconds
    start_frame: int
    end_frame: int
    annotation_data: Dict

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def total_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def file_path(self) -> str:
        """Return the video file path for compatibility."""
        return self.clip.file_path

    @property
    def fps(self) -> float:
        return self.clip.fps

    def get_frame_range(self) -> Tuple[int, int]:
        """Get the frame range for this maneuver."""
        return self.start_frame, self.end_frame


@dataclass
class DataSplit:
    """Represents train/validation/test splits of the dataset."""

    train: List[VideoClip] = field(default_factory=list)
    val: List[VideoClip] = field(default_factory=list)
    test: List[VideoClip] = field(default_factory=list)

    def get_split_stats(self) -> Dict:
        """Get statistics about the data splits including zoom distribution."""

        def get_stats(clips):
            zoom_counts = Counter(clip.zoom_level for clip in clips)
            return {
                "num_clips": len(clips),
                "total_duration": sum(clip.duration for clip in clips),
                "total_frames": sum(clip.total_frames for clip in clips),
                "zoom_distribution": dict(zoom_counts),
                "total_maneuvers": sum(len(clip.annotations) for clip in clips),
            }

        return {
            "train": get_stats(self.train),
            "val": get_stats(self.val),
            "test": get_stats(self.test),
        }


class SurfingDataLoader:
    """
    Main data loader for the surfing pose estimation dataset.

    Handles loading video clips and annotations, creating train/val/test splits,
    and providing data for pose estimation evaluation with zoom-aware processing.
    """

    def __init__(self, config: Dict):
        """
        Initialize the data loader with configuration.

        Args:
            config: Configuration dictionary containing dataset paths and settings
        """
        self.config = config
        self.data_source_config = config["data_source"]
        self.base_path = Path(self.data_source_config["base_data_path"])

        # Video and annotation paths
        self.video_clips_config = self.data_source_config["video_clips"]
        self.annotations_config = self.data_source_config["annotations"]

        # Data splits configuration
        self.splits_config = self.data_source_config["splits"]

        # Zoom handling configuration
        self.zoom_config = self.splits_config.get(
            "zoom_handling",
            {
                "enabled": True,
                "balanced_distribution": True,
                "target_distribution": {"default": 0.33, "wide": 0.33, "full": 0.34},
            },
        )

        # Camera selection configuration
        self.camera_config = self.data_source_config.get("camera_selection", {})
        self.enabled_cameras = self.camera_config.get(
            "enabled_cameras", ["SONY_300", "SONY_70"]
        )

        # Initialize containers
        self.all_clips: List[VideoClip] = []
        self.annotations_data: Dict = {}
        self.data_splits: Optional[DataSplit] = None
        self.zoom_groups: Dict[str, List[VideoClip]] = defaultdict(list)

        logger.info(f"Initialized SurfingDataLoader with base path: {self.base_path}")
        logger.info(f"Enabled cameras: {self.enabled_cameras}")
        if self.zoom_config.get("enabled", True):
            logger.info("Zoom-aware processing enabled with balanced distribution")

    def load_annotations(self) -> Dict:
        """
        Load all annotation files from the label studio exports.

        Returns:
            Dictionary mapping video files to their annotations
        """
        annotations = {}
        labels_path = self.base_path / self.annotations_config["labels_path"]

        # Load SONY_300 annotations
        sony_300_path = labels_path / self.annotations_config["sony_300_labels"]
        if sony_300_path.exists():
            for json_file in sony_300_path.glob("*.json"):
                logger.info(f"Loading annotations from {json_file}")
                with open(json_file, "r") as f:
                    data = json.load(f)
                    for item in data:
                        video_url = item.get("video_url") or item.get("data", {}).get(
                            "video_url"
                        )
                        if not video_url:
                            logger.warning(
                                "Annotation item missing video URL in %s", json_file
                            )
                            continue

                        # Extract video file path from URL
                        video_file = self._extract_video_path_from_url(video_url)
                        if not video_file:
                            continue

                        # Legacy format: annotations stored directly under "tricks"
                        legacy_tricks = item.get("tricks")

                        if legacy_tricks is None:
                            legacy_tricks = self._convert_labelstudio_annotations(item)

                        if legacy_tricks:
                            annotations[video_file] = legacy_tricks

        # Load SONY_70 annotations
        sony_70_path = labels_path / self.annotations_config["sony_70_labels"]
        if sony_70_path.exists():
            for json_file in sony_70_path.glob("*.json"):
                logger.info(f"Loading annotations from {json_file}")
                with open(json_file, "r") as f:
                    data = json.load(f)
                    for item in data:
                        video_url = item.get("video_url") or item.get("data", {}).get(
                            "video_url"
                        )
                        if not video_url:
                            logger.warning(
                                "Annotation item missing video URL in %s", json_file
                            )
                            continue

                        video_file = self._extract_video_path_from_url(video_url)
                        if not video_file:
                            continue

                        legacy_tricks = item.get("tricks")
                        if legacy_tricks is None:
                            legacy_tricks = self._convert_labelstudio_annotations(item)

                        if legacy_tricks:
                            annotations[video_file] = legacy_tricks

        self.annotations_data = annotations
        logger.info(f"Loaded annotations for {len(annotations)} video files")
        return annotations

    def _extract_video_path_from_url(self, video_url: str) -> Optional[str]:
        """
        Extract video file path from Label Studio URL format.

        Args:
            video_url: URL from Label Studio export

        Returns:
            Relative path to video file or None if invalid
        """
        # Old local Label Studio format: /data/local-files/?d=SD_02_SURF_FOOTAGE_PREPT/03_CLIPPED/h264/...
        if "?d=" in video_url:
            path_part = video_url.split("?d=")[1]
            if path_part.startswith("SD_02_SURF_FOOTAGE_PREPT/"):
                return path_part.replace("SD_02_SURF_FOOTAGE_PREPT/", "")

        # New remote Label Studio format: s3://training-data/video/<format>/...
        if video_url.startswith("s3://"):
            without_scheme = video_url[5:]
            if "/" in without_scheme:
                _, _, key_path = without_scheme.partition("/")
            else:
                key_path = ""

            prefix_map = {
                "video/h264/": "03_CLIPPED/h264/",
                "video/ffv1/": "03_CLIPPED/ffv1/",
            }

            for remote_prefix, local_prefix in prefix_map.items():
                if key_path.startswith(remote_prefix):
                    mapped = local_prefix + key_path[len(remote_prefix) :]
                    # Normalize session directories (remove variant suffixes like _WIDE)
                    parts = mapped.split("/")
                    if len(parts) >= 5:
                        session_idx = 4  # 03_CLIPPED/<fmt>/<camera>/<session>/...
                        session_name = parts[session_idx]
                        base_session, _ = self._parse_session_dir_name(session_name)
                        parts[session_idx] = base_session
                        mapped = "/".join(parts)
                    return mapped

            logger.warning("Unable to map S3 video URL to local path: %s", video_url)

        return None

    def _parse_session_dir_name(
        self, session_dir_name: str
    ) -> Tuple[str, Optional[str]]:
        """Split a session directory name into base session and optional variant suffix."""

        if not session_dir_name.startswith("SESSION"):
            return session_dir_name, None

        parts = session_dir_name.split("_")
        if len(parts) <= 2:
            return session_dir_name, None

        base_session = "_".join(parts[:2])
        variant = "_".join(parts[2:]) if len(parts) > 2 else None
        return base_session, variant or None

    def _iter_session_variant_dirs(self, camera_dir: Path):
        """Yield session directories with their base name and variant info."""

        for session_dir in sorted(camera_dir.iterdir()):
            if not session_dir.is_dir():
                continue

            base_session, variant = self._parse_session_dir_name(session_dir.name)
            if not base_session.startswith("SESSION"):
                logger.debug(
                    f"Skipping directory {session_dir} (does not look like a session folder)"
                )
                continue

            yield session_dir, base_session, variant

    def _extract_zoom_info(self, video_path: Path) -> Tuple[str, str]:
        """
        Extract zoom level and base clip ID from video filename.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (zoom_level, base_clip_id)
        """
        stem = video_path.stem

        if stem.endswith("_full"):
            return "full", stem[:-5]  # Remove '_full'
        elif stem.endswith("_wide"):
            return "wide", stem[:-5]  # Remove '_wide'
        else:
            return "default", stem

    def discover_video_clips(
        self, video_format: Optional[str] = None
    ) -> List[VideoClip]:
        """
        Discover all video clips in the specified format, grouping by zoom variations.

        Args:
            video_format: Video format to load ("h264" or "ffv1"). If None, uses config setting.

        Returns:
            List of VideoClip objects with balanced zoom distribution
        """
        # Load annotations if not already loaded
        if not self.annotations_data:
            self.annotations_data = self.load_annotations()

        # Use video format from config if not specified
        if video_format is None:
            video_format = self.video_clips_config.get("input_format", "h264")

        if video_format == "h264":
            clips_path = self.base_path / self.video_clips_config["h264_path"]
            file_extension = "*.mp4"
        elif video_format == "ffv1":
            clips_path = self.base_path / self.video_clips_config["ffv1_path"]
            file_extension = "*.mkv"
        else:
            raise ValueError(f"Unsupported video format: {video_format}")

        if not clips_path.exists():
            logger.warning(f"Video clips path does not exist: {clips_path}")
            return []

        # First pass: Group all video files by base clip ID
        zoom_groups = defaultdict(lambda: defaultdict(list))

        # Discover all video files and group by zoom variations
        for camera_dir in clips_path.iterdir():
            if not camera_dir.is_dir():
                continue

            camera_name = camera_dir.name
            logger.info(f"Discovering clips for camera: {camera_name}")

            for session_dir, session_base, _ in self._iter_session_variant_dirs(
                camera_dir
            ):
                for video_file in session_dir.glob(file_extension):
                    zoom_level, base_clip_id = self._extract_zoom_info(video_file)
                    group_key = f"{camera_name}_{session_base}_{base_clip_id}"
                    zoom_groups[group_key][zoom_level].append(video_file)

        # Second pass: Select clips with annotations using balanced zoom distribution
        selected_clips = []
        zoom_counters = Counter()
        target_dist = self.zoom_config.get(
            "target_distribution", {"default": 0.33, "wide": 0.33, "full": 0.34}
        )

        # Convert to list for shuffling to randomize selection order
        group_items = list(zoom_groups.items())
        random.shuffle(group_items)

        # Process groups with balanced zoom selection
        for group_key, zoom_variants in group_items:
            # Initialize variables for this group
            selected_file = None
            selected_zoom = None

            # Check if this base clip has annotations by looking up the base clip name
            # All zoom variants share the same base clip ID and therefore the same annotations
            base_clip_annotation_key = None

            # Extract base clip info from group key
            # Only process SONY cameras (SONY_70, SONY_300) as GP1/GP2 don't have annotations
            # Group key format: "SONY_XXX_SESSION_YYYYMMDD_CLIPID"
            # Example: "SONY_300_SESSION_060325_C0019_clip_1"
            parts = group_key.split("_")

            # Skip non-SONY cameras (GP1, GP2, etc.) as they don't have annotations
            if not group_key.startswith("SONY_"):
                continue

            # Extract camera name from group key and check if it's enabled
            parts = group_key.split("_")
            if len(parts) >= 2:
                camera_name = f"{parts[0]}_{parts[1]}"  # e.g., "SONY_300" or "SONY_70"
                if camera_name not in self.enabled_cameras:
                    continue

            if len(parts) >= 6:
                # SONY camera format: SONY_300_SESSION_060325_C0019_clip_1
                camera = f"{parts[0]}_{parts[1]}"  # e.g., "SONY_300" or "SONY_70"
                session = f"{parts[2]}_{parts[3]}"  # e.g., "SESSION_060325"
                base_clip_id = "_".join(parts[4:])  # e.g., "C0019_clip_1"

                # Create the annotation lookup key (always references the base/default clip)
                # Annotations always reference h264 paths regardless of video format being processed
                base_clip_annotation_key = (
                    f"03_CLIPPED/h264/{camera}/{session}/{base_clip_id}.mp4"
                )
            else:
                continue  # Skip malformed group keys

            # Check if base clip has annotations
            if (
                base_clip_annotation_key
                and base_clip_annotation_key in self.annotations_data
            ):
                # This base clip has annotations - all zoom variants can use these annotations
                available_zoom_variants = list(zoom_variants.keys())

                if available_zoom_variants:
                    # Select zoom level using balanced distribution
                    selected_zoom = self._select_balanced_zoom(
                        zoom_variants, zoom_counters, target_dist
                    )

                    if selected_zoom and selected_zoom in zoom_variants:
                        selected_file = zoom_variants[selected_zoom][
                            0
                        ]  # Take first video file for this zoom
                    else:
                        # Fallback: take any available zoom variant
                        selected_zoom = available_zoom_variants[0]
                        selected_file = zoom_variants[selected_zoom][0]

            # Create clip if we found an annotated file
            if selected_file and selected_zoom:
                try:
                    # Extract camera and session from path
                    path_parts = selected_file.relative_to(clips_path).parts
                    camera = path_parts[0]
                    session_variant = path_parts[1]
                    session_base, _ = self._parse_session_dir_name(session_variant)

                    clip = self._create_video_clip(
                        selected_file,
                        camera,
                        session_base,
                        video_format,
                        selected_zoom,
                    )
                    if clip and clip.annotations:  # Only add clips with annotations
                        selected_clips.append(clip)
                        zoom_counters[selected_zoom] += 1

                except Exception as e:
                    logger.warning(f"Failed to process {selected_file}: {e}")

        self.all_clips = selected_clips

        # Log zoom distribution statistics
        total_clips = len(selected_clips)
        if total_clips > 0:
            zoom_stats = {
                zoom: f"{count} ({count/total_clips:.1%})"
                for zoom, count in zoom_counters.items()
            }
            logger.info(
                f"Discovered {total_clips} clips with zoom distribution: {zoom_stats}"
            )

        return selected_clips

    def _select_balanced_zoom(
        self,
        zoom_variants: Dict[str, List],
        current_counters: Counter,
        target_distribution: Dict[str, float],
    ) -> Optional[str]:
        """
        Select zoom level to maintain balanced distribution.

        Args:
            zoom_variants: Available zoom variants for this clip
            current_counters: Current counts of each zoom level
            target_distribution: Target distribution percentages

        Returns:
            Selected zoom level or None if no variants available
        """
        available_zooms = list(zoom_variants.keys())
        if not available_zooms:
            return None

        total_clips = sum(current_counters.values())

        if total_clips == 0:
            # First clip - random selection
            return random.choice(available_zooms)

        # Calculate current distribution vs target
        zoom_priorities = []
        for zoom in available_zooms:
            current_ratio = current_counters[zoom] / total_clips
            target_ratio = target_distribution.get(zoom, 0.33)
            deficit = target_ratio - current_ratio
            zoom_priorities.append((deficit, zoom))

        # Sort by deficit (highest first) and select
        zoom_priorities.sort(reverse=True, key=lambda x: x[0])

        # Add some randomness while favoring underrepresented zooms
        if (
            len(zoom_priorities) > 1
            and zoom_priorities[0][0] - zoom_priorities[1][0] < 0.05
        ):
            # If deficit difference is small, randomly choose from top 2
            return random.choice([zoom_priorities[0][1], zoom_priorities[1][1]])
        else:
            return zoom_priorities[0][1]

    def _create_video_clip(
        self, video_path: Path, camera: str, session: str, format: str, zoom_level: str
    ) -> Optional[VideoClip]:
        """
        Create a VideoClip object from a video file.

        Args:
            video_path: Path to video file
            camera: Camera name (SONY_300, SONY_70, etc.)
            session: Session name (SESSION_060325, etc.)
            format: Video format (h264, ffv1)
            zoom_level: Zoom level (default, wide, full)

        Returns:
            VideoClip object or None if failed
        """
        try:
            # Get video metadata using OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            # Generate video ID and base clip ID
            _, base_clip_id = self._extract_zoom_info(video_path)
            video_id = f"{camera}_{session}_{video_path.stem}"

            # For annotation lookup, always use the canonical session folder
            parent_dir = f"03_CLIPPED/{format}/{camera}/{session}"
            base_filename = f"{base_clip_id}.mp4"  # Always .mp4 for annotation lookup

            if format == "ffv1":
                # For FFV1 format, map to corresponding H.264 path for annotation lookup
                annotation_lookup_path = (
                    f"03_CLIPPED/h264/{camera}/{session}/{base_filename}"
                )
            else:
                # For H.264 format, use base clip name in same directory
                annotation_lookup_path = f"{parent_dir}/{base_filename}"

            # Get annotations for this video using the appropriate lookup path
            annotations = self.annotations_data.get(annotation_lookup_path, [])

            clip = VideoClip(
                file_path=str(video_path),
                video_id=video_id,
                camera=camera,
                session=session,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                format=format,
                zoom_level=zoom_level,
                base_clip_id=base_clip_id,
                annotations=annotations,
            )

            return clip

        except Exception as e:
            logger.error(f"Error creating VideoClip for {video_path}: {e}")
            return None

    def create_data_splits(self, random_seed: Optional[int] = None) -> DataSplit:
        """
        Create train/validation/test splits ensuring no zoom variations of the same
        base clip appear across different splits.

        Args:
            random_seed: Random seed for reproducible splits

        Returns:
            DataSplit object with train/val/test clips
        """
        if random_seed is None:
            random_seed = self.splits_config.get("random_seed", 42)

        random.seed(random_seed)

        # Group clips by session AND base clip ID to prevent any leakage
        session_base_groups = defaultdict(list)
        for clip in self.all_clips:
            # Use base clip ID to ensure zoom variations don't leak across splits
            session_key = f"{clip.camera}_{clip.session}_{clip.base_clip_id}"
            session_base_groups[session_key].append(clip)

        # Split by session groups (not individual clips)
        session_keys = list(session_base_groups.keys())
        random.shuffle(session_keys)

        train_ratio = self.splits_config["train_ratio"]
        val_ratio = self.splits_config["val_ratio"]

        n_sessions = len(session_keys)
        n_train = int(n_sessions * train_ratio)
        n_val = int(n_sessions * val_ratio)

        train_sessions = session_keys[:n_train]
        val_sessions = session_keys[n_train : n_train + n_val]
        test_sessions = session_keys[n_train + n_val :]

        # Assign clips to splits based on session assignment
        train_clips = []
        val_clips = []
        test_clips = []

        for session_key in train_sessions:
            train_clips.extend(session_base_groups[session_key])

        for session_key in val_sessions:
            val_clips.extend(session_base_groups[session_key])

        for session_key in test_sessions:
            test_clips.extend(session_base_groups[session_key])

        self.data_splits = DataSplit(train=train_clips, val=val_clips, test=test_clips)

        stats = self.data_splits.get_split_stats()
        logger.info(
            f"Created zoom-aware data splits: Train({stats['train']['num_clips']} clips), "
            f"Val({stats['val']['num_clips']} clips), Test({stats['test']['num_clips']} clips)"
        )

        # Log zoom distribution per split
        for split_name, split_stats in stats.items():
            zoom_dist = split_stats["zoom_distribution"]
            logger.info(f"{split_name.title()} zoom distribution: {zoom_dist}")

        return self.data_splits

    def load_maneuvers_from_manifest(self, manifest_path: str) -> List[Maneuver]:
        """Load maneuvers from a selection manifest file

        Args:
            manifest_path: Path to selection manifest JSON file

        Returns:
            List of Maneuver objects loaded from manifest
        """
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            logger.info(f"📖 Loading maneuvers from manifest: {manifest_path}")

            maneuvers = []

            for clip_data in manifest["selected_clips"]:
                # Create a simplified VideoClip object from manifest data
                video_metadata = clip_data["video_metadata"]

                # Create clip object with manifest data
                clip = VideoClip(
                    file_path=clip_data["video_path"],
                    video_id=clip_data["clip_id"],
                    camera=clip_data["camera"],
                    session=clip_data["session"],
                    duration=video_metadata.get("duration", 0),
                    fps=video_metadata.get("fps", 25.0),
                    width=video_metadata.get("width", 1920),
                    height=video_metadata.get("height", 1080),
                    format=manifest["selection_metadata"]["video_format"],
                    zoom_level=clip_data["zoom_level"],
                    base_clip_id=clip_data["base_clip_id"],
                    annotations=[],  # Annotations handled in maneuver objects
                )

                # Create maneuver objects from manifest data
                for maneuver_data in clip_data["maneuvers"]:
                    maneuver = Maneuver(
                        clip=clip,
                        maneuver_id=maneuver_data["maneuver_id"],
                        maneuver_type=maneuver_data["maneuver_type"],
                        execution_score=maneuver_data["execution_score"],
                        start_time=maneuver_data["start_time"],
                        end_time=maneuver_data["end_time"],
                        start_frame=maneuver_data["start_frame"],
                        end_frame=maneuver_data["end_frame"],
                        annotation_data=maneuver_data["annotation_data"],
                    )
                    maneuvers.append(maneuver)

            logger.info(
                f"✅ Loaded {len(maneuvers)} maneuvers from {len(manifest['selected_clips'])} clips"
            )
            return maneuvers

        except Exception as e:
            logger.error(f"Failed to load maneuvers from manifest {manifest_path}: {e}")
            raise

    def load_video_frames(
        self, clip_or_maneuver, start_frame: int = 0, end_frame: Optional[int] = None
    ) -> np.ndarray:
        """
        Load frames from a video clip or maneuver.

        Args:
            clip_or_maneuver: VideoClip or Maneuver object
            start_frame: Starting frame index (relative to clip/maneuver)
            end_frame: Ending frame index (None for all frames)

        Returns:
            Array of frames with shape (num_frames, height, width, channels)
        """
        # Handle both VideoClip and Maneuver objects
        if isinstance(clip_or_maneuver, Maneuver):
            # For maneuvers, use the maneuver's frame range
            file_path = clip_or_maneuver.file_path
            maneuver_start, maneuver_end = clip_or_maneuver.get_frame_range()

            # Adjust frame indices to be relative to the maneuver
            actual_start = maneuver_start + start_frame
            actual_end = maneuver_start + (
                end_frame if end_frame is not None else clip_or_maneuver.total_frames
            )

            # Ensure we don't go beyond the maneuver's end
            actual_end = min(actual_end, maneuver_end)

        else:
            # For clips, use the provided range or full clip
            file_path = clip_or_maneuver.file_path
            actual_start = start_frame
            actual_end = (
                end_frame if end_frame is not None else clip_or_maneuver.total_frames
            )

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {file_path}")

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= actual_start and frame_idx < actual_end:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_idx += 1

            if frame_idx >= actual_end:
                break

        cap.release()

        if not frames:
            raise ValueError(
                f"No frames loaded from {file_path} (range: {actual_start}-{actual_end})"
            )

        return np.array(frames)

    def _convert_labelstudio_annotations(self, item: Dict) -> Optional[List[Dict]]:
        """Convert new Label Studio export format into legacy 'tricks' list."""

        results = []

        for annotation in item.get("annotations", []):
            maneuver_entries = {}

            for result in annotation.get("result", []):
                res_type = result.get("type")
                from_name = result.get("from_name")
                value = result.get("value", {})

                if res_type == "labels" and from_name in {"tricks", "maneuver"}:
                    start = value.get("start")
                    end = value.get("end")
                    labels = value.get("labels", [])

                    if not labels:
                        continue

                    maneuver_id = result.get("id")
                    if not maneuver_id:
                        maneuver_id = f"{start}_{end}_{labels[0]}"

                    entry = maneuver_entries.setdefault(
                        maneuver_id,
                        {
                            "id": maneuver_id,
                            "start": start,
                            "end": end,
                            "labels": [],
                            "score": None,
                        },
                    )

                    entry["labels"].extend(labels)

                elif res_type == "choices" and from_name == "exec_score":
                    choices = value.get("choices", [])
                    if choices:
                        related_id = result.get("id")
                        if related_id in maneuver_entries:
                            maneuver_entries[related_id]["score"] = choices[0]
                        else:
                            entry = maneuver_entries.setdefault(
                                related_id,
                                {
                                    "id": related_id,
                                    "start": value.get("start"),
                                    "end": value.get("end"),
                                    "labels": [],
                                    "score": choices[0],
                                },
                            )

            for entry in maneuver_entries.values():
                labels = entry.get("labels", [])
                if not labels:
                    continue

                score = entry.get("score") or ""
                maneuver_type = labels[-1]

                results.append(
                    {
                        "start": entry.get("start", 0.0),
                        "end": entry.get("end", 0.0),
                        "labels": [score, maneuver_type],
                    }
                )

        return results if results else None


def main():
    """Example usage of the data loader with centralized data selection."""
    print("⚠️  NOTE: This is a legacy example. Use DataSelectionManager for production.")
    print(
        "    The DataSelectionManager provides centralized, deterministic data selection."
    )
    print("    See utils/data_selection_manager.py for the recommended approach.")

    # Example configuration
    config = {
        "data_source": {
            "base_data_path": "./data/SD_02_SURF_FOOTAGE_PREPT",
            "video_clips": {
                "h264_path": "03_CLIPPED/h264",
                "ffv1_path": "03_CLIPPED/ffv1",
                "input_format": "h264",
            },
            "annotations": {
                "labels_path": "04_ANNOTATED/EXPORTED-MANEUVER-LABELS",
                "sony_300_labels": "sony_300",
                "sony_70_labels": "sony_70",
            },
            "splits": {
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "random_seed": 42,
                "zoom_handling": {
                    "enabled": True,
                    "balanced_distribution": True,
                    "target_distribution": {
                        "default": 0.33,
                        "wide": 0.33,
                        "full": 0.34,
                    },
                },
            },
        }
    }

    # Initialize data loader
    loader = SurfingDataLoader(config)

    # Load annotations and discover clips
    loader.load_annotations()
    clips = loader.discover_video_clips("h264")

    print(f"\nDiscovered {len(clips)} clips with annotations")

    # Show zoom distribution
    from collections import Counter

    zoom_dist = Counter(clip.zoom_level for clip in clips)
    print(f"Zoom distribution: {dict(zoom_dist)}")

    # Show camera distribution
    camera_dist = Counter(clip.camera for clip in clips)
    print(f"Camera distribution: {dict(camera_dist)}")

    # Example of loading from a manifest (if one exists)
    manifest_example = "./data_selections/example_selection.json"
    if Path(manifest_example).exists():
        print(f"\nLoading maneuvers from manifest: {manifest_example}")
        maneuvers = loader.load_maneuvers_from_manifest(manifest_example)
        print(f"Loaded {len(maneuvers)} maneuvers from manifest")
    else:
        print(f"\nNo manifest found at {manifest_example}")
        print("Use DataSelectionManager to generate selection manifests.")


if __name__ == "__main__":
    main()
