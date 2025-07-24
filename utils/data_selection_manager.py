#!/usr/bin/env python3
"""
Data Selection Manager for Pose Evaluation
Handles centralized, deterministic clip and maneuver selection with complete data manifests
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import cv2

from data_handling.data_loader import SurfingDataLoader, VideoClip, Maneuver

logger = logging.getLogger(__name__)


class DataSelectionManager:
    """Manages centralized, deterministic data selection for pose evaluation"""

    def __init__(self, config: Dict, run_manager=None):
        """Initialize data selection manager

        Args:
            config: Complete evaluation configuration
            run_manager: RunManager instance for output paths
        """
        self.config = config
        self.run_manager = run_manager
        self.data_source_config = config["data_source"]

        # Initialize data loader for discovery
        self.data_loader = SurfingDataLoader(config)

        # Selection storage
        self.selections_dir = None
        if run_manager:
            self.selections_dir = run_manager.run_dir / "data_selections"
            self.selections_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🎯 DataSelectionManager initialized")

    def generate_phase_selections(
        self,
        optuna_max_clips: Optional[int] = None,
        comparison_max_clips: Optional[int] = None,
        random_seed: Optional[int] = None,
        video_format: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate data selections for both Optuna and comparison phases

        Args:
            optuna_max_clips: Max clips for Optuna phase
            comparison_max_clips: Max clips for comparison phase
            random_seed: Random seed for reproducible selections
            video_format: Video format to use

        Returns:
            Dictionary with paths to generated selection manifests
        """
        if random_seed is None:
            random_seed = self.data_source_config.get("splits", {}).get(
                "random_seed", 42
            )

        if video_format is None:
            video_format = self.data_source_config["video_clips"].get(
                "input_format", "h264"
            )

        logger.info(
            f"🎲 Generating phase selections with seed: {random_seed}, format: {video_format}"
        )

        # Set global random seed for deterministic selection
        random.seed(random_seed)

        # Discover all available clips with annotations
        logger.info("🔍 Discovering available clips...")
        all_clips = self._discover_clips_deterministically(video_format, random_seed)

        manifest_paths = {}

        # Generate comparison selection first (larger set) - this is the base dataset
        comparison_clips = []
        if comparison_max_clips:
            logger.info(
                f"📝 Generating comparison selection ({comparison_max_clips} clips)"
            )
            comparison_clips = self._select_clips_balanced(
                all_clips, comparison_max_clips, random_seed  # Use base seed
            )
            comparison_manifest = self._create_selection_manifest(
                "comparison",
                comparison_clips,
                comparison_max_clips,
                random_seed,
                video_format,
            )
            comparison_path = self._save_selection_manifest(
                comparison_manifest, "comparison_selection.json"
            )
            manifest_paths["comparison"] = comparison_path

        # Generate Optuna selection as subset of comparison clips
        if optuna_max_clips and comparison_clips:
            logger.info(
                f"📝 Generating Optuna selection ({optuna_max_clips} clips) as subset of comparison"
            )
            # Take first N clips from comparison set to ensure subset relationship
            optuna_clips = comparison_clips[
                : min(optuna_max_clips, len(comparison_clips))
            ]
            optuna_manifest = self._create_selection_manifest(
                "optuna", optuna_clips, optuna_max_clips, random_seed, video_format
            )
            optuna_path = self._save_selection_manifest(
                optuna_manifest, "optuna_selection.json"
            )
            manifest_paths["optuna"] = optuna_path
        elif optuna_max_clips and not comparison_clips:
            # Fallback: generate optuna independently if no comparison
            logger.info(
                f"📝 Generating independent Optuna selection ({optuna_max_clips} clips)"
            )
            optuna_clips = self._select_clips_balanced(
                all_clips, optuna_max_clips, random_seed
            )
            optuna_manifest = self._create_selection_manifest(
                "optuna", optuna_clips, optuna_max_clips, random_seed, video_format
            )
            optuna_path = self._save_selection_manifest(
                optuna_manifest, "optuna_selection.json"
            )
            manifest_paths["optuna"] = optuna_path
        elif optuna_max_clips is None:
            # CRITICAL FIX: Use full dataset for Optuna when no limit specified
            logger.info("📝 Generating Optuna selection (full dataset)")
            optuna_clips = all_clips  # Use all available clips
            optuna_manifest = self._create_selection_manifest(
                "optuna", optuna_clips, len(optuna_clips), random_seed, video_format
            )
            optuna_path = self._save_selection_manifest(
                optuna_manifest, "optuna_selection.json"
            )
            manifest_paths["optuna"] = optuna_path

        # Generate visualization selection (subset of comparison data)
        if manifest_paths.get("comparison"):
            logger.info("🎬 Generating visualization selection")
            viz_manifest = self._create_visualization_selection(comparison_manifest)
            viz_path = self._save_selection_manifest(
                viz_manifest, "visualization_selection.json"
            )
            manifest_paths["visualization"] = viz_path

        logger.info(f"✅ Generated {len(manifest_paths)} selection manifests")
        return manifest_paths

    def _discover_clips_deterministically(
        self, video_format: str, random_seed: int
    ) -> List[VideoClip]:
        """Discover clips using deterministic ordering

        Args:
            video_format: Video format to discover
            random_seed: Random seed for reproducible discovery

        Returns:
            List of discovered VideoClip objects
        """
        # Use the data loader but make selection deterministic
        original_seed = random.getstate()
        random.seed(random_seed)

        try:
            # Load annotations first
            self.data_loader.load_annotations()

            # Discover clips with deterministic selection
            clips = self.data_loader.discover_video_clips(video_format)

            # Sort clips deterministically by key attributes
            clips.sort(
                key=lambda c: (c.camera, c.session, c.base_clip_id, c.zoom_level)
            )

            logger.info(f"🔍 Discovered {len(clips)} clips deterministically")
            return clips

        finally:
            random.setstate(original_seed)

    def _select_clips_balanced(
        self, all_clips: List[VideoClip], max_clips: int, selection_seed: int
    ) -> List[VideoClip]:
        """Select clips with balanced zoom and camera distribution

        Args:
            all_clips: All available clips
            max_clips: Maximum number to select
            selection_seed: Seed for this specific selection

        Returns:
            Selected clips with balanced distribution
        """
        if len(all_clips) <= max_clips:
            return all_clips.copy()

        # Use separate seed for this selection
        original_seed = random.getstate()
        random.seed(selection_seed)

        try:
            # Group clips by camera and zoom for balanced selection
            grouped_clips = defaultdict(lambda: defaultdict(list))
            for clip in all_clips:
                grouped_clips[clip.camera][clip.zoom_level].append(clip)

            selected_clips = []
            cameras = list(grouped_clips.keys())
            zoom_levels = ["default", "wide", "full"]

            # Calculate target distribution
            clips_per_camera = max_clips // len(cameras)
            clips_per_zoom = clips_per_camera // len(zoom_levels)

            # Select clips maintaining balance
            for camera in cameras:
                camera_clips = []
                for zoom in zoom_levels:
                    available = grouped_clips[camera][zoom]
                    if available:
                        # Shuffle for randomness within constraints
                        random.shuffle(available)
                        take_count = min(clips_per_zoom, len(available))
                        camera_clips.extend(available[:take_count])

                # Fill remaining slots for this camera
                remaining_slots = clips_per_camera - len(camera_clips)
                if remaining_slots > 0:
                    # Get all remaining clips from this camera
                    all_camera_clips = []
                    for zoom in zoom_levels:
                        all_camera_clips.extend(grouped_clips[camera][zoom])

                    # Remove already selected
                    remaining_clips = [
                        c for c in all_camera_clips if c not in camera_clips
                    ]
                    random.shuffle(remaining_clips)
                    camera_clips.extend(remaining_clips[:remaining_slots])

                selected_clips.extend(camera_clips)

            # Fill any remaining slots globally
            remaining_slots = max_clips - len(selected_clips)
            if remaining_slots > 0:
                unselected = [c for c in all_clips if c not in selected_clips]
                random.shuffle(unselected)
                selected_clips.extend(unselected[:remaining_slots])

            # Sort final selection deterministically
            selected_clips.sort(
                key=lambda c: (c.camera, c.session, c.base_clip_id, c.zoom_level)
            )

            logger.info(
                f"📊 Selected {len(selected_clips)} clips with balanced distribution"
            )
            return selected_clips[:max_clips]

        finally:
            random.setstate(original_seed)

    def _create_selection_manifest(
        self,
        phase: str,
        selected_clips: List[VideoClip],
        max_clips: int,
        random_seed: int,
        video_format: str,
    ) -> Dict[str, Any]:
        """Create complete self-contained selection manifest

        Args:
            phase: Phase name (optuna/comparison)
            selected_clips: Selected VideoClip objects
            max_clips: Maximum clips requested
            random_seed: Random seed used
            video_format: Video format used

        Returns:
            Complete selection manifest dictionary
        """
        # Calculate distributions
        zoom_dist = Counter(clip.zoom_level for clip in selected_clips)
        camera_dist = Counter(clip.camera for clip in selected_clips)

        # Collect all maneuvers with complete data
        all_maneuvers = []
        maneuver_type_dist = Counter()

        manifest_clips = []

        for clip in selected_clips:
            # Get video metadata
            video_metadata = self._get_video_metadata(clip.file_path)

            # Process maneuvers for this clip
            clip_maneuvers = []
            maneuvers = clip.get_maneuvers()

            for i, maneuver in enumerate(maneuvers):
                maneuver_data = {
                    "maneuver_id": f"{clip.camera}_{clip.session}_{clip.base_clip_id}_{clip.zoom_level}_maneuver_{i}",
                    "start_time": maneuver.start_time,
                    "end_time": maneuver.end_time,
                    "start_frame": maneuver.start_frame,
                    "end_frame": maneuver.end_frame,
                    "duration": maneuver.duration,
                    "total_frames": maneuver.total_frames,
                    "labels": maneuver.annotation_data.get("labels", []),
                    "execution_score": maneuver.execution_score,
                    "maneuver_type": maneuver.maneuver_type,
                    "channel": maneuver.annotation_data.get("channel", 0),
                    "annotation_data": maneuver.annotation_data,
                }

                clip_maneuvers.append(maneuver_data)
                all_maneuvers.append(maneuver_data)
                maneuver_type_dist[maneuver.maneuver_type] += 1

            # Create clip entry with complete data
            clip_data = {
                "clip_id": f"{clip.camera}_{clip.session}_{clip.base_clip_id}_{clip.zoom_level}",
                "base_clip_id": clip.base_clip_id,
                "zoom_level": clip.zoom_level,
                "camera": clip.camera,
                "session": clip.session,
                "video_path": clip.file_path,
                "video_metadata": video_metadata,
                "maneuvers": clip_maneuvers,
                "maneuver_count": len(clip_maneuvers),
                "total_duration": clip.duration,
            }

            manifest_clips.append(clip_data)

        # Create complete manifest
        manifest = {
            "selection_metadata": {
                "phase": phase,
                "max_clips": max_clips,
                "actual_clips": len(selected_clips),
                "random_seed": random_seed,
                "video_format": video_format,
                "cameras": list(camera_dist.keys()),
                "selection_timestamp": datetime.now().isoformat(),
                "generator": "DataSelectionManager v1.0",
            },
            "distributions": {
                "zoom_distribution": dict(zoom_dist),
                "camera_distribution": dict(camera_dist),
                "maneuver_type_distribution": dict(maneuver_type_dist),
            },
            "selected_clips": manifest_clips,
            "summary": {
                "total_clips": len(selected_clips),
                "total_maneuvers": len(all_maneuvers),
                "total_duration": sum(clip.duration for clip in selected_clips),
                "avg_maneuvers_per_clip": (
                    len(all_maneuvers) / len(selected_clips) if selected_clips else 0
                ),
            },
        }

        logger.info(
            f"📄 Created {phase} manifest: {len(selected_clips)} clips, {len(all_maneuvers)} maneuvers"
        )
        return manifest

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using OpenCV

        Args:
            video_path: Path to video file

        Returns:
            Video metadata dictionary
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return {}

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": frame_count,
                "duration": duration,
            }

        except Exception as e:
            logger.error(f"Error getting video metadata for {video_path}: {e}")
            return {}

    def _create_visualization_selection(
        self, comparison_manifest: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create visualization selection from comparison data

        Args:
            comparison_manifest: Complete comparison selection manifest

        Returns:
            Visualization selection manifest
        """
        # Get visualization config
        viz_config = self.config.get("output", {}).get("visualization", {})
        max_examples = viz_config.get("max_examples_per_model", 10)

        # Select diverse maneuvers for visualization
        all_maneuvers = []
        for clip in comparison_manifest["selected_clips"]:
            for maneuver in clip["maneuvers"]:
                maneuver["source_clip"] = {
                    "clip_id": clip["clip_id"],
                    "video_path": clip["video_path"],
                    "video_metadata": clip["video_metadata"],
                }
                all_maneuvers.append(maneuver)

        # Select diverse maneuvers (different types, cameras, scores)
        selected_maneuvers = self._select_diverse_maneuvers(all_maneuvers, max_examples)

        # Create visualization manifest
        viz_manifest = {
            "selection_metadata": {
                "phase": "visualization",
                "max_maneuvers": max_examples,
                "actual_maneuvers": len(selected_maneuvers),
                "source_phase": "comparison",
                "selection_timestamp": datetime.now().isoformat(),
                "generator": "DataSelectionManager v1.0",
            },
            "selection_criteria": {
                "diversity_factors": ["maneuver_type", "camera", "execution_score"],
                "prioritize_high_scores": True,
                "ensure_type_coverage": True,
            },
            "selected_maneuvers": selected_maneuvers,
            "summary": {
                "total_maneuvers": len(selected_maneuvers),
                "maneuver_types": list(
                    set(m["maneuver_type"] for m in selected_maneuvers)
                ),
                "cameras": list(
                    set(
                        m["source_clip"]["clip_id"].split("_")[0]
                        + "_"
                        + m["source_clip"]["clip_id"].split("_")[1]
                        for m in selected_maneuvers
                    )
                ),
                "score_range": (
                    [
                        min(int(m["execution_score"]) for m in selected_maneuvers),
                        max(int(m["execution_score"]) for m in selected_maneuvers),
                    ]
                    if selected_maneuvers
                    else [0, 0]
                ),
            },
        }

        logger.info(
            f"🎬 Created visualization selection: {len(selected_maneuvers)} maneuvers"
        )
        return viz_manifest

    def _select_diverse_maneuvers(
        self, all_maneuvers: List[Dict], max_count: int
    ) -> List[Dict]:
        """Select diverse maneuvers for visualization

        Args:
            all_maneuvers: All available maneuvers
            max_count: Maximum maneuvers to select

        Returns:
            Diversely selected maneuvers
        """
        if len(all_maneuvers) <= max_count:
            return all_maneuvers

        # Group by type for diversity
        by_type = defaultdict(list)
        for maneuver in all_maneuvers:
            by_type[maneuver["maneuver_type"]].append(maneuver)

        selected = []
        types = list(by_type.keys())
        per_type = max(1, max_count // len(types))

        # Select best examples from each type
        for maneuver_type in types:
            type_maneuvers = by_type[maneuver_type]
            # Sort by execution score (higher is better)
            type_maneuvers.sort(key=lambda m: int(m["execution_score"]), reverse=True)
            selected.extend(type_maneuvers[:per_type])

        # Fill remaining slots with highest-scoring maneuvers
        remaining = max_count - len(selected)
        if remaining > 0:
            unselected = [m for m in all_maneuvers if m not in selected]
            unselected.sort(key=lambda m: int(m["execution_score"]), reverse=True)
            selected.extend(unselected[:remaining])

        return selected[:max_count]

    def _save_selection_manifest(self, manifest: Dict[str, Any], filename: str) -> str:
        """Save selection manifest to file

        Args:
            manifest: Complete manifest dictionary
            filename: Output filename

        Returns:
            Path to saved manifest file
        """
        if not self.selections_dir:
            # Fallback if no run manager
            self.selections_dir = Path("./data_selections")
            self.selections_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = self.selections_dir / filename

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"💾 Saved selection manifest: {manifest_path}")
        return str(manifest_path)

    @staticmethod
    def load_selection_manifest(manifest_path: str) -> Dict[str, Any]:
        """Load selection manifest from file

        Args:
            manifest_path: Path to manifest file

        Returns:
            Loaded manifest dictionary
        """
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            logger.info(f"📖 Loaded selection manifest: {manifest_path}")
            return manifest

        except Exception as e:
            logger.error(f"Failed to load manifest {manifest_path}: {e}")
            raise

    def get_manifest_summary(self, manifest_path: str) -> str:
        """Get human-readable summary of selection manifest

        Args:
            manifest_path: Path to manifest file

        Returns:
            Formatted summary string
        """
        manifest = self.load_selection_manifest(manifest_path)

        metadata = manifest["selection_metadata"]
        summary = manifest["summary"]
        distributions = manifest["distributions"]

        summary_text = f"""
📊 Selection Manifest Summary: {metadata['phase'].upper()}
{'='*50}
📅 Generated: {metadata['selection_timestamp']}
🎲 Random Seed: {metadata['random_seed']}
🎥 Video Format: {metadata['video_format']}

📈 Totals:
   • Clips: {summary['total_clips']}
   • Maneuvers: {summary['total_maneuvers']}
   • Duration: {summary['total_duration']:.1f} seconds
   • Avg Maneuvers/Clip: {summary['avg_maneuvers_per_clip']:.1f}

📱 Camera Distribution:
{chr(10).join([f'   • {cam}: {count}' for cam, count in distributions['camera_distribution'].items()])}

🔍 Zoom Distribution:
{chr(10).join([f'   • {zoom}: {count}' for zoom, count in distributions['zoom_distribution'].items()])}

🏄 Maneuver Types:
{chr(10).join([f'   • {mtype}: {count}' for mtype, count in distributions['maneuver_type_distribution'].items()])}
"""

        return summary_text.strip()
