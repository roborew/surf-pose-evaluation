"""
Consensus Manager for Coordinating Pseudo-Ground-Truth Generation

Orchestrates consensus generation with leave-one-out validation and caching.
Manages which models to use for each target model's consensus to prevent
circular reasoning during Optuna optimization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from utils.consensus_generator import ConsensusGenerator
from utils.quality_filter import AdaptiveQualityFilter

logger = logging.getLogger(__name__)


class ConsensusManager:
    """
    Manages consensus generation with leave-one-out validation.

    Coordinates the generation of pseudo-ground-truth for each model being
    optimized. Implements leave-one-out logic: when generating GT for model X,
    only use predictions from other models (Y and Z), not X itself.

    Handles caching to avoid redundant computation.
    """

    def __init__(
        self,
        consensus_models: List[str],
        quality_filter: AdaptiveQualityFilter,
        cache_dir: Path,
        config: Optional[Dict] = None,
    ):
        """
        Initialize consensus manager.

        Args:
            consensus_models: Strong models for consensus
                (e.g., ['yolov8', 'pytorch_pose', 'mmpose'])
            quality_filter: Quality filter instance
            cache_dir: Directory to cache consensus data
            config: Optional configuration dictionary
        """
        self.consensus_models = consensus_models
        self.quality_filter = quality_filter
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

        # Initialize generator
        self.generator = ConsensusGenerator(
            model_names=consensus_models,
            quality_filter=quality_filter,
            config=self.config,
        )

        logger.info(
            f"ConsensusManager initialized:\n"
            f"  Consensus models: {consensus_models}\n"
            f"  Cache directory: {cache_dir}"
        )

    def get_consensus_models_for_target(self, target_model: str) -> List[str]:
        """
        Get leave-one-out model list for target model.

        Implements leave-one-out logic:
        - If target is a strong model (in consensus_models), exclude it
        - If target is a weak model (mediapipe/blazepose), use all strong models

        Args:
            target_model: Model being optimized

        Returns:
            List of models to use for consensus
        """
        if target_model in self.consensus_models:
            # Strong model: exclude itself from consensus
            models = [m for m in self.consensus_models if m != target_model]
            logger.info(
                f"Leave-one-out for {target_model}: using {models} "
                f"(excluding {target_model})"
            )
        else:
            # Weak model: use all strong models
            models = self.consensus_models
            logger.info(
                f"Weak model {target_model}: using all consensus models {models}"
            )

        if len(models) < 2:
            logger.warning(
                f"Only {len(models)} models available for consensus. "
                "Minimum 2 recommended for robust consensus."
            )

        return models

    def generate_consensus_gt(
        self,
        maneuvers: List,
        target_model: str,
        phase: str = "optuna",
    ) -> Dict[str, Any]:
        """
        Generate and cache consensus ground truth for target model.

        Args:
            maneuvers: List of Maneuver objects to process
            target_model: Model being optimized
            phase: Phase name ('optuna' or 'comparison')

        Returns:
            Dictionary mapping maneuver_id -> consensus frames
        """
        # Check cache first
        cache_file = self.cache_dir / f"{target_model}_{phase}_gt.json"

        if cache_file.exists():
            logger.info(
                f"Loading cached consensus for {target_model} from {cache_file.name}"
            )
            return self._load_from_cache(cache_file)

        logger.info(
            f"Generating consensus GT for {target_model} ({phase} phase)\n"
            f"  Processing {len(maneuvers)} maneuvers..."
        )

        # Determine which models to use (leave-one-out)
        consensus_models = self.get_consensus_models_for_target(target_model)

        # Generate consensus for all maneuvers
        gt_data = {}
        stats = {
            "total_maneuvers": len(maneuvers),
            "successful": 0,
            "failed": 0,
            "total_frames": 0,
            "avg_confidence": [],
        }

        for maneuver in tqdm(maneuvers, desc=f"Consensus GT for {target_model}"):
            try:
                # Get video path from maneuver
                video_path = maneuver.file_path

                # Generate consensus frames
                consensus_frames = self.generator.generate_consensus_for_maneuver(
                    model_names=consensus_models,
                    video_path=video_path,
                    maneuver=maneuver,
                )

                # Store consensus data
                gt_data[maneuver.maneuver_id] = {
                    "maneuver_id": maneuver.maneuver_id,
                    "maneuver_type": maneuver.maneuver_type,
                    "frames": consensus_frames,
                    "source_models": consensus_models,
                    "num_frames": len(consensus_frames),
                }

                # Update statistics
                stats["successful"] += 1
                stats["total_frames"] += len(consensus_frames)

                # Calculate average confidence
                if consensus_frames:
                    avg_conf = sum(
                        frame["confidence"].mean() for frame in consensus_frames
                    ) / len(consensus_frames)
                    stats["avg_confidence"].append(avg_conf)

            except Exception as e:
                logger.error(
                    f"Failed to generate consensus for {maneuver.maneuver_id}: {e}"
                )
                stats["failed"] += 1

        # Calculate final statistics
        if stats["avg_confidence"]:
            stats["avg_confidence"] = float(
                sum(stats["avg_confidence"]) / len(stats["avg_confidence"])
            )
        else:
            stats["avg_confidence"] = 0.0

        # Save to cache
        self._save_to_cache(cache_file, gt_data, stats)

        # Log summary
        logger.info(
            f"✓ Consensus generation complete for {target_model}:\n"
            f"  Success: {stats['successful']}/{stats['total_maneuvers']} maneuvers\n"
            f"  Total frames: {stats['total_frames']}\n"
            f"  Avg confidence: {stats['avg_confidence']:.3f}\n"
            f"  Cached to: {cache_file.name}"
        )

        return gt_data

    def _save_to_cache(
        self,
        cache_file: Path,
        gt_data: Dict,
        stats: Dict,
    ):
        """
        Save consensus data to cache file.

        Args:
            cache_file: Path to cache file
            gt_data: Consensus ground truth data
            stats: Generation statistics
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for maneuver_id, maneuver_data in gt_data.items():
            frames_serializable = []
            for frame in maneuver_data["frames"]:
                frame_serializable = {
                    "keypoints": frame["keypoints"].tolist(),
                    "confidence": frame["confidence"].tolist(),
                    "source_models": frame["source_models"],
                    "num_contributing_models": (
                        frame["num_contributing_models"].tolist()
                        if hasattr(frame["num_contributing_models"], "tolist")
                        else frame["num_contributing_models"]
                    ),
                }
                frames_serializable.append(frame_serializable)

            serializable_data[maneuver_id] = {
                **maneuver_data,
                "frames": frames_serializable,
            }

        # Create cache structure
        cache_data = {
            "version": "1.0",
            "gt_data": serializable_data,
            "stats": stats,
            "metadata": {
                "num_maneuvers": len(gt_data),
                "quality_filter_config": {
                    "weights": self.quality_filter.weights,
                    "percentile_schedule": self.quality_filter.percentile_schedule,
                },
            },
        }

        # Save to file
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"Saved consensus cache: {cache_file}")

    def _load_from_cache(self, cache_file: Path) -> Dict[str, Any]:
        """
        Load consensus data from cache file.

        Args:
            cache_file: Path to cache file

        Returns:
            Consensus ground truth data
        """
        import numpy as np

        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        # Convert lists back to numpy arrays
        gt_data = {}
        for maneuver_id, maneuver_data in cache_data["gt_data"].items():
            frames_with_arrays = []
            for frame in maneuver_data["frames"]:
                frame_with_arrays = {
                    "keypoints": np.array(frame["keypoints"]),
                    "confidence": np.array(frame["confidence"]),
                    "source_models": frame["source_models"],
                    "num_contributing_models": frame["num_contributing_models"],
                }
                frames_with_arrays.append(frame_with_arrays)

            gt_data[maneuver_id] = {
                **maneuver_data,
                "frames": frames_with_arrays,
            }

        logger.debug(f"Loaded consensus cache: {cache_file}")
        return gt_data

    def clear_cache(
        self, target_model: Optional[str] = None, phase: Optional[str] = None
    ):
        """
        Clear cached consensus data.

        Args:
            target_model: If specified, only clear cache for this model
            phase: If specified, only clear cache for this phase
        """
        if target_model and phase:
            cache_file = self.cache_dir / f"{target_model}_{phase}_gt.json"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_file.name}")
        elif target_model:
            # Clear all phases for this model
            for cache_file in self.cache_dir.glob(f"{target_model}_*.json"):
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_file.name}")
        else:
            # Clear all caches
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info(f"Cleared all caches in {self.cache_dir}")

    def get_cached_models(self, phase: str = "optuna") -> List[str]:
        """
        Get list of models that have cached consensus data.

        Args:
            phase: Phase to check ('optuna' or 'comparison')

        Returns:
            List of model names with cached data
        """
        cached_models = []
        for cache_file in self.cache_dir.glob(f"*_{phase}_gt.json"):
            model_name = cache_file.stem.replace(f"_{phase}_gt", "")
            cached_models.append(model_name)
        return cached_models


if __name__ == "__main__":
    # Demo usage
    print("ConsensusManager Demo")
    print("=" * 50)

    from unittest.mock import Mock
    import tempfile

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize manager
        quality_filter = AdaptiveQualityFilter()
        manager = ConsensusManager(
            consensus_models=["yolov8", "pytorch_pose", "mmpose"],
            quality_filter=quality_filter,
            cache_dir=Path(temp_dir) / "consensus_cache",
        )

        print(f"\nConsensus models: {manager.consensus_models}")

        # Test leave-one-out logic
        print("\nLeave-one-out test:")
        for model in ["yolov8", "pytorch_pose", "mmpose", "mediapipe", "blazepose"]:
            models = manager.get_consensus_models_for_target(model)
            print(f"  {model} -> {models}")

        # Check cached models
        cached = manager.get_cached_models("optuna")
        print(f"\nCached models: {cached if cached else 'None'}")
