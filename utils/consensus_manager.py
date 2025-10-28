"""
Consensus Manager for Coordinating Pseudo-Ground-Truth Generation

Orchestrates consensus generation with leave-one-out validation and caching.
Manages which models to use for each target model's consensus to prevent
circular reasoning during Optuna optimization.
"""

import json
import logging
import numpy as np
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
        Uses per-maneuver cache files for efficient loading and scalability.
        
        Consensus predictions are generated lazily - if a maneuver's consensus
        doesn't exist in the shared cache, it's generated on-demand and saved.

        Args:
            maneuvers: List of Maneuver objects to process
            target_model: Model being optimized
            phase: Phase name ('optuna' or 'comparison')

        Returns:
            Dictionary mapping maneuver_id -> consensus frames
        """
        # Create cache subdirectory for this model/phase
        cache_subdir = self.cache_dir / f"{target_model}_{phase}"
        cache_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔧 CONSENSUS GT FOR {target_model.upper()} ({phase})")
        print(f"   Maneuvers to process: {len(maneuvers)}")

        logger.info(
            f"Generating consensus GT for {target_model} ({phase} phase)\n"
            f"  Processing {len(maneuvers)} maneuvers (using lazy shared cache)..."
        )

        # Determine which models to use (leave-one-out)
        consensus_models = self.get_consensus_models_for_target(target_model)
        print(f"   Consensus models: {', '.join(consensus_models)}")

        # Check cache and generate for missing maneuvers
        gt_data = {}
        cache_hits = 0
        cache_misses = 0
        stats = {
            "total_maneuvers": len(maneuvers),
            "successful": 0,
            "failed": 0,
            "total_frames": 0,
            "avg_confidence": [],
        }

        for maneuver in tqdm(maneuvers, desc=f"Consensus GT for {target_model}"):
            maneuver_cache_file = cache_subdir / f"{maneuver.maneuver_id}.json"

            # Try to load from cache first
            if maneuver_cache_file.exists():
                try:
                    maneuver_data = self._load_maneuver_from_cache(maneuver_cache_file)
                    gt_data[maneuver.maneuver_id] = maneuver_data
                    cache_hits += 1

                    # Update stats
                    stats["successful"] += 1
                    stats["total_frames"] += maneuver_data["num_frames"]
                    if maneuver_data["frames"]:
                        avg_conf = sum(
                            frame["confidence"].mean()
                            for frame in maneuver_data["frames"]
                        ) / len(maneuver_data["frames"])
                        stats["avg_confidence"].append(avg_conf)
                    continue
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached consensus for {maneuver.maneuver_id}: {e}, regenerating"
                    )
                    cache_misses += 1

            # Generate fresh consensus if not in cache or cache load failed
            cache_misses += 1
            try:
                # Get video path from maneuver
                video_path = maneuver.file_path

                # Generate consensus frames from scratch
                consensus_frames = self.generator.generate_consensus_for_maneuver(
                    model_names=consensus_models,
                    video_path=video_path,
                    maneuver=maneuver,
                )

                # Store consensus data
                maneuver_data = {
                    "maneuver_id": maneuver.maneuver_id,
                    "maneuver_type": maneuver.maneuver_type,
                    "frames": consensus_frames,
                    "source_models": consensus_models,
                    "num_frames": len(consensus_frames),
                }
                gt_data[maneuver.maneuver_id] = maneuver_data

                # Save to individual cache file
                self._save_maneuver_to_cache(maneuver_cache_file, maneuver_data)

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

        # Log summary
        print(f"\n✅ CONSENSUS GENERATION COMPLETE FOR {target_model.upper()}")
        print(f"   Success: {stats['successful']}/{stats['total_maneuvers']} maneuvers")
        print(f"   Cache hits: {cache_hits}, Cache misses: {cache_misses}")
        print(f"   Total frames: {stats['total_frames']}")
        print(f"   Avg confidence: {stats['avg_confidence']:.3f}")

        # Calculate time saved
        time_saved_min = (
            cache_hits * 1.5
        )  # Assume ~90 sec per maneuver (3 models × 30 sec)
        if cache_hits > 0:
            print(f"   ⚡ Time saved from cache: ~{time_saved_min:.1f} minutes\n")
        else:
            print(f"   Cache directory: {cache_subdir}\n")

        logger.info(
            f"✓ Consensus generation complete for {target_model}:\n"
            f"  Success: {stats['successful']}/{stats['total_maneuvers']} maneuvers\n"
            f"  Cache hits: {cache_hits}, misses: {cache_misses}\n"
            f"  Total frames: {stats['total_frames']}\n"
            f"  Avg confidence: {stats['avg_confidence']:.3f}\n"
            f"  Cache directory: {cache_subdir}"
        )

        return gt_data

    def _save_maneuver_to_cache(
        self,
        cache_file: Path,
        maneuver_data: Dict,
    ):
        """
        Save single maneuver consensus data to cache file.
        Per-maneuver files for efficient loading and scalability.

        Args:
            cache_file: Path to cache file for this maneuver
            maneuver_data: Consensus data for single maneuver
        """
        # Convert numpy arrays to lists for JSON serialization
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

        # Create cache structure
        cache_data = {
            "version": "2.0",  # Updated version for per-maneuver format
            "maneuver_id": maneuver_data["maneuver_id"],
            "maneuver_type": maneuver_data["maneuver_type"],
            "source_models": maneuver_data["source_models"],
            "num_frames": maneuver_data["num_frames"],
            "frames": frames_serializable,
            "metadata": {
                "quality_filter_config": {
                    "weights": self.quality_filter.weights,
                    "percentile_schedule": self.quality_filter.percentile_schedule,
                },
            },
        }

        # Save to file atomically (write to temp file, then rename)
        temp_file = cache_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            temp_file.replace(cache_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e

        logger.debug(f"Saved consensus cache: {cache_file.name}")

    def _load_maneuver_from_cache(self, cache_file: Path) -> Dict[str, Any]:
        """
        Load single maneuver consensus data from cache file.

        Args:
            cache_file: Path to cache file for this maneuver

        Returns:
            Consensus data for single maneuver
        """
        import numpy as np

        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        # Convert lists back to numpy arrays
        frames_with_arrays = []
        for frame in cache_data["frames"]:
            frame_with_arrays = {
                "keypoints": np.array(frame["keypoints"]),
                "confidence": np.array(frame["confidence"]),
                "source_models": frame["source_models"],
                "num_contributing_models": frame["num_contributing_models"],
            }
            frames_with_arrays.append(frame_with_arrays)

        maneuver_data = {
            "maneuver_id": cache_data["maneuver_id"],
            "maneuver_type": cache_data["maneuver_type"],
            "frames": frames_with_arrays,
            "source_models": cache_data["source_models"],
            "num_frames": cache_data["num_frames"],
        }

        logger.debug(f"Loaded consensus cache: {cache_file.name}")
        return maneuver_data

    def clear_cache(
        self, target_model: Optional[str] = None, phase: Optional[str] = None
    ):
        """
        Clear cached consensus data.
        Works with per-maneuver cache directory structure.

        Args:
            target_model: If specified, only clear cache for this model
            phase: If specified, only clear cache for this phase
        """
        import shutil

        if target_model and phase:
            cache_subdir = self.cache_dir / f"{target_model}_{phase}"
            if cache_subdir.exists() and cache_subdir.is_dir():
                shutil.rmtree(cache_subdir)
                logger.info(f"Cleared cache: {cache_subdir.name}")
        elif target_model:
            # Clear all phases for this model
            for cache_subdir in self.cache_dir.glob(f"{target_model}_*"):
                if cache_subdir.is_dir():
                    shutil.rmtree(cache_subdir)
                    logger.info(f"Cleared cache: {cache_subdir.name}")
        else:
            # Clear all caches
            for cache_subdir in self.cache_dir.iterdir():
                if cache_subdir.is_dir():
                    shutil.rmtree(cache_subdir)
            logger.info(f"Cleared all caches in {self.cache_dir}")

    def get_cached_models(self, phase: str = "optuna") -> List[str]:
        """
        Get list of models that have cached consensus data.
        Works with per-maneuver cache directory structure.

        Args:
            phase: Phase to check ('optuna' or 'comparison')

        Returns:
            List of model names with cached data
        """
        cached_models = []
        for cache_subdir in self.cache_dir.iterdir():
            if cache_subdir.is_dir() and cache_subdir.name.endswith(f"_{phase}"):
                model_name = cache_subdir.name.replace(f"_{phase}", "")
                cached_models.append(model_name)
        return cached_models

    def get_cache_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the cache.
        Works with per-maneuver cache directory structure.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "total_cache_files": 0,
            "cache_directory": str(self.cache_dir),
            "cached_model_phases": {},
        }

        for cache_subdir in self.cache_dir.iterdir():
            if not cache_subdir.is_dir():
                continue

            model_phase = cache_subdir.name
            cache_files = list(cache_subdir.glob("*.json"))

            total_size = sum(f.stat().st_size for f in cache_files)

            stats["cached_model_phases"][model_phase] = {
                "num_maneuvers": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "avg_file_size_kb": (
                    (total_size / len(cache_files) / 1024) if cache_files else 0
                ),
            }
            stats["total_cache_files"] += len(cache_files)

        return stats


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
