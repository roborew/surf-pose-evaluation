"""
Consensus pseudo-ground-truth generator for pose estimation validation.

Generates high-quality consensus annotations by aggregating predictions from
multiple pose estimation models with adaptive quality filtering.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import cv2

from utils.quality_filter import AdaptiveQualityFilter

logger = logging.getLogger(__name__)


@dataclass
class ConsensusFrame:
    """Single frame consensus data."""

    frame_idx: int
    consensus_keypoints: np.ndarray  # (K, 2) or (K, 3)
    quality_scores: np.ndarray  # (K,)
    contributing_models: List[str]
    per_model_predictions: Dict[str, Dict]
    threshold_used: float
    coverage: float


@dataclass
class ConsensusManeuver:
    """Maneuver-level consensus data."""

    maneuver_id: str
    start_frame: int
    end_frame: int
    frames: List[ConsensusFrame]
    maneuver_metadata: Dict


@dataclass
class ConsensusDataset:
    """Complete consensus dataset for a validation set."""

    dataset_type: str  # 'optuna_validation' or 'comparison_test'
    consensus_models: List[str]
    excluded_model: Optional[str]  # For leave-one-out
    clips: Dict[str, List[ConsensusManeuver]]  # clip_id -> maneuvers
    generation_config: Dict
    quality_stats: Dict


class ConsensusGenerator:
    """
    Generate consensus pseudo-ground-truth from multiple pose models.

    Implements leave-one-out validation to prevent circular reasoning.
    """

    def __init__(
        self,
        consensus_models: List[str],
        quality_filter: AdaptiveQualityFilter,
        cache_path: str,
        config: Dict,
    ):
        """
        Initialize consensus generator.

        Args:
            consensus_models: List of model names for consensus
                (e.g., ['yolov8', 'pytorch_pose', 'mmpose'])
            quality_filter: Adaptive quality filter instance
            cache_path: Path to cache consensus data
            config: Configuration dictionary
        """
        self.consensus_models = consensus_models
        self.quality_filter = quality_filter
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Model instances will be loaded on demand
        self.model_instances = {}

        logger.info(f"Initialized ConsensusGenerator:")
        logger.info(f"  Models: {consensus_models}")
        logger.info(f"  Cache: {cache_path}")

    def load_model(self, model_name: str):
        """Load model instance on demand."""
        if model_name in self.model_instances:
            return self.model_instances[model_name]

        logger.info(f"Loading model: {model_name}")

        # Import model wrappers
        if model_name == "yolov8":
            from models.yolov8_wrapper import YOLOv8PoseModel

            model = YOLOv8PoseModel()
        elif model_name == "pytorch_pose":
            from models.pytorch_pose_wrapper import PyTorchPoseModel

            model = PyTorchPoseModel()
        elif model_name == "mmpose":
            from models.mmpose_wrapper import MMPoseModel

            model = MMPoseModel()
        elif model_name == "mediapipe":
            from models.mediapipe_wrapper import MediaPipePoseModel

            model = MediaPipePoseModel()
        elif model_name == "blazepose":
            from models.blazepose_wrapper import BlazePoseModel

            model = BlazePoseModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_instances[model_name] = model
        return model

    def generate_consensus_annotations(
        self,
        clips: List,  # List of ManeuverClip objects
        output_path: str,
        dataset_type: str = "validation",
        exclude_model: Optional[str] = None,
    ) -> ConsensusDataset:
        """
        Generate consensus annotations for a set of clips.

        Args:
            clips: List of ManeuverClip objects to process
            output_path: Path to save consensus data
            dataset_type: Type of dataset ('optuna_validation' or 'comparison_test')
            exclude_model: Model to exclude for leave-one-out validation

        Returns:
            ConsensusDataset object
        """
        logger.info(f"Generating consensus for {len(clips)} clips...")
        logger.info(f"Dataset type: {dataset_type}")
        if exclude_model:
            logger.info(f"Leave-one-out: excluding {exclude_model}")

        # Determine active models
        active_models = [m for m in self.consensus_models if m != exclude_model]

        if len(active_models) < 2:
            raise ValueError(
                f"Need at least 2 models for consensus, got {len(active_models)}"
            )

        logger.info(f"Active consensus models: {active_models}")

        # Generate consensus for each clip
        consensus_clips = {}
        quality_stats = {
            "total_frames": 0,
            "total_keypoints": 0,
            "filtered_keypoints": 0,
            "coverage_per_clip": [],
            "quality_score_distribution": [],
        }

        for clip in tqdm(clips, desc="Processing clips"):
            clip_consensus = self._process_clip(clip, active_models, quality_stats)
            consensus_clips[clip.clip_id] = clip_consensus

        # Create dataset object
        dataset = ConsensusDataset(
            dataset_type=dataset_type,
            consensus_models=active_models,
            excluded_model=exclude_model,
            clips=consensus_clips,
            generation_config={
                "quality_filter": {
                    "weights": self.quality_filter.weights,
                    "percentile_schedule": self.quality_filter.percentile_schedule,
                }
            },
            quality_stats=quality_stats,
        )

        # Save to disk
        self._save_consensus_dataset(dataset, output_path)

        # Log summary
        avg_coverage = (
            np.mean(quality_stats["coverage_per_clip"])
            if quality_stats["coverage_per_clip"]
            else 0
        )
        logger.info(f"Consensus generation complete:")
        logger.info(f"  Total frames: {quality_stats['total_frames']}")
        logger.info(f"  Average coverage: {avg_coverage:.1%}")
        logger.info(f"  Saved to: {output_path}")

        return dataset

    def _process_clip(
        self, clip, active_models: List[str], quality_stats: Dict
    ) -> List[ConsensusManeuver]:
        """Process single clip and generate consensus for all maneuvers."""
        clip_consensus = []

        # Load video
        video_path = clip.video_path
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.warning(f"Failed to open video: {video_path}")
            return clip_consensus

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            logger.warning(f"No frames in video: {video_path}")
            return clip_consensus

        # Process each maneuver in clip
        for maneuver in clip.maneuvers:
            maneuver_consensus = self._process_maneuver(
                frames, maneuver, active_models, quality_stats
            )
            clip_consensus.append(maneuver_consensus)

        return clip_consensus

    def _process_maneuver(
        self,
        frames: List[np.ndarray],
        maneuver,
        active_models: List[str],
        quality_stats: Dict,
    ) -> ConsensusManeuver:
        """Process single maneuver and generate frame-by-frame consensus."""
        start_frame = maneuver.start_frame
        end_frame = maneuver.end_frame

        # Extract maneuver frames
        maneuver_frames = frames[start_frame:end_frame]

        # Run all models on all frames
        model_predictions = {}
        for model_name in active_models:
            model = self.load_model(model_name)
            predictions = self._run_model_on_frames(model, maneuver_frames)
            model_predictions[model_name] = predictions

        # Generate consensus for each frame
        consensus_frames = []
        for frame_idx, frame in enumerate(maneuver_frames):
            consensus_frame = self._generate_frame_consensus(
                frame_idx + start_frame,
                model_predictions,
                frame_idx,
                active_models,
                quality_stats,
            )
            consensus_frames.append(consensus_frame)

        return ConsensusManeuver(
            maneuver_id=maneuver.maneuver_id,
            start_frame=start_frame,
            end_frame=end_frame,
            frames=consensus_frames,
            maneuver_metadata={
                "maneuver_type": getattr(maneuver, "maneuver_type", "unknown"),
                "execution_score": getattr(maneuver, "execution_score", None),
            },
        )

    def _run_model_on_frames(self, model, frames: List[np.ndarray]) -> List[Dict]:
        """Run model inference on sequence of frames."""
        predictions = []

        for frame in frames:
            try:
                result = model.predict(frame)
                predictions.append(result)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                predictions.append(
                    {"keypoints": np.zeros((1, 17, 3)), "confidence": np.zeros((1, 17))}
                )

        return predictions

    def _generate_frame_consensus(
        self,
        global_frame_idx: int,
        model_predictions: Dict[str, List[Dict]],
        local_frame_idx: int,
        active_models: List[str],
        quality_stats: Dict,
    ) -> ConsensusFrame:
        """
        Generate consensus for a single frame from multiple model predictions.

        Aggregates keypoints from all models using confidence-weighted mean.
        """
        # Collect predictions from all models for this frame
        all_keypoints = []
        all_confidences = []
        per_model_data = {}

        for model_name in active_models:
            pred = model_predictions[model_name][local_frame_idx]

            # Extract keypoints and confidence
            keypoints = pred.get("keypoints", np.array([]))
            confidence = pred.get("confidence", np.array([]))

            if len(keypoints) > 0:
                # Take first person if multiple detected
                if len(keypoints.shape) == 3:
                    keypoints = keypoints[0]
                if len(confidence.shape) == 2:
                    confidence = confidence[0]

                all_keypoints.append(keypoints)
                all_confidences.append(confidence)

                per_model_data[model_name] = {
                    "keypoints": keypoints.tolist(),
                    "confidence": confidence.tolist(),
                }

        if not all_keypoints:
            # No valid predictions
            quality_stats["total_frames"] += 1
            return ConsensusFrame(
                frame_idx=global_frame_idx,
                consensus_keypoints=np.array([]),
                quality_scores=np.array([]),
                contributing_models=[],
                per_model_predictions={},
                threshold_used=0.0,
                coverage=0.0,
            )

        # Convert to arrays
        all_keypoints = np.array(all_keypoints)  # (M, K, D)
        all_confidences = np.array(all_confidences)  # (M, K)

        # Normalize keypoint dimensions (handle 2D vs 3D)
        if all_keypoints.shape[-1] == 2:
            # Pad to 3D for consistency
            padding = np.zeros((*all_keypoints.shape[:-1], 1))
            all_keypoints = np.concatenate([all_keypoints, padding], axis=-1)

        # Compute confidence-weighted mean
        weights = all_confidences / (
            np.sum(all_confidences, axis=0, keepdims=True) + 1e-8
        )
        consensus_keypoints = np.sum(
            all_keypoints * weights[:, :, np.newaxis], axis=0
        )  # (K, D)

        # Compute quality metrics
        mean_confidence = np.mean(all_confidences, axis=0)  # (K,)

        # Stability: variance across models (lower = better agreement)
        keypoint_variance = np.var(all_keypoints, axis=0)  # (K, D)
        mean_variance = np.mean(keypoint_variance, axis=1)  # (K,)
        stability = np.exp(-mean_variance / 100.0)  # Normalize and convert to [0,1]

        # Completeness: fraction of models that detected each keypoint
        detected = all_confidences > 0.3
        completeness = np.mean(detected, axis=0)  # (K,)

        # Composite quality scores
        quality_scores = self.quality_filter.compute_composite_scores_batch(
            mean_confidence, stability, completeness
        )

        # Update stats
        quality_stats["total_frames"] += 1
        quality_stats["total_keypoints"] += len(consensus_keypoints)
        quality_stats["quality_score_distribution"].extend(quality_scores.tolist())

        return ConsensusFrame(
            frame_idx=global_frame_idx,
            consensus_keypoints=consensus_keypoints[:, :2],  # Return 2D for now
            quality_scores=quality_scores,
            contributing_models=active_models,
            per_model_predictions=per_model_data,
            threshold_used=0.5,  # Will be applied during filtering
            coverage=1.0,
        )

    def _save_consensus_dataset(self, dataset: ConsensusDataset, output_path: str):
        """Save consensus dataset to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main metadata
        metadata = {
            "dataset_type": dataset.dataset_type,
            "consensus_models": dataset.consensus_models,
            "excluded_model": dataset.excluded_model,
            "generation_config": dataset.generation_config,
            "quality_stats": dataset.quality_stats,
            "num_clips": len(dataset.clips),
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save clips separately for memory efficiency
        for clip_id, maneuvers in dataset.clips.items():
            clip_data = {
                "clip_id": clip_id,
                "maneuvers": [self._serialize_maneuver(m) for m in maneuvers],
            }

            clip_file = output_path / f"{clip_id}.json"
            with open(clip_file, "w") as f:
                json.dump(clip_data, f)

        logger.info(f"Saved consensus dataset to {output_path}")

    def _serialize_maneuver(self, maneuver: ConsensusManeuver) -> Dict:
        """Convert maneuver to JSON-serializable dict."""
        return {
            "maneuver_id": maneuver.maneuver_id,
            "start_frame": maneuver.start_frame,
            "end_frame": maneuver.end_frame,
            "maneuver_metadata": maneuver.maneuver_metadata,
            "frames": [
                {
                    "frame_idx": f.frame_idx,
                    "consensus_keypoints": f.consensus_keypoints.tolist(),
                    "quality_scores": f.quality_scores.tolist(),
                    "contributing_models": f.contributing_models,
                    "per_model_predictions": f.per_model_predictions,
                    "threshold_used": f.threshold_used,
                    "coverage": f.coverage,
                }
                for f in maneuver.frames
            ],
        }

    def _verify_no_data_leakage(
        self, optuna_clips: List, comparison_clips: List, consensus_session: str
    ):
        """Verify no session overlap between data splits."""

        def extract_session(clip):
            # Extract session name from clip path or ID
            clip_path = (
                str(clip.video_path) if hasattr(clip, "video_path") else clip.clip_id
            )
            for part in clip_path.split("/"):
                if "SESSION_" in part:
                    # Extract base session name
                    session = part.split("_clip_")[0]
                    # Remove zoom variants
                    session = session.replace("_FULL", "").replace("_WIDE", "")
                    return session
            return None

        optuna_sessions = set(
            extract_session(c) for c in optuna_clips if extract_session(c) is not None
        )
        comparison_sessions = set(
            extract_session(c)
            for c in comparison_clips
            if extract_session(c) is not None
        )

        # Check no overlap with consensus session
        if consensus_session in optuna_sessions:
            raise ValueError(
                f"Data leakage: consensus session {consensus_session} in Optuna set"
            )

        if consensus_session in comparison_sessions:
            raise ValueError(
                f"Data leakage: consensus session {consensus_session} in comparison set"
            )

        # Check no overlap between Optuna and comparison
        overlap = optuna_sessions & comparison_sessions
        if overlap:
            raise ValueError(
                f"Data leakage: sessions {overlap} appear in both Optuna and comparison"
            )

        logger.info("✅ Data leakage check passed")
        logger.info(f"  Consensus session: {consensus_session}")
        logger.info(f"  Optuna sessions: {len(optuna_sessions)}")
        logger.info(f"  Comparison sessions: {len(comparison_sessions)}")

    def generate_validation_sets(
        self, optuna_clips: List, comparison_clips: List, consensus_session: str
    ) -> Tuple[ConsensusDataset, ConsensusDataset]:
        """
        Generate consensus for both Optuna and comparison validation sets.

        Args:
            optuna_clips: Clips for Optuna validation
            comparison_clips: Clips for comparison testing
            consensus_session: Session used for consensus generation (to check for leakage)

        Returns:
            Tuple of (optuna_consensus, comparison_consensus)
        """
        # Verify no data leakage
        self._verify_no_data_leakage(optuna_clips, comparison_clips, consensus_session)

        # Generate consensus for Optuna validation
        logger.info("Generating consensus for Optuna validation...")
        optuna_consensus = self.generate_consensus_annotations(
            clips=optuna_clips,
            output_path=str(self.cache_path / "optuna_validation"),
            dataset_type="optuna_validation",
        )

        # Generate consensus for comparison testing
        logger.info("Generating consensus for comparison testing...")
        comparison_consensus = self.generate_consensus_annotations(
            clips=comparison_clips,
            output_path=str(self.cache_path / "comparison_test"),
            dataset_type="comparison_test",
        )

        return optuna_consensus, comparison_consensus


class ConsensusLoader:
    """Load pre-generated consensus datasets."""

    @staticmethod
    def load(path: str, validation_type: str = "optuna_validation") -> ConsensusDataset:
        """
        Load consensus dataset from disk.

        Args:
            path: Path to consensus dataset directory
            validation_type: Type of validation set to load

        Returns:
            ConsensusDataset object
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load all clip files
        clips = {}
        for clip_file in path.glob("*.json"):
            if clip_file.name == "metadata.json":
                continue

            with open(clip_file, "r") as f:
                clip_data = json.load(f)

            # Deserialize maneuvers
            maneuvers = []
            for m_data in clip_data["maneuvers"]:
                frames = [
                    ConsensusFrame(
                        frame_idx=f["frame_idx"],
                        consensus_keypoints=np.array(f["consensus_keypoints"]),
                        quality_scores=np.array(f["quality_scores"]),
                        contributing_models=f["contributing_models"],
                        per_model_predictions=f["per_model_predictions"],
                        threshold_used=f["threshold_used"],
                        coverage=f["coverage"],
                    )
                    for f in m_data["frames"]
                ]

                maneuvers.append(
                    ConsensusManeuver(
                        maneuver_id=m_data["maneuver_id"],
                        start_frame=m_data["start_frame"],
                        end_frame=m_data["end_frame"],
                        frames=frames,
                        maneuver_metadata=m_data["maneuver_metadata"],
                    )
                )

            clips[clip_data["clip_id"]] = maneuvers

        return ConsensusDataset(
            dataset_type=metadata["dataset_type"],
            consensus_models=metadata["consensus_models"],
            excluded_model=metadata.get("excluded_model"),
            clips=clips,
            generation_config=metadata["generation_config"],
            quality_stats=metadata["quality_stats"],
        )
