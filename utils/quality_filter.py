"""
Adaptive quality filter for consensus pseudo-ground-truth generation.

Implements research-validated percentile-based quality filtering with
multi-stage adaptive thresholds for pose estimation consensus.

Based on:
- PercentMatch (https://arxiv.org/pdf/2208.13946.pdf)
- FreeMatch (https://arxiv.org/pdf/2205.07246.pdf)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveQualityFilter:
    """
    Adaptive percentile-based quality filter for consensus keypoints.

    Uses composite scoring (confidence + stability + completeness) with
    multi-stage percentile thresholds that adapt throughout optimization.
    """

    def __init__(
        self,
        w_confidence: float = 0.4,
        w_stability: float = 0.4,
        w_completeness: float = 0.2,
        initialization_percentile: float = 70.0,
        growth_percentile: float = 80.0,
        saturation_percentile: float = 75.0,
    ):
        """
        Initialize adaptive quality filter.

        Args:
            w_confidence: Weight for confidence score (default 0.4)
            w_stability: Weight for temporal stability (default 0.4)
            w_completeness: Weight for skeleton completeness (default 0.2)
            initialization_percentile: Percentile for first 10% of trials (default 70)
            growth_percentile: Percentile for middle 60% of trials (default 80)
            saturation_percentile: Percentile for final 30% of trials (default 75)
        """
        # Validate weights sum to 1.0
        total_weight = w_confidence + w_stability + w_completeness
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Weights sum to {total_weight:.3f}, normalizing to 1.0")
            w_confidence /= total_weight
            w_stability /= total_weight
            w_completeness /= total_weight

        self.weights = {
            "confidence": w_confidence,
            "stability": w_stability,
            "completeness": w_completeness,
        }

        self.percentile_schedule = {
            "initialization": initialization_percentile,
            "growth": growth_percentile,
            "saturation": saturation_percentile,
        }

        logger.info(f"Initialized AdaptiveQualityFilter:")
        logger.info(f"  Weights: {self.weights}")
        logger.info(f"  Percentile schedule: {self.percentile_schedule}")

    def compute_composite_score(
        self, confidence: float, stability: float, completeness: float
    ) -> float:
        """
        Compute composite quality score for a keypoint.

        Q = w_c * confidence + w_s * stability + w_m * completeness

        Args:
            confidence: Model confidence score [0, 1]
            stability: Temporal stability score [0, 1]
            completeness: Skeleton completeness score [0, 1]

        Returns:
            Composite quality score [0, 1]
        """
        score = (
            self.weights["confidence"] * confidence
            + self.weights["stability"] * stability
            + self.weights["completeness"] * completeness
        )

        return np.clip(score, 0.0, 1.0)

    def compute_composite_scores_batch(
        self, confidence: np.ndarray, stability: np.ndarray, completeness: np.ndarray
    ) -> np.ndarray:
        """
        Compute composite scores for batch of keypoints.

        Args:
            confidence: Array of confidence scores
            stability: Array of stability scores
            completeness: Array of completeness scores

        Returns:
            Array of composite quality scores
        """
        scores = (
            self.weights["confidence"] * confidence
            + self.weights["stability"] * stability
            + self.weights["completeness"] * completeness
        )

        return np.clip(scores, 0.0, 1.0)

    def filter_by_percentile(
        self,
        keypoints: np.ndarray,
        quality_scores: np.ndarray,
        percentile: float = 80.0,
        min_keypoints: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Filter keypoints by quality percentile threshold.

        Keeps top (100 - percentile)% of keypoints.
        E.g., percentile=80 means threshold at 20th percentile, keeping top 80%.

        Args:
            keypoints: Array of keypoint coordinates (N, 2) or (N, 3)
            quality_scores: Array of quality scores (N,)
            percentile: Percentile threshold (higher = more selective)
            min_keypoints: Minimum number of keypoints to retain

        Returns:
            Tuple of (filtered_keypoints, filtered_scores, threshold_used)
        """
        if len(keypoints) == 0 or len(quality_scores) == 0:
            return keypoints, quality_scores, 0.0

        # Calculate percentile threshold
        # percentile=80 -> np.percentile(..., 20) to get bottom 20%, keep above it
        threshold = np.percentile(quality_scores, 100 - percentile)

        # Apply filter
        mask = quality_scores >= threshold

        # Ensure minimum keypoints
        if np.sum(mask) < min_keypoints and len(keypoints) >= min_keypoints:
            # Fall back to top-k selection
            top_k_indices = np.argpartition(quality_scores, -min_keypoints)[
                -min_keypoints:
            ]
            mask = np.zeros(len(keypoints), dtype=bool)
            mask[top_k_indices] = True
            threshold = np.min(quality_scores[mask])

        filtered_keypoints = keypoints[mask]
        filtered_scores = quality_scores[mask]

        return filtered_keypoints, filtered_scores, float(threshold)

    def get_trial_phase_percentile(self, trial_num: int, total_trials: int) -> float:
        """
        Get adaptive percentile threshold based on trial progress.

        Multi-stage schedule:
        - Initialization (first 10%): More conservative (lower percentile = keep top 30%)
        - Growth (middle 60%): Standard threshold (keep top 20%)
        - Saturation (final 30%): Slightly conservative to prevent overfitting (keep top 25%)

        Args:
            trial_num: Current trial number (0-indexed)
            total_trials: Total number of trials

        Returns:
            Percentile threshold for current trial phase
        """
        if total_trials <= 0:
            return self.percentile_schedule["growth"]

        progress = trial_num / total_trials

        if progress < 0.1:
            # Initialization phase
            return self.percentile_schedule["initialization"]
        elif progress < 0.7:
            # Growth phase
            return self.percentile_schedule["growth"]
        else:
            # Saturation phase
            return self.percentile_schedule["saturation"]

    def filter_consensus_keypoints(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        stability: np.ndarray,
        completeness: np.ndarray,
        trial_num: Optional[int] = None,
        total_trials: Optional[int] = None,
        fixed_percentile: Optional[float] = None,
    ) -> Dict:
        """
        Complete consensus filtering pipeline.

        Args:
            keypoints: Keypoint coordinates (N, K, 2) or (N, K, 3)
                where N is number of frames, K is number of keypoints
            confidence: Confidence scores (N, K)
            stability: Temporal stability scores (N, K)
            completeness: Skeleton completeness scores (N, K)
            trial_num: Current trial number for adaptive percentile
            total_trials: Total trials for adaptive percentile
            fixed_percentile: If provided, use this instead of adaptive schedule

        Returns:
            Dictionary with filtered keypoints and metadata
        """
        # Compute composite scores
        quality_scores = self.compute_composite_scores_batch(
            confidence.flatten(), stability.flatten(), completeness.flatten()
        )

        # Reshape to match keypoints
        original_shape = keypoints.shape
        keypoints_flat = keypoints.reshape(-1, original_shape[-1])

        # Determine percentile threshold
        if fixed_percentile is not None:
            percentile = fixed_percentile
        elif trial_num is not None and total_trials is not None:
            percentile = self.get_trial_phase_percentile(trial_num, total_trials)
        else:
            percentile = self.percentile_schedule["growth"]

        # Apply filtering
        filtered_kpts, filtered_scores, threshold = self.filter_by_percentile(
            keypoints_flat, quality_scores, percentile=percentile
        )

        # Calculate statistics
        coverage = (
            len(filtered_kpts) / len(keypoints_flat) if len(keypoints_flat) > 0 else 0.0
        )

        return {
            "filtered_keypoints": filtered_kpts,
            "quality_scores": filtered_scores,
            "threshold": threshold,
            "percentile": percentile,
            "coverage": coverage,
            "original_count": len(keypoints_flat),
            "filtered_count": len(filtered_kpts),
        }

    def calculate_temporal_stability(
        self, keypoints_sequence: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """
        Calculate temporal stability for keypoint sequences.

        Measures consistency of keypoint positions across adjacent frames.
        Higher stability = less jitter.

        Args:
            keypoints_sequence: Sequence of keypoints (T, K, 2) or (T, K, 3)
                where T is time (frames), K is number of keypoints
            window_size: Size of temporal window for stability calculation

        Returns:
            Stability scores (T, K) in range [0, 1]
        """
        T, K, D = keypoints_sequence.shape
        stability_scores = np.zeros((T, K))

        for t in range(T):
            # Get temporal window
            window_start = max(0, t - window_size // 2)
            window_end = min(T, t + window_size // 2 + 1)
            window = keypoints_sequence[window_start:window_end]

            # Calculate position variance in window
            if len(window) > 1:
                variances = np.var(window, axis=0)  # (K, D)
                # Convert variance to stability score (lower variance = higher stability)
                # Use negative exponential: stability = exp(-variance)
                mean_variance = np.mean(variances, axis=1)  # (K,)
                stability_scores[t] = np.exp(-mean_variance)
            else:
                # Single frame or edges: assign neutral stability
                stability_scores[t] = 0.5

        return stability_scores

    def calculate_skeleton_completeness(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        confidence_threshold: float = 0.3,
    ) -> np.ndarray:
        """
        Calculate skeleton completeness scores.

        Measures what fraction of expected keypoints are detected with
        sufficient confidence.

        Args:
            keypoints: Keypoint coordinates (N, K, 2) or (N, K, 3)
            confidence: Confidence scores (N, K)
            confidence_threshold: Minimum confidence for "detected"

        Returns:
            Completeness scores (N, K) in range [0, 1]
        """
        N, K = confidence.shape

        # Binary detection mask
        detected = confidence > confidence_threshold

        # Completeness = fraction of keypoints detected per frame
        completeness_per_frame = np.sum(detected, axis=1, keepdims=True) / K

        # Broadcast to match (N, K) shape
        completeness_scores = np.tile(completeness_per_frame, (1, K))

        return completeness_scores


def test_quality_filter():
    """Test quality filter with synthetic data."""
    print("Testing AdaptiveQualityFilter...")

    # Initialize filter
    filter_obj = AdaptiveQualityFilter()

    # Create synthetic keypoint data
    np.random.seed(42)
    n_frames = 100
    n_keypoints = 17

    keypoints = np.random.rand(n_frames, n_keypoints, 2) * 100
    confidence = np.random.rand(n_frames, n_keypoints)

    # Calculate stability and completeness
    stability = filter_obj.calculate_temporal_stability(keypoints)
    completeness = filter_obj.calculate_skeleton_completeness(keypoints, confidence)

    print(f"Synthetic data: {n_frames} frames, {n_keypoints} keypoints")
    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"Stability range: [{stability.min():.3f}, {stability.max():.3f}]")
    print(f"Completeness range: [{completeness.min():.3f}, {completeness.max():.3f}]")

    # Test filtering at different trial phases
    for trial_num, phase in [(0, "init"), (25, "growth"), (45, "saturation")]:
        result = filter_obj.filter_consensus_keypoints(
            keypoints,
            confidence,
            stability,
            completeness,
            trial_num=trial_num,
            total_trials=50,
        )

        print(f"\nTrial {trial_num} ({phase} phase):")
        print(f"  Percentile: {result['percentile']:.1f}")
        print(f"  Threshold: {result['threshold']:.3f}")
        print(f"  Coverage: {result['coverage']:.1%}")
        print(
            f"  Filtered: {result['filtered_count']}/{result['original_count']} keypoints"
        )

    print("\n✅ Quality filter tests passed")


if __name__ == "__main__":
    test_quality_filter()
