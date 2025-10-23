"""
Adaptive Quality Filter for Consensus Pseudo-Ground-Truth

Implements research-validated percentile-based filtering with composite quality scoring
to ensure high-quality consensus annotations for Optuna optimization.

Based on PercentMatch and FreeMatch research:
- Composite scoring: Q = w_c·confidence + w_s·stability + w_m·completeness
- Adaptive percentile thresholds that adjust during optimization
- Multi-stage schedule (initialization/growth/saturation)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveQualityFilter:
    """
    Adaptive quality filter using percentile-based thresholds.

    Filters keypoints based on composite quality scores that combine:
    - Confidence: Model's confidence in keypoint detection
    - Stability: Temporal consistency of keypoint positions
    - Completeness: Fraction of skeleton detected

    Uses adaptive percentile thresholds that change based on optimization progress:
    - Initialization phase (0-10%): Conservative (70th percentile)
    - Growth phase (10-70%): Standard (80th percentile)
    - Saturation phase (70-100%): Prevent overfitting (75th percentile)
    """

    def __init__(
        self,
        w_confidence: float = 0.4,
        w_stability: float = 0.4,
        w_completeness: float = 0.2,
        initialization_percentile: float = 70.0,
        growth_percentile: float = 80.0,
        saturation_percentile: float = 75.0,
        min_quality_score: float = 0.3,
    ):
        """
        Initialize adaptive quality filter.

        Args:
            w_confidence: Weight for confidence score (default 0.4)
            w_stability: Weight for temporal stability (default 0.4)
            w_completeness: Weight for skeleton completeness (default 0.2)
            initialization_percentile: Percentile for early trials (default 70.0)
            growth_percentile: Percentile for middle trials (default 80.0)
            saturation_percentile: Percentile for late trials (default 75.0)
            min_quality_score: Minimum acceptable quality score (default 0.3)
        """
        # Normalize weights to sum to 1.0
        total_weight = w_confidence + w_stability + w_completeness
        self.weights = {
            "confidence": w_confidence / total_weight,
            "stability": w_stability / total_weight,
            "completeness": w_completeness / total_weight,
        }

        self.percentile_schedule = {
            "initialization": initialization_percentile,
            "growth": growth_percentile,
            "saturation": saturation_percentile,
        }

        self.min_quality_score = min_quality_score

        logger.info(
            f"AdaptiveQualityFilter initialized:\n"
            f"  Weights: confidence={self.weights['confidence']:.2f}, "
            f"stability={self.weights['stability']:.2f}, "
            f"completeness={self.weights['completeness']:.2f}\n"
            f"  Percentiles: init={initialization_percentile:.0f}%, "
            f"growth={growth_percentile:.0f}%, sat={saturation_percentile:.0f}%"
        )

    def calculate_quality_score(
        self,
        confidence: np.ndarray,
        stability: np.ndarray,
        completeness: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate composite quality score Q.

        Q = w_c·confidence + w_s·stability + w_m·completeness

        Args:
            confidence: Confidence scores (0-1), shape: (..., num_keypoints)
            stability: Stability scores (0-1), shape: (..., num_keypoints)
            completeness: Completeness scores (0-1), shape: (..., num_keypoints)

        Returns:
            Quality scores (0-1), same shape as inputs
        """
        quality = (
            self.weights["confidence"] * confidence
            + self.weights["stability"] * stability
            + self.weights["completeness"] * completeness
        )

        return np.clip(quality, 0.0, 1.0)

    def calculate_stability(
        self,
        keypoints_sequence: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        Calculate temporal stability of keypoints.

        Stability = exp(-variance) where variance is computed over a temporal window.
        Higher stability means more consistent keypoint positions.

        Args:
            keypoints_sequence: Keypoint positions over time
                Shape: (num_frames, num_persons, num_keypoints, 2)
            window_size: Window size for stability calculation (default 5 frames)

        Returns:
            Stability scores (0-1), shape: (num_frames, num_persons, num_keypoints)
        """
        num_frames = keypoints_sequence.shape[0]
        stability_scores = np.ones_like(
            keypoints_sequence[..., 0]
        )  # Default: fully stable

        # Calculate variance over sliding window
        for i in range(num_frames):
            window_start = max(0, i - window_size // 2)
            window_end = min(num_frames, i + window_size // 2 + 1)
            window = keypoints_sequence[window_start:window_end]

            # Calculate variance of x,y coordinates
            if len(window) > 1:
                variance = np.var(
                    window, axis=0
                )  # Shape: (num_persons, num_keypoints, 2)
                avg_variance = np.mean(variance, axis=-1)  # Average over x,y

                # Convert variance to stability: exp(-variance)
                # Normalize by typical keypoint distance (100 pixels)
                normalized_variance = avg_variance / (100.0**2)
                stability_scores[i] = np.exp(-normalized_variance)

        return stability_scores

    def calculate_completeness(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        confidence_threshold: float = 0.3,
    ) -> np.ndarray:
        """
        Calculate skeleton completeness.

        Completeness = (number of detected keypoints) / (total keypoints)

        Args:
            keypoints: Keypoint positions, shape: (..., num_keypoints, 2)
            confidence: Confidence scores, shape: (..., num_keypoints)
            confidence_threshold: Threshold for considering keypoint as detected

        Returns:
            Completeness scores (0-1), shape: (...)
        """
        # Count valid keypoints (confidence above threshold)
        valid_keypoints = confidence > confidence_threshold
        num_valid = np.sum(valid_keypoints, axis=-1)
        total_keypoints = keypoints.shape[-2]

        completeness = num_valid / total_keypoints
        return completeness

    def get_threshold_for_trial(
        self,
        trial_num: int,
        total_trials: int,
    ) -> float:
        """
        Get adaptive percentile threshold based on optimization progress.

        - Initialization (0-10%): 70th percentile (keep top 30%)
        - Growth (10-70%): 80th percentile (keep top 20%)
        - Saturation (70-100%): 75th percentile (keep top 25%)

        Args:
            trial_num: Current trial number (0-indexed)
            total_trials: Total number of trials

        Returns:
            Percentile threshold to use
        """
        progress = trial_num / max(total_trials, 1)

        if progress < 0.1:
            # Initialization: conservative filtering
            return self.percentile_schedule["initialization"]
        elif progress < 0.7:
            # Growth: standard filtering
            return self.percentile_schedule["growth"]
        else:
            # Saturation: prevent overfitting
            return self.percentile_schedule["saturation"]

    def filter_keypoints(
        self,
        keypoints: np.ndarray,
        quality_scores: np.ndarray,
        threshold_percentile: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter keypoints based on quality scores and percentile threshold.

        Keeps only keypoints with quality scores above the specified percentile.

        Args:
            keypoints: Keypoint positions, shape: (num_persons, num_keypoints, 2)
            quality_scores: Quality scores, shape: (num_persons, num_keypoints)
            threshold_percentile: Percentile threshold (e.g., 80.0 = keep top 20%)

        Returns:
            Tuple of (filtered_keypoints, mask) where:
            - filtered_keypoints: Same shape as input, filtered keypoints
            - mask: Boolean mask showing which keypoints passed filter
        """
        if len(quality_scores) == 0:
            return keypoints, np.zeros_like(quality_scores, dtype=bool)

        # Calculate threshold from percentile
        threshold = np.percentile(quality_scores, 100 - threshold_percentile)
        threshold = max(threshold, self.min_quality_score)

        # Create mask for keypoints above threshold
        mask = quality_scores >= threshold

        # Log filtering statistics
        total_keypoints = quality_scores.size
        passed_keypoints = np.sum(mask)
        logger.debug(
            f"Quality filtering: {passed_keypoints}/{total_keypoints} keypoints "
            f"passed ({passed_keypoints/total_keypoints*100:.1f}%, "
            f"threshold={threshold:.3f})"
        )

        return keypoints, mask

    def filter_consensus_frame(
        self,
        frame_data: Dict,
        trial_num: Optional[int] = None,
        total_trials: Optional[int] = None,
    ) -> Dict:
        """
        Filter a consensus frame based on quality scores.

        Args:
            frame_data: Dictionary with 'keypoints', 'confidence', 'stability', 'completeness'
            trial_num: Current trial number (if in Optuna context)
            total_trials: Total number of trials (if in Optuna context)

        Returns:
            Filtered frame data with quality mask added
        """
        keypoints = frame_data["keypoints"]
        confidence = frame_data.get("confidence", np.ones_like(keypoints[..., 0]))
        stability = frame_data.get("stability", np.ones_like(keypoints[..., 0]))
        completeness = frame_data.get("completeness", np.ones_like(keypoints[..., 0]))

        # Calculate composite quality scores
        quality_scores = self.calculate_quality_score(
            confidence, stability, completeness
        )

        # Get adaptive threshold
        if trial_num is not None and total_trials is not None:
            threshold_percentile = self.get_threshold_for_trial(trial_num, total_trials)
        else:
            # Default to growth phase percentile
            threshold_percentile = self.percentile_schedule["growth"]

        # Filter keypoints
        _, mask = self.filter_keypoints(keypoints, quality_scores, threshold_percentile)

        # Return filtered frame data
        return {
            **frame_data,
            "quality_scores": quality_scores,
            "quality_mask": mask,
            "threshold_used": threshold_percentile,
        }


if __name__ == "__main__":
    # Example usage
    import doctest

    doctest.testmod()

    # Demo
    print("AdaptiveQualityFilter Demo")
    print("=" * 50)

    quality_filter = AdaptiveQualityFilter()

    # Simulate quality scores for 17 keypoints
    confidence = np.random.rand(1, 17) * 0.5 + 0.5  # 0.5-1.0
    stability = np.random.rand(1, 17) * 0.6 + 0.4  # 0.4-1.0
    completeness = np.ones((1, 17)) * 0.8

    quality_scores = quality_filter.calculate_quality_score(
        confidence, stability, completeness
    )
    print(f"\nQuality scores: {quality_scores[0][:5]}... (showing first 5)")

    # Test adaptive thresholds
    print("\nAdaptive thresholds:")
    for trial, total in [(0, 50), (25, 50), (45, 50)]:
        threshold = quality_filter.get_threshold_for_trial(trial, total)
        print(
            f"  Trial {trial}/{total} ({trial/total*100:.0f}%): {threshold:.0f}th percentile"
        )
