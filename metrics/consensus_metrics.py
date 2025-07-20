"""
Consensus-based pseudo ground truth metrics for pose estimation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import cv2

logger = logging.getLogger(__name__)


class ConsensusMetrics:
    """Calculate consensus-based pseudo ground truth and relative accuracy metrics"""

    def __init__(
        self, reference_models: List[str] = None, confidence_threshold: float = 0.7
    ):
        """Initialize consensus metrics calculator

        Args:
            reference_models: List of model names to use as reference (default: ['pytorch_pose', 'yolov8_pose', 'mmpose'])
            confidence_threshold: Minimum confidence for keypoints to be included in consensus
        """
        self.reference_models = reference_models or [
            "pytorch_pose",
            "yolov8_pose",
            "mmpose",
        ]
        self.confidence_threshold = confidence_threshold
        self.keypoint_names = None

    def create_consensus_ground_truth(
        self,
        all_model_predictions: Dict[str, List[Dict[str, Any]]],
        frame_indices: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """Create consensus ground truth from multiple model predictions

        Args:
            all_model_predictions: Dictionary of model_name -> list of predictions
            frame_indices: Optional list of frame indices to process

        Returns:
            List of consensus predictions (pseudo ground truth)
        """
        if not all_model_predictions:
            return []

        # Get the number of frames from the first model
        first_model = list(all_model_predictions.keys())[0]
        num_frames = len(all_model_predictions[first_model])

        if frame_indices is None:
            frame_indices = list(range(num_frames))

        consensus_predictions = []

        for frame_idx in frame_indices:
            frame_predictions = {}

            # Collect predictions from all models for this frame
            for model_name, predictions in all_model_predictions.items():
                if frame_idx < len(predictions):
                    frame_predictions[model_name] = predictions[frame_idx]

            # Create consensus for this frame
            consensus_frame = self._create_frame_consensus(frame_predictions, frame_idx)
            consensus_predictions.append(consensus_frame)

        logger.info(
            f"Created consensus ground truth for {len(consensus_predictions)} frames"
        )
        return consensus_predictions

    def _create_frame_consensus(
        self, frame_predictions: Dict[str, Dict[str, Any]], frame_idx: int
    ) -> Dict[str, Any]:
        """Create consensus for a single frame

        Args:
            frame_predictions: Dictionary of model_name -> prediction for this frame
            frame_idx: Frame index

        Returns:
            Consensus prediction for this frame
        """
        # Filter to only reference models
        reference_predictions = {
            model_name: pred
            for model_name, pred in frame_predictions.items()
            if model_name in self.reference_models
        }

        if not reference_predictions:
            logger.warning(
                f"No reference model predictions available for frame {frame_idx}"
            )
            return {"keypoints": [], "num_persons": 0, "frame_idx": frame_idx}

        # Extract all keypoints from reference models
        all_keypoints = []
        all_scores = []
        model_weights = []

        for model_name, prediction in reference_predictions.items():
            if "keypoints" not in prediction:
                continue

            keypoints = prediction["keypoints"]
            scores = prediction.get("scores", [])

            # Weight models based on their typical performance
            weight = self._get_model_weight(model_name)

            for person_idx, person_kpts in enumerate(keypoints):
                if len(person_kpts) == 0:
                    continue

                person_scores = (
                    scores[person_idx]
                    if person_idx < len(scores)
                    else [0.5] * len(person_kpts)
                )

                # Filter by confidence threshold
                valid_indices = [
                    i
                    for i, score in enumerate(person_scores)
                    if score >= self.confidence_threshold
                ]

                if (
                    len(valid_indices) < 2
                ):  # Need at least 2 keypoints for consensus (lowered for testing)
                    continue

                valid_keypoints = person_kpts[valid_indices]
                valid_scores = [person_scores[i] for i in valid_indices]

                all_keypoints.append(valid_keypoints)
                all_scores.append(valid_scores)
                model_weights.append(weight)

        if not all_keypoints:
            return {"keypoints": [], "num_persons": 0, "frame_idx": frame_idx}

        # Cluster keypoints to group similar predictions
        consensus_keypoints = self._cluster_keypoints(
            all_keypoints, all_scores, model_weights
        )

        return {
            "keypoints": consensus_keypoints,
            "num_persons": len(consensus_keypoints),
            "frame_idx": frame_idx,
            "consensus_confidence": self._calculate_consensus_confidence(
                all_scores, model_weights
            ),
        }

    def _cluster_keypoints(
        self,
        all_keypoints: List[np.ndarray],
        all_scores: List[List[float]],
        model_weights: List[float],
    ) -> List[np.ndarray]:
        """Cluster keypoints to create consensus predictions

        Args:
            all_keypoints: List of keypoint arrays from different models
            all_scores: List of confidence scores
            model_weights: List of model weights

        Returns:
            List of consensus keypoint arrays
        """
        if not all_keypoints:
            return []

        # Flatten all keypoints for clustering
        flat_keypoints = []
        flat_scores = []
        flat_weights = []

        for kpts, scores, weight in zip(all_keypoints, all_scores, model_weights):
            for kpt, score in zip(kpts, scores):
                flat_keypoints.append(kpt[:2])  # Use only x,y coordinates
                flat_scores.append(score)
                flat_weights.append(weight)

        if len(flat_keypoints) < 2:
            return []

        flat_keypoints = np.array(flat_keypoints)
        flat_scores = np.array(flat_scores)
        flat_weights = np.array(flat_weights)

        # Use DBSCAN to cluster similar keypoints
        clustering = DBSCAN(eps=20.0, min_samples=2).fit(flat_keypoints)
        labels = clustering.labels_

        consensus_keypoints = []

        # Process each cluster
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                continue

            cluster_mask = labels == cluster_id
            cluster_kpts = flat_keypoints[cluster_mask]
            cluster_scores = flat_scores[cluster_mask]
            cluster_weights = flat_weights[cluster_mask]

            # Calculate weighted centroid
            weighted_centroid = self._calculate_weighted_centroid(
                cluster_kpts, cluster_scores, cluster_weights
            )

            # Create consensus keypoint with confidence
            consensus_kpt = np.array(
                [weighted_centroid[0], weighted_centroid[1], np.mean(cluster_scores)]
            )
            consensus_keypoints.append(consensus_kpt)

        # Group keypoints by person (assuming COCO format with 17 keypoints)
        return self._group_keypoints_by_person(consensus_keypoints)

    def _calculate_weighted_centroid(
        self, keypoints: np.ndarray, scores: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Calculate weighted centroid of keypoints

        Args:
            keypoints: Array of keypoint coordinates
            scores: Array of confidence scores
            weights: Array of model weights

        Returns:
            Weighted centroid coordinates
        """
        # Combine scores and weights
        combined_weights = scores * weights

        if np.sum(combined_weights) == 0:
            return np.mean(keypoints, axis=0)

        weighted_sum = np.sum(keypoints * combined_weights[:, np.newaxis], axis=0)
        total_weight = np.sum(combined_weights)

        return weighted_sum / total_weight

    def _group_keypoints_by_person(
        self, consensus_keypoints: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Group consensus keypoints by person

        Args:
            consensus_keypoints: List of individual consensus keypoints

        Returns:
            List of person keypoint arrays
        """
        if not consensus_keypoints:
            return []

        # For now, assume single person and group all keypoints
        # In a more sophisticated implementation, you could use spatial clustering
        # to separate multiple people

        # Sort keypoints by confidence
        sorted_keypoints = sorted(consensus_keypoints, key=lambda x: x[2], reverse=True)

        # Take the top keypoints (assuming COCO format with 17 keypoints)
        max_keypoints = 17
        top_keypoints = sorted_keypoints[:max_keypoints]

        # Pad with zeros if needed
        while len(top_keypoints) < max_keypoints:
            top_keypoints.append(np.array([0.0, 0.0, 0.0]))

        return [np.array(top_keypoints)]

    def _get_model_weight(self, model_name: str) -> float:
        """Get weight for a model based on its typical performance

        Args:
            model_name: Name of the model

        Returns:
            Model weight (higher = more trusted)
        """
        # Weights based on typical model performance
        weights = {
            "mmpose": 1.0,  # High accuracy
            "yolov8_pose": 0.9,  # Good accuracy
            "pytorch_pose": 0.8,  # Moderate accuracy
            "blazepose": 0.7,  # Lower accuracy
            "mediapipe": 0.6,  # Lower accuracy
        }

        return weights.get(model_name.lower(), 0.5)

    def _calculate_consensus_confidence(
        self, all_scores: List[List[float]], model_weights: List[float]
    ) -> float:
        """Calculate overall confidence of consensus

        Args:
            all_scores: List of confidence scores from all models
            model_weights: List of model weights

        Returns:
            Overall consensus confidence
        """
        if not all_scores:
            return 0.0

        # Flatten all scores
        flat_scores = []
        flat_weights = []

        for scores, weight in zip(all_scores, model_weights):
            # Handle potential None values in scores
            if scores is None:
                continue
            if isinstance(scores, (list, tuple)):
                # Filter out any None values from the scores
                valid_scores = [
                    s
                    for s in scores
                    if s is not None and not (isinstance(s, float) and (s != s))
                ]  # Filter NaN
                flat_scores.extend(valid_scores)
                flat_weights.extend([weight] * len(valid_scores))
            elif isinstance(scores, (int, float)) and scores == scores:  # Check for NaN
                flat_scores.append(scores)
                flat_weights.append(weight)

        if not flat_scores:
            return 0.0

        # Calculate weighted average confidence
        weighted_sum = sum(
            score * weight for score, weight in zip(flat_scores, flat_weights)
        )
        total_weight = sum(flat_weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def calculate_relative_pck(
        self,
        model_predictions: List[Dict[str, Any]],
        consensus_predictions: List[Dict[str, Any]],
        threshold: float = 0.2,
    ) -> Dict[str, float]:
        """Calculate PCK-like metric against consensus rather than ground truth

        Args:
            model_predictions: Predictions from the model being evaluated
            consensus_predictions: Consensus predictions (pseudo ground truth)
            threshold: PCK threshold (default 0.2 for PCK@0.2)

        Returns:
            Dictionary with relative PCK metrics
        """
        if len(model_predictions) != len(consensus_predictions):
            logger.warning(
                f"Model predictions and consensus predictions have different lengths: {len(model_predictions)} vs {len(consensus_predictions)}"
            )
            logger.info("Using frame-wise alignment instead of simple truncation")

        all_distances = []
        all_normalizers = []
        correct_keypoints = []
        total_keypoints = []

        # Create frame index mappings for alignment
        model_frame_map = {}
        consensus_frame_map = {}

        # Map model predictions by frame index
        for i, pred in enumerate(model_predictions):
            frame_idx = pred.get("frame_idx", pred.get("frame_id", i))
            model_frame_map[frame_idx] = pred

        # Map consensus predictions by frame index
        for i, consensus in enumerate(consensus_predictions):
            frame_idx = consensus.get("frame_idx", consensus.get("frame_id", i))
            consensus_frame_map[frame_idx] = consensus

        # Find common frame indices
        common_frames = set(model_frame_map.keys()) & set(consensus_frame_map.keys())
        logger.info(
            f"Found {len(common_frames)} common frames out of {len(model_frame_map)} model frames and {len(consensus_frame_map)} consensus frames"
        )

        if not common_frames:
            logger.warning(
                "No common frames found between model and consensus predictions"
            )
            return {"relative_pck_error": 1.0}

        # Process only common frames
        for frame_idx in sorted(common_frames):
            pred = model_frame_map[frame_idx]
            consensus = consensus_frame_map[frame_idx]

            if "keypoints" not in pred or "keypoints" not in consensus:
                continue

            pred_kpts = pred["keypoints"]
            consensus_kpts = consensus["keypoints"]

            if len(pred_kpts) == 0 or len(consensus_kpts) == 0:
                continue

            # Match predictions to consensus (closest person)
            matches = self._match_predictions_to_consensus(pred, consensus)

            for pred_idx, consensus_idx in matches:
                if pred_idx >= len(pred_kpts) or consensus_idx >= len(consensus_kpts):
                    continue

                pred_person = pred_kpts[pred_idx]
                consensus_person = consensus_kpts[consensus_idx]

                # Calculate normalizer (head segment or torso diagonal)
                normalizer = self._calculate_normalizer(consensus_person)

                if normalizer <= 0:
                    continue

                # Calculate distances for each keypoint
                distances = []
                valid_count = 0
                correct_count = 0

                for kpt_idx in range(min(len(pred_person), len(consensus_person))):
                    pred_pt = pred_person[kpt_idx][:2]
                    consensus_pt = consensus_person[kpt_idx][:2]

                    # Check if consensus keypoint is valid (non-zero)
                    if np.any(consensus_pt != 0):
                        distance = np.linalg.norm(pred_pt - consensus_pt)
                        normalized_distance = distance / normalizer

                        distances.append(normalized_distance)
                        valid_count += 1

                        if normalized_distance <= threshold:
                            correct_count += 1

                if valid_count > 0:
                    all_distances.extend(distances)
                    all_normalizers.append(normalizer)
                    correct_keypoints.append(correct_count)
                    total_keypoints.append(valid_count)

        if not all_distances:
            return {"relative_pck_error": 1.0}

        # Calculate overall relative PCK
        total_correct = sum(correct_keypoints)
        total_valid = sum(total_keypoints)
        relative_pck = total_correct / total_valid if total_valid > 0 else 0.0

        # Calculate mean normalized distance
        mean_distance = np.mean(all_distances)

        # Calculate coverage metrics
        model_frame_count = len(model_frame_map)
        consensus_frame_count = len(consensus_frame_map)
        common_frame_count = len(common_frames)
        coverage_ratio = (
            common_frame_count / max(model_frame_count, consensus_frame_count)
            if max(model_frame_count, consensus_frame_count) > 0
            else 0.0
        )

        return {
            f"consensus_pck_{threshold}": relative_pck,
            "consensus_pck_error": 1.0 - relative_pck,
            "consensus_mean_normalized_distance": mean_distance,
            "consensus_total_correct_keypoints": total_correct,
            "consensus_total_valid_keypoints": total_valid,
            "consensus_coverage_ratio": coverage_ratio,
            "consensus_common_frames": common_frame_count,
            "model_total_frames": model_frame_count,
            "consensus_total_frames": consensus_frame_count,
        }

    def _match_predictions_to_consensus(
        self, pred: Dict[str, Any], consensus: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        """Match predicted persons to consensus persons based on spatial proximity

        Args:
            pred: Prediction with keypoint information
            consensus: Consensus prediction

        Returns:
            List of (pred_idx, consensus_idx) matches
        """
        if "keypoints" not in pred or "keypoints" not in consensus:
            return []

        pred_kpts = pred["keypoints"]
        consensus_kpts = consensus["keypoints"]

        if len(pred_kpts) == 0 or len(consensus_kpts) == 0:
            return []

        matches = []

        # Simple matching: match first person to first person
        # In a more sophisticated implementation, you could use IoU or centroid distance
        for i in range(min(len(pred_kpts), len(consensus_kpts))):
            matches.append((i, i))

        return matches

    def _calculate_normalizer(self, keypoints: np.ndarray) -> float:
        """Calculate normalization factor for PCK (head segment or torso diagonal)

        Args:
            keypoints: Keypoint array (K, 2/3)

        Returns:
            Normalization factor
        """
        if len(keypoints) < 2:
            return 1.0

        # Try head segment first (distance between ears)
        if len(keypoints) >= 4:  # Assume COCO format
            # Left ear (3), right ear (4) in COCO
            if len(keypoints) > 4:
                left_ear = keypoints[3][:2]
                right_ear = keypoints[4][:2]
                head_size = np.linalg.norm(right_ear - left_ear)
                if head_size > 0:
                    return head_size

        # Fallback: use torso diagonal
        if len(keypoints) >= 12:  # COCO format
            # Left shoulder (5), right hip (12)
            left_shoulder = keypoints[5][:2]
            right_hip = keypoints[12][:2] if len(keypoints) > 12 else keypoints[11][:2]
            torso_diagonal = np.linalg.norm(right_hip - left_shoulder)
            if torso_diagonal > 0:
                return torso_diagonal

        # Final fallback: bounding box diagonal
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        bbox_diagonal = np.sqrt(
            (x_coords.max() - x_coords.min()) ** 2
            + (y_coords.max() - y_coords.min()) ** 2
        )

        return max(bbox_diagonal, 1.0)  # Minimum 1 pixel

    def calculate_consensus_quality_metrics(
        self, consensus_predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate quality metrics for consensus predictions

        Args:
            consensus_predictions: List of consensus predictions

        Returns:
            Dictionary with consensus quality metrics
        """
        if not consensus_predictions:
            return {}

        total_frames = len(consensus_predictions)
        frames_with_consensus = 0
        total_consensus_confidence = 0.0
        consensus_stability_scores = []

        for i, consensus in enumerate(consensus_predictions):
            if consensus.get("num_persons", 0) > 0:
                frames_with_consensus += 1
                total_consensus_confidence += consensus.get("consensus_confidence", 0.0)

            # Calculate stability with previous frame
            if (
                i > 0
                and "keypoints" in consensus
                and "keypoints" in consensus_predictions[i - 1]
            ):
                prev_kpts = consensus_predictions[i - 1]["keypoints"]
                curr_kpts = consensus["keypoints"]

                if len(prev_kpts) > 0 and len(curr_kpts) > 0:
                    # Calculate average movement
                    movement = np.linalg.norm(curr_kpts[0] - prev_kpts[0], axis=1)
                    stability = 1.0 / (1.0 + np.mean(movement))
                    consensus_stability_scores.append(stability)

        metrics = {
            "consensus_coverage": (
                frames_with_consensus / total_frames if total_frames > 0 else 0.0
            ),
            "avg_consensus_confidence": (
                total_consensus_confidence / frames_with_consensus
                if frames_with_consensus > 0
                else 0.0
            ),
            "consensus_stability": (
                np.mean(consensus_stability_scores)
                if consensus_stability_scores
                else 0.0
            ),
            "total_consensus_frames": frames_with_consensus,
            "total_frames": total_frames,
        }

        return metrics
