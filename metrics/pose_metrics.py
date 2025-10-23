"""
Pose estimation metrics for evaluation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import scipy.spatial.distance as dist


class PoseMetrics:
    """Calculate pose estimation accuracy metrics"""

    def __init__(self):
        """Initialize pose metrics calculator"""
        self.keypoint_names = None

    def calculate_metrics(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate comprehensive pose estimation metrics

        Args:
            predictions: List of pose prediction results
            ground_truth: List of ground truth annotations

        Returns:
            Dictionary with calculated metrics
        """
        if not predictions or not ground_truth:
            return {}

        metrics = {}

        # Calculate PCK metrics
        pck_metrics = self.calculate_pck(predictions, ground_truth)
        metrics.update(pck_metrics)

        # Calculate MPJPE (if 3D data available)
        mpjpe_metrics = self.calculate_mpjpe(predictions, ground_truth)
        metrics.update(mpjpe_metrics)

        # Calculate temporal consistency
        temporal_metrics = self.calculate_temporal_consistency(predictions)
        metrics.update(temporal_metrics)

        # Calculate detection metrics
        detection_metrics = self.calculate_detection_metrics(predictions, ground_truth)
        metrics.update(detection_metrics)

        return metrics

    def calculate_pck(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        threshold: float = 0.2,
    ) -> Dict[str, float]:
        """Calculate Percentage of Correct Keypoints (PCK)

        Args:
            predictions: Predicted pose keypoints
            ground_truth: Ground truth keypoints
            threshold: PCK threshold (default 0.2 for PCK@0.2)

        Returns:
            Dictionary with PCK metrics
        """
        if len(predictions) != len(ground_truth):
            return {"pck_error": 1.0}

        all_distances = []
        all_normalizers = []
        correct_keypoints = []
        total_keypoints = []

        for pred, gt in zip(predictions, ground_truth):
            if "keypoints" not in pred or "keypoints" not in gt:
                continue

            pred_kpts = pred["keypoints"]  # Shape: (N, K, 2/3)
            gt_kpts = gt["keypoints"]  # Shape: (M, K, 2/3)

            # Match predictions to ground truth (closest bbox)
            matches = self._match_predictions_to_gt(pred, gt)

            for pred_idx, gt_idx in matches:
                if pred_idx >= len(pred_kpts) or gt_idx >= len(gt_kpts):
                    continue

                pred_person = pred_kpts[pred_idx]
                gt_person = gt_kpts[gt_idx]

                # Calculate normalizer (head segment or torso diagonal)
                normalizer = self._calculate_normalizer(gt_person)

                if normalizer <= 0:
                    continue

                # Calculate distances for each keypoint
                distances = []
                valid_count = 0
                correct_count = 0

                for kpt_idx in range(min(len(pred_person), len(gt_person))):
                    pred_pt = pred_person[kpt_idx][:2]
                    gt_pt = gt_person[kpt_idx][:2]

                    # Check if keypoint is valid (visible in ground truth)
                    if self._is_keypoint_valid(gt, gt_idx, kpt_idx):
                        distance = np.linalg.norm(pred_pt - gt_pt)
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
            return {"pck_error": 1.0}

        # Calculate overall PCK
        total_correct = sum(correct_keypoints)
        total_valid = sum(total_keypoints)
        pck = total_correct / total_valid if total_valid > 0 else 0.0

        # Calculate mean normalized distance
        mean_distance = np.mean(all_distances)

        return {
            f"pck_{threshold}": pck,
            "mean_normalized_distance": mean_distance,
            "total_correct_keypoints": total_correct,
            "total_valid_keypoints": total_valid,
        }

    def calculate_mpjpe(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate Mean Per Joint Position Error (3D)

        Args:
            predictions: Predicted 3D pose keypoints
            ground_truth: Ground truth 3D keypoints

        Returns:
            Dictionary with MPJPE metrics
        """
        mpjpe_distances = []

        for pred, gt in zip(predictions, ground_truth):
            if "keypoints" not in pred or "keypoints" not in gt:
                continue

            pred_kpts = pred["keypoints"]
            gt_kpts = gt["keypoints"]

            # Check if 3D data is available
            if pred_kpts.shape[-1] < 3 or gt_kpts.shape[-1] < 3:
                continue

            # Match predictions to ground truth
            matches = self._match_predictions_to_gt(pred, gt)

            for pred_idx, gt_idx in matches:
                if pred_idx >= len(pred_kpts) or gt_idx >= len(gt_kpts):
                    continue

                pred_person = pred_kpts[pred_idx][:, :3]  # (K, 3)
                gt_person = gt_kpts[gt_idx][:, :3]  # (K, 3)

                # Calculate per-joint distances
                distances = np.linalg.norm(pred_person - gt_person, axis=1)

                # Filter valid keypoints
                valid_mask = self._get_valid_keypoint_mask(gt, gt_idx)
                valid_distances = distances[valid_mask]

                if len(valid_distances) > 0:
                    mpjpe_distances.extend(valid_distances)

        if not mpjpe_distances:
            return {}

        mpjpe = np.mean(mpjpe_distances)

        return {
            "mpjpe": mpjpe,
            "mpjpe_std": np.std(mpjpe_distances),
            "mpjpe_median": np.median(mpjpe_distances),
        }

    def calculate_temporal_consistency(
        self, predictions: List[Dict[str, Any]], window_size: int = 5
    ) -> Dict[str, float]:
        """Calculate temporal consistency metrics

        Args:
            predictions: Sequential pose predictions
            window_size: Size of temporal window

        Returns:
            Dictionary with temporal consistency metrics
        """
        if len(predictions) < 2:
            return {}

        frame_differences = []
        acceleration_values = []

        for i in range(len(predictions) - 1):
            curr_pred = predictions[i]
            next_pred = predictions[i + 1]

            if "keypoints" not in curr_pred or "keypoints" not in next_pred:
                continue

            curr_kpts = curr_pred["keypoints"]
            next_kpts = next_pred["keypoints"]

            if len(curr_kpts) == 0 or len(next_kpts) == 0:
                continue

            # Take first person for simplicity
            curr_person = curr_kpts[0]
            next_person = next_kpts[0] if len(next_kpts) > 0 else curr_person

            # Calculate frame-to-frame differences
            if curr_person.shape == next_person.shape:
                diff = np.linalg.norm(next_person - curr_person, axis=1)
                frame_differences.append(np.mean(diff))

        # Calculate acceleration (second derivative)
        if len(frame_differences) >= 2:
            for i in range(len(frame_differences) - 1):
                accel = abs(frame_differences[i + 1] - frame_differences[i])
                acceleration_values.append(accel)

        metrics = {}

        if frame_differences:
            metrics.update(
                {
                    "temporal_smoothness": 1.0 / (1.0 + np.mean(frame_differences)),
                    "mean_frame_difference": np.mean(frame_differences),
                    "std_frame_difference": np.std(frame_differences),
                }
            )

        if acceleration_values:
            metrics.update(
                {
                    "temporal_stability": 1.0 / (1.0 + np.mean(acceleration_values)),
                    "mean_acceleration": np.mean(acceleration_values),
                }
            )

        return metrics

    def calculate_detection_metrics(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate person detection metrics

        Args:
            predictions: Pose predictions with person detections
            ground_truth: Ground truth person annotations

        Returns:
            Dictionary with detection metrics
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred, gt in zip(predictions, ground_truth):
            pred_count = pred.get("num_persons", 0)
            gt_count = gt.get("num_persons", 0)

            # Simple counting-based metrics
            if pred_count > 0 and gt_count > 0:
                true_positives += min(pred_count, gt_count)
                if pred_count > gt_count:
                    false_positives += pred_count - gt_count
                elif gt_count > pred_count:
                    false_negatives += gt_count - pred_count
            elif pred_count > 0 and gt_count == 0:
                false_positives += pred_count
            elif pred_count == 0 and gt_count > 0:
                false_negatives += gt_count

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_f1": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    def calculate_enhanced_detection_metrics(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate enhanced detection metrics without requiring ground truth

        Args:
            predictions: List of pose predictions

        Returns:
            Dictionary with enhanced detection metrics
        """
        if not predictions:
            return {}

        metrics = {}

        # Pose stability metrics
        stability_metrics = self._calculate_pose_stability(predictions)
        metrics.update(stability_metrics)

        # Keypoint consistency metrics
        consistency_metrics = self._calculate_keypoint_consistency(predictions)
        metrics.update(consistency_metrics)

        # Quality metrics
        quality_metrics = self._calculate_pose_quality_metrics(predictions)
        metrics.update(quality_metrics)

        # Completeness metrics
        completeness_metrics = self._calculate_skeleton_completeness(predictions)
        metrics.update(completeness_metrics)

        # Multi-person handling metrics
        multi_person_metrics = self._calculate_multi_person_metrics(predictions)
        metrics.update(multi_person_metrics)

        return metrics

    def _calculate_pose_stability(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate pose stability metrics across frames"""
        if len(predictions) < 2:
            return {}

        stability_scores = []
        jitter_scores = []

        for i in range(len(predictions) - 1):
            curr_pred = predictions[i]
            next_pred = predictions[i + 1]

            if "keypoints" not in curr_pred or "keypoints" not in next_pred:
                continue

            curr_kpts = curr_pred["keypoints"]
            next_kpts = next_pred["keypoints"]

            if len(curr_kpts) == 0 or len(next_kpts) == 0:
                continue

            # Calculate stability for each person
            for person_idx in range(min(len(curr_kpts), len(next_kpts))):
                curr_person = curr_kpts[person_idx]
                next_person = next_kpts[person_idx]

                if curr_person.shape != next_person.shape:
                    continue

                # Calculate keypoint movement
                movement = np.linalg.norm(next_person - curr_person, axis=1)
                avg_movement = np.mean(movement)

                # Stability score (inverse of movement)
                stability = 1.0 / (1.0 + avg_movement)
                stability_scores.append(stability)

                # Jitter score (standard deviation of movement)
                jitter = np.std(movement)
                jitter_scores.append(jitter)

        if not stability_scores:
            return {}

        return {
            "pose_stability_mean": np.mean(stability_scores),
            "pose_stability_std": np.std(stability_scores),
            "pose_jitter_mean": np.mean(jitter_scores),
            "pose_jitter_std": np.std(jitter_scores),
        }

    def _calculate_keypoint_consistency(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate keypoint consistency across frames"""
        if len(predictions) < 3:
            return {}

        consistency_scores = []
        confidence_stability = []

        for i in range(len(predictions) - 2):
            frame1 = predictions[i]
            frame2 = predictions[i + 1]
            frame3 = predictions[i + 2]

            if not all("keypoints" in f for f in [frame1, frame2, frame3]):
                continue

            kpts1 = frame1["keypoints"]
            kpts2 = frame2["keypoints"]
            kpts3 = frame3["keypoints"]

            if len(kpts1) == 0 or len(kpts2) == 0 or len(kpts3) == 0:
                continue

            # Calculate consistency for each person
            for person_idx in range(min(len(kpts1), len(kpts2), len(kpts3))):
                person1 = kpts1[person_idx]
                person2 = kpts2[person_idx]
                person3 = kpts3[person_idx]

                if not all(p.shape == person1.shape for p in [person2, person3]):
                    continue

                # Calculate keypoint consistency (how well keypoints track across frames)
                movement1 = np.linalg.norm(person2 - person1, axis=1)
                movement2 = np.linalg.norm(person3 - person2, axis=1)

                # Consistency: similar movement patterns
                movement_diff = np.abs(movement2 - movement1)
                consistency = 1.0 / (1.0 + np.mean(movement_diff))
                consistency_scores.append(consistency)

                # Confidence stability (if available)
                if "scores" in frame1 and "scores" in frame2 and "scores" in frame3:
                    scores1 = (
                        frame1["scores"][person_idx]
                        if person_idx < len(frame1["scores"])
                        else []
                    )
                    scores2 = (
                        frame2["scores"][person_idx]
                        if person_idx < len(frame2["scores"])
                        else []
                    )
                    scores3 = (
                        frame3["scores"][person_idx]
                        if person_idx < len(frame3["scores"])
                        else []
                    )

                    if len(scores1) == len(scores2) == len(scores3):
                        score_stability = 1.0 - np.std([scores1, scores2, scores3])
                        confidence_stability.append(score_stability)

        metrics = {}
        if consistency_scores:
            metrics.update(
                {
                    "keypoint_consistency_mean": np.mean(consistency_scores),
                    "keypoint_consistency_std": np.std(consistency_scores),
                }
            )

        if confidence_stability:
            metrics.update(
                {
                    "confidence_stability_mean": np.mean(confidence_stability),
                    "confidence_stability_std": np.std(confidence_stability),
                }
            )

        return metrics

    def _calculate_pose_quality_metrics(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate pose quality metrics based on confidence and keypoint validity"""
        all_confidences = []
        valid_keypoint_ratios = []
        high_confidence_ratios = []

        for pred in predictions:
            if "keypoints" not in pred:
                continue

            kpts = pred["keypoints"]
            if len(kpts) == 0:
                continue

            for person_kpts in kpts:
                # Extract confidence scores if available
                if "scores" in pred:
                    person_scores = pred["scores"][0] if len(pred["scores"]) > 0 else []
                    if len(person_scores) > 0:
                        all_confidences.extend(person_scores)

                        # High confidence ratio (confidence > 0.7)
                        high_conf_count = sum(1 for s in person_scores if s > 0.7)
                        high_confidence_ratios.append(
                            high_conf_count / len(person_scores)
                        )

                # Valid keypoint ratio (non-zero coordinates)
                valid_count = 0
                total_count = 0

                for kpt in person_kpts:
                    if len(kpt) >= 2:
                        total_count += 1
                        if kpt[0] != 0 or kpt[1] != 0:  # Non-zero coordinates
                            valid_count += 1

                if total_count > 0:
                    valid_keypoint_ratios.append(valid_count / total_count)

        metrics = {}

        if all_confidences:
            metrics.update(
                {
                    "avg_keypoint_confidence": np.mean(all_confidences),
                    "confidence_std": np.std(all_confidences),
                    "min_confidence": np.min(all_confidences),
                    "max_confidence": np.max(all_confidences),
                }
            )

        if high_confidence_ratios:
            metrics.update(
                {
                    "high_confidence_ratio": np.mean(high_confidence_ratios),
                }
            )

        if valid_keypoint_ratios:
            metrics.update(
                {
                    "valid_keypoint_ratio": np.mean(valid_keypoint_ratios),
                    "keypoint_completeness": np.mean(valid_keypoint_ratios),
                }
            )

        return metrics

    def _calculate_skeleton_completeness(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate skeleton completeness metrics"""
        completeness_scores = []
        anatomical_validity_scores = []

        for pred in predictions:
            if "keypoints" not in pred:
                continue

            kpts = pred["keypoints"]
            if len(kpts) == 0:
                continue

            for person_kpts in kpts:
                if len(person_kpts) < 17:  # COCO format minimum
                    continue

                # Calculate completeness (how many keypoints are detected)
                valid_kpts = sum(1 for kpt in person_kpts if kpt[0] != 0 or kpt[1] != 0)
                completeness = valid_kpts / len(person_kpts)
                completeness_scores.append(completeness)

                # Anatomical validity (basic skeleton structure)
                if len(person_kpts) >= 17:  # COCO format
                    # Check if key anatomical points are present
                    # Head, shoulders, hips, knees, ankles
                    key_points = [0, 5, 6, 11, 12, 13, 14, 15, 16]  # COCO indices
                    key_point_count = sum(
                        1
                        for i in key_points
                        if i < len(person_kpts)
                        and (person_kpts[i][0] != 0 or person_kpts[i][1] != 0)
                    )
                    anatomical_validity = key_point_count / len(key_points)
                    anatomical_validity_scores.append(anatomical_validity)

        metrics = {}

        if completeness_scores:
            metrics.update(
                {
                    "skeleton_completeness_mean": np.mean(completeness_scores),
                    "skeleton_completeness_std": np.std(completeness_scores),
                }
            )

        if anatomical_validity_scores:
            metrics.update(
                {
                    "anatomical_validity_mean": np.mean(anatomical_validity_scores),
                    "anatomical_validity_std": np.std(anatomical_validity_scores),
                }
            )

        return metrics

    def _calculate_multi_person_metrics(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate multi-person handling metrics"""
        person_counts = []
        detection_consistency = []

        for pred in predictions:
            if "keypoints" not in pred:
                continue

            kpts = pred["keypoints"]
            person_counts.append(len(kpts))

        if not person_counts:
            return {}

        # Calculate detection consistency (how stable is person count)
        if len(person_counts) > 1:
            for i in range(len(person_counts) - 1):
                consistency = 1.0 if person_counts[i] == person_counts[i + 1] else 0.0
                detection_consistency.append(consistency)

        metrics = {
            "avg_persons_detected": np.mean(person_counts),
            "max_persons_detected": np.max(person_counts),
            "min_persons_detected": np.min(person_counts),
            "person_count_std": np.std(person_counts),
        }

        if detection_consistency:
            metrics.update(
                {
                    "detection_consistency": np.mean(detection_consistency),
                }
            )

        return metrics

    def _match_predictions_to_gt(
        self, pred: Dict[str, Any], gt: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        """Match predicted persons to ground truth persons based on bbox IoU

        Args:
            pred: Prediction with bbox information
            gt: Ground truth with bbox information

        Returns:
            List of (pred_idx, gt_idx) matches
        """
        if "bbox" not in pred or "bbox" not in gt:
            # Fallback: match first person only
            pred_count = pred.get("num_persons", 0)
            gt_count = gt.get("num_persons", 0)
            if pred_count > 0 and gt_count > 0:
                return [(0, 0)]
            return []

        pred_bboxes = pred["bbox"]
        gt_bboxes = gt["bbox"]

        matches = []
        used_gt = set()

        # Calculate IoU matrix
        for pred_idx, pred_bbox in enumerate(pred_bboxes):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_bbox in enumerate(gt_bboxes):
                if gt_idx in used_gt:
                    continue

                iou = self._calculate_bbox_iou(pred_bbox, gt_bbox)
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matches.append((pred_idx, best_gt_idx))
                used_gt.add(best_gt_idx)

        return matches

    def _calculate_bbox_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_pck_with_consensus_gt(
        self,
        predictions: List[Dict[str, Any]],
        consensus_gt: Dict[str, Any],
        maneuver_id: str,
        threshold: float = 0.2,
    ) -> Dict[str, float]:
        """
        Calculate PCK against consensus pseudo-ground-truth.

        Used for Optuna optimization when no manual annotations exist.
        Compares model predictions against consensus generated from other models.

        Args:
            predictions: Model predictions for maneuver frames
                List of dicts with 'keypoints' and 'scores' keys
            consensus_gt: Consensus ground truth data for this maneuver
                Dict with 'frames' key containing consensus for each frame
            maneuver_id: ID of the maneuver being evaluated
            threshold: PCK threshold (default 0.2 for PCK@0.2)

        Returns:
            Dictionary with PCK metrics:
            - pck_0_2: PCK score at threshold 0.2
            - correct_keypoints: Number of correct keypoints
            - total_keypoints: Total keypoints evaluated
            - avg_distance: Average normalized distance
        """
        if maneuver_id not in consensus_gt:
            return {
                "pck_0_2": 0.0,
                "correct_keypoints": 0,
                "total_keypoints": 0,
                "avg_distance": 0.0,
                "error": "maneuver_not_in_consensus",
            }

        consensus_frames = consensus_gt[maneuver_id]["frames"]

        if len(predictions) != len(consensus_frames):
            # Frame count mismatch - try to align or return error
            min_frames = min(len(predictions), len(consensus_frames))
            predictions = predictions[:min_frames]
            consensus_frames = consensus_frames[:min_frames]

        correct_keypoints = 0
        total_keypoints = 0
        all_distances = []

        for pred, gt_frame in zip(predictions, consensus_frames):
            # Extract prediction keypoints (take first person if multiple)
            if pred["num_persons"] == 0:
                continue  # No detection in this frame

            pred_kpts = pred["keypoints"][0]  # Shape: (17, 2)

            # Extract consensus keypoints
            gt_kpts = gt_frame["keypoints"]  # Shape: (17, 2)
            gt_conf = gt_frame["confidence"]  # Shape: (17,)

            # Only evaluate keypoints with sufficient consensus confidence
            valid_mask = gt_conf > 0.5

            if not valid_mask.any():
                continue  # No valid keypoints in this frame

            # Calculate distances
            distances = np.linalg.norm(pred_kpts - gt_kpts, axis=-1)  # Shape: (17,)

            # Normalize by torso diameter
            torso_diameter = self._estimate_torso_diameter(gt_kpts)

            if torso_diameter > 0:
                normalized_distances = distances / torso_diameter
            else:
                # Fallback: normalize by image diagonal (assume 1920x1080)
                normalized_distances = distances / np.sqrt(1920**2 + 1080**2)

            # Check which keypoints are correct
            correct = (normalized_distances < threshold) & valid_mask

            correct_keypoints += correct.sum()
            total_keypoints += valid_mask.sum()
            all_distances.extend(normalized_distances[valid_mask].tolist())

        # Calculate final metrics
        pck = correct_keypoints / total_keypoints if total_keypoints > 0 else 0.0
        avg_distance = np.mean(all_distances) if all_distances else 0.0

        return {
            "pck_0_2": float(pck),
            "correct_keypoints": int(correct_keypoints),
            "total_keypoints": int(total_keypoints),
            "avg_distance": float(avg_distance),
        }

    def _estimate_torso_diameter(self, keypoints: np.ndarray) -> float:
        """
        Estimate torso diameter for PCK normalization.

        Uses distance between shoulders and hips as proxy for torso size.

        Args:
            keypoints: Keypoint array, shape (17, 2)

        Returns:
            Estimated torso diameter in pixels
        """
        # COCO keypoint indices:
        # 5-left_shoulder, 6-right_shoulder, 11-left_hip, 12-right_hip
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Calculate shoulder and hip widths
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        hip_width = np.linalg.norm(left_hip - right_hip)

        # Calculate torso height (average shoulder to hip distance)
        left_torso_height = np.linalg.norm(left_shoulder - left_hip)
        right_torso_height = np.linalg.norm(right_shoulder - right_hip)
        torso_height = (left_torso_height + right_torso_height) / 2

        # Torso diameter: diagonal of torso bounding box
        avg_width = (shoulder_width + hip_width) / 2
        torso_diameter = np.sqrt(avg_width**2 + torso_height**2)

        return torso_diameter if torso_diameter > 0 else 100.0  # Fallback to 100 pixels

    def _calculate_normalizer(self, keypoints: np.ndarray) -> float:
        """Calculate normalization factor for PCK (head segment or torso diagonal)

        Args:
            keypoints: Keypoint array (K, 2/3)

        Returns:
            Normalization factor
        """
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

    def _is_keypoint_valid(
        self, gt: Dict[str, Any], person_idx: int, keypoint_idx: int
    ) -> bool:
        """Check if a keypoint is valid (visible) in ground truth

        Args:
            gt: Ground truth annotation
            person_idx: Person index
            keypoint_idx: Keypoint index

        Returns:
            True if keypoint is valid/visible
        """
        # Default: assume all keypoints are valid
        if "scores" in gt:
            scores = gt["scores"]
            if person_idx < len(scores) and keypoint_idx < len(scores[person_idx]):
                return scores[person_idx][keypoint_idx] > 0.1

        # Check visibility flags if available
        if "visibility" in gt:
            visibility = gt["visibility"]
            if person_idx < len(visibility) and keypoint_idx < len(
                visibility[person_idx]
            ):
                return visibility[person_idx][keypoint_idx] > 0

        return True  # Default to valid

    def _get_valid_keypoint_mask(
        self, gt: Dict[str, Any], person_idx: int
    ) -> np.ndarray:
        """Get mask of valid keypoints for a person

        Args:
            gt: Ground truth annotation
            person_idx: Person index

        Returns:
            Boolean mask of valid keypoints
        """
        if "keypoints" not in gt:
            return np.array([])

        num_keypoints = gt["keypoints"].shape[1]
        mask = np.ones(num_keypoints, dtype=bool)

        for kpt_idx in range(num_keypoints):
            mask[kpt_idx] = self._is_keypoint_valid(gt, person_idx, kpt_idx)

        return mask
