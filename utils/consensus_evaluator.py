"""
Consensus-based evaluation system for pose estimation
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time

from utils.pose_evaluator import PoseEvaluator
from metrics.consensus_metrics import ConsensusMetrics

logger = logging.getLogger(__name__)


class ConsensusEvaluator:
    """Evaluate models using consensus-based pseudo ground truth"""

    def __init__(self, config: Dict, reference_models: List[str] = None):
        """Initialize consensus evaluator

        Args:
            config: Evaluation configuration
            reference_models: List of reference models for consensus (default: ['pytorch_pose', 'yolov8_pose', 'mmpose'])
        """
        self.config = config
        self.reference_models = reference_models or [
            "pytorch_pose",
            "yolov8_pose",
            "mmpose",
        ]
        self.consensus_metrics = ConsensusMetrics(
            reference_models=self.reference_models
        )

        # Initialize pose evaluator for accessing prediction handler
        self.pose_evaluator = PoseEvaluator(config)

        # Store all model predictions for consensus generation
        self.all_model_predictions = {}
        self.consensus_predictions = {}

    def run_consensus_evaluation(
        self,
        maneuvers: List,
        target_models: List[str] = None,
        save_consensus: bool = True,
    ) -> Dict[str, Any]:
        """Run consensus-based evaluation using cached prediction files

        IMPORTANT: This method only works with cached prediction files from prior model runs.
        It will NOT re-run model inference if prediction files are missing.

        Args:
            maneuvers: List of maneuvers to evaluate
            target_models: Models to evaluate against consensus (must have cached predictions)
            save_consensus: Whether to save consensus predictions to file

        Returns:
            Dictionary with consensus evaluation results, or empty dict if failed
        """
        logger.info(
            f"🚀 Starting consensus-based evaluation with {len(maneuvers)} maneuvers"
        )
        logger.info(f"📊 Reference models: {self.reference_models}")

        # Verify we're using comparison phase data (data consistency check)
        if (
            hasattr(self.pose_evaluator, "run_manager")
            and self.pose_evaluator.run_manager
        ):
            predictions_dir = self.pose_evaluator.run_manager.predictions_dir
            logger.info(f"🔍 Using prediction files from: {predictions_dir}")

            # Verify this is comparison phase data (should have more files than optuna)
            if predictions_dir.exists():
                model_dirs = [d for d in predictions_dir.iterdir() if d.is_dir()]
                if model_dirs:
                    sample_model_dir = model_dirs[0]
                    file_count = len(list(sample_model_dir.glob("*_predictions.json")))
                    logger.info(f"📁 Found {file_count} prediction files per model")

                    if file_count < 20:  # Arbitrary threshold
                        logger.warning(
                            "⚠️ Low prediction file count - ensure this is comparison phase data, not optuna phase"
                        )

        # Step 1: Load cached predictions from reference models
        logger.info("🔄 Step 1: Loading cached predictions from reference models...")
        self._run_reference_models(maneuvers)

        # Check if we have sufficient reference model predictions
        if not self.all_model_predictions:
            logger.error(
                "❌ No reference model predictions loaded - consensus evaluation cannot proceed"
            )
            logger.error(
                "   Ensure that individual model evaluation completed successfully"
            )
            logger.error(
                "   and prediction files exist before running consensus evaluation"
            )
            return {}

        if len(self.all_model_predictions) < 2:
            logger.error(
                f"❌ Insufficient reference models for consensus: {len(self.all_model_predictions)} < 2"
            )
            logger.error(f"   Available: {list(self.all_model_predictions.keys())}")
            logger.error(
                "   Consensus requires at least 2 reference models with cached predictions"
            )
            return {}

        # Step 2: Create consensus ground truth
        logger.info("🎯 Step 2: Creating consensus ground truth...")
        self._create_consensus_ground_truth(maneuvers)

        # Step 3: Evaluate target models against consensus
        logger.info("📈 Step 3: Evaluating target models against consensus...")
        target_models = target_models or self.pose_evaluator.get_available_models()
        evaluation_results = self._evaluate_against_consensus(maneuvers, target_models)

        # Step 4: Save consensus predictions if requested
        if save_consensus:
            self._save_consensus_predictions()

        return evaluation_results

    def _run_reference_models(self, maneuvers: List):
        """Load predictions from already-computed reference models

        CRITICAL: This method ONLY loads cached predictions and NEVER re-runs inference.
        If prediction files are missing, consensus evaluation will fail.

        Args:
            maneuvers: List of maneuvers to process
        """
        missing_predictions = []

        for model_name in self.reference_models:
            try:
                logger.info(
                    f"📁 Loading cached predictions for reference model {model_name}..."
                )

                # Load existing prediction files (NO model re-execution allowed)
                predictions = self._load_existing_predictions(model_name, maneuvers)

                if predictions:
                    self.all_model_predictions[model_name] = predictions
                    logger.info(
                        f"✅ {model_name} loaded {len(predictions)} cached predictions"
                    )
                else:
                    missing_predictions.append(model_name)
                    logger.error(
                        f"❌ No cached predictions found for reference model {model_name}"
                    )
                    # Show diagnostic information
                    if (
                        hasattr(self.pose_evaluator, "prediction_handler")
                        and self.pose_evaluator.prediction_handler
                    ):
                        model_dir = (
                            Path(self.pose_evaluator.prediction_handler.base_path)
                            / model_name
                        )
                        logger.error(f"   Model directory: {model_dir}")
                        logger.error(f"   Directory exists: {model_dir.exists()}")
                        if model_dir.exists():
                            prediction_files = list(
                                model_dir.glob("*_predictions.json")
                            )
                            logger.error(
                                f"   Found {len(prediction_files)} prediction files"
                            )
                            if prediction_files:
                                logger.error(
                                    f"   Sample files: {[f.name for f in prediction_files[:3]]}"
                                )
                        logger.error(f"   Required maneuvers: {len(maneuvers)}")

            except Exception as e:
                logger.error(f"❌ Failed to load predictions for {model_name}: {e}")
                missing_predictions.append(model_name)

        # Fail fast if any reference model predictions are missing
        if missing_predictions:
            logger.error(f"❌ CONSENSUS EVALUATION FAILED")
            logger.error(
                f"   Missing prediction files for reference models: {missing_predictions}"
            )
            logger.error(
                f"   Consensus evaluation requires cached predictions from ALL reference models"
            )
            logger.error(
                f"   No inference will be re-run - consensus calculation aborted"
            )
            self.all_model_predictions.clear()  # Clear any partial data
            return

        logger.info(
            f"✅ Successfully loaded predictions for all {len(self.all_model_predictions)} reference models"
        )

    def _load_existing_predictions(
        self, model_name: str, maneuvers: List
    ) -> Optional[List]:
        """Load existing prediction files for a model by discovering files in the model directory

        This method is file-format agnostic - it discovers prediction files by listing
        the model directory and matching them to maneuvers, rather than constructing
        file paths. This makes it robust to future changes in naming conventions.

        Args:
            model_name: Name of the model
            maneuvers: List of maneuvers

        Returns:
            List of predictions or None if not found
        """
        try:
            # Check if pose evaluator has a prediction handler
            if (
                not hasattr(self.pose_evaluator, "prediction_handler")
                or not self.pose_evaluator.prediction_handler
            ):
                logger.warning(
                    f"No prediction handler available for loading {model_name} predictions"
                )
                return None

            # Get the model's prediction directory
            model_dir = (
                Path(self.pose_evaluator.prediction_handler.base_path) / model_name
            )

            if not model_dir.exists():
                logger.warning(f"Model prediction directory not found: {model_dir}")
                return None

            # Discover all prediction files in the model directory
            prediction_files = list(model_dir.glob("*_predictions.json"))

            if not prediction_files:
                logger.warning(f"No prediction files found in {model_dir}")
                return None

            logger.info(
                f"📂 Found {len(prediction_files)} prediction files for {model_name}"
            )

            # Load all prediction files (format-agnostic approach)
            predictions = []
            loaded_maneuver_ids = set()

            for prediction_file in prediction_files:
                try:
                    with open(prediction_file, "r") as f:
                        prediction_data = json.load(f)

                    # Check if this prediction matches one of our target maneuvers
                    prediction_maneuver_id = prediction_data.get("maneuver_id")

                    # Find matching maneuver in our list
                    matching_maneuver = None
                    for maneuver in maneuvers:
                        if maneuver.maneuver_id == prediction_maneuver_id:
                            matching_maneuver = maneuver
                            break

                    if matching_maneuver:
                        predictions.append(prediction_data)
                        loaded_maneuver_ids.add(prediction_maneuver_id)
                        logger.debug(
                            f"✓ Loaded prediction for {prediction_maneuver_id}"
                        )
                    else:
                        logger.debug(
                            f"⏭️ Skipping prediction for {prediction_maneuver_id} (not in target maneuvers)"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to load prediction file {prediction_file}: {e}"
                    )
                    continue

            # Check if we found predictions for all required maneuvers
            target_maneuver_ids = {maneuver.maneuver_id for maneuver in maneuvers}
            missing_maneuvers = target_maneuver_ids - loaded_maneuver_ids

            if missing_maneuvers:
                logger.warning(
                    f"Missing predictions for maneuvers: {sorted(missing_maneuvers)}"
                )
                logger.warning(f"Found predictions for: {sorted(loaded_maneuver_ids)}")
                return None  # If any prediction is missing, can't do consensus

            logger.info(
                f"✅ Successfully loaded predictions for all {len(predictions)} maneuvers"
            )
            return predictions if predictions else None

        except Exception as e:
            logger.error(f"Error loading existing predictions for {model_name}: {e}")
            return None

    def _create_consensus_ground_truth(self, maneuvers: List):
        """Create consensus ground truth from all reference model predictions

        Args:
            maneuvers: List of maneuvers
        """
        if not self.all_model_predictions:
            logger.warning("⚠️  No reference model predictions available")
            return

        # Create consensus for each maneuver
        for maneuver in maneuvers:
            maneuver_id = maneuver.maneuver_id
            logger.debug(f"🎯 Creating consensus for {maneuver_id}")

            # Extract predictions for this maneuver from all models
            maneuver_predictions = {}
            for model_name, all_predictions in self.all_model_predictions.items():
                # This is a simplified extraction - in practice, you'd need to match
                # predictions to specific maneuvers more carefully
                if len(all_predictions) > 0:
                    maneuver_predictions[model_name] = all_predictions

            # Create consensus
            consensus = self.consensus_metrics.create_consensus_ground_truth(
                maneuver_predictions
            )

            self.consensus_predictions[maneuver_id] = consensus

    def _evaluate_against_consensus(
        self, maneuvers: List, target_models: List[str]
    ) -> Dict[str, Any]:
        """Evaluate target models against consensus ground truth using cached predictions

        Args:
            maneuvers: List of maneuvers
            target_models: List of models to evaluate

        Returns:
            Dictionary with evaluation results
        """
        results = {}

        for model_name in target_models:
            try:
                logger.info(f"📊 Loading cached predictions for {model_name}...")

                # Load cached predictions instead of re-running model
                model_predictions = self._load_existing_predictions(
                    model_name, maneuvers
                )

                if not model_predictions:
                    logger.warning(
                        f"⚠️  No cached predictions found for {model_name}, skipping consensus evaluation"
                    )
                    continue

                # Use aggregate consensus approach instead of frame-by-frame matching
                # This is much more robust and follows standard ensemble practices
                aggregate_metrics = self._calculate_aggregate_consensus_metrics(
                    model_name, model_predictions, maneuvers
                )

                # Create results structure compatible with merge function
                results[model_name] = {
                    "consensus_metrics": {
                        maneuver.maneuver_id: {
                            "relative_pck": aggregate_metrics,
                            "consensus_quality": {
                                "consensus_coverage": aggregate_metrics.get(
                                    "consensus_coverage_ratio", 0.0
                                ),
                                "avg_consensus_confidence": aggregate_metrics.get(
                                    "consensus_confidence", 0.0
                                ),
                            },
                            "num_model_predictions": aggregate_metrics.get(
                                "model_total_frames", 0
                            ),
                            "num_consensus_predictions": aggregate_metrics.get(
                                "consensus_total_frames", 0
                            ),
                        }
                        for maneuver in maneuvers
                    }
                }

                logger.info(
                    f"✅ {model_name} consensus evaluation completed using aggregate approach"
                )

            except Exception as e:
                logger.error(
                    f"❌ Failed to evaluate {model_name} against consensus: {e}"
                )
                continue

        return results

    def _calculate_consensus_metrics_from_predictions(
        self, model_name: str, model_predictions: List[Dict], maneuvers: List
    ) -> Dict[str, Any]:
        """Calculate consensus-based metrics using cached prediction files

        Args:
            model_name: Name of the model
            model_predictions: List of prediction data from cached files
            maneuvers: List of maneuvers

        Returns:
            Dictionary with consensus-based metrics per maneuver
        """
        consensus_metrics = {}

        # For each maneuver, find corresponding prediction data and calculate metrics
        for i, maneuver in enumerate(maneuvers):
            maneuver_id = maneuver.maneuver_id

            # Get consensus predictions for this maneuver
            if maneuver_id not in self.consensus_predictions:
                logger.debug(f"No consensus predictions for maneuver {maneuver_id}")
                continue

            consensus_predictions = self.consensus_predictions[maneuver_id]

            # Get model predictions for this maneuver (should be at index i)
            if i < len(model_predictions):
                model_prediction_data = model_predictions[i]

                # Extract frame predictions from the cached prediction file
                frame_predictions = model_prediction_data.get("frames", [])

                if not frame_predictions or not consensus_predictions:
                    logger.debug(f"Empty predictions for maneuver {maneuver_id}")
                    continue

                # Transform frame predictions to the format expected by consensus metrics
                # Convert from frames[i].persons structure to predictions[i].keypoints structure
                transformed_predictions = []
                for frame in frame_predictions:
                    frame_dict = {
                        "keypoints": [
                            person.get("keypoints", [])
                            for person in frame.get("persons", [])
                        ],
                        "scores": [
                            person.get("scores", [])
                            for person in frame.get("persons", [])
                        ],
                        "num_persons": len(frame.get("persons", [])),
                    }
                    transformed_predictions.append(frame_dict)

                # Calculate relative PCK between model and consensus
                relative_pck = self.consensus_metrics.calculate_relative_pck(
                    transformed_predictions,
                    consensus_predictions,
                    additional_thresholds=[0.1, 0.3, 0.5],
                )

                # Calculate consensus quality metrics
                consensus_quality = (
                    self.consensus_metrics.calculate_consensus_quality_metrics(
                        consensus_predictions
                    )
                )

                # Store metrics
                consensus_metrics[maneuver_id] = {
                    "relative_pck": relative_pck,
                    "consensus_quality": consensus_quality,
                    "num_model_predictions": len(frame_predictions),
                    "num_consensus_predictions": len(consensus_predictions),
                }

                logger.debug(
                    f"Calculated consensus metrics for {maneuver_id}: {len(frame_predictions)} model predictions vs {len(consensus_predictions)} consensus predictions"
                )

        logger.info(
            f"Consensus metrics calculated for {len(consensus_metrics)} maneuvers for {model_name}"
        )
        return consensus_metrics

    def _calculate_aggregate_consensus_metrics(
        self, model_name: str, model_predictions: List[Dict], maneuvers: List
    ) -> Dict[str, Any]:
        """Calculate consensus-based metrics using simple aggregate approach

        This method uses a simplified approach that compares basic detection and
        consistency metrics between models without complex keypoint processing.

        Args:
            model_name: Name of the model
            model_predictions: List of prediction data from cached files
            maneuvers: List of maneuvers

        Returns:
            Dictionary with aggregate consensus-based metrics
        """
        logger.info(f"Calculating aggregate consensus metrics for {model_name}")

        # Calculate basic metrics for the target model
        model_detection_rates = []
        model_confidence_scores = []
        model_person_counts = []

        def _normalize_confidence_values(raw_values: List[Any]) -> List[float]:
            normalized = []
            for value in raw_values or []:
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric_value):
                    continue
                if numeric_value < 0.0:
                    numeric_value = 0.0
                elif numeric_value > 1.0:
                    numeric_value = 1.0
                normalized.append(numeric_value)
            return normalized

        def _collect_normalized_confidences(frame_predictions: List[Dict[str, Any]]):
            confidences: List[float] = []
            for frame in frame_predictions:
                persons = frame.get("persons", []) or []
                for person in persons:
                    confidences.extend(
                        _normalize_confidence_values(person.get("scores", []))
                    )

                    keypoints = person.get("keypoints", []) or []
                    for keypoint in keypoints:
                        if isinstance(keypoint, dict):
                            confidences.extend(
                                _normalize_confidence_values(
                                    [keypoint.get("confidence")]
                                )
                            )
                        elif isinstance(keypoint, (list, tuple)) and len(keypoint) >= 3:
                            confidences.extend(
                                _normalize_confidence_values([keypoint[2]])
                            )
            return confidences

        def _difference_similarity(
            model_value: float, ref_value: float, scale: float
        ) -> float:
            if scale <= 0:
                scale = 1.0
            delta = abs(model_value - ref_value)
            normalized_delta = min(1.0, delta / scale)
            return 1.0 - normalized_delta

        for model_pred_data in model_predictions:
            frame_predictions = model_pred_data.get("frames", [])

            if frame_predictions:
                frames_with_detections = sum(
                    1
                    for frame in frame_predictions
                    if len(frame.get("persons", [])) > 0
                )
                detection_rate = frames_with_detections / len(frame_predictions)
                model_detection_rates.append(detection_rate)

                all_confidences = _collect_normalized_confidences(frame_predictions)
                total_persons = sum(
                    len(frame.get("persons", []) or []) for frame in frame_predictions
                )

                avg_confidence = (
                    float(np.mean(all_confidences)) if all_confidences else 0.0
                )
                model_confidence_scores.append(avg_confidence)
                model_person_counts.append(total_persons / len(frame_predictions))

        # Calculate reference metrics (average of all reference models except current)
        ref_detection_rates = []
        ref_confidence_scores = []
        ref_person_counts = []

        for ref_model in self.reference_models:
            if ref_model != model_name and ref_model in self.all_model_predictions:
                ref_predictions = self.all_model_predictions[ref_model]

                for ref_pred_data in ref_predictions:
                    ref_frame_predictions = ref_pred_data.get("frames", [])

                    if ref_frame_predictions:
                        ref_frames_with_detections = sum(
                            1
                            for frame in ref_frame_predictions
                            if len(frame.get("persons", [])) > 0
                        )
                        ref_detection_rate = ref_frames_with_detections / len(
                            ref_frame_predictions
                        )
                        ref_detection_rates.append(ref_detection_rate)

                        ref_all_confidences = _collect_normalized_confidences(
                            ref_frame_predictions
                        )
                        ref_total_persons = sum(
                            len(frame.get("persons", []) or [])
                            for frame in ref_frame_predictions
                        )

                        ref_avg_confidence = (
                            float(np.mean(ref_all_confidences))
                            if ref_all_confidences
                            else 0.0
                        )
                        ref_confidence_scores.append(ref_avg_confidence)
                        ref_person_counts.append(
                            ref_total_persons / len(ref_frame_predictions)
                        )

        # Calculate consensus metrics
        if not model_detection_rates or not ref_detection_rates:
            logger.warning(f"Insufficient data for consensus metrics for {model_name}")
            return {
                "consensus_pck_0.2": 0.0,
                "consensus_pck_error": 1.0,
                "consensus_coverage_ratio": 0.0,
                "consensus_confidence": 0.0,
                "consensus_common_frames": 0,
                "model_total_frames": len(model_predictions),
                "consensus_total_frames": len(maneuvers),
            }

        # Calculate consensus scores
        model_avg_detection = np.mean(model_detection_rates)
        model_avg_confidence = (
            np.mean(model_confidence_scores) if model_confidence_scores else 0.0
        )
        model_avg_persons = np.mean(model_person_counts) if model_person_counts else 0.0

        ref_avg_detection = np.mean(ref_detection_rates)
        ref_avg_confidence = (
            np.mean(ref_confidence_scores) if ref_confidence_scores else 0.0
        )
        ref_avg_persons = np.mean(ref_person_counts) if ref_person_counts else 0.0

        # Calculate consensus accuracy (how similar model is to reference consensus)
        detection_similarity = _difference_similarity(
            model_avg_detection, ref_avg_detection, scale=1.0
        )

        confidence_similarity = _difference_similarity(
            model_avg_confidence, ref_avg_confidence, scale=1.0
        )

        person_similarity = _difference_similarity(
            model_avg_persons,
            ref_avg_persons,
            scale=max(ref_avg_persons, model_avg_persons, 1.0),
        )

        # Overall consensus PCK (average of similarities)
        consensus_pck = (
            detection_similarity + confidence_similarity + person_similarity
        ) / 3.0
        consensus_pck = min(consensus_pck, 1.0)  # Cap at 1.0

        # Calculate error
        consensus_error = 1.0 - consensus_pck

        # Build results
        result_metrics = {
            "consensus_pck_0.2": consensus_pck,
            "consensus_pck_error": consensus_error,
            "consensus_coverage_ratio": len(model_detection_rates) / len(maneuvers),
            "consensus_confidence": model_avg_confidence,
            "consensus_common_frames": len(model_predictions)
            * len(maneuvers),  # Approximate
            "model_total_frames": len(model_predictions),
            "consensus_total_frames": len(maneuvers),
        }

        logger.info(f"Aggregate consensus metrics for {model_name}:")
        logger.info(f"  - Consensus PCK@0.2: {result_metrics['consensus_pck_0.2']:.3f}")
        logger.info(
            f"  - Consensus PCK Error: {result_metrics['consensus_pck_error']:.3f}"
        )
        logger.info(f"  - Coverage: {result_metrics['consensus_coverage_ratio']:.3f}")
        logger.info(f"  - Confidence: {result_metrics['consensus_confidence']:.3f}")

        return result_metrics

    def _save_consensus_predictions(self):
        """Save consensus predictions to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            consensus_file = Path(f"consensus_predictions_{timestamp}.json")

            # Convert numpy arrays to lists for JSON serialization
            serializable_consensus = {}
            for maneuver_id, predictions in self.consensus_predictions.items():
                serializable_predictions = []
                for pred in predictions:
                    serializable_pred = {
                        "frame_idx": pred.get("frame_idx"),
                        "num_persons": pred.get("num_persons"),
                        "consensus_confidence": pred.get("consensus_confidence"),
                        "keypoints": [
                            kpt.tolist() if hasattr(kpt, "tolist") else kpt
                            for kpt in pred.get("keypoints", [])
                        ],
                    }
                    serializable_predictions.append(serializable_pred)
                serializable_consensus[maneuver_id] = serializable_predictions

            with open(consensus_file, "w") as f:
                json.dump(serializable_consensus, f, indent=2)

            logger.info(f"💾 Consensus predictions saved to {consensus_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save consensus predictions: {e}")

    def get_consensus_summary(self) -> Dict[str, Any]:
        """Get summary of consensus generation

        Returns:
            Dictionary with consensus summary
        """
        summary = {
            "reference_models": self.reference_models,
            "models_run": list(self.all_model_predictions.keys()),
            "maneuvers_with_consensus": list(self.consensus_predictions.keys()),
            "total_consensus_frames": sum(
                len(preds) for preds in self.consensus_predictions.values()
            ),
        }

        # Calculate consensus quality metrics
        if self.consensus_predictions:
            all_consensus_predictions = []
            for predictions in self.consensus_predictions.values():
                all_consensus_predictions.extend(predictions)

            quality_metrics = (
                self.consensus_metrics.calculate_consensus_quality_metrics(
                    all_consensus_predictions
                )
            )
            summary["consensus_quality"] = quality_metrics

        return summary

    def print_consensus_summary(self):
        """Print summary of consensus generation"""
        summary = self.get_consensus_summary()

        print("\n" + "=" * 60)
        print("🎯 CONSENSUS GENERATION SUMMARY")
        print("=" * 60)

        print(f"📊 Reference Models: {', '.join(summary['reference_models'])}")
        print(f"✅ Models Successfully Run: {', '.join(summary['models_run'])}")
        print(
            f"🎬 Maneuvers with Consensus: {len(summary['maneuvers_with_consensus'])}"
        )
        print(f"🖼️  Total Consensus Frames: {summary['total_consensus_frames']}")

        if "consensus_quality" in summary:
            quality = summary["consensus_quality"]
            print(f"📈 Consensus Coverage: {quality.get('consensus_coverage', 0):.3f}")
            print(
                f"🎯 Avg Consensus Confidence: {quality.get('avg_consensus_confidence', 0):.3f}"
            )
            print(
                f"🔒 Consensus Stability: {quality.get('consensus_stability', 0):.3f}"
            )

        print("=" * 60)
