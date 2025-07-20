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
                        logger.error(
                            f"   Prediction handler base path: {self.pose_evaluator.prediction_handler.base_path}"
                        )
                        for maneuver in maneuvers[:2]:  # Show first 2 for brevity
                            expected_file = self.pose_evaluator.prediction_handler._get_prediction_file_path(
                                maneuver.maneuver_id, model_name
                            )
                            logger.error(f"   Expected file: {expected_file}")
                            logger.error(f"   File exists: {expected_file.exists()}")

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
        """Load existing prediction files for a model

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

            predictions = []

            # Load prediction files for each maneuver
            for maneuver in maneuvers:
                try:
                    # Try to load the prediction file for this maneuver
                    prediction_file = self.pose_evaluator.prediction_handler._get_prediction_file_path(
                        maneuver.maneuver_id, model_name
                    )

                    if prediction_file.exists():
                        import json

                        with open(prediction_file, "r") as f:
                            prediction_data = json.load(f)
                        predictions.append(prediction_data)
                    else:
                        logger.warning(f"Prediction file not found: {prediction_file}")
                        return None  # If any prediction is missing, can't do consensus

                except Exception as e:
                    logger.warning(
                        f"Failed to load prediction for maneuver {maneuver.maneuver_id}: {e}"
                    )
                    return None

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

                # Calculate consensus-based metrics using cached predictions
                consensus_metrics = self._calculate_consensus_metrics_from_predictions(
                    model_name, model_predictions, maneuvers
                )

                # Create results structure compatible with merge function
                results[model_name] = {"consensus_metrics": consensus_metrics}

                logger.info(
                    f"✅ {model_name} consensus evaluation completed using cached predictions"
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
                frame_predictions = model_prediction_data.get("frame_predictions", [])

                if not frame_predictions or not consensus_predictions:
                    logger.debug(f"Empty predictions for maneuver {maneuver_id}")
                    continue

                # Calculate relative PCK between model and consensus
                relative_pck = self.consensus_metrics.calculate_relative_pck(
                    frame_predictions, consensus_predictions
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
