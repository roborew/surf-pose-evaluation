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

        # Initialize pose evaluator for running models
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
        """Run consensus-based evaluation

        Args:
            maneuvers: List of maneuvers to evaluate
            target_models: Models to evaluate against consensus (if None, evaluate all available)
            save_consensus: Whether to save consensus predictions to file

        Returns:
            Dictionary with evaluation results
        """
        logger.info(
            f"🚀 Starting consensus-based evaluation with {len(maneuvers)} maneuvers"
        )
        logger.info(f"📊 Reference models: {self.reference_models}")

        # Step 1: Run all reference models to generate consensus
        logger.info("🔄 Step 1: Running reference models to generate consensus...")
        self._run_reference_models(maneuvers)

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
        """Run all reference models to collect predictions

        Args:
            maneuvers: List of maneuvers to process
        """
        for model_name in self.reference_models:
            try:
                logger.info(f"🔄 Running {model_name} for consensus generation...")

                # Check if model is available
                if model_name not in self.pose_evaluator.get_available_models():
                    logger.warning(f"⚠️  {model_name} not available, skipping")
                    continue

                # Run model evaluation
                model_results = self.pose_evaluator.evaluate_single_model_with_data(
                    model_name, maneuvers
                )

                # Extract predictions from results
                predictions = self._extract_predictions_from_results(
                    model_results, maneuvers
                )
                self.all_model_predictions[model_name] = predictions

                logger.info(
                    f"✅ {model_name} completed with {len(predictions)} predictions"
                )

            except Exception as e:
                logger.error(f"❌ Failed to run {model_name}: {e}")
                continue

    def _extract_predictions_from_results(
        self, model_results: Dict, maneuvers: List
    ) -> List[Dict[str, Any]]:
        """Extract pose predictions from model evaluation results

        Args:
            model_results: Results from model evaluation
            maneuvers: List of maneuvers

        Returns:
            List of pose predictions
        """
        predictions = []

        # Extract predictions from each maneuver
        for maneuver in maneuvers:
            maneuver_id = maneuver.maneuver_id

            # Look for predictions in the results
            if "maneuver_results" in model_results:
                for result in model_results["maneuver_results"]:
                    if result.get("maneuver_id") == maneuver_id:
                        # Extract pose results from the maneuver
                        pose_results = result.get("pose_results", [])
                        predictions.extend(pose_results)
                        break

        return predictions

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
        """Evaluate target models against consensus ground truth

        Args:
            maneuvers: List of maneuvers
            target_models: List of models to evaluate

        Returns:
            Dictionary with evaluation results
        """
        results = {}

        for model_name in target_models:
            try:
                logger.info(f"📊 Evaluating {model_name} against consensus...")

                # Check if model is available
                if model_name not in self.pose_evaluator.get_available_models():
                    logger.warning(f"⚠️  {model_name} not available, skipping")
                    continue

                # Run model evaluation
                model_results = self.pose_evaluator.evaluate_single_model_with_data(
                    model_name, maneuvers
                )

                # Calculate consensus-based metrics
                consensus_metrics = self._calculate_consensus_metrics_for_model(
                    model_name, model_results, maneuvers
                )

                # Combine with original results
                model_results["consensus_metrics"] = consensus_metrics
                results[model_name] = model_results

                logger.info(f"✅ {model_name} consensus evaluation completed")

            except Exception as e:
                logger.error(
                    f"❌ Failed to evaluate {model_name} against consensus: {e}"
                )
                continue

        return results

    def _calculate_consensus_metrics_for_model(
        self, model_name: str, model_results: Dict, maneuvers: List
    ) -> Dict[str, Any]:
        """Calculate consensus-based metrics for a specific model

        Args:
            model_name: Name of the model
            model_results: Results from model evaluation
            maneuvers: List of maneuvers

        Returns:
            Dictionary with consensus-based metrics
        """
        consensus_metrics = {}

        for maneuver in maneuvers:
            maneuver_id = maneuver.maneuver_id

            # Get consensus predictions for this maneuver
            if maneuver_id not in self.consensus_predictions:
                continue

            consensus_predictions = self.consensus_predictions[maneuver_id]

            # Extract model predictions for this maneuver
            model_predictions = self._extract_maneuver_predictions(
                model_results, maneuver_id
            )

            if not model_predictions or not consensus_predictions:
                continue

            # Calculate relative PCK
            relative_pck = self.consensus_metrics.calculate_relative_pck(
                model_predictions, consensus_predictions
            )

            # Calculate consensus quality
            consensus_quality = (
                self.consensus_metrics.calculate_consensus_quality_metrics(
                    consensus_predictions
                )
            )

            # Store metrics
            consensus_metrics[maneuver_id] = {
                "relative_pck": relative_pck,
                "consensus_quality": consensus_quality,
                "num_model_predictions": len(model_predictions),
                "num_consensus_predictions": len(consensus_predictions),
            }

        return consensus_metrics

    def _extract_maneuver_predictions(
        self, model_results: Dict, maneuver_id: str
    ) -> List[Dict[str, Any]]:
        """Extract predictions for a specific maneuver

        Args:
            model_results: Results from model evaluation
            maneuver_id: ID of the maneuver

        Returns:
            List of predictions for the maneuver
        """
        # Look for predictions in the results
        if "maneuver_results" in model_results:
            for result in model_results["maneuver_results"]:
                if result.get("maneuver_id") == maneuver_id:
                    return result.get("pose_results", [])

        return []

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
