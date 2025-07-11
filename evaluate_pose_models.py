#!/usr/bin/env python3
"""
Surfing Pose Estimation Evaluation Framework
Main evaluation script for comparing pose estimation models
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json
import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import mlflow
import optuna
from tqdm import tqdm

from data_handling.data_loader import SurfingDataLoader
from models.base_pose_model import BasePoseModel
from models.mediapipe_wrapper import MediaPipeWrapper
from metrics.pose_metrics import PoseMetrics
from metrics.performance_metrics import PerformanceMetrics
from utils.mlflow_utils import MLflowManager
from utils.visualization import VisualizationUtils

# Import other model wrappers when available
try:
    from models.mmpose_wrapper import MMPoseWrapper
except ImportError:
    MMPoseWrapper = None

try:
    from models.yolov8_wrapper import YOLOv8Wrapper
except ImportError:
    YOLOv8Wrapper = None

try:
    from models.hrnet_wrapper import HRNetWrapper
except ImportError:
    HRNetWrapper = None


class PoseEvaluator:
    """Main evaluation class for pose estimation models"""

    def __init__(self, config_path: str):
        """Initialize evaluator with configuration"""
        self.config = self._load_config(config_path)
        self.device = self._setup_device()

        # Initialize components
        self.data_loader = SurfingDataLoader(self.config)
        self.pose_metrics = PoseMetrics()
        self.performance_metrics = PerformanceMetrics(device=self.device)
        self.mlflow_manager = MLflowManager(self.config["mlflow"])
        self.visualizer = VisualizationUtils()

        # Model registry
        self.model_registry = self._setup_model_registry()

        # Setup logging
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_device(self) -> str:
        """Setup compute device based on availability"""
        device = self.config.get("device", "auto")

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logging.info(f"Using device: {device}")
        return device

    def _setup_model_registry(self) -> Dict[str, BasePoseModel]:
        """Setup available pose estimation models"""
        registry = {}

        # MediaPipe (always available)
        registry["mediapipe"] = MediaPipeWrapper

        # MMPose (if available)
        if MMPoseWrapper is not None:
            registry["mmpose"] = MMPoseWrapper

        # YOLOv8-Pose (if available)
        if YOLOv8Wrapper is not None:
            registry["yolov8_pose"] = YOLOv8Wrapper

        # HRNet (if available)
        if HRNetWrapper is not None:
            registry["hrnet"] = HRNetWrapper

        return registry

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("pose_evaluation.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.model_registry.keys())

    def quick_screening(self, models: List[str], num_clips: int = 50) -> Dict:
        """Quick screening evaluation with limited data"""
        logging.info(f"Starting quick screening with {num_clips} clips")

        # Load limited dataset
        clips = self.data_loader.load_clips(max_clips=num_clips)

        results = {}
        for model_name in models:
            if model_name not in self.model_registry:
                logging.warning(f"Model {model_name} not available, skipping")
                continue

            logging.info(f"Evaluating {model_name}")
            model_results = self._evaluate_single_model(model_name, clips)
            results[model_name] = model_results

        return results

    def full_evaluation(
        self,
        models: List[str],
        num_clips: Optional[int] = None,
        use_optuna: bool = False,
    ) -> Dict:
        """Comprehensive evaluation with full dataset"""
        logging.info("Starting full evaluation")

        # Load full dataset
        clips = self.data_loader.load_clips(max_clips=num_clips)

        results = {}
        for model_name in models:
            if model_name not in self.model_registry:
                logging.warning(f"Model {model_name} not available, skipping")
                continue

            logging.info(f"Evaluating {model_name}")

            if use_optuna:
                model_results = self._evaluate_with_optimization(model_name, clips)
            else:
                model_results = self._evaluate_single_model(model_name, clips)

            results[model_name] = model_results

        return results

    def _evaluate_single_model(self, model_name: str, clips: List) -> Dict:
        """Evaluate a single pose estimation model"""

        # Load model configuration
        model_config_path = f"configs/model_configs/{model_name}.yaml"
        if os.path.exists(model_config_path):
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
        else:
            model_config = {}

        # Initialize model
        model_class = self.model_registry[model_name]
        model = model_class(device=self.device, **model_config)

        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            # Log model parameters
            mlflow.log_params(model_config)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("device", self.device)
            mlflow.log_param("num_clips", len(clips))

            # Evaluation metrics
            all_pose_metrics = []
            all_performance_metrics = []

            # Process clips
            for clip in tqdm(clips, desc=f"Processing {model_name}"):
                try:
                    # Process video
                    clip_metrics = self._process_video_clip(model, clip)
                    all_pose_metrics.append(clip_metrics["pose"])
                    all_performance_metrics.append(clip_metrics["performance"])

                except Exception as e:
                    logging.error(f"Error processing clip {clip.clip_path}: {e}")
                    continue

            # Aggregate results
            aggregated_metrics = self._aggregate_metrics(
                all_pose_metrics, all_performance_metrics
            )

            # Log metrics to MLflow
            for metric_name, value in aggregated_metrics.items():
                mlflow.log_metric(metric_name, value)

            # Save model artifacts
            model_artifacts_path = f"models/{model_name}"
            os.makedirs(model_artifacts_path, exist_ok=True)
            mlflow.log_artifacts(model_artifacts_path)

            return aggregated_metrics

    def _process_video_clip(self, model: BasePoseModel, clip) -> Dict:
        """Process a single video clip with pose estimation"""

        # Load video frames
        frames = self.data_loader.load_video_frames(clip)

        # Performance tracking
        inference_times = []
        memory_usage = []

        # Process frames
        pose_results = []
        for frame in frames:
            # Measure inference time
            start_time = time.time()

            # Run pose estimation
            pose_result = model.predict(frame)

            # Record performance
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Record memory usage
            if self.device == "cuda":
                memory_usage.append(torch.cuda.memory_allocated())

            pose_results.append(pose_result)

        # Calculate pose metrics (if ground truth available)
        pose_metrics = {}
        if hasattr(clip, "annotations") and clip.annotations:
            pose_metrics = self.pose_metrics.calculate_metrics(
                pose_results, clip.annotations
            )

        # Calculate performance metrics
        performance_metrics = {
            "avg_inference_time": np.mean(inference_times),
            "fps": 1.0 / np.mean(inference_times),
            "max_memory_usage": max(memory_usage) if memory_usage else 0,
            "total_frames": len(frames),
        }

        return {"pose": pose_metrics, "performance": performance_metrics}

    def _aggregate_metrics(
        self, pose_metrics: List[Dict], performance_metrics: List[Dict]
    ) -> Dict:
        """Aggregate metrics across all clips"""
        aggregated = {}

        # Aggregate pose metrics
        if pose_metrics and pose_metrics[0]:
            pose_keys = pose_metrics[0].keys()
            for key in pose_keys:
                values = [m[key] for m in pose_metrics if key in m]
                if values:
                    aggregated[f"pose_{key}_mean"] = np.mean(values)
                    aggregated[f"pose_{key}_std"] = np.std(values)

        # Aggregate performance metrics
        perf_keys = performance_metrics[0].keys()
        for key in perf_keys:
            values = [m[key] for m in performance_metrics if key in m]
            if values:
                aggregated[f"perf_{key}_mean"] = np.mean(values)
                aggregated[f"perf_{key}_std"] = np.std(values)

        return aggregated

    def _evaluate_with_optimization(self, model_name: str, clips: List) -> Dict:
        """Evaluate model with Optuna hyperparameter optimization"""

        def objective(trial):
            # Sample hyperparameters
            config = self._sample_hyperparameters(trial, model_name)

            # Evaluate with sampled parameters
            model_class = self.model_registry[model_name]
            model = model_class(device=self.device, **config)

            # Quick evaluation on subset
            subset_clips = clips[: min(20, len(clips))]
            metrics = []

            for clip in subset_clips:
                try:
                    clip_metrics = self._process_video_clip(model, clip)
                    if clip_metrics["pose"]:
                        metrics.append(clip_metrics["pose"].get("pck_0_2", 0))
                except:
                    continue

            return np.mean(metrics) if metrics else 0

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        # Evaluate with best parameters
        best_config = study.best_params
        model_class = self.model_registry[model_name]
        model = model_class(device=self.device, **best_config)

        # Full evaluation with best parameters
        return self._evaluate_single_model(model_name, clips)

    def _sample_hyperparameters(self, trial, model_name: str) -> Dict:
        """Sample hyperparameters for optimization"""
        # This would be model-specific
        # For now, return empty dict
        return {}

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Add metadata
        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "config": self.config,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logging.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pose estimation models for surfing analysis"
    )
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--models", nargs="+", help="Models to evaluate")
    parser.add_argument(
        "--quick-test", action="store_true", help="Run quick screening test"
    )
    parser.add_argument(
        "--max-clips", type=int, help="Maximum number of clips to process"
    )
    parser.add_argument(
        "--use-optuna",
        action="store_true",
        help="Use Optuna for hyperparameter optimization",
    )
    parser.add_argument(
        "--output",
        default="results/pose_comparison_results.json",
        help="Output file path",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize evaluator
    evaluator = PoseEvaluator(args.config)

    # Determine models to evaluate
    if args.models:
        models_to_eval = args.models
    else:
        models_to_eval = evaluator.get_available_models()

    # Check available models
    available_models = evaluator.get_available_models()
    models_to_eval = [m for m in models_to_eval if m in available_models]

    if not models_to_eval:
        print("No valid models specified or available")
        print(f"Available models: {available_models}")
        return

    print(f"Evaluating models: {models_to_eval}")
    print(f"Available models: {available_models}")

    # Run evaluation
    if args.quick_test:
        max_clips = args.max_clips or 10
        results = evaluator.quick_screening(models_to_eval, num_clips=max_clips)
    else:
        results = evaluator.full_evaluation(
            models_to_eval, num_clips=args.max_clips, use_optuna=args.use_optuna
        )

    # Save results
    evaluator.save_results(results, args.output)

    # Print summary
    print("\nEvaluation Summary:")
    for model_name, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
