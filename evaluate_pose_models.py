#!/usr/bin/env python3
"""
Surfing Pose Estimation Evaluation Framework
Main evaluation script for comparing pose estimation models
"""

# Set MediaPipe/TensorFlow environment variables before any imports
import os
import platform

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# CUDA-first approach: Enable GPU on Linux, fallback to CPU on macOS for stability
if platform.system().lower() == "darwin":  # macOS
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # Keep CPU for macOS stability
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for macOS
    os.environ["TF_DISABLE_MKL"] = "1"  # Disable Intel MKL
else:  # Linux/Windows - Enable GPU acceleration
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"  # Enable GPU acceleration
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth

os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"  # Stability

import argparse
import logging
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
from utils.pose_video_visualizer import PoseVideoVisualizer
from utils.prediction_file_format import (
    PredictionFileHandler,
    get_keypoint_format_for_model,
    get_keypoint_names_for_model,
)

# Import other model wrappers when available
try:
    from models.blazepose_wrapper import BlazePoseWrapper
except ImportError:
    BlazePoseWrapper = None

try:
    from models.mmpose_wrapper import MMPoseWrapper
except ImportError:
    MMPoseWrapper = None

try:
    from models.yolov8_wrapper import YOLOv8Wrapper
except ImportError:
    YOLOv8Wrapper = None

try:
    from models.pytorch_pose_wrapper import PyTorchPoseWrapper
except ImportError:
    PyTorchPoseWrapper = None

# HRNet functionality is available through MMPose framework


class PoseEvaluator:
    """Main evaluation class for pose estimation models"""

    def __init__(self, config_path: str, video_format_override: str = None):
        """Initialize evaluator with configuration"""
        self.config = self._load_config(config_path)
        self.video_format_override = video_format_override
        self.device = self._setup_device()

        # Initialize components
        self.data_loader = SurfingDataLoader(self.config)
        self.pose_metrics = PoseMetrics()
        self.performance_metrics = PerformanceMetrics(device=self.device)
        self.mlflow_manager = MLflowManager()
        self.visualizer = VisualizationUtils()

        # Initialize video visualizer with encoding configuration
        encoding_config = (
            self.config.get("output", {}).get("visualization", {}).get("encoding", {})
        )
        self.video_visualizer = PoseVideoVisualizer(encoding_config)

        # Initialize prediction file handler
        prediction_config = self.config.get("output", {}).get("predictions", {})
        if prediction_config.get("enabled", True):
            # Use shared storage if configured, otherwise use local path
            shared_storage_path = prediction_config.get("shared_storage_path")
            if shared_storage_path:
                prediction_base_path = shared_storage_path
                logging.info(f"Using shared prediction storage: {prediction_base_path}")
            else:
                # Default to a timestamped predictions directory in the data hierarchy
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                prediction_base_path = prediction_config.get(
                    "base_path",
                    f"./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/manual_run_{timestamp}/predictions",
                )
                logging.info(f"Using local prediction storage: {prediction_base_path}")

            self.prediction_handler = PredictionFileHandler(prediction_base_path)
        else:
            self.prediction_handler = None
            logging.info("Prediction file generation disabled")

        # Model registry
        self.model_registry = self._setup_model_registry()

        # Setup logging
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_device(self) -> str:
        """Setup compute device with CUDA-first priority for production"""
        device = self.config.get("device", "auto")

        if device == "auto":
            # CUDA-first approach for production environments
            if torch.cuda.is_available():
                device = "cuda"
                logging.info(f"CUDA detected: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = "mps"
                logging.info("MPS detected (Apple Silicon)")
            else:
                device = "cpu"
                logging.info("No GPU acceleration available, using CPU")

        logging.info(f"Using device: {device}")
        return device

    def _setup_model_registry(self) -> Dict[str, BasePoseModel]:
        """Setup available pose estimation models"""
        registry = {}

        # MediaPipe (always available)
        registry["mediapipe"] = MediaPipeWrapper

        # BlazePose (if available)
        if BlazePoseWrapper is not None:
            registry["blazepose"] = BlazePoseWrapper

        # MMPose (if available)
        if MMPoseWrapper is not None:
            registry["mmpose"] = MMPoseWrapper

        # YOLOv8-Pose (if available)
        if YOLOv8Wrapper is not None:
            registry["yolov8_pose"] = YOLOv8Wrapper

        # PyTorch KeypointRCNN (if available)
        if PyTorchPoseWrapper is not None:
            registry["pytorch_pose"] = PyTorchPoseWrapper

        # HRNet functionality available through MMPose framework

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

    def _get_video_format(self) -> str:
        """Get video format to use, with command-line override support"""
        if self.video_format_override:
            return self.video_format_override
        return self.config["dataset"]["video_clips"].get("input_format", "h264")

    def quick_screening(self, models: List[str], num_clips: int = 50) -> Dict:
        """Quick screening evaluation with limited data"""
        logging.info(f"Starting quick screening with {num_clips} clips")

        # Load limited dataset - use maneuvers instead of full clips
        video_format = self._get_video_format()
        logging.info(f"Using video format: {video_format}")
        maneuvers = self.data_loader.load_maneuvers(
            max_clips=num_clips, maneuvers_per_clip=1, video_format=video_format
        )
        logging.info(f"Loaded {len(maneuvers)} maneuvers for quick screening")

        results = {}
        for model_name in models:
            if model_name not in self.model_registry:
                logging.warning(f"Model {model_name} not available, skipping")
                continue

            logging.info(f"Evaluating {model_name}")
            model_results = self._evaluate_single_model(model_name, maneuvers)
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

        # Load full dataset - use maneuvers instead of full clips
        video_format = self._get_video_format()
        logging.info(f"Using video format: {video_format}")
        maneuvers = self.data_loader.load_maneuvers(
            max_clips=num_clips, video_format=video_format
        )
        logging.info(f"Loaded {len(maneuvers)} maneuvers for full evaluation")

        results = {}
        for model_name in models:
            if model_name not in self.model_registry:
                logging.warning(f"Model {model_name} not available, skipping")
                continue

            logging.info(f"Evaluating {model_name}")

            if use_optuna:
                model_results = self._evaluate_with_optimization(model_name, maneuvers)
            else:
                model_results = self._evaluate_single_model(model_name, maneuvers)

            results[model_name] = model_results

        return results

    def _load_best_params_from_optuna(self, model_name: str) -> Dict:
        """Load best parameters from Optuna optimization results"""
        # Check if we should load best params from config
        load_config = self.config.get("models", {}).get("load_best_params", {})
        if not load_config.get("enabled", False):
            return {}

        source_path = load_config.get("source_path", "./results/best_params")
        best_params_file = Path(source_path) / "best_parameters.yaml"

        if not best_params_file.exists():
            if load_config.get("fallback_to_defaults", True):
                logging.warning(
                    f"Best parameters file not found: {best_params_file}, using defaults"
                )
                return {}
            else:
                raise FileNotFoundError(
                    f"Best parameters file not found: {best_params_file}"
                )

        try:
            with open(best_params_file, "r") as f:
                all_best_params = yaml.safe_load(f)

            model_params = all_best_params.get(model_name, {})
            if model_params:
                logging.info(f"Loaded best parameters for {model_name}: {model_params}")
            else:
                logging.warning(
                    f"No best parameters found for {model_name} in {best_params_file}"
                )

            return model_params

        except Exception as e:
            if load_config.get("fallback_to_defaults", True):
                logging.warning(f"Failed to load best parameters: {e}, using defaults")
                return {}
            else:
                raise

    def _evaluate_single_model(self, model_name: str, maneuvers: List) -> Dict:
        """Evaluate a single pose estimation model"""

        # Load model configuration
        model_config_path = f"configs/model_configs/{model_name}.yaml"
        if os.path.exists(model_config_path):
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
        else:
            model_config = {}

        # Load best parameters from Optuna if available
        best_params = self._load_best_params_from_optuna(model_name)

        # Merge configurations: best_params override model_config
        final_config = {**model_config, **best_params}

        # Log parameter source
        if best_params:
            logging.info(f"Using optimized parameters for {model_name}: {best_params}")
            param_source = "optuna_optimized"
        else:
            logging.info(f"Using default parameters for {model_name}")
            param_source = "default_config"

        # Initialize model
        model_class = self.model_registry[model_name]
        model = model_class(device=self.device, **final_config)

        # Start MLflow run
        run_name = f"{model_name}_evaluation"
        if best_params:
            run_name = f"{model_name}_evaluation_optimized"

        with mlflow.start_run(run_name=run_name):
            # Log model parameters (safely handle long strings)
            safe_config = self._sanitize_mlflow_params(final_config)

            # Add detailed description for standard evaluation
            config_preview = list(safe_config.items())[:10]
            config_text = chr(10).join([f"- {k}: {v}" for k, v in config_preview])
            if len(safe_config) > 10:
                config_text += "\n..."

            # Create detailed description
            description = f"""
STANDARD MODEL EVALUATION - {"Optimized Configuration" if best_params else "Default Configuration"}

PURPOSE: Comprehensive evaluation of {model_name} pose estimation model
DATA SCOPE: Full evaluation on {len(maneuvers)} maneuvers (complete dataset)
PARAMETER SOURCE: {param_source.replace('_', ' ').title()}

This run evaluates the {model_name} model using {'optimal hyperparameters discovered through Optuna optimization' if best_params else 'default configuration parameters'}. 
The evaluation covers the complete dataset to provide accurate performance metrics
for production deployment decisions.

EVALUATION DETAILS:
- Model: {model_name}
- Parameter source: {param_source}
- Data coverage: Complete dataset ({len(maneuvers)} maneuvers)
- Evaluation type: Comprehensive (all metrics computed)
- Expected runtime: 10-30 minutes depending on model and dataset size

CONFIGURATION USED:
{config_text}

PERFORMANCE INTERPRETATION:
- pose_pck_error_mean: Lower is better (target: <0.3 for good performance)
- perf_fps_mean: Higher is better (target: >15 FPS for real-time)
- pose_detection_f1_mean: Higher is better (target: >0.7 for reliable detection)
- Memory metrics: Lower is better for deployment efficiency

USAGE RECOMMENDATIONS:
✅ Use these metrics for production deployment decisions
✅ Compare against other models for backbone selection
✅ Cite these results in research papers or reports
✅ Use this configuration for actual surfing analysis system
            """.strip()

            mlflow.set_tag("mlflow.note.content", description)

            # Log model parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("parameter_source", param_source)
            mlflow.log_param("optimization_mode", "standard_evaluation")
            mlflow.log_param(
                "data_scope", f"complete_dataset_{len(maneuvers)}_maneuvers"
            )
            mlflow.log_param("purpose", "model_evaluation")

            # Log whether optimized parameters were used
            mlflow.log_param("uses_optimized_params", bool(best_params))

            # Log configuration parameters
            for param_name, param_value in safe_config.items():
                mlflow.log_param(param_name, param_value)

            # Run evaluation
            result = self._evaluate_single_model_internal(
                model, model_name, maneuvers, log_to_mlflow=True
            )

            # Generate sample visualizations
            if (
                self.config.get("output", {})
                .get("visualization", {})
                .get("enabled", False)
            ):
                self._create_sample_visualizations(model, model_name, maneuvers[:3])

            return result

    def _sanitize_mlflow_params(self, config: Dict, max_length: int = 500) -> Dict:
        """Sanitize configuration parameters for MLflow logging

        Args:
            config: Configuration dictionary
            max_length: Maximum string length for MLflow parameters

        Returns:
            Sanitized configuration dictionary
        """
        sanitized = {}

        def sanitize_value(value):
            if isinstance(value, str):
                if len(value) > max_length:
                    return value[: max_length - 3] + "..."
                return value
            elif isinstance(value, (list, tuple)):
                # Convert list to string representation
                list_str = str(value)
                if len(list_str) > max_length:
                    return list_str[: max_length - 3] + "..."
                return list_str
            else:
                # Convert other types to string
                str_value = str(value)
                if len(str_value) > max_length:
                    return str_value[: max_length - 3] + "..."
                return str_value

        def flatten_dict(d, parent_key="", sep="_"):
            """Flatten nested dictionary with separator"""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, sanitize_value(v)))
            return dict(items)

        # Flatten and sanitize the config
        sanitized = flatten_dict(config)

        return sanitized

    def _process_video_clip(self, model: BasePoseModel, clip) -> Dict:
        """Process a single video clip with pose estimation"""

        # Load video frames
        frames = self.data_loader.load_video_frames(clip)

        # Performance tracking
        inference_times = []
        memory_usage = []

        # Process frames
        pose_results = []
        total_frames = len(frames)

        for i, frame in enumerate(frames):
            # Show frame progress (update every 5 frames)
            if i % 5 == 0 or i == total_frames - 1:
                print(
                    f"\r      Frame {i+1}/{total_frames} [{100*(i+1)//total_frames:3d}%]",
                    end="",
                    flush=True,
                )

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

    def _process_video_maneuver(
        self,
        model: BasePoseModel,
        maneuver,
        model_name: str = None,
        generate_predictions: bool = True,
    ) -> Dict:
        """Process a single maneuver with pose estimation and optionally generate prediction files"""
        from data_handling.data_loader import Maneuver

        # Load frames for this specific maneuver
        frames = self.data_loader.load_video_frames(maneuver)

        # Performance tracking
        inference_times = []
        memory_usage = []

        # Process frames
        pose_results = []
        total_frames = len(frames)

        for i, frame in enumerate(frames):
            # Show frame progress (update every 5 frames)
            if i % 5 == 0 or i == total_frames - 1:
                print(
                    f"\r      Frame {i+1}/{total_frames} [{100*(i+1)//total_frames:3d}%]",
                    end="",
                    flush=True,
                )

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

        # Generate standardized prediction file if requested
        if generate_predictions and model_name and self.prediction_handler:
            try:
                print(
                    f"\n    📁 Generating prediction file for {maneuver.maneuver_id}..."
                )

                # Get model-specific keypoint format
                keypoint_format = get_keypoint_format_for_model(model_name)
                keypoint_names = get_keypoint_names_for_model(model_name)

                # Get model configuration
                model_config = {}
                if hasattr(model, "model_config"):
                    model_config = model.model_config
                elif hasattr(model, "get_model_info"):
                    model_config = model.get_model_info()

                # Convert pose results to standardized format
                frame_predictions = []
                for frame_idx, pose_result in enumerate(pose_results):
                    frame_prediction = (
                        self.prediction_handler.convert_model_prediction_to_standard(
                            model_result=pose_result,
                            frame_id=frame_idx,
                            absolute_frame_id=maneuver.start_frame + frame_idx,
                            timestamp=(maneuver.start_frame + frame_idx) / maneuver.fps,
                            keypoint_format=keypoint_format,
                            keypoint_names=keypoint_names,
                        )
                    )
                    frame_predictions.append(frame_prediction)

                # Create complete maneuver prediction
                maneuver_prediction = (
                    self.prediction_handler.create_maneuver_prediction(
                        maneuver=maneuver,
                        model_name=model_name,
                        model_config=model_config,
                        keypoint_format=keypoint_format,
                        keypoint_names=keypoint_names,
                        frame_predictions=frame_predictions,
                    )
                )

                # Save prediction file
                prediction_file_path = self.prediction_handler.save_prediction_file(
                    maneuver_prediction
                )
                print(
                    f"    ✅ Saved prediction file: {Path(prediction_file_path).name}"
                )
                logging.info(f"Saved prediction file: {prediction_file_path}")

            except Exception as e:
                print(f"    ❌ Failed to generate prediction file: {e}")
                logging.error(
                    f"Failed to generate prediction file for {maneuver.maneuver_id}: {e}"
                )
                import traceback

                traceback.print_exc()
        elif generate_predictions and model_name and not self.prediction_handler:
            print(
                f"    ⚠️ Prediction generation requested but no prediction handler available"
            )
        elif generate_predictions and not model_name:
            print(f"    ⚠️ Prediction generation requested but no model name provided")

        # Calculate pose metrics (if ground truth available)
        pose_metrics = {}
        if hasattr(maneuver, "annotation_data") and maneuver.annotation_data:
            # For maneuvers, we have the ground truth annotation
            pose_metrics = self.pose_metrics.calculate_metrics(
                pose_results, [maneuver.annotation_data]
            )

        # Calculate performance metrics
        performance_metrics = {
            "avg_inference_time": np.mean(inference_times),
            "fps": 1.0 / np.mean(inference_times),
            "max_memory_usage": max(memory_usage) if memory_usage else 0,
            "total_frames": len(frames),
            "maneuver_type": maneuver.maneuver_type,
            "maneuver_duration": maneuver.duration,
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
                # Only aggregate numeric values
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    aggregated[f"perf_{key}_mean"] = np.mean(numeric_values)
                    aggregated[f"perf_{key}_std"] = np.std(numeric_values)
                else:
                    # For non-numeric values, just take the first one but don't log to MLflow
                    # (MLflow only accepts numeric values for metrics)
                    pass

        return aggregated

    def _evaluate_with_optimization(self, model_name: str, maneuvers: List) -> Dict:
        """Evaluate model with Optuna hyperparameter optimization"""
        import optuna
        import numpy as np
        import time

        # Storage for all trial results
        all_trial_results = []
        best_trial_result = None
        best_score = -float("inf")

        def objective(trial):
            nonlocal best_trial_result, best_score

            # Sample hyperparameters
            config = self._sample_hyperparameters(trial, model_name)

            # Create descriptive run name with key parameters
            param_summary = self._create_param_summary(config)
            run_name = f"{model_name}_optuna_trial_{trial.number:03d}_{param_summary}"

            # Start MLflow run for this trial
            with mlflow.start_run(run_name=run_name):
                # Add detailed description for this trial
                trial_description = f"""
OPTUNA TRIAL #{trial.number:03d} - Hyperparameter Exploration

PURPOSE: Testing specific hyperparameter combination to find optimal settings
DATA SCOPE: Quick evaluation on {min(20, len(maneuvers))} maneuvers (subset for speed)
PARAMETERS TESTED: {param_summary}

This trial tests a specific combination of hyperparameters sampled by Optuna's TPE algorithm.
The goal is to quickly evaluate whether this parameter combination shows promise before
running expensive full evaluation. Results guide the optimization toward better parameters.

TRIAL DETAILS:
- Model: {model_name}
- Parameter combination: {config}
- Optimization objective: Maximize PCK@0.2 (pose accuracy)
- Data used: Subset of maneuvers for fast iteration
- Expected runtime: 1-3 minutes

INTERPRETATION:
- High trial_score (>0.5): Promising parameter combination
- Low trial_score (<0.3): Poor parameter combination  
- Compare across trials to identify parameter patterns
                """.strip()

                mlflow.set_tag("mlflow.note.content", trial_description)

                # Log trial information
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("trial_number", trial.number)
                mlflow.log_param("optimization_mode", "optuna_trial")
                mlflow.log_param(
                    "data_scope", f"subset_{min(20, len(maneuvers))}_maneuvers"
                )
                mlflow.log_param("purpose", "hyperparameter_exploration")

                # Log sampled hyperparameters
                for param_name, param_value in config.items():
                    mlflow.log_param(f"hp_{param_name}", param_value)

                # Initialize model with sampled parameters
                model_class = self.model_registry[model_name]
                model = model_class(device=self.device, **config)

                # Quick evaluation on subset for optimization
                subset_maneuvers = maneuvers[: min(20, len(maneuvers))]
                trial_metrics = []
                trial_pose_metrics = []
                trial_perf_metrics = []

                print(f"\n🔄 Trial {trial.number:03d}: {param_summary}")

                for i, maneuver in enumerate(subset_maneuvers):
                    try:
                        maneuver_metrics = self._process_video_maneuver(
                            model, maneuver, model_name, generate_predictions=False
                        )
                        if maneuver_metrics["pose"]:
                            pck_score = maneuver_metrics["pose"].get("pck_0_2", 0)
                            trial_metrics.append(pck_score)
                            trial_pose_metrics.append(maneuver_metrics["pose"])
                            trial_perf_metrics.append(maneuver_metrics["performance"])

                        # Progress indicator for trials
                        if (i + 1) % 5 == 0:
                            print(
                                f"   • Processed {i + 1}/{len(subset_maneuvers)} maneuvers..."
                            )

                    except Exception as e:
                        logging.warning(
                            f"Failed to process maneuver in trial {trial.number}: {e}"
                        )
                        continue

                # Calculate trial score (primary objective)
                trial_score = np.mean(trial_metrics) if trial_metrics else 0

                # Log trial metrics
                if trial_pose_metrics and trial_perf_metrics:
                    aggregated = self._aggregate_metrics(
                        trial_pose_metrics, trial_perf_metrics
                    )

                    # Log all aggregated metrics
                    for metric_name, value in aggregated.items():
                        mlflow.log_metric(metric_name, value)

                    # Log optimization-specific metrics
                    mlflow.log_metric("optuna_trial_score", trial_score)
                    mlflow.log_metric("num_maneuvers_processed", len(trial_metrics))

                    trial_result = {
                        "trial_number": trial.number,
                        "config": config,
                        "score": trial_score,
                        "metrics": aggregated,
                        "run_name": run_name,
                    }
                    all_trial_results.append(trial_result)

                    # Track best trial
                    if trial_score > best_score:
                        best_score = trial_score
                        best_trial_result = trial_result

                        # Add performance indicator tags
                        mlflow.set_tag("performance_tier", "best_so_far")
                        mlflow.set_tag("is_current_best", "true")
                    elif trial_score > best_score * 0.95:  # Within 5% of best
                        mlflow.set_tag("performance_tier", "high_performance")
                        mlflow.set_tag("is_current_best", "false")
                    elif trial_score > best_score * 0.8:  # Within 20% of best
                        mlflow.set_tag("performance_tier", "medium_performance")
                        mlflow.set_tag("is_current_best", "false")
                    else:
                        mlflow.set_tag("performance_tier", "low_performance")
                        mlflow.set_tag("is_current_best", "false")

                    print(
                        f"   • Trial score: {trial_score:.4f} (best so far: {best_score:.4f})"
                    )
                else:
                    print(f"   • Trial failed - no valid metrics")

                return trial_score

        # Create Optuna study with MLflow integration
        study_name = f"{model_name}_optimization_{int(time.time())}"

        # Get trial count from config or default
        n_trials = self.config.get("optuna", {}).get("n_trials", 20)

        # Add pruning for early stopping of bad trials
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Start pruning after 5 trials
            n_warmup_steps=3,  # Wait 3 steps before pruning
            interval_steps=1,  # Check every step
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible results
            pruner=pruner,
        )

        print(f"\n🔍 Starting Optuna optimization for {model_name}")
        print(f"   • Study: {study_name}")
        print(f"   • Trials: {n_trials}")
        print(f"   • Objective: Maximize PCK@0.2")
        print(f"   • Pruning: Enabled (early stopping for poor trials)")

        # Run optimization with timeout protection
        try:
            timeout_minutes = self.config.get("optuna", {}).get("timeout_minutes", 60)
            study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60)
        except KeyboardInterrupt:
            print(
                f"\n⚠️ Optimization interrupted by user after {len(all_trial_results)} trials"
            )
        except Exception as e:
            print(f"\n❌ Optimization failed: {e}")
            logging.error(f"Optuna optimization failed: {e}")
            return {"error": f"Optimization failed: {e}"}

        # Log optimization summary
        with mlflow.start_run(run_name=f"{model_name}_optuna_summary"):
            # Add detailed description for summary
            summary_description = f"""
OPTUNA OPTIMIZATION SUMMARY - Complete Analysis

PURPOSE: Comprehensive analysis of the hyperparameter optimization process
DATA SCOPE: Aggregated statistics from {len(all_trial_results)} completed trials
BEST RESULT: Trial #{best_trial_result['trial_number'] if best_trial_result else 'None'} with score {best_score:.4f}

This run provides a bird's-eye view of the entire optimization process. It aggregates
results from all individual trials to show optimization trends, convergence patterns,
and overall success metrics. Use this to understand whether the optimization worked
and what the search space exploration revealed.

OPTIMIZATION ANALYSIS:
- Total trials completed: {len(all_trial_results)}
- Search algorithm: TPE (Tree-structured Parzen Estimator)
- Objective: Maximize PCK@0.2 (percentage of correct keypoints)
- Parameter space: {', '.join(search_space.keys()) if 'search_space' in locals() else 'Model-specific hyperparameters'}
- Best trial identification: Highest scoring configuration

SUMMARY STATISTICS:
- Best trial score: {best_score:.4f}
- Mean trial score: {np.mean([t['score'] for t in all_trial_results]):.4f} ± {np.std([t['score'] for t in all_trial_results]):.4f}
- Optimization convergence: {'Good' if best_score > np.mean([t['score'] for t in all_trial_results]) + np.std([t['score'] for t in all_trial_results]) else 'Poor'}

INTERPRETATION:
- Use this run to verify optimization success
- Compare best_trial_score across different models
- Check optimization_mean_score for search space quality
- Look at best_* parameters for optimal configuration
            """.strip()

            mlflow.set_tag("mlflow.note.content", summary_description)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("optimization_mode", "summary")
            mlflow.log_param(
                "data_scope", f"statistics_from_{len(all_trial_results)}_trials"
            )
            mlflow.log_param("purpose", "optimization_analysis")

            mlflow.log_param("total_trials", len(all_trial_results))
            mlflow.log_param(
                "best_trial_number",
                best_trial_result["trial_number"] if best_trial_result else None,
            )
            mlflow.log_metric("best_trial_score", best_score)

            # Log best hyperparameters
            if best_trial_result:
                for param_name, param_value in best_trial_result["config"].items():
                    mlflow.log_param(f"best_{param_name}", param_value)

            # Log optimization statistics
            trial_scores = [t["score"] for t in all_trial_results]
            if trial_scores:
                mlflow.log_metric("optimization_mean_score", np.mean(trial_scores))
                mlflow.log_metric("optimization_std_score", np.std(trial_scores))
                mlflow.log_metric("optimization_min_score", np.min(trial_scores))
                mlflow.log_metric("optimization_max_score", np.max(trial_scores))

        # Final evaluation with best parameters on full dataset
        if best_trial_result:
            print(
                f"\n🏆 Best trial: {best_trial_result['trial_number']} (score: {best_score:.4f})"
            )
            print(
                f"   • Parameters: {self._create_param_summary(best_trial_result['config'])}"
            )

            # Run full evaluation with best config
            best_config = best_trial_result["config"]
            model_class = self.model_registry[model_name]
            model = model_class(device=self.device, **best_config)

            # Log final evaluation
            with mlflow.start_run(run_name=f"{model_name}_optuna_best_full_eval"):
                # Add detailed description for best full evaluation
                best_config_summary = self._create_param_summary(best_config)
                full_eval_description = f"""
BEST CONFIGURATION FULL EVALUATION - Production Results

PURPOSE: Complete performance evaluation using optimal hyperparameters
DATA SCOPE: Full dataset evaluation on {len(maneuvers)} maneuvers (production-scale)
OPTIMAL CONFIG: {best_config_summary} (from Trial #{best_trial_result['trial_number']})

This is the PRODUCTION-READY evaluation using the best hyperparameters discovered
during optimization. Unlike trials (which use small subsets), this run processes
your complete dataset to provide accurate, deployment-ready performance metrics.

EVALUATION DETAILS:
- Model: {model_name}
- Configuration source: Best performing trial from optimization
- Data coverage: Complete dataset ({len(maneuvers)} maneuvers)
- Evaluation type: Comprehensive (all metrics computed)
- Expected runtime: 10-30 minutes depending on dataset size

CONFIGURATION USED:
{chr(10).join([f"- {k}: {v}" for k, v in best_config.items()])}

PERFORMANCE INTERPRETATION:
- pose_pck_error_mean: Lower is better (target: <0.3 for good performance)
- perf_fps_mean: Higher is better (target: >15 FPS for real-time)
- pose_detection_f1_mean: Higher is better (target: >0.7 for reliable detection)
- Memory metrics: Lower is better for deployment efficiency

USAGE RECOMMENDATIONS:
✅ Use these metrics for production deployment decisions
✅ Compare against other models for backbone selection  
✅ Cite these results in research papers or reports
✅ Use this configuration for actual surfing analysis system

⚠️  Do not use trial results for production - they are exploration only
                """.strip()

                mlflow.set_tag("mlflow.note.content", full_eval_description)

                mlflow.log_param("model_name", model_name)
                mlflow.log_param("optimization_mode", "best_full_evaluation")
                mlflow.log_param(
                    "data_scope", f"complete_dataset_{len(maneuvers)}_maneuvers"
                )
                mlflow.log_param("purpose", "production_evaluation")
                mlflow.log_param("source_trial", best_trial_result["trial_number"])
                mlflow.log_param("config_summary", best_config_summary)

                # Log best hyperparameters
                for param_name, param_value in best_config.items():
                    mlflow.log_param(param_name, param_value)

                # Run full evaluation
                result = self._evaluate_single_model_internal(
                    model, model_name, maneuvers, log_to_mlflow=True
                )

                # Generate sample visualizations for best configuration
                if (
                    self.config.get("output", {})
                    .get("visualization", {})
                    .get("enabled", False)
                ):
                    self._create_sample_visualizations(
                        model, f"{model_name}_best", maneuvers[:3]
                    )

                return result

        else:
            print(f"\n❌ No successful trials for {model_name}")
            return {"error": "No successful trials"}

    def _sample_hyperparameters(self, trial, model_name: str) -> Dict:
        """Sample hyperparameters for optimization"""
        # Load model configuration to get search space
        model_config_path = f"configs/model_configs/{model_name}.yaml"
        if not os.path.exists(model_config_path):
            logging.warning(
                f"No model config found for {model_name}, using default parameters"
            )
            return {}

        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        # Check if optuna_search_space is defined
        if "optuna_search_space" not in model_config:
            logging.warning(f"No optuna_search_space defined in {model_config_path}")
            return {}

        search_space = model_config["optuna_search_space"]
        sampled_params = {}

        # Sample each parameter based on its type
        for param_name, param_config in search_space.items():
            param_type = param_config.get("type")

            if param_type == "uniform":
                low = param_config["low"]
                high = param_config["high"]
                sampled_params[param_name] = trial.suggest_float(param_name, low, high)

            elif param_type == "int":
                low = param_config["low"]
                high = param_config["high"]
                step = param_config.get("step", 1)
                sampled_params[param_name] = trial.suggest_int(
                    param_name, low, high, step=step
                )

            elif param_type == "categorical":
                choices = param_config["choices"]
                sampled_params[param_name] = trial.suggest_categorical(
                    param_name, choices
                )

            elif param_type == "loguniform":
                low = param_config["low"]
                high = param_config["high"]
                sampled_params[param_name] = trial.suggest_float(
                    param_name, low, high, log=True
                )

            else:
                logging.warning(
                    f"Unknown parameter type '{param_type}' for {param_name}"
                )

        logging.info(f"Sampled hyperparameters for {model_name}: {sampled_params}")
        return sampled_params

    def _create_param_summary(self, config: Dict) -> str:
        """Create a concise parameter summary for run names"""
        summary_parts = []

        # Prioritize most important parameters for the summary
        key_params = {
            "model_size": "size",
            "model_complexity": "comp",
            "confidence_threshold": "conf",
            "min_detection_confidence": "det_conf",
            "iou_threshold": "iou",
            "keypoint_threshold": "kp_thresh",
        }

        for param_name, short_name in key_params.items():
            if param_name in config:
                value = config[param_name]
                if isinstance(value, float):
                    summary_parts.append(f"{short_name}_{value:.2f}")
                else:
                    summary_parts.append(f"{short_name}_{value}")

        # Limit to 3 most important parameters to keep names readable
        return "_".join(summary_parts[:3]) if summary_parts else "default"

    def _evaluate_single_model_internal(
        self, model, model_name: str, maneuvers: List, log_to_mlflow: bool = False
    ) -> Dict:
        """Internal method to evaluate a model without starting a new MLflow run"""
        # Evaluation metrics
        all_pose_metrics = []
        all_performance_metrics = []

        # Process maneuvers
        successful_maneuvers = 0
        failed_maneuvers = 0

        for i, maneuver in enumerate(maneuvers):
            try:
                maneuver_metrics = self._process_video_maneuver(
                    model, maneuver, model_name
                )

                if maneuver_metrics["pose"]:
                    all_pose_metrics.append(maneuver_metrics["pose"])
                    all_performance_metrics.append(maneuver_metrics["performance"])
                    successful_maneuvers += 1
                else:
                    failed_maneuvers += 1

                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(
                        f"   • Progress: {i + 1}/{len(maneuvers)} maneuvers processed"
                    )

            except Exception as e:
                failed_maneuvers += 1
                logging.warning(f"Failed to process maneuver {i}: {e}")

        # Aggregate results
        if all_pose_metrics and all_performance_metrics:
            aggregated_metrics = self._aggregate_metrics(
                all_pose_metrics, all_performance_metrics
            )

            if log_to_mlflow:
                # Log metrics to current MLflow run
                for metric_name, value in aggregated_metrics.items():
                    mlflow.log_metric(metric_name, value)

                mlflow.log_param("successful_maneuvers", successful_maneuvers)
                mlflow.log_param("failed_maneuvers", failed_maneuvers)
                mlflow.log_param("total_maneuvers", len(maneuvers))

            return aggregated_metrics
        else:
            return {"error": "No successful evaluations"}

    def _create_sample_visualizations(
        self, model, model_name: str, maneuvers: List, max_maneuvers: int = 3
    ):
        """Create sample visualization videos for model evaluation

        Args:
            model: Pose estimation model instance
            model_name: Name of the model
            maneuvers: List of maneuvers to visualize
            max_maneuvers: Maximum number of maneuvers to visualize
        """
        try:
            vis_config = self.config.get("output", {}).get("visualization", {})
            if not vis_config.get("enabled", False):
                return

            max_examples = vis_config.get("max_examples_per_model", 3)
            max_maneuvers = min(max_maneuvers, max_examples)

            if not maneuvers:
                logging.warning(
                    f"No maneuvers provided for visualization of {model_name}"
                )
                return

            # Create visualization directory - use configured path or fallback to results
            shared_storage_path = vis_config.get("shared_storage_path")
            if shared_storage_path:
                # Use shared storage location
                vis_base_dir = Path(shared_storage_path)
                logging.info(f"Using shared storage location: {vis_base_dir}")
            else:
                # Fall back to local results directory
                vis_base_dir = Path(
                    self.config.get("output", {}).get("results_dir", "./results")
                )
                logging.info(f"Using local results directory: {vis_base_dir}")

            shared_pose_dir = vis_base_dir

            # Create organized directory structure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = shared_pose_dir / "visualizations" / f"{timestamp}_{model_name}"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Create metadata file for this visualization session
            metadata = {
                "model_name": model_name,
                "timestamp": timestamp,
                "config_used": vis_config,
                "max_maneuvers": max_maneuvers,
                "maneuvers_processed": [],
            }

            logging.info(
                f"Creating {max_maneuvers} sample visualizations for {model_name}"
            )

            # Process each maneuver for visualization
            for maneuver_idx, maneuver in enumerate(maneuvers[:max_maneuvers]):
                try:
                    # Load existing prediction file instead of running inference again
                    logging.info(
                        f"  Processing maneuver {maneuver_idx + 1}/{max_maneuvers}: {maneuver.maneuver_type} ({maneuver.duration:.1f}s)"
                    )

                    # Try to load existing prediction file
                    pose_results = []
                    prediction_file_path = None

                    if self.prediction_handler:
                        try:
                            # Strip any suffix like "_best" from model name for prediction file lookup
                            base_model_name = model_name.replace("_best", "").replace(
                                "_optimized", ""
                            )

                            # Try new format first (with maneuver details)
                            prediction_file_path = self.prediction_handler.get_prediction_file_path_with_details(
                                base_model_name,
                                maneuver.maneuver_type,
                                maneuver.execution_score,
                                maneuver.file_path,
                            )

                            print(
                                f"    🔍 Looking for prediction file: {Path(prediction_file_path).name}"
                            )
                            print(f"    📂 Full path: {prediction_file_path}")
                            print(
                                f"    📁 Directory exists: {Path(prediction_file_path).parent.exists()}"
                            )
                            print(
                                f"    📄 File exists: {Path(prediction_file_path).exists()}"
                            )

                            # Check if new format exists
                            if Path(prediction_file_path).exists():
                                print(
                                    f"    ✅ Found prediction file in new format (with maneuver details)"
                                )
                            else:
                                # Fallback to old method for backward compatibility
                                fallback_path = (
                                    self.prediction_handler.get_prediction_file_path(
                                        base_model_name, maneuver.maneuver_id
                                    )
                                )

                                if Path(fallback_path).exists():
                                    prediction_file_path = fallback_path
                                    print(
                                        f"    ✅ Found prediction file in old format (model folder)"
                                    )
                                else:
                                    # Try very old format for backward compatibility
                                    base_path = Path(fallback_path).parent.parent
                                    old_filename = f"{base_model_name}_{maneuver.maneuver_id}_predictions.json"
                                    old_prediction_file_path = base_path / old_filename

                                    if old_prediction_file_path.exists():
                                        prediction_file_path = str(
                                            old_prediction_file_path
                                        )
                                        print(
                                            f"    ✅ Found prediction file in very old format (root folder)"
                                        )
                                        print(
                                            f"    📂 Very old format path: {prediction_file_path}"
                                        )
                                    else:
                                        print(
                                            f"    ❌ No prediction file found in any format"
                                        )

                            if Path(prediction_file_path).exists():
                                logging.info(
                                    f"    📂 Loading existing prediction file: {Path(prediction_file_path).name}"
                                )
                                maneuver_prediction = (
                                    self.prediction_handler.load_prediction_file(
                                        prediction_file_path
                                    )
                                )

                                # Convert standardized format back to pose_results for visualization
                                for frame_pred in maneuver_prediction.frames:
                                    if frame_pred.persons:
                                        # Extract keypoints, scores, and bboxes per person
                                        keypoints_list = []
                                        scores_list = []
                                        bbox_list = []

                                        for person in frame_pred.persons:
                                            # Extract keypoints (x, y, z) or (x, y)
                                            person_keypoints = []
                                            person_scores = []

                                            for kp in person.keypoints:
                                                if kp.z is not None:
                                                    person_keypoints.append(
                                                        [kp.x, kp.y, kp.z]
                                                    )
                                                else:
                                                    person_keypoints.append(
                                                        [kp.x, kp.y]
                                                    )
                                                person_scores.append(kp.confidence)

                                            keypoints_list.append(person_keypoints)
                                            scores_list.append(person_scores)
                                            bbox_list.append(person.bbox)

                                        pose_result = {
                                            "keypoints": np.array(keypoints_list),
                                            "scores": np.array(scores_list),
                                            "bbox": np.array(bbox_list),
                                            "num_persons": len(frame_pred.persons),
                                            "metadata": {
                                                "model": model_name,
                                                "inference_time": frame_pred.inference_time,
                                                "frame_id": frame_pred.frame_id,
                                                "timestamp": frame_pred.timestamp,
                                            },
                                        }
                                    else:
                                        # No persons detected
                                        pose_result = {
                                            "keypoints": np.array([]).reshape(0, 17, 2),
                                            "scores": np.array([]).reshape(0, 17),
                                            "bbox": np.array([]).reshape(0, 4),
                                            "num_persons": 0,
                                            "metadata": {
                                                "model": model_name,
                                                "inference_time": frame_pred.inference_time,
                                                "frame_id": frame_pred.frame_id,
                                                "timestamp": frame_pred.timestamp,
                                            },
                                        }

                                    pose_results.append(pose_result)

                                logging.info(
                                    f"    ✅ Loaded {len(pose_results)} frames from prediction file"
                                )
                            else:
                                logging.info(
                                    f"    ⚠️ No prediction file found, running inference..."
                                )
                                # Fall back to running inference
                                frames = self.data_loader.load_video_frames(maneuver)
                                for frame_idx, frame in enumerate(frames):
                                    if frame_idx % 10 == 0:
                                        logging.info(
                                            f"    Frame {frame_idx + 1}/{len(frames)}"
                                        )
                                    pose_result = model.predict(frame)
                                    pose_results.append(pose_result)

                        except Exception as e:
                            logging.warning(
                                f"    ⚠️ Failed to load prediction file: {e}, running inference..."
                            )
                            # Fall back to running inference
                            frames = self.data_loader.load_video_frames(maneuver)
                            for frame_idx, frame in enumerate(frames):
                                if frame_idx % 10 == 0:
                                    logging.info(
                                        f"    Frame {frame_idx + 1}/{len(frames)}"
                                    )
                                pose_result = model.predict(frame)
                                pose_results.append(pose_result)
                    else:
                        # No prediction handler, run inference
                        logging.info(
                            f"    🔄 Running inference (no prediction handler)..."
                        )
                        frames = self.data_loader.load_video_frames(maneuver)
                        for frame_idx, frame in enumerate(frames):
                            if frame_idx % 10 == 0:
                                logging.info(f"    Frame {frame_idx + 1}/{len(frames)}")
                            pose_result = model.predict(frame)
                            pose_results.append(pose_result)

                    # Create visualization video for this maneuver
                    # Format execution score as 2-digit integer (e.g., 05, 10)
                    execution_score_str = f"{int(maneuver.execution_score):02d}"
                    output_path = (
                        vis_dir
                        / f"maneuver_{maneuver.maneuver_type}_{execution_score_str}_{Path(maneuver.file_path).stem}_poses.mp4"
                    )

                    success = self.video_visualizer.create_pose_visualization_video(
                        video_path=maneuver.file_path,
                        pose_results=pose_results,
                        output_path=str(output_path),
                        model_name=model_name,
                        kpt_thr=0.3,
                        bbox_thr=0.3,
                        max_persons=2,
                        maneuver_start_frame=maneuver.start_frame,
                        maneuver_end_frame=maneuver.end_frame,
                    )

                    if success:
                        # Log to MLflow
                        self.mlflow_manager.log_video_sample(str(output_path))
                        logging.info(
                            f"    ✅ Created visualization: {output_path.name}"
                        )

                        # Track successful maneuver in metadata
                        metadata["maneuvers_processed"].append(
                            {
                                "maneuver_index": maneuver_idx + 1,
                                "maneuver_type": maneuver.maneuver_type,
                                "maneuver_duration": maneuver.duration,
                                "source_file": Path(maneuver.file_path).name,
                                "output_video": output_path.name,
                                "prediction_file": (
                                    Path(prediction_file_path).name
                                    if prediction_file_path
                                    else None
                                ),
                                "frames_processed": len(pose_results),
                                "status": "success",
                                "used_prediction_file": prediction_file_path is not None
                                and Path(prediction_file_path).exists(),
                            }
                        )
                    else:
                        logging.warning(
                            f"    ❌ Failed to create visualization for maneuver {maneuver_idx + 1}"
                        )

                        # Track failed maneuver in metadata
                        metadata["maneuvers_processed"].append(
                            {
                                "maneuver_index": maneuver_idx + 1,
                                "maneuver_type": maneuver.maneuver_type,
                                "maneuver_duration": maneuver.duration,
                                "source_file": Path(maneuver.file_path).name,
                                "output_video": None,
                                "prediction_file": None,
                                "frames_processed": 0,
                                "status": "failed",
                            }
                        )

                except Exception as e:
                    logging.error(
                        f"    ❌ Error processing maneuver {maneuver_idx + 1}: {e}"
                    )

                    # Track error in metadata
                    metadata["maneuvers_processed"].append(
                        {
                            "maneuver_index": maneuver_idx + 1,
                            "maneuver_type": getattr(
                                maneuver, "maneuver_type", "unknown"
                            ),
                            "maneuver_duration": getattr(maneuver, "duration", 0.0),
                            "source_file": (
                                Path(maneuver.file_path).name
                                if hasattr(maneuver, "file_path")
                                else "unknown"
                            ),
                            "output_video": None,
                            "prediction_file": None,
                            "frames_processed": 0,
                            "status": "error",
                            "error_message": str(e),
                        }
                    )
                    continue

            # Save metadata file
            metadata_path = vis_dir / "visualization_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Summary
            successful_maneuvers = len(
                [m for m in metadata["maneuvers_processed"] if m["status"] == "success"]
            )
            total_maneuvers = len(metadata["maneuvers_processed"])

            logging.info(f"Completed visualization generation for {model_name}")
            logging.info(f"  📁 Saved to: {vis_dir}")
            logging.info(
                f"  📊 Success rate: {successful_maneuvers}/{total_maneuvers} maneuvers"
            )
            logging.info(
                f"  🔄 Synchronized to shared storage for cross-project access"
            )

        except Exception as e:
            logging.error(
                f"Failed to create sample visualizations for {model_name}: {e}"
            )

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
        "--video-format",
        choices=["h264", "ffv1"],
        help="Override video format for input (h264 or ffv1). If not specified, uses config file setting.",
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
    evaluator = PoseEvaluator(args.config, video_format_override=args.video_format)

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
