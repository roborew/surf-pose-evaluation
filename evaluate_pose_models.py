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

# HRNet functionality is available through MMPose framework


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

        # BlazePose (if available)
        if BlazePoseWrapper is not None:
            registry["blazepose"] = BlazePoseWrapper

        # MMPose (if available)
        if MMPoseWrapper is not None:
            registry["mmpose"] = MMPoseWrapper

        # YOLOv8-Pose (if available)
        if YOLOv8Wrapper is not None:
            registry["yolov8_pose"] = YOLOv8Wrapper

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
            # Log model parameters (safely handle long strings)
            safe_config = self._sanitize_mlflow_params(model_config)

            # Add detailed description for standard evaluation
            config_preview = list(safe_config.items())[:10]
            config_text = chr(10).join([f"- {k}: {v}" for k, v in config_preview])
            if len(safe_config) > 10:
                config_text += "\n..."

            standard_eval_description = f"""
STANDARD MODEL EVALUATION - Baseline Performance

PURPOSE: Comprehensive evaluation using default/configured hyperparameters
DATA SCOPE: Full dataset evaluation on {len(clips)} clips
CONFIGURATION: Default parameters from model config file

This is a standard evaluation run using the model's default or pre-configured
hyperparameters without optimization. Use this to establish baseline performance
or when you have known-good parameters that don't require optimization.

EVALUATION DETAILS:
- Model: {model_name}
- Configuration source: Model config file or defaults
- Data coverage: Complete dataset ({len(clips)} clips)
- Evaluation type: Comprehensive (all metrics computed)
- Hyperparameter optimization: Not used

CURRENT CONFIGURATION:
{config_text}

PERFORMANCE INTERPRETATION:
- pose_pck_error_mean: Lower is better (target: <0.3 for good performance)
- perf_fps_mean: Higher is better (target: >15 FPS for real-time)
- pose_detection_f1_mean: Higher is better (target: >0.7 for reliable detection)
- Memory metrics: Lower is better for deployment efficiency

USAGE RECOMMENDATIONS:
✅ Use as baseline for comparing optimized configurations
✅ Use when you have known-good hyperparameters
✅ Compare across different model architectures
✅ Good starting point before running optimization

💡 Consider running with --use-optuna to find better hyperparameters
            """.strip()

            mlflow.set_tag("mlflow.note.content", standard_eval_description)
            mlflow.log_params(safe_config)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("device", self.device)
            mlflow.log_param("num_clips", len(clips))
            mlflow.log_param("optimization_mode", "standard_evaluation")
            mlflow.log_param("data_scope", f"complete_dataset_{len(clips)}_clips")
            mlflow.log_param("purpose", "baseline_evaluation")

            print(f"\n🔄 Starting evaluation of {model_name}")
            print(f"   • Device: {self.device}")
            print(f"   • Clips to process: {len(clips)}")
            print(
                f"   • Model complexity: {safe_config.get('model_complexity', 'N/A')}"
            )
            print(f"   • MLflow run: {mlflow.active_run().info.run_id[:8]}...")
            print("   " + "=" * 50)

            # Evaluation metrics
            all_pose_metrics = []
            all_performance_metrics = []

            # Process clips with enhanced progress tracking
            print(f"\n📹 Processing video clips...")
            successful_clips = 0
            failed_clips = 0

            for i, clip in enumerate(
                tqdm(
                    clips,
                    desc=f"🎯 {model_name}",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
            ):
                try:
                    # Process video
                    print(
                        f"   Processing clip {i+1}/{len(clips)}: {getattr(clip, 'clip_path', 'Unknown')}",
                        end="",
                        flush=True,
                    )

                    clip_metrics = self._process_video_clip(model, clip)
                    all_pose_metrics.append(clip_metrics["pose"])
                    all_performance_metrics.append(clip_metrics["performance"])

                    successful_clips += 1
                    print(f" ✅ ({clip_metrics['performance']['fps']:.1f} FPS)")

                except Exception as e:
                    failed_clips += 1
                    print(f" ❌ Error: {str(e)[:50]}...")
                    logging.error(
                        f"Error processing clip {getattr(clip, 'clip_path', 'Unknown')}: {e}"
                    )
                    continue

            # Processing summary
            print(f"\n📊 Processing Summary for {model_name}:")
            print(
                f"   • Successful clips: {successful_clips}/{len(clips)} ({100*successful_clips/len(clips):.1f}%)"
            )
            if failed_clips > 0:
                print(f"   • Failed clips: {failed_clips}")
            if all_performance_metrics:
                avg_fps = np.mean([m["fps"] for m in all_performance_metrics])
                print(f"   • Average FPS: {avg_fps:.1f}")
            print("   " + "=" * 50)

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
DATA SCOPE: Quick evaluation on {min(20, len(clips))} clips (subset for speed)
PARAMETERS TESTED: {param_summary}

This trial tests a specific combination of hyperparameters sampled by Optuna's TPE algorithm.
The goal is to quickly evaluate whether this parameter combination shows promise before
running expensive full evaluation. Results guide the optimization toward better parameters.

TRIAL DETAILS:
- Model: {model_name}
- Parameter combination: {config}
- Optimization objective: Maximize PCK@0.2 (pose accuracy)
- Data used: Subset of clips for fast iteration
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
                mlflow.log_param("data_scope", f"subset_{min(20, len(clips))}_clips")
                mlflow.log_param("purpose", "hyperparameter_exploration")

                # Log sampled hyperparameters
                for param_name, param_value in config.items():
                    mlflow.log_param(f"hp_{param_name}", param_value)

                # Initialize model with sampled parameters
                model_class = self.model_registry[model_name]
                model = model_class(device=self.device, **config)

                # Quick evaluation on subset for optimization
                subset_clips = clips[: min(20, len(clips))]
                trial_metrics = []
                trial_pose_metrics = []
                trial_perf_metrics = []

                print(f"\n🔄 Trial {trial.number:03d}: {param_summary}")

                for i, clip in enumerate(subset_clips):
                    try:
                        clip_metrics = self._process_video_clip(model, clip)
                        if clip_metrics["pose"]:
                            pck_score = clip_metrics["pose"].get("pck_0_2", 0)
                            trial_metrics.append(pck_score)
                            trial_pose_metrics.append(clip_metrics["pose"])
                            trial_perf_metrics.append(clip_metrics["performance"])

                        # Progress indicator for trials
                        if (i + 1) % 5 == 0:
                            print(
                                f"   • Processed {i + 1}/{len(subset_clips)} clips..."
                            )

                    except Exception as e:
                        logging.warning(
                            f"Failed to process clip in trial {trial.number}: {e}"
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
                    mlflow.log_metric("num_clips_processed", len(trial_metrics))

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
DATA SCOPE: Full dataset evaluation on {len(clips)} clips (production-scale)
OPTIMAL CONFIG: {best_config_summary} (from Trial #{best_trial_result['trial_number']})

This is the PRODUCTION-READY evaluation using the best hyperparameters discovered
during optimization. Unlike trials (which use small subsets), this run processes
your complete dataset to provide accurate, deployment-ready performance metrics.

EVALUATION DETAILS:
- Model: {model_name}
- Configuration source: Best performing trial from optimization
- Data coverage: Complete dataset ({len(clips)} clips)
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
                mlflow.log_param("data_scope", f"complete_dataset_{len(clips)}_clips")
                mlflow.log_param("purpose", "production_evaluation")
                mlflow.log_param("source_trial", best_trial_result["trial_number"])
                mlflow.log_param("config_summary", best_config_summary)

                # Log best hyperparameters
                for param_name, param_value in best_config.items():
                    mlflow.log_param(param_name, param_value)

                # Run full evaluation
                return self._evaluate_single_model_internal(
                    model, model_name, clips, log_to_mlflow=True
                )

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
        self, model, model_name: str, clips: List, log_to_mlflow: bool = False
    ) -> Dict:
        """Internal method to evaluate a model without starting a new MLflow run"""
        # Evaluation metrics
        all_pose_metrics = []
        all_performance_metrics = []

        # Process clips
        successful_clips = 0
        failed_clips = 0

        for i, clip in enumerate(clips):
            try:
                clip_metrics = self._process_video_clip(model, clip)

                if clip_metrics["pose"]:
                    all_pose_metrics.append(clip_metrics["pose"])
                    all_performance_metrics.append(clip_metrics["performance"])
                    successful_clips += 1
                else:
                    failed_clips += 1

                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   • Progress: {i + 1}/{len(clips)} clips processed")

            except Exception as e:
                failed_clips += 1
                logging.warning(f"Failed to process clip {i}: {e}")

        # Aggregate results
        if all_pose_metrics and all_performance_metrics:
            aggregated_metrics = self._aggregate_metrics(
                all_pose_metrics, all_performance_metrics
            )

            if log_to_mlflow:
                # Log metrics to current MLflow run
                for metric_name, value in aggregated_metrics.items():
                    mlflow.log_metric(metric_name, value)

                mlflow.log_param("successful_clips", successful_clips)
                mlflow.log_param("failed_clips", failed_clips)
                mlflow.log_param("total_clips", len(clips))

            return aggregated_metrics
        else:
            return {"error": "No successful evaluations"}

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
