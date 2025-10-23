#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for Pose Models
Handles hyperparameter search and optimization logic with intelligent early stopping
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json

import numpy as np
import mlflow
import optuna

from utils.pose_evaluator import PoseEvaluator


class OptunaPoseOptimizer:
    """Handles Optuna hyperparameter optimization for pose models with intelligent early stopping"""

    def __init__(self, config: Dict, run_manager=None):
        """Initialize Optuna optimizer"""
        self.config = config
        self.run_manager = run_manager
        self.evaluator = PoseEvaluator(config)
        self.evaluator.run_manager = run_manager  # Set run manager for visualizations

        # Early stopping configuration
        self.early_stopping_config = config.get("optuna", {}).get(
            "early_stopping",
            {
                "enabled": True,
                "patience": 10,  # Stop if no improvement for 10 trials
                "min_trials": 15,  # Minimum trials before early stopping
                "improvement_threshold": 0.001,  # Minimum improvement to continue
                "plateau_threshold": 0.95,  # Consider plateaued if within 95% of best
            },
        )

        # Consensus validation setup
        self.use_consensus = config.get("optuna_validation", {}).get(
            "use_consensus", False
        )
        self.consensus_manager = None
        self.consensus_gt = None  # Cache for consensus ground truth

        if self.use_consensus and run_manager:
            logging.info("Consensus-based validation enabled for Optuna")
            try:
                from utils.consensus_manager import ConsensusManager
                from utils.quality_filter import AdaptiveQualityFilter

                # Initialize quality filter
                quality_filter = AdaptiveQualityFilter(
                    w_confidence=0.4, w_stability=0.4, w_completeness=0.2
                )

                # Initialize consensus manager
                consensus_cache_dir = run_manager.run_dir / "consensus_cache"
                self.consensus_manager = ConsensusManager(
                    consensus_models=["yolov8", "pytorch_pose", "mmpose"],
                    quality_filter=quality_filter,
                    cache_dir=consensus_cache_dir,
                )
                logging.info("ConsensusManager initialized successfully")
            except Exception as e:
                logging.warning(
                    f"Failed to initialize consensus validation: {e}. "
                    "Falling back to detection metrics."
                )
                self.use_consensus = False

    def optimize_model(
        self, model_name: str, maneuvers: List, memory_profiler=None
    ) -> Dict:
        """Run Optuna optimization for a single model with intelligent early stopping"""
        logging.info(f"Starting Optuna optimization for {model_name}")

        # Set MLflow experiment name from config
        experiment_name = self.config.get("mlflow", {}).get(
            "experiment_name", "surf_pose_optuna"
        )
        mlflow.set_experiment(experiment_name)
        logging.info(f"MLflow experiment set to: {experiment_name}")

        # Storage for trial results
        all_trial_results = []
        best_trial_result = None
        best_score = -float("inf")
        trials_since_improvement = 0
        start_time = time.time()

        def objective(trial):
            nonlocal best_trial_result, best_score, trials_since_improvement

            # Sample hyperparameters
            config = self._sample_hyperparameters(trial, model_name)
            param_summary = self._create_param_summary(config)
            run_name = f"{model_name}_optuna_trial_{trial.number:03d}_{param_summary}"

            # Start MLflow run for this trial
            with mlflow.start_run(run_name=run_name):
                # Signal memory profiler that MLflow run started
                if memory_profiler:
                    memory_profiler.on_mlflow_run_start()

                # Log trial information
                self._log_trial_info(trial, model_name, config, maneuvers)

                # Initialize model with sampled parameters
                try:
                    model_class = self.evaluator.model_registry[model_name]
                    model = model_class(device=self.evaluator.device, **config)
                except Exception as e:
                    logging.warning(f"Failed to initialize {model_name}: {e}")
                    mlflow.log_metric("optuna_trial_score", 0.0)
                    mlflow.set_tag("trial_status", "initialization_failed")
                    return 0.0

                # Use pre-selected maneuvers (already optimized subset)
                trial_metrics = []

                print(f"\n🔄 Trial {trial.number:03d}: {param_summary}")
                print(f"   • Using {len(maneuvers)} pre-selected maneuvers")

                # Use consensus-based validation if available
                if self.use_consensus and self.consensus_manager:
                    # Load consensus GT once (cached after first load)
                    if self.consensus_gt is None:
                        logging.info(
                            f"Loading consensus GT for {model_name} (Optuna phase)"
                        )
                        self.consensus_gt = (
                            self.consensus_manager.generate_consensus_gt(
                                maneuvers=maneuvers,
                                target_model=model_name,
                                phase="optuna",
                            )
                        )
                        mlflow.log_param("validation_method", "consensus_based")
                        mlflow.log_param(
                            "consensus_models",
                            ",".join(
                                self.consensus_manager.get_consensus_models_for_target(
                                    model_name
                                )
                            ),
                        )

                    # Calculate PCK against consensus for each maneuver
                    from metrics.pose_metrics import PoseMetrics

                    metrics_calc = PoseMetrics()

                    for i, maneuver in enumerate(maneuvers):
                        try:
                            # Run inference on this maneuver
                            predictions = []
                            cap = __import__("cv2").VideoCapture(
                                str(maneuver.file_path)
                            )

                            if maneuver.start_frame > 0:
                                cap.set(
                                    __import__("cv2").CAP_PROP_POS_FRAMES,
                                    maneuver.start_frame,
                                )

                            frame_idx = maneuver.start_frame
                            while cap.isOpened() and frame_idx < maneuver.end_frame:
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                pred = model.predict(frame)
                                predictions.append(pred)
                                frame_idx += 1

                            cap.release()

                            # Calculate PCK against consensus
                            pck_result = metrics_calc.calculate_pck_with_consensus_gt(
                                predictions=predictions,
                                consensus_gt=self.consensus_gt,
                                maneuver_id=maneuver.maneuver_id,
                                threshold=0.2,
                            )

                            if pck_result["pck_0_2"] > 0:
                                trial_metrics.append(pck_result["pck_0_2"])

                            if (i + 1) % 5 == 0:
                                avg_so_far = (
                                    np.mean(trial_metrics) if trial_metrics else 0.0
                                )
                                print(
                                    f"   • Processed {i + 1}/{len(maneuvers)} maneuvers... "
                                    f"(avg PCK: {avg_so_far:.3f})"
                                )

                        except Exception as e:
                            logging.warning(
                                f"Failed to process maneuver in trial {trial.number}: {e}"
                            )
                            continue
                else:
                    # Fallback to detection metrics (old behavior)
                    mlflow.log_param("validation_method", "detection_metrics")

                    for i, maneuver in enumerate(maneuvers):
                        try:
                            maneuver_metrics = self.evaluator._process_video_maneuver(
                                model, maneuver, model_name
                            )
                            if maneuver_metrics["pose"]:
                                pck_score = maneuver_metrics["pose"].get("pck_0_2", 0)
                                trial_metrics.append(pck_score)

                            if (i + 1) % 5 == 0:
                                print(
                                    f"   • Processed {i + 1}/{len(maneuvers)} maneuvers..."
                                )

                        except Exception as e:
                            logging.warning(
                                f"Failed to process maneuver in trial {trial.number}: {e}"
                            )
                            continue

                # Calculate trial score
                trial_score = np.mean(trial_metrics) if trial_metrics else 0

                # Log results
                mlflow.log_metric("optuna_trial_score", trial_score)
                mlflow.log_metric("pck_0_2", trial_score)  # Explicit PCK logging
                mlflow.log_metric("num_maneuvers_processed", len(trial_metrics))
                mlflow.log_param("trial_number", trial.number)

                # Check for improvement
                improvement = trial_score - best_score
                if improvement > self.early_stopping_config["improvement_threshold"]:
                    best_score = trial_score
                    best_trial_result = {
                        "trial_number": trial.number,
                        "config": config,
                        "score": trial_score,
                        "run_name": run_name,
                    }
                    trials_since_improvement = 0
                    print(
                        f"   ✅ New best score: {trial_score:.4f} (improvement: {improvement:.4f})"
                    )
                else:
                    trials_since_improvement += 1
                    print(
                        f"   • Trial score: {trial_score:.4f} (best: {best_score:.4f}, no improvement: {trials_since_improvement})"
                    )

                # Early stopping check
                if self._should_stop_early(
                    trial.number, trials_since_improvement, trial_score, best_score
                ):
                    print(f"   🛑 Early stopping triggered after {trial.number} trials")
                    raise optuna.exceptions.OptunaError("Early stopping triggered")

                # Signal memory profiler that MLflow run is ending
                if memory_profiler:
                    memory_profiler.on_mlflow_run_end()

                return trial_score

        # Create and run study
        study_name = f"{model_name}_optimization_{int(time.time())}"
        n_trials = self.config.get("optuna", {}).get("n_trials", 20)
        max_timeout = self.config.get("optuna", {}).get("timeout_minutes", 300)

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )

        print(f"\n🔍 Starting Optuna optimization for {model_name}")
        print(f"   • Study: {study_name}")
        print(f"   • Max trials: {n_trials}")
        print(f"   • Max timeout: {max_timeout} minutes")
        print(f"   • Early stopping: {self.early_stopping_config['enabled']}")

        try:
            study.optimize(objective, n_trials=n_trials, timeout=max_timeout * 60)
        except optuna.exceptions.OptunaError as e:
            if "Early stopping triggered" in str(e):
                logging.info(f"Early stopping triggered for {model_name}")
            else:
                logging.warning(f"Optimization interrupted: {e}")
        except (KeyboardInterrupt, Exception) as e:
            logging.warning(f"Optimization interrupted: {e}")

        # Log optimization summary
        elapsed_time = time.time() - start_time
        completed_trials = len(study.trials)
        logging.info(
            f"✅ {model_name} optimization completed: {completed_trials} trials in {elapsed_time/60:.1f} minutes"
        )

        return self._finalize_optimization(
            model_name, maneuvers, best_trial_result, best_score
        )

    def _should_stop_early(
        self,
        trial_number: int,
        trials_since_improvement: int,
        current_score: float,
        best_score: float,
    ) -> bool:
        """Determine if optimization should stop early"""
        if not self.early_stopping_config["enabled"]:
            return False

        # Minimum trials before early stopping
        if trial_number < self.early_stopping_config["min_trials"]:
            return False

        # Patience-based stopping
        if trials_since_improvement >= self.early_stopping_config["patience"]:
            return True

        # Plateau-based stopping (if score is very close to best)
        if best_score > 0:
            score_ratio = current_score / best_score
            if score_ratio >= self.early_stopping_config["plateau_threshold"]:
                return True

        return False

    def _sample_hyperparameters(self, trial, model_name: str) -> Dict:
        """Sample hyperparameters for optimization"""
        model_config_path = f"configs/model_configs/{model_name}.yaml"
        if not Path(model_config_path).exists():
            logging.warning(f"No model config found for {model_name}")
            return {}

        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        if "optuna_search_space" not in model_config:
            logging.warning(f"No optuna_search_space defined in {model_config_path}")
            return {}

        search_space = model_config["optuna_search_space"]
        sampled_params = {}

        for param_name, param_config in search_space.items():
            param_type = param_config.get("type")

            if param_type == "uniform":
                sampled_params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_type == "int":
                sampled_params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1),
                )
            elif param_type == "categorical":
                sampled_params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_type == "loguniform":
                sampled_params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=True
                )

        return sampled_params

    def _create_param_summary(self, config: Dict) -> str:
        """Create concise parameter summary for run names"""
        key_params = {
            "model_size": "size",
            "confidence_threshold": "conf",
            "min_detection_confidence": "det_conf",
            "iou_threshold": "iou",
        }

        summary_parts = []
        for param_name, short_name in key_params.items():
            if param_name in config:
                value = config[param_name]
                if isinstance(value, float):
                    summary_parts.append(f"{short_name}_{value:.2f}")
                else:
                    summary_parts.append(f"{short_name}_{value}")

        return "_".join(summary_parts[:3]) if summary_parts else "default"

    def _log_trial_info(self, trial, model_name: str, config: Dict, maneuvers: List):
        """Log trial information to MLflow"""
        trial_description = f"""
OPTUNA TRIAL #{trial.number:03d} - Hyperparameter Exploration

PURPOSE: Testing specific hyperparameter combination to find optimal settings
DATA SCOPE: Quick evaluation on {min(20, len(maneuvers))} maneuvers (subset for speed)
PARAMETERS TESTED: {self._create_param_summary(config)}

This trial tests a specific combination of hyperparameters sampled by Optuna's TPE algorithm.
Results guide the optimization toward better parameters.

TRIAL DETAILS:
- Model: {model_name}
- Parameter combination: {config}
- Optimization objective: Maximize PCK@0.2 (pose accuracy)
- Data used: Subset of maneuvers for fast iteration
        """.strip()

        mlflow.set_tag("mlflow.note.content", trial_description)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("trial_number", trial.number)

        # Log all hyperparameters
        for param_name, param_value in config.items():
            mlflow.log_param(param_name, param_value)

    def _finalize_optimization(
        self,
        model_name: str,
        maneuvers: List,
        best_trial_result: Dict,
        best_score: float,
    ) -> Dict:
        """Run final evaluation with best parameters"""
        if not best_trial_result:
            logging.error(f"No successful trials for {model_name}")
            return {"error": "No successful trials"}

        logging.info(
            f"Best trial: {best_trial_result['trial_number']} (score: {best_score:.4f})"
        )

        # Run full evaluation with best config
        best_config = best_trial_result["config"]
        model_class = self.evaluator.model_registry[model_name]
        model = model_class(device=self.evaluator.device, **best_config)

        # Final evaluation with full dataset
        with mlflow.start_run(run_name=f"{model_name}_optuna_best_full_eval"):
            self._log_best_evaluation_info(
                model_name, best_config, best_trial_result, maneuvers
            )

            # Log best hyperparameters
            for param_name, param_value in best_config.items():
                mlflow.log_param(param_name, param_value)

            # Run full evaluation
            result = self.evaluator._evaluate_model_internal(
                model, model_name, maneuvers
            )

            # Log metrics to MLflow
            if isinstance(result, dict) and "error" not in result:
                for metric_name, value in result.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric_name, value)

            # Include best parameters in the result for save_best_parameters to use
            if isinstance(result, dict):
                result["best_config"] = best_config
                result["best_trial_number"] = best_trial_result["trial_number"]
                result["best_score"] = best_score

            return result

    def _log_best_evaluation_info(
        self,
        model_name: str,
        best_config: Dict,
        best_trial_result: Dict,
        maneuvers: List,
    ):
        """Log information about the best configuration evaluation"""
        best_config_summary = self._create_param_summary(best_config)

        description = f"""
BEST CONFIGURATION FULL EVALUATION - Production Results

PURPOSE: Complete performance evaluation using optimal hyperparameters
DATA SCOPE: Full dataset evaluation on {len(maneuvers)} maneuvers (production-scale)
OPTIMAL CONFIG: {best_config_summary} (from Trial #{best_trial_result['trial_number']})

This is the PRODUCTION-READY evaluation using the best hyperparameters discovered
during optimization. This run processes the complete dataset to provide accurate,
deployment-ready performance metrics.

CONFIGURATION USED:
{chr(10).join([f"- {k}: {v}" for k, v in best_config.items()])}

USAGE RECOMMENDATIONS:
✅ Use these metrics for production deployment decisions
✅ Compare against other models for backbone selection
✅ Use this configuration for actual surfing analysis system
        """.strip()

        mlflow.set_tag("mlflow.note.content", description)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("optimization_mode", "best_full_evaluation")
        mlflow.log_param("source_trial", best_trial_result["trial_number"])
        mlflow.log_param("config_summary", best_config_summary)

    def save_best_parameters(self, model_results: Dict[str, Dict]) -> bool:
        """Save best parameters to file"""
        if not self.run_manager:
            logging.warning("No run manager available, cannot save best parameters")
            return False

        best_params = {}
        for model_name, result in model_results.items():
            if "error" not in result and isinstance(result, dict):
                # Extract actual best parameters from optimization result
                if "best_config" in result:
                    best_params[model_name] = result["best_config"]
                    logging.info(
                        f"Extracted best parameters for {model_name}: {result['best_config']}"
                    )
                else:
                    logging.warning(
                        f"No best_config found in result for {model_name}, optimization may have failed"
                    )
                    best_params[model_name] = {
                        "error": "optimization_failed_no_best_config"
                    }
            else:
                logging.warning(
                    f"Skipping {model_name} - result contains error or invalid format"
                )
                best_params[model_name] = {
                    "error": "optimization_failed_or_invalid_result"
                }

        if not any(
            isinstance(params, dict) and "error" not in params
            for params in best_params.values()
        ):
            logging.error("No valid optimized parameters found for any model!")
            return False

        try:
            best_params_file = self.run_manager.best_params_dir / "best_parameters.yaml"
            with open(best_params_file, "w") as f:
                yaml.dump(best_params, f, default_flow_style=False)

            logging.info(f"Saved best parameters to {best_params_file}")

            # Log summary of what was saved
            valid_models = [
                name
                for name, params in best_params.items()
                if isinstance(params, dict) and "error" not in params
            ]
            failed_models = [
                name
                for name, params in best_params.items()
                if isinstance(params, dict) and "error" in params
            ]

            logging.info(
                f"✅ Saved optimized parameters for {len(valid_models)} models: {valid_models}"
            )
            if failed_models:
                logging.warning(
                    f"⚠️ Failed to get parameters for {len(failed_models)} models: {failed_models}"
                )

            return True

        except Exception as e:
            logging.error(f"Failed to save best parameters: {e}")
            return False
