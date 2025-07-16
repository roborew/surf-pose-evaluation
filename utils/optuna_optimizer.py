#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for Pose Models
Handles hyperparameter search and optimization logic
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
    """Handles Optuna hyperparameter optimization for pose models"""

    def __init__(self, config: Dict, run_manager=None):
        """Initialize Optuna optimizer"""
        self.config = config
        self.run_manager = run_manager
        self.evaluator = PoseEvaluator(config)

    def optimize_model(self, model_name: str, maneuvers: List) -> Dict:
        """Run Optuna optimization for a single model"""
        logging.info(f"Starting Optuna optimization for {model_name}")

        # Set MLflow experiment name from config
        experiment_name = self.config.get("mlflow", {}).get("experiment_name", "surf_pose_optuna")
        mlflow.set_experiment(experiment_name)
        logging.info(f"MLflow experiment set to: {experiment_name}")

        # Storage for trial results
        all_trial_results = []
        best_trial_result = None
        best_score = -float("inf")

        def objective(trial):
            nonlocal best_trial_result, best_score

            # Sample hyperparameters
            config = self._sample_hyperparameters(trial, model_name)
            param_summary = self._create_param_summary(config)
            run_name = f"{model_name}_optuna_trial_{trial.number:03d}_{param_summary}"

            # Start MLflow run for this trial
            with mlflow.start_run(run_name=run_name):
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

                # Quick evaluation on subset
                subset_maneuvers = maneuvers[: min(20, len(maneuvers))]
                trial_metrics = []

                print(f"\n🔄 Trial {trial.number:03d}: {param_summary}")

                for i, maneuver in enumerate(subset_maneuvers):
                    try:
                        maneuver_metrics = self.evaluator._process_video_maneuver(
                            model, maneuver, model_name
                        )
                        if maneuver_metrics["pose"]:
                            pck_score = maneuver_metrics["pose"].get("pck_0_2", 0)
                            trial_metrics.append(pck_score)

                        if (i + 1) % 5 == 0:
                            print(
                                f"   • Processed {i + 1}/{len(subset_maneuvers)} maneuvers..."
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
                mlflow.log_metric("num_maneuvers_processed", len(trial_metrics))

                # Track best trial
                if trial_score > best_score:
                    best_score = trial_score
                    best_trial_result = {
                        "trial_number": trial.number,
                        "config": config,
                        "score": trial_score,
                        "run_name": run_name,
                    }

                print(
                    f"   • Trial score: {trial_score:.4f} (best so far: {best_score:.4f})"
                )
                return trial_score

        # Create and run study
        study_name = f"{model_name}_optimization_{int(time.time())}"
        n_trials = self.config.get("optuna", {}).get("n_trials", 20)

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )

        print(f"\n🔍 Starting Optuna optimization for {model_name}")
        print(f"   • Study: {study_name}")
        print(f"   • Trials: {n_trials}")

        try:
            timeout_minutes = self.config.get("optuna", {}).get("timeout_minutes", 60)
            study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60)
        except (KeyboardInterrupt, Exception) as e:
            logging.warning(f"Optimization interrupted: {e}")

        return self._finalize_optimization(
            model_name, maneuvers, best_trial_result, best_score
        )

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
        mlflow.log_param("optimization_mode", "optuna_trial")

        # Log sampled hyperparameters
        for param_name, param_value in config.items():
            mlflow.log_param(f"hp_{param_name}", param_value)

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
            if "error" not in result and hasattr(result, "get"):
                # Extract parameters from result (this would need to be implemented based on result structure)
                # For now, this is a placeholder
                best_params[model_name] = {
                    "placeholder": "implement_parameter_extraction"
                }

        try:
            best_params_file = self.run_manager.best_params_dir / "best_parameters.yaml"
            with open(best_params_file, "w") as f:
                yaml.dump(best_params, f, default_flow_style=False)

            logging.info(f"Best parameters saved to {best_params_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to save best parameters: {e}")
            return False
