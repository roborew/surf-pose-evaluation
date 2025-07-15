#!/usr/bin/env python3
"""
Production Evaluation Runner with Organized Run Management
Runs Optuna optimization followed by comprehensive comparison
"""

import argparse
import logging
import sys
import subprocess
import json
import yaml
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.run_manager import RunManager


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run production evaluation with organized run management"
    )

    # Run organization arguments
    parser.add_argument(
        "--run-name",
        type=str,
        help="Custom name for this run (will be prefixed with timestamp)",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up old runs (keep only last 5)"
    )

    # Evaluation arguments
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mediapipe", "blazepose", "yolov8_pose", "pytorch_pose", "mmpose"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--max-clips", type=int, help="Maximum number of clips to process"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation_config_production_optuna.yaml",
        help="Base configuration file for Optuna phase",
    )
    parser.add_argument(
        "--comparison-config",
        type=str,
        default="configs/evaluation_config_production_comparison.yaml",
        help="Configuration file for comparison phase",
    )
    parser.add_argument(
        "--optuna-trials", type=int, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--skip-optuna", action="store_true", help="Skip Optuna optimization phase"
    )
    parser.add_argument(
        "--skip-comparison", action="store_true", help="Skip comparison phase"
    )
    parser.add_argument(
        "--optuna-only", action="store_true", help="Run only Optuna optimization phase"
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Run only model comparison phase (requires existing best parameters)",
    )

    return parser.parse_args()


def run_optuna_phase(run_manager: RunManager, args) -> str:
    """Run Optuna optimization phase"""
    logger = logging.getLogger(__name__)

    logger.info("🔍 Starting Optuna optimization phase...")

    # Create Optuna-specific config
    optuna_config = run_manager.create_config_for_phase(
        "optuna", args.config, args.max_clips
    )

    # Build command
    cmd = [
        "python",
        "evaluate_pose_models.py",
        "--config",
        optuna_config,
        "--use-optuna",
        "--models",
    ] + args.models

    if args.max_clips:
        cmd.extend(["--max-clips", str(args.max_clips)])

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        # Run with real-time output streaming
        result = subprocess.run(cmd, check=True, text=True)
        logger.info("✅ Optuna optimization completed successfully")
        return optuna_config
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Optuna optimization failed with exit code: {e.returncode}")
        raise


def extract_best_parameters(run_manager: RunManager, models: list):
    """Extract best parameters from Optuna MLflow runs"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Extracting best parameters from Optuna results")

    try:
        import mlflow
        import os

        # Connect to MLflow
        mlflow_dir = run_manager.mlflow_dir
        tracking_uri = str(mlflow_dir)
        logger.info(f"🔗 Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        # Check if MLflow directory exists and has content
        if not mlflow_dir.exists():
            logger.error(f"❌ MLflow directory does not exist: {mlflow_dir}")
            return False

        # List contents of MLflow directory for debugging
        try:
            mlflow_contents = list(mlflow_dir.iterdir())
            logger.info(
                f"📁 MLflow directory contents: {[item.name for item in mlflow_contents]}"
            )
        except Exception as e:
            logger.error(f"❌ Could not list MLflow directory contents: {e}")

        # Get the Optuna experiment with timestamp suffix
        experiment_name = f"surf_pose_production_optuna_{run_manager.timestamp}"
        logger.info(f"🔍 Looking for experiment: {experiment_name}")

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.error(f"❌ Optuna experiment not found: {experiment_name}")

            # Try to list available experiments for debugging
            try:
                all_experiments = mlflow.search_experiments()
                experiment_names = [exp.name for exp in all_experiments]
                logger.info(
                    f"📋 Available experiments ({len(experiment_names)}): {experiment_names}"
                )

                # Try to find any experiment with "optuna" in the name
                optuna_experiments = [
                    name for name in experiment_names if "optuna" in name.lower()
                ]
                if optuna_experiments:
                    logger.info(
                        f"🔍 Found Optuna-related experiments: {optuna_experiments}"
                    )

                    # Try using the most recent Optuna experiment
                    if len(optuna_experiments) == 1:
                        logger.info(
                            f"🔄 Attempting to use found Optuna experiment: {optuna_experiments[0]}"
                        )
                        experiment = mlflow.get_experiment_by_name(
                            optuna_experiments[0]
                        )
                        if experiment:
                            logger.info(
                                f"✅ Successfully found alternative experiment: {optuna_experiments[0]}"
                            )
                else:
                    logger.warning("⚠️ No Optuna-related experiments found")

            except Exception as e:
                logger.error(f"❌ Could not list experiments: {e}")

            if not experiment:
                return False

        logger.info(
            f"✅ Found experiment: {experiment.name} (ID: {experiment.experiment_id})"
        )

        best_params = {}

        # For each model, find the best_full_eval run
        for model in models:
            logger.info(f"🔍 Searching for best parameters for model: {model}")

            try:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName LIKE '{model}_optuna_best_full_eval'",
                    max_results=1,
                )

                if runs.empty:
                    logger.warning(f"⚠️ No best full eval run found for {model}")

                    # Try alternative search patterns
                    alternative_patterns = [
                        f"run_name LIKE '{model}_optuna_best%'",
                        f"tags.mlflow.runName LIKE '{model}_best%'",
                        f"params.model_name = '{model}' AND tags.mlflow.runName LIKE '%best%'",
                    ]

                    for pattern in alternative_patterns:
                        try:
                            runs = mlflow.search_runs(
                                experiment_ids=[experiment.experiment_id],
                                filter_string=pattern,
                                max_results=1,
                            )
                            if not runs.empty:
                                logger.info(
                                    f"✅ Found run using alternative pattern: {pattern}"
                                )
                                break
                        except Exception as e:
                            logger.debug(f"Alternative pattern failed: {pattern} - {e}")

                    if runs.empty:
                        logger.warning(f"⚠️ No runs found for {model} with any pattern")
                        continue

                run = runs.iloc[0]
                model_params = {}

                # Extract parameters (filter out non-hyperparameters)
                params_dict = run.params if hasattr(run, "params") else {}
                if hasattr(params_dict, "items"):
                    for param_name, param_value in params_dict.items():
                        if not param_name.startswith(
                            ("model_name", "optimization_mode", "data_scope", "purpose")
                        ):
                            model_params[param_name] = param_value
                else:
                    # If params is a Series, convert to dict
                    params_dict = (
                        run.params.to_dict() if hasattr(run.params, "to_dict") else {}
                    )
                    for param_name, param_value in params_dict.items():
                        if not param_name.startswith(
                            ("model_name", "optimization_mode", "data_scope", "purpose")
                        ):
                            model_params[param_name] = param_value

                best_params[model] = model_params
                logger.info(f"✅ Extracted best parameters for {model}: {model_params}")

            except Exception as e:
                logger.error(f"❌ Failed to extract parameters for {model}: {e}")
                continue

        if not best_params:
            logger.error("❌ No best parameters found for any model")
            return False

        # Save best parameters
        best_params_file = run_manager.best_params_dir / "best_parameters.yaml"
        with open(best_params_file, "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)

        logger.info(f"💾 Best parameters saved to {best_params_file}")
        logger.info(
            f"📊 Successfully extracted parameters for {len(best_params)} models: {list(best_params.keys())}"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Failed to extract best parameters: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def run_comparison_phase(run_manager: RunManager, args) -> str:
    """Run comprehensive comparison phase"""
    logger = logging.getLogger(__name__)

    logger.info("📊 Starting comprehensive comparison phase...")

    # Create comparison-specific config
    comparison_config = run_manager.create_config_for_phase(
        "comparison",
        args.comparison_config,
        args.max_clips,
    )

    # Build command
    cmd = [
        "python",
        "evaluate_pose_models.py",
        "--config",
        comparison_config,
        "--models",
    ] + args.models

    if args.max_clips:
        cmd.extend(["--max-clips", str(args.max_clips)])

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        # Run with real-time output streaming
        result = subprocess.run(cmd, check=True, text=True)
        logger.info("✅ Comprehensive comparison completed successfully")
        return comparison_config
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Model comparison failed with exit code: {e.returncode}")
        raise


def generate_summary_report(run_manager: RunManager, models: list):
    """Generate a summary report comparing the results"""
    logger = logging.getLogger(__name__)
    logger.info("📋 Generating summary report")

    try:
        import mlflow
        import pandas as pd
        from datetime import datetime

        mlflow.set_tracking_uri(str(run_manager.mlflow_dir))

        # Get comparison experiment
        experiment = mlflow.get_experiment_by_name("surf_pose_production_comparison")
        if not experiment:
            logger.error("❌ Comparison experiment not found")
            return False

        # Get all runs from comparison
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.pose_pck_error_mean ASC"],  # Best accuracy first
        )

        if runs.empty:
            logger.error("❌ No comparison runs found")
            return False

        # Create summary
        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "models_evaluated": models,
            "dataset_size": args.max_clips or "full",
            "results": [],
        }

        for _, run in runs.iterrows():
            model_result = {
                "model": run.get("params.model_name", "unknown"),
                "run_name": run.get("tags.mlflow.runName", "unknown"),
                "accuracy": {
                    "pck_error_mean": run.get("metrics.pose_pck_error_mean", None),
                    "detection_f1": run.get("metrics.pose_detection_f1_mean", None),
                },
                "performance": {
                    "fps_mean": run.get("metrics.perf_fps_mean", None),
                    "inference_time_ms": run.get(
                        "metrics.perf_avg_inference_time_mean", None
                    ),
                    "memory_usage_gb": run.get(
                        "metrics.perf_max_memory_usage_mean", None
                    ),
                },
            }
            summary["results"].append(model_result)

        # Save summary
        summary_file = run_manager.run_dir / "production_evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📊 Summary report saved to {summary_file}")

        # Print summary to console
        print("\n" + "=" * 60)
        print("🏆 PRODUCTION EVALUATION SUMMARY")
        print("=" * 60)

        for result in summary["results"]:
            print(f"\n📍 {result['model'].upper()}")
            print(
                f"   • Accuracy (PCK Error): {result['accuracy']['pck_error_mean']:.4f}"
            )
            print(f"   • Detection F1: {result['accuracy']['detection_f1']:.4f}")
            print(f"   • Speed (FPS): {result['performance']['fps_mean']:.2f}")
            print(
                f"   • Inference Time: {result['performance']['inference_time_ms']:.2f}ms"
            )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to generate summary report: {e}")
        return False


def main():
    """Main production evaluation workflow"""
    # Record start time for total execution tracking
    start_time = time.time()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Parse arguments
    args = parse_arguments()

    # Initialize run manager (always organized)
    run_manager = RunManager(run_name=args.run_name, max_clips=args.max_clips)

    # Print run information
    run_manager.print_run_info()

    # Optional cleanup
    if args.cleanup:
        run_manager.cleanup_old_runs()

    # Track results
    results = {"optuna_phase": None, "comparison_phase": None, "configs_used": []}

    success = True

    try:
        # Phase 1: Optuna Optimization
        if not args.skip_optuna and not args.comparison_only:
            optuna_config = run_optuna_phase(run_manager, args)
            results["optuna_phase"] = "completed"
            results["configs_used"].append(optuna_config)

            # Extract best parameters
            if not extract_best_parameters(run_manager, args.models):
                logger.error("❌ Failed at parameter extraction phase")
                success = False
        else:
            logger.info("⏭️ Skipping Optuna optimization phase")
            results["optuna_phase"] = "skipped"

        # Phase 2: Comprehensive Comparison
        if not args.skip_comparison and not args.optuna_only and success:
            comparison_config = run_comparison_phase(run_manager, args)
            results["comparison_phase"] = "completed"
            results["configs_used"].append(comparison_config)

            # Generate summary report
            if not generate_summary_report(run_manager, args.models):
                logger.error("❌ Failed at summary generation phase")
                success = False
        else:
            logger.info("⏭️ Skipping comparison phase")
            results["comparison_phase"] = "skipped"

        # Calculate total execution time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        # Add timing info to results
        results["total_execution_time"] = {
            "seconds": total_time,
            "formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "human_readable": (
                f"{hours}h {minutes}m {seconds}s"
                if hours > 0
                else f"{minutes}m {seconds}s"
            ),
        }

        # Create run summary
        run_manager.create_run_summary(results)

        if success:
            # Print final information
            print("\n" + "=" * 60)
            print("🎉 PRODUCTION EVALUATION COMPLETED")
            print("=" * 60)
            run_manager.print_run_info()

            print(f"\n📊 Results Summary:")
            print(f"   Models: {', '.join(args.models)}")
            print(f"   Max clips: {args.max_clips or 'full dataset'}")
            print(f"   Optuna Phase: {results['optuna_phase']}")
            print(f"   Comparison Phase: {results['comparison_phase']}")
            print(f"   Configs Used: {len(results['configs_used'])}")
            print(
                f"   ⏱️ Total Execution Time: {results['total_execution_time']['human_readable']}"
            )

            print(f"\n🔍 View Results:")
            print(
                f"   MLflow UI: mlflow ui --backend-store-uri {run_manager.mlflow_dir}"
            )
            print(f"   Run Summary: {run_manager.run_dir}/run_summary.json")
            print(f"   Predictions: {run_manager.predictions_dir}")
            print(f"   Visualizations: {run_manager.visualizations_dir}")

            # Show shared directory info
            shared_uri = RunManager.get_shared_mlflow_uri()
            print(f"\n🌐 Shared MLflow Access:")
            print(f"   All Experiments: mlflow ui --backend-store-uri {shared_uri}")

            logger.info(
                f"🎯 Production evaluation completed successfully in {results['total_execution_time']['human_readable']}!"
            )

    except Exception as e:
        # Calculate total time even on failure
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        results["total_execution_time"] = {
            "seconds": total_time,
            "formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "human_readable": (
                f"{hours}h {minutes}m {seconds}s"
                if hours > 0
                else f"{minutes}m {seconds}s"
            ),
        }

        logger.error(
            f"❌ Production evaluation failed after {results['total_execution_time']['human_readable']}: {e}"
        )
        results["error"] = str(e)
        run_manager.create_run_summary(results)
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
