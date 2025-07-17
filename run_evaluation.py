#!/usr/bin/env python3
"""
Surfing Pose Estimation Evaluation Framework
Main evaluation script with organized run management
Runs Optuna optimization followed by comprehensive comparison
"""

import argparse
import logging
import sys
import json
import yaml
from pathlib import Path
import time
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.run_manager import RunManager
from utils.pose_evaluator import PoseEvaluator
from utils.optuna_optimizer import OptunaPoseOptimizer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run pose estimation evaluation with organized run management"
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


def run_optuna_phase(run_manager: RunManager, args, optuna_maneuvers: List) -> Dict:
    """Run Optuna optimization phase with pre-selected data"""
    logger = logging.getLogger(__name__)

    logger.info("🔍 Starting Optuna optimization phase...")

    # Create Optuna-specific config
    optuna_config_path = run_manager.create_config_for_phase(
        "optuna", args.config, args.max_clips
    )

    # Load config
    with open(optuna_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize optimizer with pre-selected data
    optimizer = OptunaPoseOptimizer(config, run_manager)

    logger.info(
        f"Using pre-selected {len(optuna_maneuvers)} maneuvers for optimization"
    )

    # Run optimization for each model
    results = {}
    for model_name in args.models:
        # Check model availability using a temporary evaluator
        temp_evaluator = PoseEvaluator(config)
        if model_name not in temp_evaluator.get_available_models():
            logger.warning(f"Model {model_name} not available, skipping")
            continue

        logger.info(f"Optimizing {model_name}")
        model_result = optimizer.optimize_model(model_name, optuna_maneuvers)
        results[model_name] = model_result

    # Save best parameters
    optimizer.save_best_parameters(results)

    logger.info("✅ Optuna optimization completed successfully")
    return {"config_path": optuna_config_path, "results": results}


# The extract_best_parameters function has been moved to OptunaPoseOptimizer class


def run_comparison_phase(
    run_manager: RunManager,
    args,
    comparison_maneuvers: List,
    visualization_manifest_path: str,
) -> Dict:
    """Run comprehensive comparison phase with pre-selected data"""
    logger = logging.getLogger(__name__)

    logger.info("📊 Starting comprehensive comparison phase...")

    # Create comparison-specific config
    comparison_config_path = run_manager.create_config_for_phase(
        "comparison",
        args.comparison_config,
        args.max_clips,
    )

    # Load config
    with open(comparison_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set MLflow experiment name from comparison config
    import mlflow

    experiment_name = config.get("mlflow", {}).get(
        "experiment_name", "surf_pose_comparison"
    )
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: {experiment_name}")

    # Force h264 for macOS comparison (override any config issues)
    video_format_override = (
        "h264" if "macos" in comparison_config_path.lower() else None
    )
    if video_format_override:
        # Fix config before initializing evaluator
        config["data_source"]["video_clips"]["input_format"] = video_format_override

    # Initialize evaluator with pre-selected data
    evaluator = PoseEvaluator(config)
    evaluator.run_manager = run_manager  # Set run manager for visualizations

    logger.info(
        f"Using pre-selected {len(comparison_maneuvers)} maneuvers for comparison"
    )

    # Run evaluation for each model with MLflow logging
    results = {}
    for model_name in args.models:
        if model_name not in evaluator.get_available_models():
            logger.warning(f"Model {model_name} not available, skipping")
            continue

        logger.info(f"Running comparison evaluation for {model_name}")

        # Start MLflow run for comparison
        run_name = f"{model_name}_comparison_eval"
        with mlflow.start_run(run_name=run_name):
            # Log run info
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("phase", "comparison")
            mlflow.log_param("max_clips", args.max_clips)

            # Run single model evaluation with pre-selected data
            model_result = evaluator.evaluate_single_model_with_data(
                model_name, comparison_maneuvers, visualization_manifest_path
            )

            # Log metrics to MLflow - Fix: Access metrics correctly
            if isinstance(model_result, dict) and "error" not in model_result:
                # Log all available metrics
                for metric_name, value in model_result.items():
                    if isinstance(value, (int, float)):
                        # Convert numpy types to native Python types
                        if hasattr(value, "item"):
                            value = value.item()
                        mlflow.log_metric(metric_name, value)

            results[model_name] = model_result

    logger.info("✅ Comprehensive comparison completed successfully")
    return {"config_path": comparison_config_path, "results": results}


def generate_summary_report(
    run_manager: RunManager, models: list, max_clips: int = None
):
    """Generate a summary report comparing the results"""
    logger = logging.getLogger(__name__)
    logger.info("📋 Generating summary report")

    try:
        import mlflow
        import pandas as pd
        from datetime import datetime

        mlflow.set_tracking_uri(str(run_manager.mlflow_dir))

        # Get comparison experiment - Fix: Use dynamic experiment name from config
        # Try to load the comparison config to get the correct experiment name
        comparison_config_files = list(
            run_manager.run_dir.glob("comparison_config_*.yaml")
        )
        experiment_name = "surf_pose_production_comparison"  # default

        if comparison_config_files:
            try:
                import yaml

                with open(comparison_config_files[0], "r") as f:
                    config = yaml.safe_load(f)
                experiment_name = config.get("mlflow", {}).get(
                    "experiment_name", experiment_name
                )
            except Exception as e:
                logger.warning(f"Could not read comparison config: {e}")

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            # Try to find any experiment with "comparison" in the name
            all_experiments = mlflow.search_experiments()
            comparison_experiments = [
                exp for exp in all_experiments if "comparison" in exp.name.lower()
            ]
            if comparison_experiments:
                experiment = comparison_experiments[0]  # Use the first one found
                logger.info(f"Using experiment: {experiment.name}")
            else:
                logger.error(
                    f"❌ No comparison experiment found (looking for '{experiment_name}')"
                )
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
            "dataset_size": max_clips or "full",
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

    # Set MLflow tracking URI immediately after run manager creation
    import mlflow

    mlflow.set_tracking_uri(str(run_manager.mlflow_dir))
    logger.info(f"🔗 MLflow tracking URI set to: {run_manager.mlflow_dir}")

    # Optional cleanup
    if args.cleanup:
        run_manager.cleanup_old_runs()

    # Track results
    results = {"optuna_phase": None, "comparison_phase": None, "configs_used": []}

    success = True

    try:
        # Generate centralized data selections ONCE for all phases
        logger.info("🎯 Generating centralized data selections...")

        # Load base config to determine proper max_clips for each phase
        with open(args.config, "r") as f:
            base_config = yaml.safe_load(f)

        # Generate data selection manifests
        manifest_paths = run_manager.generate_data_selections(
            config=base_config,
            optuna_max_clips=args.max_clips,  # Smaller subset for Optuna
            comparison_max_clips=args.max_clips,  # Full requested set for comparison
        )

        # Load pre-selected data from manifests
        optuna_maneuvers = []
        comparison_maneuvers = []
        visualization_manifest_path = None

        if manifest_paths.get("optuna"):
            from data_handling.data_loader import SurfingDataLoader

            temp_loader = SurfingDataLoader(base_config)
            optuna_maneuvers = temp_loader.load_maneuvers_from_manifest(
                manifest_paths["optuna"]
            )
            logger.info(f"📖 Loaded {len(optuna_maneuvers)} maneuvers for Optuna phase")

        if manifest_paths.get("comparison"):
            from data_handling.data_loader import SurfingDataLoader

            temp_loader = SurfingDataLoader(base_config)
            comparison_maneuvers = temp_loader.load_maneuvers_from_manifest(
                manifest_paths["comparison"]
            )
            logger.info(
                f"📖 Loaded {len(comparison_maneuvers)} maneuvers for comparison phase"
            )

        if manifest_paths.get("visualization"):
            visualization_manifest_path = manifest_paths["visualization"]
            logger.info(
                f"📖 Visualization manifest ready: {Path(visualization_manifest_path).name}"
            )

        # Phase 1: Optuna Optimization
        if not args.skip_optuna and not args.comparison_only:
            if optuna_maneuvers:
                optuna_result = run_optuna_phase(run_manager, args, optuna_maneuvers)
                results["optuna_phase"] = "completed"
                results["configs_used"].append(optuna_result["config_path"])
                results["optuna_results"] = optuna_result["results"]
            else:
                logger.warning("⚠️ No Optuna data available, skipping optimization")
                results["optuna_phase"] = "skipped"
        else:
            logger.info("⏭️ Skipping Optuna optimization phase")
            results["optuna_phase"] = "skipped"

        # Phase 2: Comprehensive Comparison
        if not args.skip_comparison and not args.optuna_only:
            if comparison_maneuvers:
                comparison_result = run_comparison_phase(
                    run_manager, args, comparison_maneuvers, visualization_manifest_path
                )
                results["comparison_phase"] = "completed"
                results["configs_used"].append(comparison_result["config_path"])
                results["comparison_results"] = comparison_result["results"]

                # Generate summary report
                if not generate_summary_report(
                    run_manager, args.models, args.max_clips
                ):
                    logger.error("❌ Failed at summary generation phase")
                    success = False
            else:
                logger.warning("⚠️ No comparison data available, skipping comparison")
                results["comparison_phase"] = "skipped"
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
