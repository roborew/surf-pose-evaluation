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
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.run_manager import RunManager
from utils.pose_evaluator import PoseEvaluator
from utils.optuna_optimizer import OptunaPoseOptimizer
from utils.memory_profiler import MemoryProfiler


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
        "--max-clips",
        type=int,
        help="[DEPRECATED] Maximum clips for both phases. Use --optuna-max-clips and --comparison-max-clips instead",
    )
    parser.add_argument(
        "--optuna-max-clips",
        type=int,
        help="Maximum number of clips for Optuna optimization phase (overrides config file and --eval-mode)",
    )
    parser.add_argument(
        "--comparison-max-clips",
        type=int,
        help="Maximum number of clips for comparison phase (overrides config file and --eval-mode)",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["quick", "comprehensive", "auto"],
        default="comprehensive",
        help="""Evaluation mode (default: comprehensive):
  quick        - Use quick_test from configs (faster, fewer clips)
  comprehensive - Use comprehensive_test from configs (thorough, more clips)  
  auto         - Use whatever is enabled in each config file""",
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
        "--coco-images",
        type=int,
        default=100,
        help="Number of COCO images for ground truth validation (default: 100 for production accuracy)",
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


def get_clips_from_config(
    config_path: str, eval_mode: str = "comprehensive"
) -> Optional[int]:
    """Extract clip count from config based on evaluation mode

    Args:
        config_path: Path to configuration file
        eval_mode: Evaluation mode ("quick", "comprehensive", "auto")

    Returns:
        Number of clips from config, or None if not found
    """
    logger = logging.getLogger(__name__)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        eval_config = config.get("evaluation", {})

        if eval_mode == "comprehensive":
            # Try comprehensive_test first
            if eval_config.get("comprehensive_test", {}).get("enabled", False):
                clips = eval_config["comprehensive_test"].get("num_clips")
                if clips is not None:
                    logger.info(
                        f"📊 Using comprehensive_test clips from {Path(config_path).name}: {clips}"
                    )
                    return clips
        elif eval_mode == "quick":
            # Try quick_test first
            if eval_config.get("quick_test", {}).get("enabled", False):
                clips = eval_config["quick_test"].get("num_clips")
                if clips is not None:
                    logger.info(
                        f"⚡ Using quick_test clips from {Path(config_path).name}: {clips}"
                    )
                    return clips
        elif eval_mode == "auto":
            # Current behavior - try comprehensive, then quick
            if eval_config.get("comprehensive_test", {}).get("enabled", False):
                clips = eval_config["comprehensive_test"].get("num_clips")
                if clips is not None:
                    logger.info(
                        f"📊 Auto-detected comprehensive_test clips from {Path(config_path).name}: {clips}"
                    )
                    return clips
            elif eval_config.get("quick_test", {}).get("enabled", False):
                clips = eval_config["quick_test"].get("num_clips")
                if clips is not None:
                    logger.info(
                        f"⚡ Auto-detected quick_test clips from {Path(config_path).name}: {clips}"
                    )
                    return clips

        # Fallback: try the other mode if primary mode didn't work
        if eval_mode == "comprehensive":
            if eval_config.get("quick_test", {}).get("enabled", False):
                clips = eval_config["quick_test"].get("num_clips")
                if clips is not None:
                    logger.warning(
                        f"⚠️ comprehensive_test not available, falling back to quick_test: {clips}"
                    )
                    return clips
        elif eval_mode == "quick":
            if eval_config.get("comprehensive_test", {}).get("enabled", False):
                clips = eval_config["comprehensive_test"].get("num_clips")
                if clips is not None:
                    logger.warning(
                        f"⚠️ quick_test not available, falling back to comprehensive_test: {clips}"
                    )
                    return clips

        logger.warning(
            f"⚠️ No valid evaluation config found in {Path(config_path).name}"
        )
        return None

    except Exception as e:
        logger.warning(f"⚠️ Failed to load config {config_path}: {e}")
        return None


def resolve_clip_counts(
    args, optuna_config_path: str, comparison_config_path: str
) -> tuple[Optional[int], Optional[int]]:
    """Resolve clip counts using the new configuration hierarchy

    Priority (highest to lowest):
    1. CLI Parameters (--optuna-max-clips, --comparison-max-clips)
    2. Legacy CLI Parameter (--max-clips) with deprecation warning
    3. Evaluation Mode + Config Files (--eval-mode)
    4. None (full dataset)

    Args:
        args: Parsed command line arguments
        optuna_config_path: Path to Optuna configuration file
        comparison_config_path: Path to comparison configuration file

    Returns:
        Tuple of (optuna_clips, comparison_clips)
    """
    logger = logging.getLogger(__name__)

    # Resolve Optuna clips
    if args.optuna_max_clips is not None:
        optuna_clips = args.optuna_max_clips
        logger.info(f"🔧 Using CLI --optuna-max-clips: {optuna_clips}")
    elif args.max_clips is not None:  # backward compatibility
        optuna_clips = args.max_clips
        logger.warning(
            "⚠️ --max-clips is deprecated for optuna phase. Use --optuna-max-clips instead"
        )
        logger.info(f"🔧 Using deprecated --max-clips for optuna: {optuna_clips}")
    else:
        # Get from config based on eval_mode
        optuna_clips = get_clips_from_config(optuna_config_path, args.eval_mode)
        if optuna_clips is None:
            logger.info("🔧 No optuna clips specified, using full dataset")

    # Resolve Comparison clips
    if args.comparison_max_clips is not None:
        comparison_clips = args.comparison_max_clips
        logger.info(f"🔧 Using CLI --comparison-max-clips: {comparison_clips}")
    elif args.max_clips is not None:  # backward compatibility
        comparison_clips = args.max_clips
        logger.warning(
            "⚠️ --max-clips is deprecated for comparison phase. Use --comparison-max-clips instead"
        )
        logger.info(
            f"🔧 Using deprecated --max-clips for comparison: {comparison_clips}"
        )
    else:
        # Get from config based on eval_mode
        comparison_clips = get_clips_from_config(comparison_config_path, args.eval_mode)
        if comparison_clips is None:
            logger.warning("⚠️ No comparison clips found in config, using default: 50")
            comparison_clips = 50  # Safe fallback

    return optuna_clips, comparison_clips


def validate_parameters(args) -> tuple[list[str], list[str]]:
    """Validate parameter combinations and provide clear error messages

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (errors, warnings) as lists of strings
    """
    errors = []
    warnings = []

    # Check for deprecated usage
    if args.max_clips is not None:
        warnings.append(
            "--max-clips is deprecated. Use --optuna-max-clips and --comparison-max-clips for better control"
        )

    # Check for conflicting phase options
    phase_options = [
        args.skip_optuna,
        args.skip_comparison,
        args.optuna_only,
        args.comparison_only,
    ]
    if sum(phase_options) > 1:
        errors.append(
            "Cannot combine multiple phase control options (--skip-optuna, --skip-comparison, --optuna-only, --comparison-only)"
        )

    # Validate config files exist
    if not Path(args.config).exists():
        errors.append(f"Optuna config file not found: {args.config}")

    if not Path(args.comparison_config).exists():
        errors.append(f"Comparison config file not found: {args.comparison_config}")

    # Validate eval_mode (should be caught by argparse, but double-check)
    if args.eval_mode not in ["quick", "comprehensive", "auto"]:
        errors.append(f"Invalid eval-mode: {args.eval_mode}")

    return errors, warnings


def run_optuna_phase(run_manager: RunManager, args, optuna_maneuvers: List) -> Dict:
    """Run Optuna optimization phase with pre-selected data and dynamic time allocation"""
    logger = logging.getLogger(__name__)

    logger.info("🔍 Starting Optuna optimization phase with dynamic time allocation...")

    # Create Optuna-specific config
    optuna_config_path = run_manager.create_config_for_phase(
        "optuna", args.config, args.max_clips
    )

    # Load config
    with open(optuna_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.optuna_trials:
        if "optuna" not in config:
            config["optuna"] = {}
        config["optuna"]["n_trials"] = args.optuna_trials
        logger.info(
            f"🔧 Overriding Optuna trials from command line: {args.optuna_trials}"
        )

        # Save updated config
        with open(optuna_config_path, "w") as f:
            yaml.safe_dump(config, f)

    # Initialize dynamic optimizer for intelligent time allocation
    from utils.dynamic_optimizer import create_dynamic_optimizer_from_config

    dynamic_optimizer = create_dynamic_optimizer_from_config(config)

    # Initialize optimizer with pre-selected data
    optimizer = OptunaPoseOptimizer(config, run_manager)

    logger.info(
        f"Using pre-selected {len(optuna_maneuvers)} maneuvers for optimization"
    )

    # Run optimization for each model with dynamic time allocation
    results = {}
    for model_name in args.models:
        # Check model availability using a temporary evaluator
        temp_evaluator = PoseEvaluator(config)
        if model_name not in temp_evaluator.get_available_models():
            logger.warning(f"Model {model_name} not available, skipping")
            continue

        # Get allocated time for this model
        allocated_time = dynamic_optimizer.get_time_for_model(model_name)
        logger.info(
            f"Optimizing {model_name} with {allocated_time/3600:.1f}h allocated time"
        )

        # Update config with model-specific timeout
        if "optuna" not in config:
            config["optuna"] = {}
        config["optuna"]["timeout_minutes"] = int(allocated_time / 60)

        # Run optimization
        start_time = time.time()
        model_result = optimizer.optimize_model(model_name, optuna_maneuvers)
        end_time = time.time()

        # Track results for dynamic allocation
        trials_completed = (
            len(model_result.get("trials", [])) if isinstance(model_result, dict) else 0
        )
        best_score = (
            model_result.get("pck_0_2", 0.0) if isinstance(model_result, dict) else 0.0
        )
        time_taken = end_time - start_time

        # Check if early stopping occurred (this would need to be tracked in the optimizer)
        early_stopped = False  # TODO: Track this in the optimizer

        # Update dynamic optimizer
        dynamic_optimizer.update_model_result(
            model_name=model_name,
            trials_completed=trials_completed,
            best_score=best_score,
            time_taken=time_taken,
            early_stopped=early_stopped,
        )

        results[model_name] = model_result

    # Save best parameters
    optimizer.save_best_parameters(results)

    # Save and display optimization summary
    summary_path = run_manager.reports_dir / "dynamic_optimization_summary.json"
    dynamic_optimizer.save_summary(summary_path)
    dynamic_optimizer.print_summary()

    logger.info(
        "✅ Optuna optimization completed successfully with dynamic time allocation"
    )
    return {
        "config_path": optuna_config_path,
        "results": results,
        "dynamic_summary": summary_path,
    }


# The extract_best_parameters function has been moved to OptunaPoseOptimizer class


def run_coco_validation_phase(
    run_manager: RunManager, args, coco_annotations_path: str
) -> Dict:
    """Run COCO validation phase for ground truth PCK scores using optimized parameters"""
    logger = logging.getLogger(__name__)

    logger.info("🧪 Starting COCO validation phase with optimized parameters...")

    # Create COCO-specific config (use comparison config as base)
    coco_config_path = run_manager.create_config_for_phase(
        "coco_validation",
        args.comparison_config,
        None,  # No clip limit for COCO
    )

    # Load config
    with open(coco_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Configure best parameters loading for COCO validation - REQUIRE optimized parameters
    if "models" not in config:
        config["models"] = {}

    # Check if best parameters exist and are valid
    best_params_file = run_manager.best_params_dir / "best_parameters.yaml"
    if best_params_file.exists():
        # Check if the parameters file contains actual optimized parameters
        try:
            with open(best_params_file, "r") as f:
                best_params = yaml.safe_load(f)

            # Check if we have valid parameters (not just placeholders or errors)
            valid_params = {}
            for model_name, params in best_params.items():
                if (
                    isinstance(params, dict)
                    and "error" not in params
                    and "placeholder" not in params
                ):
                    valid_params[model_name] = params

            if valid_params:
                config["models"]["load_best_params"] = {
                    "enabled": True,
                    "source_path": str(run_manager.best_params_dir),
                    "fallback_to_defaults": False,  # FAIL EARLY if parameters missing
                }
                logger.info(
                    f"✅ Configured COCO validation to use optimized parameters for models: {list(valid_params.keys())}"
                )
            else:
                logger.error(
                    "❌ Best parameters file exists but contains no valid optimized parameters!"
                )
                logger.error(
                    "COCO validation requires optimized parameters - stopping to prevent false positives."
                )
                return {
                    "error": "No valid optimized parameters available for COCO validation"
                }

        except Exception as e:
            logger.error(f"❌ Failed to validate best parameters file: {e}")
            logger.error(
                "COCO validation requires optimized parameters - stopping to prevent false positives."
            )
            return {"error": f"Failed to validate best parameters: {e}"}
    else:
        logger.error("❌ No optimized parameters found!")
        logger.error(
            "COCO validation requires optimized parameters - stopping to prevent false positives."
        )
        return {"error": "No optimized parameters file found for COCO validation"}

    # Save updated config
    with open(coco_config_path, "w") as f:
        yaml.safe_dump(config, f)

    # Set MLflow experiment name for COCO validation
    import mlflow

    experiment_name = (
        config.get("mlflow", {})
        .get("experiment_name", "surf_pose_coco_validation")
        .replace("comparison", "coco_validation")
    )
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: {experiment_name}")

    # Initialize evaluator
    evaluator = PoseEvaluator(config)
    evaluator.run_manager = run_manager

    logger.info(
        f"Running COCO validation for ground truth PCK scores ({args.coco_images} images)"
    )

    # Run COCO validation
    coco_results = evaluator.run_coco_validation_phase(
        models=args.models,
        coco_annotations_path=coco_annotations_path,
        coco_images_path=None,  # Will download images as needed
        max_images=args.coco_images,  # Configurable via --coco-images (default: 100)
    )

    logger.info("✅ COCO validation phase completed successfully")
    return {"config_path": coco_config_path, "results": coco_results}


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

    # Configure best parameters loading for comparison phase - REQUIRE optimized parameters
    if "models" not in config:
        config["models"] = {}

    # Check if best parameters exist and are valid
    best_params_file = run_manager.best_params_dir / "best_parameters.yaml"
    if best_params_file.exists():
        # Check if the parameters file contains actual optimized parameters
        try:
            with open(best_params_file, "r") as f:
                best_params = yaml.safe_load(f)

            # Check if we have valid parameters (not just placeholders or errors)
            valid_params = {}
            for model_name, params in best_params.items():
                if (
                    isinstance(params, dict)
                    and "error" not in params
                    and "placeholder" not in params
                ):
                    valid_params[model_name] = params

            if valid_params:
                config["models"]["load_best_params"] = {
                    "enabled": True,
                    "source_path": str(run_manager.best_params_dir),
                    "fallback_to_defaults": False,  # FAIL EARLY if parameters missing
                }
                logger.info(
                    f"✅ Configured comparison to use optimized parameters for models: {list(valid_params.keys())}"
                )
            else:
                logger.error(
                    "❌ Best parameters file exists but contains no valid optimized parameters!"
                )
                logger.error(
                    "Comparison phase requires optimized parameters - stopping to prevent false positives."
                )
                return {
                    "error": "No valid optimized parameters available for comparison phase"
                }

        except Exception as e:
            logger.error(f"❌ Failed to validate best parameters file: {e}")
            logger.error(
                "Comparison phase requires optimized parameters - stopping to prevent false positives."
            )
            return {"error": f"Failed to validate best parameters: {e}"}
    else:
        logger.error("❌ No optimized parameters found!")
        logger.error(
            "Comparison phase requires optimized parameters - stopping to prevent false positives."
        )
        return {"error": "No optimized parameters file found for comparison phase"}

    # Save updated config
    with open(comparison_config_path, "w") as f:
        yaml.safe_dump(config, f)

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


def run_consensus_phase(
    run_manager: RunManager, args, comparison_maneuvers: List
) -> Optional[Dict]:
    """Run consensus evaluation phase to calculate relative PCK metrics"""
    logger = logging.getLogger(__name__)

    try:
        # Import consensus evaluator
        from utils.consensus_evaluator import ConsensusEvaluator

        # Load comparison config
        comparison_config_files = list(
            run_manager.run_dir.glob("comparison_config_*.yaml")
        )
        if not comparison_config_files:
            logger.error("No comparison config found for consensus evaluation")
            return None

        with open(comparison_config_files[0], "r") as f:
            config = yaml.safe_load(f)

        # Initialize consensus evaluator with reference models
        reference_models = ["pytorch_pose", "yolov8_pose", "mmpose"]
        available_reference_models = [m for m in reference_models if m in args.models]

        logger.info(f"🔍 Consensus evaluation check:")
        logger.info(f"   Reference models required: {reference_models}")
        logger.info(f"   Models in args: {args.models}")
        logger.info(f"   Available reference models: {available_reference_models}")
        logger.info(f"   Prediction directory: {run_manager.predictions_dir}")
        logger.info(
            f"   Prediction directory exists: {run_manager.predictions_dir.exists()}"
        )

        if len(available_reference_models) < 2:
            logger.warning(
                f"⚠️ Need at least 2 reference models for consensus, have {len(available_reference_models)}"
            )
            logger.warning("⏭️ Skipping consensus evaluation")
            return None

        logger.info(
            f"🎯 Using reference models for consensus: {available_reference_models}"
        )

        consensus_evaluator = ConsensusEvaluator(
            config, reference_models=available_reference_models
        )

        # CRITICAL: Set run_manager on consensus evaluator's pose_evaluator
        # to ensure it uses the correct prediction file paths
        consensus_evaluator.pose_evaluator.run_manager = run_manager

        # Also update the prediction handler to use the correct run-specific path
        prediction_config = config.get("output", {}).get("predictions", {})
        if prediction_config.get("enabled", True):
            from utils.prediction_file_format import PredictionFileHandler

            consensus_evaluator.pose_evaluator.prediction_handler = (
                PredictionFileHandler(str(run_manager.predictions_dir))
            )
            logger.info(
                f"🔗 Updated consensus evaluator to use prediction path: {run_manager.predictions_dir}"
            )

        # Run consensus evaluation using maneuvers that actually have prediction files
        logger.info(
            "🔍 Determining consensus dataset from available prediction files..."
        )

        # Get maneuvers that actually have prediction files (from comparison phase execution)
        prediction_dir = run_manager.predictions_dir
        if prediction_dir.exists() and list(prediction_dir.iterdir()):
            # Find a model directory that has predictions
            model_dirs = [d for d in prediction_dir.iterdir() if d.is_dir()]
            if model_dirs:
                sample_model_dir = model_dirs[0]
                prediction_files = list(sample_model_dir.glob("*_predictions.json"))

                # Extract maneuver IDs from prediction files
                actual_maneuver_ids = set()
                for pred_file in prediction_files:
                    try:
                        with open(pred_file, "r") as f:
                            pred_data = json.load(f)
                            maneuver_id = pred_data.get("maneuver_id")
                            if maneuver_id:
                                actual_maneuver_ids.add(maneuver_id)
                    except Exception as e:
                        logger.warning(
                            f"Failed to read prediction file {pred_file}: {e}"
                        )

                # Filter comparison_maneuvers to only include those with predictions
                consensus_maneuvers = [
                    maneuver
                    for maneuver in comparison_maneuvers
                    if maneuver.maneuver_id in actual_maneuver_ids
                ]

                logger.info(f"📊 Found {len(prediction_files)} prediction files")
                logger.info(
                    f"🎯 Using {len(consensus_maneuvers)} maneuvers for consensus (from {len(comparison_maneuvers)} available)"
                )
            else:
                logger.warning("No model prediction directories found")
                consensus_maneuvers = []
        else:
            logger.warning("Prediction directory not found or empty")
            consensus_maneuvers = []

        if consensus_maneuvers:
            consensus_results = consensus_evaluator.run_consensus_evaluation(
                consensus_maneuvers, target_models=args.models, save_consensus=True
            )
        else:
            logger.error(
                "❌ No valid consensus maneuvers found - skipping consensus evaluation"
            )
            consensus_results = {}

        if consensus_results:
            logger.info("✅ Consensus evaluation completed successfully")
            logger.info(
                f"   Consensus results for models: {list(consensus_results.keys())}"
            )
        else:
            logger.warning("⚠️ Consensus evaluation returned empty results")

        return consensus_results

    except ImportError:
        logger.error(
            "❌ ConsensusEvaluator not available - skipping consensus evaluation"
        )
        return None
    except Exception as e:
        logger.error(f"❌ Consensus evaluation failed: {e}")
        import traceback

        logger.error(f"   Traceback: {traceback.format_exc()}")
        return None


def merge_consensus_with_comparison(
    comparison_results: Dict[str, Dict], consensus_results: Dict[str, Dict]
) -> Dict[str, Dict]:
    """Merge consensus metrics into comparison results"""
    logger = logging.getLogger(__name__)

    merged_results = comparison_results.copy()

    for model_name, consensus_data in consensus_results.items():
        if model_name in merged_results:
            # Extract consensus metrics
            consensus_metrics = consensus_data.get("consensus_metrics", {})

            if consensus_metrics:
                # Calculate aggregated consensus metrics
                all_relative_pck = []
                all_relative_pck_02 = []
                all_consensus_coverage = []
                all_consensus_confidence = []

                for maneuver_id, maneuver_metrics in consensus_metrics.items():
                    # Fix: Consensus metrics are now stored directly in relative_pck dict
                    relative_pck = maneuver_metrics.get("relative_pck", {})
                    consensus_quality = maneuver_metrics.get("consensus_quality", {})

                    # Extract metrics from the relative_pck dict directly
                    if "consensus_pck_error" in relative_pck:
                        all_relative_pck.append(
                            1.0 - relative_pck["consensus_pck_error"]
                        )
                    if "consensus_pck_0.2" in relative_pck:
                        all_relative_pck_02.append(relative_pck["consensus_pck_0.2"])

                    # Backward compatibility with old naming
                    if "relative_pck_error" in relative_pck:
                        all_relative_pck.append(
                            1.0 - relative_pck["relative_pck_error"]
                        )
                    if "relative_pck_0.2" in relative_pck:
                        all_relative_pck_02.append(relative_pck["relative_pck_0.2"])

                    # Fix: Use the correct key names from our updated metrics
                    if "consensus_coverage_ratio" in relative_pck:
                        all_consensus_coverage.append(
                            relative_pck["consensus_coverage_ratio"]
                        )

                    # Also check consensus_quality for backward compatibility and new format
                    if "consensus_coverage" in consensus_quality:
                        all_consensus_coverage.append(
                            consensus_quality["consensus_coverage"]
                        )
                    if "avg_consensus_confidence" in consensus_quality:
                        all_consensus_confidence.append(
                            consensus_quality["avg_consensus_confidence"]
                        )

                    # Also check relative_pck for newer format
                    if "consensus_coverage" in relative_pck:
                        all_consensus_coverage.append(
                            relative_pck["consensus_coverage"]
                        )
                    if "consensus_confidence" in relative_pck:
                        all_consensus_confidence.append(
                            relative_pck["consensus_confidence"]
                        )

                # Add aggregated consensus-based metrics
                if all_relative_pck:
                    merged_results[model_name]["pose_consensus_pck_error_mean"] = (
                        1.0 - np.mean(all_relative_pck)
                    )
                if all_relative_pck_02:
                    merged_results[model_name]["pose_consensus_pck_0.2_mean"] = np.mean(
                        all_relative_pck_02
                    )
                if all_consensus_coverage:
                    merged_results[model_name]["pose_consensus_coverage_mean"] = (
                        np.mean(all_consensus_coverage)
                    )
                if all_consensus_confidence:
                    merged_results[model_name]["pose_consensus_confidence_mean"] = (
                        np.mean(all_consensus_confidence)
                    )

                logger.info(f"✅ Added consensus metrics for {model_name}")
            else:
                logger.warning(f"⚠️ No consensus metrics found for {model_name}")
        else:
            logger.warning(f"⚠️ Model {model_name} not found in comparison results")

    return merged_results


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

        # Get COCO validation data from separate COCO experiment
        coco_data = {}
        try:
            # Find COCO validation experiment
            all_experiments = mlflow.search_experiments()
            coco_experiments = [
                exp
                for exp in all_experiments
                if "coco" in exp.name.lower() and "validation" in exp.name.lower()
            ]

            if coco_experiments:
                coco_experiment = coco_experiments[0]  # Use the most recent one
                logger.info(f"Found COCO experiment: {coco_experiment.name}")

                # Get COCO runs
                coco_runs = mlflow.search_runs(
                    experiment_ids=[coco_experiment.experiment_id]
                )

                # Create mapping of model_name -> COCO metrics
                for _, coco_run in coco_runs.iterrows():
                    model_name = coco_run.get("params.model_name")
                    if model_name:
                        coco_data[model_name] = {
                            "coco_pck_0.1": coco_run.get("metrics.coco_pck_0.1"),
                            "coco_pck_0.2": coco_run.get("metrics.coco_pck_0.2"),
                            "coco_pck_0.3": coco_run.get("metrics.coco_pck_0.3"),
                            "coco_pck_0.5": coco_run.get("metrics.coco_pck_0.5"),
                            "coco_pck_error_mean": coco_run.get(
                                "metrics.coco_pck_error_mean"
                            ),
                            "coco_detection_f1": coco_run.get(
                                "metrics.coco_detection_f1"
                            ),
                            "coco_fps_mean": coco_run.get("metrics.coco_fps_mean"),
                            "coco_inference_time_ms": coco_run.get(
                                "metrics.coco_inference_time_ms"
                            ),
                            "coco_images_processed": coco_run.get(
                                "metrics.coco_total_images_processed"
                            ),
                        }
                logger.info(f"Retrieved COCO data for {len(coco_data)} models")
            else:
                logger.warning("⚠️ No COCO validation experiment found")
        except Exception as e:
            logger.warning(f"⚠️ Could not retrieve COCO data: {e}")

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
                    # Legacy PCK (usually null without ground truth)
                    "pck_error_mean": run.get("metrics.pose_pck_error_mean", None),
                    "detection_f1": run.get("metrics.pose_detection_f1_mean", None),
                    # Consensus-based metrics (synthetic ground truth from 3 reference models) - PRIMARY METRICS
                    "consensus_pck_error": run.get(
                        "metrics.pose_consensus_pck_error_mean", None
                    ),
                    "consensus_pck_0.2": run.get(
                        "metrics.pose_consensus_pck_0.2_mean", None
                    ),
                    "consensus_coverage": run.get(
                        "metrics.pose_consensus_coverage_mean", None
                    ),
                    "consensus_confidence": run.get(
                        "metrics.pose_consensus_confidence_mean", None
                    ),
                    # COCO Ground Truth Validation - REFERENCE METRICS (from separate COCO experiment)
                    "coco_pck_0.1": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_pck_0.1", None),
                    "coco_pck_0.2": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_pck_0.2", None),
                    "coco_pck_0.3": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_pck_0.3", None),
                    "coco_pck_0.5": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_pck_0.5", None),
                    "coco_pck_error_mean": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_pck_error_mean", None),
                    "coco_detection_f1": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_detection_f1", None),
                    # Enhanced detection metrics
                    "pose_stability_mean": run.get(
                        "metrics.pose_pose_stability_mean_mean", None
                    ),
                    "keypoint_consistency_mean": run.get(
                        "metrics.pose_keypoint_consistency_mean_mean", None
                    ),
                    "avg_keypoint_confidence": run.get(
                        "metrics.pose_avg_keypoint_confidence_mean", None
                    ),
                    "skeleton_completeness_mean": run.get(
                        "metrics.pose_skeleton_completeness_mean_mean", None
                    ),
                    "detection_consistency": run.get(
                        "metrics.pose_detection_consistency_mean", None
                    ),
                },
                "performance": {
                    # Production dataset performance
                    "fps_mean": run.get("metrics.perf_fps_mean", None),
                    "inference_time_ms": run.get(
                        "metrics.perf_avg_inference_time_mean", None
                    ),
                    "memory_usage_mb": run.get(
                        "metrics.perf_max_memory_usage_mean", None
                    ),  # Already in MB from the evaluator
                    # COCO validation performance (from separate COCO experiment)
                    "coco_fps_mean": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_fps_mean", None),
                    "coco_inference_time_ms": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_inference_time_ms", None),
                    "coco_images_processed": coco_data.get(
                        run.get("params.model_name", ""), {}
                    ).get("coco_images_processed", None),
                    # Model characteristics
                    "model_size_mb": run.get("metrics.perf_model_size_mb_mean", None),
                    "memory_efficiency": run.get(
                        "metrics.perf_memory_efficiency_mean", None
                    ),
                    "theoretical_fps": run.get(
                        "metrics.perf_theoretical_fps_mean", None
                    ),
                    # Enhanced memory metrics
                    "avg_memory_usage": run.get(
                        "metrics.perf_avg_memory_usage_mean", None
                    ),
                    "memory_std": run.get("metrics.perf_memory_std_mean", None),
                    "memory_peak_to_avg_ratio": run.get(
                        "metrics.perf_memory_peak_to_avg_ratio_mean", None
                    ),
                    # Comprehensive performance metrics
                    "avg_cpu_utilization": run.get(
                        "metrics.perf_avg_cpu_utilization_mean", None
                    ),
                    "efficiency_score": run.get(
                        "metrics.perf_efficiency_score_mean", None
                    ),
                    "throughput_per_mb": run.get(
                        "metrics.perf_throughput_per_mb_mean", None
                    ),
                    "speed_memory_ratio": run.get(
                        "metrics.perf_speed_memory_ratio_mean", None
                    ),
                    "p95_inference_time_ms": run.get(
                        "metrics.perf_p95_inference_time_ms_mean", None
                    ),
                    "p99_inference_time_ms": run.get(
                        "metrics.perf_p99_inference_time_ms_mean", None
                    ),
                    "single_frame_throughput_fps": run.get(
                        "metrics.perf_single_frame_throughput_fps_mean", None
                    ),
                    "batch_throughput_fps": run.get(
                        "metrics.perf_batch_throughput_fps_mean", None
                    ),
                },
            }
            summary["results"].append(model_result)

        # Add summary statistics
        if summary["results"]:
            # Calculate best performers
            valid_results = [
                r
                for r in summary["results"]
                if r["performance"]["fps_mean"] is not None
            ]
            if valid_results:
                # Best FPS
                best_fps = max(
                    valid_results, key=lambda x: x["performance"]["fps_mean"] or 0
                )
                summary["best_performers"] = {
                    "fastest_model": best_fps["model"],
                    "fastest_fps": best_fps["performance"]["fps_mean"],
                    "most_memory_efficient": min(
                        valid_results,
                        key=lambda x: x["performance"]["avg_memory_usage"]
                        or float("inf"),
                    )["model"],
                    "smallest_model": min(
                        valid_results,
                        key=lambda x: x["performance"]["model_size_mb"] or float("inf"),
                    )["model"],
                }

        # Save summary (handle NaN values for JSON serialization)
        summary_file = run_manager.run_dir / "production_evaluation_summary.json"

        # Replace any remaining NaN values with null for proper JSON
        def replace_nan_recursive(obj):
            if isinstance(obj, dict):
                return {k: replace_nan_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan_recursive(item) for item in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None  # Convert NaN to null in JSON
            else:
                return obj

        clean_summary = replace_nan_recursive(summary)

        with open(summary_file, "w") as f:
            json.dump(clean_summary, f, indent=2)

        logger.info(f"📊 Summary report saved to {summary_file}")

        # Print summary to console
        print("\n" + "=" * 60)
        print("🏆 PRODUCTION EVALUATION SUMMARY")
        print("=" * 60)

        for result in summary["results"]:
            print(f"\n📍 {result['model'].upper()}")

            # Handle None values safely
            pck_error = result["accuracy"]["pck_error_mean"]
            detection_f1 = result["accuracy"]["detection_f1"]

            # COCO ground truth metrics
            coco_pck_02 = result["accuracy"]["coco_pck_0.2"]
            coco_pck_05 = result["accuracy"]["coco_pck_0.5"]
            coco_pck_error = result["accuracy"]["coco_pck_error_mean"]

            fps = result["performance"]["fps_mean"]
            inference_time = result["performance"]["inference_time_ms"]
            model_size = result["performance"]["model_size_mb"]
            memory_efficiency = result["performance"]["memory_efficiency"]
            avg_memory = result["performance"]["avg_memory_usage"]
            memory_peak_ratio = result["performance"]["memory_peak_to_avg_ratio"]

            # COCO performance
            coco_fps = result["performance"]["coco_fps_mean"]
            coco_images = result["performance"]["coco_images_processed"]

            # Display consensus-based accuracy metrics FIRST (primary metrics)
            consensus_pck_error = result["accuracy"]["consensus_pck_error"]
            consensus_pck_02 = result["accuracy"]["consensus_pck_0.2"]
            consensus_coverage = result["accuracy"]["consensus_coverage"]
            consensus_confidence = result["accuracy"]["consensus_confidence"]

            print("   🎯 Consensus-based Accuracy (Synthetic GT from 3 models):")
            print(
                f"     • Consensus PCK Error: {consensus_pck_error:.4f}"
                if consensus_pck_error is not None
                else "     • Consensus PCK Error: N/A"
            )
            print(
                f"     • Consensus PCK@0.2: {consensus_pck_02:.4f}"
                if consensus_pck_02 is not None
                else "     • Consensus PCK@0.2: N/A"
            )
            print(
                f"     • Consensus Coverage: {consensus_coverage:.3f}"
                if consensus_coverage is not None
                else "     • Consensus Coverage: N/A"
            )
            print(
                f"     • Consensus Confidence: {consensus_confidence:.3f}"
                if consensus_confidence is not None
                else "     • Consensus Confidence: N/A"
            )

            # Display COCO ground truth accuracy (reference metrics)
            print("   🏆 COCO Ground Truth Validation (Reference):")
            print(
                f"     • PCK@0.2: {coco_pck_02:.3f}"
                if coco_pck_02 is not None
                else "     • PCK@0.2: N/A"
            )
            print(
                f"     • PCK@0.5: {coco_pck_05:.3f}"
                if coco_pck_05 is not None
                else "     • PCK@0.5: N/A"
            )
            print(
                f"     • PCK Error: {coco_pck_error:.4f}"
                if coco_pck_error is not None
                else "     • PCK Error: N/A"
            )
            print(
                f"     • COCO FPS: {coco_fps:.1f}"
                if coco_fps is not None
                else "     • COCO FPS: N/A"
            )
            print(
                f"     • Images Tested: {coco_images}"
                if coco_images is not None
                else "     • Images Tested: N/A"
            )

            # Display basic metrics
            print("   📊 Basic Metrics:")
            print(
                f"   • Legacy PCK Error: {pck_error:.4f}"
                if pck_error is not None
                else "   • Legacy PCK Error: N/A"
            )
            print(
                f"   • Detection F1: {detection_f1:.4f}"
                if detection_f1 is not None
                else "   • Detection F1: N/A"
            )
            print(
                f"   • Speed (FPS): {fps:.2f}"
                if fps is not None
                else "   • Speed (FPS): N/A"
            )
            print(
                f"   • Inference Time: {inference_time:.2f}ms"
                if inference_time is not None
                else "   • Inference Time: N/A"
            )
            print(
                f"   • Model Size: {model_size:.1f}MB"
                if model_size is not None
                else "   • Model Size: N/A"
            )
            print(
                f"   • Avg Memory: {avg_memory:.1f}MB"
                if avg_memory is not None
                else "   • Avg Memory: N/A"
            )
            print(
                f"   • Memory Efficiency: {memory_efficiency:.2f}"
                if memory_efficiency is not None
                else "   • Memory Efficiency: N/A"
            )
            print(
                f"   • Memory Stability: {memory_peak_ratio:.2f}"
                if memory_peak_ratio is not None
                else "   • Memory Stability: N/A"
            )

            # Display enhanced accuracy metrics
            pose_stability = result["accuracy"]["pose_stability_mean"]
            keypoint_consistency = result["accuracy"]["keypoint_consistency_mean"]
            avg_confidence = result["accuracy"]["avg_keypoint_confidence"]
            skeleton_completeness = result["accuracy"]["skeleton_completeness_mean"]

            print(
                f"   • Pose Stability: {pose_stability:.3f}"
                if pose_stability is not None
                else "   • Pose Stability: N/A"
            )
            print(
                f"   • Keypoint Consistency: {keypoint_consistency:.3f}"
                if keypoint_consistency is not None
                else "   • Keypoint Consistency: N/A"
            )
            print(
                f"   • Avg Confidence: {avg_confidence:.3f}"
                if avg_confidence is not None
                else "   • Avg Confidence: N/A"
            )
            print(
                f"   • Skeleton Completeness: {skeleton_completeness:.3f}"
                if skeleton_completeness is not None
                else "   • Skeleton Completeness: N/A"
            )

            # Display comprehensive performance metrics
            efficiency_score = result["performance"]["efficiency_score"]
            throughput_per_mb = result["performance"]["throughput_per_mb"]
            p95_inference = result["performance"]["p95_inference_time_ms"]

            print(
                f"   • Efficiency Score: {efficiency_score:.3f}"
                if efficiency_score is not None
                else "   • Efficiency Score: N/A"
            )
            print(
                f"   • Throughput/MB: {throughput_per_mb:.1f}"
                if throughput_per_mb is not None
                else "   • Throughput/MB: N/A"
            )
            print(
                f"   • P95 Inference: {p95_inference:.1f}ms"
                if p95_inference is not None
                else "   • P95 Inference: N/A"
            )

        # Display best performers summary
        if "best_performers" in summary:
            print("\n" + "🏆 BEST PERFORMERS" + "=" * 45)
            best = summary["best_performers"]
            print(
                f"   • Fastest Model: {best['fastest_model']} ({best['fastest_fps']:.1f} FPS)"
            )
            print(f"   • Most Memory Efficient: {best['most_memory_efficient']}")
            print(f"   • Smallest Model: {best['smallest_model']}")

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

    # Validate parameters and display errors/warnings
    errors, warnings = validate_parameters(args)

    if warnings:
        for warning in warnings:
            print(f"⚠️ WARNING: {warning}")
        print()  # Add blank line after warnings

    if errors:
        print("❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"   • {error}")
        print("\nPlease fix the above errors and try again.")
        sys.exit(1)

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

    # Initialize memory profiler
    memory_profiler = MemoryProfiler(
        enable_tracemalloc=True,
        monitoring_interval=2.0,  # 2-second intervals for detailed tracking
        enable_continuous_monitoring=True,
        save_snapshots=True,
    )

    try:
        # Start memory profiling
        memory_profiler.start_profiling(run_manager.run_dir)
        memory_profiler.log_milestone("run_started")

        # Generate centralized data selections ONCE for all phases
        logger.info("🎯 Generating centralized data selections...")

        # Load comparison config as base (broader camera selection)
        # This ensures both phases get the full camera coverage
        with open(args.comparison_config, "r") as f:
            base_config = yaml.safe_load(f)

        # Load optuna config for camera filtering preferences
        with open(args.config, "r") as f:
            optuna_config = yaml.safe_load(f)

        logger.info(
            f"🎯 Using comparison config as base for data selection (broader camera coverage)"
        )
        logger.info(f"🔧 Using optuna config for camera filtering preferences")

        # Use new configuration resolution system
        logger.info(f"🎛️ Using evaluation mode: {args.eval_mode}")
        optuna_clips, comparison_clips = resolve_clip_counts(
            args, args.config, args.comparison_config
        )

        manifest_paths = run_manager.generate_data_selections(
            config=base_config,
            optuna_max_clips=optuna_clips,
            comparison_max_clips=comparison_clips,
            optuna_config=optuna_config,
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
            memory_profiler.log_milestone("optuna_phase_start")
            if optuna_maneuvers:
                optuna_result = run_optuna_phase(run_manager, args, optuna_maneuvers)
                results["optuna_phase"] = "completed"
                results["configs_used"].append(optuna_result["config_path"])
                results["optuna_results"] = optuna_result["results"]
                memory_profiler.log_milestone("optuna_phase_completed")
            else:
                logger.warning("⚠️ No Optuna data available, skipping optimization")
                results["optuna_phase"] = "skipped"
        else:
            logger.info("⏭️ Skipping Optuna optimization phase")
            results["optuna_phase"] = "skipped"

        # Phase 2: COCO Ground Truth Validation (AFTER Optuna to use optimized parameters)
        coco_results = None
        if not args.skip_comparison and not args.optuna_only:
            # Run COCO validation for ground truth PCK scores using optimized parameters (if available)
            coco_annotations_path = "data/SD_01_SURF_FOOTAGE_SOURCE/03_THIRD_PARTY_DATA_SETS/COCO_2017_annotations/person_keypoints_val2017.json"

            if Path(coco_annotations_path).exists():
                if results["optuna_phase"] == "completed":
                    logger.info(
                        "🧪 COCO annotations found, running ground truth validation with optimized parameters..."
                    )
                else:
                    logger.info(
                        "🧪 COCO annotations found, running ground truth validation with default parameters..."
                    )

                coco_validation_result = run_coco_validation_phase(
                    run_manager, args, coco_annotations_path
                )
                results["coco_validation_phase"] = "completed"
                results["configs_used"].append(coco_validation_result["config_path"])
                results["coco_validation_results"] = coco_validation_result["results"]
                results["coco_validation_config"] = {
                    "max_images_requested": args.coco_images,
                    "annotations_path": coco_annotations_path,
                }
                coco_results = coco_validation_result["results"]
            else:
                logger.warning(
                    f"⚠️ COCO annotations not found at {coco_annotations_path}"
                )
                logger.warning("⏭️ Skipping COCO validation phase")
                results["coco_validation_phase"] = "skipped"

        # Phase 3: Production Dataset Comparison (Multi-step)
        if not args.skip_comparison and not args.optuna_only:
            memory_profiler.log_milestone("comparison_phase_start")
            if comparison_maneuvers:
                # Step 3a: Run individual model evaluations
                logger.info("📊 Step 3a: Running individual model evaluations...")
                comparison_result = run_comparison_phase(
                    run_manager,
                    args,
                    comparison_maneuvers,
                    visualization_manifest_path,
                )
                memory_profiler.log_milestone("individual_model_evaluations_completed")

                # Step 3b: Run consensus evaluation to add relative PCK metrics
                logger.info("🎯 Step 3b: Calculating consensus-based metrics...")
                consensus_result = run_consensus_phase(
                    run_manager, args, comparison_maneuvers
                )

                # Step 3c: Merge consensus results into comparison results
                if consensus_result and comparison_result:
                    merged_results = merge_consensus_with_comparison(
                        comparison_result["results"], consensus_result
                    )
                    comparison_result["results"] = merged_results
                    logger.info("✅ Merged consensus metrics with comparison results")

                    # CRITICAL FIX: Log consensus metrics to MLflow AFTER merging
                    import mlflow

                    for model_name, model_result in merged_results.items():
                        # Find and restart the MLflow run for this model to add consensus metrics
                        mlflow.set_tracking_uri(str(run_manager.mlflow_dir))

                        # Get the experiment - use pattern matching since name is timestamped
                        # Look for experiment like: surf_pose_production_comparison_20250720_193441
                        experiments = mlflow.search_experiments()
                        experiment = None
                        for exp in experiments:
                            if exp.name and exp.name.startswith(
                                "surf_pose_production_comparison"
                            ):
                                experiment = exp
                                break
                        if experiment:
                            # Find the run for this model
                            runs = mlflow.search_runs(
                                experiment_ids=[experiment.experiment_id],
                                filter_string=f"params.model_name = '{model_name}'",
                            )

                            if not runs.empty:
                                run_id = runs.iloc[0]["run_id"]

                                # Log consensus metrics to the existing run
                                with mlflow.start_run(run_id=run_id):
                                    for metric_name, value in model_result.items():
                                        if metric_name.startswith(
                                            "pose_consensus_"
                                        ) and isinstance(value, (int, float)):
                                            if hasattr(value, "item"):
                                                value = value.item()
                                            mlflow.log_metric(metric_name, value)
                                            logger.debug(
                                                f"Logged consensus metric {metric_name}: {value} for {model_name}"
                                            )

                                logger.info(
                                    f"✅ Logged consensus metrics to MLflow for {model_name}"
                                )
                            else:
                                logger.warning(
                                    f"⚠️ Could not find MLflow run for {model_name} to log consensus metrics"
                                )
                        else:
                            logger.warning(
                                "⚠️ Could not find comparison experiment to log consensus metrics"
                            )
                elif not consensus_result:
                    logger.error(
                        "❌ Consensus evaluation failed - no consensus metrics will be available"
                    )
                    logger.error(
                        "   This means prediction files from individual model evaluation were not found"
                    )
                    logger.error(f"   Available models: {args.models}")
                    logger.error(
                        f"   Prediction directory: {run_manager.predictions_dir}"
                    )
                    logger.error(
                        f"   Prediction files exist: {list(run_manager.predictions_dir.iterdir()) if run_manager.predictions_dir.exists() else 'None'}"
                    )
                    logger.error(
                        "   Consensus evaluation requires cached prediction files - no inference will be re-run"
                    )
                else:
                    logger.warning(
                        "⚠️ Comparison result missing - cannot merge consensus metrics"
                    )

                results["comparison_phase"] = "completed"
                results["configs_used"].append(comparison_result["config_path"])
                results["comparison_results"] = comparison_result["results"]
                memory_profiler.log_milestone("comparison_phase_completed")

                # Generate summary report
                if not generate_summary_report(
                    run_manager, args.models, args.max_clips
                ):
                    logger.error("❌ Failed at summary generation phase")
                    success = False
                memory_profiler.log_milestone("summary_report_generated")
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
            print(f"   Phase 1 - Optuna Optimization: {results['optuna_phase']}")

            coco_phase_status = results.get("coco_validation_phase", "skipped")
            if coco_phase_status == "completed":
                if results["optuna_phase"] == "completed":
                    coco_desc = f"{coco_phase_status} (with optimized params)"
                else:
                    coco_desc = f"{coco_phase_status} (with default params)"
            else:
                coco_desc = coco_phase_status
            print(f"   Phase 2 - COCO Validation: {coco_desc}")

            print(f"   Phase 3 - Production Comparison: {results['comparison_phase']}")
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
        memory_profiler.log_milestone("evaluation_failed")
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

    finally:
        # Stop memory profiling and generate final report
        try:
            memory_profiler.log_milestone("evaluation_finished")
            final_memory_stats = memory_profiler.stop_profiling()

            # Log final memory statistics to MLflow
            try:
                import mlflow

                # Check if MLflow has an active run
                active_run = mlflow.active_run()
                if active_run is None:
                    logger.debug(
                        "No active MLflow run - skipping final memory stats logging"
                    )
                elif final_memory_stats and "statistics" in final_memory_stats:
                    stats = final_memory_stats["statistics"]
                    mlflow.log_metric(
                        "final_memory_peak_mb", stats["process_memory"]["peak_mb"]
                    )
                    mlflow.log_metric(
                        "final_memory_increase_mb",
                        stats["process_memory"]["increase_from_start_mb"],
                    )
                    mlflow.log_metric(
                        "final_cpu_peak_percent", stats["cpu"]["peak_percent"]
                    )
                    if "gpu" in stats:
                        mlflow.log_metric("final_gpu_peak_mb", stats["gpu"]["peak_mb"])

                    # Log memory efficiency analysis
                    if "analysis" in final_memory_stats:
                        analysis = final_memory_stats["analysis"]
                        mlflow.log_metric(
                            "memory_growth_rate_mb_per_sec",
                            analysis["memory_growth_rate_mb_per_second"],
                        )
                        mlflow.log_param(
                            "memory_efficiency", analysis["memory_efficiency"]
                        )
                        mlflow.log_param(
                            "potential_memory_leak", analysis["potential_memory_leak"]
                        )
            except Exception as e:
                logger.debug(
                    f"Failed to log final memory stats to MLflow (normal if no experiment active): {e}"
                )

            logger.info(
                f"📊 Memory profiling completed. Peak memory: {final_memory_stats['statistics']['process_memory']['peak_mb']:.1f}MB"
            )
        except Exception as e:
            logger.error(f"Failed to stop memory profiling: {e}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
