#!/usr/bin/env python3
"""
Core Pose Evaluation Logic
Handles model evaluation, metrics calculation, and result generation
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json
from datetime import datetime

import torch
import numpy as np
import mlflow

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

# Import model wrappers when available
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


class PoseEvaluator:
    """Core pose estimation evaluation functionality"""

    def __init__(self, config: Dict, video_format_override: str = None):
        """Initialize evaluator with configuration"""
        self.config = config
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

        # Run manager for output paths (set later if available)
        self.run_manager = None

        # Initialize prediction file handler
        self._setup_prediction_handler()

        # Model registry
        self.model_registry = self._setup_model_registry()

    def _setup_device(self) -> str:
        """Setup compute device with CUDA-first priority"""
        device = self.config.get("device", "auto")

        if device == "auto":
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

    def _setup_prediction_handler(self):
        """Setup prediction file handler"""
        prediction_config = self.config.get("output", {}).get("predictions", {})
        if prediction_config.get("enabled", True):
            shared_storage_path = prediction_config.get("shared_storage_path")
            if shared_storage_path:
                prediction_base_path = shared_storage_path
                logging.info(f"Using shared prediction storage: {prediction_base_path}")
            else:
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

        return registry

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.model_registry.keys())

    def _get_video_format(self) -> str:
        """Get video format to use, with command-line override support"""
        if self.video_format_override:
            return self.video_format_override
        return self.config["data_source"]["video_clips"].get("input_format", "h264")

    def evaluate_models(
        self, models: List[str], max_clips: Optional[int] = None
    ) -> Dict:
        """Evaluate multiple models with standard configuration"""
        logging.info("Starting model evaluation")

        # Load dataset
        video_format = self._get_video_format()
        logging.info(f"Using video format: {video_format}")
        maneuvers = self.data_loader.load_maneuvers(
            max_clips=max_clips, video_format=video_format
        )
        logging.info(f"Loaded {len(maneuvers)} maneuvers for evaluation")

        results = {}
        for model_name in models:
            if model_name not in self.model_registry:
                logging.warning(f"Model {model_name} not available, skipping")
                continue

            logging.info(f"Evaluating {model_name}")
            model_results = self._evaluate_single_model(model_name, maneuvers)
            results[model_name] = model_results

        return results

    def _evaluate_single_model(self, model_name: str, maneuvers: List) -> Dict:
        """Evaluate a single pose estimation model"""
        # Load model configuration
        model_config_path = f"configs/model_configs/{model_name}.yaml"
        if Path(model_config_path).exists():
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
        else:
            model_config = {}

        # Load best parameters if available
        best_params = self._load_best_params_from_optuna(model_name)
        final_config = {**model_config, **best_params}

        # Initialize model
        model_class = self.model_registry[model_name]
        model = model_class(device=self.device, **final_config)

        # Run evaluation
        result = self._evaluate_model_internal(model, model_name, maneuvers)

        # Generate visualizations if enabled
        if self.config.get("output", {}).get("visualization", {}).get("enabled", False):
            self._create_sample_visualizations(model, model_name, maneuvers[:3])

        return result

    def _load_best_params_from_optuna(self, model_name: str) -> Dict:
        """Load best parameters from Optuna optimization results"""
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
                logging.warning(f"No best parameters found for {model_name}")

            return model_params

        except Exception as e:
            if load_config.get("fallback_to_defaults", True):
                logging.warning(f"Failed to load best parameters: {e}, using defaults")
                return {}
            else:
                raise

    def _evaluate_model_internal(self, model, model_name: str, maneuvers: List) -> Dict:
        """Internal method to evaluate a model"""
        all_pose_metrics = []
        all_performance_metrics = []
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
            aggregated_metrics.update(
                {
                    "successful_maneuvers": successful_maneuvers,
                    "failed_maneuvers": failed_maneuvers,
                    "total_maneuvers": len(maneuvers),
                }
            )
            return aggregated_metrics
        else:
            return {"error": "No successful evaluations"}

    def _process_video_maneuver(
        self, model: BasePoseModel, maneuver, model_name: str = None
    ) -> Dict:
        """Process a single maneuver with pose estimation"""
        frames = self.data_loader.load_video_frames(maneuver)

        inference_times = []
        memory_usage = []
        pose_results = []
        total_frames = len(frames)

        for i, frame in enumerate(frames):
            if i % 5 == 0 or i == total_frames - 1:
                print(
                    f"\r      Frame {i+1}/{total_frames} [{100*(i+1)//total_frames:3d}%]",
                    end="",
                    flush=True,
                )

            start_time = time.time()
            pose_result = model.predict(frame)
            inference_time = time.time() - start_time

            inference_times.append(inference_time)
            if self.device == "cuda":
                memory_usage.append(torch.cuda.memory_allocated())
            pose_results.append(pose_result)

        # Generate prediction files if enabled
        if self.prediction_handler and model_name:
            self._generate_prediction_file(maneuver, model_name, pose_results)

        # Calculate metrics
        pose_metrics = {}
        if hasattr(maneuver, "annotation_data") and maneuver.annotation_data:
            pose_metrics = self.pose_metrics.calculate_metrics(
                pose_results, [maneuver.annotation_data]
            )

        performance_metrics = {
            "avg_inference_time": np.mean(inference_times),
            "fps": 1.0 / np.mean(inference_times),
            "max_memory_usage": max(memory_usage) if memory_usage else 0,
            "total_frames": len(frames),
            "maneuver_type": maneuver.maneuver_type,
            "maneuver_duration": maneuver.duration,
        }

        return {"pose": pose_metrics, "performance": performance_metrics}

    def _generate_prediction_file(self, maneuver, model_name: str, pose_results: List):
        """Generate standardized prediction file"""
        try:
            logging.debug(f"Generating prediction file for {maneuver.maneuver_id}")

            keypoint_format = get_keypoint_format_for_model(model_name)
            keypoint_names = get_keypoint_names_for_model(model_name)

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

            maneuver_prediction = self.prediction_handler.create_maneuver_prediction(
                maneuver=maneuver,
                model_name=model_name,
                model_config={},
                keypoint_format=keypoint_format,
                keypoint_names=keypoint_names,
                frame_predictions=frame_predictions,
            )

            prediction_file_path = self.prediction_handler.save_prediction_file(
                maneuver_prediction
            )
            logging.info(f"Saved prediction file: {Path(prediction_file_path).name}")

        except Exception as e:
            logging.error(f"Failed to generate prediction file: {e}")

    def _aggregate_metrics(
        self, pose_metrics: List[Dict], performance_metrics: List[Dict]
    ) -> Dict:
        """Aggregate metrics across all maneuvers"""
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
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    aggregated[f"perf_{key}_mean"] = np.mean(numeric_values)
                    aggregated[f"perf_{key}_std"] = np.std(numeric_values)

        return aggregated

    def _create_sample_visualizations(self, model, model_name: str, maneuvers: List):
        """Create sample visualization videos"""
        from utils.pose_video_visualizer import PoseVideoVisualizer

        logging.info(f"Creating sample visualizations for {model_name}")

        # Get visualization config
        viz_config = self.config.get("output", {}).get("visualization", {})
        max_examples = viz_config.get("max_examples_per_model", 3)

        # Get output directory from run manager if available
        viz_dir = None
        if hasattr(self, "run_manager") and self.run_manager:
            viz_dir = self.run_manager.visualizations_dir
        else:
            # Fallback to config path
            viz_dir = Path(viz_config.get("shared_storage_path", "./visualizations"))

        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualizer
        encoding_config = viz_config.get("encoding", {})
        visualizer = PoseVideoVisualizer(encoding_config)

        created_count = 0
        for i, maneuver in enumerate(maneuvers[:max_examples]):
            try:
                # Generate pose predictions for this maneuver
                pose_results = self._generate_pose_predictions(model, maneuver)

                if not pose_results:
                    logging.warning(
                        f"No pose results for maneuver {i}, skipping visualization"
                    )
                    continue

                # Create output filename
                video_stem = Path(maneuver["video_path"]).stem
                output_filename = f"{model_name}_{video_stem}_visualization.mp4"
                output_path = viz_dir / output_filename

                # Create visualization video
                success = visualizer.create_pose_visualization_video(
                    video_path=maneuver["video_path"],
                    pose_results=pose_results,
                    output_path=str(output_path),
                    model_name=model_name,
                )

                if success:
                    created_count += 1
                    logging.info(
                        f"Created visualization {created_count}: {output_filename}"
                    )
                else:
                    logging.warning(f"Failed to create visualization for {video_stem}")

            except Exception as e:
                logging.error(f"Error creating visualization for maneuver {i}: {e}")

        logging.info(f"Created {created_count} visualization videos for {model_name}")

    def _generate_pose_predictions(self, model, maneuver: Dict) -> List[Dict]:
        """Generate pose predictions for a maneuver for visualization"""
        try:
            # Load video frames using the data loader
            frames = self.data_loader.load_video_frames(maneuver)

            pose_results = []
            for i, frame in enumerate(frames):
                # Get pose prediction for this frame
                pose_result = model.predict(frame)

                # Convert to format expected by visualizer
                if pose_result:
                    # Add frame index to the result
                    if isinstance(pose_result, dict):
                        pose_result["frame_id"] = i
                    pose_results.append(pose_result)

            logging.info(
                f"Generated {len(pose_results)} pose predictions for visualization"
            )
            return pose_results

        except Exception as e:
            logging.error(f"Failed to generate pose predictions: {e}")
            return []
