"""
Run Management Utility for Timestamp-Based Organization
Automatically organizes runs into separate folders with timestamps
"""

import os
import json
import yaml
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class RunManager:
    """Manages run-specific organization with timestamps"""

    def __init__(
        self,
        run_name: Optional[str] = None,
        max_clips: Optional[int] = None,
    ):
        """Initialize run manager with organized directory structure"""
        self.max_clips = max_clips

        # Generate timestamp and run name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or "evaluation"

        # Create organized run directory
        self.base_results_dir = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results"
        )
        self.run_dir = self.base_results_dir / "runs" / f"{self.timestamp}_{self.run_name}"

        # Create all subdirectories
        self._create_directories()

        # Create run metadata
        self._create_run_metadata()

        logger.info(f"🏗️ Created organized run directory: {self.run_dir}")

    @property
    def mlflow_dir(self) -> Path:
        """MLflow tracking directory for this run"""
        return self.run_dir / "mlflow"

    @property
    def predictions_dir(self) -> Path:
        """Predictions output directory for this run"""
        return self.run_dir / "predictions"

    @property
    def visualizations_dir(self) -> Path:
        """Visualizations output directory for this run"""
        return self.run_dir / "visualizations"

    @property
    def best_params_dir(self) -> Path:
        """Best parameters directory for this run"""
        return self.run_dir / "best_params"

    @property
    def reports_dir(self) -> Path:
        """Reports directory for this run"""
        return self.run_dir / "reports"

    @property
    def data_selections_dir(self) -> Path:
        """Data selections directory for this run"""
        return self.run_dir / "data_selections"

    @property
    def data_splits_dir(self) -> Path:
        """Data splits directory for this run"""
        return self.run_dir / "data_splits"

    def _create_directories(self):
        """Create all necessary directories for this run"""
        directories = [
            self.run_dir,
            self.mlflow_dir,
            self.predictions_dir,
            self.visualizations_dir,
            self.best_params_dir,
            self.reports_dir,
            self.data_selections_dir,
            self.data_splits_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Created {len(directories)} directories for run")

    def _create_run_metadata(self):
        """Create metadata file for this run"""
        import sys

        metadata = {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "created_at": datetime.now().isoformat(),
            "command_line": " ".join(sys.argv),  # Record exact command used
            "machine_info": {
                "hostname": os.uname().nodename,
                "platform": os.uname().sysname,
                "user": os.environ.get("USER", "unknown"),
            },
            "directories": {
                "run_dir": str(self.run_dir),
                "mlflow_dir": str(self.mlflow_dir),
                "predictions_dir": str(self.predictions_dir),
                "visualizations_dir": str(self.visualizations_dir),
                "best_params_dir": str(self.best_params_dir),
                "reports_dir": str(self.reports_dir),
                "data_selections_dir": str(self.data_selections_dir),
                "data_splits_dir": str(self.data_splits_dir),
            },
        }

        metadata_file = self.run_dir / "run_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"📋 Created run metadata: {metadata_file}")

    @staticmethod
    def detect_existing_results() -> bool:
        """Check if there are existing results in the shared directory"""
        shared_results = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results"
        )
        return (shared_results / "runs").exists() and any(
            (shared_results / "runs").iterdir()
        )

    def get_mlflow_config(self, base_experiment_name: str) -> Dict[str, Any]:
        """Get MLflow configuration for this run"""
        return {
            "enabled": True,
            "tracking_uri": f"file://{self.mlflow_dir.absolute()}",
            "experiment_name": f"{base_experiment_name}_{self.timestamp}",
            "artifact_location": str(self.mlflow_dir / "artifacts"),
            "run_name": self.run_name,
            "tags": {
                "run_timestamp": self.timestamp,
                "run_name": self.run_name,
                "machine": os.uname().nodename,
            },
        }

    def get_predictions_config(self, enabled: bool = True) -> Dict[str, Any]:
        """Get predictions configuration for this run"""
        return {
            "enabled": enabled,
            "base_path": str(self.predictions_dir),
            "shared_storage_path": str(self.predictions_dir),
            "format": "json",
            "include_metadata": True,
            "compress": False,
        }

    def get_visualizations_config(self, enabled: bool = True) -> Dict[str, Any]:
        """Get visualizations configuration for this run"""
        return {
            "enabled": enabled,
            "save_overlay_videos": True,
            "save_keypoint_plots": True,
            "save_comparison_plots": True,
            "max_examples_per_model": 10,
            "shared_storage_path": str(self.visualizations_dir),
            "encoding": {
                "format": "h264",
                "quality": {"crf": 23, "preset": "fast"},
                "pixel_format": "yuv420p",
                "audio": {"enabled": True, "codec": "copy"},
                "container": "mp4",
            },
        }

    def get_best_params_config(self) -> Dict[str, Any]:
        """Get best parameters configuration for this run"""
        return {
            "enabled": True,
            "save_path": str(self.best_params_dir),
            "format": "yaml",
        }

    def create_config_for_phase(
        self, phase: str, base_config_path: str, max_clips: Optional[int] = None
    ) -> str:
        """Create run-specific config file for a phase"""
        # Load base config
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Config already has proper data_source structure - no translation needed

        # Ensure video format stays consistent with source config
        if "data_source" in config and "video_clips" in config["data_source"]:
            # Preserve the original input_format from source config
            original_format = config["data_source"]["video_clips"].get(
                "input_format", "h264"
            )
            # Force string type to prevent YAML conversion issues
            config["data_source"]["video_clips"]["input_format"] = str(original_format)

            # Debug logging to track format preservation
            logger.info(f"🎥 Preserving video format: {original_format}")

            # Explicit check for macOS configs to force h264
            if "macos" in base_config_path.lower() and original_format != "h264":
                logger.warning(f"⚠️ macOS config had {original_format}, forcing h264")
                config["data_source"]["video_clips"]["input_format"] = "h264"

        # Update with run-specific settings
        config["mlflow"] = self.get_mlflow_config(
            config.get("mlflow", {}).get("experiment_name", "surf_pose_evaluation")
        )

        # Create output section with proper nesting for evaluate_pose_models.py compatibility
        if "output" not in config:
            config["output"] = {}

        # Preserve original enabled/disabled settings from production config
        original_predictions_enabled = (
            config.get("output", {}).get("predictions", {}).get("enabled", True)
        )
        original_visualizations_enabled = (
            config.get("output", {}).get("visualization", {}).get("enabled", True)
        )

        config["output"]["predictions"] = self.get_predictions_config(
            enabled=original_predictions_enabled
        )
        config["output"]["visualization"] = self.get_visualizations_config(
            enabled=original_visualizations_enabled
        )
        config["best_params"] = self.get_best_params_config()

        # Update max_clips if provided
        if max_clips is not None:
            if "evaluation" not in config:
                config["evaluation"] = {}
            # Set max_clips in both quick_test and comprehensive_test
            if "quick_test" in config["evaluation"]:
                config["evaluation"]["quick_test"]["num_clips"] = max_clips
            if "comprehensive_test" in config["evaluation"]:
                config["evaluation"]["comprehensive_test"]["num_clips"] = max_clips

        # Create config file in run directory
        config_filename = f"{phase}_config_{self.timestamp}.yaml"
        config_path = self.run_dir / config_filename

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"📄 Created {phase} config: {config_path}")
        return str(config_path)

    def generate_data_selections(
        self,
        config: Dict[str, Any],
        optuna_max_clips: Optional[int] = None,
        comparison_max_clips: Optional[int] = None,
        optuna_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Generate data selections for both phases using DataSelectionManager"""
        from utils.data_selection_manager import DataSelectionManager

        # Initialize data selection manager
        selection_manager = DataSelectionManager(config, self)

        # Generate selections for both phases
        manifest_paths = selection_manager.generate_phase_selections(
            optuna_max_clips=optuna_max_clips,
            comparison_max_clips=comparison_max_clips,
            optuna_config=optuna_config,
        )

        # Update run metadata with selection info
        self._update_run_metadata_with_selections(manifest_paths)

        logger.info(f"🎯 Generated data selections for {len(manifest_paths)} phases")
        return manifest_paths

    def generate_data_splits(
        self,
        config: Dict[str, Any],
        random_seed: Optional[int] = None,
    ) -> Dict[str, str]:
        """Generate and save data splits to the run folder"""
        from data_handling.data_loader import SurfingDataLoader
        import json

        # Initialize data loader
        loader = SurfingDataLoader(config)

        # Load annotations and discover clips
        logger.info("📋 Loading annotations...")
        annotations = loader.load_annotations()

        video_format = config['data_source']['video_clips'].get('input_format', 'h264')
        logger.info(f"🎥 Discovering video clips (format: {video_format})...")
        clips = loader.discover_video_clips(video_format)

        logger.info(f"📊 Found {len(clips)} video clips")

        # Set clips for split creation
        loader.all_clips = clips

        # Create data splits
        logger.info("🔄 Creating data splits...")
        splits = loader.create_data_splits(random_seed)

        # Get split statistics
        stats = splits.get_split_stats()
        logger.info(f"📈 Split statistics: {stats}")

        # Save splits to JSON files in run folder
        split_files = {}
        
        def save_split(clips, split_name):
            split_data = []
            for clip in clips:
                clip_data = {
                    'video_path': str(clip.file_path),
                    'video_id': clip.video_id,
                    'camera': clip.camera,
                    'session': clip.session,
                    'duration': clip.duration,
                    'fps': clip.fps,
                    'width': clip.width,
                    'height': clip.height,
                    'format': clip.format,
                    'zoom_level': clip.zoom_level,
                    'base_clip_id': clip.base_clip_id,
                    'annotations': clip.annotations
                }
                split_data.append(clip_data)
            
            output_file = self.data_splits_dir / f'{split_name}_split.json'
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2, default=str)
            
            logger.info(f"💾 Saved {len(split_data)} clips to {output_file}")
            return str(output_file)

        # Save each split
        split_files['train'] = save_split(splits.train, 'train')
        split_files['val'] = save_split(splits.val, 'val')
        split_files['test'] = save_split(splits.test, 'test')

        # Save split metadata
        split_metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'random_seed': random_seed or config['data_source']['splits'].get('random_seed', 42),
            'video_format': video_format,
            'total_clips': len(clips),
            'split_statistics': stats,
            'split_files': split_files
        }

        metadata_file = self.data_splits_dir / 'splits_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(split_metadata, f, indent=2, default=str)

        logger.info(f"✅ Generated data splits: {split_files}")
        return split_files

    def _update_run_metadata_with_selections(self, manifest_paths: Dict[str, str]):
        """Update run metadata file with data selection information

        Args:
            manifest_paths: Dictionary mapping phase names to manifest paths
        """
        metadata_file = self.run_dir / "run_metadata.json"

        try:
            # Load existing metadata
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Add data selection information
            metadata["data_selections"] = {
                "enabled": True,
                "manifest_files": manifest_paths,
                "selection_dir": str(self.data_selections_dir),
            }

            # Save updated metadata
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"📋 Updated run metadata with data selection info")

        except Exception as e:
            logger.warning(f"Failed to update run metadata with selections: {e}")

    def get_data_selection_manifest(self, phase: str) -> Optional[str]:
        """Get path to data selection manifest for a phase

        Args:
            phase: Phase name (optuna/comparison/visualization)

        Returns:
            Path to manifest file or None if not found
        """
        return self.data_selection_manifests.get(phase)

    def print_data_selection_summary(self):
        """Print summary of data selections for this run"""
        if not self.data_selection_manifests:
            print("   📊 Data Selections: None generated")
            return

        print(f"   📊 Data Selections:")
        for phase, manifest_path in self.data_selection_manifests.items():
            try:
                from utils.data_selection_manager import DataSelectionManager

                manager = DataSelectionManager({}, run_manager=self)
                summary = manager.get_manifest_summary(manifest_path)
                # Extract key stats from summary
                lines = summary.split("\n")
                clips_line = next((line for line in lines if "• Clips:" in line), "")
                maneuvers_line = next(
                    (line for line in lines if "• Maneuvers:" in line), ""
                )
                print(
                    f"     • {phase.title()}: {clips_line.strip().replace('• ', '')} | {maneuvers_line.strip().replace('• ', '')}"
                )
            except Exception as e:
                print(
                    f"     • {phase.title()}: manifest available ({Path(manifest_path).name})"
                )

    @staticmethod
    def list_previous_runs() -> List[Dict[str, Any]]:
        """List all previous runs with metadata"""
        shared_results = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results"
        )
        runs_dir = shared_results / "runs"

        if not runs_dir.exists():
            return []

        runs = []
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                metadata_file = run_dir / "run_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        runs.append(metadata)
                    except (json.JSONDecodeError, KeyError):
                        # Fallback for runs without proper metadata
                        runs.append({"run_name": run_dir.name, "timestamp": "unknown"})

        return runs

    def cleanup_old_runs(self, keep_last: int = 5):
        """Clean up old runs, keeping only the most recent ones"""
        runs = self.list_previous_runs()

        if len(runs) <= keep_last:
            logger.info(f"🗂️ Only {len(runs)} runs found, no cleanup needed")
            return

        runs_to_delete = runs[keep_last:]
        shared_results = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results"
        )
        runs_dir = shared_results / "runs"

        for run_metadata in runs_to_delete:
            run_name = run_metadata["run_name"]
            run_path = runs_dir / run_name

            if run_path.exists():
                shutil.rmtree(run_path)
                logger.info(f"🗑️ Deleted old run: {run_name}")

        logger.info(f"🧹 Cleanup complete: kept {keep_last} most recent runs")

    def create_run_summary(
        self, results: Dict[str, Any], memory_stats: Dict[str, Any] = None
    ):
        """Create a comprehensive summary of the run results including memory and performance metrics"""
        import platform
        import psutil

        # Base summary structure
        summary = {
            "run_info": {
                "name": self.run_name,
                "timestamp": self.timestamp,
                "completed_at": datetime.now().isoformat(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            },
            "results": convert_numpy_types(results),
            "directories": {
                "mlflow": str(self.mlflow_dir),
                "predictions": str(self.predictions_dir),
                "visualizations": str(self.visualizations_dir),
                "best_params": str(self.best_params_dir),
            },
        }

        # Add comprehensive memory and performance statistics
        if memory_stats:
            duration_seconds = memory_stats.get("statistics", {}).get(
                "duration_seconds", 0
            )

            # Convert duration to human-readable format
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            seconds = int(duration_seconds % 60)

            if hours > 0:
                duration_human = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_human = f"{minutes}m {seconds}s"
            else:
                duration_human = f"{seconds}s"

            summary["memory_profiling"] = {
                "enabled": True,
                "duration_seconds": duration_seconds,
                "duration_human_readable": duration_human,
                "snapshots_collected": memory_stats.get("statistics", {}).get(
                    "snapshots_count", 0
                ),
                "process_memory": {
                    "peak_mb": memory_stats.get("statistics", {})
                    .get("process_memory", {})
                    .get("peak_mb", 0),
                    "mean_mb": memory_stats.get("statistics", {})
                    .get("process_memory", {})
                    .get("mean_mb", 0),
                    "increase_from_start_mb": memory_stats.get("statistics", {})
                    .get("process_memory", {})
                    .get("increase_from_start_mb", 0),
                    "std_mb": memory_stats.get("statistics", {})
                    .get("process_memory", {})
                    .get("std_mb", 0),
                },
                "cpu": {
                    "peak_percent": memory_stats.get("statistics", {})
                    .get("cpu", {})
                    .get("peak_percent", 0),
                    "mean_percent": memory_stats.get("statistics", {})
                    .get("cpu", {})
                    .get("mean_percent", 0),
                    "std_percent": memory_stats.get("statistics", {})
                    .get("cpu", {})
                    .get("std_percent", 0),
                },
                "gpu": memory_stats.get("statistics", {}).get("gpu", {}),
                "analysis": memory_stats.get("analysis", {}),
                "efficiency": memory_stats.get("analysis", {}).get(
                    "memory_efficiency", "unknown"
                ),
                "potential_memory_leak": memory_stats.get("analysis", {}).get(
                    "potential_memory_leak", False
                ),
            }
        else:
            summary["memory_profiling"] = {"enabled": False}

        # Add system information
        memory_info = psutil.virtual_memory()
        summary["system_info"] = {
            "total_ram_gb": round(memory_info.total / (1024**3), 2),
            "available_ram_gb": round(memory_info.available / (1024**3), 2),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
        }

        # Try to add GPU information
        try:
            import torch

            if torch.cuda.is_available():
                summary["system_info"]["gpu"] = {
                    "available": True,
                    "device_name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                    ),
                }
            else:
                summary["system_info"]["gpu"] = {"available": False}
        except Exception:
            summary["system_info"]["gpu"] = {"available": "unknown"}

        # Add high-level performance metrics summary
        if "optuna_results" in results:
            summary["performance_summary"] = self._extract_performance_summary(
                results["optuna_results"]
            )

        # Add comparison results summary if available
        if "comparison_results" in results:
            summary["comparison_summary"] = self._extract_comparison_summary(
                results["comparison_results"]
            )

        # Add COCO validation summary if available
        if "coco_validation_results" in results:
            summary["coco_summary"] = self._extract_coco_summary(
                results["coco_validation_results"]
            )

        # Add data selection summary
        if self.data_selection_manifests:
            summary["data_selections"] = {}
            for phase, manifest_path in self.data_selection_manifests.items():
                try:
                    with open(manifest_path, "r") as f:
                        manifest_data = json.load(f)
                    summary["data_selections"][phase] = {
                        "total_clips": len(manifest_data.get("clips", [])),
                        "total_maneuvers": len(
                            set(
                                clip.get("maneuver_id")
                                for clip in manifest_data.get("clips", [])
                            )
                        ),
                        "cameras": list(
                            set(
                                clip.get("camera")
                                for clip in manifest_data.get("clips", [])
                            )
                        ),
                        "manifest_path": str(manifest_path),
                    }
                except Exception as e:
                    summary["data_selections"][phase] = {
                        "error": str(e),
                        "manifest_path": str(manifest_path),
                    }

        summary_file = self.run_dir / "run_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📊 Created comprehensive run summary: {summary_file}")

    def _extract_performance_summary(self, optuna_results: Dict) -> Dict:
        """Extract high-level performance metrics from Optuna results"""
        summary = {}

        for model_name, model_data in optuna_results.items():
            if isinstance(model_data, dict):
                summary[model_name] = {
                    "best_trial": model_data.get("best_trial_number", "unknown"),
                    "best_score": model_data.get("best_score", 0),
                    "successful_maneuvers": model_data.get("successful_maneuvers", 0),
                    "failed_maneuvers": model_data.get("failed_maneuvers", 0),
                    "avg_fps": model_data.get("perf_fps_mean", 0),
                    "avg_inference_time_ms": (
                        model_data.get("perf_avg_inference_time_mean", 0) * 1000
                        if model_data.get("perf_avg_inference_time_mean")
                        else 0
                    ),
                    "avg_memory_mb": model_data.get("perf_avg_memory_usage_mean", 0),
                    "max_memory_mb": model_data.get("perf_max_memory_usage_mean", 0),
                    "model_size_mb": model_data.get("perf_model_size_mb_mean", 0),
                    "avg_cpu_percent": model_data.get(
                        "perf_avg_cpu_utilization_mean", 0
                    ),
                }

        return summary

    def _extract_comparison_summary(self, comparison_results: Dict) -> Dict:
        """Extract high-level metrics from comparison results"""
        summary = {}

        for model_name, model_data in comparison_results.items():
            if isinstance(model_data, dict):
                summary[model_name] = {
                    "detection_f1": model_data.get("pose_detection_f1_mean", 0),
                    "consensus_pck_error": model_data.get(
                        "pose_consensus_pck_error_mean", 0
                    ),
                    "consensus_pck_0.2": model_data.get(
                        "pose_consensus_pck_0.2_mean", 0
                    ),
                    "consensus_coverage": model_data.get(
                        "pose_consensus_coverage_mean", 0
                    ),
                    "fps_mean": model_data.get("perf_fps_mean", 0),
                    "inference_time_ms": (
                        model_data.get("perf_avg_inference_time_mean", 0) * 1000
                        if model_data.get("perf_avg_inference_time_mean")
                        else 0
                    ),
                    "memory_efficiency": model_data.get(
                        "perf_memory_efficiency_mean", 0
                    ),
                    "successful_maneuvers": model_data.get("successful_maneuvers", 0),
                    "failed_maneuvers": model_data.get("failed_maneuvers", 0),
                }

        return summary

    def _extract_coco_summary(self, coco_results: Dict) -> Dict:
        """Extract high-level metrics from COCO validation results"""
        summary = {}

        for model_name, model_data in coco_results.items():
            if isinstance(model_data, dict):
                summary[model_name] = {
                    "pck_0.2": model_data.get("coco_pck_0.2", 0),
                    "pck_0.5": model_data.get("coco_pck_0.5", 0),
                    "pck_error_mean": model_data.get("coco_pck_error_mean", 0),
                    "detection_f1": model_data.get("coco_detection_f1", 0),
                    "fps_mean": model_data.get("coco_fps_mean", 0),
                    "inference_time_ms": model_data.get("coco_inference_time_ms", 0),
                    "images_processed": model_data.get(
                        "coco_total_images_processed", 0
                    ),
                }

        return summary

    def print_run_info(self):
        """Print information about this run"""
        print(f"\n🗂️ Run Information:")
        print(f"   Name: {self.run_name}")
        print(f"   Timestamp: {self.timestamp}")
        print(f"   Directory: {self.run_dir}")
        print(f"   MLflow: {self.mlflow_dir}")
        print(f"   Predictions: {self.predictions_dir}")
        print(f"   Visualizations: {self.visualizations_dir}")
        self.print_data_selection_summary()

    @staticmethod
    def get_shared_mlflow_uri() -> str:
        """Get the URI for accessing all MLflow experiments in the shared directory"""
        shared_results = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results"
        )
        return f"file://{shared_results.absolute()}"

    @staticmethod
    def list_all_experiments() -> List[Dict[str, Any]]:
        """List all MLflow experiments across all runs"""
        shared_results = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results"
        )
        runs_dir = shared_results / "runs"

        if not runs_dir.exists():
            return []

        experiments = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                mlflow_dir = run_dir / "mlruns"
                if mlflow_dir.exists():
                    # Find experiment directories (numbered folders)
                    for exp_dir in mlflow_dir.iterdir():
                        if exp_dir.is_dir() and exp_dir.name.isdigit():
                            meta_file = exp_dir / "meta.yaml"
                            if meta_file.exists():
                                try:
                                    with open(meta_file, "r") as f:
                                        meta = yaml.safe_load(f)
                                    experiments.append(
                                        {
                                            "experiment_id": exp_dir.name,
                                            "name": meta.get("name", "unknown"),
                                            "run_name": run_dir.name,
                                            "mlflow_dir": str(mlflow_dir),
                                            "tracking_uri": f"file://{mlflow_dir.absolute()}",
                                        }
                                    )
                                except (yaml.YAMLError, KeyError):
                                    pass

        return experiments
