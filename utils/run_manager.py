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
        """Initialize run manager with timestamp-based organization

        Args:
            run_name: Optional custom run name
            max_clips: Number of clips (used for naming)
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create descriptive run name
        if run_name:
            self.run_name = f"{self.timestamp}_{run_name}"
        else:
            clips_suffix = f"_{max_clips}clips" if max_clips else "_full"
            self.run_name = f"{self.timestamp}{clips_suffix}"

        # Use shared POSE directory for multi-machine sync
        self.base_results_dir = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results"
        )
        self.run_dir = self.base_results_dir / "runs" / self.run_name

        # Run-specific subdirectories
        self.mlflow_dir = self.run_dir / "mlruns"
        self.predictions_dir = self.run_dir / "predictions"
        self.visualizations_dir = self.run_dir / "visualizations"
        self.best_params_dir = self.run_dir / "best_params"
        self.reports_dir = self.run_dir / "reports"

        # Create directories
        self._create_directories()

        # Create run metadata
        self._create_run_metadata()

        logger.info(f"🗂️ Created organized run: {self.run_name}")
        logger.info(f"📁 Run directory: {self.run_dir}")
        logger.info(f"🔗 Shared results location: {self.base_results_dir}")

    def _create_directories(self):
        """Create all run-specific directories"""
        directories = [
            self.run_dir,
            self.mlflow_dir,
            self.predictions_dir,
            self.visualizations_dir,
            self.best_params_dir,
            self.reports_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _create_run_metadata(self):
        """Create metadata file for this run"""
        metadata = {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "created_at": datetime.now().isoformat(),
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
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results"
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

    def get_predictions_config(self) -> Dict[str, Any]:
        """Get predictions configuration for this run"""
        return {
            "enabled": True,
            "base_path": str(self.predictions_dir),
            "shared_storage_path": str(self.predictions_dir),
            "format": "json",
            "include_metadata": True,
            "compress": False,
        }

    def get_visualizations_config(self) -> Dict[str, Any]:
        """Get visualizations configuration for this run"""
        return {
            "enabled": True,
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
        config["output"]["predictions"] = self.get_predictions_config()
        config["output"]["visualization"] = self.get_visualizations_config()
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

    @staticmethod
    def list_previous_runs() -> List[Dict[str, Any]]:
        """List all previous runs with metadata"""
        shared_results = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results"
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
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results"
        )
        runs_dir = shared_results / "runs"

        for run_metadata in runs_to_delete:
            run_name = run_metadata["run_name"]
            run_path = runs_dir / run_name

            if run_path.exists():
                shutil.rmtree(run_path)
                logger.info(f"🗑️ Deleted old run: {run_name}")

        logger.info(f"🧹 Cleanup complete: kept {keep_last} most recent runs")

    def create_run_summary(self, results: Dict[str, Any]):
        """Create a summary of the run results"""
        summary = {
            "run_info": {
                "name": self.run_name,
                "timestamp": self.timestamp,
                "completed_at": datetime.now().isoformat(),
            },
            "results": convert_numpy_types(results),
            "directories": {
                "mlflow": str(self.mlflow_dir),
                "predictions": str(self.predictions_dir),
                "visualizations": str(self.visualizations_dir),
                "best_params": str(self.best_params_dir),
            },
        }

        summary_file = self.run_dir / "run_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📊 Created run summary: {summary_file}")

    def print_run_info(self):
        """Print information about this run"""
        print(f"\n🗂️ Run Information:")
        print(f"   Name: {self.run_name}")
        print(f"   Timestamp: {self.timestamp}")
        print(f"   Directory: {self.run_dir}")
        print(f"   MLflow: {self.mlflow_dir}")
        print(f"   Predictions: {self.predictions_dir}")
        print(f"   Visualizations: {self.visualizations_dir}")

    @staticmethod
    def get_shared_mlflow_uri() -> str:
        """Get the URI for accessing all MLflow experiments in the shared directory"""
        shared_results = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results"
        )
        return f"file://{shared_results.absolute()}"

    @staticmethod
    def list_all_experiments() -> List[Dict[str, Any]]:
        """List all MLflow experiments across all runs"""
        shared_results = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results"
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
