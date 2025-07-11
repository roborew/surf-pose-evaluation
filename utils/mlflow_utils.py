"""
MLflow utilities for experiment tracking
"""

import os
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from typing import Dict, Any, Optional, List
import json
import tempfile
import shutil
from pathlib import Path
import logging


class MLflowManager:
    """MLflow experiment management utilities"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MLflow manager

        Args:
            config: MLflow configuration dictionary
        """
        self.config = config
        self.tracking_uri = config.get("tracking_uri", "./results/mlruns")
        self.experiment_name = config.get(
            "experiment_name", "pose_estimation_comparison"
        )
        self.artifact_location = config.get("artifact_location", None)

        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name, artifact_location=self.artifact_location
                )
                logging.info(
                    f"Created new MLflow experiment: {self.experiment_name} (ID: {experiment_id})"
                )
            else:
                logging.info(
                    f"Using existing MLflow experiment: {self.experiment_name}"
                )

            mlflow.set_experiment(self.experiment_name)

        except Exception as e:
            logging.error(f"Failed to setup MLflow experiment: {e}")
            raise

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLflow run

        Args:
            run_name: Name for the run
            tags: Additional tags for the run

        Returns:
            MLflow run context manager
        """
        run_tags = {}
        if tags:
            run_tags.update(tags)

        return mlflow.start_run(run_name=run_name, tags=run_tags)

    def log_model_params(self, model, model_name: str):
        """Log model parameters and configuration

        Args:
            model: Model instance
            model_name: Name of the model
        """
        try:
            # Get model info
            model_info = model.get_model_info()

            # Log basic parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", model_info.get("type", "unknown"))
            mlflow.log_param("num_keypoints", model_info.get("num_keypoints", 0))
            mlflow.log_param("device", model.device)

            # Log model-specific configuration
            for key, value in model.model_config.items():
                mlflow.log_param(f"config_{key}", value)

            # Log model info as JSON
            mlflow.log_text(json.dumps(model_info, indent=2), "model_info.json")

        except Exception as e:
            logging.warning(f"Failed to log model parameters: {e}")

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information

        Args:
            dataset_info: Dataset metadata
        """
        try:
            # Log dataset parameters
            mlflow.log_param("num_clips", dataset_info.get("num_clips", 0))
            mlflow.log_param("num_frames", dataset_info.get("num_frames", 0))
            mlflow.log_param(
                "video_format", dataset_info.get("video_format", "unknown")
            )
            mlflow.log_param("dataset_split", dataset_info.get("split", "unknown"))

            # Log zoom distribution if available
            if "zoom_distribution" in dataset_info:
                zoom_dist = dataset_info["zoom_distribution"]
                for zoom_type, count in zoom_dist.items():
                    mlflow.log_param(f"zoom_{zoom_type}_count", count)

            # Log dataset info as artifact
            mlflow.log_text(json.dumps(dataset_info, indent=2), "dataset_info.json")

        except Exception as e:
            logging.warning(f"Failed to log dataset info: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation metrics

        Args:
            metrics: Dictionary of metrics
            step: Step number for time series metrics
        """
        try:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, bool)):
                    mlflow.log_metric(metric_name, value, step=step)

        except Exception as e:
            logging.warning(f"Failed to log metrics: {e}")

    def log_performance_metrics(self, performance_metrics: Dict[str, Any]):
        """Log performance benchmarking results

        Args:
            performance_metrics: Performance measurement results
        """
        try:
            # Log timing metrics
            timing_keys = [
                "mean_inference_time_ms",
                "std_inference_time_ms",
                "min_inference_time_ms",
                "max_inference_time_ms",
                "median_inference_time_ms",
                "fps",
            ]

            for key in timing_keys:
                if key in performance_metrics:
                    mlflow.log_metric(key, performance_metrics[key])

            # Log memory metrics
            memory_keys = [
                k for k in performance_metrics.keys() if "memory" in k.lower()
            ]
            for key in memory_keys:
                if isinstance(performance_metrics[key], (int, float)):
                    mlflow.log_metric(key, performance_metrics[key])

            # Log throughput metrics
            throughput_keys = [
                "samples_per_second",
                "total_samples_processed",
                "total_time_seconds",
            ]

            for key in throughput_keys:
                if key in performance_metrics:
                    mlflow.log_metric(key, performance_metrics[key])

            # Log device info as parameters
            if "device_info" in performance_metrics:
                device_info = performance_metrics["device_info"]
                for key, value in device_info.items():
                    if isinstance(value, (str, int, float)):
                        mlflow.log_param(f"device_{key}", value)

        except Exception as e:
            logging.warning(f"Failed to log performance metrics: {e}")

    def log_pose_visualization(self, image_path: str, artifact_name: str = None):
        """Log pose visualization images

        Args:
            image_path: Path to visualization image
            artifact_name: Name for the artifact (defaults to filename)
        """
        try:
            if os.path.exists(image_path):
                if artifact_name is None:
                    artifact_name = os.path.basename(image_path)

                mlflow.log_artifact(image_path, "visualizations")

        except Exception as e:
            logging.warning(f"Failed to log visualization: {e}")

    def log_video_sample(self, video_path: str, artifact_name: str = None):
        """Log video sample with pose annotations

        Args:
            video_path: Path to annotated video
            artifact_name: Name for the artifact
        """
        try:
            if os.path.exists(video_path):
                if artifact_name is None:
                    artifact_name = os.path.basename(video_path)

                mlflow.log_artifact(video_path, "video_samples")

        except Exception as e:
            logging.warning(f"Failed to log video sample: {e}")

    def save_model_artifacts(self, model, model_name: str):
        """Save model artifacts and weights

        Args:
            model: Model instance
            model_name: Model name for saving
        """
        try:
            # Create temporary directory for model artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(temp_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)

                # Save model configuration
                config_path = os.path.join(model_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(model.model_config, f, indent=2)

                # Save model info
                info_path = os.path.join(model_dir, "model_info.json")
                with open(info_path, "w") as f:
                    json.dump(model.get_model_info(), f, indent=2)

                # Log as artifacts
                mlflow.log_artifacts(model_dir, f"models/{model_name}")

        except Exception as e:
            logging.warning(f"Failed to save model artifacts: {e}")

    def log_comparison_results(self, comparison_results: Dict[str, Any]):
        """Log model comparison results

        Args:
            comparison_results: Results from model comparison
        """
        try:
            # Log summary metrics
            if "summary" in comparison_results:
                summary = comparison_results["summary"]

                # Log best performing models
                if summary.get("fastest_model"):
                    mlflow.log_param("fastest_model", summary["fastest_model"])
                if summary.get("most_memory_efficient"):
                    mlflow.log_param(
                        "most_memory_efficient", summary["most_memory_efficient"]
                    )
                if summary.get("highest_throughput"):
                    mlflow.log_param(
                        "highest_throughput", summary["highest_throughput"]
                    )

            # Create comparison table
            comparison_table = self._create_comparison_table(comparison_results)
            mlflow.log_text(comparison_table, "model_comparison.md")

            # Log full results as JSON
            results_json = json.dumps(comparison_results, indent=2, default=str)
            mlflow.log_text(results_json, "full_comparison_results.json")

        except Exception as e:
            logging.warning(f"Failed to log comparison results: {e}")

    def _create_comparison_table(self, results: Dict[str, Any]) -> str:
        """Create markdown table for model comparison

        Args:
            results: Comparison results

        Returns:
            Markdown formatted comparison table
        """
        table_lines = [
            "# Model Comparison Results",
            "",
            "| Model | Inference Time (ms) | FPS | Memory (MB) | PCK@0.2 |",
            "|-------|---------------------|-----|-------------|---------|",
        ]

        for model_name, model_results in results.items():
            if model_name == "summary" or "error" in model_results:
                continue

            inference_time = model_results.get("mean_inference_time_ms", "N/A")
            fps = model_results.get("fps", "N/A")
            memory = model_results.get("process_ram_peak_mb", "N/A")
            pck = model_results.get("pose_pck_0.2_mean", "N/A")

            # Format numbers
            if isinstance(inference_time, (int, float)):
                inference_time = f"{inference_time:.2f}"
            if isinstance(fps, (int, float)):
                fps = f"{fps:.1f}"
            if isinstance(memory, (int, float)):
                memory = f"{memory:.1f}"
            if isinstance(pck, (int, float)):
                pck = f"{pck:.3f}"

            table_lines.append(
                f"| {model_name} | {inference_time} | {fps} | {memory} | {pck} |"
            )

        return "\n".join(table_lines)

    def get_experiment_runs(self) -> List[Dict[str, Any]]:
        """Get all runs from current experiment

        Returns:
            List of run information dictionaries
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return []

            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return runs.to_dict("records")

        except Exception as e:
            logging.warning(f"Failed to get experiment runs: {e}")
            return []

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple MLflow runs

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Comparison results
        """
        try:
            runs_data = []

            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                runs_data.append(
                    {
                        "run_id": run_id,
                        "run_name": run.info.run_name,
                        "params": run.data.params,
                        "metrics": run.data.metrics,
                        "tags": run.data.tags,
                    }
                )

            return {
                "runs": runs_data,
                "comparison_timestamp": mlflow.utils.time.get_current_time_millis(),
            }

        except Exception as e:
            logging.warning(f"Failed to compare runs: {e}")
            return {}

    def cleanup_old_runs(self, max_runs: int = 50):
        """Clean up old experiment runs

        Args:
            max_runs: Maximum number of runs to keep
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
            )

            if len(runs) > max_runs:
                old_runs = runs.iloc[max_runs:]
                for _, run in old_runs.iterrows():
                    mlflow.delete_run(run.run_id)
                    logging.info(f"Deleted old run: {run.run_id}")

        except Exception as e:
            logging.warning(f"Failed to cleanup old runs: {e}")

    def export_experiment(self, output_path: str):
        """Export experiment data to file

        Args:
            output_path: Path to export file
        """
        try:
            runs = self.get_experiment_runs()

            export_data = {
                "experiment_name": self.experiment_name,
                "export_timestamp": mlflow.utils.time.get_current_time_millis(),
                "runs": runs,
            }

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logging.info(f"Exported experiment data to {output_path}")

        except Exception as e:
            logging.warning(f"Failed to export experiment: {e}")
