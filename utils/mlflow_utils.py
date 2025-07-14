"""
MLflow Utilities for Multi-Run Management
Provides tools for viewing and managing MLflow experiments across all runs
"""

import os
import json
import yaml
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import mlflow

logger = logging.getLogger(__name__)


class MLflowManager:
    """Manages MLflow experiments across all runs in the shared directory"""

    def __init__(self):
        self.shared_results_dir = Path(
            "./data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results"
        )
        self.runs_dir = self.shared_results_dir / "runs"

    def log_video_sample(self, video_path: str):
        """Log a video sample as an MLflow artifact

        Args:
            video_path: Path to the video file to log
        """
        try:
            if os.path.exists(video_path):
                # Log the video file as an artifact
                mlflow.log_artifact(video_path, "sample_videos")
                logger.info(f"Logged video sample: {Path(video_path).name}")
            else:
                logger.warning(f"Video file not found for logging: {video_path}")
        except Exception as e:
            logger.error(f"Failed to log video sample {video_path}: {e}")

    def list_all_experiments(self) -> List[Dict[str, Any]]:
        """List all MLflow experiments across all runs"""
        if not self.runs_dir.exists():
            return []

        experiments = []
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
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

                                    # Load run metadata if available
                                    run_metadata_file = run_dir / "run_metadata.json"
                                    run_metadata = {}
                                    if run_metadata_file.exists():
                                        with open(run_metadata_file, "r") as f:
                                            run_metadata = json.load(f)

                                    experiments.append(
                                        {
                                            "experiment_id": exp_dir.name,
                                            "experiment_name": meta.get(
                                                "name", "unknown"
                                            ),
                                            "run_name": run_dir.name,
                                            "run_timestamp": run_metadata.get(
                                                "timestamp", "unknown"
                                            ),
                                            "created_at": run_metadata.get(
                                                "created_at", "unknown"
                                            ),
                                            "machine": run_metadata.get(
                                                "machine_info", {}
                                            ).get("hostname", "unknown"),
                                            "mlflow_dir": str(mlflow_dir),
                                            "tracking_uri": f"file://{mlflow_dir.absolute()}",
                                            "artifact_location": meta.get(
                                                "artifact_location", ""
                                            ),
                                            "lifecycle_stage": meta.get(
                                                "lifecycle_stage", "active"
                                            ),
                                        }
                                    )
                                except (yaml.YAMLError, json.JSONDecodeError, KeyError):
                                    pass

        return experiments

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of all experiments"""
        experiments = self.list_all_experiments()

        summary = {
            "total_experiments": len(experiments),
            "total_runs": len(set(exp["run_name"] for exp in experiments)),
            "machines": list(set(exp["machine"] for exp in experiments)),
            "experiment_types": {},
            "recent_experiments": [],
        }

        # Categorize experiments by type
        for exp in experiments:
            exp_name = exp["experiment_name"]
            if "optuna" in exp_name.lower():
                exp_type = "optuna"
            elif "comparison" in exp_name.lower():
                exp_type = "comparison"
            else:
                exp_type = "other"

            if exp_type not in summary["experiment_types"]:
                summary["experiment_types"][exp_type] = 0
            summary["experiment_types"][exp_type] += 1

        # Get recent experiments (last 10)
        summary["recent_experiments"] = experiments[:10]

        return summary

    def start_mlflow_ui(
        self, port: int = 5000, host: str = "localhost"
    ) -> subprocess.Popen:
        """Start MLflow UI for all experiments"""
        tracking_uri = f"file://{self.shared_results_dir.absolute()}"

        cmd = [
            "mlflow",
            "ui",
            "--backend-store-uri",
            tracking_uri,
            "--host",
            host,
            "--port",
            str(port),
        ]

        logger.info(f"Starting MLflow UI with command: {' '.join(cmd)}")
        logger.info(f"MLflow UI will be available at: http://{host}:{port}")

        # Start MLflow UI in background
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        return process

    def create_consolidated_tracking_uri(self) -> str:
        """Create a consolidated tracking URI that includes all experiments"""
        return f"file://{self.shared_results_dir.absolute()}"

    def export_experiment_data(self, output_file: Optional[str] = None) -> str:
        """Export all experiment data to a JSON file"""
        experiments = self.list_all_experiments()
        summary = self.get_experiment_summary()

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": summary,
            "experiments": experiments,
        }

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"mlflow_experiments_export_{timestamp}.json"

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported experiment data to: {output_path}")
        return str(output_path)

    def cleanup_old_experiments(self, keep_last_n_runs: int = 5):
        """Clean up old experiment runs, keeping only the most recent ones"""
        if not self.runs_dir.exists():
            logger.info("No runs directory found, nothing to clean up")
            return

        # Get all run directories sorted by timestamp
        run_dirs = []
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                run_dirs.append(run_dir)

        # Sort by directory name (which includes timestamp)
        run_dirs.sort(key=lambda x: x.name, reverse=True)

        if len(run_dirs) <= keep_last_n_runs:
            logger.info(f"Only {len(run_dirs)} runs found, no cleanup needed")
            return

        # Remove old runs
        runs_to_remove = run_dirs[keep_last_n_runs:]
        for run_dir in runs_to_remove:
            logger.info(f"Removing old run: {run_dir.name}")
            import shutil

            shutil.rmtree(run_dir)

        logger.info(f"Cleanup complete: kept {keep_last_n_runs} most recent runs")

    def print_experiments_summary(self):
        """Print a formatted summary of all experiments"""
        summary = self.get_experiment_summary()

        print("\n" + "=" * 60)
        print("🧪 MLFLOW EXPERIMENTS SUMMARY")
        print("=" * 60)

        print(f"📊 Total Experiments: {summary['total_experiments']}")
        print(f"🗂️ Total Runs: {summary['total_runs']}")
        print(f"💻 Machines: {', '.join(summary['machines'])}")

        print(f"\n📈 Experiment Types:")
        for exp_type, count in summary["experiment_types"].items():
            print(f"   • {exp_type.capitalize()}: {count}")

        print(f"\n🕐 Recent Experiments:")
        for exp in summary["recent_experiments"][:5]:
            print(f"   • {exp['experiment_name']} ({exp['run_name']})")
            print(f"     Machine: {exp['machine']}, Created: {exp['created_at']}")

        print(f"\n🔍 Access All Experiments:")
        tracking_uri = self.create_consolidated_tracking_uri()
        print(f"   MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")
        print(f"   Tracking URI: {tracking_uri}")

    def get_run_comparison(self) -> Dict[str, Any]:
        """Compare results across different runs"""
        experiments = self.list_all_experiments()

        # Group experiments by run
        runs_data = {}
        for exp in experiments:
            run_name = exp["run_name"]
            if run_name not in runs_data:
                runs_data[run_name] = {
                    "run_name": run_name,
                    "timestamp": exp["run_timestamp"],
                    "machine": exp["machine"],
                    "experiments": [],
                }
            runs_data[run_name]["experiments"].append(exp)

        # Sort runs by timestamp
        sorted_runs = sorted(
            runs_data.values(), key=lambda x: x["timestamp"], reverse=True
        )

        return {"total_runs": len(sorted_runs), "runs": sorted_runs}


def main():
    """CLI interface for MLflow management"""
    import argparse

    parser = argparse.ArgumentParser(description="MLflow Multi-Run Management")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument(
        "--summary", action="store_true", help="Show experiments summary"
    )
    parser.add_argument(
        "--export", type=str, help="Export experiment data to JSON file"
    )
    parser.add_argument(
        "--ui", action="store_true", help="Start MLflow UI for all experiments"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for MLflow UI")
    parser.add_argument("--cleanup", type=int, help="Clean up old runs (keep last N)")
    parser.add_argument(
        "--compare", action="store_true", help="Compare results across runs"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    mlflow_manager = MLflowManager()

    if args.list:
        experiments = mlflow_manager.list_all_experiments()
        print(json.dumps(experiments, indent=2))

    elif args.summary:
        mlflow_manager.print_experiments_summary()

    elif args.export:
        mlflow_manager.export_experiment_data(args.export)

    elif args.ui:
        process = mlflow_manager.start_mlflow_ui(port=args.port)
        try:
            print(f"MLflow UI running at http://localhost:{args.port}")
            print("Press Ctrl+C to stop...")
            process.wait()
        except KeyboardInterrupt:
            print("\nStopping MLflow UI...")
            process.terminate()

    elif args.cleanup:
        mlflow_manager.cleanup_old_experiments(args.cleanup)

    elif args.compare:
        comparison = mlflow_manager.get_run_comparison()
        print(json.dumps(comparison, indent=2))

    else:
        mlflow_manager.print_experiments_summary()


if __name__ == "__main__":
    main()
