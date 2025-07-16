#!/usr/bin/env python3
"""
MLflow UI Launcher for Individual Experiment Runs
Shows available experiment runs and lets user choose which one to view
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.mlflow_utils import MLflowManager


def choose_experiment_run(mlflow_manager):
    """Let user choose which experiment run to view"""
    # Get all available runs
    runs_dir = mlflow_manager.shared_results_dir / "runs"

    if not runs_dir.exists():
        print(f"❌ No runs directory found at: {runs_dir}")
        return None

    # Find all run directories with mlruns
    available_runs = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            mlruns_dir = run_dir / "mlruns"
            if mlruns_dir.exists():
                # Check if it has actual MLflow data
                has_data = any(mlruns_dir.iterdir())
                available_runs.append(
                    {
                        "name": run_dir.name,
                        "path": run_dir,
                        "mlruns_path": mlruns_dir,
                        "has_data": has_data,
                    }
                )

    if not available_runs:
        print("❌ No experiment runs with MLflow data found")
        return None

    # Show available runs
    print("📋 Available Experiment Runs:")
    print("=" * 50)
    for i, run in enumerate(available_runs, 1):
        status = "✅ Has data" if run["has_data"] else "⚠️  Empty"
        print(f"{i:2d}. {run['name']} ({status})")

    print(f"{len(available_runs) + 1:2d}. View all runs (consolidated)")
    print("=" * 50)

    # Get user choice
    try:
        choice = input(f"\nSelect run to view (1-{len(available_runs) + 1}): ").strip()
        choice_num = int(choice)

        if choice_num == len(available_runs) + 1:
            # Return None to use consolidated view
            return None
        elif 1 <= choice_num <= len(available_runs):
            selected_run = available_runs[choice_num - 1]
            if not selected_run["has_data"]:
                print(
                    f"⚠️  Warning: {selected_run['name']} appears to have no MLflow data"
                )
                proceed = input("Continue anyway? (y/N): ").strip().lower()
                if proceed != "y":
                    return None
            return selected_run["mlruns_path"]
        else:
            print("❌ Invalid choice")
            return None

    except (ValueError, KeyboardInterrupt):
        print("\n❌ Invalid input or cancelled")
        return None


def main():
    """Launch MLflow UI for selected experiment"""
    parser = argparse.ArgumentParser(description="Launch MLflow UI for experiment runs")
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for MLflow UI (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for MLflow UI (default: localhost)",
    )
    parser.add_argument(
        "--run-name", type=str, help="Specific run name to view (skips selection menu)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show experiments summary before starting UI",
    )

    args = parser.parse_args()

    # Initialize MLflow manager
    mlflow_manager = MLflowManager()

    # Show summary if requested
    if args.summary:
        mlflow_manager.print_experiments_summary()
        print()

    # Choose which experiment to view
    if args.run_name:
        # Use specified run name
        mlruns_path = (
            mlflow_manager.shared_results_dir / "runs" / args.run_name / "mlruns"
        )
        if not mlruns_path.exists():
            print(f"❌ Run '{args.run_name}' not found or has no MLflow data")
            sys.exit(1)
    else:
        # Let user choose
        mlruns_path = choose_experiment_run(mlflow_manager)
        if mlruns_path is None:
            print("ℹ️  Using consolidated view of all experiments")
            mlruns_path = mlflow_manager.shared_results_dir

    # Start MLflow UI
    print(f"🚀 Starting MLflow UI...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"📂 Data directory: {mlruns_path}")
    print()

    try:
        import subprocess

        cmd = [
            "mlflow",
            "ui",
            "--backend-store-uri",
            f"file://{mlruns_path.absolute()}",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]

        print(f"📋 Running: {' '.join(cmd)}")

        process = subprocess.Popen(cmd)
        print(f"✅ MLflow UI running at: http://{args.host}:{args.port}")
        print(f"🔍 Viewing experiment data from: {mlruns_path.name}")
        print()
        print("Press Ctrl+C to stop the MLflow UI...")

        # Wait for process to complete or be interrupted
        process.wait()

    except KeyboardInterrupt:
        print("\n🛑 Stopping MLflow UI...")
        process.terminate()
        process.wait()
        print("✅ MLflow UI stopped")

    except Exception as e:
        print(f"❌ Error starting MLflow UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
