#!/usr/bin/env python3
"""
MLflow UI Launcher for Shared POSE Directory
Starts MLflow UI to view all experiments across all runs
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.mlflow_utils import MLflowManager


def main():
    """Launch MLflow UI for all experiments"""
    parser = argparse.ArgumentParser(
        description="Launch MLflow UI for all experiments in shared POSE directory"
    )
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

    # Start MLflow UI
    print(f"🚀 Starting MLflow UI for all experiments...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"📂 Shared directory: {mlflow_manager.shared_results_dir}")
    print()

    try:
        process = mlflow_manager.start_mlflow_ui(port=args.port, host=args.host)
        print(f"✅ MLflow UI running at: http://{args.host}:{args.port}")
        print(f"🔍 View all experiments from all runs in one interface")
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
