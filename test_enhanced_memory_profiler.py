#!/usr/bin/env python3
"""
Test enhanced memory profiler with dynamic MLflow detection
"""

import mlflow
import time
import tempfile
from pathlib import Path
from utils.memory_profiler import MemoryProfiler


def test_enhanced_memory_profiler():
    """Test enhanced memory profiler with dynamic MLflow integration"""

    print("🧪 Testing Enhanced Memory Profiler...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Set up MLflow tracking
        mlflow_uri = f"file://{temp_path}/test_mlruns"
        mlflow.set_tracking_uri(mlflow_uri)

        print(f"📂 MLflow URI: {mlflow_uri}")

        # Initialize enhanced memory profiler
        profiler = MemoryProfiler(
            enable_tracemalloc=True,
            monitoring_interval=0.5,  # Fast for testing
            enable_continuous_monitoring=True,
            save_snapshots=False,  # Don't save snapshots for this test
            mlflow_buffer_size=20,  # Small buffer for testing
        )

        try:
            print("\n🔍 Phase 1: Start profiler with no MLflow experiment")
            profiler.start_profiling(temp_path)
            time.sleep(2)  # Let it collect data without MLflow

            print("\n🔍 Phase 2: Start first MLflow experiment")
            mlflow.set_experiment("test_experiment_1")

            with mlflow.start_run(run_name="test_run_1"):
                print("✅ Experiment 1 started - should see retroactive logging")
                time.sleep(3)  # Let it log some data

                # Check metrics
                run = mlflow.active_run()
                client = mlflow.tracking.MlflowClient()
                metrics = client.get_run(run.info.run_id).data.metrics

                memory_metrics = [
                    key
                    for key in metrics.keys()
                    if key.startswith(("memory_", "cpu_", "gpu_"))
                ]
                print(f"📈 Experiment 1 - Found {len(memory_metrics)} memory metrics")

                if memory_metrics:
                    print("✅ SUCCESS: Memory metrics found in first experiment!")
                    for metric in sorted(memory_metrics)[:5]:
                        print(f"   • {metric}")
                else:
                    print("❌ No memory metrics found in first experiment")

            print("\n🔍 Phase 3: Start second MLflow experiment")
            time.sleep(1)  # Brief gap

            mlflow.set_experiment("test_experiment_2")
            with mlflow.start_run(run_name="test_run_2"):
                print("✅ Experiment 2 started - should see transition and new logging")
                time.sleep(2)

                # Check metrics
                run = mlflow.active_run()
                client = mlflow.tracking.MlflowClient()
                metrics = client.get_run(run.info.run_id).data.metrics

                memory_metrics = [
                    key
                    for key in metrics.keys()
                    if key.startswith(("memory_", "cpu_", "gpu_"))
                ]
                print(f"📈 Experiment 2 - Found {len(memory_metrics)} memory metrics")

                if memory_metrics:
                    print("✅ SUCCESS: Memory metrics found in second experiment!")
                else:
                    print("❌ No memory metrics found in second experiment")

            print("\n🔍 Phase 4: No active experiment")
            time.sleep(2)  # Let it run without experiments

        finally:
            # Stop profiling
            final_stats = profiler.stop_profiling()
            print(
                f"\n📊 Final stats: {len(profiler.logged_snapshots)} snapshots logged to MLflow"
            )
            print(f"📊 Buffer size: {len(profiler.mlflow_buffer)} snapshots")
            print(f"📊 Total snapshots: {len(profiler.snapshots)}")

        print("\n🎉 Enhanced memory profiler test completed!")
        return True


if __name__ == "__main__":
    success = test_enhanced_memory_profiler()
    exit(0 if success else 1)
