#!/usr/bin/env python3
"""
Memory Profiler for Pose Estimation Evaluation
Provides comprehensive memory and performance monitoring with MLflow integration
"""

import tracemalloc
import psutil
import torch
import time
import mlflow
import threading
import json
import warnings
import os
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Suppress protobuf warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot"""

    timestamp: float
    elapsed_time: float
    system_memory_percent: float
    system_memory_available_gb: float
    system_memory_used_gb: float
    process_memory_rss_mb: float
    process_memory_vms_mb: float
    process_memory_percent: float
    cpu_percent: float
    gpu_available: bool = False
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_max_allocated_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    tracemalloc_current_mb: float = 0.0
    tracemalloc_peak_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class MemoryProfiler:
    """Comprehensive memory profiler with MLflow integration and time-series tracking"""

    def __init__(
        self,
        enable_tracemalloc: bool = True,
        monitoring_interval: float = 2.0,
        enable_continuous_monitoring: bool = True,
        save_snapshots: bool = True,
        mlflow_buffer_size: int = 100,
    ):
        """
        Initialize memory profiler

        Args:
            enable_tracemalloc: Enable Python tracemalloc for detailed memory tracking
            monitoring_interval: Interval in seconds for continuous monitoring
            enable_continuous_monitoring: Whether to run background monitoring
            save_snapshots: Whether to save detailed snapshots to disk
            mlflow_buffer_size: Number of recent snapshots to buffer for retroactive MLflow logging
        """
        self.enable_tracemalloc = enable_tracemalloc
        self.monitoring_interval = monitoring_interval
        self.enable_continuous_monitoring = enable_continuous_monitoring
        self.save_snapshots = save_snapshots
        self.mlflow_buffer_size = mlflow_buffer_size

        # Tracking variables
        self.start_time = None
        self.initial_memory = None
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        # MLflow state tracking
        self.current_experiment_id = None
        self.current_run_id = None
        self.mlflow_buffer: List[MemorySnapshot] = (
            []
        )  # Recent snapshots for retroactive logging
        self.logged_snapshots = (
            set()
        )  # Track which snapshots have been logged to avoid duplicates

        # Event-driven MLflow integration
        self.mlflow_run_active = False
        self.force_log_next = (
            threading.Event()
        )  # Signal to force logging on next snapshot

        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import GPUtil

                self.gputil_available = True
            except ImportError:
                self.gputil_available = False
                logger.warning(
                    "GPUtil not available. GPU utilization monitoring disabled."
                )
        else:
            self.gputil_available = False

        # Initialize tracemalloc
        if self.enable_tracemalloc:
            tracemalloc.start()
            logger.info("Tracemalloc enabled for detailed memory tracking")

    def start_profiling(self, run_dir: Optional[Path] = None):
        """Start memory profiling with optional run directory for saving snapshots"""
        self.start_time = time.time()
        self.initial_memory = self._get_current_process_memory()
        self.run_dir = run_dir

        # Take initial snapshot
        initial_snapshot = self._take_snapshot()
        self.snapshots.append(initial_snapshot)

        # Start continuous monitoring if enabled
        if self.enable_continuous_monitoring:
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(
                target=self._continuous_monitoring
            )
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()

        logger.info(
            f"Memory profiling started. Initial memory: {self.initial_memory:.1f} MB"
        )

        # Log initial state to MLflow
        self._log_snapshot_to_mlflow(initial_snapshot, "start")

    def _continuous_monitoring(self):
        """Background thread for continuous memory monitoring with event-driven MLflow integration"""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)

                # Maintain rolling buffer for retroactive MLflow logging
                self.mlflow_buffer.append(snapshot)
                if len(self.mlflow_buffer) > self.mlflow_buffer_size:
                    self.mlflow_buffer.pop(0)

                # Event-driven MLflow logging
                should_log_to_mlflow = False

                # Force log if signaled
                if self.force_log_next.is_set():
                    should_log_to_mlflow = True
                    self.force_log_next.clear()
                    logger.debug("🎯 Force logging triggered")

                # Log if MLflow run is explicitly active
                elif self.mlflow_run_active and mlflow.active_run():
                    should_log_to_mlflow = True

                # Fallback: Check for MLflow state changes (less frequent)
                elif not self.mlflow_run_active:
                    self._check_mlflow_state_change()
                    should_log_to_mlflow = mlflow.active_run() is not None

                # Log to MLflow if conditions are met
                if should_log_to_mlflow:
                    self._log_snapshot_to_mlflow(snapshot, "continuous")

                # Save snapshot to disk if enabled
                if self.save_snapshots and self.run_dir:
                    self._save_snapshot_to_disk(snapshot)

            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")

    def _check_mlflow_state_change(self):
        """Check if MLflow experiment state has changed and handle transitions"""
        try:
            active_run = mlflow.active_run()

            if active_run is None:
                # No active run - reset state if we were tracking one
                if self.current_run_id is not None:
                    logger.debug(f"MLflow run ended: {self.current_run_id}")
                    self.current_experiment_id = None
                    self.current_run_id = None
                return

            current_run_id = active_run.info.run_id
            current_experiment_id = active_run.info.experiment_id

            # Check if we've switched to a new run
            if current_run_id != self.current_run_id:
                logger.info(
                    f"🔄 New MLflow run detected: {current_run_id} (experiment: {current_experiment_id})"
                )

                # Retroactively log recent buffer data to the new run
                if self.mlflow_buffer:
                    self._retroactive_log_buffer()

                # Update state
                self.current_run_id = current_run_id
                self.current_experiment_id = current_experiment_id

        except Exception as e:
            logger.debug(f"Error checking MLflow state: {e}")

    def _retroactive_log_buffer(self):
        """Retroactively log buffered snapshots to the newly active MLflow run"""
        if not self.mlflow_buffer:
            return

        try:
            active_run = mlflow.active_run()
            if active_run is None:
                return

            logger.info(
                f"📊 Retroactively logging {len(self.mlflow_buffer)} buffered snapshots to MLflow"
            )

            # Log all buffered snapshots
            for snapshot in self.mlflow_buffer:
                snapshot_id = f"{snapshot.timestamp}_{snapshot.elapsed_time}"

                # Skip if already logged to avoid duplicates
                if snapshot_id in self.logged_snapshots:
                    continue

                self._log_snapshot_to_mlflow_direct(snapshot, "retroactive")
                self.logged_snapshots.add(snapshot_id)

        except Exception as e:
            logger.error(f"Error in retroactive logging: {e}")

    def _take_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0

        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)

        # GPU memory (if available)
        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        gpu_max_allocated_mb = 0.0
        gpu_utilization_percent = 0.0

        if self.gpu_available:
            gpu_allocated_mb = torch.cuda.memory_allocated() / 1e6
            gpu_reserved_mb = torch.cuda.memory_reserved() / 1e6
            gpu_max_allocated_mb = torch.cuda.max_memory_allocated() / 1e6

            # GPU utilization (if GPUtil available)
            if self.gputil_available:
                try:
                    import GPUtil

                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_utilization_percent = gpus[0].load * 100
                except Exception as e:
                    logger.debug(f"GPU utilization query failed: {e}")

        # Tracemalloc (if enabled)
        tracemalloc_current_mb = 0.0
        tracemalloc_peak_mb = 0.0
        if self.enable_tracemalloc:
            try:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_current_mb = current / 1e6
                tracemalloc_peak_mb = peak / 1e6
            except Exception as e:
                logger.debug(f"Tracemalloc query failed: {e}")

        return MemorySnapshot(
            timestamp=current_time,
            elapsed_time=elapsed_time,
            system_memory_percent=system_memory.percent,
            system_memory_available_gb=system_memory.available / 1e9,
            system_memory_used_gb=system_memory.used / 1e9,
            process_memory_rss_mb=process_memory.rss / 1e6,
            process_memory_vms_mb=process_memory.vms / 1e6,
            process_memory_percent=process.memory_percent(),
            cpu_percent=cpu_percent,
            gpu_available=self.gpu_available,
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
            gpu_max_allocated_mb=gpu_max_allocated_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            tracemalloc_current_mb=tracemalloc_current_mb,
            tracemalloc_peak_mb=tracemalloc_peak_mb,
        )

    def _log_snapshot_to_mlflow(self, snapshot: MemorySnapshot, phase: str):
        """Log snapshot metrics to MLflow with timestamp - only if experiment is active"""
        try:
            # Check for duplicate logging
            snapshot_id = f"{snapshot.timestamp}_{snapshot.elapsed_time}"
            if snapshot_id in self.logged_snapshots:
                return  # Already logged this snapshot

            # Check if MLflow has an active run
            active_run = mlflow.active_run()
            if active_run is None:
                # No active run - skip logging but don't error
                logger.debug(
                    f"No active MLflow run - skipping memory metrics logging for phase: {phase}"
                )
                return

            # Verify we have a valid experiment ID
            try:
                experiment_id = active_run.info.experiment_id
                if experiment_id is None:
                    logger.debug(
                        f"No valid experiment ID - skipping memory metrics logging for phase: {phase}"
                    )
                    return
            except Exception as exp_err:
                logger.debug(
                    f"Could not get experiment ID - skipping memory metrics logging: {exp_err}"
                )
                return

            # System metrics
            mlflow.log_metric(
                "memory_system_percent",
                snapshot.system_memory_percent,
                step=int(snapshot.elapsed_time),
            )
            mlflow.log_metric(
                "memory_system_available_gb",
                snapshot.system_memory_available_gb,
                step=int(snapshot.elapsed_time),
            )
            mlflow.log_metric(
                "memory_system_used_gb",
                snapshot.system_memory_used_gb,
                step=int(snapshot.elapsed_time),
            )

            # Process metrics
            mlflow.log_metric(
                "memory_process_rss_mb",
                snapshot.process_memory_rss_mb,
                step=int(snapshot.elapsed_time),
            )
            mlflow.log_metric(
                "memory_process_vms_mb",
                snapshot.process_memory_vms_mb,
                step=int(snapshot.elapsed_time),
            )
            mlflow.log_metric(
                "memory_process_percent",
                snapshot.process_memory_percent,
                step=int(snapshot.elapsed_time),
            )

            # CPU metrics
            mlflow.log_metric(
                "cpu_percent", snapshot.cpu_percent, step=int(snapshot.elapsed_time)
            )

            # GPU metrics (if available)
            if snapshot.gpu_available:
                mlflow.log_metric(
                    "gpu_allocated_mb",
                    snapshot.gpu_allocated_mb,
                    step=int(snapshot.elapsed_time),
                )
                mlflow.log_metric(
                    "gpu_reserved_mb",
                    snapshot.gpu_reserved_mb,
                    step=int(snapshot.elapsed_time),
                )
                mlflow.log_metric(
                    "gpu_max_allocated_mb",
                    snapshot.gpu_max_allocated_mb,
                    step=int(snapshot.elapsed_time),
                )
                mlflow.log_metric(
                    "gpu_utilization_percent",
                    snapshot.gpu_utilization_percent,
                    step=int(snapshot.elapsed_time),
                )

            # Tracemalloc metrics (if enabled)
            if self.enable_tracemalloc:
                mlflow.log_metric(
                    "tracemalloc_current_mb",
                    snapshot.tracemalloc_current_mb,
                    step=int(snapshot.elapsed_time),
                )
                mlflow.log_metric(
                    "tracemalloc_peak_mb",
                    snapshot.tracemalloc_peak_mb,
                    step=int(snapshot.elapsed_time),
                )

            # Mark as successfully logged
            self.logged_snapshots.add(snapshot_id)

        except Exception as e:
            logger.debug(
                f"Failed to log metrics to MLflow (this is normal during startup): {e}"
            )

    def _log_snapshot_to_mlflow_direct(self, snapshot: MemorySnapshot, phase: str):
        """Direct MLflow logging for retroactive data - assumes active run exists"""
        try:
            # Mark this snapshot as logged to prevent duplicates
            snapshot_id = f"{snapshot.timestamp}_{snapshot.elapsed_time}"

            # System metrics
            mlflow.log_metric(
                "memory_system_percent",
                snapshot.system_memory_percent,
                step=int(snapshot.elapsed_time),
            )
            mlflow.log_metric(
                "memory_system_available_gb",
                snapshot.system_memory_available_gb,
                step=int(snapshot.elapsed_time),
            )
            mlflow.log_metric(
                "memory_system_used_gb",
                snapshot.system_memory_used_gb,
                step=int(snapshot.elapsed_time),
            )

            # Process metrics
            mlflow.log_metric(
                "memory_process_rss_mb",
                snapshot.process_memory_rss_mb,
                step=int(snapshot.elapsed_time),
            )
            mlflow.log_metric(
                "memory_process_vms_mb",
                snapshot.process_memory_vms_mb,
                step=int(snapshot.elapsed_time),
            )
            mlflow.log_metric(
                "memory_process_percent",
                snapshot.process_memory_percent,
                step=int(snapshot.elapsed_time),
            )

            # CPU metrics
            mlflow.log_metric(
                "cpu_percent", snapshot.cpu_percent, step=int(snapshot.elapsed_time)
            )

            # GPU metrics (if available)
            if snapshot.gpu_available:
                mlflow.log_metric(
                    "gpu_allocated_mb",
                    snapshot.gpu_allocated_mb,
                    step=int(snapshot.elapsed_time),
                )
                mlflow.log_metric(
                    "gpu_reserved_mb",
                    snapshot.gpu_reserved_mb,
                    step=int(snapshot.elapsed_time),
                )
                mlflow.log_metric(
                    "gpu_max_allocated_mb",
                    snapshot.gpu_max_allocated_mb,
                    step=int(snapshot.elapsed_time),
                )
                mlflow.log_metric(
                    "gpu_utilization_percent",
                    snapshot.gpu_utilization_percent,
                    step=int(snapshot.elapsed_time),
                )

            # Tracemalloc metrics (if enabled)
            if self.enable_tracemalloc:
                mlflow.log_metric(
                    "tracemalloc_current_mb",
                    snapshot.tracemalloc_current_mb,
                    step=int(snapshot.elapsed_time),
                )
                mlflow.log_metric(
                    "tracemalloc_peak_mb",
                    snapshot.tracemalloc_peak_mb,
                    step=int(snapshot.elapsed_time),
                )

            # Mark as successfully logged
            self.logged_snapshots.add(snapshot_id)

        except Exception as e:
            logger.debug(f"Failed to log metrics directly to MLflow: {e}")

    def _save_snapshot_to_disk(self, snapshot: MemorySnapshot):
        """Save snapshot to disk for detailed analysis"""
        if not self.run_dir:
            return

        snapshots_dir = self.run_dir / "memory_snapshots"
        snapshots_dir.mkdir(exist_ok=True)

        # Save individual snapshot
        snapshot_file = snapshots_dir / f"snapshot_{snapshot.timestamp:.3f}.json"
        try:
            with open(snapshot_file, "w") as f:
                json.dump(snapshot.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save snapshot to disk: {e}")

    def log_milestone(self, milestone_name: str):
        """Log a milestone with current memory stats"""
        snapshot = self._take_snapshot()
        self.snapshots.append(snapshot)

        # Log to MLflow with milestone tag (if available)
        self._log_snapshot_to_mlflow(snapshot, f"milestone_{milestone_name}")

        # Log milestone-specific metrics (if MLflow is available)
        try:
            active_run = mlflow.active_run()
            if active_run is None:
                logger.debug(
                    f"No active MLflow run - skipping milestone metrics for: {milestone_name}"
                )
            else:
                mlflow.log_metric(
                    f"milestone_{milestone_name}_memory_mb",
                    snapshot.process_memory_rss_mb,
                )
                mlflow.log_metric(
                    f"milestone_{milestone_name}_cpu_percent", snapshot.cpu_percent
                )
                if snapshot.gpu_available:
                    mlflow.log_metric(
                        f"milestone_{milestone_name}_gpu_mb", snapshot.gpu_allocated_mb
                    )
        except Exception as e:
            logger.debug(
                f"Failed to log milestone metrics (normal during startup): {e}"
            )

        logger.info(
            f"Milestone '{milestone_name}': Memory={snapshot.process_memory_rss_mb:.1f}MB, "
            f"CPU={snapshot.cpu_percent:.1f}%, GPU={snapshot.gpu_allocated_mb:.1f}MB"
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.snapshots:
            return {}

        latest_snapshot = self.snapshots[-1]

        # Calculate statistics over all snapshots
        process_memory_values = [s.process_memory_rss_mb for s in self.snapshots]
        cpu_values = [s.cpu_percent for s in self.snapshots]

        stats = {
            "latest": latest_snapshot.to_dict(),
            "statistics": {
                "duration_seconds": latest_snapshot.elapsed_time,
                "snapshots_count": len(self.snapshots),
                "process_memory": {
                    "current_mb": latest_snapshot.process_memory_rss_mb,
                    "peak_mb": max(process_memory_values),
                    "mean_mb": np.mean(process_memory_values),
                    "std_mb": np.std(process_memory_values),
                    "increase_from_start_mb": (
                        latest_snapshot.process_memory_rss_mb - process_memory_values[0]
                        if process_memory_values
                        else 0
                    ),
                },
                "cpu": {
                    "current_percent": latest_snapshot.cpu_percent,
                    "peak_percent": max(cpu_values),
                    "mean_percent": np.mean(cpu_values),
                    "std_percent": np.std(cpu_values),
                },
            },
        }

        # Add GPU statistics if available
        if latest_snapshot.gpu_available:
            gpu_values = [s.gpu_allocated_mb for s in self.snapshots]
            stats["statistics"]["gpu"] = {
                "current_mb": latest_snapshot.gpu_allocated_mb,
                "peak_mb": max(gpu_values),
                "mean_mb": np.mean(gpu_values),
                "std_mb": np.std(gpu_values),
            }

        return stats

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        stats = self.get_memory_stats()

        # Add additional analysis
        if len(self.snapshots) > 1:
            # Memory growth analysis
            first_memory = self.snapshots[0].process_memory_rss_mb
            last_memory = self.snapshots[-1].process_memory_rss_mb
            memory_growth_rate = (
                (last_memory - first_memory) / self.snapshots[-1].elapsed_time
                if self.snapshots[-1].elapsed_time > 0
                else 0
            )

            stats["analysis"] = {
                "memory_growth_rate_mb_per_second": memory_growth_rate,
                "potential_memory_leak": memory_growth_rate > 1.0,  # >1MB/sec growth
                "memory_efficiency": (
                    "good"
                    if memory_growth_rate < 0.5
                    else "moderate" if memory_growth_rate < 2.0 else "poor"
                ),
            }

        return stats

    def save_full_report(self, filepath: Path):
        """Save complete memory profiling report"""
        report = {
            "profiling_info": {
                "start_time": (
                    datetime.fromtimestamp(self.start_time).isoformat()
                    if self.start_time
                    else None
                ),
                "duration_seconds": (
                    self.snapshots[-1].elapsed_time if self.snapshots else 0
                ),
                "monitoring_interval": self.monitoring_interval,
                "total_snapshots": len(self.snapshots),
                "tracemalloc_enabled": self.enable_tracemalloc,
                "gpu_available": self.gpu_available,
            },
            "summary": self.generate_summary_report(),
            "all_snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Full memory report saved to {filepath}")

    def _get_current_process_memory(self) -> float:
        """Get current process memory in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1e6

    def stop_profiling(self) -> Dict[str, Any]:
        """Stop memory profiling and return final stats"""
        # Stop continuous monitoring
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)

        # Take final snapshot
        final_snapshot = self._take_snapshot()
        self.snapshots.append(final_snapshot)
        self._log_snapshot_to_mlflow(final_snapshot, "stop")

        # Stop tracemalloc
        if self.enable_tracemalloc:
            tracemalloc.stop()

        # Generate final report
        final_stats = self.generate_summary_report()

        # Save complete report if run directory is available
        if self.run_dir:
            report_path = self.run_dir / "memory_profiling_report.json"
            self.save_full_report(report_path)

        logger.info(
            f"Memory profiling stopped. Final memory: {final_snapshot.process_memory_rss_mb:.1f}MB, "
            f"Peak memory: {final_stats['statistics']['process_memory']['peak_mb']:.1f}MB"
        )

        return final_stats

    # Event-driven MLflow integration methods

    def on_mlflow_run_start(self, run_id: str = None, experiment_name: str = None):
        """Signal that an MLflow run has started - triggers immediate logging"""
        try:
            active_run = mlflow.active_run()
            if active_run:
                self.current_run_id = active_run.info.run_id
                self.current_experiment_id = active_run.info.experiment_id
                self.mlflow_run_active = True

                logger.info(
                    f"🔄 MLflow run started: {self.current_run_id} - Enabling real-time memory logging"
                )

                # Immediately log recent buffer data to the new run
                if self.mlflow_buffer:
                    self._retroactive_log_buffer()

                # Force logging on next snapshot
                self.force_log_next.set()

            else:
                logger.warning(
                    "on_mlflow_run_start called but no active MLflow run detected"
                )

        except Exception as e:
            logger.error(f"Error in on_mlflow_run_start: {e}")

    def on_mlflow_run_end(self):
        """Signal that an MLflow run has ended"""
        try:
            if self.mlflow_run_active:
                logger.info(
                    f"🔚 MLflow run ended: {self.current_run_id} - Stopping real-time logging"
                )

                # Take a final snapshot and log it
                if self.current_run_id:
                    final_snapshot = self._take_snapshot()
                    self.snapshots.append(final_snapshot)
                    # Don't log the final snapshot to MLflow as the run is ending

                self.mlflow_run_active = False
                self.current_run_id = None
                self.current_experiment_id = None

        except Exception as e:
            logger.error(f"Error in on_mlflow_run_end: {e}")

    def force_log_current_state(self):
        """Force immediate logging of current memory state to active MLflow run"""
        try:
            active_run = mlflow.active_run()
            if active_run:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                self._log_snapshot_to_mlflow(snapshot, "forced")
                logger.debug(
                    f"💾 Forced memory logging to run: {active_run.info.run_id}"
                )
            else:
                logger.debug("force_log_current_state called but no active MLflow run")

        except Exception as e:
            logger.error(f"Error in force_log_current_state: {e}")

    def flush_buffer_to_mlflow(self):
        """Flush all buffered data to the current active MLflow run"""
        try:
            active_run = mlflow.active_run()
            if active_run and self.mlflow_buffer:
                logger.info(
                    f"🚀 Flushing {len(self.mlflow_buffer)} buffered snapshots to MLflow run: {active_run.info.run_id}"
                )
                self._retroactive_log_buffer()

        except Exception as e:
            logger.error(f"Error in flush_buffer_to_mlflow: {e}")
