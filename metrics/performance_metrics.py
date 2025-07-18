"""
Performance metrics for pose estimation models
"""

import time
import psutil
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Comprehensive performance metrics for pose estimation models"""

    def __init__(self, device: str = "cpu"):
        """Initialize performance metrics

        Args:
            device: Device to monitor (cpu, cuda, mps)
        """
        self.device = device
        self.process = psutil.Process()

    def measure_inference_time(
        self, model, test_frame: np.ndarray, num_runs: int = 10, warmup_runs: int = 3
    ) -> Dict[str, float]:
        """Measure detailed inference timing metrics

        Args:
            model: Pose estimation model
            test_frame: Test frame for inference
            num_runs: Number of inference runs to measure
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing metrics
        """
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                _ = model.predict(test_frame)
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
                continue

        # Actual timing measurements
        inference_times = []
        preprocess_times = []
        postprocess_times = []

        for _ in range(num_runs):
            try:
                # Measure total inference time
                start_time = time.perf_counter()
                result = model.predict(test_frame)
                end_time = time.perf_counter()

                total_time = (end_time - start_time) * 1000  # Convert to ms
                inference_times.append(total_time)

                # Try to get detailed timing if available
                if hasattr(model, "get_timing_breakdown"):
                    timing_breakdown = model.get_timing_breakdown()
                    preprocess_times.append(
                        timing_breakdown.get("preprocess_time_ms", 0)
                    )
                    postprocess_times.append(
                        timing_breakdown.get("postprocess_time_ms", 0)
                    )

            except Exception as e:
                logger.warning(f"Inference timing run failed: {e}")
                continue

        if not inference_times:
            return {}

        # Calculate timing statistics
        timing_metrics = {
            "avg_inference_time_ms": np.mean(inference_times),
            "min_inference_time_ms": np.min(inference_times),
            "max_inference_time_ms": np.max(inference_times),
            "std_inference_time_ms": np.std(inference_times),
            "median_inference_time_ms": np.median(inference_times),
            "p95_inference_time_ms": np.percentile(inference_times, 95),
            "p99_inference_time_ms": np.percentile(inference_times, 99),
            "fps": 1000.0 / np.mean(inference_times),
            "throughput_fps": 1000.0 / np.mean(inference_times),
        }

        # Add detailed timing if available
        if preprocess_times:
            timing_metrics.update(
                {
                    "avg_preprocess_time_ms": np.mean(preprocess_times),
                    "avg_postprocess_time_ms": np.mean(postprocess_times),
                    "preprocess_ratio": np.mean(preprocess_times)
                    / np.mean(inference_times),
                    "postprocess_ratio": np.mean(postprocess_times)
                    / np.mean(inference_times),
                }
            )

        return timing_metrics

    def measure_memory_usage(
        self, model, test_frame: np.ndarray, num_runs: int = 5
    ) -> Dict[str, float]:
        """Measure detailed memory usage metrics

        Args:
            model: Pose estimation model
            test_frame: Test frame for inference
            num_runs: Number of runs to measure memory

        Returns:
            Dictionary with memory metrics
        """
        memory_usage = []
        peak_memory_usage = []
        memory_after_gc = []

        for _ in range(num_runs):
            try:
                # Measure memory before inference
                memory_before = self._get_current_memory_mb()

                # Run inference
                result = model.predict(test_frame)

                # Measure memory after inference
                memory_after = self._get_current_memory_mb()
                memory_usage.append(memory_after - memory_before)

                # Measure peak memory
                peak_memory = self._get_peak_memory_mb()
                peak_memory_usage.append(peak_memory)

                # Force garbage collection and measure
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                memory_after_gc.append(self._get_current_memory_mb())

            except Exception as e:
                logger.warning(f"Memory measurement run failed: {e}")
                continue

        if not memory_usage:
            return {}

        # Calculate memory statistics
        memory_metrics = {
            "avg_memory_increase_mb": np.mean(memory_usage),
            "max_memory_increase_mb": np.max(memory_usage),
            "std_memory_increase_mb": np.std(memory_usage),
            "avg_peak_memory_mb": np.mean(peak_memory_usage),
            "max_peak_memory_mb": np.max(peak_memory_usage),
        }

        if memory_after_gc:
            memory_metrics.update(
                {
                    "avg_memory_after_gc_mb": np.mean(memory_after_gc),
                    "memory_leak_mb": (
                        np.mean(memory_after_gc) - memory_before
                        if "memory_before" in locals()
                        else 0
                    ),
                }
            )

        return memory_metrics

    def measure_throughput(
        self, model, test_frames: List[np.ndarray], batch_size: int = 1
    ) -> Dict[str, float]:
        """Measure throughput metrics

        Args:
            model: Pose estimation model
            test_frames: List of test frames
            batch_size: Batch size for processing

        Returns:
            Dictionary with throughput metrics
        """
        if len(test_frames) < 2:
            return {}

        # Measure single frame throughput
        single_frame_times = []
        for frame in test_frames[: min(10, len(test_frames))]:
            try:
                start_time = time.perf_counter()
                _ = model.predict(frame)
                end_time = time.perf_counter()
                single_frame_times.append((end_time - start_time) * 1000)
            except Exception as e:
                logger.warning(f"Single frame throughput measurement failed: {e}")
                continue

        # Measure batch processing if supported
        batch_throughput = None
        if hasattr(model, "predict_batch") and len(test_frames) >= batch_size:
            try:
                batch_frames = test_frames[:batch_size]
                start_time = time.perf_counter()
                _ = model.predict_batch(batch_frames)
                end_time = time.perf_counter()
                batch_time = (end_time - start_time) * 1000
                batch_throughput = (batch_size * 1000) / batch_time
            except Exception as e:
                logger.warning(f"Batch throughput measurement failed: {e}")

        # Calculate throughput metrics
        throughput_metrics = {
            "avg_single_frame_time_ms": (
                np.mean(single_frame_times) if single_frame_times else 0
            ),
            "single_frame_throughput_fps": (
                1000.0 / np.mean(single_frame_times) if single_frame_times else 0
            ),
            "single_frame_throughput_std": (
                np.std(single_frame_times) if single_frame_times else 0
            ),
        }

        if batch_throughput is not None:
            throughput_metrics.update(
                {
                    "batch_throughput_fps": batch_throughput,
                    "batch_efficiency": (
                        batch_throughput / (1000.0 / np.mean(single_frame_times))
                        if single_frame_times
                        else 0
                    ),
                }
            )

        return throughput_metrics

    def measure_cpu_utilization(
        self, model, test_frame: np.ndarray, duration_seconds: float = 5.0
    ) -> Dict[str, float]:
        """Measure CPU utilization during inference

        Args:
            model: Pose estimation model
            test_frame: Test frame for inference
            duration_seconds: Duration to measure CPU usage

        Returns:
            Dictionary with CPU metrics
        """
        cpu_percentages = []
        start_time = time.time()

        # Monitor CPU usage during repeated inference
        while time.time() - start_time < duration_seconds:
            try:
                # Get CPU usage before inference
                cpu_before = psutil.cpu_percent(interval=0.1)

                # Run inference
                _ = model.predict(test_frame)

                # Get CPU usage after inference
                cpu_after = psutil.cpu_percent(interval=0.1)

                cpu_percentages.append(max(cpu_before, cpu_after))

            except Exception as e:
                logger.warning(f"CPU measurement failed: {e}")
                break

        if not cpu_percentages:
            return {}

        return {
            "avg_cpu_utilization": np.mean(cpu_percentages),
            "max_cpu_utilization": np.max(cpu_percentages),
            "std_cpu_utilization": np.std(cpu_percentages),
            "cpu_measurement_duration_s": duration_seconds,
        }

    def measure_model_efficiency(
        self, model, test_frame: np.ndarray
    ) -> Dict[str, float]:
        """Measure model efficiency metrics

        Args:
            model: Pose estimation model
            test_frame: Test frame for inference

        Returns:
            Dictionary with efficiency metrics
        """
        try:
            # Get model size
            model_size_mb = self._get_model_size_mb(model)

            # Measure inference time
            timing_metrics = self.measure_inference_time(model, test_frame, num_runs=5)

            # Measure memory usage
            memory_metrics = self.measure_memory_usage(model, test_frame, num_runs=3)

            if not timing_metrics or not memory_metrics:
                return {}

            avg_inference_time = timing_metrics.get("avg_inference_time_ms", 0)
            avg_memory_usage = memory_metrics.get("avg_memory_increase_mb", 0)

            # Calculate efficiency metrics
            efficiency_metrics = {
                "model_size_mb": model_size_mb,
                "inference_time_ms": avg_inference_time,
                "memory_usage_mb": avg_memory_usage,
                "speed_memory_ratio": avg_inference_time / max(avg_memory_usage, 1.0),
                "efficiency_score": 1.0
                / (
                    1.0
                    + avg_inference_time * avg_memory_usage / max(model_size_mb, 1.0)
                ),
                "throughput_per_mb": (
                    (1000.0 / avg_inference_time) / max(model_size_mb, 1.0)
                    if avg_inference_time > 0
                    else 0
                ),
            }

            return efficiency_metrics

        except Exception as e:
            logger.warning(f"Model efficiency measurement failed: {e}")
            return {}

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        elif self.device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            try:
                return torch.mps.current_allocated_memory() / (1024**2)
            except Exception:
                pass

        # Fallback to process memory
        return self.process.memory_info().rss / (1024**2)

    def _get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        elif self.device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            try:
                return torch.mps.current_allocated_memory() / (1024**2)
            except Exception:
                pass

        # Fallback to process memory
        return self.process.memory_info().rss / (1024**2)

    def _get_model_size_mb(self, model) -> float:
        """Get model size in MB"""
        try:
            # Try to get from model's performance metrics
            if hasattr(model, "get_performance_metrics"):
                metrics = model.get_performance_metrics()
                return metrics.get("model_size_mb", 0.0)

            # Try to calculate from model parameters
            if hasattr(model, "model") and hasattr(model.model, "parameters"):
                total_params = sum(p.numel() for p in model.model.parameters())
                # Rough estimate: 4 bytes per parameter for float32
                return total_params * 4 / (1024**2)

            # Fallback to default estimates based on model type
            model_name = model.__class__.__name__.lower()
            if "mediapipe" in model_name:
                return 5.0
            elif "blazepose" in model_name:
                return 150.0
            elif "mmpose" in model_name:
                return 180.0
            elif "yolo" in model_name:
                return 6.2
            elif "pytorch" in model_name:
                return 25.0
            else:
                return 50.0  # Default estimate

        except Exception as e:
            logger.warning(f"Failed to get model size: {e}")
            return 50.0  # Default fallback

    def get_comprehensive_metrics(
        self, model, test_frames: List[np.ndarray]
    ) -> Dict[str, float]:
        """Get comprehensive performance metrics

        Args:
            model: Pose estimation model
            test_frames: List of test frames

        Returns:
            Dictionary with all performance metrics
        """
        if not test_frames:
            return {}

        test_frame = test_frames[0]

        # Collect all metrics
        all_metrics = {}

        # Timing metrics
        timing_metrics = self.measure_inference_time(model, test_frame)
        all_metrics.update(timing_metrics)

        # Memory metrics
        memory_metrics = self.measure_memory_usage(model, test_frame)
        all_metrics.update(memory_metrics)

        # Throughput metrics
        throughput_metrics = self.measure_throughput(model, test_frames)
        all_metrics.update(throughput_metrics)

        # CPU metrics
        cpu_metrics = self.measure_cpu_utilization(model, test_frame)
        all_metrics.update(cpu_metrics)

        # Efficiency metrics
        efficiency_metrics = self.measure_model_efficiency(model, test_frame)
        all_metrics.update(efficiency_metrics)

        return all_metrics
