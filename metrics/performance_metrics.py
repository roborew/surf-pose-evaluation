"""
Performance metrics for pose estimation models
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import gc


class PerformanceMetrics:
    """Performance measurement utilities for pose estimation models"""

    def __init__(self, device: str = "cpu"):
        """Initialize performance metrics

        Args:
            device: Compute device ('cpu', 'cuda', 'mps')
        """
        self.device = device
        self.reset_counters()

    def reset_counters(self):
        """Reset all performance counters"""
        self.inference_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_memory_usage = []
        self.throughput_samples = []

    def start_monitoring(self, monitor_system: bool = True):
        """Start system resource monitoring

        Args:
            monitor_system: Whether to monitor system-wide resources
        """
        self.monitoring = True
        if monitor_system:
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=1.0)

    def measure_inference_time(
        self, model, input_data, num_runs: int = 1
    ) -> Dict[str, float]:
        """Measure inference time for a model

        Args:
            model: Model to benchmark
            input_data: Input data for inference
            num_runs: Number of inference runs

        Returns:
            Dictionary with timing metrics
        """
        times = []

        # Warmup runs
        for _ in range(min(3, num_runs)):
            _ = model.predict(input_data)

        # Synchronize GPU if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Actual timing runs
        for _ in range(num_runs):
            start_time = time.perf_counter()

            result = model.predict(input_data)

            # Synchronize GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(inference_time)

        self.inference_times.extend(times)

        return {
            "mean_inference_time_ms": np.mean(times),
            "std_inference_time_ms": np.std(times),
            "min_inference_time_ms": np.min(times),
            "max_inference_time_ms": np.max(times),
            "median_inference_time_ms": np.median(times),
            "fps": 1000.0 / np.mean(times),
        }

    def measure_memory_usage(self, model, input_data) -> Dict[str, float]:
        """Measure memory usage during inference

        Args:
            model: Model to benchmark
            input_data: Input data for inference

        Returns:
            Dictionary with memory metrics
        """
        # Force garbage collection
        gc.collect()

        # Measure baseline memory
        baseline_memory = self._get_memory_usage()

        # Run inference and measure peak memory
        peak_memory = baseline_memory.copy()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Single inference run
        result = model.predict(input_data)

        # Measure peak memory
        current_memory = self._get_memory_usage()

        for key in current_memory:
            peak_memory[key] = max(peak_memory[key], current_memory[key])

        # Calculate memory increase
        memory_increase = {}
        for key in baseline_memory:
            memory_increase[f"{key}_increase_mb"] = (
                peak_memory[key] - baseline_memory[key]
            )

        return {
            **memory_increase,
            **{f"{key}_baseline_mb": value for key, value in baseline_memory.items()},
            **{f"{key}_peak_mb": value for key, value in peak_memory.items()},
        }

    def measure_throughput(
        self, model, input_batch: List, duration_seconds: float = 30.0
    ) -> Dict[str, float]:
        """Measure model throughput

        Args:
            model: Model to benchmark
            input_batch: Batch of input data
            duration_seconds: Duration to run throughput test

        Returns:
            Dictionary with throughput metrics
        """
        processed_samples = 0
        start_time = time.time()
        end_time = start_time + duration_seconds

        batch_times = []

        while time.time() < end_time:
            batch_start = time.time()

            # Process batch
            results = model.predict_batch(input_batch)

            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)

            processed_samples += len(input_batch)

        total_time = time.time() - start_time

        return {
            "total_samples_processed": processed_samples,
            "total_time_seconds": total_time,
            "samples_per_second": processed_samples / total_time,
            "mean_batch_time_ms": np.mean(batch_times) * 1000,
            "std_batch_time_ms": np.std(batch_times) * 1000,
        }

    def benchmark_model(
        self,
        model,
        test_inputs: List,
        num_timing_runs: int = 10,
        throughput_duration: float = 30.0,
    ) -> Dict[str, Any]:
        """Comprehensive model benchmarking

        Args:
            model: Model to benchmark
            test_inputs: List of test inputs
            num_timing_runs: Number of timing measurement runs
            throughput_duration: Duration for throughput test

        Returns:
            Comprehensive benchmark results
        """
        results = {}

        if not test_inputs:
            return {"error": "No test inputs provided"}

        # Single input timing
        single_input = test_inputs[0]
        timing_metrics = self.measure_inference_time(
            model, single_input, num_timing_runs
        )
        results.update(timing_metrics)

        # Memory usage
        memory_metrics = self.measure_memory_usage(model, single_input)
        results.update(memory_metrics)

        # Throughput (if multiple inputs available)
        if len(test_inputs) > 1:
            throughput_metrics = self.measure_throughput(
                model, test_inputs[: min(10, len(test_inputs))], throughput_duration
            )
            results.update(throughput_metrics)

        # Model information
        model_info = model.get_model_info()
        results["model_info"] = model_info

        # Device information
        results["device_info"] = self._get_device_info()

        return results

    def compare_models(
        self, models: Dict[str, Any], test_inputs: List
    ) -> Dict[str, Dict]:
        """Compare performance of multiple models

        Args:
            models: Dictionary of model_name -> model_instance
            test_inputs: List of test inputs

        Returns:
            Comparison results for all models
        """
        results = {}

        for model_name, model in models.items():
            print(f"Benchmarking {model_name}...")
            try:
                model_results = self.benchmark_model(model, test_inputs)
                results[model_name] = model_results
            except Exception as e:
                results[model_name] = {"error": str(e)}

        # Add comparison summary
        results["summary"] = self._create_comparison_summary(results)

        return results

    def _monitor_resources(self):
        """Monitor system resources in background thread"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)

                # Memory usage
                memory_info = psutil.virtual_memory()
                self.memory_usage.append(memory_info.percent)

                # GPU memory (if available)
                if self.device == "cuda" and torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    self.gpu_memory_usage.append(gpu_memory)

                time.sleep(0.5)  # Sample every 500ms

            except Exception:
                break

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage

        Returns:
            Dictionary with memory usage in MB
        """
        memory_usage = {}

        # System RAM
        memory_info = psutil.virtual_memory()
        memory_usage["system_ram"] = memory_info.used / 1024**2

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        memory_usage["process_ram"] = process_memory.rss / 1024**2

        # GPU memory
        if self.device == "cuda" and torch.cuda.is_available():
            memory_usage["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**2
            memory_usage["gpu_reserved"] = torch.cuda.memory_reserved() / 1024**2
            memory_usage["gpu_max_allocated"] = (
                torch.cuda.max_memory_allocated() / 1024**2
            )
        elif self.device == "mps" and torch.backends.mps.is_available():
            # MPS doesn't have direct memory querying, use process memory as proxy
            memory_usage["mps_allocated"] = process_memory.rss / 1024**2

        return memory_usage

    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information

        Returns:
            Dictionary with device details
        """
        device_info = {
            "device": self.device,
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "total_ram_gb": psutil.virtual_memory().total / 1024**3,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            device_info.update(
                {
                    "gpu_name": torch.cuda.get_device_name(),
                    "gpu_count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
                    / 1024**3,
                }
            )
        elif self.device == "mps" and torch.backends.mps.is_available():
            device_info["mps_available"] = True

        return device_info

    def _create_comparison_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create comparison summary from benchmark results

        Args:
            results: Benchmark results for all models

        Returns:
            Summary comparison metrics
        """
        summary = {
            "fastest_model": None,
            "most_memory_efficient": None,
            "highest_throughput": None,
            "speed_ranking": [],
            "memory_ranking": [],
            "throughput_ranking": [],
        }

        # Extract valid results (exclude errors)
        valid_results = {
            name: data
            for name, data in results.items()
            if "error" not in data and name != "summary"
        }

        if not valid_results:
            return summary

        # Speed comparison
        speed_data = [
            (name, data.get("mean_inference_time_ms", float("inf")))
            for name, data in valid_results.items()
        ]
        speed_data.sort(key=lambda x: x[1])
        summary["speed_ranking"] = speed_data
        summary["fastest_model"] = speed_data[0][0] if speed_data else None

        # Memory comparison
        memory_data = [
            (name, data.get("process_ram_peak_mb", float("inf")))
            for name, data in valid_results.items()
        ]
        memory_data.sort(key=lambda x: x[1])
        summary["memory_ranking"] = memory_data
        summary["most_memory_efficient"] = memory_data[0][0] if memory_data else None

        # Throughput comparison
        throughput_data = [
            (name, data.get("samples_per_second", 0))
            for name, data in valid_results.items()
        ]
        throughput_data.sort(key=lambda x: x[1], reverse=True)
        summary["throughput_ranking"] = throughput_data
        summary["highest_throughput"] = (
            throughput_data[0][0] if throughput_data else None
        )

        return summary

    def get_monitoring_summary(self) -> Dict[str, float]:
        """Get summary of monitored resources

        Returns:
            Summary of resource usage during monitoring
        """
        summary = {}

        if self.cpu_usage:
            summary.update(
                {
                    "mean_cpu_usage_percent": np.mean(self.cpu_usage),
                    "max_cpu_usage_percent": np.max(self.cpu_usage),
                    "std_cpu_usage_percent": np.std(self.cpu_usage),
                }
            )

        if self.memory_usage:
            summary.update(
                {
                    "mean_memory_usage_percent": np.mean(self.memory_usage),
                    "max_memory_usage_percent": np.max(self.memory_usage),
                    "std_memory_usage_percent": np.std(self.memory_usage),
                }
            )

        if self.gpu_memory_usage:
            summary.update(
                {
                    "mean_gpu_memory_gb": np.mean(self.gpu_memory_usage),
                    "max_gpu_memory_gb": np.max(self.gpu_memory_usage),
                    "std_gpu_memory_gb": np.std(self.gpu_memory_usage),
                }
            )

        return summary
