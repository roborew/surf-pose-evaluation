#!/usr/bin/env python3
"""
Quick analysis of memory profiling data from JSON
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_memory_data():
    # Load the memory profiling data
    report_path = Path(
        "data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/20250728_173010_test_run_5/memory_profiling_report.json"
    )

    print("📊 Loading memory profiling data...")
    with open(report_path) as f:
        data = json.load(f)

    # Extract key information
    summary = data["summary"]
    stats = summary["statistics"]
    analysis = summary["analysis"]

    print("\n🎯 Memory Profiling Summary")
    print("=" * 50)
    print(f"⏱️  Duration: {stats['duration_seconds']/60:.1f} minutes")
    print(f"📸 Snapshots: {stats['snapshots_count']:,}")
    print(f"📈 Peak Memory: {stats['process_memory']['peak_mb']:.1f} MB")
    print(
        f"📊 Memory Growth: {stats['process_memory']['increase_from_start_mb']:.1f} MB"
    )
    print(f"🖥️  Peak CPU: {stats['cpu']['peak_percent']:.1f}%")
    print(f"🎮 Peak GPU: {stats['gpu']['peak_mb']:.1f} MB")
    print(f"📉 Growth Rate: {analysis['memory_growth_rate_mb_per_second']:.2f} MB/s")
    print(f"✅ Efficiency: {analysis['memory_efficiency']}")
    print(f"⚠️  Memory Leak: {'Yes' if analysis['potential_memory_leak'] else 'No'}")

    # Convert snapshots to DataFrame
    df = pd.DataFrame(data["all_snapshots"])

    print(f"\n📋 Data Points Available:")
    print(f"   • Process Memory (RSS): {df['process_memory_rss_mb'].count():,} points")
    print(f"   • CPU Usage: {df['cpu_percent'].count():,} points")
    print(f"   • GPU Memory: {df['gpu_allocated_mb'].count():,} points")
    print(f"   • Python Memory: {df['tracemalloc_current_mb'].count():,} points")

    # Find memory phases
    print(f"\n🔍 Memory Analysis:")
    memory_values = df["process_memory_rss_mb"]
    print(f"   • Starting Memory: {memory_values.iloc[0]:.1f} MB")
    print(f"   • Peak Memory: {memory_values.max():.1f} MB")
    print(f"   • Final Memory: {memory_values.iloc[-1]:.1f} MB")
    print(f"   • Average Memory: {memory_values.mean():.1f} MB")
    print(f"   • Memory Volatility: {memory_values.std():.1f} MB")

    # Time analysis
    time_hours = df["elapsed_time"].max() / 3600
    print(f"\n⏰ Timeline:")
    print(f"   • Total Runtime: {time_hours:.2f} hours")
    print(
        f"   • Sampling Rate: {stats['snapshots_count']/stats['duration_seconds']:.1f} samples/second"
    )

    # GPU analysis if available
    if "gpu_allocated_mb" in df.columns and df["gpu_allocated_mb"].max() > 0:
        gpu_peak = df["gpu_allocated_mb"].max()
        gpu_avg = df["gpu_allocated_mb"].mean()
        print(f"\n🎮 GPU Analysis:")
        print(f"   • Peak GPU Memory: {gpu_peak:.1f} MB")
        print(f"   • Average GPU Memory: {gpu_avg:.1f} MB")
        print(f"   • GPU Utilization: {(gpu_avg/gpu_peak)*100:.1f}% of peak")

    print(f"\n💡 Recommendations:")
    if analysis["memory_growth_rate_mb_per_second"] > 1.0:
        print("   ⚠️  High memory growth rate detected - investigate potential leaks")
    if stats["process_memory"]["peak_mb"] > 8000:
        print("   📈 High peak memory usage - consider optimization")
    if stats["cpu"]["peak_percent"] > 80:
        print("   🖥️  High CPU usage detected")
    else:
        print("   ✅ Memory usage patterns look reasonable")

    print(f"\n📁 Full data available in: {report_path}")
    print("📊 Run 'python generate_memory_visualizations.py' for charts")


if __name__ == "__main__":
    analyze_memory_data()
