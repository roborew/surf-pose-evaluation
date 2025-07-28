#!/usr/bin/env python3
"""
Generate memory visualizations for test run
"""

from pathlib import Path
import sys

sys.path.append(".")

from utils.memory_visualizer import create_memory_visualizations
import logging


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Path to your test run
    run_dir = Path(
        "data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/runs/20250728_173010_test_run_5"
    )
    report_path = run_dir / "memory_profiling_report.json"

    print(f"🎨 Generating memory visualizations for test run...")
    print(f"📊 Report: {report_path}")
    print(f"📁 Output: {run_dir}")

    try:
        # Generate all visualizations
        results = create_memory_visualizations(report_path, run_dir)

        print("\n✅ Generated visualizations:")
        for viz_type, path in results.items():
            print(f"  📈 {viz_type}: {path}")

        print(f"\n🎯 Open these files to view your memory profiling data:")
        print(f"  Dashboard: open {run_dir}/memory_profiling_dashboard.png")
        print(f"  Timeline: open {run_dir}/memory_timeline.png")
        print(f"  CSV Data: {run_dir}/memory_data.csv")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
