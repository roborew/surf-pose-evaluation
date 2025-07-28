#!/usr/bin/env python3
"""
Memory Profiling Visualization Utility
Generates time-based graphs from memory profiling data
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryVisualizer:
    """Utility for visualizing memory profiling data"""

    def __init__(self, report_path: Path):
        """
        Initialize visualizer with memory profiling report

        Args:
            report_path: Path to memory_profiling_report.json
        """
        self.report_path = Path(report_path)
        self.data = None
        self.df = None
        self.load_data()

    def load_data(self):
        """Load memory profiling data from report"""
        try:
            with open(self.report_path, "r") as f:
                self.data = json.load(f)

            # Convert snapshots to DataFrame for easier plotting
            snapshots = self.data.get("all_snapshots", [])
            if snapshots:
                self.df = pd.DataFrame(snapshots)

                # Convert timestamps to datetime objects
                if "timestamp" in self.df.columns:
                    self.df["datetime"] = pd.to_datetime(self.df["timestamp"], unit="s")

                logger.info(
                    f"Loaded {len(snapshots)} memory snapshots for visualization"
                )
            else:
                logger.warning("No snapshot data found in report")

        except Exception as e:
            logger.error(f"Failed to load memory profiling data: {e}")
            raise

    def create_comprehensive_dashboard(self, output_dir: Optional[Path] = None) -> Path:
        """
        Create comprehensive memory profiling dashboard

        Args:
            output_dir: Directory to save plots (defaults to same as report)

        Returns:
            Path to the saved dashboard image
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available for visualization")

        if output_dir is None:
            output_dir = self.report_path.parent

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Memory Profiling Dashboard", fontsize=16, fontweight="bold")

        # Plot 1: Memory Usage Over Time
        ax1 = axes[0, 0]
        ax1.plot(
            self.df["elapsed_time"],
            self.df["process_memory_rss_mb"],
            label="Process Memory (RSS)",
            linewidth=2,
            color="blue",
        )
        ax1.plot(
            self.df["elapsed_time"],
            self.df["process_memory_vms_mb"],
            label="Virtual Memory (VMS)",
            linewidth=1,
            alpha=0.7,
            color="lightblue",
        )

        if "tracemalloc_current_mb" in self.df.columns:
            ax1.plot(
                self.df["elapsed_time"],
                self.df["tracemalloc_current_mb"],
                label="Python Memory (tracemalloc)",
                linewidth=1,
                alpha=0.8,
                color="green",
            )

        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Memory (MB)")
        ax1.set_title("Memory Usage Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: System Memory and CPU
        ax2 = axes[0, 1]
        ax2_cpu = ax2.twinx()

        # System memory percentage
        line1 = ax2.plot(
            self.df["elapsed_time"],
            self.df["system_memory_percent"],
            color="orange",
            linewidth=2,
            label="System Memory %",
        )

        # CPU percentage on secondary y-axis
        line2 = ax2_cpu.plot(
            self.df["elapsed_time"],
            self.df["cpu_percent"],
            color="red",
            linewidth=2,
            label="CPU %",
        )

        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("System Memory (%)", color="orange")
        ax2_cpu.set_ylabel("CPU Usage (%)", color="red")
        ax2.set_title("System Resources Over Time")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper left")
        ax2.grid(True, alpha=0.3)

        # Plot 3: GPU Memory (if available)
        ax3 = axes[1, 0]
        if self.df["gpu_available"].any():
            ax3.plot(
                self.df["elapsed_time"],
                self.df["gpu_allocated_mb"],
                label="GPU Allocated",
                linewidth=2,
                color="purple",
            )
            ax3.plot(
                self.df["elapsed_time"],
                self.df["gpu_reserved_mb"],
                label="GPU Reserved",
                linewidth=1,
                alpha=0.7,
                color="mediumpurple",
            )
            ax3.set_ylabel("GPU Memory (MB)")
            ax3.set_title("GPU Memory Usage Over Time")
            ax3.legend()
        else:
            ax3.text(
                0.5,
                0.5,
                "GPU Not Available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=14,
                alpha=0.5,
            )
            ax3.set_title("GPU Memory Usage (Not Available)")

        ax3.set_xlabel("Time (seconds)")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Memory Growth Rate
        ax4 = axes[1, 1]
        if len(self.df) > 1:
            # Calculate memory growth rate (MB/second)
            time_diff = self.df["elapsed_time"].diff()
            memory_diff = self.df["process_memory_rss_mb"].diff()
            growth_rate = memory_diff / time_diff

            # Smooth the growth rate to reduce noise
            window_size = min(10, len(growth_rate) // 5)
            if window_size > 1:
                growth_rate_smooth = growth_rate.rolling(
                    window=window_size, center=True
                ).mean()
            else:
                growth_rate_smooth = growth_rate

            ax4.plot(
                self.df["elapsed_time"][1:],
                growth_rate_smooth[1:],
                linewidth=2,
                color="darkgreen",
                alpha=0.8,
            )
            ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax4.axhline(
                y=1,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="1 MB/s (potential concern)",
            )
            ax4.set_ylabel("Growth Rate (MB/s)")
            ax4.set_title("Memory Growth Rate Over Time")
            ax4.legend()
        else:
            ax4.text(
                0.5,
                0.5,
                "Insufficient Data",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=14,
                alpha=0.5,
            )
            ax4.set_title("Memory Growth Rate (Insufficient Data)")

        ax4.set_xlabel("Time (seconds)")
        ax4.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()

        # Add summary statistics as text
        if self.data and "summary" in self.data:
            summary = self.data["summary"]
            stats_text = self._generate_stats_text(summary)
            fig.text(
                0.02,
                0.02,
                stats_text,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

        # Save dashboard
        dashboard_path = output_dir / "memory_profiling_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Memory profiling dashboard saved to {dashboard_path}")
        return dashboard_path

    def create_timeline_plot(self, output_dir: Optional[Path] = None) -> Path:
        """
        Create a detailed timeline plot showing milestones

        Args:
            output_dir: Directory to save plot

        Returns:
            Path to saved plot
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available for visualization")

        if output_dir is None:
            output_dir = self.report_path.parent

        # Create timeline plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot main memory line
        ax.plot(
            self.df["elapsed_time"],
            self.df["process_memory_rss_mb"],
            linewidth=3,
            color="blue",
            alpha=0.8,
            label="Process Memory (MB)",
        )

        # Add milestone markers (would need to parse from MLflow or logs)
        # For now, we can identify potential milestones from memory changes
        memory_changes = self.df["process_memory_rss_mb"].diff().abs()
        significant_changes = memory_changes > memory_changes.quantile(0.9)

        if significant_changes.any():
            milestone_times = self.df.loc[significant_changes, "elapsed_time"]
            milestone_memory = self.df.loc[significant_changes, "process_memory_rss_mb"]

            ax.scatter(
                milestone_times,
                milestone_memory,
                color="red",
                s=100,
                alpha=0.7,
                zorder=5,
                label="Significant Memory Changes",
            )

        # Add secondary y-axis for CPU
        ax2 = ax.twinx()
        ax2.plot(
            self.df["elapsed_time"],
            self.df["cpu_percent"],
            color="orange",
            alpha=0.6,
            linewidth=2,
            label="CPU %",
        )

        # Formatting
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Memory (MB)", fontsize=12, color="blue")
        ax2.set_ylabel("CPU Usage (%)", fontsize=12, color="orange")
        ax.set_title("Detailed Evaluation Timeline", fontsize=14, fontweight="bold")

        # Legends
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

        # Grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save timeline
        timeline_path = output_dir / "memory_timeline.png"
        plt.savefig(timeline_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Memory timeline plot saved to {timeline_path}")
        return timeline_path

    def _generate_stats_text(self, summary: Dict[str, Any]) -> str:
        """Generate summary statistics text for plots"""
        if "statistics" not in summary:
            return ""

        stats = summary["statistics"]
        text_lines = []

        # Duration
        duration = stats.get("duration_seconds", 0)
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = (
            f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            if hours > 0
            else f"{minutes:02d}:{seconds:02d}"
        )
        text_lines.append(f"Duration: {duration_str}")

        # Memory stats
        if "process_memory" in stats:
            mem = stats["process_memory"]
            text_lines.append(f"Peak Memory: {mem.get('peak_mb', 0):.1f} MB")
            text_lines.append(
                f"Memory Increase: {mem.get('increase_from_start_mb', 0):.1f} MB"
            )

        # CPU stats
        if "cpu" in stats:
            cpu = stats["cpu"]
            text_lines.append(f"Peak CPU: {cpu.get('peak_percent', 0):.1f}%")
            text_lines.append(f"Avg CPU: {cpu.get('mean_percent', 0):.1f}%")

        # GPU stats
        if "gpu" in stats:
            gpu = stats["gpu"]
            text_lines.append(f"Peak GPU: {gpu.get('peak_mb', 0):.1f} MB")

        # Analysis
        if "analysis" in summary:
            analysis = summary["analysis"]
            efficiency = analysis.get("memory_efficiency", "unknown")
            text_lines.append(f"Memory Efficiency: {efficiency.title()}")

            growth_rate = analysis.get("memory_growth_rate_mb_per_second", 0)
            text_lines.append(f"Growth Rate: {growth_rate:.3f} MB/s")

        return "\n".join(text_lines)

    def export_csv_data(self, output_path: Optional[Path] = None) -> Path:
        """
        Export memory profiling data to CSV for further analysis

        Args:
            output_path: Path for CSV file

        Returns:
            Path to saved CSV file
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available for export")

        if output_path is None:
            output_path = self.report_path.parent / "memory_profiling_data.csv"

        # Select relevant columns for export
        export_columns = [
            "elapsed_time",
            "process_memory_rss_mb",
            "process_memory_vms_mb",
            "system_memory_percent",
            "cpu_percent",
        ]

        # Add GPU columns if available
        if "gpu_allocated_mb" in self.df.columns:
            export_columns.extend(
                ["gpu_allocated_mb", "gpu_reserved_mb", "gpu_utilization_percent"]
            )

        # Add tracemalloc columns if available
        if "tracemalloc_current_mb" in self.df.columns:
            export_columns.extend(["tracemalloc_current_mb", "tracemalloc_peak_mb"])

        # Filter columns that exist in the dataframe
        export_columns = [col for col in export_columns if col in self.df.columns]

        self.df[export_columns].to_csv(output_path, index=False)

        logger.info(f"Memory profiling data exported to {output_path}")
        return output_path


def create_memory_visualizations(
    report_path: Path, output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Convenience function to create all memory visualizations

    Args:
        report_path: Path to memory profiling report
        output_dir: Directory to save visualizations

    Returns:
        Dictionary mapping visualization type to file path
    """
    visualizer = MemoryVisualizer(report_path)

    if output_dir is None:
        output_dir = report_path.parent

    results = {}

    try:
        results["dashboard"] = visualizer.create_comprehensive_dashboard(output_dir)
        results["timeline"] = visualizer.create_timeline_plot(output_dir)
        results["csv_data"] = visualizer.export_csv_data(output_dir / "memory_data.csv")

        logger.info(f"Created {len(results)} memory visualizations in {output_dir}")

    except Exception as e:
        logger.error(f"Failed to create memory visualizations: {e}")
        raise

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate memory profiling visualizations"
    )
    parser.add_argument("report_path", help="Path to memory_profiling_report.json")
    parser.add_argument("--output-dir", help="Output directory for visualizations")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir) if args.output_dir else None
    results = create_memory_visualizations(Path(args.report_path), output_dir)

    print("Generated visualizations:")
    for viz_type, path in results.items():
        print(f"  {viz_type}: {path}")
