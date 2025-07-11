"""
Visualization utilities for pose estimation results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
import pandas as pd
from pathlib import Path
import json


class VisualizationUtils:
    """Utilities for visualizing pose estimation results and comparisons"""

    def __init__(self):
        """Initialize visualization utilities"""
        # Set up matplotlib style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # COCO skeleton connections
        self.coco_skeleton = [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ]

        # Colors for different models
        self.model_colors = {
            "mediapipe": (0, 255, 0),  # Green
            "mmpose": (255, 0, 0),  # Red (includes HRNet backbones)
            "yolov8_pose": (0, 0, 255),  # Blue
            "blazepose": (255, 0, 255),  # Magenta
        }

    def draw_pose_on_image(
        self,
        image: np.ndarray,
        pose_result: Dict[str, Any],
        model_name: str = "unknown",
        thickness: int = 2,
        radius: int = 3,
    ) -> np.ndarray:
        """Draw pose keypoints and skeleton on image

        Args:
            image: Input image (H, W, C)
            pose_result: Pose estimation results
            model_name: Name of the model for color coding
            thickness: Line thickness for skeleton
            radius: Circle radius for keypoints

        Returns:
            Image with pose visualization
        """
        vis_image = image.copy()

        if pose_result.get("num_persons", 0) == 0:
            return vis_image

        # Get color for this model
        color = self.model_colors.get(model_name, (255, 255, 255))

        keypoints = pose_result["keypoints"]
        scores = pose_result.get("scores", None)

        # Draw each person
        for person_idx in range(keypoints.shape[0]):
            person_kpts = keypoints[person_idx]
            person_scores = scores[person_idx] if scores is not None else None

            # Draw keypoints
            for kpt_idx, (x, y) in enumerate(person_kpts[:, :2]):
                if person_scores is None or person_scores[kpt_idx] > 0.3:
                    cv2.circle(vis_image, (int(x), int(y)), radius, color, -1)

            # Draw skeleton
            self._draw_skeleton(vis_image, person_kpts, person_scores, color, thickness)

        # Add model name label
        cv2.putText(
            vis_image, model_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

        return vis_image

    def create_comparison_grid(
        self,
        image: np.ndarray,
        pose_results: Dict[str, Dict[str, Any]],
        output_path: str = None,
    ) -> np.ndarray:
        """Create comparison grid showing multiple model results

        Args:
            image: Original image
            pose_results: Dictionary of model_name -> pose_result
            output_path: Optional path to save the grid

        Returns:
            Grid image with all model comparisons
        """
        num_models = len(pose_results)
        if num_models == 0:
            return image

        # Calculate grid dimensions
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols

        # Create grid
        h, w = image.shape[:2]
        grid_image = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # Fill grid with model results
        for idx, (model_name, pose_result) in enumerate(pose_results.items()):
            row = idx // cols
            col = idx % cols

            # Draw pose on image copy
            vis_image = self.draw_pose_on_image(image, pose_result, model_name)

            # Place in grid
            y_start = row * h
            y_end = (row + 1) * h
            x_start = col * w
            x_end = (col + 1) * w

            grid_image[y_start:y_end, x_start:x_end] = vis_image

        # Save if requested
        if output_path:
            cv2.imwrite(output_path, grid_image)

        return grid_image

    def plot_performance_comparison(
        self, results: Dict[str, Dict[str, Any]], output_path: str = None
    ) -> plt.Figure:
        """Create performance comparison plots

        Args:
            results: Model comparison results
            output_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        # Extract performance metrics
        models = []
        inference_times = []
        memory_usage = []
        fps_values = []

        for model_name, model_results in results.items():
            if model_name == "summary" or "error" in model_results:
                continue

            models.append(model_name)
            inference_times.append(model_results.get("mean_inference_time_ms", 0))
            memory_usage.append(model_results.get("process_ram_peak_mb", 0))
            fps_values.append(model_results.get("fps", 0))

        if not models:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "No valid results to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Pose Estimation Model Performance Comparison", fontsize=16)

        # Inference time comparison
        axes[0, 0].bar(
            models, inference_times, color=sns.color_palette("husl", len(models))
        )
        axes[0, 0].set_title("Inference Time Comparison")
        axes[0, 0].set_ylabel("Time (ms)")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Memory usage comparison
        axes[0, 1].bar(
            models, memory_usage, color=sns.color_palette("husl", len(models))
        )
        axes[0, 1].set_title("Memory Usage Comparison")
        axes[0, 1].set_ylabel("Memory (MB)")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # FPS comparison
        axes[1, 0].bar(models, fps_values, color=sns.color_palette("husl", len(models)))
        axes[1, 0].set_title("Frames Per Second Comparison")
        axes[1, 0].set_ylabel("FPS")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Speed vs Memory scatter plot
        axes[1, 1].scatter(
            memory_usage, fps_values, s=100, c=range(len(models)), cmap="viridis"
        )
        axes[1, 1].set_xlabel("Memory Usage (MB)")
        axes[1, 1].set_ylabel("FPS")
        axes[1, 1].set_title("Speed vs Memory Trade-off")

        # Add model labels to scatter plot
        for i, model in enumerate(models):
            axes[1, 1].annotate(
                model,
                (memory_usage[i], fps_values[i]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_accuracy_comparison(
        self, results: Dict[str, Dict[str, Any]], output_path: str = None
    ) -> plt.Figure:
        """Create accuracy comparison plots

        Args:
            results: Model comparison results with accuracy metrics
            output_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        # Extract accuracy metrics
        models = []
        pck_scores = []
        detection_f1 = []
        temporal_smoothness = []

        for model_name, model_results in results.items():
            if model_name == "summary" or "error" in model_results:
                continue

            models.append(model_name)
            pck_scores.append(model_results.get("pose_pck_0.2_mean", 0))
            detection_f1.append(model_results.get("pose_detection_f1_mean", 0))
            temporal_smoothness.append(
                model_results.get("pose_temporal_smoothness_mean", 0)
            )

        if not models:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "No accuracy results to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Create radar chart for accuracy metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # PCK comparison
        ax1.bar(models, pck_scores, color=sns.color_palette("viridis", len(models)))
        ax1.set_title("PCK@0.2 Accuracy Comparison")
        ax1.set_ylabel("PCK@0.2 Score")
        ax1.tick_params(axis="x", rotation=45)
        ax1.set_ylim(0, 1)

        # Create radar chart for multiple metrics
        metrics = ["PCK@0.2", "Detection F1", "Temporal Smoothness"]

        # Normalize values to 0-1 range for radar chart
        metric_values = np.array([pck_scores, detection_f1, temporal_smoothness]).T

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        ax2 = plt.subplot(122, projection="polar")

        colors = sns.color_palette("husl", len(models))
        for i, model in enumerate(models):
            values = metric_values[i].tolist()
            values += values[:1]  # Complete the circle

            ax2.plot(angles, values, "o-", linewidth=2, label=model, color=colors[i])
            ax2.fill(angles, values, alpha=0.25, color=colors[i])

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title("Multi-Metric Accuracy Comparison")
        ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig

    def create_detailed_report(
        self, results: Dict[str, Dict[str, Any]], output_dir: str
    ) -> None:
        """Create detailed visualization report

        Args:
            results: Model comparison results
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Performance comparison
        perf_fig = self.plot_performance_comparison(results)
        perf_fig.savefig(
            output_path / "performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close(perf_fig)

        # Accuracy comparison
        acc_fig = self.plot_accuracy_comparison(results)
        acc_fig.savefig(
            output_path / "accuracy_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close(acc_fig)

        # Create summary table
        self._create_summary_table(results, output_path / "summary_table.png")

        # Save raw results as JSON
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Detailed report saved to {output_dir}")

    def _draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ):
        """Draw skeleton connections on image

        Args:
            image: Image to draw on
            keypoints: Keypoints array (K, 2/3)
            scores: Confidence scores (K,)
            color: Line color (B, G, R)
            thickness: Line thickness
        """
        for connection in self.coco_skeleton:
            if len(keypoints) > max(connection):
                pt1_idx, pt2_idx = connection
                pt1 = keypoints[pt1_idx][:2]
                pt2 = keypoints[pt2_idx][:2]

                # Check confidence scores if available
                if scores is not None:
                    if scores[pt1_idx] < 0.3 or scores[pt2_idx] < 0.3:
                        continue

                cv2.line(
                    image,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    color,
                    thickness,
                )

    def _create_summary_table(
        self, results: Dict[str, Dict[str, Any]], output_path: str
    ):
        """Create summary table visualization

        Args:
            results: Model comparison results
            output_path: Path to save table image
        """
        # Extract data for table
        table_data = []
        for model_name, model_results in results.items():
            if model_name == "summary" or "error" in model_results:
                continue

            row = {
                "Model": model_name,
                "Inference Time (ms)": f"{model_results.get('mean_inference_time_ms', 0):.1f}",
                "FPS": f"{model_results.get('fps', 0):.1f}",
                "Memory (MB)": f"{model_results.get('process_ram_peak_mb', 0):.1f}",
                "PCK@0.2": f"{model_results.get('pose_pck_0.2_mean', 0):.3f}",
                "Detection F1": f"{model_results.get('pose_detection_f1_mean', 0):.3f}",
            }
            table_data.append(row)

        if not table_data:
            return

        # Create table plot
        fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.8 + 2))
        ax.axis("tight")
        ax.axis("off")

        df = pd.DataFrame(table_data)
        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.title("Model Performance Summary", fontsize=16, fontweight="bold", pad=20)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def visualize_temporal_consistency(
        self,
        temporal_results: List[Dict[str, Any]],
        model_name: str,
        output_path: str = None,
    ) -> plt.Figure:
        """Visualize temporal consistency of pose predictions

        Args:
            temporal_results: Sequential pose prediction results
            model_name: Name of the model
            output_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        if not temporal_results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No temporal data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Extract keypoint positions over time
        frames = []
        keypoint_positions = []

        for frame_idx, result in enumerate(temporal_results):
            if result.get("num_persons", 0) > 0:
                keypoints = result["keypoints"][0]  # First person
                frames.append(frame_idx)
                keypoint_positions.append(keypoints[:, :2])  # x, y coordinates

        if not frames:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No valid poses detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Calculate movement over time for key joints
        key_joints = [0, 5, 6, 11, 12]  # nose, shoulders, hips
        joint_names = ["Nose", "L_Shoulder", "R_Shoulder", "L_Hip", "R_Hip"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot position changes
        for joint_idx, joint_name in zip(key_joints, joint_names):
            x_positions = [
                pos[joint_idx][0] for pos in keypoint_positions if joint_idx < len(pos)
            ]
            y_positions = [
                pos[joint_idx][1] for pos in keypoint_positions if joint_idx < len(pos)
            ]

            axes[0].plot(
                frames[: len(x_positions)],
                x_positions,
                label=f"{joint_name}_X",
                alpha=0.7,
            )
            axes[1].plot(
                frames[: len(y_positions)],
                y_positions,
                label=f"{joint_name}_Y",
                alpha=0.7,
            )

        axes[0].set_title(f"Temporal Consistency: {model_name} - X Coordinates")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("X Position (pixels)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Temporal Consistency: {model_name} - Y Coordinates")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Y Position (pixels)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig
