#!/usr/bin/env python3
"""
Regenerate visualization videos from existing prediction files
Uses the fixed visualization code with your saved predictions
"""

import sys
import json
import logging
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.pose_video_visualizer import PoseVideoVisualizer
from data_handling.data_loader import SurfingDataLoader


def regenerate_visualizations_from_predictions(run_path: str):
    """Regenerate visualizations using existing prediction files"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    run_dir = Path(run_path)
    if not run_dir.exists():
        logger.error(f"❌ Run directory not found: {run_dir}")
        return False

    predictions_dir = run_dir / "predictions"
    visualizations_dir = run_dir / "visualizations"

    if not predictions_dir.exists():
        logger.error(f"❌ No predictions directory found: {predictions_dir}")
        return False

    # Create visualizations directory
    visualizations_dir.mkdir(exist_ok=True)

    logger.info(f"🎬 Regenerating visualizations from: {predictions_dir}")
    logger.info(f"📁 Output directory: {visualizations_dir}")

    # Load comparison config to get encoding settings
    config_files = list(run_dir.glob("comparison_config_*.yaml"))
    encoding_config = {}

    if config_files:
        try:
            with open(config_files[0], "r") as f:
                config = yaml.safe_load(f)
            encoding_config = (
                config.get("output", {}).get("visualization", {}).get("encoding", {})
            )
        except Exception as e:
            logger.warning(f"Could not read config: {e}")

    # Initialize visualizer
    visualizer = PoseVideoVisualizer(encoding_config)

    # Find all model directories
    model_dirs = [d for d in predictions_dir.iterdir() if d.is_dir()]

    total_created = 0

    for model_dir in model_dirs:
        model_name = model_dir.name
        logger.info(f"\n🔄 Processing {model_name}...")

        # Find all prediction files for this model
        prediction_files = list(model_dir.glob("*_predictions.json"))

        if not prediction_files:
            logger.warning(f"⚠️ No prediction files found for {model_name}")
            continue

        logger.info(f"Found {len(prediction_files)} prediction files")

        model_created = 0

        for pred_file in prediction_files:
            try:
                # Load prediction data
                with open(pred_file, "r") as f:
                    prediction_data = json.load(f)

                # Extract video information
                video_path = prediction_data["video_path"]
                maneuver_id = prediction_data["maneuver_id"]

                # Check if video file exists
                if not Path(video_path).exists():
                    logger.warning(f"⚠️ Video file not found: {video_path}")
                    continue

                # Convert prediction frames to visualization format
                pose_results = []
                for frame_data in prediction_data["frames"]:
                    if frame_data["persons"]:
                        # Convert to the format expected by visualizer
                        frame_result = {
                            "num_persons": len(frame_data["persons"]),
                            "keypoints": [],
                            "scores": [],
                            "bbox": [],
                        }

                        for person in frame_data["persons"]:
                            keypoints = []
                            scores = []

                            for kpt in person["keypoints"]:
                                keypoints.append([kpt["x"], kpt["y"]])
                                scores.append(kpt["confidence"])

                            frame_result["keypoints"].append(keypoints)
                            frame_result["scores"].append(scores)

                            # Add bbox if available
                            if person.get("bbox"):
                                bbox = person["bbox"]
                                frame_result["bbox"].append(
                                    [
                                        bbox["x1"],
                                        bbox["y1"],
                                        bbox["x2"],
                                        bbox["y2"],
                                        bbox.get("confidence", 1.0),
                                    ]
                                )

                        pose_results.append(frame_result)
                    else:
                        # Empty frame
                        pose_results.append(
                            {
                                "num_persons": 0,
                                "keypoints": [],
                                "scores": [],
                                "bbox": [],
                            }
                        )

                if not pose_results:
                    logger.warning(f"⚠️ No pose data found in {pred_file.name}")
                    continue

                # Create output filename
                video_stem = Path(video_path).stem
                output_filename = f"{model_name}_{maneuver_id}_visualization.mp4"
                output_path = visualizations_dir / output_filename

                # Get maneuver frame range
                start_frame = prediction_data.get("start_frame", 0)
                end_frame = prediction_data.get(
                    "end_frame", start_frame + len(pose_results)
                )

                logger.info(f"  • Creating: {output_filename}")

                # Create visualization
                success = visualizer.create_pose_visualization_video(
                    video_path=video_path,
                    pose_results=pose_results,
                    output_path=str(output_path),
                    model_name=model_name,
                    maneuver_start_frame=start_frame,
                    maneuver_end_frame=end_frame,
                )

                if success:
                    model_created += 1
                    total_created += 1
                    file_size = output_path.stat().st_size / (1024 * 1024)
                    logger.info(f"    ✅ Created ({file_size:.2f} MB)")
                else:
                    logger.warning(f"    ❌ Failed to create visualization")

            except Exception as e:
                logger.error(f"❌ Error processing {pred_file.name}: {e}")
                continue

        logger.info(f"✅ {model_name}: Created {model_created} visualizations")

    logger.info(f"\n🎉 Complete! Created {total_created} visualization videos")
    logger.info(f"📁 Visualizations saved to: {visualizations_dir}")

    # List created files
    viz_files = list(visualizations_dir.glob("*.mp4"))
    if viz_files:
        logger.info(f"\n📋 Created files:")
        for viz_file in sorted(viz_files):
            file_size = viz_file.stat().st_size / (1024 * 1024)
            logger.info(f"  • {viz_file.name} ({file_size:.2f} MB)")

    return total_created > 0


def main():
    if len(sys.argv) != 2:
        print("Usage: python regenerate_visualizations.py <run_directory>")
        print(
            "Example: python regenerate_visualizations.py data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/runs/20250716_171307_production_test_4090_2clips_all_models"
        )
        sys.exit(1)

    run_path = sys.argv[1]
    success = regenerate_visualizations_from_predictions(run_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
