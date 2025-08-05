#!/usr/bin/env python3
"""
Test script for zoom-aware data loading functionality.

This script demonstrates how the framework handles zoom variations to prevent
data leakage while maintaining balanced distribution across zoom levels.

NOTE: This test uses some internal methods for validation purposes.
Production code should use DataSelectionManager for centralized data selection.
"""

import sys
from pathlib import Path
import json

# Add the current directory to the path (for direct imports)
sys.path.append(str(Path(__file__).parent))

from data_handling.data_loader import SurfingDataLoader


def test_zoom_aware_loading():
    """Test the zoom-aware data loading functionality."""

    # Configuration for testing (updated structure)
    config = {
        "data_source": {
            "base_data_path": "./data/SD_02_SURF_FOOTAGE_PREPT",
            "video_clips": {
                "h264_path": "03_CLIPPED/h264",
                "ffv1_path": "03_CLIPPED/ffv1",
                "input_format": "h264",
            },
            "annotations": {
                "labels_path": "04_ANNOTATED/EXPORTED-MANEUVER-LABELS",
                "sony_300_labels": "sony_300",
                "sony_70_labels": "sony_70",
            },
            "splits": {
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "random_seed": 42,
                "zoom_handling": {
                    "enabled": True,
                    "balanced_distribution": True,
                    "target_distribution": {
                        "default": 0.33,
                        "wide": 0.33,
                        "full": 0.34,
                    },
                },
            },
            "camera_selection": {"enabled_cameras": ["SONY_300", "SONY_70"]},
        }
    }

    print("🔍 Testing Zoom-Aware Data Loading")
    print("=" * 50)
    print("⚠️  NOTE: This test uses internal methods for validation.")
    print("   Production code should use DataSelectionManager instead.")

    # Initialize data loader
    loader = SurfingDataLoader(config)

    # Load annotations
    print("\n📋 Loading annotations...")
    annotations = loader.load_annotations()
    print(f"   Found annotations for {len(annotations)} video files")

    # Discover clips with zoom-aware processing
    print("\n🎥 Discovering video clips with zoom-aware processing...")
    clips = loader.discover_video_clips("h264")
    print(f"   Selected {len(clips)} clips total")

    # Analyze zoom distribution
    zoom_counts = {}
    base_clip_counts = {}

    for clip in clips:
        # Count zoom levels
        zoom_counts[clip.zoom_level] = zoom_counts.get(clip.zoom_level, 0) + 1

        # Count unique base clips to ensure no duplicates
        if clip.base_clip_id in base_clip_counts:
            print(f"⚠️  WARNING: Duplicate base clip detected: {clip.base_clip_id}")
            print(f"   Existing: {base_clip_counts[clip.base_clip_id]}")
            print(f"   Current:  {clip.file_path}")
        base_clip_counts[clip.base_clip_id] = clip.file_path

    print(f"\n📊 Zoom Distribution Analysis:")
    total_clips = len(clips)
    for zoom_level, count in zoom_counts.items():
        percentage = (count / total_clips) * 100
        print(f"   {zoom_level:>7}: {count:>3} clips ({percentage:>5.1f}%)")

    print(f"\n✅ Data Leakage Check:")
    print(f"   Unique base clips: {len(base_clip_counts)}")
    print(f"   Total clips:       {total_clips}")
    print(f"   Ratio:             {total_clips/len(base_clip_counts):.2f}")

    if total_clips == len(base_clip_counts):
        print("   ✅ PASS: No data leakage detected (1:1 ratio)")
    else:
        print("   ❌ FAIL: Potential data leakage detected!")

    # Create data splits (for testing purposes only)
    print(f"\n🔄 Creating train/val/test splits...")
    loader.all_clips = clips  # Set clips for split creation
    splits = loader.create_data_splits()

    # Analyze splits
    stats = splits.get_split_stats()
    print(f"\n📈 Split Statistics:")

    for split_name in ["train", "val", "test"]:
        split_stats = stats[split_name]
        print(f"\n   {split_name.upper()} SET:")
        print(f"     Clips:     {split_stats['num_clips']}")
        print(f"     Duration:  {split_stats['total_duration']:.1f}s")
        print(f"     Maneuvers: {split_stats['total_maneuvers']}")
        print(f"     Zoom distribution:")
        for zoom, count in split_stats["zoom_distribution"].items():
            pct = (
                (count / split_stats["num_clips"]) * 100
                if split_stats["num_clips"] > 0
                else 0
            )
            print(f"       {zoom:>7}: {count:>2} ({pct:>4.1f}%)")

    # Check for cross-split leakage
    print(f"\n🔒 Cross-Split Leakage Check:")
    train_base_clips = {clip.base_clip_id for clip in splits.train}
    val_base_clips = {clip.base_clip_id for clip in splits.val}
    test_base_clips = {clip.base_clip_id for clip in splits.test}

    train_val_overlap = train_base_clips & val_base_clips
    train_test_overlap = train_base_clips & test_base_clips
    val_test_overlap = val_base_clips & test_base_clips

    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("   ✅ PASS: No base clip appears in multiple splits")
    else:
        print("   ❌ FAIL: Base clip leakage detected!")
        if train_val_overlap:
            print(f"      Train/Val overlap: {train_val_overlap}")
        if train_test_overlap:
            print(f"      Train/Test overlap: {train_test_overlap}")
        if val_test_overlap:
            print(f"      Val/Test overlap: {val_test_overlap}")

    # Show some example clips
    print(f"\n🎬 Example Selected Clips:")
    example_clips = clips[:5]
    for i, clip in enumerate(example_clips):
        print(f"   {i+1}. {clip.video_id}")
        print(f"      File: {Path(clip.file_path).name}")
        print(f"      Zoom: {clip.zoom_level}")
        print(f"      Base: {clip.base_clip_id}")
        print(f"      Maneuvers: {len(clip.annotations)}")

    print(f"\n🎉 Zoom-aware data loading test completed!")
    print("\n💡 For production use:")
    print("   • Use DataSelectionManager for centralized data selection")
    print("   • Generate selection manifests for reproducible experiments")
    print("   • Load data from manifests using load_maneuvers_from_manifest()")


if __name__ == "__main__":
    test_zoom_aware_loading()
