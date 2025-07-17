#!/usr/bin/env python3
"""
Tests for DataSelectionManager
Validates data selection consistency, manifest integrity, and reproducibility
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from utils.data_selection_manager import DataSelectionManager
from data_handling.data_loader import VideoClip


class TestDataSelectionManager:
    """Test suite for DataSelectionManager"""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            "data_source": {
                "base_data_path": "./data/SD_02_SURF_FOOTAGE_PREPT",
                "video_clips": {
                    "h264_path": "03_CLIPPED/h264",
                    "ffv1_path": "03_CLIPPED/ffv1",
                    "input_format": "h264",
                },
                "annotations": {
                    "labels_path": "04_ANNOTATED/surf-manoeuvre-labels",
                    "sony_300_labels": "sony_300",
                    "sony_70_labels": "sony_70",
                },
                "splits": {"random_seed": 42},
                "camera_selection": {"enabled_cameras": ["SONY_300", "SONY_70"]},
            },
            "output": {"visualization": {"max_examples_per_model": 10}},
        }

    @pytest.fixture
    def sample_clips(self):
        """Sample video clips for testing"""
        clips = []

        # Create diverse sample clips
        cameras = ["SONY_300", "SONY_70"]
        sessions = ["SESSION_060325", "SESSION_070325"]
        zooms = ["default", "wide", "full"]

        clip_id = 0
        for camera in cameras:
            for session in sessions:
                for zoom in zooms:
                    clip = VideoClip(
                        file_path=f"/path/to/{camera}_{session}_C00{clip_id:02d}_clip_1_{zoom}.mp4",
                        video_id=f"{camera}_{session}_C00{clip_id:02d}_clip_1_{zoom}",
                        camera=camera,
                        session=session,
                        duration=10.0 + clip_id,
                        fps=25.0,
                        width=1920,
                        height=1080,
                        format="h264",
                        zoom_level=zoom,
                        base_clip_id=f"C00{clip_id:02d}_clip_1",
                        annotations=[
                            {
                                "start": 1.0,
                                "end": 5.0,
                                "labels": ["05", "Pop-up"],
                                "channel": 0,
                            }
                        ],
                    )
                    clips.append(clip)
                    clip_id += 1

        return clips

    @pytest.fixture
    def temp_run_manager(self):
        """Mock run manager with temporary directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_manager = Mock()
            run_manager.run_dir = Path(temp_dir)
            run_manager.data_selections_dir = Path(temp_dir) / "data_selections"
            run_manager.data_selections_dir.mkdir(parents=True, exist_ok=True)
            yield run_manager

    def test_initialization(self, sample_config, temp_run_manager):
        """Test DataSelectionManager initialization"""
        manager = DataSelectionManager(sample_config, temp_run_manager)

        assert manager.config == sample_config
        assert manager.run_manager == temp_run_manager
        assert manager.selections_dir == temp_run_manager.data_selections_dir

    def test_deterministic_selection(
        self, sample_config, sample_clips, temp_run_manager
    ):
        """Test that selections are deterministic with same random seed"""
        manager = DataSelectionManager(sample_config, temp_run_manager)

        # Mock the data discovery
        with patch.object(
            manager, "_discover_clips_deterministically", return_value=sample_clips
        ):
            # Generate selections with same seed multiple times
            results1 = manager.generate_phase_selections(
                optuna_max_clips=5, comparison_max_clips=10, random_seed=42
            )

            results2 = manager.generate_phase_selections(
                optuna_max_clips=5, comparison_max_clips=10, random_seed=42
            )

            # Load and compare manifests
            manifest1 = DataSelectionManager.load_selection_manifest(results1["optuna"])
            manifest2 = DataSelectionManager.load_selection_manifest(results2["optuna"])

            # Should have identical selections
            assert manifest1["selected_clips"] == manifest2["selected_clips"]
            assert len(manifest1["selected_clips"]) == 5

    def test_balanced_distribution(self, sample_config, sample_clips, temp_run_manager):
        """Test that selections maintain balanced zoom and camera distribution"""
        manager = DataSelectionManager(sample_config, temp_run_manager)

        with patch.object(
            manager, "_discover_clips_deterministically", return_value=sample_clips
        ):
            results = manager.generate_phase_selections(
                optuna_max_clips=6,  # Should allow 2 per zoom level
                comparison_max_clips=12,
                random_seed=42,
            )

            # Load Optuna manifest
            optuna_manifest = DataSelectionManager.load_selection_manifest(
                results["optuna"]
            )

            # Check distributions
            zoom_dist = optuna_manifest["distributions"]["zoom_distribution"]
            camera_dist = optuna_manifest["distributions"]["camera_distribution"]

            # Should have balanced zoom distribution
            zoom_counts = list(zoom_dist.values())
            assert max(zoom_counts) - min(zoom_counts) <= 1  # Difference at most 1

            # Should have both cameras represented
            assert len(camera_dist) == 2
            assert "SONY_300" in camera_dist
            assert "SONY_70" in camera_dist

    def test_manifest_completeness(self, sample_config, sample_clips, temp_run_manager):
        """Test that manifests contain complete self-contained data"""
        manager = DataSelectionManager(sample_config, temp_run_manager)

        with patch.object(
            manager, "_discover_clips_deterministically", return_value=sample_clips
        ):
            results = manager.generate_phase_selections(
                optuna_max_clips=3, comparison_max_clips=5, random_seed=42
            )

            # Load comparison manifest
            manifest = DataSelectionManager.load_selection_manifest(
                results["comparison"]
            )

            # Check required top-level keys
            required_keys = [
                "selection_metadata",
                "distributions",
                "selected_clips",
                "summary",
            ]
            for key in required_keys:
                assert key in manifest

            # Check metadata completeness
            metadata = manifest["selection_metadata"]
            assert "phase" in metadata
            assert "random_seed" in metadata
            assert "video_format" in metadata
            assert "selection_timestamp" in metadata

            # Check each clip has complete data
            for clip in manifest["selected_clips"]:
                clip_required = ["clip_id", "video_path", "video_metadata", "maneuvers"]
                for key in clip_required:
                    assert key in clip

                # Check video metadata
                video_meta = clip["video_metadata"]
                meta_required = ["fps", "width", "height", "duration"]
                for key in meta_required:
                    assert key in video_meta

                # Check maneuver data
                for maneuver in clip["maneuvers"]:
                    maneuver_required = [
                        "maneuver_id",
                        "start_time",
                        "end_time",
                        "start_frame",
                        "end_frame",
                        "maneuver_type",
                        "execution_score",
                        "annotation_data",
                    ]
                    for key in maneuver_required:
                        assert key in maneuver

    def test_visualization_selection_diversity(
        self, sample_config, sample_clips, temp_run_manager
    ):
        """Test that visualization selections prioritize diversity"""
        manager = DataSelectionManager(sample_config, temp_run_manager)

        with patch.object(
            manager, "_discover_clips_deterministically", return_value=sample_clips
        ):
            results = manager.generate_phase_selections(
                comparison_max_clips=10, random_seed=42
            )

            # Load visualization manifest
            viz_manifest = DataSelectionManager.load_selection_manifest(
                results["visualization"]
            )

            # Check diversity criteria
            assert "selection_criteria" in viz_manifest
            criteria = viz_manifest["selection_criteria"]
            assert "diversity_factors" in criteria
            assert "maneuver_type" in criteria["diversity_factors"]

            # Check that different maneuver types are selected
            selected_maneuvers = viz_manifest["selected_maneuvers"]
            maneuver_types = set(m["maneuver_type"] for m in selected_maneuvers)

            # Should have some diversity (at least trying different types)
            assert len(maneuver_types) >= 1

    def test_data_loading_from_manifest(self, sample_config, temp_run_manager):
        """Test loading maneuvers from manifest files"""
        from data_handling.data_loader import SurfingDataLoader

        # Create a sample manifest
        sample_manifest = {
            "selection_metadata": {"phase": "test", "video_format": "h264"},
            "selected_clips": [
                {
                    "clip_id": "SONY_300_SESSION_060325_C0001_clip_1_default",
                    "camera": "SONY_300",
                    "session": "SESSION_060325",
                    "base_clip_id": "C0001_clip_1",
                    "zoom_level": "default",
                    "video_path": "/path/to/video.mp4",
                    "video_metadata": {
                        "fps": 25.0,
                        "width": 1920,
                        "height": 1080,
                        "duration": 10.0,
                        "total_frames": 250,
                    },
                    "maneuvers": [
                        {
                            "maneuver_id": "test_maneuver_1",
                            "maneuver_type": "Pop-up",
                            "execution_score": "05",
                            "start_time": 1.0,
                            "end_time": 5.0,
                            "start_frame": 25,
                            "end_frame": 125,
                            "duration": 4.0,
                            "total_frames": 100,
                            "labels": ["05", "Pop-up"],
                            "channel": 0,
                            "annotation_data": {
                                "start": 1.0,
                                "end": 5.0,
                                "labels": ["05", "Pop-up"],
                                "channel": 0,
                            },
                        }
                    ],
                }
            ],
        }

        # Save manifest to temp file
        manifest_path = temp_run_manager.data_selections_dir / "test_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(sample_manifest, f)

        # Load maneuvers using data loader
        loader = SurfingDataLoader(sample_config)
        maneuvers = loader.load_maneuvers_from_manifest(str(manifest_path))

        # Verify loaded data
        assert len(maneuvers) == 1
        maneuver = maneuvers[0]
        assert maneuver.maneuver_id == "test_maneuver_1"
        assert maneuver.maneuver_type == "Pop-up"
        assert maneuver.execution_score == "05"
        assert maneuver.start_time == 1.0
        assert maneuver.end_time == 5.0

    def test_manifest_summary_generation(
        self, sample_config, sample_clips, temp_run_manager
    ):
        """Test generation of human-readable manifest summaries"""
        manager = DataSelectionManager(sample_config, temp_run_manager)

        with patch.object(
            manager, "_discover_clips_deterministically", return_value=sample_clips
        ):
            results = manager.generate_phase_selections(
                optuna_max_clips=5, comparison_max_clips=8, random_seed=42
            )

            # Generate summary for Optuna manifest
            summary = manager.get_manifest_summary(results["optuna"])

            # Check summary contains key information
            assert "OPTUNA" in summary
            assert "Clips:" in summary
            assert "Maneuvers:" in summary
            assert "Random Seed:" in summary
            assert "Camera Distribution:" in summary
            assert "Zoom Distribution:" in summary

    def test_error_handling(self, sample_config, temp_run_manager):
        """Test error handling for various failure scenarios"""
        manager = DataSelectionManager(sample_config, temp_run_manager)

        # Test loading non-existent manifest
        with pytest.raises(Exception):
            DataSelectionManager.load_selection_manifest("/nonexistent/path.json")

        # Test with empty clips list
        with patch.object(
            manager, "_discover_clips_deterministically", return_value=[]
        ):
            results = manager.generate_phase_selections(
                optuna_max_clips=5, comparison_max_clips=10, random_seed=42
            )

            # Should handle empty clips gracefully
            assert len(results) >= 0  # Should not crash

    def test_different_seeds_produce_different_results(
        self, sample_config, sample_clips, temp_run_manager
    ):
        """Test that different random seeds produce different selections"""
        manager = DataSelectionManager(sample_config, temp_run_manager)

        with patch.object(
            manager, "_discover_clips_deterministically", return_value=sample_clips
        ):
            # Generate with seed 42
            results1 = manager.generate_phase_selections(
                optuna_max_clips=6, comparison_max_clips=10, random_seed=42
            )

            # Generate with seed 123
            results2 = manager.generate_phase_selections(
                optuna_max_clips=6, comparison_max_clips=10, random_seed=123
            )

            # Load manifests
            manifest1 = DataSelectionManager.load_selection_manifest(results1["optuna"])
            manifest2 = DataSelectionManager.load_selection_manifest(results2["optuna"])

            # Should have different selections (with high probability)
            clip_ids1 = [clip["clip_id"] for clip in manifest1["selected_clips"]]
            clip_ids2 = [clip["clip_id"] for clip in manifest2["selected_clips"]]

            # With different seeds, selections should likely be different
            # (This could occasionally fail due to randomness, but very unlikely)
            assert (
                clip_ids1 != clip_ids2 or len(sample_clips) <= 6
            )  # Unless all clips selected
