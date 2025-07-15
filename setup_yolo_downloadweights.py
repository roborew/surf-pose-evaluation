#!/usr/bin/env python3
"""
YOLOv8 Pose Weights Download Script

This script pre-downloads YOLOv8 pose model weights to avoid download issues during inference.
Downloads all standard YOLOv8 pose models (nano, small, medium, large, extra-large) to the
models/yolov8_pose directory.

Usage:
    python setup_yolo_downloadweights.py [--models n,s,m,l,x] [--force]

Arguments:
    --models: Comma-separated list of model sizes to download (default: all)
    --force: Force re-download even if weights exist
"""

import argparse
import logging
import requests
import torch
from pathlib import Path
import hashlib
import sys
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configurations
YOLO_MODELS = {
    "n": {
        "filename": "yolov8n-pose.pt",
        "size_mb": 6.2,
        "description": "YOLOv8 Nano Pose",
    },
    "s": {
        "filename": "yolov8s-pose.pt",
        "size_mb": 11.6,
        "description": "YOLOv8 Small Pose",
    },
    "m": {
        "filename": "yolov8m-pose.pt",
        "size_mb": 26.4,
        "description": "YOLOv8 Medium Pose",
    },
    "l": {
        "filename": "yolov8l-pose.pt",
        "size_mb": 50.5,
        "description": "YOLOv8 Large Pose",
    },
    "x": {
        "filename": "yolov8x-pose.pt",
        "size_mb": 90.7,
        "description": "YOLOv8 Extra-Large Pose",
    },
}

# Download URLs (in order of preference)
DOWNLOAD_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.0.0/{filename}",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/{filename}",
]


class YOLOWeightDownloader:
    """Handles downloading and validating YOLOv8 pose weights"""

    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, model_size: str, force: bool = False) -> bool:
        """Download a specific YOLOv8 pose model

        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            force: Force re-download even if file exists

        Returns:
            True if successful, False otherwise
        """
        if model_size not in YOLO_MODELS:
            logger.error(f"Unknown model size: {model_size}")
            return False

        model_info = YOLO_MODELS[model_size]
        filename = model_info["filename"]
        filepath = self.weights_dir / filename

        # Check if already exists and valid
        if filepath.exists() and not force:
            if self._validate_model_file(filepath):
                logger.info(
                    f"✅ {model_info['description']} already exists and is valid"
                )
                return True
            else:
                logger.warning(
                    f"⚠️  {filename} exists but is corrupted, re-downloading..."
                )
                filepath.unlink()

        logger.info(
            f"📥 Downloading {model_info['description']} ({model_info['size_mb']:.1f} MB)..."
        )

        # Try each download URL
        for url_template in DOWNLOAD_URLS:
            url = url_template.format(filename=filename)
            if self._download_from_url(url, filepath):
                if self._validate_model_file(filepath):
                    logger.info(f"✅ Successfully downloaded {filename}")
                    return True
                else:
                    logger.warning(f"⚠️  Downloaded {filename} is corrupted")
                    filepath.unlink()

        logger.error(f"❌ Failed to download {filename} from all sources")
        return False

    def _download_from_url(self, url: str, filepath: Path) -> bool:
        """Download file from URL with progress tracking

        Args:
            url: Download URL
            filepath: Destination file path

        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info(f"Trying: {url}")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress update every MB
                        if downloaded % (1024 * 1024) == 0 and total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.info(
                                f"Progress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)"
                            )

            logger.info(f"Download completed: {downloaded / (1024*1024):.1f} MB")
            return True

        except Exception as e:
            logger.warning(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            return False

    def _validate_model_file(self, filepath: Path) -> bool:
        """Validate that downloaded model file is valid PyTorch model

        Args:
            filepath: Path to model file

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file size
            if filepath.stat().st_size < 1024 * 1024:  # Less than 1MB is suspicious
                logger.warning(
                    f"File {filepath.name} is too small ({filepath.stat().st_size} bytes)"
                )
                return False

            # Try to load with PyTorch
            state_dict = torch.load(filepath, map_location="cpu")

            # Basic validation - should have model structure
            if not isinstance(state_dict, dict):
                logger.warning(
                    f"File {filepath.name} doesn't contain a valid state dict"
                )
                return False

            # Check for essential keys that YOLOv8 models should have
            essential_keys = ["model", "epoch", "date"]
            missing_keys = [key for key in essential_keys if key not in state_dict]
            if missing_keys:
                logger.warning(
                    f"File {filepath.name} missing essential keys: {missing_keys}"
                )
                return False

            logger.debug(f"✅ {filepath.name} validation passed")
            return True

        except Exception as e:
            logger.warning(f"Validation failed for {filepath.name}: {e}")
            return False

    def list_downloaded_weights(self) -> Dict[str, bool]:
        """List status of all YOLOv8 pose weights

        Returns:
            Dictionary mapping model size to availability status
        """
        status = {}
        for size, info in YOLO_MODELS.items():
            filepath = self.weights_dir / info["filename"]
            status[size] = filepath.exists() and self._validate_model_file(filepath)
        return status

    def get_weights_summary(self) -> str:
        """Get a formatted summary of downloaded weights"""
        status = self.list_downloaded_weights()
        lines = [f"YOLOv8 Pose Weights Status (Directory: {self.weights_dir}):"]
        lines.append("=" * 60)

        for size, info in YOLO_MODELS.items():
            status_icon = "✅" if status[size] else "❌"
            lines.append(
                f"{status_icon} {info['description']:<25} ({info['size_mb']:.1f} MB)"
            )

        lines.append("=" * 60)
        available_count = sum(status.values())
        lines.append(f"Available: {available_count}/{len(YOLO_MODELS)} models")

        return "\n".join(lines)


def main():
    """Main function to handle command line arguments and run downloads"""
    parser = argparse.ArgumentParser(
        description="Download YOLOv8 pose model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_yolo_downloadweights.py                    # Download all models
  python setup_yolo_downloadweights.py --models n,s       # Download nano and small only
  python setup_yolo_downloadweights.py --force            # Force re-download all
  python setup_yolo_downloadweights.py --list             # List current status
        """,
    )

    parser.add_argument(
        "--models",
        type=str,
        default="n,s,m,l,x",
        help="Comma-separated list of model sizes to download (n,s,m,l,x). Default: all",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if weights already exist",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List current status of downloaded weights and exit",
    )

    parser.add_argument(
        "--weights-dir",
        type=str,
        default="models/yolov8_pose",
        help="Directory to store weights (default: models/yolov8_pose)",
    )

    args = parser.parse_args()

    # Setup weights directory
    weights_dir = Path(args.weights_dir)
    downloader = YOLOWeightDownloader(weights_dir)

    # Handle list command
    if args.list:
        print(downloader.get_weights_summary())
        return

    # Parse model sizes
    try:
        model_sizes = [size.strip().lower() for size in args.models.split(",")]
        invalid_sizes = [size for size in model_sizes if size not in YOLO_MODELS]
        if invalid_sizes:
            logger.error(f"Invalid model sizes: {invalid_sizes}")
            logger.error(f"Valid sizes: {', '.join(YOLO_MODELS.keys())}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing model sizes: {e}")
        sys.exit(1)

    # Download models
    logger.info(
        f"Starting download of {len(model_sizes)} model(s): {', '.join(model_sizes)}"
    )

    success_count = 0
    total_count = len(model_sizes)

    for model_size in model_sizes:
        if downloader.download_model(model_size, force=args.force):
            success_count += 1
        else:
            logger.error(f"Failed to download YOLOv8-{model_size}-pose")

    # Final summary
    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count}/{total_count} successful")

    if success_count == total_count:
        logger.info("🎉 All downloads completed successfully!")
        print("\nYou can now use the YOLOv8 wrapper without download issues.")
    else:
        logger.error(f"❌ {total_count - success_count} download(s) failed")
        print("\nSome downloads failed. Check your internet connection and try again.")
        sys.exit(1)

    # Show final status
    print("\n" + downloader.get_weights_summary())


if __name__ == "__main__":
    main()
