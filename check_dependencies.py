#!/usr/bin/env python3
"""
Dependency checker for Surf Pose Evaluation project
Checks all required packages and system requirements
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Python Version Check:")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python version is compatible (3.8+)")
        return True
    else:
        print("   ❌ Python version is incompatible (requires 3.8+)")
        return False


def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally check version"""
    try:
        if import_name is None:
            import_name = package_name

        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")

        if min_version:
            from packaging import version as pkg_version

            current = pkg_version.parse(version)
            minimum = pkg_version.parse(min_version)

            if current >= minimum:
                print(f"   ✅ {package_name}: {version}")
                return True
            else:
                print(
                    f"   ⚠️  {package_name} version {version} may be too old (recommended: {min_version}+)"
                )
                return False
        else:
            print(f"   ✅ {package_name}: {version}")
            return True
    except ImportError:
        print(f"   ❌ {package_name}: Not installed")
        return False


def check_torch_gpu():
    """Check PyTorch GPU support"""
    print("\n🔥 PyTorch GPU Support:")
    try:
        import torch

        print(f"   PyTorch: {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"   ✅ CUDA: Available ({torch.cuda.get_device_name()})")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("   ⚠️  CUDA: Not available")

        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print("   ✅ MPS: Available (Apple Silicon)")
        else:
            print("   ⚠️  MPS: Not available")

        # Check CPU
        print("   ✅ CPU: Always available")

        return True
    except Exception as e:
        print(f"   ❌ PyTorch check failed: {e}")
        return False


def check_opencv():
    """Check OpenCV installation"""
    print("\n📷 OpenCV Check:")
    try:
        import cv2

        version = cv2.__version__
        print(f"   ✅ OpenCV: {version}")

        # Check video codecs
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        print("   ✅ MP4 codec: Available")

        return True
    except Exception as e:
        print(f"   ❌ OpenCV check failed: {e}")
        return False


def check_ultralytics():
    """Check Ultralytics YOLO"""
    print("\n🎯 Ultralytics YOLO Check:")
    try:
        from ultralytics import YOLO
        import ultralytics

        print(f"   ✅ Ultralytics: {ultralytics.__version__}")

        # Test model loading
        print("   Testing YOLOv8 model download...")
        model = YOLO("yolov8n-pose.pt")
        print("   ✅ YOLOv8 pose model: Loaded successfully")

        return True
    except Exception as e:
        print(f"   ❌ Ultralytics check failed: {e}")
        return False


def check_mlflow():
    """Check MLflow"""
    print("\n📊 MLflow Check:")
    try:
        import mlflow
        import tempfile
        import shutil
        from pathlib import Path

        print(f"   ✅ MLflow: {mlflow.__version__}")

        # Use a temporary directory for testing MLflow
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test tracking with temporary directory
            mlflow.set_tracking_uri(f"file://{temp_path}")
            mlflow.set_experiment("test_experiment")
            print("   ✅ MLflow tracking: Working")

        print("   ✅ Test directory automatically cleaned up")

        return True
    except Exception as e:
        print(f"   ❌ MLflow check failed: {e}")
        return False


def check_optuna():
    """Check Optuna"""
    print("\n🔍 Optuna Check:")
    try:
        import optuna

        print(f"   ✅ Optuna: {optuna.__version__}")

        # Test study creation
        study = optuna.create_study(direction="maximize")
        print("   ✅ Optuna study creation: Working")

        return True
    except Exception as e:
        print(f"   ❌ Optuna check failed: {e}")
        return False


def check_data_structure():
    """Check data directory structure"""
    print("\n📁 Data Structure Check:")

    base_path = Path("./data/SD_02_SURF_FOOTAGE_PREPT")
    if not base_path.exists():
        print("   ❌ Base data path not found")
        return False

    print("   ✅ Base data path exists")

    # Check video clips
    h264_path = base_path / "03_CLIPPED/h264"
    if h264_path.exists():
        sony_70_path = h264_path / "SONY_70"
        if sony_70_path.exists():
            video_files = list(sony_70_path.rglob("*.mp4"))
            print(f"   ✅ Found {len(video_files)} video files in SONY_70")
            return True
        else:
            print("   ❌ SONY_70 directory not found")
            return False
    else:
        print("   ❌ h264 video directory not found")
        return False


def check_disk_space():
    """Check available disk space"""
    print("\n💾 Disk Space Check:")
    try:
        import shutil

        total, used, free = shutil.disk_usage("./")
        free_gb = free // (1024**3)
        print(f"   Available space: {free_gb} GB")

        if free_gb > 10:
            print("   ✅ Sufficient disk space")
            return True
        else:
            print("   ⚠️  Low disk space (< 10 GB)")
            return False
    except Exception as e:
        print(f"   ❌ Disk space check failed: {e}")
        return False


def main():
    """Run all dependency checks"""
    print("🔧 Surf Pose Evaluation - macOS Dependency Check")
    print("=" * 50)

    checks = [
        check_python_version(),
        check_torch_gpu(),
        check_opencv(),
        check_ultralytics(),
        check_mlflow(),
        check_optuna(),
        check_data_structure(),
        check_disk_space(),
    ]

    print("\n" + "=" * 50)
    print("📋 Summary:")

    if all(checks):
        print("✅ All dependencies are satisfied!")
        print("🚀 Ready to run surf pose evaluation")
        return True
    else:
        print("❌ Some dependencies are missing or incompatible")
        print("Please install missing packages or fix issues above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
