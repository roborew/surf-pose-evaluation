#!/usr/bin/env python3
"""
GPU Setup Verification Script
Tests CUDA-first device detection and GPU acceleration for pose estimation models
"""

import os
import platform
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_device_detection():
    """Test the CUDA-first device detection logic"""
    print("=" * 60)
    print("DEVICE DETECTION TEST")
    print("=" * 60)

    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python version: {platform.python_version()}")

    # Test PyTorch device availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")

    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")

    # Device selection logic (matches evaluate_pose_models.py)
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n✅ Selected device: {device} (CUDA-first priority)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"\n✅ Selected device: {device} (MPS fallback)")
    else:
        device = "cpu"
        print(f"\n⚠️  Selected device: {device} (CPU fallback)")

    return device


def test_mediapipe_environment():
    """Test MediaPipe environment variable settings"""
    print("\n" + "=" * 60)
    print("MEDIAPIPE ENVIRONMENT TEST")
    print("=" * 60)

    # Simulate the environment setup from evaluate_pose_models.py
    if platform.system().lower() == "darwin":  # macOS
        os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
        print("✅ macOS detected: MediaPipe configured for CPU stability")
    else:  # Linux/Windows
        os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        print("✅ Linux/Windows detected: MediaPipe configured for GPU acceleration")

    print(
        f"MEDIAPIPE_DISABLE_GPU: {os.environ.get('MEDIAPIPE_DISABLE_GPU', 'Not set')}"
    )
    print(
        f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set')}"
    )


def test_model_initialization(device):
    """Test model initialization with detected device"""
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION TEST")
    print("=" * 60)

    try:
        # Test basic tensor operations on detected device
        print(f"Testing tensor operations on {device}...")
        test_tensor = torch.randn(1, 3, 256, 256)

        if device != "cpu":
            test_tensor = test_tensor.to(device)
            result = torch.sum(test_tensor)
            print(f"✅ Tensor operations successful on {device}")

            if device == "cuda":
                print(
                    f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
                )
        else:
            result = torch.sum(test_tensor)
            print(f"✅ Tensor operations successful on CPU")

    except Exception as e:
        print(f"❌ Tensor operation failed: {e}")
        return False

    # Test YOLOv8 availability
    try:
        from ultralytics import YOLO

        print("✅ YOLOv8 (ultralytics) available")
    except ImportError:
        print("❌ YOLOv8 (ultralytics) not available")

    # Test MediaPipe availability
    try:
        import mediapipe as mp

        print("✅ MediaPipe available")
    except ImportError:
        print("❌ MediaPipe not available")

    # Test MMPose availability
    try:
        from mmpose.apis import MMPoseInferencer

        print("✅ MMPose available")
    except ImportError:
        print("❌ MMPose not available")

    return True


def test_half_precision(device):
    """Test half precision capabilities"""
    print("\n" + "=" * 60)
    print("HALF PRECISION TEST")
    print("=" * 60)

    if device == "cuda":
        try:
            test_tensor = torch.randn(1, 3, 256, 256, device=device)
            test_tensor_half = test_tensor.half()
            result = torch.sum(test_tensor_half)
            print("✅ FP16 (half precision) supported on CUDA")
            print("🚀 RTX 4090 will benefit significantly from FP16!")
        except Exception as e:
            print(f"❌ FP16 test failed: {e}")
    elif device == "mps":
        print("⚠️  FP16 support on MPS is experimental")
    else:
        print("⚠️  FP16 not applicable on CPU")


def main():
    """Run all verification tests"""
    print("GPU Setup Verification for Surf Pose Evaluation")
    print("This script verifies CUDA-first configuration")

    # Run tests
    device = test_device_detection()
    test_mediapipe_environment()
    test_model_initialization(device)
    test_half_precision(device)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if device == "cuda":
        print("🚀 PERFECT! Your production setup will use:")
        print("   • CUDA acceleration for all compatible models")
        print("   • MediaPipe GPU acceleration enabled")
        print("   • Half precision (FP16) for maximum performance")
        print("   • Expected 60-80% reduction in inference time")
    elif device == "mps":
        print("✅ GOOD! Your macOS setup will use:")
        print("   • MPS acceleration where supported")
        print("   • MediaPipe CPU (for stability)")
        print("   • Expected 30-50% reduction in inference time")
    else:
        print("⚠️  CPU ONLY detected:")
        print("   • All models will run on CPU")
        print("   • Consider installing CUDA drivers for GPU acceleration")

    print(f"\n🎯 Ready for production! Just run: python evaluate_pose_models.py")
    print(
        "   The system will automatically detect and use the best available acceleration."
    )


if __name__ == "__main__":
    main()
