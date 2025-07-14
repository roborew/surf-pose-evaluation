# Surf Pose Evaluation Setup Guide

Simple setup guide for running pose estimation evaluation with the proven two-phase approach: compile MMPose first, then create environment.

## Platform-Specific Setup

Choose your setup based on your machine:

### For macOS (Apple Silicon/Intel)

```bash
# Phase 1: Pre-compile MMPose (creates conda cache)
./setup_mmpose_macos.sh

# Phase 2: Create main environment (uses cached MMPose)
conda env create -f environment_macos.yml
conda activate surf_pose_eval

# Phase 3: Add MMPose packages to environment_macos.yml
# (The setup script will tell you exactly what to add)

# Test setup
python -c "import torch; import mediapipe; import ultralytics; from mmpose.apis import MMPoseInferencer; print('All models ready!')"
```

### For Production (RTX 3090/4090 Linux)

```bash
# Phase 1: Pre-compile MMPose (creates conda cache)
./setup_mmpose_production.sh

# Phase 2: Create main environment (uses cached MMPose packages)
conda env create -f environment.yml
conda activate surf_pose_eval

# Test setup
python -c "import torch; print('CUDA:', torch.cuda.is_available()); from mmpose.apis import MMPoseInferencer; print('MMPose ready!')"
```

## Running Evaluations

All models are directly available in the main environment:

```bash
# Activate main environment
conda activate surf_pose_eval

# Run full evaluation with all models
python evaluate_pose_models.py configs/evaluation_config_macos.yaml  # macOS
python evaluate_pose_models.py configs/evaluation_config_production_optuna.yaml  # Production

# Run specific models
python evaluate_pose_models.py configs/evaluation_config_macos.yaml --models mediapipe yolov8_pose mmpose

# Production evaluation with optimization
python run_production_evaluation.py --models mmpose mediapipe yolov8_pose
```

## How It Works

**Two-Phase Approach (Proven from SurfAnalysis):**

1. **Pre-compilation Phase**: Setup script creates temporary environment, builds MMPose from source with proper CUDA alignment, and caches the compilation
2. **Environment Creation Phase**: Main environment references MMPose packages in pip section, conda finds and uses the cached compilation
3. **Direct Usage**: All models available in single environment, simple wrapper used for MMPose

## Files Structure

```
surf_pose_evaluation/
├── environment_macos.yml          # macOS main environment
├── environment.yml                # Production main environment (CUDA 12.1)
├── setup_mmpose_macos.sh          # MMPose pre-compilation for macOS
├── setup_mmpose_production.sh     # MMPose pre-compilation for production
└── models/
    └── mmpose_wrapper.py           # Simple direct MMPose wrapper
```

## Key Features

- **CUDA 12.1 Support**: Production environment uses pytorch-cuda=12.1 to match system CUDA 12.6
- **NumPy 1.24.x**: Avoids NumPy 2.x compatibility issues
- **Cached Compilation**: MMPose built once, reused by main environment
- **No Redundant Packages**: System FFmpeg and CUDA used, no conda duplicates

## Troubleshooting

### If MMPose pre-compilation fails:

```bash
# Check CUDA version alignment (production)
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Re-run pre-compilation
./setup_mmpose_production.sh     # Production
./setup_mmpose_macos.sh          # macOS
```

### If main environment has issues:

```bash
# Remove and recreate
conda env remove -n surf_pose_eval
conda env create -f environment.yml          # Production
conda env create -f environment_macos.yml    # macOS
```

### If NumPy compatibility issues:

```bash
# Check NumPy version (should be 1.24.x)
python -c "import numpy; print(numpy.__version__)"

# If NumPy 2.x, downgrade:
pip install "numpy<2.0" --force-reinstall
```

This approach replicates the exact method that worked in SurfAnalysis - build MMPose first to create conda cache, then main environment uses the cached packages directly!
