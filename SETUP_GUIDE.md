# Surf Pose Evaluation Setup Guide

Simple setup guide for running pose estimation evaluation with MMPose directly available in the main environment.

## Platform-Specific Setup

Choose your setup based on your machine:

### For macOS (Apple Silicon/Intel)

```bash
# 1. Create main environment
conda env create -f environment_macos.yml
conda activate surf_pose_eval

# 2. Build and install MMPose
./setup_mmpose_macos.sh

# 3. Test setup
python -c "import torch; import mediapipe; import ultralytics; from mmpose.apis import MMPoseInferencer; print('All models ready!')"
```

### For Production (RTX 3090/4090 Linux)

```bash
# 1. Create main environment
conda env create -f environment.yml
conda activate surf_pose_eval

# 2. Build and install MMPose with CUDA support
./setup_mmpose_production.sh

# 3. Test setup
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
```

## How It Works

**Simple Approach:**

- **Build Phase**: Setup script builds MMPose from source with proper CUDA alignment, creating cached compilation
- **Install Phase**: Same script installs MMPose directly in main environment using cached compilation
- **Direct Usage**: All models available in single environment, simple wrapper used for MMPose

## Files Structure

```
surf_pose_evaluation/
├── environment_macos.yml          # macOS main environment
├── environment.yml                # Production main environment
├── setup_mmpose_macos.sh          # MMPose build & install for macOS
├── setup_mmpose_production.sh     # MMPose build & install for production
└── models/
    └── mmpose_wrapper.py           # Simple direct MMPose wrapper
```

## Troubleshooting

### If MMPose fails to build:

```bash
# Check CUDA version alignment (production)
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Re-run setup if needed
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

This approach eliminates complexity while ensuring MMPose is directly available for fast, simple execution!
