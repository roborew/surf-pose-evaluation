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
# Phase 1: Pre-compile MMPose with complete model zoo (creates conda cache)
./setup_mmpose_production.sh

# Phase 2: Create main environment (uses cached MMPose packages)
./create_surf_pose_env.sh

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

## Complete Model Zoo Installation

**🚨 CRITICAL: Why We Install from Source**

The updated setup now installs MMDetection from source instead of via `mim install` to ensure the **complete model zoo** is available:

- **Problem**: `mim install mmdet` only installs the core package without configuration files
- **Solution**: Source installation includes the full `configs/` directory with all model configurations
- **Result**: RTMDet, human detection, and all model configs are properly accessible

### What You Get:

✅ **Complete MMDetection configs**: All RTMDet, COCO, and specialized model configurations  
✅ **Human detection models**: Person detection configs that MMPose needs  
✅ **Full model zoo**: Access to 200+ pre-configured detection models  
✅ **No missing configs**: Eliminates "Cannot find model" errors

### Verification:

The setup scripts now verify that the complete model zoo is installed:

```bash
# Check model zoo installation
python -c "
import mmdet, os
configs_path = os.path.join(os.path.dirname(mmdet.__file__), '..', 'configs')
rtmdet_configs = os.path.join(configs_path, 'rtmdet')
if os.path.exists(rtmdet_configs):
    print(f'✅ Complete model zoo: {len(os.listdir(rtmdet_configs))} RTMDet configs')
else:
    print('❌ Incomplete installation')
"
```

## Files Structure

```
surf_pose_evaluation/
├── environment_macos.yml          # macOS main environment
├── environment.yml                # Production main environment (CUDA 12.1)
├── setup_mmpose_macos.sh          # MMPose pre-compilation for macOS
├── setup_mmpose_production.sh     # MMPose + MMDet source compilation for production
├── create_surf_pose_env.sh        # Creates working environment with complete zoo
└── models/
    └── mmpose_wrapper.py           # Simple direct MMPose wrapper
```

## Key Features

- **CUDA 12.1 Support**: Production environment uses pytorch-cuda=12.1 to match system CUDA 12.6
- **Complete Model Zoo**: Source installation ensures all detection models are available
- **NumPy 1.24.x**: Avoids NumPy 2.x compatibility issues
- **Cached Compilation**: MMPose built once, reused by main environment
- **No Missing Configs**: Full MMDetection configuration library installed

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

### If "Cannot find model" errors occur:

```bash
# Verify complete model zoo installation
python -c "
import mmdet, os
mmdet_path = os.path.dirname(mmdet.__file__)
configs_path = os.path.join(os.path.dirname(mmdet_path), 'configs')
print(f'Configs available: {os.path.exists(configs_path)}')
if os.path.exists(configs_path):
    rtmdet_path = os.path.join(configs_path, 'rtmdet')
    print(f'RTMDet configs: {os.path.exists(rtmdet_path)}')
    if os.path.exists(rtmdet_path):
        print(f'Config count: {len([f for f in os.listdir(rtmdet_path) if f.endswith(\".py\")])}')"

# If configs are missing, reinstall from source:
cd ../mmdetection
pip install -v -e .
```

### If main environment has issues:

```bash
# Remove and recreate
conda env remove -n surf_pose_eval
./create_surf_pose_env.sh         # Production
conda env create -f environment_macos.yml    # macOS
```

### If NumPy compatibility issues:

```bash
# Check NumPy version (should be 1.24.x)
python -c "import numpy; print(numpy.__version__)"

# If NumPy 2.x, downgrade:
pip install "numpy<2.0" --force-reinstall
```

## Why This Approach Works

This method replicates the exact approach that worked in SurfAnalysis:

1. **Build from source first** to create conda cache with complete packages
2. **Main environment uses cached packages** directly from the compilation
3. **Complete model zoo** ensures all detection models are accessible
4. **No registry conflicts** because all configs are properly installed

The key insight: `mim install` is great for minimal installations, but for comprehensive frameworks like MMDetection, **source installation is essential** to get the complete model ecosystem.
