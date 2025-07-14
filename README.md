# 🏄‍♂️ Surf Pose Evaluation

A comprehensive evaluation framework for pose estimation models on surf footage, designed for production use with organized run management and automatic GPU acceleration.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Quick Start](#quick-start)
4. [The Evaluation Pipeline](#the-evaluation-pipeline)
5. [Configuration](#configuration)
6. [Running Evaluations](#running-evaluations)
7. [Understanding Results](#understanding-results)
8. [GPU Acceleration](#gpu-acceleration)
9. [Multi-Machine Setup](#multi-machine-setup)
10. [Troubleshooting](#troubleshooting)

## Overview

This system evaluates pose estimation models (YOLOv8, MediaPipe, MMPose, BlazePose, PyTorch Pose) on surf footage using a **two-phase approach**:

1. **Phase 1: Optuna Optimization** - Find optimal parameters for surf conditions
2. **Phase 2: Model Comparison** - Compare models using optimized parameters

### Key Features

- **Automated Pipeline**: Complete evaluation with a single command
- **GPU Acceleration**: CUDA-first with automatic fallback (MPS/CPU)
- **Organized Results**: Timestamp-based runs with MLflow tracking
- **Multi-Machine Support**: Shared storage for team collaboration
- **Surf-Optimized**: Parameters tuned for challenging surf conditions

## Setup

### 1. Environment Setup

Create and activate the conda environment:

```bash
# For macOS (Apple Silicon/Intel)
conda env create -f environment_macos.yml
conda activate surf_pose_eval

# For Linux/Windows
conda env create -f environment.yml
conda activate surf_pose_eval
```

### 2. Verify GPU Setup

Test your GPU acceleration:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"

# Or run the comprehensive GPU verification test
python tests/verify_gpu_setup.py
```

Expected output:

- **Linux with NVIDIA GPU**: `CUDA: True, MPS: False`
- **macOS with Apple Silicon**: `CUDA: False, MPS: True`
- **CPU-only systems**: `CUDA: False, MPS: False`

### 3. Data Setup

Ensure your data follows this structure:

```
data/SD_02_SURF_FOOTAGE_PREPT/
├── 03_CLIPPED/
│   ├── h264/           # H.264 video clips
│   └── ffv1/           # FFV1 video clips
├── 04_ANNOTATED/
│   └── surf-manoeuvre-labels/
│       ├── sony_300/   # Sony 300mm annotations
│       └── sony_70/    # Sony 70mm annotations
└── 05_ANALYSED_DATA/
    └── POSE/
        └── results/    # Evaluation results (auto-created)
```

## Quick Start

### Run Complete Evaluation

```bash
# Full pipeline (optimization + comparison)
python run_production_evaluation.py

# Test with limited clips (faster)
python run_production_evaluation.py --max-clips 20

# Custom run name
python run_production_evaluation.py --run-name "rtx4090_test"
```

### View Results

```bash
# Start MLflow UI for all experiments
python start_mlflow_ui.py

# View results summary
python utils/mlflow_utils.py --summary
```

## The Evaluation Pipeline

### Phase 1: Optuna Optimization (Automatic)

**Purpose**: Find optimal parameters for surf conditions  
**Duration**: ~2-5 hours (depending on hardware)  
**Config**: `configs/evaluation_config_production_optuna.yaml`

- Tests 50 clips with 100 optimization trials
- Explores confidence thresholds: 0.05-1.0 (full range for surf conditions)
- Tests model sizes: nano → large
- **No visualizations/predictions** (speed focused)
- **Output**: `results/best_params/best_parameters.yaml`

### Phase 2: Model Comparison (Automatic)

**Purpose**: Compare models using optimal parameters  
**Duration**: ~1-3 hours (depending on dataset size)  
**Config**: `configs/evaluation_config_production_comparison.yaml`

- Uses full dataset (200+ clips)
- Loads optimal parameters from Phase 1
- **Generates visualizations** and prediction files
- **Output**: Complete analysis with MLflow tracking

### Parameter Flow Between Phases

```yaml
# Phase 1 Output: results/best_params/best_parameters.yaml
mediapipe:
  min_detection_confidence: 0.12 # Optimized for surf conditions
  model_complexity: 2
yolov8_pose:
  model_size: "l" # Large model for accuracy
  confidence_threshold: 0.08 # Low threshold for challenging frames
```

These parameters automatically flow to Phase 2 for final comparison.

## Configuration

### Production Configs (Used by Pipeline)

- `evaluation_config_production_optuna.yaml` - Phase 1 optimization
- `evaluation_config_production_comparison.yaml` - Phase 2 comparison

### Model Configs (Surf-Optimized)

Each model has optimized parameter ranges for surf conditions:

- `model_configs/yolov8_pose.yaml` - YOLOv8 with expanded model sizes
- `model_configs/mediapipe.yaml` - MediaPipe with full confidence range
- `model_configs/blazepose.yaml` - BlazePose optimization
- `model_configs/mmpose.yaml` - MMPose configuration
- `model_configs/pytorch_pose.yaml` - PyTorch Pose settings

### Key Surf Optimizations

1. **Extended Confidence Range**: 0.05-1.0 (instead of 0.3-0.9)
2. **All Model Sizes**: From nano to extra-large
3. **Surf-Specific Parameters**: Segmentation, static image mode
4. **Challenging Conditions**: Optimized for spray, motion blur, lighting

## Running Evaluations

### Complete Pipeline

```bash
# Standard full evaluation
python run_production_evaluation.py

# With custom settings
python run_production_evaluation.py --max-clips 50 --run-name "experiment_v2"
```

### Individual Phases

```bash
# Only optimization phase
python run_production_evaluation.py --optuna-only

# Only comparison phase (requires existing parameters)
python run_production_evaluation.py --comparison-only

# Skip optimization (use existing parameters)
python run_production_evaluation.py --skip-optuna
```

### Direct Model Evaluation

```bash
# Test specific models
python evaluate_pose_models.py configs/evaluation_config_macos.yaml

# With custom models
python evaluate_pose_models.py configs/evaluation_config_macos.yaml --models yolov8_pose mediapipe
```

## Understanding Results

### Results Structure

```
data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/
└── runs/
    └── 20240315_143022_rtx4090_test/    # Timestamped run
        ├── mlruns/                      # MLflow experiments
        ├── best_params/                 # Optimal parameters
        ├── predictions/                 # JSON prediction files
        ├── visualizations/              # Video overlays
        ├── reports/                     # Performance reports
        ├── run_metadata.json           # Run information
        └── run_summary.json             # Final results
```

### File Naming Convention

**Predictions**: `maneuver_{type}_{score}_{video}_predictions.json`  
**Visualizations**: `maneuver_{type}_{score}_{video}_poses.mp4`

Example: `maneuver_cutback_85_SONY_70_SESSION_020325_C0010_clip_4_predictions.json`

### MLflow Experiments

Each run creates timestamped experiments:

- `surf_pose_production_optuna_20240315_143022` (Phase 1)
- `surf_pose_production_comparison_20240315_143022` (Phase 2)

### Key Metrics

- **mAP (mean Average Precision)**: Overall pose accuracy
- **PCK (Percentage of Correct Keypoints)**: Keypoint accuracy at different thresholds
- **FPS**: Inference speed
- **GPU Utilization**: Hardware efficiency

## GPU Acceleration

The system automatically detects and uses optimal acceleration:

### Production (Linux with NVIDIA GPU)

- **CUDA acceleration** for all compatible models
- **FP16 precision** for RTX 4090/3090 performance
- **MediaPipe GPU** acceleration enabled
- **Expected**: 60-80% faster inference

### Development (macOS)

- **MPS acceleration** where supported (YOLOv8, PyTorch)
- **CPU fallback** for MediaPipe/BlazePose (stability)
- **Expected**: 30-50% faster inference

### Automatic Configuration

No manual setup required! The system:

1. **Detects hardware** (CUDA → MPS → CPU)
2. **Configures models** for optimal performance
3. **Enables FP16** on compatible GPUs
4. **Logs acceleration status** for verification

## Multi-Machine Setup

### Shared Storage

All results use shared storage for team collaboration:

```
data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/
```

### Multi-Machine Workflow

**Machine 1 (Development)**:

```bash
python run_production_evaluation.py --max-clips 20 --run-name "dev_test"
```

**Machine 2 (Production)**:

```bash
python run_production_evaluation.py --run-name "production_eval"
```

**View Combined Results**:

```bash
python start_mlflow_ui.py --summary
```

### Benefits

- **No conflicts**: Timestamp-based isolation
- **Team collaboration**: Shared MLflow experiments
- **Easy comparison**: All results in single UI
- **Complete provenance**: Machine and user tracking

## Troubleshooting

### Common Issues

**1. No GPU acceleration detected**

```bash
# Check drivers and PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
```

**2. MLflow UI won't start**

```bash
# Check if MLflow is installed
pip install mlflow
python start_mlflow_ui.py
```

**3. Memory issues**

```bash
# Reduce batch size or use CPU fallback
python run_production_evaluation.py --max-clips 10
```

**4. Data path issues**

```bash
# Verify data structure
ls -la data/SD_02_SURF_FOOTAGE_PREPT/03_CLIPPED/h264/
```

### Debug Commands

```bash
# Check system status
python -c "from utils.run_manager import RunManager; rm = RunManager(); rm.print_run_info()"

# List all experiments  
python utils/mlflow_utils.py --list

# Check specific run
ls -la data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/runs/

# Run system tests
python tests/verify_gpu_setup.py          # GPU acceleration
python tests/test_prediction_system.py    # Prediction files
python tests/test_zoom_loading.py         # Data loading
```

### Performance Optimization

**For RTX 4090/3090**:

- Ensure CUDA 11.8+ and compatible PyTorch
- Enable FP16 precision (automatic)
- Monitor `nvidia-smi` for GPU utilization

**For macOS**:

- Use Metal Performance Shaders (automatic)
- Monitor Activity Monitor for GPU usage
- Consider CPU fallback for stability

---

## Getting Started Summary

1. **Setup environment**: `conda env create -f environment_macos.yml` (or `environment.yml`)
2. **Activate**: `conda activate surf_pose_eval`
3. **Run evaluation**: `python run_production_evaluation.py`
4. **View results**: `python start_mlflow_ui.py`

The system handles everything else automatically! 🚀
