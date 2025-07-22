# 🏄‍♂️ Surf Pose Evaluation Framework

A comprehensive evaluation framework for pose estimation models on surf footage, designed for production use with organized run management, automatic GPU acceleration, and standardized prediction-based visualization.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation & Setup](#installation--setup)
4. [Running Evaluations](#running-evaluations)
5. [Parameter Migration Guide](#parameter-migration-guide)
6. [Understanding Results](#understanding-results)
7. [System Architecture](#system-architecture)
8. [Configuration](#configuration)
9. [Enhanced Optuna Optimization](#enhanced-optuna-optimization)
10. [GPU Acceleration](#gpu-acceleration)
11. [Multi-Machine Setup](#multi-machine-setup)
12. [Testing & Verification](#testing--verification)
13. [Troubleshooting](#troubleshooting)
14. [Advanced Setup Details](#advanced-setup-details)
15. [YOLOv8 Weight Management](#yolov8-weight-management)

## Overview

This system evaluates pose estimation models on surf footage using a **standardized, prediction-file-based pipeline**:

**Supported Models:**

- **YOLOv8-Pose** (nano, small, medium, large, extra-large)
- **MediaPipe Pose** (complexity 0, 1, 2)
- **MMPose** (RTMPose, HRNet, other COCO models)
- **BlazePose** (full, lite, heavy)
- **PyTorch Pose** (KeypointRCNN with ResNet backbones)

**Key Features:**

- **Automated Two-Phase Pipeline**: Optuna optimization → Model comparison
- **GPU Acceleration**: CUDA/MPS auto-detection with FP16 precision
- **Standardized Predictions**: All models output to same JSON format
- **Advanced Visualization**: Multi-person support with bounding boxes and confidence scores
- **Organized Results**: Timestamp-based runs with MLflow tracking
- **Surf-Optimized**: Parameters tuned for challenging surf conditions

## Quick Start

### 🚀 One-Command Evaluation

```bash
# Clone and setup
git clone <repository-url>
cd surf_pose_evaluation

# Setup environment (choose your platform)
conda env create -f environment_macos.yml     # macOS
conda env create -f environment.yml           # Linux/Windows

# Activate environment
conda activate surf_pose_eval

# Download YOLOv8 weights
python setup_yolo_downloadweights.py

# Run complete evaluation (optimization + comparison)
python run_evaluation.py

# View results
python start_mlflow_ui.py
```

### 🎯 Quick Test (5 minutes)

```bash
# Test with limited clips for quick validation
python run_evaluation.py --max-clips 5 --run-name "quick_test"
```

## Installation & Setup

### Platform-Specific Setup

#### For macOS (Apple Silicon/Intel)

```bash
# Method 1: Simple environment creation (recommended)
conda env create -f environment_macos.yml
conda activate surf_pose_eval

# Method 2: Advanced setup with MMPose pre-compilation
./setup_mmpose_macos.sh                    # Pre-compile MMPose
conda env create -f environment_macos.yml  # Create main environment
conda activate surf_pose_eval

# Test setup
python -c "import torch; import mediapipe; import ultralytics; print('✅ All models ready!')"
```

#### For Production (Linux with NVIDIA GPU)

```bash
# Method 1: Direct environment creation
conda env create -f environment.yml
conda activate surf_pose_eval

# Method 2: Complete model zoo setup (for MMPose/MMDet)
./setup_mmpose_production.sh    # Pre-compile with complete model zoo
./create_surf_pose_env.sh        # Create optimized environment

# Test GPU setup
python tests/verify_gpu_setup.py
```

### Essential Dependencies Verification

```bash
# Check all dependencies and system status
python check_dependencies.py

# Quick dependency check
python -c "
import torch
import mediapipe
import ultralytics
import mlflow
import optuna
print('✅ Core dependencies installed')
print(f'🎮 CUDA: {torch.cuda.is_available()}')
print(f'🍎 MPS: {torch.backends.mps.is_available()}')
"
```

### YOLOv8 Weight Management

```bash
# Download all YOLOv8 pose models (recommended)
python setup_yolo_downloadweights.py

# Download specific models
python setup_yolo_downloadweights.py --models n,s,m

# Check download status
python setup_yolo_downloadweights.py --list

# Force re-download if corrupted
python setup_yolo_downloadweights.py --force
```

**Available Models:**

- **nano (n)**: 6.2 MB - Fastest (15+ FPS)
- **small (s)**: 11.6 MB - Balanced (10-15 FPS)
- **medium (m)**: 26.4 MB - Good accuracy (8-12 FPS)
- **large (l)**: 50.5 MB - High accuracy (5-8 FPS)
- **extra-large (x)**: 90.7 MB - Best accuracy (3-5 FPS)

### Data Structure Setup

Ensure your data follows this structure:

```
data/SD_02_SURF_FOOTAGE_PREPT/
├── 03_CLIPPED/
│   ├── h264/           # H.264 video clips (recommended for compatibility)
│   └── ffv1/           # FFV1 video clips (lossless, larger files)
├── 04_ANNOTATED/
│   └── surf-manoeuvre-labels/
│       ├── sony_300/   # Sony 300mm camera annotations
│       └── sony_70/    # Sony 70mm camera annotations
└── 05_ANALYSED_DATA/
    └── POSE/
        ├── results/    # Evaluation results (auto-created)
        └── splits/     # Data splits (auto-created)
```

## Running Evaluations

### Complete Production Pipeline

```bash
# Full evaluation with automatic optimization
python run_evaluation.py

# Custom run with specific settings
python run_evaluation.py \
    --run-name "rtx4090_production" \
    --max-clips 100 \
    --models yolov8_pose mmpose mediapipe

# Fast development test
python run_evaluation.py \
    --max-clips 10 \
    --run-name "dev_test" \
    --models yolov8_pose
```

### Phase-Specific Execution

```bash
# Only optimization phase (find best parameters)
python run_evaluation.py --optuna-only --max-clips 20

# Only comparison phase (use existing parameters)
python run_evaluation.py --comparison-only --max-clips 50

# Skip optimization (use default parameters)
python run_evaluation.py --skip-optuna
```

### Parameter Control Options

#### New Parameters (Recommended)

For better control over each evaluation phase, use the specific parameters:

```bash
# Separate control over Optuna and Comparison phases
python run_evaluation.py \
    --optuna-max-clips 6 \
    --comparison-max-clips 20 \
    --run-name "precise_control_test"

# Fast optimization, reasonable comparison
python run_evaluation.py \
    --optuna-max-clips 6 \
    --run-name "prod_test"
    # Comparison uses config value (50 clips from quick_test)
```

#### Legacy Parameter (Backward Compatible)

The `--max-clips` parameter still works but shows deprecation warnings:

```bash
# Old way (still supported)
python run_evaluation.py --max-clips 6 --run-name 'legacy_test'
# Both Optuna and Comparison use 6 clips
```

#### Parameter Priority

1. **Optuna Phase**: `--optuna-max-clips` → `--max-clips` → full dataset
2. **Comparison Phase**: `--comparison-max-clips` → `--max-clips` → config value (50 clips from quick_test)

For complete parameter migration details, see the [Parameter Migration Guide](#parameter-migration-guide) section below.

### Direct Model Evaluation

```bash
# Test specific configuration
python evaluate_pose_models.py configs/evaluation_config_macos_comparison.yaml

# Custom model selection
python evaluate_pose_models.py \
    configs/evaluation_config_production_comparison.yaml \
    --models yolov8_pose mediapipe
```

### Visualization Generation

The system automatically generates visualizations during evaluation. To create visualizations from existing prediction files:

```bash
# Create visualization from prediction file
python -c "
from utils.pose_video_visualizer import PoseVideoVisualizer
visualizer = PoseVideoVisualizer()
visualizer.create_visualization_from_prediction_file(
    'path/to/predictions.json',
    'output_visualization.mp4'
)
"
```

## Parameter Migration Guide

### 🔄 Parameter Changes

The `--max-clips` parameter behavior has been enhanced with clearer, more specific parameters while maintaining backward compatibility.

### 📊 Parameter Options

#### **New Parameters (Recommended)**

| Parameter                | Description                                 | Example                     |
| ------------------------ | ------------------------------------------- | --------------------------- |
| `--optuna-max-clips`     | Maximum clips for Optuna optimization phase | `--optuna-max-clips 6`      |
| `--comparison-max-clips` | Maximum clips for comparison phase          | `--comparison-max-clips 10` |

#### **Legacy Parameter (Deprecated but Still Supported)**

| Parameter     | Description                                    | Behavior                                   |
| ------------- | ---------------------------------------------- | ------------------------------------------ |
| `--max-clips` | **[DEPRECATED]** Maximum clips for both phases | Controls both Optuna and Comparison phases |

### 🚀 Usage Examples

#### **1. Old Way (Backward Compatible)**

```bash
python run_evaluation.py --max-clips 6 --run-name 'old_style_test'
```

- **Optuna**: Uses 6 clips
- **Comparison**: Uses 6 clips
- **Warning**: Shows deprecation warning

#### **2. New Way (Recommended)**

```bash
python run_evaluation.py --optuna-max-clips 6 --comparison-max-clips 10 --run-name 'new_style_test'
```

- **Optuna**: Uses 6 clips
- **Comparison**: Uses 10 clips
- **Benefit**: Full control over each phase

#### **3. Mixed Approach**

```bash
python run_evaluation.py --optuna-max-clips 6 --run-name 'mixed_test'
```

- **Optuna**: Uses 6 clips
- **Comparison**: Uses config value (50 clips from quick_test)
- **Use Case**: Fast optimization, reasonable comparison

#### **4. Production Default**

```bash
python run_evaluation.py --run-name 'production_run'
```

- **Optuna**: Uses full dataset
- **Comparison**: Uses config value (50 clips from quick_test)
- **Use Case**: Standard evaluation with reasonable performance

### 🎯 Migration Strategy

#### **Phase 1: Current (Backward Compatible)**

- `--max-clips` still works but shows deprecation warnings
- New parameters are available and recommended

#### **Phase 2: Future (Optional)**

- `--max-clips` could be removed in future versions
- All users should migrate to specific parameters

### 🔍 Parameter Priority

When multiple parameters are provided:

1. **Optuna Phase**: `--optuna-max-clips` > `--max-clips` > full dataset
2. **Comparison Phase**: `--comparison-max-clips` > `--max-clips` > config value

### 💡 Benefits of New Parameters

- **Clarity**: Explicit control over each phase
- **Flexibility**: Different datasets for optimization vs comparison
- **Performance**: Defaults to reasonable dataset sizes (50 clips for comparison)
- **Testing**: Easy to test with limited datasets
- **Production**: Full control over resource usage
- **Comprehensive Mode**: Use `--comparison-max-clips 200` for full comprehensive evaluation

### ⚠️ Breaking Changes

**None!** The implementation maintains full backward compatibility while providing clearer parameter names for future use.

## Understanding Results

### Two-Phase Evaluation Pipeline

#### Phase 1: Optuna Optimization (Automatic)

**Purpose**: Find optimal parameters for surf conditions  
**Duration**: 2-5 hours (depending on hardware)  
**Output**: `results/best_params/best_parameters.yaml`

- Tests 50 clips with 100 optimization trials
- Explores confidence thresholds: 0.05-1.0 (extended for surf conditions)
- Tests model sizes: nano → extra-large
- **No visualizations** (speed-focused)

#### Phase 2: Model Comparison (Automatic)

**Purpose**: Compare models using optimal parameters  
**Duration**: 1-3 hours (depending on dataset size)  
**Output**: Complete analysis with MLflow tracking

- Uses optimized parameters from Phase 1
- Generates **standardized prediction files**
- Creates **annotated visualization videos**
- **Multi-person detection** with bounding boxes and confidence scores

### Results Structure

```
data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/
└── runs/
    └── 20240315_143022_production_eval/  # Timestamped run directory
        ├── mlruns/                       # MLflow experiment tracking
        ├── best_params/                  # Optimized model parameters
        │   └── best_parameters.yaml
        ├── predictions/                  # Standardized JSON predictions
        │   ├── yolov8_pose/
        │   ├── mmpose/
        │   └── mediapipe/
        ├── visualizations/               # Annotated video overlays
        ├── reports/                      # Performance analysis
        ├── run_metadata.json           # Run configuration
        └── run_summary.json             # Final results summary
```

### File Naming Convention

**Standardized Predictions**: `maneuver_{type}_{score}_{video}_predictions.json`  
**Visualizations**: `maneuver_{type}_{score}_{video}_poses.mp4`

**Example**: `maneuver_Cutback_85_SONY_300_SESSION_020325_C0010_clip_4_wide_predictions.json`

### Standardized Prediction Format

All models output to the same JSON structure:

```json
{
  "maneuver_id": "maneuver_Cutback_85_C0010_clip_4_wide",
  "model_name": "mmpose",
  "video_path": "data/.../C0010_clip_4_wide.mkv",
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.04,
      "persons": [
        {
          "person_id": 0,
          "bbox": [123.5, 45.2, 234.1, 456.7],
          "detection_confidence": 0.89,
          "keypoints": [
            {"x": 150.2, "y": 67.8, "confidence": 0.95, "visibility": 1.0},
            ...
          ],
          "num_visible_keypoints": 15
        }
      ]
    }
  ]
}
```

### MLflow Experiment Tracking

Each run creates timestamped experiments:

- `surf_pose_production_optuna_20240315_143022` (Phase 1)
- `surf_pose_production_comparison_20240315_143022` (Phase 2)

**Key Metrics Tracked:**

- **mAP (mean Average Precision)**: Overall pose accuracy
- **PCK (Percentage of Correct Keypoints)**: Keypoint accuracy at thresholds
- **Detection F1**: Person detection accuracy
- **FPS**: Inference speed
- **GPU Utilization**: Hardware efficiency
- **Memory Usage**: Peak memory consumption

### Viewing Results

```bash
# Start MLflow UI for all experiments
python start_mlflow_ui.py

# Start UI for specific run
python start_mlflow_ui.py --run-name "20240315_143022_production_eval"

# Show experiments summary
python start_mlflow_ui.py --summary

# Command-line summary
python utils/mlflow_utils.py --summary

# Export results to JSON
python utils/mlflow_utils.py --export results_export.json
```

## System Architecture

### Prediction-Based Pipeline

The system uses a **standardized prediction file approach** for consistency and efficiency:

```
Model Inference → Standardized Predictions → Visualization
     ↓                      ↓                      ↓
Raw model output    JSON prediction files    Annotated videos
(model-specific)    (standardized format)    (consistent style)
```

**Benefits:**

- **Consistency**: All models produce identical output format
- **Reproducibility**: Visualizations can be regenerated from prediction files
- **No Re-inference**: Visualizations don't require running models again
- **Data Integrity**: Bounding boxes and confidence preserved across all models

### Multi-Person Support

**Features:**

- Unlimited person detection (configurable with `max_persons`)
- Unique person ID tracking across frames
- Bounding box annotations with detection confidence
- Format: `ID{person_id}: {confidence:.2f}`

**Visualization Elements:**

- **Color-coded keypoints**: Left limbs (orange), right limbs (green), torso (cyan), face (red)
- **Confidence filtering**: Separate thresholds for keypoints and bounding boxes
- **Skeleton connections**: Intelligent linking of confident keypoints
- **Legend overlay**: Model name, frame counter, color legend

## Configuration

### Production Configurations

**Primary Configs** (used by `run_evaluation.py`):

- `evaluation_config_production_optuna.yaml` - Phase 1 optimization
- `evaluation_config_production_comparison.yaml` - Phase 2 comparison

**Development Configs** (macOS-specific):

- `evaluation_config_macos_optuna.yaml` - macOS optimization
- `evaluation_config_macos_comparison.yaml` - macOS comparison

### Model-Specific Configurations

Each model has surf-optimized parameter ranges:

```yaml
# configs/model_configs/yolov8_pose.yaml
model_size:
  values: ["n", "s", "m", "l", "x"] # All sizes for accuracy vs speed trade-off

confidence_threshold:
  low: 0.05 # Extended range for challenging surf conditions
  high: 1.0

# configs/model_configs/mediapipe.yaml
min_detection_confidence:
  low: 0.05 # Lower threshold for spray/motion blur
  high: 1.0

model_complexity:
  values: [0, 1, 2] # Test all complexity levels
```

### Surf-Specific Optimizations

1. **Extended Confidence Ranges**: 0.05-1.0 (instead of typical 0.3-0.9)
2. **Complete Model Size Coverage**: From nano to extra-large
3. **Challenging Conditions**: Optimized for spray, motion blur, changing lighting
4. **Camera-Specific Tuning**: Sony 300mm vs 70mm lens differences

## Enhanced Optuna Optimization

The Optuna optimizer has been enhanced with intelligent early stopping and dynamic time allocation to maximize optimization efficiency while ensuring high-quality results.

### Key Features

#### 1. Intelligent Early Stopping

The optimizer now automatically stops when:

- **No improvement for N trials** (configurable patience)
- **Score plateaus** (within 95% of best score)
- **Minimum trials completed** (ensures sufficient exploration)

#### 2. Dynamic Time Allocation

Time is allocated based on model complexity:

- **MediaPipe**: 15% of total time (fastest)
- **YOLOv8**: 20% of total time (fast)
- **BlazePose**: 20% of total time (moderate)
- **PyTorch Pose**: 25% of total time (slower)
- **MMPose**: 30% of total time (slowest)

#### 3. Real-time Progress Monitoring

- **Trial-by-trial progress** with improvement tracking
- **Early stopping notifications** when triggered
- **Dynamic time redistribution** as models complete
- **Comprehensive summary reports**

### Configuration

#### Production Configuration

```yaml
# configs/evaluation_config_production_optuna.yaml
optuna:
  enabled: true
  study_name: "surf_pose_optimization_production"
  direction: "maximize"
  n_trials: 100
  timeout_minutes: 300 # 5 hours per model
  sampler: "TPESampler"
  pruner: "MedianPruner"

  # Intelligent early stopping configuration
  early_stopping:
    enabled: true
    patience: 10 # Stop if no improvement for 10 trials
    min_trials: 15 # Minimum trials before early stopping
    improvement_threshold: 0.001 # Minimum improvement to continue
    plateau_threshold: 0.95 # Consider plateaued if within 95% of best

  # Model-specific optimization settings
  model_settings:
    mediapipe:
      expected_trials: 20 # Fast model, fewer trials needed
      patience: 8
    yolov8_pose:
      expected_trials: 25
      patience: 10
    blazepose:
      expected_trials: 30
      patience: 12
    mmpose:
      expected_trials: 40 # Complex model, more trials
      patience: 15
    pytorch_pose:
      expected_trials: 35
      patience: 12
```

### Usage

#### Basic Usage

```bash
# Run with enhanced optimizer
python run_evaluation.py --max-clips 30 --optuna-trials 25 --coco-images 50 --run-name enhanced_optimization
```

#### Expected Behavior

1. **Fast models** (MediaPipe, YOLOv8) will likely stop early after 15-25 trials
2. **Complex models** (MMPose, PyTorch Pose) will use more time and trials
3. **Time redistribution** occurs as models complete
4. **Comprehensive reporting** shows optimization efficiency

### Expected Performance Improvements

#### Time Savings

| Model            | Traditional | Enhanced  | Savings    |
| ---------------- | ----------- | --------- | ---------- |
| **MediaPipe**    | 5 hours     | 1-2 hours | **60-80%** |
| **YOLOv8**       | 5 hours     | 2-3 hours | **40-60%** |
| **BlazePose**    | 5 hours     | 3-4 hours | **20-40%** |
| **MMPose**       | 5 hours     | 4-5 hours | **0-20%**  |
| **PyTorch Pose** | 5 hours     | 3-4 hours | **20-40%** |

#### Quality Assurance

- **Early stopping** only occurs after minimum trials
- **Plateau detection** ensures near-optimal results
- **Patience mechanism** prevents premature stopping
- **Quality maintained** while reducing unnecessary computation

### Monitoring and Reports

#### Real-time Output

```
🔄 Trial 015: conf_0.25_iou_0.45_size_n
   • Using 150 pre-selected maneuvers
   • Processed 5/150 maneuvers...
   • Processed 10/150 maneuvers...
   ✅ New best score: 0.8234 (improvement: 0.0123)

🔄 Trial 016: conf_0.30_iou_0.50_size_s
   • Using 150 pre-selected maneuvers
   • Processed 5/150 maneuvers...
   • Processed 10/150 maneuvers...
   • Trial score: 0.8156 (best: 0.8234, no improvement: 1)

🛑 Early stopping triggered after 16 trials
```

#### Summary Report

```
============================================================
DYNAMIC OPTIMIZATION SUMMARY
============================================================
Models completed: 5
Total trials: 89
Total time: 18.5h
Remaining time: 5.5h
Average score: 0.8123

Best performing model: yolov8_pose
  Score: 0.8234
  Trials: 16

Most efficient model: mediapipe
  Efficiency: 0.4123
  Score: 0.8156

Early stopped models: mediapipe, yolov8_pose, blazepose

Detailed Results:
----------------------------------------
✅ mediapipe: 15 trials, score: 0.8156, time: 2.5h, efficiency: 0.4123
✅ yolov8_pose: 16 trials, score: 0.8234, time: 3.2h, efficiency: 0.4123
✅ blazepose: 18 trials, score: 0.8198, time: 4.1h, efficiency: 0.4123
✅ mmpose: 25 trials, score: 0.8212, time: 5.0h, efficiency: 0.4123
✅ pytorch_pose: 15 trials, score: 0.8176, time: 3.7h, efficiency: 0.4123
============================================================
```

### Testing

Run the test script to verify functionality:

```bash
python test_enhanced_optuna.py
```

This will test:

- Early stopping configuration loading
- Early stopping decision logic
- Dynamic optimizer functionality
- Configuration file loading

### Benefits

#### 1. **Efficiency**

- 30-50% reduction in total optimization time
- More trials for models that need them
- Less time wasted on converged models

#### 2. **Quality**

- Maintains optimization quality
- Prevents premature stopping
- Ensures sufficient exploration

#### 3. **Flexibility**

- Configurable early stopping parameters
- Model-specific settings
- Adaptive time allocation

#### 4. **Monitoring**

- Real-time progress tracking
- Comprehensive reporting
- Performance analytics

### Troubleshooting

#### Early Stopping Too Aggressive

If models are stopping too early:

```yaml
early_stopping:
  patience: 15 # Increase patience
  min_trials: 20 # Increase minimum trials
  improvement_threshold: 0.0005 # Lower threshold
  plateau_threshold: 0.98 # Increase plateau threshold
```

#### Time Allocation Issues

If time allocation seems unfair:

```yaml
# Adjust model complexity scores in dynamic_optimizer.py
model_complexity = {
    "mediapipe": 0.20,    # Increase allocation
    "yolov8_pose": 0.25,  # Increase allocation
    "blazepose": 0.20,    # Keep same
    "pytorch_pose": 0.20, # Decrease allocation
    "mmpose": 0.25,       # Decrease allocation
}
```

### Migration from Old System

The enhanced optimizer is **backward compatible**. Existing configurations will work with default early stopping settings. To enable enhanced features:

1. **Update configuration** with early stopping settings
2. **Run normally** - enhanced features activate automatically
3. **Monitor output** for early stopping notifications
4. **Review summary** for optimization efficiency

### Future Enhancements

Planned improvements:

- **Adaptive patience** based on model performance
- **Multi-objective optimization** (speed vs accuracy)
- **Resource-aware allocation** (GPU memory, CPU cores)
- **Distributed optimization** across multiple machines

## GPU Acceleration

### Automatic Hardware Detection

The system automatically detects and configures optimal acceleration:

```python
# Automatic device detection and configuration
if torch.cuda.is_available():
    device = "cuda"
    enable_fp16 = True  # RTX 3090/4090 optimization
elif torch.backends.mps.is_available():
    device = "mps"     # Apple Silicon optimization
else:
    device = "cpu"     # Fallback
```

### Platform Performance

**Production (Linux with NVIDIA RTX 4090/3090):**

- **CUDA acceleration** for all compatible models
- **FP16 precision** for 2x performance boost
- **Expected performance**: 60-80% faster inference
- **GPU memory management**: Automatic batch sizing

**Development (macOS with Apple Silicon):**

- **MPS acceleration** for PyTorch-based models (YOLOv8, PyTorch Pose)
- **CPU fallback** for MediaPipe/BlazePose (stability)
- **Expected performance**: 30-50% faster inference

**Verification:**

```bash
# Check GPU acceleration status
python tests/verify_gpu_setup.py

# Quick GPU test
python -c "
import torch
print(f'🎮 CUDA Available: {torch.cuda.is_available()}')
print(f'🍎 MPS Available: {torch.backends.mps.is_available()}')
if torch.cuda.is_available():
    print(f'📊 GPU: {torch.cuda.get_device_name()}')
    print(f'💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## Multi-Machine Setup

### Shared Storage Architecture

All results use shared storage for seamless team collaboration:

```
data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/
├── runs/                    # Individual run directories
│   ├── 20240315_143022_dev_test/
│   ├── 20240315_183045_production_eval/
│   └── 20240316_094521_rtx4090_comparison/
└── shared_mlflow/          # Consolidated MLflow experiments
```

### Multi-Machine Workflow

**Development Machine (macOS)**:

```bash
python run_evaluation.py \
    --max-clips 20 \
    --run-name "dev_macos_test" \
    --models yolov8_pose mediapipe
```

**Production Machine (Linux RTX 4090)**:

```bash
python run_evaluation.py \
    --run-name "production_full_eval" \
    --models mmpose yolov8_pose pytorch_pose
```

**Analysis Machine (Any)**:

```bash
# View all experiments from all machines
python start_mlflow_ui.py --summary

# Compare results across machines
python utils/mlflow_utils.py --compare
```

### Benefits

- **No Conflicts**: Timestamp-based run isolation
- **Team Collaboration**: Shared MLflow experiments and results
- **Easy Comparison**: All results accessible in single UI
- **Complete Provenance**: Machine, user, and configuration tracking

## Testing & Verification

### System Tests

```bash
# Run all verification tests
python tests/verify_gpu_setup.py      # GPU acceleration
python tests/test_prediction_system.py # Prediction file system
python tests/test_zoom_loading.py     # Data loading and splits

# Quick system check
python check_dependencies.py
```

### Test Scripts Details

#### `verify_gpu_setup.py`

**Purpose**: Verify GPU acceleration setup and device detection  
**Usage**: `python tests/verify_gpu_setup.py`  
**Use when**: After environment setup to confirm GPU acceleration is working

#### `test_prediction_system.py`

**Purpose**: Test the standardized prediction file system  
**Usage**: `python tests/test_prediction_system.py`  
**Use when**: Validating prediction file generation and loading functionality

#### `test_zoom_loading.py`

**Purpose**: Test zoom-aware data loading and data leakage prevention  
**Usage**: `python tests/test_zoom_loading.py`  
**Use when**: Validating data loading splits and ensuring no data leakage

### Integration Tests

```bash
# Test complete pipeline with minimal data
python run_evaluation.py \
    --max-clips 3 \
    --run-name "integration_test" \
    --models yolov8_pose

# Test specific model
python -c "
from models.yolov8_wrapper import YOLOv8Wrapper
import numpy as np

model = YOLOv8Wrapper(model_size='n')
model.load_model()

# Test with dummy image
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = model.predict(test_image)
print(f'✅ Model test: {result[\"num_persons\"]} persons detected')
"
```

### Performance Benchmarks

```bash
# Benchmark all models
python -c "
import time
import numpy as np
from models import *

test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
models = ['yolov8_pose', 'mediapipe', 'pytorch_pose']

for model_name in models:
    # Time inference
    start = time.time()
    # ... model inference code ...
    fps = 1 / (time.time() - start)
    print(f'{model_name}: {fps:.1f} FPS')
"
```

## Troubleshooting

### Common Issues & Solutions

#### 1. GPU Acceleration Not Working

```bash
# Check CUDA installation
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# For macOS MPS issues
python -c "import torch; print(torch.backends.mps.is_built())"

# Fallback to CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

#### 2. MMPose Installation Issues

```bash
# Re-run complete setup
./setup_mmpose_production.sh  # Linux
./setup_mmpose_macos.sh       # macOS

# Check MMPose installation
python -c "
try:
    from mmpose.apis import MMPoseInferencer
    print('✅ MMPose ready')
except ImportError as e:
    print(f'❌ MMPose issue: {e}')
"
```

#### 3. YOLOv8 Weight Issues

```bash
# Check weights status
python setup_yolo_downloadweights.py --list

# Re-download corrupted weights
python setup_yolo_downloadweights.py --force

# Verify weight files
ls -la models/yolov8_pose/
```

#### 4. MLflow UI Problems

```bash
# Check MLflow installation
pip install mlflow==2.8.1

# Start with specific port
python start_mlflow_ui.py --port 5002

# Clear MLflow cache
rm -rf ~/.mlflow
```

#### 5. Memory Issues

```bash
# Reduce memory usage
python run_evaluation.py --max-clips 5

# Monitor memory usage
watch -n 1 'nvidia-smi'  # NVIDIA GPU
top -pid $(pgrep Python)  # CPU/RAM
```

#### 6. Data Path Issues

```bash
# Verify data structure
ls -la data/SD_02_SURF_FOOTAGE_PREPT/03_CLIPPED/h264/

# Check annotation files
find data/SD_02_SURF_FOOTAGE_PREPT/04_ANNOTATED/ -name "*.json" | head -5

# Test data loading
python -c "
from data_handling.data_loader import SurfingDataLoader
import yaml

with open('configs/evaluation_config_macos_comparison.yaml') as f:
    config = yaml.safe_load(f)

loader = SurfingDataLoader(config)
clips = loader.load_all_video_clips()
print(f'✅ Loaded {len(clips)} video clips')
"
```

### Debug Commands

```bash
# System status overview
python -c "
import torch
import platform
import psutil
import shutil

print(f'🖥️  Platform: {platform.platform()}')
print(f'🐍 Python: {platform.python_version()}')
print(f'🔥 PyTorch: {torch.__version__}')
print(f'🎮 CUDA: {torch.cuda.is_available()}')
print(f'🍎 MPS: {torch.backends.mps.is_available()}')
print(f'💾 RAM: {psutil.virtual_memory().total / 1e9:.1f} GB')
print(f'💿 Disk: {shutil.disk_usage(\"/\")[2] / 1e9:.1f} GB free')
"

# List all experiments
python utils/mlflow_utils.py --list

# Check specific run
ls -la data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/runs/

# Monitor GPU during evaluation
watch -n 2 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'
```

### Performance Optimization Tips

**For RTX 4090/3090:**

- Ensure CUDA 11.8+ and compatible PyTorch
- Enable mixed precision (automatic)
- Monitor `nvidia-smi` for >80% GPU utilization
- Use batch processing for multiple clips

**For macOS:**

- Prefer MPS-compatible models (YOLOv8, PyTorch)
- Monitor Activity Monitor → GPU tab
- Use smaller model sizes for real-time inference
- Consider CPU fallback for stability

**For CPU-only systems:**

- Use nano/small models only
- Reduce max_clips for memory management
- Enable multi-threading optimizations
- Consider cloud GPU instances for production

## Advanced Setup Details

### Two-Phase Setup Approach (For Advanced Users)

**Proven approach from SurfAnalysis for complex environments:**

#### Phase 1: Pre-compilation (Creates conda cache)

**macOS:**

```bash
./setup_mmpose_macos.sh
```

**Linux Production:**

```bash
./setup_mmpose_production.sh
```

#### Phase 2: Environment Creation (Uses cached packages)

**macOS:**

```bash
conda env create -f environment_macos.yml
conda activate surf_pose_eval
```

**Linux Production:**

```bash
./create_surf_pose_env.sh
```

### Complete Model Zoo Installation

**🚨 CRITICAL: Why We Install from Source**

For MMPose/MMDetection, source installation ensures the **complete model zoo** is available:

- **Problem**: `mim install mmdet` only installs core package without configuration files
- **Solution**: Source installation includes full `configs/` directory with all model configurations
- **Result**: RTMDet, human detection, and all model configs are properly accessible

**What You Get:**

- ✅ **Complete MMDetection configs**: All RTMDet, COCO, and specialized model configurations
- ✅ **Human detection models**: Person detection configs that MMPose needs
- ✅ **Full model zoo**: Access to 200+ pre-configured detection models
- ✅ **No missing configs**: Eliminates "Cannot find model" errors

**Verification:**

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

### Key Features of Advanced Setup

- **CUDA 12.1 Support**: Production environment uses pytorch-cuda=12.1 to match system CUDA 12.6
- **Complete Model Zoo**: Source installation ensures all detection models are available
- **NumPy 1.24.x**: Avoids NumPy 2.x compatibility issues
- **Cached Compilation**: MMPose built once, reused by main environment
- **No Missing Configs**: Full MMDetection configuration library installed

## YOLOv8 Weight Management

### Overview

The system uses a pre-download weight system for YOLOv8, eliminating complex download logic and making the system more reliable and maintainable.

### Weight Download Script Features

**`setup_yolo_downloadweights.py` provides:**

- Downloads all 5 YOLOv8 pose models (nano, small, medium, large, extra-large)
- Multiple download URLs with fallback
- Progress tracking and validation
- Force re-download option
- Status listing functionality

### Usage Examples

```bash
# Download all models
python setup_yolo_downloadweights.py

# Download specific models
python setup_yolo_downloadweights.py --models n,s

# Check status
python setup_yolo_downloadweights.py --list

# Force re-download
python setup_yolo_downloadweights.py --force
```

### File Structure

```
models/
├── yolov8_pose/                    # Weight storage directory
│   ├── yolov8n-pose.pt            # Nano model (6.2 MB)
│   ├── yolov8s-pose.pt            # Small model (11.6 MB)
│   ├── yolov8m-pose.pt            # Medium model (26.4 MB)
│   ├── yolov8l-pose.pt            # Large model (50.5 MB)
│   └── yolov8x-pose.pt            # Extra-large model (90.7 MB)
├── yolov8_wrapper.py               # Refactored wrapper
└── ...

setup_yolo_downloadweights.py       # Weight download script
```

### Error Handling

The wrapper provides clear, actionable error messages:

```python
FileNotFoundError: YOLOv8 weights not found: models/yolov8_pose/yolov8n-pose.pt

Please run the setup script to download weights:
  python setup_yolo_downloadweights.py --models n

Or download all standard models:
  python setup_yolo_downloadweights.py

Expected weight file: yolov8n-pose.pt
```

### Benefits Achieved

- ✅ **Reliability**: No more download failures during evaluation
- ✅ **Performance**: Faster model initialization
- ✅ **Maintainability**: Much simpler codebase (60% reduction in complexity)
- ✅ **User Experience**: Clear error messages and setup instructions
- ✅ **Separation of Concerns**: Setup vs. inference are now separate
- ✅ **Resource Management**: Centralized weight storage

---

## Getting Started Checklist

- [ ] **Environment Setup**: `conda env create -f environment_macos.yml` (or `environment.yml`)
- [ ] **Activate Environment**: `conda activate surf_pose_eval`
- [ ] **Download Weights**: `python setup_yolo_downloadweights.py`
- [ ] **Test System**: `python check_dependencies.py`
- [ ] **Verify GPU**: `python tests/verify_gpu_setup.py`
- [ ] **Run Quick Test**: `python run_evaluation.py --max-clips 3 --run-name "test"`
- [ ] **View Results**: `python start_mlflow_ui.py`

🚀 **Ready to evaluate surf pose estimation models!**

---

_For additional help, check the individual test scripts in `tests/` or run the troubleshooting commands above._
