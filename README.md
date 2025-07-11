# Surfing Pose Estimation Evaluation Framework

A comprehensive system for comparing and evaluating multiple pose estimation libraries for surfing performance analysis using MLflow experiment tracking and Optuna hyperparameter optimization.

## Project Overview

This framework systematically compares pose estimation models to determine the optimal backbone for surfing action recognition. It supports both local development (macOS) and production evaluation (Linux/CUDA) environments.

### Supported Models

- **MMPose** - Extensive model zoo with high accuracy
- **MediaPipe Pose** - Fast, edge-optimized for real-time inference
- **YOLOv8-Pose** - Unified detection and pose estimation
- **HRNet** - High-accuracy reference implementation

> **Note**: OpenPose excluded due to poor performance (5 FPS vs 20-60 FPS for modern alternatives) and installation complexity.

## Quick Start

### 1. Environment Setup

**Choose your deployment environment:**

| Environment           | Hardware      | Acceleration | Best For                               |
| --------------------- | ------------- | ------------ | -------------------------------------- |
| **macOS Development** | Apple Silicon | MPS          | Development, testing, quick iterations |
| **Linux Production**  | NVIDIA GPU    | CUDA         | Comprehensive evaluation, research     |

#### macOS Setup (Development)

```bash
conda env create -f environment_macos.yml
conda activate surf_pose_eval

# Verify installation
python test_zoom_loading.py
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Capabilities:** ✅ MediaPipe ✅ Basic YOLOv8 ✅ MPS acceleration ❌ MMPose ❌ HRNet

#### Linux Setup (Production)

```bash
# Try main environment first
conda env create -f environment.yml
conda activate surf_pose_eval

# If conflicts, use fallback:
# conda env create -f environment_simple.yml && pip install mmengine mmcv mmdet mmpose ultralytics

# Verify installation
python test_zoom_loading.py
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Capabilities:** ✅ All models ✅ CUDA acceleration ✅ Large batches ✅ FFV1 lossless video

### 2. Quick Test

```bash
# macOS Development Test
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --quick-test --models mediapipe --max-clips 5

# Linux Production Test
python evaluate_pose_models.py \
    --config configs/evaluation_config.yaml \
    --quick-test --models mediapipe --max-clips 10
```

### 3. Full Evaluation

```bash
# Run in tmux for long sessions (Linux)
tmux new-session -d -s pose_eval
tmux send-keys -t pose_eval "conda activate surf_pose_eval" Enter
tmux send-keys -t pose_eval "python evaluate_pose_models.py --config configs/evaluation_config.yaml --models mediapipe mmpose yolov8_pose hrnet --max-clips 300" Enter
tmux attach -t pose_eval
```

## Project Structure

```
surf_pose_evaluation/
├── README.md                       # This file
├── environment*.yml                # Environment configurations
├── evaluate_pose_models.py        # Main evaluation script
├── configs/
│   ├── evaluation_config*.yaml     # Environment-specific configs
│   └── model_configs/              # Individual model configurations
├── models/                         # Pose model wrappers
├── metrics/                        # Evaluation metrics
├── data_handling/                  # Zoom-aware data loading
├── utils/                          # MLflow, visualization utilities
├── data/                           # Your dataset
└── results/                        # MLflow experiments and outputs
```

## Key Features

### Zoom-Aware Data Loading

**Prevents data leakage** from zoom variations:

- Each clip has 3 zoom levels (default, wide, full)
- Intelligent selection ensures only one zoom per base clip
- Balanced distribution (~33% each zoom) with no leakage

```bash
python test_zoom_loading.py  # Verify this works
```

### Multi-Environment Support

- **Development (macOS)**: Fast iteration with lightweight models
- **Production (Linux)**: Comprehensive evaluation with all models
- **Automatic acceleration**: MPS on macOS, CUDA on Linux

### Comprehensive Metrics

- **Accuracy**: PCK@0.2, MPJPE, detection metrics
- **Performance**: Inference latency, memory usage, FPS
- **Temporal**: Frame-to-frame consistency and stability

## Configuration

### Files

- `configs/evaluation_config_macos.yaml` - Development (MPS, reduced clips, lightweight models)
- `configs/evaluation_config.yaml` - Production (CUDA, full clips, all models)

### Video Formats

- **H264**: Compressed videos for development
- **FFV1**: Lossless videos for production evaluation

## Performance Expectations

| Environment | Hardware      | MediaPipe | MMPose    | YOLOv8    | Memory                   |
| ----------- | ------------- | --------- | --------- | --------- | ------------------------ |
| **macOS**   | Apple Silicon | 15-25 FPS | N/A       | 10-20 FPS | 2-4GB RAM                |
| **Linux**   | RTX 4090      | 30-60 FPS | 10-20 FPS | 20-40 FPS | 8-16GB RAM + 4-12GB VRAM |

## Development Workflow

### 1. Local Development (macOS)

```bash
# Quick iteration cycle
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --quick-test --models mediapipe --max-clips 5 --verbose
```

### 2. Transfer to Production

```bash
# Git method (recommended)
git add . && git commit -m "Updated framework" && git push
# On training machine: git pull

# Or rsync method
rsync -av --exclude results/ --exclude __pycache__/ \
    surf_pose_evaluation/ user@training-machine:~/surf_pose_evaluation/
```

### 3. Production Evaluation

```bash
ssh user@training-machine
cd ~/surf_pose_evaluation
conda activate surf_pose_eval
python evaluate_pose_models.py --config configs/evaluation_config.yaml --models all
```

## MLflow Integration

All experiments automatically logged with:

- Model parameters and hyperparameters
- Evaluation metrics and performance benchmarks
- Video samples with pose overlays
- Model artifacts and checkpoints

```bash
# Local MLflow UI
mlflow ui --backend-store-uri ./results/mlruns
# Open http://localhost:5000

# Remote access (training machines)
ssh -L 5000:localhost:5000 user@training-machine
# Then run mlflow ui on remote machine
```

### MLflow Compare-Runs Configuration

For optimal model comparison in the MLflow UI (http://127.0.0.1:5000/#/compare-runs):

**Essential Parameters to Add:**

- `model_name` - Model identifier (yolov8_pose, mediapipe)
- `device` - Compute device (cuda, mps, cpu)
- `confidence_threshold` - Detection threshold
- `model_size` - YOLOv8 variant (n,s,m,l) or MediaPipe complexity (0,1,2)

**Key Metrics to Compare:**

- `pose_pck_error_mean` - Accuracy (lower is better)
- `perf_fps_mean` - Speed (higher is better)
- `pose_detection_f1_mean` - Detection quality (higher is better)
- `perf_avg_inference_time_mean` - Inference time (lower is better)
- `pose_detection_precision_mean` - Precision
- `pose_detection_recall_mean` - Recall

**Analysis Priority:**

1. Sort by `pose_pck_error_mean` for best accuracy
2. Filter `pose_detection_f1_mean` > 0.5 for viable models
3. Plot `perf_fps_mean` vs `pose_pck_error_mean` for speed/accuracy trade-offs

## Usage Examples

### Python API

```python
from evaluate_pose_models import PoseEvaluator

# Quick screening
evaluator = PoseEvaluator(config_path="configs/evaluation_config.yaml")
results = evaluator.quick_screening(
    models=["mediapipe", "yolov8_pose"],
    num_clips=50
)

# Comprehensive evaluation with optimization
results = evaluator.full_evaluation(
    models=["mmpose", "hrnet", "mediapipe", "yolov8_pose"],
    num_clips=300,
    use_optuna=True
)
```

### Command Line

```bash
# Compare specific models
python evaluate_pose_models.py \
    --models mediapipe mmpose yolov8_pose \
    --max-clips 100 \
    --video-format ffv1

# Quick development test
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --quick-test --verbose
```

## Troubleshooting

### macOS Issues

- **MPS not available**: Requires macOS 12.3+
- **Memory errors**: Reduce batch size and clip count
- **Package conflicts**: Use `environment_simple.yml`

### Linux Issues

- **CUDA errors**: Check drivers with `nvidia-smi`
- **Out of memory**: Reduce batch size
- **MMPose installation**: Try manual `pip install mmengine mmcv mmdet mmpose`

### General Issues

```bash
# Test core functionality
python test_zoom_loading.py
python -c "from models.mediapipe_wrapper import MediaPipeWrapper; print('OK')"

# Check acceleration
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

## Data Requirements

**Minimum (Testing)**: 10-20 labeled clips, balanced cameras/sessions  
**Recommended (Production)**: 200-300 clips, multiple cameras (SONY_300, SONY_70, GP1, GP2), various conditions

## Contributing

To add new pose models:

1. Inherit from `BasePoseModel` in `models/base_pose_model.py`
2. Add configuration in `configs/model_configs/`
3. Update environment files with dependencies
4. Test on both macOS and Linux environments

## Results

Results stored in multiple formats:

- JSON files for programmatic analysis
- MLflow experiments for interactive exploration
- Visualization plots and video overlays
- Performance benchmarks and model comparisons

---

For questions or issues, check the troubleshooting section above or examine the test scripts for debugging.
