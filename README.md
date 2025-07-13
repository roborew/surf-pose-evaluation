# Surfing Pose Estimation Evaluation Framework

A comprehensive system for comparing and evaluating multiple pose estimation libraries for surfing performance analysis using MLflow experiment tracking, Optuna hyperparameter optimization, and configurable video visualization.

## Project Overview

This framework systematically compares pose estimation models to determine the optimal backbone for surfing action recognition. It supports both local development (macOS) and production evaluation (Linux/CUDA) environments with advanced visualization capabilities and cross-project data sharing.

### Supported Models

- **MediaPipe Pose** - Fast, edge-optimized for real-time inference ✅
- **BlazePose** - Google's optimized real-time 3D pose estimation ✅
- **YOLOv8-Pose** - Unified detection and pose estimation ✅
- **MMPose** - Industry-standard framework (includes HRNet backbones) ✅
- **PyTorch KeypointRCNN** - Torchvision's pose estimation model ✅

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

**Capabilities:** ✅ MediaPipe ✅ BlazePose ✅ YOLOv8 ✅ PyTorch ✅ MPS acceleration ✅ MMPose

#### Linux Setup (Production)

```bash
# Create the production environment
conda env create -f environment.yml
conda activate surf_pose_eval

# Verify installation
python test_zoom_loading.py
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

**Key features of the Linux environment:**

- ✅ Python 3.10 for better compatibility
- ✅ OpenCV via pip to avoid JPEG conflicts
- ✅ All pose estimation libraries included
- ✅ CUDA 11.8 support maintained
- ✅ Complete MMPose ecosystem (mmengine, mmcv, mmdet, mmpose)
- ✅ All visualization and analysis tools

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

### 3. Full Evaluation with Visualization

```bash
# Run in tmux for long sessions (Linux)
tmux new-session -d -s pose_eval
tmux send-keys -t pose_eval "conda activate surf_pose_eval" Enter
tmux send-keys -t pose_eval "python evaluate_pose_models.py --config configs/evaluation_config.yaml --models mediapipe blazepose yolov8_pose mmpose --max-clips 300" Enter
tmux attach -t pose_eval
```

## Environment Files

The project includes two optimized environment files:

- **`environment.yml`** - Linux/Ubuntu production environment with CUDA support
- **`environment_macos.yml`** - macOS development environment with MPS acceleration

Both environments include all necessary packages for complete pose estimation evaluation, including:
- All pose estimation models (MediaPipe, BlazePose, YOLOv8, MMPose, PyTorch KeypointRCNN)
- Video processing and visualization tools
- MLflow experiment tracking
- Optuna hyperparameter optimization
- Development and testing utilities

## Project Structure

```
surf_pose_evaluation/
├── README.md                       # This comprehensive guide
├── environment.yml                 # Linux/Ubuntu production environment
├── environment_macos.yml           # macOS development environment
├── evaluate_pose_models.py        # Main evaluation script
├── configs/
│   ├── evaluation_config*.yaml     # Environment-specific configs
│   ├── evaluation_config_production.yaml  # Lossless video format
│   └── model_configs/              # Individual model configurations
├── models/                         # Pose model wrappers
├── metrics/                        # Evaluation metrics
├── data_handling/                  # Zoom-aware data loading
├── utils/
│   ├── mlflow_utils.py            # MLflow experiment tracking
│   ├── visualization.py           # Static visualization utilities
│   └── pose_video_visualizer.py   # Video visualization with pose overlays
├── data/
│   └── SD_02_SURF_FOOTAGE_PREPT/
│       └── 05_ANALYSED_DATA/
│           └── POSE/               # Shared storage for cross-project sync
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

### Advanced Visualization System

**High-quality pose video generation with:**

- Color-coded anatomical regions (left/right limbs, torso, face)
- Professional video encoding (H.264, H.265, lossless formats)
- Audio preservation from original footage
- Model identification and frame counters
- Automatic MLflow integration

**Shared Storage Integration:**

- Cross-project synchronization via configurable shared storage
- Timestamped directories prevent overwrites
- Comprehensive metadata tracking
- Organized directory structure for team collaboration

### Hyperparameter Optimization

**Optuna integration for automatic model tuning:**

- TPE (Tree-structured Parzen Estimator) sampling
- Early pruning for poor trials
- Multi-objective optimization (accuracy, speed, memory)
- Comprehensive trial tracking in MLflow

### Comprehensive Metrics

- **Accuracy**: PCK@0.2, MPJPE, detection metrics
- **Performance**: Inference latency, memory usage, FPS
- **Temporal**: Frame-to-frame consistency and stability

## Configuration

### Configuration Files

| File                                | Purpose            | Environment   | Video Format  |
| ----------------------------------- | ------------------ | ------------- | ------------- |
| `evaluation_config_macos.yaml`      | Development        | Apple Silicon | H.264 MP4     |
| `evaluation_config.yaml`            | Production         | CUDA/Linux    | H.264 MP4     |
| `evaluation_config_production.yaml` | Production Testing | CUDA/Linux    | Lossless MKV  |
| `evaluation_config_local.yaml`      | Local Development  | Any           | Local storage |

### Video Encoding Formats

**Configurable video encoding for different use cases:**

#### H.264 (Default - Recommended for Annotated Results)

```yaml
encoding:
  format: "h264"
  quality:
    crf: 23 # 18-28 range, lower = better quality
    preset: "fast" # ultrafast to veryslow
  pixel_format: "yuv420p"
  container: "mp4"
```

**Characteristics:**

- ✅ Small file sizes (good for sharing)
- ✅ Wide compatibility (plays everywhere)
- ✅ Fast encoding (quick processing)
- ❌ Lossy compression (not suitable for production testing)

#### Lossless (Production Testing)

```yaml
encoding:
  format: "lossless"
  quality:
    crf: 0 # Lossless
    preset: "veryslow" # Best compression
  pixel_format: "yuv444p" # Full chroma
  container: "mkv"
```

**Characteristics:**

- ✅ Perfect quality (no compression artifacts)
- ✅ Suitable for production testing
- ❌ Very large files (10-50x larger)
- ❌ Very slow encoding (10x longer)

#### Other Formats Available

- **H.265/HEVC**: Better compression, slower encoding
- **FFV1**: True lossless, open source
- **ProRes**: Professional quality for macOS

### Shared Storage Configuration

**Flexible storage options for cross-project collaboration:**

```yaml
output:
  visualization:
    # Shared storage (default)
    shared_storage_path: "data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE"

    # Local storage (development)
    # shared_storage_path: null  # Omit for local storage

    # Custom path
    # shared_storage_path: "/path/to/your/custom/storage"
```

**Storage Benefits:**

- **Cross-project sharing**: Results synced across projects
- **Timestamped directories**: No overwrites, chronological ordering
- **Metadata tracking**: Complete processing history
- **Team collaboration**: Centralized access to visualizations

## Performance Expectations

| Environment | Hardware      | MediaPipe | MMPose    | YOLOv8    | PyTorch   | Memory                   |
| ----------- | ------------- | --------- | --------- | --------- | --------- | ------------------------ |
| **macOS**   | Apple Silicon | 15-25 FPS | 8-15 FPS  | 10-20 FPS | 5-12 FPS  | 2-4GB RAM                |
| **Linux**   | RTX 4090      | 30-60 FPS | 10-20 FPS | 20-40 FPS | 15-25 FPS | 8-16GB RAM + 4-12GB VRAM |

## Development Workflow

### 1. Local Development (macOS)

```bash
# Quick iteration cycle
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --quick-test --models mediapipe --max-clips 5 --verbose
```

### 2. Hyperparameter Optimization

```bash
# Automatic model tuning with Optuna
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --models mediapipe --use-optuna --max-clips 50
```

### 3. Production Evaluation with Lossless Video

```bash
# Production testing with lossless format
python evaluate_pose_models.py \
    --config configs/evaluation_config_production.yaml \
    --models mediapipe yolov8_pose --max-clips 100
```

### 4. Transfer to Production

```bash
# Git method (recommended)
git add . && git commit -m "Updated framework" && git push
# On training machine: git pull

# Or rsync method
rsync -av --exclude results/ --exclude __pycache__/ \
    surf_pose_evaluation/ user@training-machine:~/surf_pose_evaluation/
```

## MLflow Integration

All experiments automatically logged with:

- Model parameters and hyperparameters
- Evaluation metrics and performance benchmarks
- Video samples with pose overlays
- Model artifacts and checkpoints
- Optuna trial results and optimization history

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
    models=["mmpose", "mediapipe", "blazepose", "yolov8_pose"],
    num_clips=300,
    use_optuna=True
)
```

### Command Line Examples

```bash
# Compare specific models with visualizations
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --models mediapipe yolov8_pose \
    --max-clips 100

# Quick development test
python evaluate_pose_models.py \
    --config configs/evaluation_config_local.yaml \
    --quick-test --verbose

# Production evaluation with lossless video
python evaluate_pose_models.py \
    --config configs/evaluation_config_production.yaml \
    --models all --max-clips 300

# Hyperparameter optimization
python evaluate_pose_models.py \
    --config configs/evaluation_config.yaml \
    --models mediapipe --use-optuna
```

## Visualization Features

### Automatic Video Generation

**Sample videos created for each model showing:**

- Color-coded pose keypoints and skeleton
- Anatomical region highlighting (left/right limbs, torso, face)
- Model identification and frame counters
- Professional video encoding with audio preservation

### Shared Storage Integration

**Cross-project synchronization:**

```
data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/
├── visualizations/
│   ├── 20241214_143052_mediapipe/
│   │   ├── clip_1_C0019_clip_1_poses.mp4
│   │   ├── clip_2_C0019_clip_2_poses.mp4
│   │   └── visualization_metadata.json
│   └── 20241214_143125_yolov8_pose/
│       ├── clip_1_C0019_clip_1_poses.mp4
│       └── visualization_metadata.json
```

### Video Format Options

| Format       | Use Case           | File Size  | Quality | Encoding Time |
| ------------ | ------------------ | ---------- | ------- | ------------- |
| H.264 MP4    | Annotated results  | 10-20 MB   | Good    | Fast          |
| H.265 MP4    | Better compression | 5-12 MB    | High    | Medium        |
| Lossless MKV | Production testing | 200-500 MB | Perfect | Slow          |
| FFV1 MKV     | Archival quality   | 100-300 MB | Perfect | Medium        |

## Troubleshooting

### macOS Issues

- **MPS not available**: Requires macOS 12.3+
- **Memory errors**: Reduce batch size and clip count
- **Package conflicts**: Use `environment_simple.yml`
- **MediaPipe issues**: Update to latest version (0.10.21+)

### Linux Issues

- **CUDA errors**: Check drivers with `nvidia-smi`
- **Out of memory**: Reduce batch size
- **MMPose installation**: Try manual `pip install mmengine mmcv mmdet mmpose`
- **FFmpeg not found**: Install with `sudo apt install ffmpeg`

### Visualization Issues

- **Video encoding fails**: Check FFmpeg installation
- **Large file sizes**: Use H.264 format with higher CRF values
- **Slow encoding**: Use faster presets (ultrafast, superfast)
- **Shared storage access**: Verify path permissions

### General Issues

```bash
# Test core functionality
python test_zoom_loading.py
python -c "from models.mediapipe_wrapper import MediaPipeWrapper; print('OK')"

# Check acceleration
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"

# Test video encoding
python -c "import subprocess; subprocess.run(['ffmpeg', '-version'])"
```

## Data Requirements

**Minimum (Testing)**: 10-20 labeled clips, balanced cameras/sessions  
**Recommended (Production)**: 200-300 clips, multiple cameras (SONY_300, SONY_70, GP1, GP2), various conditions

## Advanced Features

### Optuna Hyperparameter Optimization

**Automatic model tuning with:**

- TPE (Tree-structured Parzen Estimator) sampling
- Early pruning for poor-performing trials
- Multi-objective optimization (accuracy, speed, memory)
- Comprehensive trial tracking in MLflow

```bash
# Enable optimization in config
optuna:
  enabled: true
  n_trials: 50
  timeout_minutes: 180
  objective:
    primary_metric: "pck_0.2"
    secondary_metrics: ["inference_latency_ms", "memory_usage_gb"]
    weights: [0.7, 0.2, 0.1]
```

### Model-Specific Configurations

Each model has dedicated configuration files in `configs/model_configs/`:

- `mediapipe.yaml` - MediaPipe-specific parameters
- `yolov8_pose.yaml` - YOLOv8 model variants and thresholds
- `mmpose.yaml`