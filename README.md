# Surfing Pose Estimation Evaluation Framework

A comprehensive system for comparing and evaluating multiple pose estimation libraries for surfing performance analysis using MLflow experiment tracking and Optuna hyperparameter optimization.

## Project Overview

This framework systematically compares pose estimation models to determine the optimal backbone for surfing action recognition. It supports both local testing with H264 video and production evaluation with FFV1 lossless video on training machines.

## Supported Pose Estimation Libraries

- **MMPose** - Current baseline with extensive model zoo
- **MediaPipe Pose** - Fast, edge-optimized for real-time inference
- **YOLOv8-Pose** - Unified detection and pose estimation
- **HRNet** - High-accuracy reference implementation
- **BlazePose** - Mobile-first design for lightweight deployment

> **Note**: OpenPose has been excluded due to installation complexity, poor performance (5 FPS vs 20-60 FPS for modern alternatives), and limited practical applicability for real-time surfing analysis.

## Project Structure

```
surf_pose_evaluation/
├── README.md                       # This file
├── DEPLOYMENT_GUIDE.md            # Detailed deployment instructions
├── environment.yml                 # Linux/CUDA environment (training machines)
├── environment_macos.yml           # macOS/Apple Silicon environment
├── environment_simple.yml          # Simplified environment (fallback)
├── requirements.txt               # pip requirements (fallback)
├── test_zoom_loading.py           # Data loading verification script
├── evaluate_pose_models.py        # Main evaluation script
├── configs/
│   ├── evaluation_config.yaml      # Production configuration (Linux/CUDA)
│   ├── evaluation_config_macos.yaml # Development configuration (macOS)
│   └── model_configs/              # Individual model configurations
├── models/
│   ├── base_pose_model.py          # Abstract base class for pose models
│   ├── mediapipe_wrapper.py        # MediaPipe Pose implementation
│   ├── mmpose_wrapper.py           # MMPose implementation
│   ├── yolov8_wrapper.py           # YOLOv8-Pose implementation
│   └── hrnet_wrapper.py            # HRNet implementation
├── metrics/
│   ├── pose_metrics.py             # PCK, MPJPE, temporal consistency
│   └── performance_metrics.py      # Latency, memory, throughput
├── data_handling/
│   ├── data_loader.py              # Zoom-aware video and annotation loading
│   └── video_utils.py              # Video processing utilities
├── utils/
│   ├── mlflow_utils.py             # MLflow experiment management
│   ├── optuna_utils.py             # Hyperparameter optimization
│   └── visualization.py           # Result visualization tools
├── data/                           # Your actual dataset
│   └── SD_02_SURF_FOOTAGE_PREPT/   # Video clips and annotations
└── results/
    ├── pose_comparison_results.json
    ├── mlruns/                     # MLflow experiment artifacts
    └── visualizations/             # Generated plots and videos
```

## Environment Setup

The framework supports two deployment environments with different capabilities:

### Option 1: macOS Local Development (Apple Silicon)

**Best for**: Development, testing, quick iterations

```bash
# Clone and setup
cd surf_pose_evaluation
conda env create -f environment_macos.yml
conda activate surf_pose_eval
```

**Capabilities**:

- ✅ MediaPipe (optimized for Apple Silicon)
- ✅ Basic YOLOv8-Pose
- ✅ MPS acceleration
- ❌ MMPose (complex dependencies)
- ❌ HRNet (CUDA required)

**Hardware Requirements**:

- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3+ (for MPS support)
- 8GB+ unified memory

### Option 2: Linux Training Machines (NVIDIA GPU)

**Best for**: Production evaluation, comprehensive comparison

```bash
cd surf_pose_evaluation

# Try main environment first
conda env create -f environment.yml
conda activate surf_pose_eval

# If conflicts occur, use simplified version
conda env create -f environment_simple.yml
conda activate surf_pose_eval

# Add remaining models manually
pip install mmengine mmcv mmdet mmpose ultralytics
```

**Capabilities**:

- ✅ All pose models (MediaPipe, MMPose, YOLOv8, HRNet)
- ✅ CUDA acceleration
- ✅ Large batch processing
- ✅ FFV1 lossless video support

**Hardware Requirements**:

- NVIDIA GPU (RTX 3060+ recommended, RTX 4090 optimal)
- 8GB+ VRAM for most models
- 16GB+ RAM (32GB+ for large batches)
- Fast SSD storage for video files

### Option 3: Fallback Installation

If conda environments fail:

```bash
pip install -r requirements.txt
# Then manually install specific model libraries as needed
```

## Quick Start & Testing

### 1. Verify Installation

Test your environment setup first:

```bash
# Test zoom-aware data loading (critical for preventing data leakage)
python test_zoom_loading.py

# Test individual model wrappers
python -c "from models.mediapipe_wrapper import MediaPipeWrapper; print('MediaPipe OK')"

# macOS only - test MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Linux only - test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Quick Development Test (macOS)

```bash
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --quick-test \
    --models mediapipe \
    --max-clips 5 \
    --verbose
```

### 3. Quick Development Test (Linux)

```bash
python evaluate_pose_models.py \
    --config configs/evaluation_config.yaml \
    --quick-test \
    --models mediapipe \
    --max-clips 10 \
    --verbose
```

### 4. Production Evaluation (Linux Training Machines)

```bash
# Run in tmux for long sessions
tmux new-session -d -s pose_eval
tmux send-keys -t pose_eval "conda activate surf_pose_eval" Enter
tmux send-keys -t pose_eval "python evaluate_pose_models.py --config configs/evaluation_config.yaml --models mediapipe mmpose yolov8_pose hrnet --video-format ffv1 --max-clips 300" Enter

# Monitor progress
tmux attach -t pose_eval
```

## Key Features & Data Handling

### Zoom-Aware Data Loading

The framework includes sophisticated data handling that **prevents data leakage** from zoom variations:

- **Problem**: Each labeled clip has 3 zoom levels (default, wide, full) but annotations reference only one
- **Solution**: Intelligent selection ensuring only one zoom variation per base clip
- **Benefit**: Balanced distribution (~33% each zoom level) with no data leakage across train/val/test splits

Verify this works:

```bash
python test_zoom_loading.py
```

Expected output shows balanced zoom distribution and no base clip duplication across splits.

### Supported Video Formats

- **H264**: Compressed videos for local development (smaller files)
- **FFV1**: Lossless videos for production evaluation (highest quality)

## Configuration

### Development Configuration (`configs/evaluation_config_macos.yaml`)

- `device: "mps"` for Apple Silicon acceleration
- Reduced clip counts for faster iteration
- Only lightweight models enabled
- H264 video format

### Production Configuration (`configs/evaluation_config.yaml`)

- `device: "cuda"` for GPU acceleration
- Full clip counts for comprehensive evaluation
- All pose models enabled
- FFV1 video format support

## Evaluation Metrics

### Quantitative Metrics

- **PCK@0.2**: Percentage of Correct Keypoints at 20% threshold
- **MPJPE**: Mean Per Joint Position Error (3D models)
- **Inference Latency**: Milliseconds per frame
- **Memory Usage**: GPU VRAM consumption
- **Temporal Consistency**: Frame-to-frame stability

### Qualitative Assessment

- Robustness to occlusion (water spray, surfboard blocking)
- Performance in challenging lighting conditions
- Accuracy across different wave conditions and surfer skill levels

## MLflow Integration

All experiments are automatically logged to MLflow with:

- Model parameters and hyperparameters
- Evaluation metrics and performance benchmarks
- Video samples with pose overlay visualizations
- Model artifacts and checkpoints

Access the MLflow UI:

```bash
mlflow ui --backend-store-uri ./results/mlruns
# Open http://localhost:5000
```

For remote training machines, use SSH tunneling:

```bash
ssh -L 5000:localhost:5000 user@training-machine
# Then run mlflow ui on the remote machine
```

## Performance Expectations

### macOS Performance (Apple Silicon)

- **MediaPipe**: ~15-25 FPS
- **Memory Usage**: 2-4GB RAM
- **Good for**: Code development, algorithm testing, quick validation

### Linux Training Machine Performance (RTX 4090)

- **MediaPipe**: ~30-60 FPS
- **MMPose**: ~10-20 FPS with high accuracy
- **YOLOv8**: ~20-40 FPS with good balance
- **HRNet**: ~5-15 FPS with highest accuracy
- **Memory Usage**: 8-16GB RAM + 4-12GB VRAM

## Development Workflow

### Phase 1: Local Development (macOS)

1. **Code Development**: Write and test new features locally
2. **Quick Validation**: Test with MediaPipe on small datasets
3. **Debug Issues**: Fast iteration with immediate feedback

```bash
# Typical local development cycle
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --quick-test \
    --models mediapipe \
    --max-clips 5 \
    --verbose
```

### Phase 2: Transfer to Training Machine

```bash
# Option 1: rsync (exclude large files)
rsync -av --exclude results/ --exclude __pycache__/ \
    surf_pose_evaluation/ user@training-machine:~/surf_pose_evaluation/

# Option 2: Git (recommended)
git add .
git commit -m "Updated pose evaluation framework"
git push
# Then on training machine: git pull
```

### Phase 3: Production Evaluation

```bash
# SSH to training machine
ssh user@training-machine

# Activate environment and run full evaluation
cd ~/surf_pose_evaluation
conda activate surf_pose_eval

# Comprehensive evaluation
python evaluate_pose_models.py \
    --config configs/evaluation_config.yaml \
    --models mediapipe mmpose yolov8_pose hrnet \
    --video-format ffv1 \
    --max-clips 300
```

## Troubleshooting

### Common macOS Issues

1. **MPS not available**: Requires macOS 12.3+, check with:

   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **Memory errors**: Reduce batch size and clip count in config

3. **Package conflicts**: Try simplified environment:
   ```bash
   conda env create -f environment_simple.yml
   ```

### Common Linux Issues

1. **CUDA errors**: Check driver compatibility:

   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Out of memory**: Reduce batch size or use gradient checkpointing

3. **Slow performance**: Verify GPU utilization with `nvidia-smi`

4. **MMPose installation issues**: Try manual installation:
   ```bash
   pip install mmengine mmcv mmdet mmpose
   ```

## Usage Examples

### Quick Screening Test

```python
from evaluate_pose_models import PoseEvaluator

evaluator = PoseEvaluator(config_path="configs/evaluation_config.yaml")
results = evaluator.quick_screening(
    models=["mediapipe", "yolov8_pose"],
    num_clips=50
)
```

### Comprehensive Evaluation

```python
evaluator = PoseEvaluator(config_path="configs/evaluation_config.yaml")
results = evaluator.full_evaluation(
    models=["mmpose", "hrnet", "mediapipe", "yolov8_pose"],
    num_clips=300,
    use_optuna=True
)
```

## Results and Analysis

Results are stored in multiple formats:

- JSON files for programmatic analysis
- MLflow experiments for interactive exploration
- Visualization plots for presentation
- Video overlays for qualitative assessment

## Contributing

When adding new pose estimation models:

1. Inherit from `BasePoseModel` in `models/base_pose_model.py`
2. Implement required methods with consistent interface
3. Add model configuration in `configs/model_configs/`
4. Update environment files with any new dependencies
5. Test on both macOS and Linux if possible

## Data Requirements

### Minimum Requirements for Testing

- 10-20 labeled video clips for quick testing
- Balanced distribution across cameras and sessions
- At least 3 different lighting conditions

### Recommended for Production

- 200-300 labeled video clips
- Multiple cameras (SONY_300, SONY_70, GP1, GP2)
- Various sessions and environmental conditions
- Balanced maneuver distribution

## Citation

If you use this framework in your research, please cite:

```bibtex
[Your dissertation citation will go here]
```

## Additional Resources

- **DEPLOYMENT_GUIDE.md**: Detailed deployment instructions for different environments
- **test_zoom_loading.py**: Comprehensive test script for data loading verification
- **configs/**: Example configurations for different use cases

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).
