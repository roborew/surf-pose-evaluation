# Deployment Guide: macOS vs Training Machine

This guide explains how to set up and run the pose evaluation framework on different environments.

## Environment Comparison

| Aspect           | macOS (Local Dev)     | Training Machine (Production)    |
| ---------------- | --------------------- | -------------------------------- |
| **Hardware**     | Apple Silicon M1/M2   | NVIDIA RTX 4090                  |
| **Acceleration** | MPS (Metal)           | CUDA                             |
| **Memory**       | 8-16GB unified        | 32GB+ RAM + 24GB VRAM            |
| **Video Format** | H264                  | FFV1 (lossless)                  |
| **Models**       | MediaPipe, basic YOLO | All models (MMPose, HRNet, etc.) |
| **Dataset Size** | 10-50 clips           | 200-300 clips                    |
| **Purpose**      | Development, testing  | Full evaluation, research        |

## Setup Instructions

### macOS Local Development

#### 1. Installation

```bash
cd surf_pose_evaluation
conda env create -f environment_macos.yml
conda activate surf_pose_eval
```

#### 2. Configuration

Uses `configs/evaluation_config_macos.yaml` with:

- `device: "mps"` for Apple Silicon acceleration
- Reduced clip counts for faster iteration
- Only lightweight models enabled

#### 3. Usage

```bash
# Quick development test
python evaluate_pose_models.py \
    --config configs/evaluation_config_macos.yaml \
    --quick-test \
    --models mediapipe \
    --max-clips 10

# Test specific components
python models/mediapipe_wrapper.py
python data/data_loader.py
```

#### 4. Limitations

- No MMPose (complex dependencies on Apple Silicon)
- Limited to CPU/MPS processing
- Smaller datasets for memory constraints
- H264 video only (FFV1 may not be available locally)

### Training Machine Production

#### 1. Installation

```bash
cd surf_pose_evaluation

# Try main environment first
conda env create -f environment.yml
conda activate surf_pose_eval

# If conflicts, use simplified version
conda env create -f environment_simple.yml
conda activate surf_pose_eval

# Add remaining models
pip install mmengine mmcv mmdet mmpose ultralytics
```

#### 2. Configuration

Uses `configs/evaluation_config.yaml` with:

- `device: "cuda"` for GPU acceleration
- Full clip counts for comprehensive evaluation
- All pose models enabled

#### 3. Usage

```bash
# Full evaluation (run in tmux for long sessions)
tmux new-session -d -s pose_eval
tmux send-keys -t pose_eval "conda activate surf_pose_eval" Enter
tmux send-keys -t pose_eval "python evaluate_pose_models.py --config configs/evaluation_config.yaml --models mediapipe mmpose yolov8_pose hrnet --video-format ffv1" Enter

# Monitor progress
tmux attach -t pose_eval

# Quick screening of models
python evaluate_pose_models.py \
    --config configs/evaluation_config.yaml \
    --quick-test \
    --models mediapipe mmpose yolov8_pose \
    --max-clips 50
```

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
# Option 1: rsync
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

## Data Management

### Local Development Data

- Use H264 compressed videos (smaller file sizes)
- Subset of sessions/cameras for testing
- Store in local `data/` directory

### Training Machine Data

- Use FFV1 lossless videos (highest quality)
- Full dataset with all sessions and cameras
- Potentially mounted from network storage

## Results Management

### Local Results

```bash
# View local MLflow results
mlflow ui --backend-store-uri ./results/mlruns
# Access at http://localhost:5000
```

### Training Machine Results

```bash
# Option 1: SSH tunnel for MLflow UI
ssh -L 5000:localhost:5000 user@training-machine
# Then on training machine: mlflow ui --backend-store-uri ./results/mlruns
# Access at http://localhost:5000 on local machine

# Option 2: Download results
rsync -av user@training-machine:~/surf_pose_evaluation/results/ ./results/
```

## Performance Expectations

### macOS Performance

- **MediaPipe**: ~15-25 FPS on Apple Silicon
- **Memory Usage**: 2-4GB RAM
- **Good for**: Code development, algorithm testing, quick validation

### Training Machine Performance

- **MediaPipe**: ~30-60 FPS on RTX 4090
- **MMPose**: ~10-20 FPS with high accuracy
- **YOLOv8**: ~20-40 FPS with good balance
- **Memory Usage**: 8-16GB RAM + 4-12GB VRAM
- **Good for**: Comprehensive evaluation, final model selection

## Troubleshooting

### macOS Issues

1. **MPS not available**: Check macOS version (requires 12.3+)
2. **Memory errors**: Reduce batch size and clip count
3. **Package conflicts**: Use pip installation as fallback

### Training Machine Issues

1. **CUDA errors**: Check driver compatibility
2. **Out of memory**: Reduce batch size or use gradient checkpointing
3. **Slow performance**: Verify GPU utilization with `nvidia-smi`

## File Structure Sync

Keep these files synchronized between environments:

**Always Sync:**

- `configs/` - Configuration files
- `models/` - Model wrapper code
- `metrics/` - Evaluation metrics
- `data/data_loader.py` - Data loading logic
- `evaluate_pose_models.py` - Main evaluation script

**Environment Specific:**

- `environment_macos.yml` vs `environment.yml`
- `configs/evaluation_config_macos.yaml` vs `configs/evaluation_config.yaml`

**Don't Sync:**

- `results/` - Environment-specific results
- `__pycache__/` - Python cache files
- `.conda/` - Conda environment files

This workflow allows you to develop efficiently on macOS while leveraging the full power of GPU training machines for comprehensive evaluation.
