# GPU Acceleration Optimization Guide

## Current Issues Identified

### Models Using GPU Acceleration:

- ✅ **YOLOv8-Pose**: Properly GPU-accelerated on both CUDA and MPS
- ⚠️ **PyTorch Pose**: CUDA support but MPS disabled on macOS

### Models NOT Using GPU:

- ❌ **MediaPipe**: Explicitly disabled via `MEDIAPIPE_DISABLE_GPU=1`
- ❌ **BlazePose**: Inherits MediaPipe's CPU-only limitation
- ⚠️ **MMPose**: CUDA support but forces CPU when MPS is requested

## Optimization Steps

### For Linux (RTX 4090) - Enable CUDA Acceleration

#### 1. Enable MediaPipe GPU Support

Remove or modify the GPU disable settings in:

**File: `evaluate_pose_models.py` (lines 10-11)**

```python
# BEFORE (CPU-only):
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # Force CPU only
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Disable GPU memory growth

# AFTER (GPU-enabled):
os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"  # Enable GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth
```

**File: `models/mediapipe_wrapper.py` (lines 47-48)**

```python
# BEFORE (CPU-only):
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # Force CPU only
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Disable GPU memory growth

# AFTER (GPU-enabled):
os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"  # Enable GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth
```

#### 2. Enable MMPose CUDA Support

**File: `models/mmpose_wrapper.py` (line 41)**

```python
# BEFORE (forces CPU on non-CUDA):
self.mmpose_device = device if device != "mps" else "cpu"

# AFTER (uses CUDA when available):
self.mmpose_device = device if device in ["cuda", "cpu"] else "cpu"
```

#### 3. Verify PyTorch CUDA Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name()}')"
```

#### 4. Production Config Optimization

Update `configs/evaluation_config_production_optuna.yaml` and `configs/evaluation_config_production_comparison.yaml`:

```yaml
models:
  default_settings:
    device: "cuda"
    half_precision: true # Enable FP16 for faster inference on RTX 4090

hardware:
  gpu:
    device_id: 0
    memory_fraction: 0.9
    allow_growth: true
```

### For macOS (MPS) - Optimize Metal Performance Shaders

#### 1. Keep MediaPipe CPU-only (recommended for stability)

MediaPipe GPU support on macOS is experimental. Keep current CPU settings.

#### 2. Enable PyTorch MPS Support

**File: `models/pytorch_pose_wrapper.py` (lines 28-30)**

```python
# BEFORE (forces CPU on MPS):
if device == "mps":
    print("Warning: PyTorch KeypointRCNN has MPS compatibility issues, forcing CPU usage")
    device = "cpu"

# AFTER (enable MPS with fallback):
if device == "mps":
    if torch.backends.mps.is_available():
        print("Using MPS acceleration for PyTorch KeypointRCNN")
    else:
        print("MPS not available, falling back to CPU")
        device = "cpu"
```

#### 3. Optimize YOLOv8 for MPS

**File: `configs/model_configs/yolov8_pose.yaml`**

```yaml
# Performance settings for MPS
half_precision: false # MPS doesn't support FP16 well
device_specific_optimization: true
```

#### 4. macOS Config Optimization

**File: `configs/evaluation_config_macos.yaml`**

```yaml
models:
  default_settings:
    device: "mps"
    half_precision: false # MPS compatibility

hardware:
  device:
    primary: "mps"
    fallback: "cpu"
```

## Verification Commands

### Linux CUDA Verification:

```bash
# Check GPU utilization during inference
nvidia-smi -l 1

# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Check CUDA in Python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"
```

### macOS MPS Verification:

```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available()); print(torch.backends.mps.is_built())"

# Monitor process and memory during inference
top -pid $(pgrep -f "python.*evaluate_pose_models")
```

## Expected Performance Improvements

### Linux (RTX 4090):

- **MediaPipe**: 2-5x speedup with GPU acceleration
- **YOLOv8**: Already optimized, should see full GPU utilization
- **MMPose**: 3-10x speedup with CUDA
- **Overall**: Expect 60-80% reduction in inference time

### macOS (MPS):

- **YOLOv8**: Should see GPU utilization in Activity Monitor
- **PyTorch Pose**: 2-3x speedup over CPU
- **MediaPipe/BlazePose**: Keep CPU for stability
- **Overall**: Expect 30-50% reduction in inference time

## Troubleshooting

### If GPU Memory Issues Occur:

```python
# Add to model loading:
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8)
```

### If CUDA Out of Memory:

- Reduce batch size to 1
- Enable gradient checkpointing
- Use mixed precision training

### If MPS Issues Occur:

- Fall back to CPU for problematic models
- Monitor memory usage with Activity Monitor
- Ensure macOS >= 12.3 for stable MPS support
