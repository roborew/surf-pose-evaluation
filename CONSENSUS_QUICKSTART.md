# Consensus-Based Optuna - Quick Start Guide

## 🎯 What Is This?

Consensus-based validation solves the problem of **PCK scores being 0** during Optuna optimization. Since surf footage lacks manual pose annotations, we generate pseudo-ground-truth by running multiple strong models and using their consensus as the "truth".

## 🚀 Run It (3 Commands)

### 1. Quick Test (5 minutes)

```bash
python run_evaluation.py \
    --eval-mode quick \
    --run-name "test_consensus" \
    @configs/evaluation_config_production_optuna.yaml
```

### 2. Full Optuna (30-60 minutes)

```bash
python run_evaluation.py \
    --eval-mode optuna \
    --run-name "surf_optuna" \
    @configs/evaluation_config_production_optuna.yaml
```

### 3. Check Results

```bash
cd runs/test_consensus
mlflow ui
# Open http://localhost:5000 in browser
```

## ✅ Verify It Works

### Check 1: Console Output

Look for:

```
✓ Consensus-based validation enabled for Optuna
✓ Loading consensus GT for yolov8 (Optuna phase)
✓ Consensus generation complete for yolov8:
    Success: 75/75 maneuvers
```

### Check 2: Files Created

```bash
ls runs/test_consensus/consensus_cache/
# Should see: yolov8_optuna_gt.json, pytorch_pose_optuna_gt.json, etc.
```

### Check 3: MLflow Metrics

In MLflow UI, look for:

- **pck_0_2 metric**: Should be > 0 (e.g., 0.45, 0.62)
- **validation_method param**: `consensus_based`

## 🎓 How It Works

### The Problem

```
❌ Surf footage has no manual pose annotations
❌ PCK = 0.0 (can't measure accuracy without ground truth)
❌ Optuna can't optimize (no meaningful metric)
```

### The Solution

```
✅ Run 3 strong models (YOLOv8, PyTorch Pose, MMPose) on each clip
✅ Average their predictions = "consensus ground truth"
✅ Compare optimization target against consensus
✅ PCK > 0 (now we can measure accuracy!)
✅ Optuna optimizes successfully
```

### Leave-One-Out Logic

```
Optimizing YOLOv8?     → Use PyTorch Pose + MMPose consensus
Optimizing PyTorch?    → Use YOLOv8 + MMPose consensus
Optimizing MMPose?     → Use YOLOv8 + PyTorch Pose consensus
Optimizing MediaPipe?  → Use all 3 strong models
Optimizing BlazePose?  → Use all 3 strong models
```

## 📊 Expected Results

| Model        | Typical PCK@0.2 |
| ------------ | --------------- |
| YOLOv8       | 0.55 - 0.75     |
| PyTorch Pose | 0.60 - 0.80     |
| MMPose       | 0.58 - 0.78     |
| MediaPipe    | 0.30 - 0.50     |
| BlazePose    | 0.35 - 0.55     |

## 🐛 Troubleshooting

### PCK still 0.0?

Check config has:

```yaml
optuna_validation:
  use_consensus: true # ← Must be true!
```

### First trial slow?

**Normal!** First trial generates consensus (~5-10 min).  
Subsequent trials use cache (~30-60 sec).

### No consensus_cache folder?

Check logs for errors:

```bash
grep -i "consensus" runs/test_consensus/logs/*.log
```

## 📝 Configuration

Edit `configs/evaluation_config_production_optuna.yaml`:

```yaml
# Enable consensus validation
optuna_validation:
  use_consensus: true # ← Turn on/off here
  num_clips: 75 # Number of clips for Optuna
  cameras: [SONY_300] # Which camera(s) to use

# Models to optimize
models_to_optimize:
  - yolov8
  - pytorch_pose
  - mmpose
  - mediapipe
  - blazepose
```

## 🎯 What You Get

### Before Consensus

```
Trial 1: PCK = 0.0
Trial 2: PCK = 0.0
Trial 3: PCK = 0.0
❌ No improvement possible
```

### After Consensus

```
Trial 1: PCK = 0.523
Trial 2: PCK = 0.547 (+0.024)
Trial 3: PCK = 0.561 (+0.014)
✅ Clear optimization progress!
```

## 📚 More Information

- **Full details:** See `CONSENSUS_IMPLEMENTATION_COMPLETE.md`
- **Technical docs:** See `CONSENSUS_IMPLEMENTATION_STATUS.md`
- **Implementation plan:** See `consensus-based-optuna-optimization.plan.md`

---

**Status:** ✅ Ready to use  
**Quick test time:** ~5 minutes  
**Full Optuna time:** ~30-60 minutes
