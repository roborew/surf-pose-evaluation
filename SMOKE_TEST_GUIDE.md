# Smoke Test Guide - Full Suite Validation
## October 27, 2025

### Purpose
Validate the complete pipeline with all models before committing to a 21-hour full comparison run.

---

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
cd /Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation
chmod +x smoke_test_full_suite.sh
./smoke_test_full_suite.sh
```

### Option 2: Manual Command

```bash
# Start tmux session for persistence
tmux new -s smoke_test

# Run smoke test
python run_evaluation.py \
  --run-name "smoke_test_full_suite_oct27" \
  --config configs/evaluation_config_production_optuna.yaml \
  --comparison-config configs/evaluation_config_production_comparison.yaml \
  --optuna-trials 5 \
  --optuna-max-clips 10 \
  --comparison-max-clips 10 \
  --eval-mode quick \
  2>&1 | tee smoke_test_full_suite.log

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t smoke_test
```

---

## What Gets Tested

| Component                    | Smoke Test              | Full Run              |
| ---------------------------- | ----------------------- | --------------------- |
| **Models**                   | All 5 models            | All 5 models          |
| **Optuna Trials**            | 5 per model             | 100 per model         |
| **Optuna Clips**             | 10 clips                | 200 clips             |
| **COCO Images**              | 10 images               | 100 images            |
| **Comparison Clips**         | 10 clips                | 715 clips             |
| **Consensus Cache**          | Generate/load           | Reuse from smoke test |
| **PyTorch Search Space**     | Constrained (0.25-0.65) | Same                  |
| **MLflow Logging**           | Single run              | Single run            |
| **Estimated Time**           | **15-20 minutes**       | **~21 hours**         |

---

## Monitoring Progress

### Watch the Log in Real-Time

```bash
tail -f smoke_test_full_suite.log
```

### Key Milestones to Watch For

1. **Consensus Cache (0-5 min)**
   ```
   ✅ Found valid consensus cache: 20251025_202539_final_full_pose_comparison
   ```
   OR
   ```
   📦 Generating consensus cache for Optuna validation...
   ```

2. **PyTorch Constrained Search (2-5 min)**
   ```
   Trial 1: confidence_threshold=0.456 (should be 0.25-0.65)
   ```
   ❌ **ABORT IF YOU SEE:** `confidence_threshold=0.98+`

3. **Optuna Trials Complete (5-10 min)**
   ```
   🎯 Best trial: 3
   📊 Best score: 0.xxx
   ✓ Saved best parameters to: best_params/pytorch_pose.yaml
   ```

4. **COCO Validation (10-12 min)**
   ```
   🎯 COCO Validation: pytorch_pose
   📊 Processing 10 images...
   ✓ COCO PCK@0.5: 0.xxx
   ```

5. **Comparison Phase (12-20 min)**
   ```
   📊 Comparison Evaluation: pytorch_pose
   Processing 10 maneuvers...
   ✓ Detection rate: 0.xxx
   ✓ Consensus PCK@0.2: 0.xxx
   ```

6. **Success (20 min)**
   ```
   ============================================
   📊 Evaluation Complete!
   ============================================
   ```

---

## Success Criteria

### ✅ Smoke Test PASSES if:

1. **No Crashes**
   - No OOM errors
   - No GPU crashes
   - Process completes cleanly

2. **PyTorch Uses Constrained Search**
   - `confidence_threshold` trials between 0.25-0.65
   - Never exceeds 0.7

3. **All Models Log Metrics**
   - Check MLflow UI shows 5 models × 3 phases
   - Each model has Optuna, COCO, and comparison results

4. **PyTorch Performance Acceptable**
   - Detection rate > 70%
   - Consensus PCK@0.2 > 70%
   - (Even with only 10 clips, should show improvement from Oct 25)

5. **Consensus Cache Works**
   - Either loads existing cache OR generates new one
   - No errors about missing predictions

---

## Checking Results

### View MLflow UI

```bash
cd data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/smoke_test_full_suite_oct27/
mlflow ui --port 5001
```

Open browser: http://localhost:5001

### Key Metrics to Check (in MLflow)

| Model         | Detection Rate | Consensus PCK@0.2 | COCO PCK@0.5 |
| ------------- | -------------- | ----------------- | ------------ |
| **PyTorch**   | > 70%          | > 70%             | > 60%        |
| **YOLOv8**    | > 80%          | > 80%             | > 70%        |
| **MMPose**    | > 85%          | > 85%             | > 75%        |
| **MediaPipe** | > 60%          | > 60%             | > 50%        |
| **BlazePose** | > 60%          | > 60%             | > 50%        |

### Check for the Oct 25 PyTorch Bug

```bash
# Should show confidence in range 0.25-0.65
cat data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/smoke_test_full_suite_oct27/best_params/pytorch_pose.yaml | grep confidence_threshold
```

❌ **ABORT FULL RUN IF:** `confidence_threshold: 0.98+`  
✅ **PROCEED IF:** `confidence_threshold: 0.25-0.65`

---

## If Smoke Test Fails

### Scenario 1: Crashes at Consensus Generation

**Symptoms:**
- Process stops during "Generating consensus cache"
- No MLflow runs created

**Diagnosis:**
```bash
# Check if consensus models can load
python -c "from models.yolov8_wrapper import YOLOv8Wrapper; print('YOLOv8 OK')"
python -c "from models.pytorch_pose_wrapper import PyTorchPoseWrapper; print('PyTorch OK')"
python -c "from models.mmpose_wrapper import MMPoseWrapper; print('MMPose OK')"
```

**Solution:**
- Verify model weights exist in `models/yolov8_pose/`, `models/pytorch_pose/`, etc.
- Check GPU is accessible: `nvidia-smi`

---

### Scenario 2: PyTorch Still Uses High Confidence

**Symptoms:**
- Log shows `Trial 1: confidence_threshold=0.98+`
- Detection rate < 50%

**Diagnosis:**
```bash
# Verify constrained ranges in config
grep -A5 "confidence_threshold:" configs/model_configs/pytorch_pose.yaml
```

**Expected:**
```yaml
confidence_threshold:
  type: "uniform"
  low: 0.25    # ← Must be 0.25
  high: 0.65   # ← Must be 0.65
```

**Solution:**
- Re-pull latest changes: `git pull`
- Manually edit `configs/model_configs/pytorch_pose.yaml`
- Re-run smoke test

---

### Scenario 3: OOM Error

**Symptoms:**
- Process killed with "Killed" message
- `dmesg` shows OOM killer

**Diagnosis:**
```bash
dmesg | grep -i "out of memory"
free -h
nvidia-smi
```

**Solution:**
- Reduce `--optuna-max-clips` to 5
- Reduce `--comparison-max-clips` to 5
- Process models sequentially: `--models yolov8`, then `--models pytorch_pose`, etc.

---

### Scenario 4: Hangs Without Progress

**Symptoms:**
- Log stops updating
- btop shows 100% CPU but no GPU activity
- No new MLflow metrics

**Diagnosis:**
```bash
# Check if process is truly stuck (no I/O)
sudo iotop -o  # Shows disk activity
sudo nethogs   # Shows network activity
```

**Solution:**
- Kill process: `pkill -9 python`
- Add timeout to Optuna trial:
  ```bash
  # In configs/evaluation_config_production_optuna.yaml
  timeout_minutes: 30  # Per model (originally 300)
  ```
- Re-run smoke test

---

## After Smoke Test Passes

### Option 1: Quick Full Run (Recommended)

Reuse the smoke test's consensus cache for immediate full run:

```bash
tmux new -s fullrun

python run_evaluation.py \
  --run-name "full_pose_comparison_oct27_final" \
  --config configs/evaluation_config_production_optuna.yaml \
  --comparison-config configs/evaluation_config_production_comparison.yaml \
  2>&1 | tee full_pose_comparison_oct27_final.log

# Detach: Ctrl+B, then D
```

**Estimated time:** ~21 hours

---

### Option 2: Overnight Run with Fresh Cache

Generate a completely new consensus cache for the full run:

```bash
# Delete smoke test cache to force regeneration
rm -rf data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/smoke_test_full_suite_oct27/consensus_cache

# Run full comparison
tmux new -s fullrun

python run_evaluation.py \
  --run-name "full_pose_comparison_oct27_final" \
  --config configs/evaluation_config_production_optuna.yaml \
  --comparison-config configs/evaluation_config_production_comparison.yaml \
  2>&1 | tee full_pose_comparison_oct27_final.log
```

**Estimated time:** ~24 hours (includes cache generation)

---

## Monitoring the Full Run

### Periodic Check (Every 2-4 Hours)

```bash
# Quick status
tail -n 50 full_pose_comparison_oct27_final.log

# Check MLflow for progress
cd data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/full_pose_comparison_oct27_final/
mlflow ui --port 5001 &
```

### Expected Timeline (Full Run)

| Time     | Phase                          | What to See                                  |
| -------- | ------------------------------ | -------------------------------------------- |
| 0-1 hr   | Consensus generation           | "Generating consensus cache..."              |
| 1-5 hrs  | YOLOv8 Optuna (100 trials)     | "Trial 50/100..."                            |
| 5-9 hrs  | PyTorch Optuna                 | "pytorch_pose Trial 50/100..."               |
| 9-13 hrs | MMPose Optuna                  | "mmpose Trial 50/100..."                     |
| 13-15    | MediaPipe/BlazePose Optuna     | Fast (minimal optimization)                  |
| 15-16    | COCO Validation (all models)   | "COCO Validation: 100 images..."             |
| 16-21    | Comparison (715 clips × 5)     | "Processing maneuver 500/715..."             |
| ~21 hrs  | Complete                       | "📊 Evaluation Complete!"                    |

---

## Emergency Stop

If you need to abort the run:

```bash
# Attach to tmux session
tmux attach -t fullrun  # or smoke_test

# Ctrl+C to stop gracefully
# OR Ctrl+B, then :kill-session to force stop

# If unresponsive, kill process
pkill -9 python
```

**Note:** Partial results will be saved to MLflow (completed trials/phases only).

---

## Summary

1. **Run smoke test first** (15-20 min)
2. **Verify all success criteria** (especially PyTorch)
3. **If smoke test passes:** Run full comparison (21 hrs)
4. **Monitor periodically** (every 2-4 hrs)
5. **Collect results** in single MLflow run for report

**Confidence level after smoke test:** High—you've validated the entire pipeline with real data.

---

**Created:** October 27, 2025  
**For:** Full pose comparison on robuntu30 (3090)  
**Estimated smoke test time:** 15-20 minutes  
**Estimated full run time:** ~21 hours

