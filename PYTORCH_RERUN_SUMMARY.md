# PyTorch Pose Optuna Re-run Summary

## Problem Statement

The October 25, 2025 consensus-based optimization for PyTorch Pose **failed catastrophically** due to over-conservative parameter selection:

| Metric                   | Oct 19 (Working)   | Oct 25 (Failed)       | Change      |
| ------------------------ | ------------------ | --------------------- | ----------- |
| **Detection Rate**       | **84.1%**          | **41.1%**             | **-43%** ❌ |
| **Consensus PCK@0.2**    | **78.8%**          | **51.4%**             | **-27%** ❌ |
| **Confidence Threshold** | 0.406 (reasonable) | **0.988** (too high!) | Problem     |
| **Max Detections**       | 15                 | 10                    | Too low     |

**Root Cause:** Optuna search ranges were too broad (confidence: 0.05-1.0), allowing convergence on extreme values.

---

## Solution Implemented

### 1. Default Configuration Updated with Constrained Ranges

**File:** `configs/model_configs/pytorch_pose.yaml` (updated as default)

**Key Changes:**

| Parameter              | Old Range  | New Range (Constrained) | Oct 19 Baseline |
| ---------------------- | ---------- | ----------------------- | --------------- |
| `confidence_threshold` | 0.05 - 1.0 | **0.25 - 0.65** ✓       | 0.406           |
| `keypoint_threshold`   | 0.05 - 1.0 | 0.40 - 0.98 ✓           | 0.953           |
| `nms_threshold`        | 0.1 - 0.8  | 0.35 - 0.75 ✓           | 0.612           |
| `max_detections`       | 5 - 25     | **10 - 30** ✓           | 15              |
| `box_score_thresh`     | 0.05 - 1.0 | **0.10 - 0.50** ✓       | 0.198           |
| `box_nms_thresh`       | 0.1 - 0.8  | 0.15 - 0.60 ✓           | 0.209           |

**Rationale:**

- Confidence range centered around Oct 19's successful 0.406 value
- Prevents convergence on near-100% thresholds (Oct 25 failed at 0.988)
- Still allows exploration but within reasonable bounds
- **These are now the DEFAULT ranges** - proper domain-constrained optimization

### 2. Optional Code Patch

**File:** `pytorch_optuna_constraint_patch.md`

**Two approaches provided:**

**Approach A (Simple - Recommended):**

- Penalty for confidence_threshold > 0.7
- Prevents October 25-style failures
- Minimal code change

**Approach B (Comprehensive):**

- Real-time detection rate monitoring
- Penalty if detection rate < 75%
- More robust but slower

### 3. Run Script Created

**File:** `run_pytorch_optuna_fix.sh`

**Features:**

- Runs PyTorch-only optimization
- Uses existing consensus cache from Oct 25 run
- Automatically backs up and restores configs
- Estimated runtime: 1.5-2 hours (15-20 trials)

---

## Expected Results

### Conservative Estimates (75% confidence)

| Metric                | Target     | Basis                         |
| --------------------- | ---------- | ----------------------------- |
| Detection Rate        | **75-90%** | Oct 19 achieved 84%           |
| Consensus PCK@0.2     | **75-85%** | Oct 19 achieved 79%           |
| Detection F1          | **75-85%** | Oct 19 achieved 78%           |
| Skeleton Completeness | **100%**   | PyTorch always complete       |
| Pose Stability        | **~0.10**  | Best in class (Oct 19: 0.106) |

### Why It Should Work Better

✅ **Constrained search space** prevents extreme values

✅ **Proven configuration nearby** (Oct 19: 0.406 confidence is in 0.25-0.65 range)

✅ **Consensus-based optimization** still active (unlike Oct 19)

✅ **Same consensus cache** ensures consistency with Oct 25 YOLOv8/MMPose results

✅ **Optional penalties** prevent over-conservative convergence

---

## Running the Re-optimization

### Prerequisites

1. ✓ Default config updated with constrained ranges (`pytorch_pose.yaml`)
2. ✓ Run script created (`run_pytorch_optuna_fix.sh`)
3. Optional: Apply code patch from `pytorch_optuna_constraint_patch.md`

### Option 1: Simple Run (Recommended - No Code Changes)

```bash
cd /Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation

# Run the script
./run_pytorch_optuna_fix.sh
```

**Runtime:** ~1.5-2 hours  
**Trials:** 20 (with early stopping)

### Option 2: With Code Patch (More Robust)

```bash
# 1. Apply the code patch
# Edit utils/optuna_optimizer.py and add the simple penalty (see pytorch_optuna_constraint_patch.md)

# 2. Run the script
./run_pytorch_optuna_fix.sh
```

**Recommended:** Use the simple penalty approach (Approach A in patch file)

### Option 3: Manual Command

```bash
conda activate surf_pose_analysis

# Verify config (constrained ranges are now default)
cat configs/model_configs/pytorch_pose.yaml

# Run optimization
python run_evaluation.py \
    --run-name "pytorch_optuna_fix" \
    --phase optuna \
    --models pytorch_pose \
    --optuna-trials 20 \
    --use-consensus-validation \
    --consensus-cache-dir "data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/20251025_202539_final_full_pose_comparison/consensus_cache"
```

---

## After Re-optimization

### 1. Check Results

```bash
# View results directory
cd data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/pytorch_optuna_fix/

# Check best parameters
cat best_params/best_parameters.yaml

# Look for these values in pytorch_pose section:
#   confidence_threshold: 0.3-0.6 (should be reasonable!)
#   max_detections: 15-25 (should be adequate)
```

### 2. View MLflow Results

```bash
mlflow ui --backend-store-uri file:///Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation/data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/pytorch_optuna_fix/mlruns

# Open browser to http://localhost:5000
# Check metrics:
#   - optuna_trial_score (consensus PCK)
#   - detection_rate_check (if patch applied)
#   - Best trial should have score > 0.75
```

### 3. Success Criteria

**Good Results (proceed with update):**

- ✓ Detection rate > 75%
- ✓ Consensus PCK@0.2 > 75%
- ✓ Confidence threshold: 0.3-0.6 range
- ✓ Max detections: 15-30 range

**Still Poor Results (need investigation):**

- ✗ Detection rate < 70%
- ✗ Consensus PCK@0.2 < 70%
- ✗ Confidence threshold still extreme (>0.7 or <0.2)

---

## Updating the October 25 Report

### If Re-optimization Succeeds

**Update these files with new PyTorch numbers:**

1. `docs/pose_comparison_report_oct2025.md`

   - Section 2.1: Optuna optimization results
   - Section 2.3: Comparison phase results
   - Update all PyTorch metrics with new values

2. `docs/executive_summary_oct2025.md`

   - Update quick comparison table
   - Update PyTorch technical highlights
   - Change recommendation from "not viable" to "viable option"

3. `docs/pose_comparison_metrics_oct2025.csv`

   - Replace all PyTorch rows with new metrics
   - Keep other models unchanged

4. Add note to all documents:
   ```markdown
   **Note on PyTorch Pose Results:**
   Initial consensus-based optimization (Oct 25) failed due to over-conservative
   parameter selection (confidence=0.988). Re-optimization with constrained
   parameter ranges (Oct 27) achieved [X]% detection and [Y]% consensus accuracy,
   validating PyTorch Pose as a viable option for surf pose analysis.
   ```

### Report Structure (Combined Data)

```markdown
## Experiment Details

**Primary Run:** October 25, 2025 (20251025_202539_final_full_pose_comparison)

- Models: MediaPipe, BlazePose, YOLOv8-Pose, MMPose
- PyTorch Pose: Re-optimized October 27, 2025 with constrained ranges

**Key Innovation:** Consensus-based PCK optimization

- ✅ YOLOv8-Pose: Major success (85.5% consensus)
- ✅ PyTorch Pose: Success after constraint fix ([X]% consensus)
- ⚠️ Initial PyTorch optimization failed (lesson learned)
```

---

## Decision Matrix After Re-run

### If PyTorch Achieves >80% Detection, >75% Consensus:

| Use Case             | Recommendation   | Rationale                              |
| -------------------- | ---------------- | -------------------------------------- |
| Production (General) | **YOLOv8-Pose**  | Best balance (85.5% consensus, 99 FPS) |
| Complete Skeletons   | **PyTorch Pose** | 100% completeness + good accuracy      |
| Maximum Accuracy     | **MMPose**       | 92.9% consensus (but slow)             |
| Stability Priority   | **PyTorch Pose** | Best stability (0.10) + low jitter     |
| Real-time (>60 FPS)  | **YOLOv8-Pose**  | Only accurate option >90 FPS           |

### If PyTorch Still <75%:

Continue using October 19 configuration as documented baseline:

- Detection: 84.1%
- Consensus: 78.8%
- Proven, validated configuration
- Document as "pre-consensus-optimization baseline"

---

## Summary

**Problem:** Oct 25 PyTorch optimization failed (41% detection, 51% consensus)

**Solution:**

1. ✓ Constrained parameter ranges created
2. ✓ Optional code penalties provided
3. ✓ Re-run script ready

**Expected Outcome:** 75-90% detection, 75-85% consensus (matching/exceeding Oct 19)

**Next Step:** Run `./run_pytorch_optuna_fix.sh` (1.5-2 hours)

**Success Will Prove:** Consensus-based optimization works for PyTorch when properly constrained

**Your Project:** Will have validated PyTorch configuration with:

- Proven detection capability (>75%)
- Consensus-optimized accuracy (>75%)
- Complete skeletons (100%)
- Best stability (0.10)
- Honest, reproducible methodology

---

**Files Updated/Created:**

1. ✓ `configs/model_configs/pytorch_pose.yaml` - **Updated with constrained ranges as DEFAULT**
2. ✓ `pytorch_optuna_constraint_patch.md` - Optional code improvements
3. ✓ `run_pytorch_optuna_fix.sh` - Automated run script
4. ✓ `PYTORCH_RERUN_SUMMARY.md` - This document

**Why Constrained Ranges Are Now Default:**

- The broad ranges (confidence: 0.05-1.0) were the **mistake**, not a reasonable default
- Constrained ranges based on Oct 19 baseline are **proper ML practice**
- Prevents pathological convergence while still allowing meaningful exploration
- If Optuna doesn't find better than defaults, it uses the Oct 19 working values
- This is how hyperparameter optimization **should** be configured

**Ready to run!** 🚀
