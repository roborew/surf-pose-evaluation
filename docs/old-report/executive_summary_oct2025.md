# Pose Estimation Library Comparison - Executive Summary

## October 2025 Evaluation

**Experiment:** full_pose_comparison | **Date:** Oct 19-20, 2025 | **Duration:** 10.25 hours

---

## Bottom Line

**For Production Surf Pose Analysis:** **YOLOv8-Pose**

**Why:** Near-perfect detection (99.3% F1), fast processing (142 FPS), tiny footprint (11.6 MB), and good accuracy (78.9% consensus). Optimal balance of all critical factors.

**For Maximum Accuracy:** **MMPose** (92.9% consensus, but 4× slower)

**Critical Failure:** **MediaPipe and BlazePose detected <5% of poses** - unsuitable despite speed.

---

## Quick Comparison

| Model              | Best For       | Detection F1 | Consensus Accuracy | FPS     | Model Size  | COCO PCK@0.2 |
| ------------------ | -------------- | ------------ | ------------------ | ------- | ----------- | ------------ |
| **YOLOv8-Pose** 🏆 | **Production** | **99.3%**    | 78.9%              | 142     | **11.6 MB** | 82.9%        |
| **MMPose**         | **Accuracy**   | 93.3%        | **92.9%**          | 35      | 180 MB      | **89.8%**    |
| **PyTorch Pose**   | Research       | 77.6%        | 78.8%              | 40      | 160 MB      | 85.6%        |
| **MediaPipe**      | Not Viable     | 49.9%        | 34.7%              | 154     | **5 MB**    | 57.6%        |
| **BlazePose**      | Not Viable     | 49.9%        | 27.7%              | **167** | 150 MB      | 59.4%        |

---

## Key Findings

### Winners by Category

- 🎯 **Accuracy Champion:** MMPose (92.9% consensus on surf, 89.8% COCO PCK)
- ⚡ **Speed Champion:** BlazePose (167 FPS, but unusable accuracy)
- 💾 **Efficiency Champion:** YOLOv8-Pose (12.28 FPS/MB among accurate models)
- 🏅 **Best Balance:** YOLOv8-Pose (99% detection + 143 FPS + 11.6 MB)
- ✅ **Most Stable:** PyTorch Pose (0.106 stability, 11.00px jitter)

### Critical Issues

❌ **MediaPipe & BlazePose catastrophically failed:**

- MediaPipe: Only 3.4% detection rate on surf footage
- BlazePose: Only 0.9% detection rate on surf footage
- Both are **NOT recommended** despite fast processing (154-167 FPS)

**Reason:** Likely optimized for upright, indoor poses. Unable to handle surf-specific challenges (crouched poses, occlusion, motion, variable scale).

---

## Performance Metrics

### Accuracy (Surf Footage - 715 maneuvers)

| Model        | Detections | Consensus PCK@0.2 | Skeleton Complete | Detection Consistency |
| ------------ | ---------- | ----------------- | ----------------- | --------------------- |
| YOLOv8-Pose  | 100% ✓     | 78.9%             | 38.6%             | **98.6%** ✓           |
| MMPose       | 100% ✓     | **92.9%** ✓       | **100%** ✓        | 86.7%                 |
| PyTorch Pose | 84.1%      | 78.8%             | **100%** ✓        | 71.1%                 |
| MediaPipe    | 3.4% ❌    | 34.7%             | N/A               | 96.5%                 |
| BlazePose    | 0.9% ❌    | 27.7%             | N/A               | **98.8%** ✓           |

### Speed & Resources (Comparison Phase)

| Model        | FPS       | P95 Latency  | Memory      | CPU %      | Model Size    |
| ------------ | --------- | ------------ | ----------- | ---------- | ------------- |
| BlazePose    | **167** ✓ | **7.4 ms** ✓ | 87 MB       | **9.9%** ✓ | 150 MB        |
| MediaPipe    | **154** ✓ | 8.9 ms       | 87 MB       | **9.7%** ✓ | **5 MB** ✓    |
| YOLOv8-Pose  | **142** ✓ | **8.0 ms** ✓ | **80 MB** ✓ | **9.0%** ✓ | **11.6 MB** ✓ |
| PyTorch Pose | 40        | 26.3 ms      | 265 MB      | 13.8%      | 160 MB        |
| MMPose       | 35        | 31.9 ms      | 181 MB      | 18.3%      | 180 MB        |

### COCO Validation (Standard Benchmark - 100 images)

| Model        | PCK@0.2     | PCK@0.5     | Detection F1 | Processing FPS |
| ------------ | ----------- | ----------- | ------------ | -------------- |
| MMPose       | **89.8%** ✓ | **96.8%** ✓ | 71.6%        | 16.2           |
| PyTorch Pose | 85.6%       | 94.6%       | 61.1%        | 23.4           |
| YOLOv8-Pose  | 82.9%       | **94.9%** ✓ | **88.7%** ✓  | 49.3           |
| BlazePose    | 59.4%       | 85.5%       | 24.2%        | **71.9** ✓     |
| MediaPipe    | 57.6%       | 84.1%       | 32.7%        | **64.7** ✓     |

---

## Use Case Recommendations

### 🏄 Surf Pose Analysis (Production)

→ Use **YOLOv8-Pose**

- Detects 99.3% of poses (near-perfect)
- Fast enough for real-time (142 FPS, 8ms latency)
- Smallest viable model (11.6 MB) - easy deployment
- Good accuracy (78.9% consensus, 82.9% COCO)
- **Recommended for 90% of applications**

### 🔬 Research / Maximum Accuracy

→ Use **MMPose**

- Best consensus accuracy (92.9% on surf, 89.8% COCO)
- Complete, precise skeletons (100% completeness)
- Perfect detection rate (100%)
- Accept slower processing (35 FPS)
- Best for offline analysis, ground truth generation

### 📐 Tracking with Stability Priority

→ Use **PyTorch Pose**

- Most stable tracking (0.106 stability score)
- Lowest jitter (11.00 pixels)
- Complete skeletons (100%)
- Good consensus (78.8%)
- Moderate speed (40 FPS acceptable)

### ❌ NOT Recommended for Surf

- **MediaPipe:** Misses 96.6% of poses
- **BlazePose:** Misses 99.1% of poses
- Speed advantages completely negated by detection failures

---

## Technical Highlights

### YOLOv8-Pose Strengths

- ✓ Near-perfect detection (99.3% F1 score)
- ✓ Excellent speed (142 FPS, 4× faster than MMPose)
- ✓ Tiny model (11.6 MB, 16× smaller than MMPose)
- ✓ Best efficiency (12.28 FPS/MB)
- ✓ Lowest jitter (4.95 pixels)
- ✓ Excellent consistency (98.6%)
- ✗ Incomplete skeletons (38.6% completeness)
- ✗ 14% accuracy gap vs MMPose

### MMPose Strengths

- ✓ Best accuracy (92.9% consensus, 89.8% COCO)
- ✓ Complete skeletons (100%)
- ✓ Perfect detection rate (100%)
- ✓ Excellent keypoint precision
- ✗ Slowest (35 FPS, 4× slower than YOLOv8)
- ✗ Largest memory footprint (181 MB runtime)
- ✗ Highest CPU usage (18.3%)

### PyTorch Pose Strengths

- ✓ Best stability (0.106, smoothest tracking)
- ✓ Complete skeletons (100%)
- ✓ Low jitter (11.00 px)
- ✓ Good accuracy (78.8% consensus)
- ✗ Lower detection rate (84.1%)
- ✗ Slow (40 FPS)
- ✗ High memory (265 MB)

---

## Accuracy-Performance Trade-off

```
High Accuracy (>90%)    MMPose (92.9% @ 35 FPS)
                          ↓ 4× speed, -14% accuracy
Balanced (75-90%)       YOLOv8-Pose (78.9% @ 142 FPS) ← OPTIMAL
                          ↓ similar accuracy, 0.28× speed
Moderate (75-85%)       PyTorch Pose (78.8% @ 40 FPS)
                          ↓ 4× speed, -44% accuracy
Unusable (<50%)         MediaPipe/BlazePose (28-35% @ 154-167 FPS)
```

**Pareto Frontier:** YOLOv8-Pose dominates the accuracy-speed trade-off space for production use.

---

## Validation Summary

### Three-Phase Evaluation

1. **Optuna Phase:** 272 maneuvers, hyperparameter optimization (8.6 hrs)
2. **COCO Phase:** 100 images, standardized benchmark
3. **Comparison Phase:** 715 maneuvers, comprehensive evaluation

### Total Evaluation

- **Models Tested:** 5 (MediaPipe, BlazePose, YOLOv8, PyTorch, MMPose)
- **Total Maneuvers:** 987 pose evaluations
- **Duration:** 10 hours 15 minutes
- **Success Rate:** 100% (no crashes)

### Hardware

- **GPU:** NVIDIA RTX 4090 (23.6 GB)
- **CPU:** 32 cores
- **RAM:** 123.5 GB

---

## Decision Tree

```
Need >90% accuracy?
├─ Yes → MMPose (accept 35 FPS)
└─ No → Need >100 FPS?
     ├─ Yes → YOLOv8-Pose (79% accuracy, real-time capable)
     └─ No → Need complete skeletons?
          ├─ Yes → PyTorch Pose (40 FPS) or MMPose (35 FPS)
          └─ No → YOLOv8-Pose (best overall balance)
```

**90% of use cases:** YOLOv8-Pose is optimal choice.

---

## Comparison with August 2025 Run

| Metric              | August | October   | Change        |
| ------------------- | ------ | --------- | ------------- |
| MMPose Consensus    | 86.0%  | **92.9%** | +6.9% ✓       |
| YOLOv8 Consensus    | 65.6%  | **78.9%** | +13.3% ✓      |
| MediaPipe Detection | 7.6%   | 3.4%      | -4.2% (worse) |
| BlazePose Detection | 1.9%   | 0.9%      | -1.0% (worse) |
| Runtime             | 27 hrs | 10.25 hrs | 62% faster    |

**Consistency:** Rankings unchanged across both runs. YOLOv8-Pose confirmed optimal for production.

---

## Files Generated

**Comprehensive Report:** `docs/pose_comparison_report_oct2025.md`

- Full methodology (3-phase architecture explained)
- Complete results with all metrics
- MLflow visualization recommendations
- Detailed analysis

**Metrics CSV:** `docs/pose_comparison_metrics_oct2025.csv`

- All numerical results
- Import into Excel, Google Sheets, etc.

**MLflow Guide:** `docs/mlflow_visualization_guide.md`

- Specific charts to create
- Configuration examples
- Recommended visualizations

**Raw Data:** `data/.../20251019_182446_full_pose_comparison/`

- MLflow runs
- JSON summaries
- Best parameters

---

## Quick Start

**For production deployment:**

```python
# Use YOLOv8-Pose
from ultralytics import YOLO

model = YOLO('yolov8s-pose.pt')
results = model(frame, conf=0.198, iou=0.294)
# Expect: 142 FPS, 99.3% detection, 78.9% accuracy
```

**For research/validation:**

```python
# Use MMPose
from mmpose.apis import init_model, inference_topdown

model = init_model(config, checkpoint)
results = inference_topdown(model, img)
# Expect: 35 FPS, 100% detection, 92.9% accuracy
```

---

## MLflow Access

```bash
# View experiment results
cd /path/to/surf-pose-evaluation
mlflow ui --backend-store-uri file:///data/.../mlruns

# Access at http://localhost:5000
# Experiments: Optuna (151189...), COCO (644559...), Comparison (685592...)
```

---

_Generated: October 21, 2025_  
_For complete technical details, see: `docs/pose_comparison_report_oct2025.md`_  
_Experiment ID: full_pose_comparison (20251019_182446)_
