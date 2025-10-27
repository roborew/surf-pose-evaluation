# Pose Estimation Evaluation Reports - October 2025

## Navigation and Documentation Index

**Experiment:** final_full_pose_comparison  
**Run Date:** October 25-26, 2025  
**Key Innovation:** Consensus-Based PCK Optimization

---

## Quick Start

### For Decision Makers (5 minutes)

→ Read: [`executive_summary_oct2025.md`](executive_summary_oct2025.md)

**Bottom Line:** Use **YOLOv8-Pose (Large, consensus-tuned)** for surf pose applications. It achieves 85.5% consensus accuracy (+6.6% improvement), 98.5% detection F1, and 99 FPS. MediaPipe and BlazePose are not viable (<5% detection rate).

### For Technical Teams (30 minutes)

→ Read: [`pose_comparison_report_oct2025.md`](pose_comparison_report_oct2025.md)  
→ Reference: [`pose_comparison_metrics_oct2025.csv`](pose_comparison_metrics_oct2025.csv)

**Key Findings:**

- YOLOv8 consensus-based optimization: Major success (+6.6% accuracy)
- PyTorch consensus-based optimization: Over-conservative (41% detection rate)
- MMPose: Unexpected regression (67.9% vs 92.9% in previous run)
- MediaPipe/BlazePose: Catastrophic failures (3.4%, 0.9% detection rates)

### For Data Analysis (60 minutes)

→ Setup: [`mlflow_visualization_guide.md`](mlflow_visualization_guide.md)  
→ Data: `data/.../20251025_202539_final_full_pose_comparison/mlruns/`

**Focus Charts:**

- Chart 10: Consensus accuracy comparison (primary result)
- Chart 14: Detection vs consensus scatter (PyTorch failure)
- Chart 2: Optimization convergence (consensus approach validation)

---

## Available Reports

### 1. Comprehensive Technical Report

**File:** [`pose_comparison_report_oct2025.md`](pose_comparison_report_oct2025.md) (~60 pages)

**Contents:**

- Executive summary with key findings
- Detailed methodology (consensus-based optimization explained)
- Complete results across all three phases:
  - Optuna optimization (200 maneuvers)
  - COCO validation (100 images)
  - Comprehensive comparison (715 maneuvers)
- Model-specific analysis (strengths/weaknesses)
- Consensus optimization impact assessment
- MLflow visualization recommendations
- Conclusions and recommendations

**Best For:**

- Understanding the complete evaluation methodology
- Analyzing model performance in depth
- Accessing detailed metrics and statistical analysis
- Learning about consensus-based optimization approach
- Troubleshooting and implementation guidance

### 2. Executive Summary

**File:** [`executive_summary_oct2025.md`](executive_summary_oct2025.md) (~8 pages)

**Contents:**

- Bottom-line recommendation (YOLOv8-Pose Large)
- Quick comparison table (all models)
- Key findings by category (winners and failures)
- Performance metrics summary
- Use case recommendations
- Consensus optimization results analysis
- Decision tree for model selection
- Quick start code examples

**Best For:**

- Fast decision-making
- Project planning and budgeting
- Stakeholder presentations
- Model selection guidance

### 3. Metrics CSV

**File:** [`pose_comparison_metrics_oct2025.csv`](pose_comparison_metrics_oct2025.csv) (~70 rows)

**Contents:**

- All numerical results in tabular format
- Organized by category:
  - Accuracy (Optuna, COCO, Surf)
  - Performance (speed, latency, resources)
  - Stability (jitter, consistency, completeness)
  - Optimization (parameters, trials)
  - Historical comparison (vs Oct 19 run)
- Winner column for each metric
- Phase identifier for context

**Best For:**

- Importing into Excel, Google Sheets, R, Python
- Creating custom visualizations
- Statistical analysis
- Comparative studies
- Publication supplements

### 4. MLflow Visualization Guide

**File:** [`mlflow_visualization_guide.md`](mlflow_visualization_guide.md) (~25 pages)

**Contents:**

- MLflow setup and access instructions
- 14 recommended chart configurations with step-by-step instructions
- Custom visualization examples (radar charts, decision matrices)
- Python code for external analysis (Plotly, Matplotlib)
- Data export procedures
- Chart recommendations by audience (technical, executive, research)
- Troubleshooting guide
- Best practices for annotations and color coding

**Best For:**

- Creating presentation-ready visualizations
- Exploring MLflow experiment data
- Generating publication figures
- Interactive data analysis

---

## Key Findings at a Glance

### Winner: YOLOv8-Pose (Large, Consensus-Tuned) 🏆

| Metric            | Value     | Rank |
| ----------------- | --------- | ---- |
| Consensus PCK@0.2 | **85.5%** | 1st  |
| Detection F1      | **98.5%** | 1st  |
| FPS (Comparison)  | 99.2      | 3rd  |
| Model Size        | 50.5 MB   | 2nd  |
| COCO PCK@0.2      | 87.4%     | 2nd  |
| P95 Latency       | 11.3 ms   | 3rd  |

**Improvement vs Traditional Tuning:** +6.6% consensus accuracy

### Runner-up: MMPose

| Metric            | Value     | Rank       |
| ----------------- | --------- | ---------- |
| COCO PCK@0.2      | **89.8%** | 1st        |
| Skeleton Complete | **100%**  | 1st (tied) |
| Consensus PCK@0.2 | 67.9%     | 2nd        |
| Detection F1      | 93.3%     | 2nd        |
| FPS (Comparison)  | 32.7      | 5th        |

**Note:** Regression from 92.9% → 67.9% consensus vs previous run

### Not Recommended: PyTorch Pose (Current Tuning)

| Metric               | Value     | Issue               |
| -------------------- | --------- | ------------------- |
| Detection Rate       | **41.1%** | 58.9% poses missed! |
| Consensus PCK@0.2    | 51.4%     | Poor accuracy       |
| Confidence Threshold | 0.988     | Too restrictive     |

**Cause:** Consensus-based optimization with overly narrow parameter ranges

### Not Viable: MediaPipe & BlazePose

- MediaPipe: **3.4% detection rate** (96.6% poses missed)
- BlazePose: **0.9% detection rate** (99.1% poses missed)
- Speed advantages (154-167 FPS) completely negated by accuracy failures

---

## Experiment Overview

### Three-Phase Evaluation Architecture

#### Phase 1: Optuna Hyperparameter Optimization

- **Dataset:** 200 surf maneuvers (50 clips, SONY_300)
- **Duration:** 5.6 hours
- **Method:** **Consensus-based PCK optimization** (YOLOv8, PyTorch)
- **Objective:** Find optimal model configurations for surf poses
- **Key Innovation:** Using consensus with other models as optimization objective

**Results:**

- YOLOv8: 10 trials → best score 0.718 (trial 10) → selected 'large' variant
- PyTorch: 11 trials → best score 0.781 (trial 11) → over-conservative params
- MMPose/MediaPipe/BlazePose: Pre-determined configurations (0 trials)

#### Phase 2: COCO Validation

- **Dataset:** 100 images from COCO 2017 validation set
- **Purpose:** Standard benchmark comparison
- **Metrics:** PCK@0.1/0.2/0.3/0.5, detection precision/recall/F1

**Results:**

- MMPose: Best COCO accuracy (89.8% PCK@0.2)
- YOLOv8: Strong performance (87.4% PCK@0.2)
- PyTorch: Excellent F1 balance (79.3%)
- Lightweight models: Poor accuracy (<60% PCK@0.2)

#### Phase 3: Comprehensive Surf Comparison

- **Dataset:** 715 surf maneuvers (200 clips, SONY_300 + SONY_70)
- **Duration:** ~12 hours (main phase)
- **Metrics:** Consensus PCK, detection, performance, stability, completeness

**Results:**

- YOLOv8: **Dominant performance** (85.5% consensus, 98.5% detection)
- MMPose: Good but regressed (67.9% consensus)
- PyTorch: Detection failure (41.1% rate)
- MediaPipe/BlazePose: Unusable (<5% detection)

### Hardware Configuration

- **GPU:** NVIDIA GeForce RTX 4090 (23.6 GB VRAM)
- **CPU:** 32 cores, Linux 6.14.0
- **RAM:** 123.5 GB total, 102.2 GB available
- **Platform:** robuntu40 (Ubuntu-based Linux)
- **Python:** 3.10.18

### Total Experiment Statistics

- **Duration:** 20 hours 58 minutes
- **Models Tested:** 5
- **Total Evaluations:** 1,015 (200 Optuna + 100 COCO + 715 Comparison)
- **Total Frames Processed:** ~100,000+
- **Peak Memory:** 16.3 GB process memory
- **Mean CPU:** 34.7%
- **Success Rate:** 100% (no crashes)

---

## Data Availability

### Full Results Directory

```
/Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation/
  data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/
    POSE_EXPERIMENTS/results/runs/
      20251025_202539_final_full_pose_comparison/
```

### Key Files

**Summary Files:**

- `run_summary.json` (1540 lines) - Complete results across all phases
- `production_evaluation_summary.json` - Executive metrics snapshot
- `memory_profiling_report.json` - System resource analysis

**Configuration Files:**

- `best_params/best_parameters.yaml` - Optimized hyperparameters
- `optuna_config_20251025_202539.yaml` - Optimization phase config
- `coco_validation_config_20251025_202539.yaml` - COCO phase config
- `comparison_config_20251025_202539.yaml` - Comparison phase config

**MLflow Data:**

- `mlruns/` - Three experiment directories (Optuna, COCO, Comparison)
  - Run metadata, parameters, metrics
  - Model artifacts
  - Per-maneuver and aggregated results

**Data Selections:**

- `data_selections/optuna_selection.json` - 200 maneuvers for optimization
- `data_selections/comparison_selection.json` - 715 maneuvers for evaluation
- `data_splits/` - Train/val/test splits (211/45/46 clips)

**Optimization Reports:**

- `reports/dynamic_optimization_summary.json` - Per-model optimization timing

### MLflow Access

```bash
# Start MLflow UI
cd /Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation
mlflow ui --backend-store-uri file:///Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation/data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/20251025_202539_final_full_pose_comparison/mlruns

# Open browser to http://localhost:5000
```

---

## Quick Reference: Model Selection

### Decision Flow

```
┌─────────────────────────────────────────────────┐
│  Need pose estimation for surf footage?        │
└─────────────────┬───────────────────────────────┘
                  │
                  ├─> Use YOLOv8-Pose (Large) ✓
                  │   - 85.5% consensus accuracy
                  │   - 98.5% detection F1
                  │   - 99 FPS real-time
                  │   - 50.5 MB model size
                  │   - Consensus-optimized
                  │
                  └─> Special requirements?
                      │
                      ├─> COCO benchmark → MMPose
                      │   (89.8% COCO PCK@0.2)
                      │
                      ├─> 100% complete skeletons → MMPose
                      │   (67.9% consensus, 100% complete)
                      │
                      └─> All other cases → YOLOv8-Pose
```

### Use Case Matrix

| Requirement        | Model           | Consensus | FPS  | Size     | Completeness |
| ------------------ | --------------- | --------- | ---- | -------- | ------------ |
| Production         | **YOLOv8-Pose** | **85.5%** | 99   | 50.5 MB  | 42.5%        |
| Research           | **YOLOv8-Pose** | **85.5%** | 99   | 50.5 MB  | 42.5%        |
| Real-time video    | **YOLOv8-Pose** | **85.5%** | 99   | 50.5 MB  | 42.5%        |
| Complete skeletons | MMPose          | 67.9%     | 33   | 180 MB   | **100%**     |
| COCO validation    | MMPose          | 67.9%     | 33   | 180 MB   | **100%**     |
| Embedded/edge      | **YOLOv8-Pose** | **85.5%** | 99   | 50.5 MB  | 42.5%        |
| Offline analysis   | MMPose          | 67.9%     | 33   | 180 MB   | **100%**     |
| Mobile/lightweight | **Not viable**  | <40%      | 154+ | 5-150 MB | N/A          |

---

## Implementation Examples

### YOLOv8-Pose (Recommended)

```python
from ultralytics import YOLO

# Load YOLOv8 Large model (consensus-tuned)
model = YOLO('yolov8l-pose.pt')

# Use optimized parameters from Optuna
results = model(
    frame,
    conf=0.056,       # Consensus-optimized confidence threshold
    iou=0.529,        # Consensus-optimized IoU threshold
    half=True,        # FP16 for speed
    max_det=600,      # Max detections per image
    device='cuda:0'   # GPU acceleration
)

# Expected performance
# - 99 FPS (NVIDIA RTX 4090)
# - 98.5% detection F1
# - 85.5% consensus accuracy
# - 11.3ms P95 latency
```

### MMPose (COCO Benchmark)

```python
from mmpose.apis import init_model, inference_topdown

# Load MMPose model
config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
checkpoint = 'checkpoints/hrnet_w48_coco_256x192.pth'
model = init_model(config, checkpoint, device='cuda:0')

# Use optimized parameters
model.cfg.model.test_cfg.detection_threshold = 0.406
model.cfg.model.test_cfg.pose_threshold = 0.953

# Run inference
results = inference_topdown(model, img)

# Expected performance
# - 33 FPS (NVIDIA RTX 4090)
# - 89.8% COCO PCK@0.2
# - 100% skeleton completeness
```

---

## Comparison with Previous Evaluation

### October 19, 2025 vs October 25, 2025

| Aspect                      | Oct 19             | Oct 25                       | Change                |
| --------------------------- | ------------------ | ---------------------------- | --------------------- |
| **Methodology**             | Detection-based    | **Consensus-based**          | Key innovation        |
| **YOLOv8 Model**            | 's' (11.6 MB)      | **'l' (50.5 MB)**            | Larger, more accurate |
| **YOLOv8 Consensus**        | 78.9%              | **85.5%**                    | **+6.6%** ✓           |
| **PyTorch Detection**       | 84.1%              | **41.1%**                    | **-43%** ❌           |
| **PyTorch Consensus**       | 78.8%              | **51.4%**                    | **-27%** ❌           |
| **MMPose Consensus**        | 92.9%              | **67.9%**                    | **-25%** ❌           |
| **Optuna Maneuvers**        | 272                | 200                          | Smaller dataset       |
| **Optuna Trials (YOLOv8)**  | 0 (pre-determined) | **10** (consensus-optimized) | Active optimization   |
| **Optuna Trials (PyTorch)** | 0 (pre-determined) | **11** (consensus-optimized) | Active optimization   |
| **Duration**                | 10.25 hours        | 20.98 hours                  | 2× longer             |

**Key Lessons:**

✅ **Consensus optimization successful for YOLOv8:** Large model selection and parameter tuning significantly improved accuracy

❌ **Consensus optimization failed for PyTorch:** Over-conservative parameters (conf=0.988) caused detection collapse

⚠️ **MMPose regression:** Requires investigation - likely dataset or ensemble interaction effects

**Recommendation:** Continue using consensus-based optimization but with:

- Broader parameter ranges (especially for confidence thresholds)
- Detection rate constraints (>80% minimum)
- Validation on held-out test set during optimization

---

## Consensus-Based Optimization Explained

### What Changed

**Traditional Approach (Oct 19):**

- Objective: Maximize detection rate and F1 score
- Risk: Optimizes for detecting any pose, not accurate poses

**Consensus Approach (Oct 25):**

- **Objective: Maximize agreement (PCK) with ensemble of other models**
- Benefit: Optimizes for accurate pose localization
- Trade-off: May reduce detection rate if accuracy prioritized

### Why It Worked for YOLOv8

1. **Broader search space:** Tested multiple model sizes ('n', 's', 'm', 'l', 'x')
2. **Selected larger model:** 'l' variant (50.5 MB) for better accuracy
3. **Lower confidence threshold:** 0.056 (aggressive detection)
4. **High max detections:** 600 (captures all candidates)
5. **Result:** 6.6% accuracy improvement while maintaining high detection (98.5%)

### Why It Failed for PyTorch

1. **Over-conservative confidence:** 0.988 (very restrictive)
2. **Low max detections:** 10 (may miss multi-person scenes)
3. **Narrow parameter ranges:** TPE converged to extreme values
4. **Result:** 43% detection drop, 27% accuracy drop

### Recommendations for Future

**For All Models:**

- Use consensus PCK as primary objective ✓
- Include detection rate as constraint (>80%)
- Set permissive parameter ranges:
  - Confidence: 0.2 - 0.9 (not 0.0 - 1.0)
  - Max detections: 20 - 300 (not 1 - 1000)
- Test on validation set during optimization
- Monitor both accuracy and detection throughout trials

---

## Citation

If using these results in publications or presentations, please cite:

```
Surf Pose Estimation Library Comparison with Consensus-Based Optimization
October 2025 Comprehensive Evaluation
Experiment: final_full_pose_comparison (20251025_202539)

Dataset:
- Optuna: 200 surf maneuvers (SONY_300)
- COCO: 100 validation images
- Comparison: 715 surf maneuvers (SONY_300 + SONY_70)

Models: MediaPipe, BlazePose, YOLOv8-Pose (Large), PyTorch Pose, MMPose
Platform: Linux, NVIDIA RTX 4090, 32 cores, 123.5 GB RAM
Duration: 20 hours 58 minutes

Key Innovation: Consensus-based PCK as optimization objective
Key Finding: YOLOv8-Pose Large achieves 85.5% consensus accuracy (+6.6% improvement)

Generated: October 27, 2025
```

---

## Contact and Support

**For questions about:**

- Methodology → See comprehensive report Section 1
- Results → See executive summary or metrics CSV
- Visualizations → See MLflow guide
- Implementation → See quick start examples above
- Raw data → See data availability section

**Files in this documentation:**

1. `README_OCT2025_REPORTS.md` - This file (navigation and overview)
2. `pose_comparison_report_oct2025.md` - Full technical report (~60 pages)
3. `executive_summary_oct2025.md` - Executive summary (~8 pages)
4. `pose_comparison_metrics_oct2025.csv` - All metrics in CSV format (~70 rows)
5. `mlflow_visualization_guide.md` - Visualization instructions (~25 pages)

---

## Quick Facts

- **Total Pages:** ~100+ across all documents
- **Total Metrics Tracked:** 65+
- **Experiment Duration:** 20.98 hours
- **Models Evaluated:** 5
- **Poses Analyzed:** 1,015 (across all phases)
- **Winner:** YOLOv8-Pose (Large, consensus-tuned)
- **Key Innovation:** Consensus-based PCK optimization
- **Success Story:** YOLOv8 (+6.6% accuracy)
- **Failure Story:** PyTorch (41% detection rate)
- **Not Viable:** MediaPipe, BlazePose (<5% detection)

---

_Documentation Generated: October 27, 2025_  
_Experiment ID: final_full_pose_comparison (20251025_202539)_  
_Platform: Linux (robuntu40) with NVIDIA RTX 4090_
