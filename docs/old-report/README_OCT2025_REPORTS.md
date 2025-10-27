# October 2025 Pose Estimation Reports - Index

This directory contains comprehensive documentation for the October 19-20, 2025 pose estimation library comparison experiment.

---

## 📊 Generated Reports

### 1. Main Comprehensive Report

**File:** [`pose_comparison_report_oct2025.md`](./pose_comparison_report_oct2025.md)  
**Length:** ~60 pages  
**Contents:**

- Complete methodology explanation (3-phase architecture)
- Library selection rationale
- Optuna optimization results
- COCO validation benchmarks
- Comprehensive surf comparison (715 maneuvers)
- Detailed model analysis with strengths/weaknesses
- Performance-accuracy trade-offs
- MLflow visualization recommendations
- Comparison with August 2025 experiment
- Decision frameworks and recommendations

**Audience:** Technical teams, researchers, developers

---

### 2. Executive Summary

**File:** [`executive_summary_oct2025.md`](./executive_summary_oct2025.md)  
**Length:** ~8 pages  
**Contents:**

- Bottom-line recommendation (YOLOv8-Pose for production)
- Quick comparison table
- Key findings summary
- Performance metrics overview
- Use case recommendations
- Decision tree
- Quick start code examples

**Audience:** Management, quick reference, presentations

---

### 3. Metrics CSV

**File:** [`pose_comparison_metrics_oct2025.csv`](./pose_comparison_metrics_oct2025.csv)  
**Rows:** 65 metrics × 5 models  
**Contents:**

- All numerical results in tabular format
- Organized by category (Accuracy, Performance, Resources, Stability)
- Separate sections for Optuna, COCO, and Comparison phases
- Winner column for each metric
- Import into Excel, Google Sheets, Pandas, etc.

**Audience:** Data analysis, charting, further processing

---

### 4. MLflow Visualization Guide

**File:** [`mlflow_visualization_guide.md`](./mlflow_visualization_guide.md)  
**Length:** ~25 pages  
**Contents:**

- MLflow UI setup instructions
- 12 recommended visualizations with step-by-step creation
- Chart configuration examples
- Custom visualization code (Python)
- Data export instructions
- Best practices and tips
- Troubleshooting guide

**Audience:** Anyone creating charts/visualizations from results

---

## 🎯 Quick Navigation

### Looking for...

**Overall recommendation?**  
→ Start with [`executive_summary_oct2025.md`](./executive_summary_oct2025.md)

**Technical details and methodology?**  
→ Read [`pose_comparison_report_oct2025.md`](./pose_comparison_report_oct2025.md)

**Raw numbers for analysis?**  
→ Import [`pose_comparison_metrics_oct2025.csv`](./pose_comparison_metrics_oct2025.csv)

**Want to create visualizations?**  
→ Follow [`mlflow_visualization_guide.md`](./mlflow_visualization_guide.md)

**Previous August 2025 results?**  
→ See [`pose_comparison_report.md`](./pose_comparison_report.md) (older)

---

## 📈 Key Findings at a Glance

### Winner: YOLOv8-Pose (Production)

- **Detection:** 99.3% F1 score
- **Speed:** 142 FPS
- **Accuracy:** 78.9% consensus, 82.9% COCO PCK@0.2
- **Size:** 11.6 MB

### Runner-up: MMPose (Research/Accuracy)

- **Detection:** 93.3% F1 score
- **Speed:** 35 FPS
- **Accuracy:** 92.9% consensus, 89.8% COCO PCK@0.2
- **Size:** 180 MB

### Critical Failure: MediaPipe & BlazePose

- Detected only 3.4% and 0.9% of poses respectively
- **Not suitable** for surf pose analysis despite fast processing

---

## 🔬 Experiment Overview

**Run Name:** full_pose_comparison  
**Date:** October 19-20, 2025  
**Duration:** 10 hours 15 minutes  
**Platform:** Linux (robuntu40) with NVIDIA RTX 4090

**Evaluation Structure:**

- **Phase 1:** Optuna optimization (272 maneuvers)
- **Phase 2:** COCO validation (100 images)
- **Phase 3:** Comprehensive comparison (715 maneuvers)

**Models Tested:**

1. MediaPipe (Google, 5 MB)
2. BlazePose (Google, 150 MB)
3. YOLOv8-Pose (Ultralytics, 11.6 MB) ✓ **Recommended**
4. PyTorch Pose (Torchvision, 160 MB)
5. MMPose (OpenMMLab, 180 MB) ✓ **Most Accurate**

---

## 📁 Data Availability

**Raw Experiment Data:**

```
/Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation/data/
  SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/
  20251019_182446_full_pose_comparison/
```

**Contents:**

- `run_summary.json` - Complete results (1540 lines)
- `production_evaluation_summary.json` - Summary metrics
- `best_params/best_parameters.yaml` - Optimized hyperparameters
- `mlruns/` - MLflow tracking data
- `predictions/` - Model predictions
- `visualizations/` - Generated visualizations
- `memory_profiling_report.json` - Resource usage

**MLflow Experiments:**

- Optuna: `151189311028917651`
- COCO: `644559105068258782`
- Comparison: `685592705955115462`

---

## 🚀 Quick Start

### View MLflow Results

```bash
cd /Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation
mlflow ui --backend-store-uri file:///Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation/data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/20251019_182446_full_pose_comparison/mlruns

# Access at http://localhost:5000
```

### Use Recommended Model (YOLOv8-Pose)

```python
from ultralytics import YOLO

model = YOLO('yolov8s-pose.pt')
results = model(frame, conf=0.198, iou=0.294)
# Expect: 142 FPS, 99.3% detection, 78.9% accuracy
```

### Analyze CSV Data

```python
import pandas as pd

df = pd.read_csv('docs/pose_comparison_metrics_oct2025.csv')
# Filter for specific phase
surf_accuracy = df[df['Phase'] == 'Comparison']
print(surf_accuracy[['Model', 'Metric', 'Winner']])
```

---

## 📝 Citation

If using these results in publications or presentations:

```
Surf Pose Estimation Library Comparison
October 2025 Comprehensive Evaluation
Experiment: full_pose_comparison (20251019_182446)
Platform: Linux, NVIDIA RTX 4090
Dataset: 715 surf maneuvers + 100 COCO validation images
Models: MediaPipe, BlazePose, YOLOv8-Pose, PyTorch Pose, MMPose
```

---

## 🔗 Related Files

**Configuration Files:**

- `configs/evaluation_config_production_comparison.yaml`
- `configs/evaluation_config_production_optuna.yaml`
- `configs/model_configs/*.yaml`

**Previous Experiments:**

- August 2025 report: [`pose_comparison_report.md`](./pose_comparison_report.md)
- August 2025 metrics: [`pose_comparison_metrics.csv`](./pose_comparison_metrics.csv)

---

## 📞 Support

For questions about the reports:

1. Check the **comprehensive report** for detailed methodology
2. Review the **MLflow guide** for visualization questions
3. Examine **raw data** in the experiment directory
4. Compare with **August 2025** results for consistency validation

---

_Index created: October 21, 2025_  
_Reports reflect experiment: 20251019_182446_full_pose_comparison_
