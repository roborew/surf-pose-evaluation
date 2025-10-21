# MLflow Visualization Guide

## Pose Estimation Experiment - October 2025

This guide provides step-by-step instructions for creating visualizations from the MLflow tracking data to illustrate the pose estimation comparison results.

---

## 1. Setup and Access

### 1.1 Start MLflow UI

```bash
# Navigate to project directory
cd /Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation

# Start MLflow server
mlflow ui --backend-store-uri file:///Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation/data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/20251019_182446_full_pose_comparison/mlruns

# Access in browser
# URL: http://localhost:5000
```

### 1.2 Experiment IDs

The October 2025 run contains three experiments:

| Experiment Name       | Experiment ID        | Phase                 | Runs     |
| --------------------- | -------------------- | --------------------- | -------- |
| Optuna Optimization   | `151189311028917651` | Hyperparameter tuning | 5 models |
| COCO Validation       | `644559105068258782` | Standard benchmark    | 5 models |
| Comparison Evaluation | `685592705955115462` | Main surf comparison  | 5 models |

---

## 2. Recommended Visualizations by Experiment

### 2.1 Optuna Experiment (151189311028917651)

#### Chart 1: Detection Rate Comparison

**Purpose:** Show catastrophic failure of MediaPipe/BlazePose

**Steps:**

1. Navigate to Optuna experiment
2. Select all 5 runs
3. Click "Chart" tab
4. Configure:
   - Chart Type: **Bar**
   - X-axis: `param.model_name`
   - Y-axis: `metric.pose_detection_rate_mean`
   - Color: `param.model_name`
5. Title: "Detection Rate by Model (Optuna Phase - 272 Maneuvers)"

**Expected Result:**

- YOLOv8/MMPose: ~1.0 (100%)
- PyTorch: ~0.92 (92%)
- MediaPipe: ~0.045 (4.5%)
- BlazePose: ~0.012 (1.2%)

#### Chart 2: FPS vs Detection F1 (Speed-Accuracy Trade-off)

**Purpose:** Visualize performance-accuracy balance

**Steps:**

1. Select all 5 runs in Optuna experiment
2. Chart Type: **Scatter**
3. Configure:
   - X-axis: `metric.perf_fps_mean`
   - Y-axis: `metric.pose_detection_f1_mean`
   - Color: `param.model_name`
   - Point size: constant (10)
4. Title: "Speed vs Accuracy Trade-off (Optuna Phase)"

**Expected Clustering:**

- Top-right (ideal): YOLOv8 (~229 FPS, 0.986 F1)
- Top-center: MMPose (~43 FPS, 0.932 F1)
- Middle-low: PyTorch (~41 FPS, 0.840 F1)
- Left edge (poor): MediaPipe/BlazePose (150-166 FPS, ~0.50 F1)

#### Chart 3: Processing Time Distribution

**Purpose:** Compare optimization time per model

**Steps:**

1. Use data from `dynamic_optimization_summary.json`
2. Create bar chart in external tool (Excel, matplotlib):
   - X-axis: Model names
   - Y-axis: Processing time (hours)
   - Data:
     - MediaPipe: 1.22 hrs
     - BlazePose: 1.34 hrs
     - YOLOv8: 0.87 hrs
     - PyTorch: 2.66 hrs
     - MMPose: 2.55 hrs

#### Chart 4: Memory and CPU Utilization

**Purpose:** Resource consumption comparison

**Steps:**

1. In MLflow, create grouped bar chart
2. Metrics to compare:
   - `metric.perf_max_memory_usage_mean`
   - `metric.perf_avg_cpu_utilization_mean`
3. Group by: `param.model_name`

**Note:** Memory values are 0.0 for MediaPipe/BlazePose (tracking limitation)

---

### 2.2 COCO Validation (644559105068258782)

#### Chart 5: PCK Score Comparison (Multiple Thresholds)

**Purpose:** Show accuracy progression across thresholds

**Steps:**

1. Navigate to COCO experiment
2. Select all 5 model runs
3. Chart Type: **Grouped Bar**
4. Metrics (create 4 series):
   - `metric.coco_pck_0.1`
   - `metric.coco_pck_0.2`
   - `metric.coco_pck_0.3`
   - `metric.coco_pck_0.5`
5. X-axis: Model names
6. Title: "COCO PCK Scores at Different Thresholds"

**Expected Pattern:**

- MMPose: Highest at all thresholds (75.5%, 89.8%, 93.5%, 96.8%)
- Increasing trend from PCK@0.1 to PCK@0.5 for all models
- Large gap between research models and lightweight models

#### Chart 6: Precision-Recall Trade-off

**Purpose:** Visualize detection characteristics

**Steps:**

1. Chart Type: **Scatter**
2. Configure:
   - X-axis: `metric.coco_detection_recall`
   - Y-axis: `metric.coco_detection_precision`
   - Point size: `metric.coco_detection_f1` (F1 score)
   - Color: `param.model_name`
3. Add diagonal line y=x (perfect balance)
4. Title: "COCO Detection: Precision vs Recall"

**Expected Pattern:**

- Top-left: MediaPipe/BlazePose (perfect precision, low recall)
- Top-right (ideal): YOLOv8 (balanced)
- Right side: MMPose/PyTorch (high recall, lower precision)

#### Chart 7: COCO Processing Speed

**Purpose:** Compare inference speed on standard images

**Steps:**

1. Chart Type: **Bar**
2. Metrics:
   - Y-axis: `metric.coco_fps_mean`
   - Error bars: `metric.coco_inference_time_std_ms`
3. X-axis: Model names
4. Title: "COCO Processing Speed (100 Images)"

**Expected Result:**

- BlazePose: ~72 FPS (fastest)
- MediaPipe: ~65 FPS
- YOLOv8: ~49 FPS
- PyTorch: ~23 FPS
- MMPose: ~16 FPS (slowest)

#### Chart 8: PCK Error Heatmap

**Purpose:** Visualize keypoint localization errors

**Steps:**

1. Export metrics to external tool
2. Create heatmap with:
   - Rows: Models
   - Columns: PCK thresholds (0.1, 0.2, 0.3, 0.5)
   - Values: 1 - PCK score (error rate)
   - Color scale: Red (high error) to Green (low error)

---

### 2.3 Comparison Experiment (685592705955115462)

#### Chart 9: Consensus Accuracy Comparison

**Purpose:** Primary accuracy metric for surf footage

**Steps:**

1. Navigate to Comparison experiment
2. Select all 5 runs
3. Chart Type: **Horizontal Bar**
4. Configure:
   - X-axis: `metric.pose_consensus_pck_0.2_mean`
   - Y-axis: Model names (sorted by accuracy)
   - Color: Green (>0.7), Yellow (0.3-0.7), Red (<0.3)
5. Title: "Consensus PCK@0.2 (715 Surf Maneuvers)"

**Expected Order (best to worst):**

1. MMPose: 92.9%
2. YOLOv8: 78.9%
3. PyTorch: 78.8%
4. MediaPipe: 34.7%
5. BlazePose: 27.7%

#### Chart 10: Detection Performance Metrics

**Purpose:** Comprehensive detection assessment

**Steps:**

1. Create grouped bar chart with 3 metrics:
   - `metric.pose_detection_f1_mean`
   - `metric.pose_detection_rate_mean`
   - `metric.pose_detection_consistency_mean`
2. X-axis: Models
3. Title: "Detection Performance Metrics (Surf Comparison)"

#### Chart 11: Stability and Jitter Comparison

**Purpose:** Show tracking quality

**Steps:**

1. Chart Type: **Line** or **Bar**
2. Metrics (dual Y-axis):
   - Left Y-axis: `metric.pose_pose_stability_mean_mean` (lower is better)
   - Right Y-axis: `metric.pose_pose_jitter_mean_mean` (px, lower is better)
3. X-axis: Models (exclude MediaPipe/BlazePose - insufficient data)
4. Title: "Pose Stability and Jitter (Surf Tracking)"

**Expected:**

- PyTorch: Best stability (0.106), moderate jitter (11.00px)
- MMPose: Good stability (0.112), high jitter (56.54px)
- YOLOv8: Higher instability (0.623), lowest jitter (4.95px)

#### Chart 12: Speed vs Memory Footprint

**Purpose:** Resource efficiency visualization

**Steps:**

1. Chart Type: **Bubble**
2. Configure:
   - X-axis: `metric.perf_fps_mean`
   - Y-axis: `metric.perf_max_memory_usage_mean`
   - Bubble size: Model size (from params)
   - Color: Model accuracy (consensus PCK@0.2)
3. Title: "Performance vs Resources (Comparison Phase)"

**Expected Quadrants:**

- Top-right (fast, light memory): YOLOv8, MediaPipe, BlazePose
- Top-left (slow, light memory): None
- Bottom-right (fast, heavy memory): None
- Bottom-left (slow, heavy memory): PyTorch, MMPose

**Ideal position:** High FPS, low memory - YOLOv8 wins among accurate models

---

## 3. Custom Visualizations (Create Externally)

### 3.1 Comprehensive Radar Chart

**Purpose:** Multi-dimensional model comparison

**Steps:**

1. Use matplotlib, Plotly, or Excel
2. Axes (normalized 0-1, higher is better):
   - Detection F1
   - Consensus PCK@0.2
   - FPS (normalized to 0-250)
   - Efficiency (FPS/MB)
   - Stability (inverted - 1-stability)
3. One polygon per model
4. Title: "Multi-Dimensional Model Comparison"

**Code Example (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Detection F1', 'Consensus\nPCK@0.2', 'FPS\n(norm)', 'Efficiency', 'Stability\n(inv)']
models = {
    'YOLOv8': [0.993, 0.789, 0.57, 1.0, 0.377],  # Normalized
    'MMPose': [0.933, 0.929, 0.14, 0.015, 0.888],
    'PyTorch': [0.776, 0.788, 0.16, 0.020, 0.894],
}

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

for model, values in models.items():
    values += values[:1]  # Complete the circle
    angles_plot = angles + angles[:1]
    ax.plot(angles_plot, values, 'o-', linewidth=2, label=model)
    ax.fill(angles_plot, values, alpha=0.15)

ax.set_xticks(angles)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.set_title('Multi-Dimensional Model Comparison', size=16, pad=20)
plt.tight_layout()
plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
```

### 3.2 Decision Matrix

**Purpose:** Guide model selection

**Create in Excel/Google Sheets:**

|                      | Accuracy Priority | Speed Priority | Balanced     | Embedded |
| -------------------- | ----------------- | -------------- | ------------ | -------- |
| **High Acc (>90%)**  | MMPose            | -              | -            | -        |
| **Med Acc (75-90%)** | YOLOv8            | YOLOv8         | **YOLOv8** ✓ | YOLOv8   |
| **Low Acc (<75%)**   | -                 | -              | PyTorch      | -        |
| **Any Acc**          | -                 | BlazePose\*    | -            | -        |

\*BlazePose fast but unusable accuracy

### 3.3 Optuna Convergence Plot

**Purpose:** Show optimization progression (if trial data available)

**Note:** Current run shows 0 trials (pre-determined configs used), but structure for future:

```python
import plotly.graph_objects as go

# If Optuna trial data available
fig = go.Figure()
for model in ['mediapipe', 'yolov8_pose', 'mmpose']:
    trials = optuna_study.trials_dataframe()
    fig.add_trace(go.Scatter(
        x=trials.number,
        y=trials.value,
        mode='lines+markers',
        name=model
    ))

fig.update_layout(
    title='Optuna Optimization Convergence',
    xaxis_title='Trial Number',
    yaxis_title='Objective Value (Detection F1)',
    hovermode='x unified'
)
```

---

## 4. Exporting Data from MLflow

### 4.1 Export Metrics to CSV

```bash
# Using MLflow CLI
mlflow experiments csv -x 685592705955115462 -o comparison_metrics.csv

# Or using Python API
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment("685592705955115462")
runs = client.search_runs(experiment.experiment_id)

import pandas as pd
data = []
for run in runs:
    row = {'model': run.data.params.get('model_name')}
    row.update(run.data.metrics)
    data.append(row)

df = pd.DataFrame(data)
df.to_csv('comparison_export.csv', index=False)
```

### 4.2 Query Specific Metrics

```python
from mlflow import MlflowClient

client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["685592705955115462"],
    filter_string="",
    order_by=["metrics.pose_detection_f1_mean DESC"]
)

for run in runs:
    model = run.data.params.get('model_name')
    f1 = run.data.metrics.get('pose_detection_f1_mean')
    fps = run.data.metrics.get('perf_fps_mean')
    print(f"{model}: F1={f1:.3f}, FPS={fps:.1f}")
```

---

## 5. Presentation-Ready Charts

### 5.1 For Technical Presentations

**Must-Have Charts:**

1. Detection Rate Comparison (shows MediaPipe/BlazePose failure)
2. COCO PCK Scores (standardized accuracy)
3. FPS vs Detection F1 Scatter (trade-off visualization)
4. Consensus PCK Horizontal Bar (surf accuracy)

### 5.2 For Executive Summary

**Essential Charts:**

1. Consensus PCK@0.2 Bar Chart (primary metric)
2. Speed vs Accuracy Scatter (with annotations)
3. Model Size vs Efficiency Bubble Chart

### 5.3 For Research Paper

**Comprehensive Set:**

1. All COCO metrics (PCK at all thresholds, detection metrics)
2. Stability and consistency plots
3. Resource utilization comparison
4. Failure mode analysis (MediaPipe/BlazePose)
5. Radar chart for multi-dimensional comparison

---

## 6. Tips and Best Practices

### 6.1 MLflow UI Tips

**Filtering:**

- Use filter: `param.model_name = "yolov8_pose"` to isolate models
- Sort by metrics: Click column headers in table view

**Comparing Runs:**

- Select 2+ runs with checkboxes
- Click "Compare" button
- View side-by-side metrics and parameters

**Downloading Artifacts:**

- Click run → "Artifacts" tab
- Download predictions, configs, etc.

### 6.2 Visualization Best Practices

**Color Coding:**

- Green: Good performance (YOLOv8, MMPose)
- Yellow: Moderate (PyTorch)
- Red: Poor (MediaPipe, BlazePose)

**Annotations:**

- Highlight "Production Recommended" (YOLOv8)
- Mark "Best Accuracy" (MMPose)
- Flag "Unsuitable" (MediaPipe/BlazePose)

**Scale Considerations:**

- Use log scale for FPS if range >10×
- Normalize metrics to 0-1 for radar charts
- Show error bars/confidence intervals where available

### 6.3 Common Pitfalls

❌ **Don't:**

- Compare different phases directly (Optuna vs Comparison datasets differ)
- Show raw metric names (use human-readable labels)
- Omit units (FPS, ms, MB, %)
- Use too many colors (limit to 5 models)

✅ **Do:**

- Include sample sizes in titles (e.g., "715 maneuvers")
- Show both absolute and relative performance
- Highlight recommended model clearly
- Provide context for metrics (what's "good")

---

## 7. Quick Reference: Key Metrics

### Most Important Metrics

| Metric Name (MLflow)          | Display Name           | Good Value | Chart Type   |
| ----------------------------- | ---------------------- | ---------- | ------------ |
| `pose_detection_f1_mean`      | Detection F1           | >0.95      | Bar, Scatter |
| `pose_consensus_pck_0.2_mean` | Consensus PCK@0.2      | >0.80      | Bar          |
| `perf_fps_mean`               | Processing Speed (FPS) | >100       | Bar, Scatter |
| `coco_pck_0.2`                | COCO PCK@0.2           | >0.80      | Grouped Bar  |
| `perf_max_memory_usage_mean`  | Memory (MB)            | <200       | Bubble, Bar  |
| `pose_pose_jitter_mean_mean`  | Jitter (pixels)        | <15        | Line, Bar    |

### Metric Relationships

```
Detection F1 = 2 × (Precision × Recall) / (Precision + Recall)
Efficiency = FPS / Model Size (MB)
Consensus PCK = Agreement with other models within threshold
```

---

## 8. Troubleshooting

### MLflow UI Not Starting

```bash
# Check if port 5000 in use
lsof -i :5000

# Use different port
mlflow ui --port 5001 --backend-store-uri file:///.../mlruns
```

### Missing Metrics

- Some metrics (e.g., stability) only available when sufficient detections
- MediaPipe/BlazePose missing stability data due to low detection rate
- Check `successful_maneuvers` count

### Export Errors

```python
# If CSV export fails, use programmatic approach
import mlflow
mlflow.set_tracking_uri("file:///.../mlruns")
# Then use MlflowClient as shown above
```

---

## 9. Next Steps

After creating visualizations:

1. **Include in Report:** Embed key charts in main comparison report
2. **Create Slide Deck:** Presentation with top 5-6 charts
3. **Archive:** Save high-resolution PNGs for publication
4. **Share:** Export interactive Plotly charts for web viewing

**Recommended Tool Stack:**

- **Interactive:** Plotly (HTML exports)
- **Static:** Matplotlib (publication quality)
- **Quick:** MLflow built-in charts
- **Presentation:** Export to PowerPoint/Keynote

---

_Guide created: October 21, 2025_  
_Experiment: full_pose_comparison (20251019_182446)_  
_MLflow URI: file:///.../20251019_182446_full_pose_comparison/mlruns_
