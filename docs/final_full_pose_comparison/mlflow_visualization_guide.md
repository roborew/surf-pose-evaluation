# MLflow Visualization Guide

## October 2025 Pose Estimation Evaluation

**Experiment:** final_full_pose_comparison | **Run Date:** Oct 25-26, 2025

This guide provides specific instructions for creating visualizations in MLflow to analyze and present the pose estimation comparison results, with emphasis on the **consensus-based optimization** approach.

---

## 1. Setup and Access

### Starting MLflow UI

```bash
# Navigate to project root
cd /Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation

# Start MLflow UI pointing to this run's data
mlflow ui --backend-store-uri file:///Users/robo/05_Repos/01_PROJECTS/CV-PROJ/surf-pose-evaluation/data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/20251025_202539_final_full_pose_comparison/mlruns

# Access in browser
# http://localhost:5000
```

### Experiment Structure

This run contains **three experiments:**

1. **Optuna Phase** - Hyperparameter optimization (200 maneuvers)
   - Focus: Consensus-based optimization trials for YOLOv8 and PyTorch
2. **COCO Validation** - Standard benchmark (100 images)
   - Focus: Generalization assessment
3. **Comparison Phase** - Comprehensive evaluation (715 maneuvers)
   - Focus: Real-world performance on surf footage

---

## 2. Recommended Visualizations by Experiment

### 2.1 Optuna Experiment Visualizations

#### Chart 1: Detection Rate Comparison (Bar Chart)

**Purpose:** Immediately identify MediaPipe/BlazePose failures

**Configuration:**

- Chart Type: Bar Chart (Horizontal)
- Metric: `pose_detection_rate_mean`
- X-axis: Detection rate (0.0 to 1.0)
- Y-axis: Model names
- Color: Gradient (red < 0.5 < green)

**Steps:**

1. Select Optuna experiment
2. Click "Charts" → "Create Chart"
3. Select all 5 model runs
4. Choose metric: `pose_detection_rate_mean`
5. Set chart type: Bar Chart
6. Sort: Descending

**Expected Result:**

- YOLOv8-Pose, MMPose: 1.0 (green, 100%)
- PyTorch Pose: 0.733 (yellow, 73.3%)
- MediaPipe: 0.046 (red, 4.6%)
- BlazePose: 0.013 (red, 1.3%)

**Insight:** Clear visualization of detection failures

#### Chart 2: Consensus Optimization Convergence (Line Chart)

**Purpose:** Show how YOLOv8 and PyTorch optimized via consensus PCK

**Configuration:**

- Chart Type: Line Chart
- X-axis: Trial number (0-11)
- Y-axis: Best score (consensus PCK@0.2)
- Lines: Separate for YOLOv8 (10 trials) and PyTorch (11 trials)

**Steps:**

1. Filter runs by model: yolov8_pose AND pytorch_pose
2. Filter by: `best_trial_number` exists
3. Plot: `best_score` over trial number
4. Annotate best trials (YOLOv8 trial 10, PyTorch trial 11)

**Expected Pattern:**

- YOLOv8: Convergence to 0.718 at trial 10
- PyTorch: Convergence to 0.781 at trial 11
- Shows optimization process working

**Key Insight:** Validates consensus-based optimization approach

#### Chart 3: Detection F1 vs FPS (Scatter Plot)

**Purpose:** Accuracy-speed trade-off visualization

**Configuration:**

- Chart Type: Scatter Plot
- X-axis: `perf_fps_mean`
- Y-axis: `pose_detection_f1_mean`
- Point size: `perf_model_size_mb_mean`
- Labels: Model names

**Steps:**

1. Select all 5 runs from Optuna phase
2. Create scatter: FPS (x) vs F1 (y)
3. Size by model size
4. Label each point

**Expected Clusters:**

- Top-right (ideal): YOLOv8 (97.6% F1, 162 FPS)
- Top-left: MMPose (94.7% F1, 44 FPS)
- Middle: PyTorch (80.1% F1, 41 FPS)
- Bottom-right (fast, inaccurate): MediaPipe/BlazePose (50% F1, 150-167 FPS)

**Insight:** YOLOv8 dominates Pareto frontier

#### Chart 4: Processing Time Distribution (Box Plot)

**Purpose:** Show inference time variability

**Configuration:**

- Chart Type: Box Plot
- Metric: `perf_avg_inference_time_mean` and `_std`
- Y-axis: Model names
- X-axis: Time in milliseconds

**Expected Values:**

- MediaPipe: 6.73 ± 1.08 ms (consistent, fast)
- BlazePose: 6.00 ± 0.44 ms (fastest, most consistent)
- YOLOv8: 6.16 ± 0.20 ms (fast, very consistent)
- PyTorch: 24.11 ± 0.47 ms (slower)
- MMPose: 22.98 ± 1.12 ms (slower, more variable)

**Insight:** YOLOv8 combines speed with consistency

#### Chart 5: Memory vs CPU Utilization (Bubble Chart)

**Purpose:** Resource usage comparison

**Configuration:**

- Chart Type: Scatter (bubble)
- X-axis: `perf_avg_memory_usage_mean` (MB)
- Y-axis: `perf_avg_cpu_utilization_mean` (%)
- Bubble size: `perf_fps_mean`
- Color: By model

**Expected Pattern:**

- Small memory, low CPU: MediaPipe/BlazePose (32 MB, 11%)
- Large memory, medium CPU: YOLOv8 (902 MB, 11%)
- Medium memory, high CPU: MMPose (283 MB, 22%)

**Insight:** YOLOv8 uses more RAM but low CPU; MMPose CPU-intensive

### 2.2 COCO Validation Visualizations

#### Chart 6: PCK Score Progression (Grouped Bar Chart)

**Purpose:** Compare accuracy across PCK thresholds

**Configuration:**

- Chart Type: Grouped Bar Chart
- X-axis: Model names
- Y-axis: PCK score (0.0 to 1.0)
- Groups: PCK@0.1, @0.2, @0.3, @0.5
- Colors: Different per threshold

**Metrics:**

- `coco_pck_0.1`
- `coco_pck_0.2`
- `coco_pck_0.3`
- `coco_pck_0.5`

**Expected Pattern:**

All models improve from @0.1 to @0.5:

- MMPose: 75.5% → 96.8% (best)
- YOLOv8: 70.5% → 95.2%
- PyTorch: 69.8% → 95.1%
- MediaPipe/BlazePose: ~40% → ~85%

**Insight:** Research models (MMPose, YOLOv8, PyTorch) significantly better than lightweight

#### Chart 7: COCO Detection Metrics (Radar Chart)

**Purpose:** Multi-dimensional detection assessment

**Axes (0-1 scale):**

1. Precision (`coco_detection_precision`)
2. Recall (`coco_detection_recall`)
3. F1 Score (`coco_detection_f1`)
4. PCK@0.2 (`coco_pck_0.2`)
5. Processing Speed (normalized `coco_fps_mean` / max_fps)

**Expected Shapes:**

- PyTorch: Balanced pentagon (best F1)
- YOLOv8: Strong recall, good PCK, moderate precision
- MMPose: Strong recall, excellent PCK, weak precision
- MediaPipe/BlazePose: Extreme precision, very weak recall

**Create Externally:** Export metrics and use Python/R:

```python
import plotly.graph_objects as go

models = ['MediaPipe', 'BlazePose', 'YOLOv8', 'PyTorch', 'MMPose']
categories = ['Precision', 'Recall', 'F1', 'PCK@0.2', 'Speed (norm)']

# Data from run_summary.json
data = {
    'YOLOv8': [0.587, 1.000, 0.740, 0.874, 0.621],
    'PyTorch': [0.968, 0.671, 0.793, 0.865, 0.323],
    'MMPose': [0.559, 0.996, 0.716, 0.898, 0.219],
    'MediaPipe': [1.000, 0.196, 0.327, 0.576, 0.904],
    'BlazePose': [1.000, 0.138, 0.242, 0.594, 1.000]
}

fig = go.Figure()
for model in models:
    fig.add_trace(go.Scatterpolar(
        r=data[model],
        theta=categories,
        name=model
    ))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
fig.show()
```

#### Chart 8: Precision-Recall Scatter

**Purpose:** Visualize detection trade-offs

**Configuration:**

- Chart Type: Scatter Plot
- X-axis: `coco_detection_recall`
- Y-axis: `coco_detection_precision`
- Point size: `coco_detection_f1` (larger = better)
- Labels: Model names
- Diagonal line: F1 contours

**Expected Positions:**

- Top-left (high precision, low recall): MediaPipe (1.0, 0.196), BlazePose (1.0, 0.138)
- Top-middle (best F1): PyTorch (0.968, 0.671) ← Largest point
- Middle-right (high recall): YOLOv8 (0.587, 1.0), MMPose (0.559, 0.996)

**Insight:** PyTorch best balance; YOLOv8/MMPose favor recall

#### Chart 9: COCO Processing Speed (Box Plot)

**Purpose:** Show speed distribution on COCO

**Configuration:**

- Chart Type: Box Plot with whiskers
- Y-axis: Model names
- X-axis: Inference time (ms)
- Show: mean, std, min, max

**Metrics:**

- `coco_inference_time_ms` (mean)
- `coco_inference_time_std_ms` (std)
- `coco_min_inference_time_ms`
- `coco_max_inference_time_ms`

**Expected Ranges:**

- BlazePose: 7.1 - 29.1 ms (mean 14.0)
- MediaPipe: 7.7 - 24.1 ms (mean 15.4)
- YOLOv8: 12.2 - 96.3 ms (mean 22.5, high variance)
- PyTorch: 21.9 - 113.3 ms (mean 43.3, high variance)
- MMPose: 32.6 - 148.7 ms (mean 63.8, moderate variance)

**Insight:** Lightweight models more consistent; research models vary with image complexity

### 2.3 Comparison Phase Visualizations

#### Chart 10: Consensus Accuracy Comparison (Horizontal Bar)

**Purpose:** **PRIMARY METRIC - Show consensus-based optimization results**

**Configuration:**

- Chart Type: Horizontal Bar Chart
- Metric: `pose_consensus_pck_0.2_mean`
- X-axis: Consensus PCK@0.2 (0.0 to 1.0)
- Y-axis: Model names
- Color: Green gradient (darker = better)
- Annotations: Percentage labels

**Steps:**

1. Select Comparison experiment
2. All 5 model runs
3. Sort descending by consensus PCK
4. Highlight YOLOv8 (85.5%) as best

**Expected Values:**

- YOLOv8-Pose: **85.5%** (dark green) ← WINNER
- MMPose: 67.9% (light green)
- PyTorch: 51.4% (yellow/orange)
- MediaPipe: 38.4% (orange/red)
- BlazePose: 31.2% (red)

**Insight:** **YOLOv8 consensus optimization dramatically successful (+6.6% vs traditional)**

**Include Annotation:** "+6.6% vs traditional tuning" on YOLOv8 bar

#### Chart 11: Detection Performance Matrix (Table/Heatmap)

**Purpose:** Compare all detection metrics at once

**Configuration:**

- Chart Type: Table or Heatmap
- Rows: Models
- Columns: Detection metrics
- Color: Green (good) to red (bad)

**Columns:**

1. Detection Rate (`pose_detection_rate_mean`)
2. Detection F1 (`pose_detection_f1_mean`)
3. True Positives (`pose_true_positives_mean`)
4. False Negatives (`pose_false_negatives_mean`)
5. Consistency (`pose_detection_consistency_mean`)

**Expected Heatmap:**

| Model     | Det Rate | F1       | TP      | FN      | Consistency |
| --------- | -------- | -------- | ------- | ------- | ----------- |
| YOLOv8    | 1.00 🟢  | 0.985 🟢 | 76.3 🟢 | 0.0 🟢  | 0.969 🟢    |
| MMPose    | 1.00 🟢  | 0.933 🟢 | 76.3 🟢 | 0.0 🟢  | 0.867 🟡    |
| PyTorch   | 0.41 🔴  | 0.661 🟡 | 31.1 🟡 | 45.2 🔴 | 0.912 🟢    |
| MediaPipe | 0.03 🔴  | 0.499 🔴 | 2.3 🔴  | 74.0 🔴 | 0.965 🟢    |
| BlazePose | 0.01 🔴  | 0.499 🔴 | 0.8 🔴  | 75.5 🔴 | 0.988 🟢    |

**Insight:** Clear visual of detection failures

#### Chart 12: Pose Quality Metrics (Grouped Bar)

**Purpose:** Compare stability and completeness

**Configuration:**

- Chart Type: Grouped Bar Chart
- X-axis: Model names (only YOLOv8, PyTorch, MMPose - others insufficient detections)
- Y-axis: Metric value
- Groups: Stability (inverted), Jitter (inverted), Completeness

**Metrics (normalized 0-1):**

- Pose Stability (inverted): `1 - pose_pose_stability_mean_mean`
- Jitter (inverted): `1 - (pose_pose_jitter_mean_mean / max_jitter)`
- Skeleton Completeness: `pose_skeleton_completeness_mean_mean`

**Expected Values:**

- PyTorch: High stability (0.910), low jitter norm (0.80), complete skeleton (1.0)
- MMPose: Good stability (0.888), lowest jitter norm (0.02), complete skeleton (1.0)
- YOLOv8: Lower stability (0.415), best jitter (0.90), incomplete skeleton (0.425)

**Insight:** PyTorch/MMPose better quality but YOLOv8 better accuracy (trade-off)

#### Chart 13: Speed vs Memory Footprint (Bubble Chart)

**Purpose:** Resource efficiency comparison

**Configuration:**

- Chart Type: Scatter (bubble)
- X-axis: FPS (`perf_fps_mean`)
- Y-axis: Memory MB (`perf_max_memory_usage_mean`)
- Bubble size: Consensus PCK@0.2 (larger = more accurate)
- Color: By model
- Quadrant lines at 50 FPS, 200 MB

**Expected Positions:**

- Top-right (fast, small memory): MediaPipe/BlazePose (150-167 FPS, 32 MB) - small bubbles (low accuracy)
- Top-middle (fast, moderate memory): YOLOv8 (99 FPS, 206 MB) - **LARGE bubble (85.5%)**
- Bottom-left (slow, high memory): PyTorch (40 FPS, 272 MB), MMPose (33 FPS, 182 MB) - medium bubbles

**Insight:** YOLOv8 optimal trade-off (fast + accurate), lightweight models unusable despite speed

#### Chart 14: Detection Rate vs Consensus Accuracy (Scatter)

**Purpose:** **Show PyTorch over-tuning problem**

**Configuration:**

- Chart Type: Scatter Plot
- X-axis: `pose_detection_rate_mean`
- Y-axis: `pose_consensus_pck_0.2_mean`
- Point size: Detection F1
- Labels: Model names
- Highlight: PyTorch Pose with annotation

**Expected Pattern:**

- Ideal (top-right): YOLOv8 (100% detection, 85.5% consensus) ← BEST
- Good (top-right): MMPose (100% detection, 67.9% consensus)
- **PROBLEM** (middle): PyTorch (41.1% detection, 51.4% consensus) ← OVER-TUNED
- Poor (bottom-left): MediaPipe (3.4% detection, 38.4% consensus)
- Poor (bottom-left): BlazePose (0.9% detection, 31.2% consensus)

**Annotation on PyTorch:** "Over-conservative consensus tuning (conf=0.988)"

**Insight:** **Visualizes key finding - PyTorch optimization failure**

---

## 3. Custom Visualizations (Create Externally)

### Visualization 15: Comprehensive Comparison Radar Chart

**Purpose:** Multi-dimensional model assessment across all key metrics

**Create with Python/R using exported data**

```python
import plotly.graph_objects as go
import pandas as pd

# Normalized metrics (0-1, higher is better)
models = ['YOLOv8-Pose', 'MMPose', 'PyTorch Pose']
metrics = [
    'Detection F1',
    'Consensus PCK@0.2',
    'FPS (normalized)',
    'Efficiency (normalized)',
    'Stability (inverted)',
    'Skeleton Complete'
]

data = {
    'YOLOv8-Pose': [0.985, 0.855, 0.593, 0.869, 0.415, 0.425],
    'MMPose': [0.933, 0.679, 0.196, 0.080, 0.888, 1.000],
    'PyTorch Pose': [0.661, 0.514, 0.239, 0.111, 0.910, 1.000]
}

fig = go.Figure()
for model in models:
    fig.add_trace(go.Scatterpolar(
        r=data[model],
        theta=metrics,
        fill='toself',
        name=model
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Comprehensive Model Comparison (Normalized Metrics)"
)
fig.write_html("comprehensive_radar.html")
```

**Insight:** YOLOv8 largest area (best overall); MMPose strongest in completeness; PyTorch best stability but poor detection

### Visualization 16: Optimization Journey Timeline

**Purpose:** Show the 20.98-hour experiment timeline

**Create with Gantt chart:**

```python
import plotly.figure_factory as ff
import pandas as pd

tasks = [
    dict(Task="MediaPipe Optuna", Start='2025-10-25 20:25', Finish='2025-10-25 21:12', Resource='Optuna'),
    dict(Task="BlazePose Optuna", Start='2025-10-25 21:12', Finish='2025-10-25 22:02', Resource='Optuna'),
    dict(Task="YOLOv8 Optuna", Start='2025-10-25 22:02', Finish='2025-10-25 22:32', Resource='Optuna'),
    dict(Task="PyTorch Optuna", Start='2025-10-25 22:32', Finish='2025-10-26 00:16', Resource='Optuna'),
    dict(Task="MMPose Optuna", Start='2025-10-26 00:16', Finish='2025-10-26 02:01', Resource='Optuna'),
    dict(Task="COCO Validation", Start='2025-10-26 02:01', Finish='2025-10-26 03:30', Resource='COCO'),
    dict(Task="Comparison Phase", Start='2025-10-26 03:30', Finish='2025-10-26 16:23', Resource='Comparison')
]

fig = ff.create_gantt(tasks, index_col='Resource', show_colorbar=True, group_tasks=True)
fig.update_layout(title="Experiment Execution Timeline (20.98 hours)")
fig.write_html("experiment_timeline.html")
```

### Visualization 17: Decision Matrix

**Purpose:** Guide model selection based on requirements

**Create as table/flowchart:**

| Requirement                  | Recommended Model | Rationale                              |
| ---------------------------- | ----------------- | -------------------------------------- |
| Production deployment        | **YOLOv8-Pose**   | 85.5% accuracy, 99 FPS, 50.5 MB        |
| Research/max accuracy        | **YOLOv8-Pose**   | Best consensus PCK (consensus-tuned)   |
| Real-time video (>60 FPS)    | **YOLOv8-Pose**   | Only accurate model achieving high FPS |
| Complete skeletons required  | **MMPose**        | 100% completeness, 67.9% consensus     |
| COCO benchmark validation    | **MMPose**        | Best COCO PCK (89.8%)                  |
| Embedded/edge deployment     | **YOLOv8-Pose**   | 50.5 MB compact, efficient             |
| Offline analysis (no RT req) | **MMPose**        | Best generalization to standard poses  |
| Mobile/ultra-lightweight     | **Not viable**    | MediaPipe/BlazePose <5% detection rate |

---

## 4. Data Export for External Analysis

### Export MLflow Metrics to CSV

```bash
# Export all metrics from Comparison experiment
mlflow experiments csv -e comparison_experiment_id -o comparison_metrics.csv

# Or use Python API
```

```python
import mlflow
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("file:///path/to/mlruns")

# Get experiment
experiment = mlflow.get_experiment_by_name("Comparison")

# Get all runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Extract specific metrics
metrics_of_interest = [
    'metrics.pose_consensus_pck_0.2_mean',
    'metrics.pose_detection_f1_mean',
    'metrics.perf_fps_mean',
    'metrics.pose_pose_stability_mean_mean',
    'metrics.pose_skeleton_completeness_mean_mean'
]

df = runs[['tags.model_name'] + metrics_of_interest]
df.to_csv('comparison_key_metrics.csv', index=False)
```

### Query Specific Metrics

```python
# Get YOLOv8 consensus optimization trials
yolo_runs = mlflow.search_runs(
    experiment_ids=[optuna_exp_id],
    filter_string="tags.model_name = 'yolov8_pose'"
)

# Plot optimization convergence
import matplotlib.pyplot as plt
trials = yolo_runs.sort_values('params.trial_number')
plt.plot(trials['params.trial_number'], trials['metrics.best_score'])
plt.xlabel('Trial Number')
plt.ylabel('Consensus PCK@0.2')
plt.title('YOLOv8 Consensus-Based Optimization Convergence')
plt.savefig('yolo_optimization_convergence.png')
```

---

## 5. Presentation-Ready Chart Recommendations

### For Technical Presentations (Conference/Paper)

**Essential Charts:**

1. **Consensus Accuracy Comparison (Chart 10)** - Primary result
2. **PCK Score Progression (Chart 6)** - COCO validation
3. **FPS vs Detection F1 (Chart 3)** - Trade-off analysis
4. **Comprehensive Radar (Chart 15)** - Multi-dimensional comparison
5. **Detection Rate vs Consensus (Chart 14)** - **Highlights PyTorch over-tuning**

**Supporting Charts:**

- Optimization convergence (Chart 2) - Shows consensus-based optimization
- Detection performance matrix (Chart 11) - Detailed breakdown
- Processing speed comparison (Chart 4/9) - Efficiency analysis

### For Executive Summary

**Essential Charts:**

1. **Consensus Accuracy Comparison (Chart 10)** - Bottom line
2. **Decision Matrix (Chart 17)** - Actionable recommendations
3. **Speed vs Memory Bubble (Chart 13)** - Resource trade-offs
4. **COCO Radar Chart (Chart 7)** - Validation credibility

### For Research Paper

**Essential Figures:**

1. **Figure 1:** Comprehensive radar chart (all metrics)
2. **Figure 2:** Consensus accuracy comparison + optimization convergence
3. **Figure 3:** COCO PCK progression (all thresholds)
4. **Figure 4:** Detection rate vs consensus accuracy (PyTorch failure analysis)
5. **Figure 5:** Precision-recall scatter (COCO validation)
6. **Table 1:** Complete metrics table (from CSV)
7. **Table 2:** Optimized hyperparameters

---

## 6. Tips and Best Practices

### Color Coding Recommendations

**Consistent Color Scheme:**

- 🟢 **Green:** Good performance (>80%)
- 🟡 **Yellow/Orange:** Moderate (50-80%)
- 🔴 **Red:** Poor performance (<50%)
- 🔵 **Blue:** YOLOv8-Pose (winner)
- 🟣 **Purple:** MMPose (runner-up)
- 🟠 **Orange:** PyTorch Pose (problematic)

### Annotation Guidelines

**Always Include:**

- Axis labels with units (%, FPS, ms, MB)
- Model names as labels
- Winner highlighting (bold, larger, or callout)
- Context annotations (e.g., "Real-time = 30 FPS")
- **Consensus optimization notes** for YOLOv8 and PyTorch

**Key Annotations:**

- YOLOv8: "+6.6% improvement with consensus tuning"
- PyTorch: "Over-conservative tuning (conf=0.988)"
- MediaPipe/BlazePose: "Detection failure: <5%"

### Common Pitfalls to Avoid

❌ **Don't:**

- Compare lightweight models (MediaPipe/BlazePose) as viable alternatives (they're not)
- Use linear scales for memory when values range 5-900 MB (use log scale)
- Show only speed without accuracy context
- Omit error bars or std dev where available
- Forget to normalize different-scale metrics (FPS vs PCK)

✅ **Do:**

- Emphasize YOLOv8's dominance (consensus optimization success)
- Highlight PyTorch's over-tuning failure (learning lesson)
- Show MMPose as COCO benchmark leader
- Include confidence intervals where available
- Use consistent color schemes across all charts
- Annotate consensus-based optimization impact

---

## 7. Quick Reference: Most Important Metrics

| Metric Name                            | What It Measures                       | Good Value | Phase      |
| -------------------------------------- | -------------------------------------- | ---------- | ---------- |
| `pose_consensus_pck_0.2_mean`          | **PRIMARY: Consensus accuracy**        | **>80%**   | Comparison |
| `pose_detection_f1_mean`               | Person detection reliability           | >95%       | All        |
| `coco_pck_0.2`                         | COCO keypoint accuracy                 | >85%       | COCO       |
| `perf_fps_mean`                        | Processing speed                       | >30 FPS    | All        |
| `perf_p95_inference_time_ms_mean`      | Latency (95th percentile)              | <50 ms     | Comparison |
| `pose_skeleton_completeness_mean_mean` | Skeleton completeness                  | >90%       | Comparison |
| `pose_pose_stability_mean_mean`        | Tracking stability (lower better)      | <0.2       | Comparison |
| `best_score`                           | **Optimization objective (consensus)** | Max        | Optuna     |

### Metric Relationships

- **Detection F1** must be high (>95%) for accuracy metrics to be meaningful
- **Consensus PCK** directly reflects optimization quality for YOLOv8/PyTorch
- **COCO PCK** validates generalization but may not predict surf performance
- **FPS** and **latency** are inversely related (faster = lower latency)
- **Skeleton completeness** and **accuracy** can trade off (YOLOv8 vs MMPose)

---

## 8. Chart Configuration Examples

### MLflow Built-in Charts

**Bar Chart Configuration:**

```json
{
  "chartType": "bar",
  "metrics": ["pose_consensus_pck_0.2_mean"],
  "orientation": "horizontal",
  "colorScale": "greens",
  "sortBy": "descending",
  "showValues": true
}
```

**Scatter Plot Configuration:**

```json
{
  "chartType": "scatter",
  "xMetric": "perf_fps_mean",
  "yMetric": "pose_consensus_pck_0.2_mean",
  "sizeMetric": "pose_detection_f1_mean",
  "colorBy": "tags.model_name",
  "showLabels": true
}
```

### Python Plotly Examples

**Interactive Comparison Table:**

```python
import plotly.graph_objects as go

models = ['MediaPipe', 'BlazePose', 'YOLOv8', 'PyTorch', 'MMPose']
metrics = ['Detection %', 'Consensus %', 'FPS', 'Size (MB)']

# From run_summary.json
values = [
    [3.4, 38.4, 153.9, 5.0],      # MediaPipe
    [0.9, 31.2, 167.2, 150.0],    # BlazePose
    [100.0, 85.5, 99.2, 50.5],    # YOLOv8 ← BEST
    [41.1, 51.4, 39.9, 160.0],    # PyTorch
    [100.0, 67.9, 32.7, 180.0]    # MMPose
]

fig = go.Figure(data=[go.Table(
    header=dict(values=['Model'] + metrics),
    cells=dict(values=[models] + list(map(list, zip(*values))))
)])
fig.write_html("comparison_table.html")
```

---

## 9. Troubleshooting

### Issue: Metrics Not Showing

**Cause:** Metric names may differ between runs

**Solution:**

```python
# List all available metrics for a run
run = mlflow.get_run(run_id)
print(run.data.metrics.keys())

# Search for metrics containing keyword
matching_metrics = [k for k in run.data.metrics.keys() if 'consensus' in k.lower()]
print(matching_metrics)
```

### Issue: Too Many Data Points

**Cause:** 715 maneuvers × 5 models = many data points

**Solution:** Use aggregated metrics (means) rather than per-maneuver values

```python
# Filter for aggregate metrics only
agg_metrics = [m for m in metrics if '_mean' in m or '_std' in m]
```

### Issue: Can't Compare Across Experiments

**Solution:** Use MLflow compare runs feature

```bash
mlflow runs compare --experiment-ids optuna_id,coco_id,comparison_id
```

---

## 10. Key Visualizations Summary

| Chart # | Title                            | Purpose                                   | Priority     |
| ------- | -------------------------------- | ----------------------------------------- | ------------ |
| 10      | Consensus Accuracy Comparison    | **Primary result - optimization success** | **CRITICAL** |
| 14      | Detection vs Consensus (PyTorch) | **Show over-tuning failure**              | **CRITICAL** |
| 2       | Optimization Convergence         | **Validate consensus approach**           | **HIGH**     |
| 3       | FPS vs Detection F1              | Speed-accuracy trade-off                  | HIGH         |
| 6       | COCO PCK Progression             | Standard benchmark validation             | HIGH         |
| 13      | Speed vs Memory Bubble           | Resource efficiency                       | MEDIUM       |
| 11      | Detection Performance Matrix     | Detailed breakdown                        | MEDIUM       |
| 7       | COCO Radar Chart                 | Multi-dimensional COCO                    | MEDIUM       |
| 15      | Comprehensive Radar              | Overall comparison                        | HIGH         |
| 1       | Detection Rate Bar Chart         | Identify failures                         | MEDIUM       |

**Focus on Charts 10, 14, and 2 to communicate the core consensus-based optimization story.**

---

## Resources

**MLflow Documentation:** https://mlflow.org/docs/latest/index.html  
**Plotly Documentation:** https://plotly.com/python/  
**Experiment Data:** `/path/to/20251025_202539_final_full_pose_comparison/mlruns`

**Contact:** See main report for experiment details and raw data access

---

_Guide Generated: October 27, 2025_  
_Experiment: final_full_pose_comparison (20251025_202539)_  
_Focus: Consensus-Based Optimization Visualization_
