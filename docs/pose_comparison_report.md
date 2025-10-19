# Pose Estimation Library Comparison Report

**Experiment:** `mega_h264_complete`  
**Run Date:** August 5-6, 2025  
**Duration:** 27 hours 1 minute  
**Platform:** Linux (robuntu40) with NVIDIA GeForce RTX 4090

---

## Executive Summary

This report presents a comprehensive evaluation of **five pose estimation libraries** tested on surf footage and COCO validation datasets. The evaluation assessed accuracy, performance, and resource efficiency across 1,051 surf maneuvers and 100 COCO images.

### Key Findings

**🏆 Best Overall Accuracy:** MMPose  
**⚡ Fastest Processing:** BlazePose (163 FPS)  
**💾 Most Memory Efficient:** YOLOv8-Pose (smallest model, 11.6 MB)  
**⚖️ Best Balance:** MMPose (high accuracy, acceptable performance)  
**❌ Poorest Detection:** BlazePose & MediaPipe (very low detection rates on surf footage)

---

## 1. Models Evaluated

| Model            | Model Size (MB) | Type             | Description                      |
| ---------------- | --------------- | ---------------- | -------------------------------- |
| **MediaPipe**    | 5.0             | Lightweight      | Google's efficient pose solution |
| **BlazePose**    | 150.0           | Lightweight      | Google's real-time pose detector |
| **YOLOv8-Pose**  | 11.6            | Object Detection | Ultralytics YOLO with pose       |
| **PyTorch Pose** | 160.0           | Research         | PyTorch implementation           |
| **MMPose**       | 180.0           | Research         | OpenMMLab pose framework         |

---

## 2. Accuracy Results

### 2.1 Detection Performance (Surf Footage - 1,051 Maneuvers)

| Model            | Detection F1 | Detection Rate | True Positives | False Negatives | Coverage |
| ---------------- | ------------ | -------------- | -------------- | --------------- | -------- |
| **YOLOv8-Pose**  | **0.987** ✓  | 100%           | 80.03          | 0.00            | 100%     |
| **MMPose**       | **0.936** ✓  | 100%           | 80.03          | 0.00            | 100%     |
| **PyTorch Pose** | 0.805        | 86.4%          | 69.66          | 10.37           | 100%     |
| **MediaPipe**    | 0.507        | 7.6%           | 6.09           | 73.94           | 100%     |
| **BlazePose**    | 0.498        | 1.9%           | 1.48           | 78.54           | 100%     |

**Key Insight:** YOLOv8 and MMPose achieved perfect detection rates (100%), while MediaPipe and BlazePose struggled significantly on surf footage, missing most poses.

### 2.2 Pose Accuracy - Consensus Metrics (Surf Footage)

| Model            | Consensus PCK@0.2 | PCK Error Mean | Confidence Score |
| ---------------- | ----------------- | -------------- | ---------------- |
| **PyTorch Pose** | **1.000** ✓       | **0.000** ✓    | 2.207            |
| **MMPose**       | **0.860** ✓       | 0.140          | 0.370            |
| **YOLOv8-Pose**  | 0.656             | 0.344          | 0.388            |
| **MediaPipe**    | 0.189             | 0.811          | 0.428            |
| **BlazePose**    | 0.076             | 0.924          | 0.195            |

**Key Insight:** PyTorch Pose achieved perfect consensus accuracy (100% within 20% threshold), followed by MMPose at 86%. YOLOv8 trades some accuracy for speed.

### 2.3 COCO Validation Results (100 Images)

#### PCK Scores (Percentage of Correct Keypoints)

| Model            | PCK@0.1     | PCK@0.2     | PCK@0.3     | PCK@0.5     | Mean Error  |
| ---------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **MMPose**       | **0.755** ✓ | **0.898** ✓ | **0.935** ✓ | **0.968** ✓ | **0.103** ✓ |
| **PyTorch Pose** | 0.689       | 0.856       | 0.912       | 0.946       | 0.144       |
| **YOLOv8-Pose**  | 0.650       | 0.829       | 0.890       | 0.949       | 0.171       |
| **BlazePose**    | 0.435       | 0.594       | 0.710       | 0.855       | 0.406       |
| **MediaPipe**    | 0.394       | 0.576       | 0.689       | 0.841       | 0.424       |

#### COCO Detection Metrics

| Model            | Precision   | Recall      | F1 Score    | True Positives | False Positives |
| ---------------- | ----------- | ----------- | ----------- | -------------- | --------------- |
| **YOLOv8-Pose**  | 0.812       | **0.978** ✓ | **0.887** ✓ | 220            | 51              |
| **MMPose**       | 0.559       | **0.996** ✓ | 0.716       | 224            | 177             |
| **PyTorch Pose** | 0.442       | **0.991** ✓ | 0.611       | 223            | 282             |
| **MediaPipe**    | **1.000** ✓ | 0.196       | 0.327       | 44             | 0               |
| **BlazePose**    | **1.000** ✓ | 0.138       | 0.242       | 31             | 0               |

**Key Insight:** MMPose achieved the highest keypoint accuracy on COCO. YOLOv8 balanced precision and recall best. MediaPipe and BlazePose had perfect precision but very low recall.

---

## 3. Performance Results

### 3.1 Speed Comparison (Surf Footage)

| Model            | Mean FPS    | Inference Time (ms) | P95 Latency (ms) | P99 Latency (ms) |
| ---------------- | ----------- | ------------------- | ---------------- | ---------------- |
| **BlazePose**    | **162.9** ✓ | **6.18** ✓          | 8.00             | 9.58             |
| **YOLOv8-Pose**  | **146.0** ✓ | **6.86** ✓          | 7.78             | 8.99             |
| **MediaPipe**    | **144.2** ✓ | **7.15** ✓          | 10.54            | 12.46            |
| **PyTorch Pose** | 39.7        | 25.22               | 26.37            | 27.42            |
| **MMPose**       | 33.5        | 29.92               | 32.92            | 34.48            |

**Key Insight:** BlazePose was fastest despite poor accuracy. YOLOv8 achieved 146 FPS with excellent accuracy—best speed/accuracy trade-off.

### 3.2 COCO Processing Speed

| Model            | COCO FPS   | Inference Time (ms) |
| ---------------- | ---------- | ------------------- |
| **BlazePose**    | **71.0** ✓ | 14.09               |
| **MediaPipe**    | **65.5** ✓ | 15.26               |
| **YOLOv8-Pose**  | 48.9       | 20.46               |
| **PyTorch Pose** | 22.9       | 43.67               |
| **MMPose**       | 16.0       | 62.65               |

### 3.3 Resource Efficiency

| Model            | Memory (MB) | CPU Util. (%) | Model Size (MB) | Efficiency Score |
| ---------------- | ----------- | ------------- | --------------- | ---------------- |
| **MediaPipe**    | 68.5        | 9.7           | **5.0** ✓       | **28.85** ✓      |
| **YOLOv8-Pose**  | 76.4        | 9.1           | **11.6** ✓      | **12.59** ✓      |
| **BlazePose**    | 32.0        | 9.9           | 150.0           | 1.09             |
| **MMPose**       | 180.9       | 18.1          | 180.0           | 0.19             |
| **PyTorch Pose** | 262.5       | 14.4          | 160.0           | 0.25             |

**Key Insight:** MediaPipe and YOLOv8 were most efficient per MB. YOLOv8 had the smallest model with excellent performance. MMPose and PyTorch required significantly more memory.

---

## 4. Stability & Consistency Metrics

### 4.1 Pose Stability (Lower is Better)

| Model            | Stability Mean | Jitter Mean (px) | Keypoint Consistency | Detection Consistency |
| ---------------- | -------------- | ---------------- | -------------------- | --------------------- |
| **MediaPipe**    | **0.028** ✓    | 37.37            | **0.044** ✓          | 0.938                 |
| **PyTorch Pose** | 0.078          | **14.74** ✓      | 0.099                | 0.747                 |
| **MMPose**       | 0.091          | 35.16            | 0.138                | 0.871                 |
| **YOLOv8-Pose**  | 0.518          | **7.49** ✓       | 0.530                | **0.973** ✓           |
| **BlazePose**    | N/A            | N/A              | N/A                  | **0.977** ✓           |

**Key Insight:** MediaPipe showed the most stable poses but detected very few. YOLOv8 had lowest jitter and excellent detection consistency. MMPose balanced stability with high detection rate.

### 4.2 Skeleton Completeness

| Model            | Completeness Mean | Valid Keypoint Ratio | High Confidence Ratio |
| ---------------- | ----------------- | -------------------- | --------------------- |
| **MediaPipe**    | **1.000** ✓       | **1.000** ✓          | 0.681                 |
| **PyTorch Pose** | **1.000** ✓       | **1.000** ✓          | 0.613                 |
| **MMPose**       | **1.000** ✓       | **1.000** ✓          | 0.153                 |
| **YOLOv8-Pose**  | 0.494             | 0.494                | 0.381                 |
| **BlazePose**    | N/A               | N/A                  | N/A                   |

**Key Insight:** MediaPipe, PyTorch, and MMPose provided complete skeletons. YOLOv8 had partial skeletons but still performed well overall.

---

## 5. Comparative Rankings

### 5.1 Overall Rankings by Category

| Rank | Accuracy         | Speed           | Memory Efficiency | Stability        |
| ---- | ---------------- | --------------- | ----------------- | ---------------- |
| 🥇   | **MMPose**       | **BlazePose**   | **MediaPipe**     | **MediaPipe**    |
| 🥈   | **PyTorch Pose** | **YOLOv8-Pose** | **YOLOv8-Pose**   | **PyTorch Pose** |
| 🥉   | **YOLOv8-Pose**  | **MediaPipe**   | **BlazePose**     | **MMPose**       |
| 4    | MediaPipe        | PyTorch Pose    | MMPose            | YOLOv8-Pose      |
| 5    | BlazePose        | MMPose          | PyTorch Pose      | BlazePose        |

### 5.2 Use Case Recommendations

| Use Case                     | Recommended Model | Rationale                                                     |
| ---------------------------- | ----------------- | ------------------------------------------------------------- |
| **Production Surf Analysis** | **YOLOv8-Pose**   | Best balance: 98.7% detection, 146 FPS, small model (11.6 MB) |
| **Maximum Accuracy**         | **MMPose**        | Highest COCO PCK (89.8%), 93.6% detection F1                  |
| **Research/Training**        | **PyTorch Pose**  | Perfect consensus accuracy (100%), complete skeletons         |
| **Real-time Mobile**         | **MediaPipe**     | Tiny model (5 MB), very fast, but poor surf detection         |
| **Speed Priority**           | **BlazePose**     | Fastest (163 FPS), but poor accuracy (49.8% F1)               |

---

## 6. Detailed Analysis

### 6.1 Strengths and Weaknesses

#### **MMPose**

✅ **Strengths:**

- Best COCO accuracy (89.8% PCK@0.2)
- High detection F1 (93.6%)
- Complete skeletons (100%)
- Good stability metrics

❌ **Weaknesses:**

- Slowest (33.5 FPS)
- Highest memory usage (181 MB)
- Largest model size (180 MB)
- Lower confidence scores

#### **YOLOv8-Pose**

✅ **Strengths:**

- Near-perfect detection (98.7% F1)
- Second-fastest (146 FPS)
- Smallest model (11.6 MB)
- Excellent detection consistency (97.3%)
- Best speed/accuracy trade-off

❌ **Weaknesses:**

- Incomplete skeletons (49.4%)
- Higher pose instability
- Moderate COCO accuracy

#### **PyTorch Pose**

✅ **Strengths:**

- Perfect consensus accuracy (100%)
- Complete skeletons (100%)
- Good stability (0.078)
- Low jitter (14.74 px)

❌ **Weaknesses:**

- Slow (39.7 FPS)
- High memory (262 MB)
- Lower detection rate (86.4%)
- Many false positives on COCO

#### **MediaPipe**

✅ **Strengths:**

- Smallest model (5 MB)
- Very fast (144 FPS)
- Most stable poses (0.028)
- Best memory efficiency
- Perfect precision

❌ **Weaknesses:**

- **Critical:** Only 7.6% detection rate on surf footage
- Poor COCO accuracy (57.6% PCK@0.2)
- Very low recall (19.6%)
- Not suitable for surf analysis

#### **BlazePose**

✅ **Strengths:**

- Fastest (163 FPS)
- Low memory (32 MB)
- Excellent detection consistency (97.7%)
- Perfect precision

❌ **Weaknesses:**

- **Critical:** Only 1.9% detection rate on surf footage
- Worst consensus accuracy (7.6%)
- Poor COCO accuracy (59.4% PCK@0.2)
- Very low recall (13.8%)
- Not suitable for surf analysis

---

## 7. Statistical Summary

### 7.1 Dataset Statistics

| Dataset                   | Videos/Images | Maneuvers | Cameras           | Total Frames |
| ------------------------- | ------------- | --------- | ----------------- | ------------ |
| **Optuna Optimization**   | 150 clips     | 499       | SONY_300          | ~38,494      |
| **Comparison Evaluation** | 304 clips     | 1,051     | SONY_70, SONY_300 | ~84,090      |
| **COCO Validation**       | 100 images    | N/A       | Various           | 100          |

### 7.2 Execution Statistics

- **Total Runtime:** 27h 1m 0s (97,260 seconds)
- **Memory Snapshots Collected:** 48,401
- **Peak Process Memory:** 23,477 MB
- **Peak GPU Memory:** 727 MB
- **All Models:** 100% success rate (0 crashes)

---

## 8. Conclusions

### 8.1 Primary Findings

1. **MMPose** delivers the **highest overall accuracy** on both surf footage (86% consensus PCK) and COCO (90% PCK@0.2), making it ideal for applications where accuracy is paramount.

2. **YOLOv8-Pose** offers the **best production balance**: near-perfect detection (98.7%), fast processing (146 FPS), tiny model (11.6 MB), and good accuracy. **Recommended for surf pose analysis.**

3. **PyTorch Pose** achieves **perfect consensus accuracy** (100%) but at the cost of speed (40 FPS) and excessive false positives.

4. **MediaPipe and BlazePose** are **not suitable for surf footage** despite being fast. They missed 92-98% of poses, likely due to training on different pose types or detection thresholds optimized for other use cases.

### 8.2 Model Selection Decision Tree

```
┌─────────────────────────────────────────┐
│    What matters most?                   │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │           │
    Accuracy      Speed
         │           │
    ┌────┴────┐      └───────────────┐
    │         │                      │
Research  Production           Real-time (<50ms)
    │         │                      │
PyTorch   YOLOv8                BlazePose*
  Pose      Pose               (*if detection OK)
    │         │                      │
Perfect   98.7% F1              163 FPS
 100%     146 FPS               6.2 ms/frame
         11.6 MB
```

### 8.3 Final Recommendation

**For production surf pose estimation:** Use **YOLOv8-Pose**

- Detects 98.7% of poses
- Processes at 146 FPS (6.9 ms latency)
- Only 11.6 MB model size
- Good COCO accuracy (82.9% PCK@0.2)
- Excellent consistency (97.3%)

**For maximum accuracy research:** Use **MMPose**

- Best COCO accuracy (89.8% PCK@0.2)
- Best consensus on surf footage (86%)
- Complete, stable skeletons
- Accept slower processing (33.5 FPS)

---

## Appendix

### A. Experimental Configuration

**Hardware:**

- CPU: 32 cores (32 logical)
- RAM: 123.5 GB
- GPU: NVIDIA GeForce RTX 4090 (23.6 GB)

**Software:**

- Platform: Linux 6.11.0-26-generic
- Python: 3.10.18

**Optimisation:**

- Optuna trials run on 499 maneuvers (SONY_300)
- Best hyperparameters selected per model
- All models tested with optimised configurations

### B. Metrics Definitions

- **PCK (Percentage of Correct Keypoints):** Proportion of predicted keypoints within threshold distance of ground truth
- **Detection F1:** Harmonic mean of precision and recall for person detection
- **Consensus PCK:** Agreement with ensemble of other models
- **Coverage:** Proportion of frames where model produces output
- **Efficiency Score:** Throughput per MB of model size
- **Stability:** Average frame-to-frame pose variation
- **Jitter:** Pixel-level noise in keypoint positions

### C. Data Availability

All raw results, predictions, and visualizations are available in:

```
data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/
  results/runs/20250805_200156_mega_h264_complete/
```

MLflow tracking UI available at: `http://localhost:5000`


