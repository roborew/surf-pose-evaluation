# Pose Estimation Library Comparison - Executive Summary

**Experiment:** mega_h264_complete | **Date:** Aug 5-6, 2025 | **Duration:** 27 hours

---

## 🎯 Bottom Line

**Recommended for Production:** **YOLOv8-Pose**  
**Why:** Best balance of accuracy (98.7% detection), speed (146 FPS), and efficiency (11.6 MB model)

---

## 📊 Quick Comparison

| Model              | Best For              | Detection F1 | FPS     | Model Size  | COCO PCK@0.2 |
| ------------------ | --------------------- | ------------ | ------- | ----------- | ------------ |
| **YOLOv8-Pose** 🏆 | **Production**        | **98.7%**    | 146     | **11.6 MB** | 82.9%        |
| **MMPose**         | **Accuracy**          | 93.6%        | 33      | 180 MB      | **89.8%**    |
| **PyTorch Pose**   | Research              | 80.5%        | 40      | 160 MB      | 85.6%        |
| **MediaPipe**      | Mobile (with caveats) | 50.7%        | 144     | **5 MB**    | 57.6%        |
| **BlazePose**      | Speed only            | 49.8%        | **163** | 150 MB      | 59.4%        |

---

## ✅ Key Findings

### Winners by Category

- 🎯 **Accuracy Champion:** MMPose (89.8% COCO PCK, 93.6% detection F1)
- ⚡ **Speed Champion:** BlazePose (163 FPS, but poor accuracy)
- 💾 **Efficiency Champion:** YOLOv8-Pose (12.59 throughput/MB)
- 🏅 **Best Balance:** YOLOv8-Pose (excellent across all metrics)

### Critical Issues

❌ **MediaPipe & BlazePose failed on surf footage:**

- MediaPipe: Only detected 7.6% of poses
- BlazePose: Only detected 1.9% of poses
- Both are **NOT recommended** for surf analysis despite being fast

---

## 📈 Performance Metrics

### Accuracy (Surf Footage - 1,051 maneuvers)

| Model        | Detections | Consensus Accuracy | Skeleton Complete |
| ------------ | ---------- | ------------------ | ----------------- |
| YOLOv8-Pose  | 100% ✓     | 65.6%              | 49.4%             |
| MMPose       | 100% ✓     | 86.0% ✓            | 100% ✓            |
| PyTorch Pose | 86.4%      | **100%** ✓         | 100% ✓            |
| MediaPipe    | 7.6% ❌    | 18.9%              | 100% ✓            |
| BlazePose    | 1.9% ❌    | 7.6%               | N/A               |

### Speed & Resources

| Model        | FPS       | Latency (P95) | Memory      | CPU %      |
| ------------ | --------- | ------------- | ----------- | ---------- |
| BlazePose    | **163** ✓ | 8.0 ms        | **32 MB** ✓ | **9.9%** ✓ |
| YOLOv8-Pose  | **146** ✓ | **7.8 ms** ✓  | 76 MB       | **9.1%** ✓ |
| MediaPipe    | **144** ✓ | 10.5 ms       | 69 MB       | 9.7%       |
| PyTorch Pose | 40        | 26.4 ms       | 263 MB      | 14.4%      |
| MMPose       | 33        | 32.9 ms       | 181 MB      | 18.1%      |

---

## 🎓 Recommendations

### Use Case Guide

**🏄 Surf Pose Analysis (Production)**
→ Use **YOLOv8-Pose**

- Detects 98.7% of poses
- Fast enough for real-time (146 FPS)
- Small model (11.6 MB) - easy deployment
- Good accuracy (82.9% COCO PCK)

**🔬 Research / Maximum Accuracy**
→ Use **MMPose**

- Best COCO accuracy (89.8%)
- Best surf consensus (86.0%)
- Complete, stable skeletons
- Accept slower processing (33 FPS)

**📐 Perfect Consensus Matching**
→ Use **PyTorch Pose**

- 100% consensus accuracy
- Complete skeletons
- Good stability (low jitter: 14.74px)
- Moderate speed (40 FPS)

**❌ NOT Recommended for Surf**

- MediaPipe: Misses 92% of poses
- BlazePose: Misses 98% of poses

---

## 💡 Technical Highlights

### YOLOv8-Pose Strengths

- ✓ Near-perfect detection (98.7% F1)
- ✓ Excellent speed (146 FPS)
- ✓ Tiny model (11.6 MB)
- ✓ Best detection consistency (97.3%)
- ✓ Lowest jitter (7.49 px)
- ✗ Incomplete skeletons (49.4%)

### MMPose Strengths

- ✓ Best COCO accuracy (89.8%)
- ✓ High detection F1 (93.6%)
- ✓ Complete skeletons (100%)
- ✓ Good stability
- ✗ Slowest (33.5 FPS)
- ✗ Largest model (180 MB)

---

## 📁 Full Results Available

**Report:** `docs/pose_comparison_report.md`  
**Metrics CSV:** `docs/pose_comparison_metrics.csv`  
**Raw Data:** `data/.../20250805_200156_mega_h264_complete/`  
**MLflow UI:** `http://localhost:5000`

---

## 🔢 Experiment Scale

- ✓ **5 models** evaluated
- ✓ **1,051 surf maneuvers** tested
- ✓ **100 COCO images** validated
- ✓ **27 hours** total runtime
- ✓ **100% success rate** (no crashes)
- ✓ **48,401 memory snapshots** collected

---

_Generated: October 14, 2025_  
_For complete analysis, see full report: `docs/pose_comparison_report.md`_
