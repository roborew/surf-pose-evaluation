# Consensus Implementation - Complete File Tree

## 📁 Files Created/Modified

```
surf-pose-evaluation/
│
├── 🆕 utils/
│   ├── quality_filter.py               ✅ NEW (240 LOC)
│   │   └── AdaptiveQualityFilter class
│   │       ├── calculate_quality_score()
│   │       ├── calculate_stability()
│   │       ├── calculate_completeness()
│   │       ├── get_threshold_for_trial()
│   │       └── filter_keypoints()
│   │
│   ├── consensus_generator.py          ✅ NEW (230 LOC)
│   │   └── ConsensusGenerator class
│   │       ├── load_model()
│   │       ├── run_inference_on_maneuver()
│   │       ├── aggregate_predictions()
│   │       └── generate_consensus_for_maneuver()
│   │
│   ├── consensus_manager.py            ✅ NEW (300 LOC)
│   │   └── ConsensusManager class
│   │       ├── get_consensus_models_for_target()
│   │       ├── generate_consensus_gt()
│   │       ├── _save_to_cache()
│   │       ├── _load_from_cache()
│   │       └── clear_cache()
│   │
│   └── optuna_optimizer.py             🔧 MODIFIED (+100 LOC)
│       └── OptunaPoseOptimizer class
│           ├── __init__()              ← Added consensus setup
│           └── optimize_model()
│               └── objective()         ← Added consensus-based PCK
│
├── 🆕 metrics/
│   └── pose_metrics.py                 🔧 EXTENDED (+140 LOC)
│       └── PoseMetrics class
│           ├── calculate_pck_with_consensus_gt()  ← KEY METHOD!
│           └── _estimate_torso_diameter()
│
├── 🆕 configs/
│   ├── consensus_config.yaml           ✅ NEW (80 LOC)
│   │   ├── consensus.strong_models
│   │   ├── consensus.weak_models
│   │   ├── consensus.quality_filter
│   │   ├── consensus.leave_one_out
│   │   └── validation settings
│   │
│   ├── evaluation_config_production_optuna.yaml  🔧 MODIFIED
│   │   ├── optuna_validation          ← NEW SECTION
│   │   │   ├── use_consensus: true
│   │   │   ├── num_clips: 75
│   │   │   └── cameras: [SONY_300]
│   │   ├── models_to_optimize          ← NEW SECTION
│   │   └── consensus_config            ← NEW REFERENCE
│   │
│   └── evaluation_config_production_comparison.yaml  🔧 MODIFIED
│       ├── comparison_validation       ← NEW SECTION
│       │   ├── use_consensus: true
│       │   ├── num_clips: 200
│       │   ├── cameras: [SONY_300, SONY_70]
│       │   └── exclude_optuna_clips: true
│       └── consensus_config            ← NEW REFERENCE
│
├── 🆕 tests/
│   └── test_consensus_integration.py   ✅ NEW (200 LOC)
│       ├── TestBaselineBehavior
│       ├── TestModelWrapperInterfaces
│       ├── TestManeuverStructure
│       ├── TestPCKCalculation
│       └── TestDataLeakagePrevention
│
└── 🆕 docs/ (New documentation)
    ├── CONSENSUS_QUICKSTART.md         ✅ NEW
    ├── CONSENSUS_IMPLEMENTATION_COMPLETE.md  ✅ NEW
    ├── CONSENSUS_IMPLEMENTATION_STATUS.md    🔧 UPDATED
    ├── IMPLEMENTATION_SUMMARY.md       ✅ NEW
    └── IMPLEMENTATION_FILE_TREE.md     ✅ NEW (this file)
```

---

## 📊 Statistics

### Code Changes

| Category         | Files  | Lines of Code |
| ---------------- | ------ | ------------- |
| New core modules | 3      | 770           |
| Extended modules | 2      | 240           |
| Configuration    | 3      | 80            |
| Tests            | 1      | 200           |
| Documentation    | 5      | 120           |
| **Total**        | **14** | **1,410**     |

### File Breakdown

| File                                  | Status   | LOC  | Purpose                       |
| ------------------------------------- | -------- | ---- | ----------------------------- |
| `utils/quality_filter.py`             | NEW      | 240  | Adaptive quality filtering    |
| `utils/consensus_generator.py`        | NEW      | 230  | Model inference & aggregation |
| `utils/consensus_manager.py`          | NEW      | 300  | Orchestration & caching       |
| `utils/optuna_optimizer.py`           | MODIFIED | +100 | Consensus integration         |
| `metrics/pose_metrics.py`             | EXTENDED | +140 | PCK with consensus GT         |
| `configs/consensus_config.yaml`       | NEW      | 80   | Consensus configuration       |
| `configs/*_optuna.yaml`               | MODIFIED | +30  | Optuna config updates         |
| `configs/*_comparison.yaml`           | MODIFIED | +30  | Comparison config updates     |
| `tests/test_consensus_integration.py` | NEW      | 200  | Safety tests                  |

---

## 🔑 Key Files Explained

### Core Infrastructure

#### `utils/quality_filter.py`

**Purpose:** Filter pseudo-ground-truth based on quality  
**Key method:** `calculate_quality_score(confidence, stability, completeness)`  
**Research-based:** Uses percentile thresholds (70/80/75)

#### `utils/consensus_generator.py`

**Purpose:** Generate consensus from multiple models  
**Key method:** `generate_consensus_for_maneuver(models, video, maneuver)`  
**Handles:** Model loading, inference, aggregation

#### `utils/consensus_manager.py`

**Purpose:** Orchestrate consensus generation with leave-one-out  
**Key method:** `generate_consensus_gt(maneuvers, target_model, phase)`  
**Features:** Caching, leave-one-out logic, statistics

### Integration Points

#### `utils/optuna_optimizer.py`

**Changes:**

- Added consensus setup in `__init__()` (lines 43-74)
- Modified `objective()` to use consensus (lines 129-194)
- Added MLflow logging enhancements

**Key addition:**

```python
if self.use_consensus and self.consensus_manager:
    pck_result = metrics_calc.calculate_pck_with_consensus_gt(...)
    trial_score = pck_result['pck_0_2']
```

#### `metrics/pose_metrics.py`

**Changes:**

- Added `calculate_pck_with_consensus_gt()` method
- Added `_estimate_torso_diameter()` helper

**Key addition:**

```python
def calculate_pck_with_consensus_gt(predictions, consensus_gt, maneuver_id):
    # Compare predictions against consensus
    # Return PCK score (now > 0!)
```

### Configuration

#### `configs/consensus_config.yaml`

**Defines:**

- Strong models: yolov8, pytorch_pose, mmpose
- Weak models: mediapipe, blazepose
- Quality filter weights: 0.4/0.4/0.2
- Percentile schedule: 70/80/75

#### `configs/evaluation_config_production_optuna.yaml`

**Added:**

```yaml
optuna_validation:
  use_consensus: true # ← Enable consensus!
  num_clips: 75
  cameras: [SONY_300]
```

---

## 🎯 Runtime Artifacts

When you run the pipeline, these files are created:

```
runs/test_consensus/
│
├── consensus_cache/              ← Consensus GT files
│   ├── yolov8_optuna_gt.json
│   ├── pytorch_pose_optuna_gt.json
│   ├── mmpose_optuna_gt.json
│   ├── mediapipe_optuna_gt.json
│   └── blazepose_optuna_gt.json
│
├── mlruns/                       ← MLflow tracking
│   └── {experiment_id}/
│       └── {run_id}/
│           ├── metrics/
│           │   ├── pck_0_2      ← KEY METRIC!
│           │   └── optuna_trial_score
│           └── params/
│               ├── validation_method
│               ├── consensus_models
│               └── trial_number
│
└── logs/                         ← Execution logs
    └── evaluation_*.log
```

### Consensus Cache File Structure

```json
{
  "version": "1.0",
  "gt_data": {
    "maneuver_id_001": {
      "maneuver_id": "SONY_300_..._maneuver_0",
      "maneuver_type": "Pop-up",
      "frames": [
        {
          "keypoints": [[x1,y1], [x2,y2], ...],  // 17 keypoints
          "confidence": [c1, c2, ...],            // 17 confidences
          "source_models": ["yolov8", "mmpose"],
          "num_contributing_models": [2, 2, ...]
        }
      ],
      "source_models": ["yolov8", "mmpose"],
      "num_frames": 58
    }
  },
  "stats": {
    "total_maneuvers": 75,
    "successful": 75,
    "failed": 0,
    "total_frames": 4350,
    "avg_confidence": 0.723
  }
}
```

---

## 🔄 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Pipeline Start                                           │
│    python run_evaluation.py --eval-mode optuna ...          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. OptunaPoseOptimizer.__init__()                           │
│    - Check config: use_consensus = true?                    │
│    - Initialize ConsensusManager                            │
│    - Set consensus_gt = None (lazy load)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. First Trial: objective(trial)                            │
│    - consensus_gt is None?                                  │
│    - YES → Generate consensus GT                            │
│      ├─ ConsensusManager.generate_consensus_gt()            │
│      ├─ ConsensusGenerator.generate_consensus_for_maneuver()│
│      ├─ Load models, run inference                          │
│      ├─ Aggregate predictions (mean)                        │
│      ├─ Apply quality filtering                             │
│      └─ Save to cache: {model}_optuna_gt.json               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Calculate PCK                                            │
│    - Run model with trial parameters                        │
│    - Get predictions for all 75 clips                       │
│    - PoseMetrics.calculate_pck_with_consensus_gt()          │
│      ├─ Compare predictions vs consensus                    │
│      ├─ Normalize by torso diameter                         │
│      └─ Return PCK score (> 0!)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Subsequent Trials                                        │
│    - consensus_gt already loaded (cached)                   │
│    - Use cached consensus → Fast!                           │
│    - Calculate PCK against same consensus                   │
│    - Log metrics to MLflow                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Verification Checklist

After implementation, verify:

- [x] Files created: 14 files
- [x] Core modules: 3 new files
- [x] Integration: 2 files modified
- [x] Configuration: 3 files updated
- [x] Tests: 1 new test file
- [x] Documentation: 5 guides created
- [x] Total LOC: 1,410 lines

**Status:** ✅ All files accounted for and documented

---

## 📚 Related Documentation

- **Quick Start:** `CONSENSUS_QUICKSTART.md`
- **Full Details:** `CONSENSUS_IMPLEMENTATION_COMPLETE.md`
- **Status:** `CONSENSUS_IMPLEMENTATION_STATUS.md`
- **Summary:** `IMPLEMENTATION_SUMMARY.md`
- **This Document:** `IMPLEMENTATION_FILE_TREE.md`

---

**Generated:** October 23, 2025  
**Status:** ✅ Implementation Complete  
**Ready for:** Testing and validation
