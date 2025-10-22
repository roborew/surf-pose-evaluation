# Consensus-Based Optuna Optimization System

## Overview

This document describes the consensus-based pseudo-ground-truth system implemented to fix broken Optuna hyperparameter optimization for surf pose estimation.

### Problem Statement

Previously, Optuna optimization was broken because:

- Surf footage lacks manual pose annotations (only temporal/maneuver labels exist)
- Without ground truth, `pck_0_2` always returned 0
- Optuna couldn't optimize parameters meaningfully

### Solution

Implemented research-validated consensus-based pseudo-ground-truth generation:

1. **Multi-model consensus**: Generate high-quality pseudo-GT from YOLOv8-Pose, PyTorch Pose (RTMPose), and MMPose
2. **Adaptive quality filtering**: Use percentile-based filtering with composite scores (confidence + stability + completeness)
3. **Leave-one-out validation**: Prevent circular reasoning by excluding model being optimized from its own consensus
4. **Session-level data splitting**: Prevent data leakage by keeping FULL/WIDE/standard variants together

## Architecture

### Core Components

```
utils/
├── session_analyzer.py          # Analyze sessions, select best for consensus
├── quality_filter.py            # Adaptive percentile quality filtering
├── consensus_generator.py       # Generate consensus pseudo-ground-truth
└── optuna_optimizer.py          # Modified to use consensus validation

metrics/
└── pose_metrics.py              # Extended with consensus-based PCK calculation

configs/
├── consensus_config.yaml        # Shared consensus configuration
├── evaluation_config_production_optuna.yaml      # References consensus config
└── evaluation_config_production_comparison.yaml  # References consensus config

scripts/
└── generate_consensus.py        # Standalone consensus generation script
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   1. Session Analysis                        │
│  Analyze all sessions → Select best SONY_300 for consensus  │
│  Split remaining sessions: 50% Optuna, 50% Comparison       │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              2. Consensus Generation (Upfront)               │
│  Run YOLOv8 + PyTorch Pose + MMPose on consensus session    │
│  → Compute quality scores (confidence + stability + comp.)   │
│  → Apply percentile filtering → Generate consensus keypoints │
│  → Save to cache with leave-one-out variants                 │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              3. Optuna Optimization (Per Model)              │
│  For model X:                                                 │
│    - Load consensus from models Y & Z (leave-one-out)        │
│    - Sample hyperparameters                                   │
│    - Run inference                                            │
│    - Calculate PCK against consensus (adaptive percentile)   │
│    - Repeat for 50 trials                                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│           4. Comparison Testing (Larger Dataset)             │
│  Use optimized parameters + consensus on 200 clips           │
│  → Final model comparison with reliable PCK scores           │
└─────────────────────────────────────────────────────────────┘
```

### Data Splits (No Leakage)

```
Total Sessions:
├── Consensus Set (1 session, SONY_300 only)
│   └── SESSION_070325 (43 clips, ~150 maneuvers)
│       - Used to generate pseudo-GT
│       - NEVER used for evaluation
│
├── Optuna Validation Set (~50% of remaining)
│   └── SONY_300 + SONY_70 clips
│       - Different sessions from consensus
│       - Used for hyperparameter tuning (75 clips)
│
└── Comparison Test Set (~50% of remaining)
    └── SONY_300 + SONY_70 clips
        - Different sessions from both consensus AND Optuna
        - Used for final model comparison (200 clips)
```

**Session Grouping**: FULL/WIDE/standard variants of same session stay together to prevent leakage.

## Usage

### Step 1: Session Analysis & Consensus Generation

```bash
# Analyze sessions and generate consensus (first time only)
python scripts/generate_consensus.py --config configs/consensus_config.yaml

# To force regeneration:
python scripts/generate_consensus.py --force

# To only analyze without generating:
python scripts/generate_consensus.py --analyze-only
```

This script:

1. Analyzes all SONY_300 and SONY_70 sessions
2. Recommends best session for consensus (most maneuvers + diversity)
3. Updates `configs/consensus_config.yaml` with session assignments
4. Generates consensus pseudo-GT for Optuna and comparison sets
5. Saves to `./data/consensus_cache/`

### Step 2: Run Optuna with Consensus Validation

```bash
# Run Optuna optimization (now uses consensus)
python run_evaluation.py --run-optuna --eval-mode comprehensive_test

# Or with custom run name:
python run_evaluation.py \
  --run-optuna \
  --eval-mode comprehensive_test \
  --run-name "optuna_with_consensus_v1"
```

Optuna will now:

- Use consensus-based PCK scores (not broken detection metrics)
- Apply leave-one-out validation (model excluded from own consensus)
- Use adaptive percentile filtering (70% → 80% → 75% across trials)
- Run 50 trials per model (reduced from 100 to prevent overfitting)

### Step 3: Run Comparison with Consensus

```bash
# Run final comparison (now uses consensus)
python run_evaluation.py --run-comparison --eval-mode comprehensive_test
```

## Configuration

### Main Consensus Config

`configs/consensus_config.yaml` controls all consensus settings:

```yaml
consensus:
  generation:
    consensus_session: "SESSION_070325" # Auto-selected
    consensus_models: ["yolov8", "pytorch_pose", "mmpose"]
    excluded_models: ["mediapipe", "blazepose"]

  quality_filter:
    composite_weights:
      confidence: 0.4
      stability: 0.4
      completeness: 0.2
    percentile_schedule:
      initialization: 70.0 # First 10% of trials
      growth: 80.0 # Middle 60% of trials
      saturation: 75.0 # Final 30% of trials
```

### Evaluation Configs

Both `evaluation_config_production_optuna.yaml` and `evaluation_config_production_comparison.yaml` now reference the shared consensus config:

```yaml
# Reference shared consensus config
consensus_config: "configs/consensus_config.yaml"

optuna_validation:
  use_consensus: true # Enable consensus-based validation
  num_clips: 75
```

## Technical Details

### Adaptive Percentile Quality Filtering

Based on research (PercentMatch, FreeMatch), uses multi-stage percentile thresholds:

**Composite Score**: `Q = 0.4*confidence + 0.4*stability + 0.2*completeness`

**Adaptive Percentile**:

- **Initialization** (first 10% trials): 70th percentile → keep top 30% (conservative)
- **Growth** (middle 60% trials): 80th percentile → keep top 20% (standard)
- **Saturation** (final 30% trials): 75th percentile → keep top 25% (prevent overfitting)

### Leave-One-Out Validation

Prevents circular reasoning where model evaluates against its own predictions:

```python
# When optimizing YOLOv8:
consensus = mean(PyTorch Pose predictions, MMPose predictions)

# When optimizing PyTorch Pose:
consensus = mean(YOLOv8 predictions, MMPose predictions)

# When optimizing MediaPipe (excluded from consensus):
consensus = mean(YOLOv8 predictions, PyTorch Pose predictions, MMPose predictions)
```

### Temporal Stability Calculation

Measures keypoint consistency across adjacent frames:

```python
stability = exp(-variance_across_frames / 100)
```

Lower variance = higher stability = better consensus quality.

### Skeleton Completeness

Fraction of expected keypoints detected with sufficient confidence:

```python
completeness = (num_keypoints_with_confidence > 0.3) / total_keypoints
```

## Validation & Quality Checks

### Data Leakage Verification

The system automatically verifies:

1. Consensus session NOT in Optuna set
2. Consensus session NOT in comparison set
3. No overlap between Optuna and comparison sets
4. FULL/WIDE/standard variants stay together

### Consensus Quality Metrics

Track to ensure high-quality pseudo-GT:

- **Inter-model agreement**: How well models agree on keypoint positions
- **Keypoint coverage**: Fraction of frames with valid consensus after filtering
- **Quality score distribution**: Check most scores are high
- **Temporal consistency**: Smooth trajectories (no jitter)

Target: >90% frame coverage after quality filtering

## Benefits Over Previous Approach

| Aspect                  | Old (Broken)      | New (Consensus)                  |
| ----------------------- | ----------------- | -------------------------------- |
| **Ground Truth**        | None for surf     | Consensus pseudo-GT              |
| **PCK Score**           | Always 0          | Meaningful 0-1 range             |
| **Optuna Optimization** | Flat/random       | Clear convergence                |
| **Validation**          | Detection metrics | True PCK with adaptive filtering |
| **Data Leakage**        | Possible          | Prevented by session grouping    |
| **Research Validated**  | No                | Yes (PercentMatch, FreeMatch)    |

## Expected Results

### Before Consensus (Broken)

```
Trial 000: PCK@0.2 = 0.000
Trial 010: PCK@0.2 = 0.000
Trial 020: PCK@0.2 = 0.000
...
Best params: Random (no signal)
```

### After Consensus (Working)

```
Trial 000: PCK@0.2 = 0.623
Trial 010: PCK@0.2 = 0.681
Trial 020: PCK@0.2 = 0.724
...
Best params: Meaningful improvement over defaults
```

## Troubleshooting

### Consensus Data Not Found

```
ERROR: Consensus data not found at ./data/consensus_cache/optuna_validation
```

**Solution**: Run consensus generation first:

```bash
python scripts/generate_consensus.py
```

### Low Consensus Coverage

If quality reports show <70% coverage:

- Check model predictions are valid
- Review quality threshold settings
- Ensure video files are accessible

### Optuna Still Returns 0

If Optuna still returns PCK = 0:

- Verify `use_consensus: true` in config
- Check consensus data loaded successfully
- Review logs for errors in consensus loading

## Future Enhancements

Potential improvements:

1. **Per-class percentile normalization**: Prevent easy poses from dominating
2. **Ablation-based weighting**: Dynamically weight models by reliability
3. **Multi-view triangulation**: If multi-camera data is synchronized
4. **Active learning**: Selectively request manual annotations for low-consensus regions

## References

Research foundations:

- **PercentMatch**: Percentile-based pseudo-label filtering (https://arxiv.org/pdf/2208.13946.pdf)
- **FreeMatch**: Adaptive threshold adjustment (https://arxiv.org/pdf/2205.07246.pdf)
- **Semi-supervised learning**: Ground-truth-free hyperparameter tuning (https://arxiv.org/html/2412.01116v1)

## File Changes Summary

### New Files Created

- `utils/session_analyzer.py`
- `utils/quality_filter.py`
- `utils/consensus_generator.py`
- `configs/consensus_config.yaml`
- `scripts/generate_consensus.py`
- `docs/CONSENSUS_SYSTEM_README.md`

### Modified Files

- `metrics/pose_metrics.py` - Added `calculate_metrics_with_consensus()`
- `utils/optuna_optimizer.py` - Added consensus validation support
- `configs/evaluation_config_production_optuna.yaml` - Added consensus reference
- `configs/evaluation_config_production_comparison.yaml` - Added consensus reference

### Legacy Code Removed

- `_calculate_detection_metrics_without_ground_truth()` usage in Optuna (lines 96-103)
- Broken PCK fallback to 0

## Support

For issues or questions:

1. Check logs in `results/runs/{timestamp}/logs/`
2. Review consensus quality reports in `results/consensus_quality_reports/`
3. Verify data splits have no leakage
4. Ensure all dependencies installed (opencv-python, tqdm, numpy, etc.)
