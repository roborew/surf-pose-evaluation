# Consensus-Based Optuna Optimization - Quick Start

## What It Does

Fixes Optuna optimization for surf footage by using **consensus pseudo-ground-truth** from multiple models instead of broken detection metrics.

## How to Use

### One Command - Everything Automated

**Optuna only:**

```bash
python run_evaluation.py --run-optuna --run-name "my_experiment"
```

**Full pipeline (Optuna + Comparison):**

```bash
python run_evaluation.py --run-name "my_experiment"
```

**That's it.** The pipeline automatically:

1. Generates consensus pseudo-GT **once** at the start
2. Runs Optuna optimization with PCK scores (uses consensus)
3. Runs comparison validation (uses **same consensus** from this run)
4. Logs everything to MLflow
5. Saves best parameters

### What Gets Created

```
.../results/runs/20251022_130000_my_experiment/
├── consensus_cache/              # Pseudo ground truth
│   ├── optuna_validation/        # GT for 75 Optuna clips
│   └── comparison_test/          # GT for 200 comparison clips
├── consensus_quality_reports/    # Quality metrics
├── mlflow_tracking/              # PCK scores per trial
├── best_params.yaml              # Optimal hyperparameters
└── logs/                         # Full logs
```

Everything in ONE directory. Nothing in project root.

## Configuration

### Default (Automated)

`configs/evaluation_config_production_optuna.yaml`:

```yaml
optuna_validation:
  use_consensus: true
  consensus_run_path: null # null = auto-generate in this run
```

### Reuse Existing Consensus

If you want to reuse consensus from a previous run:

```yaml
optuna_validation:
  use_consensus: true
  consensus_run_path: ".../runs/20251022_120000_previous_run"
```

## View Results

### Check Logs

```bash
tail -f .../runs/20251022_130000_my_experiment/logs/run_evaluation.log
```

### MLflow UI

```bash
cd .../runs/20251022_130000_my_experiment
mlflow ui --backend-store-uri file://$(pwd)/mlflow_tracking
```

Open http://localhost:5000 to see PCK improvement graph across trials.

### Best Parameters

```bash
cat .../runs/20251022_130000_my_experiment/best_params.yaml
```

## Expected Output

```
GENERATING CONSENSUS PSEUDO-GROUND-TRUTH
==================================================
Generating consensus GT for 75 Optuna clips...
✅ Generated consensus for Optuna validation
Generating consensus GT for 200 comparison clips...
✅ Generated consensus for comparison testing

OPTUNA OPTIMIZATION
==================================================
Using consensus from specified path: .../runs/20251022_130000_my_experiment
✅ Loaded consensus data: 75 clips

Trial 000/50: PCK@0.2 = 0.623
Trial 001/50: PCK@0.2 = 0.645 ✅ New best
Trial 002/50: PCK@0.2 = 0.638
...
Trial 049/50: PCK@0.2 = 0.681 ✅ New best

✅ Optimization complete!
💾 Best parameters saved
```

## How It Works

### Consensus Generation

1. Pipeline selects validation clips (75 for Optuna, 200 for comparison)
2. Runs 3 models (YOLOv8, PyTorch Pose, MMPose) on those clips
3. Generates consensus by averaging high-quality keypoints
4. Applies adaptive quality filtering (confidence, stability, completeness)
5. Saves pseudo-GT to run directory

### Optuna Optimization

1. Each trial runs the model being optimized on the 75 validation clips
2. Compares predictions to consensus pseudo-GT (from those same clips)
3. Calculates PCK@0.2 score
4. Logs to MLflow
5. Finds best hyperparameters

### Data Splitting

- **Optuna clips**: 75 clips used for generating Optuna GT and running trials
- **Comparison clips**: 200 separate clips for final testing (no overlap with Optuna)

No data leakage - Optuna and comparison use completely separate clips.

## Troubleshooting

### "Falling back to detection metrics"

- Check `use_consensus: true` in config
- Check logs for consensus generation errors

### "No improvement in Optuna"

- Check logs for PCK scores - should be 0.5-0.8 range, not 0
- Verify consensus was generated successfully
- Check consensus cache has files in `optuna_validation/`

### "Consensus generation failed"

- Check video clips exist and are readable
- Check logs in `.../runs/{timestamp}_{name}/logs/`
- Verify models (yolov8, pytorch_pose, mmpose) are working

## Cleanup Old Artifacts

If you have old files in project root from previous implementations:

```bash
cd surf-pose-evaluation/
rm -rf predictions/ consensus_cache/ consensus_predictions_*.json
ls -la . | grep consensus  # Should be empty
```

All new artifacts go to run directories automatically.

## Key Files

### In Config

- `configs/evaluation_config_production_optuna.yaml` - Optuna settings
- `configs/consensus_config.yaml` - Consensus generation settings

### In Run Directory

- `consensus_cache/` - Generated pseudo-GT annotations
  - `optuna_validation/` - GT for Optuna (75 clips)
  - `comparison_test/` - GT for comparison (200 clips)
- `consensus_quality_reports/` - Quality metrics for GT
- `mlflow_tracking/*/metrics/pck_0_2` - PCK scores per trial
- `best_params.yaml` - Optimized hyperparameters
- `logs/run_evaluation.log` - Full execution log

### In Project Root

- Nothing! All clean.

## That's It

Just run:

```bash
python run_evaluation.py --run-optuna --run-name "my_experiment"
```

Check MLflow for results. Done.
