# Surf Pose Evaluation

A comprehensive evaluation framework for pose estimation models on surf footage, designed for production use with organized run management and multi-machine collaboration.

## Features

- **Organized Run Management**: Timestamp-based run organization with isolated directories
- **Multi-Machine Collaboration**: Shared results directory for team collaboration
- **MLflow Integration**: Comprehensive experiment tracking with consolidated view
- **Automated Workflows**: Two-phase evaluation (Optuna optimization + comparison)
- **Aligned File Naming**: Consistent naming between predictions and visualizations
- **Performance Metrics**: Comprehensive accuracy and performance benchmarking
- **GPU Acceleration**: Automatic CUDA/MPS detection with optimal performance settings

## Quick Start

### Basic Usage

```bash
# Run full evaluation with organized runs
python run_production_evaluation.py

# Run with specific number of clips
python run_production_evaluation.py --max-clips 20

# Run with custom name
python run_production_evaluation.py --run-name "experiment_v1"

# Skip phases
python run_production_evaluation.py --skip-optuna
python run_production_evaluation.py --skip-comparison
```

### View Results

```bash
# Start MLflow UI for all experiments
python start_mlflow_ui.py

# Show experiments summary
python utils/mlflow_utils.py --summary

# View specific run's MLflow
mlflow ui --backend-store-uri data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/runs/TIMESTAMP_NAME/mlruns
```

## GPU Acceleration

The system automatically detects and uses the best available acceleration:
- **Linux/Production**: CUDA + FP16 precision (RTX 4090 optimized)  
- **macOS/Development**: MPS acceleration where supported
- **Fallback**: CPU with optimized settings

No configuration needed - just run! Test your setup: `python verify_gpu_setup.py`

## Directory Structure

### Shared Results Directory

All results are stored in the shared POSE directory for multi-machine collaboration:

```
data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/
├── runs/                           # Organized runs
│   ├── 20240315_143022_20clips/    # Timestamped run directory
│   │   ├── mlruns/                 # MLflow experiments for this run
│   │   ├── predictions/            # Model predictions
│   │   ├── visualizations/         # Pose visualizations
│   │   ├── best_params/            # Optimized parameters
│   │   ├── reports/                # Evaluation reports
│   │   ├── run_metadata.json      # Run information
│   │   └── run_summary.json       # Results summary
│   └── 20240315_150000_full/       # Another run
└── ...
```

### File Naming Conventions

**Predictions**: `maneuver_{type}_{score}_{video_stem}_predictions.json`
**Visualizations**: `maneuver_{type}_{score}_{video_stem}_poses.mp4`

Examples:

- `maneuver_cutback_8.5_SONY_70_SESSION_020325_C0010_clip_4_predictions.json`
- `maneuver_cutback_8.5_SONY_70_SESSION_020325_C0010_clip_4_poses.mp4`

## Multi-Machine Workflow

### Setup

1. **Ensure shared directory access**: All machines should have access to the POSE directory
2. **Run evaluations**: Each machine can run evaluations independently
3. **View consolidated results**: Use MLflow UI to view all experiments from all machines

### Example Multi-Machine Usage

**Machine 1 (20 clips test)**:

```bash
python run_production_evaluation.py --max-clips 20 --run-name "quick_test"
```

**Machine 2 (full dataset)**:

```bash
python run_production_evaluation.py --run-name "full_eval"
```

**View all results**:

```bash
python start_mlflow_ui.py --summary
```

## Run Management

### Organized Runs (Default)

Every run creates a timestamped directory with complete isolation:

- **Automatic organization**: No manual cleanup needed
- **Machine tracking**: Each run records which machine created it
- **Timestamped experiments**: MLflow experiments include timestamps
- **Easy comparison**: Compare results across different runs/machines

### Key Commands

```bash
# Clean up old runs (keep last 5)
python run_production_evaluation.py --cleanup

# List all previous runs
python utils/mlflow_utils.py --compare

# Export experiment data
python utils/mlflow_utils.py --export results_backup.json
```

## MLflow Integration

### Experiment Organization

- **Timestamped experiments**: `surf_pose_production_optuna_20240315_143022`
- **Machine tagging**: Each run tagged with hostname and user
- **Consolidated view**: Single MLflow UI shows all experiments from all runs
- **Artifact organization**: Run-specific artifact storage

### Accessing MLflow

```bash
# View all experiments from all runs
python start_mlflow_ui.py

# View specific run
mlflow ui --backend-store-uri data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/runs/TIMESTAMP/mlruns

# Get tracking URI for programmatic access
python -c "from utils.run_manager import RunManager; print(RunManager.get_shared_mlflow_uri())"
```

## Configuration

### Base Configuration

The system uses `configs/evaluation_config_production.yaml` as the base configuration. The run manager automatically creates run-specific configs with updated paths.

### Key Configuration Sections

- **Data paths**: Automatically updated for each run
- **MLflow settings**: Timestamped experiments with run-specific tracking
- **Output directories**: Isolated per run
- **Model parameters**: Consistent across runs

## Development

### Adding New Models

1. Create model wrapper in `models/`
2. Add to `models/enabled_models` in config
3. Run evaluation to test integration

### Extending Metrics

1. Add metric calculation to `metrics/`
2. Update config to include new metrics
3. Verify MLflow logging captures new metrics

### Custom Workflows

```python
from utils.run_manager import RunManager

# Create custom run
run_manager = RunManager(run_name="custom_experiment")

# Get run-specific config
config_path = run_manager.create_config_for_phase(
    "custom", "configs/base_config.yaml"
)

# Access directories
print(f"Predictions: {run_manager.predictions_dir}")
print(f"MLflow: {run_manager.mlflow_dir}")
```

## Benefits

### For Development

- **Isolated experiments**: No conflicts between different runs
- **Easy rollback**: Each run is completely self-contained
- **Clear organization**: Timestamp-based naming eliminates confusion

### For Production

- **Multi-machine support**: Team can run evaluations simultaneously
- **Comprehensive tracking**: All experiments tracked in single MLflow instance
- **Automated cleanup**: Optional cleanup of old runs
- **Consistent naming**: Aligned file naming across all outputs

### For Analysis

- **Easy comparison**: Compare results across runs and machines
- **Complete provenance**: Track which machine ran which experiment
- **Exportable data**: Export experiment data for external analysis

## Troubleshooting

### Common Issues

1. **Directory permissions**: Ensure all machines have read/write access to POSE directory
2. **MLflow conflicts**: Each run uses isolated MLflow tracking
3. **Disk space**: Use cleanup functionality to manage storage

### Debug Commands

```bash
# Check run manager status
python -c "from utils.run_manager import RunManager; rm = RunManager(); rm.print_run_info()"

# List all experiments
python utils/mlflow_utils.py --list

# Check directory structure
ls -la data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE/results/runs/
```

## Migration Notes

The system no longer supports legacy mode. All runs use the organized structure with the shared POSE directory. This ensures:

- **Consistent behavior**: All runs work the same way
- **Multi-machine compatibility**: Shared directory enables collaboration
- **Simplified maintenance**: Single code path reduces complexity

For any questions or issues, refer to the run metadata files or MLflow experiment tracking for detailed information about each run.
