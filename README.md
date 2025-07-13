# Surfing Pose Estimation Evaluation Framework

A comprehensive framework for evaluating pose estimation models on surfing footage, with automated hyperparameter optimization and production-ready model comparison.

## 🚀 Quick Start

### Production Evaluation (Recommended)

Run the complete two-phase automated pipeline:

```bash
# Full automated pipeline (Optuna + Model Comparison)
python run_production_evaluation.py

# Custom models and clips
python run_production_evaluation.py --models mediapipe yolov8_pose --max-clips 100

# Skip optimization and use existing best parameters
python run_production_evaluation.py --skip-optuna

# Run only optimization phase
python run_production_evaluation.py --optuna-only

# Run only comparison phase (requires existing best parameters)
python run_production_evaluation.py --comparison-only
```

### Manual Evaluation

For development and testing:

```bash
# Quick test with default models
python evaluate_pose_models.py --config configs/evaluation_config.yaml --quick-test

# Full evaluation with Optuna optimization
python evaluate_pose_models.py --config configs/evaluation_config_production.yaml --use-optuna --models mediapipe yolov8_pose

# Standard evaluation without optimization
python evaluate_pose_models.py --config configs/evaluation_config_production.yaml --models mediapipe yolov8_pose
```

## 📁 Project Structure

```
surf_pose_evaluation/
├── configs/
│   ├── evaluation_config.yaml                    # Development config
│   ├── evaluation_config_production.yaml         # Legacy production config
│   ├── evaluation_config_production_optuna.yaml  # Phase 1: Optimization
│   ├── evaluation_config_production_comparison.yaml # Phase 2: Comparison
│   └── model_configs/                            # Model-specific configs
├── data/                                         # Dataset location
├── models/                                       # Model wrappers
├── metrics/                                      # Evaluation metrics
├── results/                                      # Output directory
│   ├── mlruns/                                  # MLflow tracking
│   ├── best_params/                             # Optuna results
│   ├── predictions/                             # Model predictions
│   └── visualizations/                          # Sample videos
├── evaluate_pose_models.py                      # Main evaluation script
├── run_production_evaluation.py                 # Automated pipeline
└── README.md                                    # This file
```

## 🔄 Two-Phase Production Workflow

### Phase 1: Hyperparameter Optimization

**Purpose**: Find optimal hyperparameters for each model using Optuna

**Configuration**: `configs/evaluation_config_production_optuna.yaml`

**MLflow Experiment**: `surf_pose_production_optuna`

**Key Features**:

- Uses subset of data (50 clips) for speed
- Runs 50 trials per model by default
- Saves best parameters to `results/best_params/`
- Focuses on PCK@0.2 optimization
- Minimal visualization for speed

**Run Types**:

- `{model}_optuna_trial_XXX`: Individual optimization trials
- `{model}_optuna_summary`: Aggregated optimization statistics
- `{model}_optuna_best_full_eval`: Best configuration on full data

### Phase 2: Model Comparison

**Purpose**: Compare models using optimal hyperparameters on full dataset

**Configuration**: `configs/evaluation_config_production_comparison.yaml`

**MLflow Experiment**: `surf_pose_production_comparison`

**Key Features**:

- Uses full dataset (200 clips)
- Loads best parameters from Phase 1
- Comprehensive metrics and visualizations
- Production-ready results
- Generates prediction files

**Run Types**:

- `{model}_evaluation_optimized`: Final model evaluation with best params

## 🎯 Available Models

| Model        | Wrapper              | Status              | Notes               |
| ------------ | -------------------- | ------------------- | ------------------- |
| MediaPipe    | `MediaPipeWrapper`   | ✅ Always available | CPU-optimized, fast |
| BlazePose    | `BlazePoseWrapper`   | ⚠️ Optional         | GPU-accelerated     |
| YOLOv8-Pose  | `YOLOv8Wrapper`      | ⚠️ Optional         | Real-time detection |
| PyTorch Pose | `PyTorchPoseWrapper` | ⚠️ Optional         | KeypointRCNN        |
| MMPose       | `MMPoseWrapper`      | ⚠️ Optional         | Research models     |

## 📊 MLflow Experiments

### Experiment Organization

The framework creates separate MLflow experiments for different purposes:

1. **`surf_pose_production_optuna`**: Hyperparameter optimization

   - Trial runs: Individual parameter combinations
   - Summary runs: Optimization statistics
   - Best eval runs: Full evaluation with best parameters

2. **`surf_pose_production_comparison`**: Final model comparison

   - Evaluation runs: Production-ready model comparisons
   - Uses optimized parameters from Phase 1

3. **Legacy experiments**: Single-phase runs (backward compatibility)

### Viewing Results

Start MLflow UI to view results:

```bash
mlflow ui --backend-store-uri ./results/mlruns --port 5000
```

Navigate to http://localhost:5000 to explore:

- Model performance comparisons
- Hyperparameter optimization progress
- Detailed metrics and visualizations
- Run artifacts and logs

## 🔧 Configuration Files

### Production Configurations

#### `evaluation_config_production_optuna.yaml`

- **Purpose**: Hyperparameter optimization
- **Dataset**: Subset (50 clips) for speed
- **Optuna**: Enabled with 50 trials
- **Visualization**: Disabled for speed
- **Experiment**: `surf_pose_production_optuna`

#### `evaluation_config_production_comparison.yaml`

- **Purpose**: Model comparison with best parameters
- **Dataset**: Full dataset (200 clips)
- **Optuna**: Disabled
- **Visualization**: Full visualization suite
- **Experiment**: `surf_pose_production_comparison`
- **Parameter Loading**: Loads best params from Phase 1

### Development Configurations

#### `evaluation_config.yaml`

- **Purpose**: Development and testing
- **Dataset**: Configurable size
- **Optuna**: Optional
- **Experiment**: `surf_pose_evaluation`

## 🎛️ Hyperparameter Optimization

### Configuring Search Spaces

Edit model config files in `configs/model_configs/{model_name}.yaml`:

```yaml
# Example: yolov8_pose.yaml
model_size: "n" # Default value

optuna_search_space:
  model_size:
    type: "categorical"
    choices: ["n", "s", "m", "l", "x"]

  conf:
    type: "uniform"
    low: 0.1
    high: 0.9

  iou:
    type: "uniform"
    low: 0.3
    high: 0.8
```

### Supported Parameter Types

- **`categorical`**: Discrete choices
- **`uniform`**: Continuous range
- **`int`**: Integer range
- **`loguniform`**: Log-scale continuous range

### Optimization Metrics

Primary objective: **PCK@0.2** (Percentage of Correct Keypoints at 0.2 threshold)

Secondary considerations:

- Inference speed (FPS)
- Memory usage
- Detection accuracy

## 📈 Metrics and Evaluation

### Pose Accuracy Metrics

- **PCK@0.1, 0.2, 0.3**: Percentage of Correct Keypoints at different thresholds
- **MPJPE**: Mean Per Joint Position Error
- **Detection F1**: Person detection accuracy

### Performance Metrics

- **FPS**: Frames per second (higher is better)
- **Inference Time**: Average time per frame (lower is better)
- **Memory Usage**: Peak GPU/CPU memory (lower is better)
- **Model Size**: Storage requirements

### Interpretation Guidelines

| Metric       | Good | Acceptable | Poor |
| ------------ | ---- | ---------- | ---- |
| PCK@0.2      | >0.7 | 0.5-0.7    | <0.5 |
| FPS          | >15  | 10-15      | <10  |
| Detection F1 | >0.8 | 0.6-0.8    | <0.6 |

## 🎥 Visualization and Outputs

### Prediction Files

Location: `results/predictions/{model_name}/`

Format: Standardized JSON with:

- Frame-by-frame keypoint predictions
- Confidence scores
- Bounding boxes
- Metadata (model config, timing)

**New Aligned Naming Structure:**

Predictions and visualizations now use the same naming pattern for easy correlation:

```
# Prediction files
results/predictions/{model_name}/maneuver_{type}_{score}_{video_stem}_predictions.json

# Visualization files
results/visualizations/{timestamp}_{model_name}/maneuver_{type}_{score}_{video_stem}_poses.mp4
```

**Example:**

```
# Prediction
results/predictions/yolov8_pose/maneuver_Pumping_08_C0010_clip_4_predictions.json

# Corresponding visualization
results/visualizations/20240315_143022_yolov8_pose/maneuver_Pumping_08_C0010_clip_4_poses.mp4
```

This makes it easy to:

- ✅ Identify maneuver type and execution score from filename
- ✅ Match predictions with their corresponding visualizations
- ✅ Sort files by score or maneuver type
- ✅ Quickly find specific maneuvers

### Visualization Videos

Location: `results/visualizations/` or shared storage

Features:

- Pose overlay on original footage
- Keypoint connections
- Confidence visualization
- Multiple person support

### Reports

- **HTML Report**: Comprehensive evaluation summary
- **JSON Summary**: Machine-readable results
- **MLflow Artifacts**: Detailed run information

## 🔄 Automated Pipeline Usage

### Basic Usage

```bash
# Complete pipeline with all models
python run_production_evaluation.py

# Specific models only
python run_production_evaluation.py --models mediapipe yolov8_pose

# Limited dataset for testing
python run_production_evaluation.py --max-clips 50
```

### Advanced Usage

```bash
# Skip optimization if you have existing best parameters
python run_production_evaluation.py --skip-optuna

# Run only optimization phase
python run_production_evaluation.py --optuna-only

# Run only comparison phase (requires existing best parameters)
python run_production_evaluation.py --comparison-only

# Custom model selection
python run_production_evaluation.py --models mediapipe blazepose --max-clips 100
```

### Pipeline Outputs

The automated pipeline generates:

1. **Best Parameters**: `results/best_params/best_parameters.yaml`
2. **Summary Report**: `results/production_evaluation_summary.json`
3. **MLflow Experiments**: Two separate experiments for optimization and comparison
4. **Prediction Files**: Standardized JSON predictions for each model
5. **Visualizations**: Sample videos with pose overlays

## 🛠️ Development and Customization

### Adding New Models

1. Create model wrapper inheriting from `BasePoseModel`
2. Add to model registry in `evaluate_pose_models.py`
3. Create config file in `configs/model_configs/`
4. Define Optuna search space in config

### Custom Metrics

1. Add metric calculation to `metrics/pose_metrics.py`
2. Update aggregation in `_aggregate_metrics()`
3. Configure in evaluation config files

### Dataset Integration

1. Update `data_handling/data_loader.py` for new data sources
2. Modify annotation format handling
3. Update visualization components

## 🚨 Troubleshooting

### Common Issues

**MLflow UI not starting**:

```bash
# Check if port is in use
lsof -i :5000
# Use different port
mlflow ui --backend-store-uri ./results/mlruns --port 5001
```

**CUDA out of memory**:

- Reduce batch size in model configs
- Use smaller models (e.g., YOLOv8n instead of YOLOv8x)
- Enable mixed precision training

**Missing prediction files**:

- Check `results/predictions/` directory
- Verify model ran successfully in MLflow
- Check logs for prediction generation errors

**Optuna optimization fails**:

- Verify search space configuration
- Check model config files exist
- Ensure sufficient disk space for trials

### Performance Optimization

**Speed up evaluation**:

- Use `--max-clips` to limit dataset size
- Disable visualization during optimization
- Use faster video formats (h264 vs ffv1)

**Reduce memory usage**:

- Process clips sequentially
- Clear GPU cache between models
- Use CPU-only models for large datasets

## 📚 References and Citation

If you use this framework in your research, please cite:

```bibtex
@software{surf_pose_evaluation,
  title={Surfing Pose Estimation Evaluation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/surf_pose_evaluation}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

- Check the troubleshooting section above
- Review MLflow logs for detailed error messages
- Open an issue on GitHub with:
  - Error messages
  - Configuration files used
  - System information (OS, GPU, Python version)
