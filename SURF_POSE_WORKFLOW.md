# 🏄‍♂️ Surf Pose Evaluation - Complete Workflow Guide

## 🎯 **Overview**

This system evaluates pose estimation models for surf footage using a **two-phase approach**:

1. **Phase 1: Optuna Optimization** - Find best parameters for each model
2. **Phase 2: Model Comparison** - Compare models using optimal parameters

## 🚀 **Quick Start - Run Everything**

```bash
# Complete automated pipeline
python run_production_evaluation.py

# Or with limited clips for testing
python run_production_evaluation.py --max-clips 20
```

## 📋 **What Happens Automatically**

### **Phase 1: Optuna Hyperparameter Optimization**

- **Config**: `configs/evaluation_config_production_optuna.yaml`
- **Purpose**: Find optimal parameters for surf footage conditions
- **Dataset**: 50 clips (smaller for speed)
- **Trials**: 100 trials with 5-hour timeout
- **Visualizations**: ❌ OFF (for speed)
- **Predictions**: ❌ OFF (for speed)
- **Output**: Best parameters → `./results/best_params/best_parameters.yaml`

### **Phase 2: Model Comparison**

- **Config**: `configs/evaluation_config_production_comparison.yaml`
- **Purpose**: Compare models using optimal parameters
- **Dataset**: 200 clips (full dataset)
- **Input**: Best parameters from Phase 1
- **Visualizations**: ✅ ON (final analysis)
- **Predictions**: ✅ ON (final analysis)
- **Output**: Comparison results, visualizations, prediction files

## 🔧 **How Best Parameters Flow Between Phases**

### **Phase 1 Output**

```yaml
# ./results/best_params/best_parameters.yaml
mediapipe:
  min_detection_confidence: 0.15
  min_tracking_confidence: 0.12
  model_complexity: 2
  # ... other optimized parameters

yolov8_pose:
  model_size: "m"
  confidence_threshold: 0.08
  # ... other optimized parameters
```

### **Phase 2 Input**

The comparison config automatically loads these:

```yaml
# configs/evaluation_config_production_comparison.yaml
models:
  load_best_params:
    enabled: true
    source_path: "./results/best_params"
    fallback_to_defaults: true
```

## 📊 **Surf-Specific Optimizations**

All model configs have been optimized for surf conditions:

### **Full Confidence Threshold Range**

- **Problem**: Surf footage has challenging conditions (spray, motion blur) BUT also clear frames
- **Solution**: Expanded from 0.3-0.9 to **0.05-1.0** (full range exploration)
- **Rationale**: Let Optuna find optimal thresholds - might be low (0.1) for challenging conditions OR high (0.9) for clear footage

### **Expanded Model Sizes**

- **YOLOv8**: Now tests n, s, m, **l** (added large model)
- **MediaPipe/BlazePose**: Tests all complexity levels (0, 1, 2)

### **Surf-Specific Parameters**

- **Segmentation**: Added for better person detection
- **Static image mode**: Better for challenging frames
- **Increased detection limits**: More detections per frame

## 🎛️ **Manual Control Options**

### **Run Only Optuna (Find Best Parameters)**

```bash
python run_production_evaluation.py --optuna-only
```

### **Run Only Comparison (Use Existing Parameters)**

```bash
python run_production_evaluation.py --comparison-only
```

### **Skip Optuna (Use Existing Parameters)**

```bash
python run_production_evaluation.py --skip-optuna
```

## 📈 **Results Structure**

After running the full pipeline:

```
results/
├── best_params/
│   └── best_parameters.yaml          # Optimal parameters from Optuna
├── mlruns/
│   ├── surf_pose_production_optuna/   # Optuna experiment results
│   └── surf_pose_production_comparison/ # Comparison experiment results
├── visualizations/                    # Sample videos with pose overlays
├── predictions/                       # JSON prediction files
└── production_evaluation_summary.json # Final comparison summary
```

## 🔍 **Configuration Files Explained**

### **Production Pipeline Configs (Used by `run_production_evaluation.py`)**

- `evaluation_config_production_optuna.yaml` - Phase 1 (Optuna optimization)
- `evaluation_config_production_comparison.yaml` - Phase 2 (Model comparison)

### **Model Configs (Optimized for Surf)**

- `model_configs/mediapipe.yaml` - MediaPipe search space
- `model_configs/blazepose.yaml` - BlazePose search space
- `model_configs/yolov8_pose.yaml` - YOLOv8 search space
- `model_configs/mmpose.yaml` - MMPose search space
- `model_configs/pytorch_pose.yaml` - PyTorch search space

### **Development/Debug Config**

- `evaluation_config_macos.yaml` - Used by debug scripts only

## 🎥 **Visualization & Prediction Strategy**

### **During Optuna (Phase 1)**

- **Focus**: Speed and parameter optimization
- **Visualizations**: ❌ Disabled
- **Predictions**: ❌ Disabled
- **Only metrics logged** for parameter selection

### **During Comparison (Phase 2)**

- **Focus**: Quality analysis with optimal parameters
- **Visualizations**: ✅ Enabled (sample videos with pose overlays)
- **Predictions**: ✅ Enabled (JSON files for all clips)
- **Full analysis** with best parameters

## 🚨 **Important Notes**

1. **Predictions and visualizations are ONLY generated in Phase 2** with optimal parameters
2. **Best parameters automatically flow** from Phase 1 to Phase 2
3. **The pipeline handles everything** - no manual parameter copying needed
4. **Use `--max-clips` for testing** to avoid long runtimes

## 🔧 **Troubleshooting**

### **If Optuna Fails**

```bash
# Skip Optuna and use default parameters
python run_production_evaluation.py --skip-optuna
```

### **If You Want Different Models**

```bash
# Test specific models only
python run_production_evaluation.py --models yolov8_pose mediapipe
```

### **If You Want More Visualizations**

Edit `configs/evaluation_config_production_comparison.yaml`:

```yaml
visualization:
  max_examples_per_model: 20 # Increase from 10
```

## 📝 **Summary**

- **One command runs everything**: `python run_production_evaluation.py`
- **Two phases**: Optuna optimization → Model comparison
- **Parameters flow automatically** between phases
- **Visualizations/predictions only in Phase 2** with optimal parameters
- **Surf-optimized configurations** already implemented
