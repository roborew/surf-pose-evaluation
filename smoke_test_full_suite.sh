#!/bin/bash
# Smoke test for full pose comparison pipeline
# Tests all models with consensus-based PCK validation
# Estimated runtime: 15-20 minutes

set -e

echo "=============================================="
echo "Full Suite Smoke Test - All Models"
echo "=============================================="
echo ""
echo "This will test the complete pipeline:"
echo "  • All 5 models (YOLOv8, PyTorch, MMPose, MediaPipe, BlazePose)"
echo "  • Consensus cache generation/loading"
echo "  • Optuna optimization (5 trials per model)"
echo "  • COCO validation (10 images)"
echo "  • Comparison evaluation (10 clips)"
echo ""
echo "Expected runtime: 15-20 minutes"
echo "All data logged to single MLflow run"
echo ""

read -p "Start smoke test? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Pre-flight checks
echo ""
echo "=== Pre-flight Checks ==="
echo ""

# Check conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "surf_pose_analysis" ]] && [[ "${CONDA_DEFAULT_ENV}" != "surf_pose_eval" ]]; then
    echo "❌ Wrong conda environment: ${CONDA_DEFAULT_ENV}"
    echo "   Run: conda activate surf_pose_analysis"
    exit 1
fi
echo "✓ Conda environment: ${CONDA_DEFAULT_ENV}"

# Check configs exist
if [ ! -f "configs/evaluation_config_production_optuna.yaml" ]; then
    echo "❌ Missing: configs/evaluation_config_production_optuna.yaml"
    exit 1
fi
echo "✓ Found: configs/evaluation_config_production_optuna.yaml"

if [ ! -f "configs/evaluation_config_production_comparison.yaml" ]; then
    echo "❌ Missing: configs/evaluation_config_production_comparison.yaml"
    exit 1
fi
echo "✓ Found: configs/evaluation_config_production_comparison.yaml"

# Check PyTorch config has constrained ranges
if ! grep -q "low: 0.25" configs/model_configs/pytorch_pose.yaml; then
    echo "⚠️  Warning: PyTorch config may not have constrained ranges"
    echo "   Expected 'low: 0.25' in confidence_threshold section"
else
    echo "✓ PyTorch config has constrained ranges"
fi

# Check for previous consensus cache (optional)
CACHE_FOUND=""
if [ -d "data/SD_02_SURF_FOOTAGE_PREPT/05_ANALYSED_DATA/POSE_EXPERIMENTS/results/runs/20251025_202539_final_full_pose_comparison/consensus_cache" ]; then
    CACHE_FOUND="yes"
    echo "✓ Found existing consensus cache (will be reused)"
else
    echo "ℹ️  No existing cache found (will generate fresh)"
fi

echo ""
echo "=== Starting Smoke Test ==="
echo ""
echo "Log file: smoke_test_full_suite.log"
echo "Detach with: Ctrl+B, then D"
echo ""
sleep 2

# Run in tmux for persistence
tmux new -s smoke_test -d "
cd $(pwd) && \
conda activate ${CONDA_DEFAULT_ENV} && \
python run_evaluation.py \
  --run-name 'smoke_test_full_suite_oct27' \
  --config configs/evaluation_config_production_optuna.yaml \
  --comparison-config configs/evaluation_config_production_comparison.yaml \
  --optuna-trials 5 \
  --optuna-max-clips 10 \
  --comparison-max-clips 10 \
  --eval-mode quick \
  2>&1 | tee smoke_test_full_suite.log
"

echo "✓ Smoke test started in tmux session 'smoke_test'"
echo ""
echo "Monitor progress:"
echo "  tail -f smoke_test_full_suite.log"
echo ""
echo "Attach to session:"
echo "  tmux attach -t smoke_test"
echo ""
echo "What to watch for:"
echo "  1. Consensus cache loading/generation (first few minutes)"
echo "  2. 'pytorch_pose' using confidence range 0.25-0.65"
echo "  3. Each model completing 5 Optuna trials"
echo "  4. COCO validation results"
echo "  5. Comparison phase completing"
echo ""
echo "Success indicators:"
echo "  • No crashes or OOM errors"
echo "  • All models report metrics to MLflow"
echo "  • PyTorch detection rate > 70%"
echo "  • Log ends with 'Evaluation complete'"
echo ""

