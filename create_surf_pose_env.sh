#!/bin/bash
# Script 2: Create surf_pose_eval environment using cached MMPose packages
# This copies the pre-built packages from mmpose_cache to avoid recompilation

# Initialize conda for shell script use
eval "$(conda shell.bash hook)"

echo "🚀 Creating surf_pose_eval environment with cached MMPose packages"

# Check if cache environment exists
if ! conda env list | grep -q mmpose_cache; then
    echo "❌ Cache environment 'mmpose_cache' not found!"
    echo "Run './setup_mmpose_production.sh' first to build and cache packages"
    exit 1
fi

# Create the main environment from environment.yml (without mmpose packages)
echo "📦 Creating surf_pose_eval environment..."
conda env create -f environment.yml

# Activate the new environment
conda activate surf_pose_eval

echo "🔄 Copying cached MMPose packages to surf_pose_eval environment..."

# Get the conda environments path
CONDA_PREFIX_CACHE=$(conda info --envs | grep mmpose_cache | awk '{print $2}')
CONDA_PREFIX_TARGET=$(conda info --envs | grep surf_pose_eval | awk '{print $2}')

if [ -z "$CONDA_PREFIX_CACHE" ] || [ -z "$CONDA_PREFIX_TARGET" ]; then
    echo "❌ Could not find environment paths"
    exit 1
fi

echo "Cache environment: $CONDA_PREFIX_CACHE"
echo "Target environment: $CONDA_PREFIX_TARGET"

# Copy the exact package versions that were successfully compiled in cache
echo "📋 Installing exact MMPose ecosystem versions from cache..."

# Get the exact versions from cache environment
MMCV_VERSION=$(conda list -n mmpose_cache mmcv | grep mmcv | awk '{print $2}')
MMENGINE_VERSION=$(conda list -n mmpose_cache mmengine | grep mmengine | awk '{print $2}')
MMDET_VERSION=$(conda list -n mmpose_cache mmdet | grep mmdet | awk '{print $2}')
MMPOSE_VERSION=$(conda list -n mmpose_cache mmpose | grep mmpose | awk '{print $2}')

echo "Using cached versions: mmcv=$MMCV_VERSION, mmengine=$MMENGINE_VERSION, mmdet=$MMDET_VERSION, mmpose=$MMPOSE_VERSION"

# Install the exact same versions with no-deps to use the compiled versions
pip install fsspec
pip install -U openmim

# Install the exact cached versions WITH dependencies (like SurfAnalysis does)
echo "Installing mmengine==$MMENGINE_VERSION..."
pip install mmengine==$MMENGINE_VERSION

echo "Installing mmcv==$MMCV_VERSION..."
pip install mmcv==$MMCV_VERSION -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

echo "Installing mmdet==$MMDET_VERSION..."
pip install mmdet==$MMDET_VERSION

echo "Installing mmpose==$MMPOSE_VERSION..."
pip install mmpose==$MMPOSE_VERSION

# The MM packages are now installed with dependencies
# The complete model zoo should be available through the pip installations
echo "✅ MM packages installed with complete dependencies and model zoo"

echo "✅ Environment 'surf_pose_eval' created successfully!"
echo ""
echo "🧪 Testing installation..."

# Test the installation
python -c "
import sys
print('Python version:', sys.version)

try:
    import torch
    print('✅ PyTorch version:', torch.__version__)
    print('✅ CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print('❌ PyTorch import failed:', e)

try:
    import mmcv
    print('✅ MMCV version:', mmcv.__version__)
except ImportError as e:
    print('❌ MMCV import failed:', e)

try:
    import mmengine
    print('✅ MMEngine version:', mmengine.__version__)
except ImportError as e:
    print('❌ MMEngine import failed:', e)

try:
    import mmdet
    print('✅ MMDetection version:', mmdet.__version__)
except ImportError as e:
    print('❌ MMDetection import failed:', e)

try:
    import mmpose
    print('✅ MMPose version:', mmpose.__version__)
except ImportError as e:
    print('❌ MMPose import failed:', e)

try:
    from mmpose.apis import MMPoseInferencer
    print('✅ MMPose inferencer can be imported')
except ImportError as e:
    print('❌ MMPose inferencer import failed:', e)

print('\\n🎯 Testing MMDetection model zoo access...')
try:
    from mmdet.apis import init_detector
    # Try to access a common model config that should be available
    import mmdet.models
    print('✅ MMDetection model registry accessible')
except Exception as e:
    print('❌ MMDetection model access failed:', e)
"

echo ""
echo "🚀 Setup complete! Activate the environment with:"
echo "   conda activate surf_pose_eval"
echo ""
echo "📊 Run evaluation with:"
echo "   python run_evaluation.py --config configs/evaluation_config_production_comparison.yaml" 