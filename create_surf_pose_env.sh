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

echo "🔧 Copying MMPose packages from cache..."

# Get the package locations from cache environment
CACHE_ENV_PATH=$(conda info --envs | grep mmpose_cache | awk '{print $NF}')
TARGET_ENV_PATH=$(conda info --envs | grep surf_pose_eval | awk '{print $NF}')

echo "Cache environment: $CACHE_ENV_PATH"
echo "Target environment: $TARGET_ENV_PATH"

# Copy the installed packages from cache to target environment
echo "📚 Installing cached OpenMMLab packages..."

# Method 1: Use conda to clone specific packages
conda activate mmpose_cache
MMCV_VERSION=$(python -c "import mmcv; print(mmcv.__version__)" 2>/dev/null)
MMENGINE_VERSION=$(python -c "import mmengine; print(mmengine.__version__)" 2>/dev/null)
MMDET_VERSION=$(python -c "import mmdet; print(mmdet.__version__)" 2>/dev/null)

conda activate surf_pose_eval

# Install the exact versions found in cache
if [ ! -z "$MMCV_VERSION" ]; then
    echo "Installing mmcv==$MMCV_VERSION"
    pip install mmcv==$MMCV_VERSION -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
fi

if [ ! -z "$MMENGINE_VERSION" ]; then
    echo "Installing mmengine==$MMENGINE_VERSION"
    pip install mmengine==$MMENGINE_VERSION
fi

if [ ! -z "$MMDET_VERSION" ]; then
    echo "Installing mmdet==$MMDET_VERSION"
    pip install mmdet==$MMDET_VERSION
fi

# For MMPose, we need to copy the development installation
echo "🏗️  Setting up MMPose development installation..."
ORIGINAL_DIR=$(pwd)

# Check if mmpose source exists
if [ -d "../mmpose" ]; then
    cd ../mmpose
    pip install -v -e .
else
    echo "❌ MMPose source directory not found at ../mmpose"
    echo "Clone MMPose manually or run setup_mmpose_production.sh again"
    cd "$ORIGINAL_DIR"
    exit 1
fi

cd "$ORIGINAL_DIR"

echo "✅ surf_pose_eval environment ready!"
echo ""
echo "🎯 To use the environment:"
echo "  conda activate surf_pose_eval"
echo "  python evaluate_pose_models.py ..."
echo ""
echo "Testing installation:"
python -c "
try:
    import mmcv, mmengine, mmdet, mmpose
    print('✅ All MMPose packages successfully installed')
    print(f'  mmcv: {mmcv.__version__}')
    print(f'  mmengine: {mmengine.__version__}')
    print(f'  mmdet: {mmdet.__version__}')
    print(f'  mmpose: {mmpose.__version__}')
except Exception as e:
    print(f'❌ Installation test failed: {e}')
" 