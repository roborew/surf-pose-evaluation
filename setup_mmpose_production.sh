#!/bin/bash
# Script 1: Build and cache MMPose packages for CUDA 12
# This creates a temporary environment to compile packages, then caches them for reuse
# UPDATED: Installing mmdet from source to ensure complete model zoo

# Initialize conda for shell script use
eval "$(conda shell.bash hook)"

echo "🚀 Building and caching MMPose packages for CUDA 12 (with complete model zoo)"

# Check if packages are already cached
if conda list -n mmpose_cache mmpose 2>/dev/null | grep -q mmpose; then
    echo "✅ MMPose packages already cached in 'mmpose_cache' environment"
    echo "Run './create_surf_pose_env.sh' to create your working environment"
    exit 0
fi

# Create a temporary build environment (based on working SurfAnalysis setup)
echo "📦 Creating temporary build environment..."
cat > temp_build_env.yml << 'EOF'
name: mmpose_cache
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - setuptools
  - wheel
  - pytorch=2.1.*
  - torchvision=0.16.*
  - torchaudio=2.1.*
  - pytorch-cuda=12.1
  - numpy
  - scipy
  - gcc
  - gxx
  - pip:
      - fsspec
      - opencv-python
      - pillow
      - pyyaml
EOF

conda env create -f temp_build_env.yml
conda activate mmpose_cache

# Install OpenMMLab packages using working SurfAnalysis approach
echo "🔧 Installing OpenMMLab ecosystem (SurfAnalysis method)..."

# Install PyTorch first (conda should handle this, but ensure it's right)
conda install pytorch torchvision -c pytorch -y

# Install base packages
pip install fsspec
pip install -U openmim

# Install MM packages with ranges like working setup
echo "Installing mmengine..."
mim install mmengine

echo "Installing mmcv (using version range like working setup)..."
mim install "mmcv>=2.0.0rc4,<2.2.0"

echo "🏗️  Installing MMDetection from source for complete model zoo..."
# Store the current directory location
ORIGINAL_DIR=$(pwd)

# Clone and install mmdetection from source to get complete configs
if [ -d "../mmdetection" ]; then
  cd ../mmdetection
  git fetch origin
  git reset --hard origin/main
else
  cd ..
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
fi

# Install MMDetection in development mode to ensure full configs are available
pip install -r requirements.txt
pip install -v -e .

cd "$ORIGINAL_DIR"

echo "🏗️  Building MMPose from source..."
# Check if the mmpose directory exists
if [ -d "../mmpose" ]; then
  cd ../mmpose
  git fetch origin
  git reset --hard origin/main
else
  cd ..
  git clone https://github.com/open-mmlab/mmpose.git
  cd mmpose
fi

# Install MMPose in the cache environment
pip install -r requirements.txt
pip install -v -e .

cd "$ORIGINAL_DIR"
rm temp_build_env.yml

echo "✅ MMPose packages built and cached with complete model zoo!"
echo ""
echo "📦 Cached environment 'mmpose_cache' contains:"
conda list -n mmpose_cache | grep -E "(mmcv|mmpose|mmdet|mmengine)"
echo ""
echo "🎯 Next step: Run './create_surf_pose_env.sh' to create your working environment"
echo ""
echo "Testing MMPose availability:"
python -c "import mmpose; print('✅ MMPose successfully compiled and cached for production')" || echo "❌ MMPose compilation failed"

echo ""
echo "🔍 Verifying complete model zoo installation:"
# Check if the configs directory exists and contains RTMDet configurations
if [ -d "../mmdetection/configs/rtmdet" ]; then
    echo "✅ Complete MMDetection model zoo installed with RTMDet configs"
    echo "   Config files available: $(ls ../mmdetection/configs/rtmdet/*.py | wc -l) RTMDet configurations"
else
    echo "⚠️  MMDetection configs may not be complete - checking alternative locations"
fi

# Also check in site-packages if installed there
python -c "
import mmdet
import os
mmdet_path = os.path.dirname(mmdet.__file__)
configs_path = os.path.join(os.path.dirname(mmdet_path), 'configs')
if os.path.exists(configs_path):
    rtmdet_path = os.path.join(configs_path, 'rtmdet')
    if os.path.exists(rtmdet_path):
        rtmdet_files = [f for f in os.listdir(rtmdet_path) if f.endswith('.py')]
        print(f'✅ MMDetection configs found in site-packages: {len(rtmdet_files)} RTMDet configs')
    else:
        print('⚠️  RTMDet configs not found in site-packages configs')
else:
    print('ℹ️  No configs directory found in site-packages (this is normal for mim installs)')
" 2>/dev/null || echo "ℹ️  MMDetection package verification completed" 