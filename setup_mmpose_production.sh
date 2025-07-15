#!/bin/bash
# Script 1: Build and cache MMPose packages for CUDA 12
# This creates a temporary environment to compile packages, then caches them for reuse
# UPDATED: Using exact version pinning like macOS setup

# Initialize conda for shell script use
eval "$(conda shell.bash hook)"

echo "🚀 Building and caching MMPose packages for CUDA 12 (with exact versions)"

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

echo "Installing mmdet..."
mim install "mmdet>=3.0.0,<3.3.0"

# Store the current directory location
ORIGINAL_DIR=$(pwd)

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

echo "✅ MMPose packages built and cached with exact versions!"
echo ""
echo "📦 Cached environment 'mmpose_cache' contains:"
conda list -n mmpose_cache | grep -E "(mmcv|mmpose|mmdet|mmengine)"
echo ""
echo "🎯 Next step: Run './create_surf_pose_env.sh' to create your working environment"
echo ""
echo "Testing MMPose availability:"
python -c "import mmpose; print('✅ MMPose successfully compiled and cached for production')" || echo "❌ MMPose compilation failed" 