#!/bin/bash
# MMPose compilation script - creates conda cache for reuse
# Replicates the proven approach where SurfAnalysis compiled MMPose first

# Initialize conda for shell script use
eval "$(conda shell.bash hook)"

echo "🚀 Setting up MMPose compilation (creates conda cache for surf_pose_eval)"

# Check if MMPose is already compiled and cached
if python -c "import mmpose; print('MMPose already available')" 2>/dev/null; then
    echo "✅ MMPose already compiled and available"
    echo "You can now create the surf_pose_eval environment:"
    echo "  conda env create -f environment.yml"
    exit 0
fi

# Create minimal build environment (matching proven SurfAnalysis approach)
cat > environment_mmpose_build.yml << 'EOF'
name: mmpose_build
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - numpy
  - pandas
  - matplotlib
  - jupyter
  - tqdm
  - pytorch=2.1.*
  - torchvision=0.16.*
  - pytorch-cuda=12.1
  - gcc_linux-64=9.*
  - gxx_linux-64=9.*
  - scikit-learn
  - scipy
  - seaborn
  - pip:
      - opencv-python
      - pillow
      - pyyaml
      - fsspec
EOF

echo "📦 Creating MMPose build environment..."
conda env create -f environment_mmpose_build.yml

# Activate and install OpenMMLab packages
conda activate mmpose_build

# Use conda-installed GCC and set C++17 compilation flags
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CXXFLAGS="-std=c++17"
export CPPFLAGS="-std=c++17"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

echo "🔧 Installing OpenMMLab ecosystem..."
pip install fsspec
pip install -U openmim

# Install pre-compiled MMCV (avoid compilation issues)
echo "Installing pre-compiled MMCV..."
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

mim install mmengine==0.8.4
mim install mmdet==3.1.0

# Store the current directory location
ORIGINAL_DIR=$(pwd)

echo "🏗️  Building MMPose from source (creates conda cache)..."
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

# Install from source with proper compilation flags - this creates the cached compilation
pip install -r requirements.txt
pip install -v -e .

# Navigate back to the original directory
cd "$ORIGINAL_DIR"

# Clean up build environment
conda env remove -n mmpose_build -y
rm environment_mmpose_build.yml

echo "✅ MMPose compilation complete and cached!"
echo ""
echo "🎯 Next steps:"
echo "1. Create surf_pose_eval environment: conda env create -f environment.yml"
echo "2. The environment.yml will use the cached MMPose packages"
echo "3. Run evaluations: python evaluate_pose_models.py ..."

echo ""
echo "Testing MMPose availability in base environment:"
python -c "import mmpose; print('✅ MMPose successfully compiled and cached')" || echo "❌ MMPose compilation failed" 