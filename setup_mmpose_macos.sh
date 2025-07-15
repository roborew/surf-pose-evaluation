#!/bin/bash
# MMPose setup for macOS (Apple Silicon/Intel) - CPU/MPS only
# Adds MMPose to your existing environment_macos.yml setup

# Initialize conda for shell script use
eval "$(conda shell.bash hook)"

echo "🚀 Setting up MMPose for macOS (adding to existing environment)"

# Check if MMPose is already compiled and cached
if python -c "import mmpose; print('MMPose already available')" 2>/dev/null; then
    echo "✅ MMPose already compiled and available"
    echo "You can now create/update the surf_pose_eval environment:"
    echo "  conda env create -f environment_macos.yml"
    exit 0
fi

# Create minimal build environment matching your macOS setup
cat > environment_mmpose_build_macos.yml << 'EOF'
name: mmpose_build_macos
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  # Match your environment_macos.yml setup
  - python=3.9
  - numpy=1.24.3
  - pandas=2.0.3
  - matplotlib=3.7.2
  - jupyter=1.0.0
  - tqdm=4.65.0
  - pyyaml=6.0
  
  # PyTorch for macOS (CPU + Metal Performance Shaders)
  - pytorch::pytorch=2.0.1
  - pytorch::torchvision=0.15.2
  - pytorch::torchaudio=2.0.2
  
  - scikit-learn=1.3.0
  - scipy=1.11.1
  - seaborn=0.12.2
  
  - pip=23.2.1
  - pip:
      - opencv-python==4.8.0.76
      - pillow==10.0.0
      - rich==13.4.2
      - psutil==5.9.5
EOF

echo "📦 Creating MMPose build environment for macOS..."
conda env create -f environment_mmpose_build_macos.yml

# Activate and install OpenMMLab packages (CPU/MPS only)
conda activate mmpose_build_macos

echo "🔧 Installing OpenMMLab ecosystem for macOS..."
pip install -U openmim
mim install mmengine==0.8.4
mim install mmcv==2.0.1
mim install mmdet==3.1.0

# Store the current directory location
ORIGINAL_DIR=$(pwd)

echo "🏗️  Building MMPose from source for macOS (creates conda cache)..."
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

# Install from source - this creates the cached compilation
pip install -r requirements.txt
pip install -v -e .

# Navigate back to the original directory
cd "$ORIGINAL_DIR"

# Clean up build environment
conda env remove -n mmpose_build_macos -y
rm environment_mmpose_build_macos.yml

echo "✅ MMPose compilation complete and cached for macOS!"
echo ""
echo "🎯 Next steps:"
echo "1. Update your environment_macos.yml to include MMPose packages in pip section"
echo "2. Create/update environment: conda env create -f environment_macos.yml"
echo "3. MMPose will use the cached compilation from this build"

echo ""
echo "Add these lines to your environment_macos.yml pip section:"
echo "      # OpenMMLab MMPose (compiled and cached)"
echo "      - openmim>=0.3.7"
echo "      - mmengine==0.8.4"
echo "      - mmcv==2.0.1"
echo "      - mmdet==3.1.0"
echo "      - mmpose==1.1.0"

echo ""
echo "Testing MMPose availability:"
python -c "import mmpose; print('✅ MMPose successfully compiled and cached for macOS')" || echo "❌ MMPose compilation failed" 