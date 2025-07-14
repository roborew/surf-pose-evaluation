#!/bin/bash
# MMPose setup for macOS (Apple Silicon/Intel)
# Based on proven SurfAnalysis approach: build first, then make available in main environment

# Initialize conda for shell script use
eval "$(conda shell.bash hook)"

echo "Setting up MMPose for macOS - Phase 1: Build MMPose"

# Create dedicated MMPose environment for building
cat > environment_mmpose_build.yml << 'EOF'
name: mmpose_build
channels:
  - pytorch
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
  - scipy
  - seaborn
  # PyTorch for macOS (CPU + MPS)
  - pytorch::pytorch=2.0.1
  - pytorch::torchvision=0.15.2
  - pip:
      - opencv-python
      - pillow
      - pyyaml
      - fsspec
EOF

echo "Creating MMPose build environment..."
conda env create -f environment_mmpose_build.yml

# Activate and install OpenMMLab packages
conda activate mmpose_build

echo "Installing OpenMMLab ecosystem..."
pip install fsspec
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0" 

# Store the current directory location
ORIGINAL_DIR=$(pwd)

echo "Cloning and building MMPose from source..."
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

# Install from source (this creates the cached compilation)
pip install -r requirements.txt
pip install -v -e .

# Navigate back to the original directory
cd "$ORIGINAL_DIR"

echo "✅ Phase 1 complete: MMPose built and cached"

# Phase 2: Update main environment to include MMPose (now that it's compiled)
echo "Phase 2: Adding MMPose to main surf_pose_eval environment..."

conda activate surf_pose_eval

echo "Installing MMPose in main environment (using cached compilation)..."
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"

# Install MMPose from the compiled source
cd ../mmpose
pip install -v -e .
cd "$ORIGINAL_DIR"

# Clean up temporary environment file
rm environment_mmpose_build.yml

echo "✅ MMPose setup complete for macOS!"
echo ""
echo "Testing installation in main environment:"
conda activate surf_pose_eval
python -c "import torch; import mediapipe; import ultralytics; from mmpose.apis import MMPoseInferencer; print('All models available in main environment!')"

echo ""
echo "Ready to run evaluations:"
echo "  python evaluate_pose_models.py configs/evaluation_config_macos.yaml --models mmpose" 