#!/bin/bash
# MMPose setup for production - replicating proven SurfAnalysis approach
# save as setup_mmpose_production.sh

# Initialize conda for shell script use
eval "$(conda shell.bash hook)"

echo "Setting up MMPose using proven SurfAnalysis approach..."

# Create minimal build environment (matching SurfAnalysis exactly)
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
  - pytorch
  - torchvision
  - gcc 
  - gxx
  - scikit-learn
  - scipy
  - seaborn
  - pip:
      - opencv-python
      - pillow
      - pyyaml
      - fsspec
EOF

echo "Creating MMPose build environment..."
conda env create -f environment_mmpose_build.yml

# Activate and install OpenMMLab packages (following SurfAnalysis pattern)
conda activate mmpose_build

echo "Installing PyTorch and dependencies..."
conda install pytorch torchvision -c pytorch
pip install fsspec

echo "Installing OpenMMLab ecosystem..."
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0" 

# Store the current directory location
ORIGINAL_DIR=$(pwd)

echo "Building MMPose from source (creates cached compilation)..."
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

echo "✅ MMPose built and cached successfully!"

# Now MMPose should be available in conda cache for the main environment
echo ""
echo "Phase 2: Installing MMPose in main surf_pose_eval environment..."

conda activate surf_pose_eval

echo "Installing MMPose dependencies in main environment..."
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"

# Install MMPose from the compiled source (using cached compilation)
cd ../mmpose
pip install -v -e .
cd "$ORIGINAL_DIR"

# Clean up temporary environment and file
conda env remove -n mmpose_build -y
rm environment_mmpose_build.yml

echo "✅ MMPose setup complete for production!"
echo ""
echo "Testing installation in main environment:"
conda activate surf_pose_eval
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import mmcv; print('MMCV version:', mmcv.__version__)"
python -c "from mmpose.apis import MMPoseInferencer; print('✅ MMPose ready!')"

echo ""
echo "Ready to run evaluations:"
echo "  python evaluate_pose_models.py configs/evaluation_config_production_optuna.yaml --models mmpose" 