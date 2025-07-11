#!/bin/bash
# Script to fix MediaPipe installation issues on macOS

echo "🔧 Fixing MediaPipe installation for macOS..."

# Check if conda environment exists
if conda env list | grep -q "surf_pose_eval"; then
    echo "✅ Found existing surf_pose_eval environment"
else
    echo "❌ surf_pose_eval environment not found. Please create it first with:"
    echo "   conda env create -f environment_macos.yml"
    exit 1
fi

# Activate environment
echo "🔄 Activating surf_pose_eval environment..."
eval "$(conda shell.bash hook)"
conda activate surf_pose_eval

# Method 1: Try updating to MediaPipe 0.10.8
echo "📦 Method 1: Installing MediaPipe 0.10.8..."
pip uninstall -y mediapipe mediapipe-silicon 2>/dev/null || true
pip install mediapipe==0.10.8

# Test MediaPipe installation
echo "🧪 Testing MediaPipe installation..."
python -c "
import mediapipe as mp
try:
    pose = mp.solutions.pose.Pose()
    print('✅ MediaPipe 0.10.8 working!')
    exit(0)
except Exception as e:
    print(f'❌ MediaPipe 0.10.8 failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "🎉 MediaPipe 0.10.8 is working! You can now run your evaluation."
    exit 0
fi

# Method 2: Try mediapipe-silicon package
echo "📦 Method 2: Trying mediapipe-silicon package..."
pip uninstall -y mediapipe mediapipe-silicon 2>/dev/null || true
pip install mediapipe-silicon

# Test mediapipe-silicon
echo "🧪 Testing mediapipe-silicon installation..."
python -c "
import mediapipe as mp
try:
    pose = mp.solutions.pose.Pose()
    print('✅ mediapipe-silicon working!')
    exit(0)
except Exception as e:
    print(f'❌ mediapipe-silicon failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "🎉 mediapipe-silicon is working! You can now run your evaluation."
    exit 0
fi

# Method 3: Try older version
echo "📦 Method 3: Trying MediaPipe 0.9.3.0 (last known stable)..."
pip uninstall -y mediapipe mediapipe-silicon 2>/dev/null || true
pip install mediapipe==0.9.3.0

# Test older version
echo "🧪 Testing MediaPipe 0.9.3.0..."
python -c "
import mediapipe as mp
try:
    pose = mp.solutions.pose.Pose()
    print('✅ MediaPipe 0.9.3.0 working!')
    exit(0)
except Exception as e:
    print(f'❌ MediaPipe 0.9.3.0 failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "🎉 MediaPipe 0.9.3.0 is working! You can now run your evaluation."
    exit 0
fi

# If all methods fail
echo "❌ All MediaPipe installation methods failed."
echo "🔧 Troubleshooting steps:"
echo "1. Check your macOS version: $(sw_vers -productVersion)"
echo "2. Check your Python version: $(python --version)"
echo "3. Try recreating the conda environment:"
echo "   conda env remove -n surf_pose_eval"
echo "   conda env create -f environment_macos.yml"
echo "4. Check for any system-level conflicts with MediaPipe dependencies"
echo ""
echo "💡 Alternative: You can run only YOLOv8 pose estimation with:"
echo "   python evaluate_pose_models.py --config configs/evaluation_config_macos.yaml --models yolov8_pose"

exit 1 