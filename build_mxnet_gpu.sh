#!/usr/bin/env bash
set -e

echo "=== MXNet GPU Build Script (CUDA 12.x + Python Virtual Env) ==="

# Step 1: Prerequisites
echo "[1/7] Installing Python dependencies..."
pip install -U setuptools wheel numpy cython requests opencv-python

# Step 2: Clone MXNet source
echo "[2/7] Cloning MXNet..."
if [ ! -d "mxnet" ]; then
    git clone --recursive https://github.com/apache/mxnet.git
fi
cd mxnet
git checkout v1.x

# Step 3: Prepare build directory
echo "[3/7] Creating build directory..."
mkdir -p build && cd build

# Step 4: Configure CMake
echo "[4/7] Configuring CMake for CUDA 12.x..."
cmake .. \
  -DUSE_CUDA=1 \
  -DUSE_CUDNN=1 \
  -DUSE_NCCL=0 \
  -DUSE_MKLDNN=1 \
  -DUSE_OPENCV=1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2

# Step 5: Compile MXNet core
echo "[5/7] Building MXNet (this may take 15–30 minutes)..."
make -j$(nproc)

# Step 6: Build Python wheel
echo "[6/7] Building Python wheel..."
cd ../python
pip install -r requirements.txt
python setup.py bdist_wheel

# Step 7: Install and verify
echo "[7/7] Installing MXNet wheel..."
pip install dist/mxnet-*.whl

echo "=== Build complete! Verifying installation... ==="
python - <<'EOF'
import mxnet as mx
print("MXNet version:", mx.__version__)
print("Detected GPUs:", mx.context.num_gpus())
print("Runtime features:", mx.runtime.Features())
EOF

echo "✅ MXNet GPU build complete. You can now run your project with --gpu 0."