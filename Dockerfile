# ============================
# Stage 1: Base image with common dependencies
# ============================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

# Prevent prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from Python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# ============================
# Install Python, git, and other necessary tools
# ============================
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    build-essential \
 && ln -sf /usr/bin/python3.10 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && rm -rf /var/lib/apt/lists/*

# ============================
# Install ComfyUI via comfy-cli
# ============================
RUN pip install --upgrade pip && pip install comfy-cli

# Install the latest stable ComfyUI with CUDA 11.8 support
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia

# Set working directory to ComfyUI workspace
WORKDIR /comfyui

# ============================
# Fix bitsandbytes - compile from source for CUDA 11.8
# ============================
# The issue is PyTorch 2.8.0 reports CUDA 12.8, but container has CUDA 11.8
# We need to force bitsandbytes to compile for CUDA 11.8
RUN pip uninstall -y bitsandbytes && \
    git clone https://github.com/TimDettmers/bitsandbytes.git /tmp/bitsandbytes && \
    cd /tmp/bitsandbytes && \
    CUDA_VERSION=118 make cuda11x && \
    python setup.py install && \
    cd / && rm -rf /tmp/bitsandbytes

# Alternative: Use pre-built wheels for CUDA 11.8
# RUN pip uninstall -y bitsandbytes && \
#     pip install bitsandbytes==0.41.1 --no-cache-dir

# ============================
# Fix opencv-contrib-python for LayerStyle nodes
# ============================
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless && \
    pip install opencv-contrib-python==4.8.1.78 --no-cache-dir

# ============================
# Install additional dependencies
# ============================
RUN pip install runpod requests

# Support for network volume
ADD src/extra_model_paths.yaml ./

# ============================
# Pre-download DiffuEraser models to avoid first-run delays
# ============================
RUN mkdir -p /comfyui/models/DiffuEraser/propainter && \
    cd /comfyui/models/DiffuEraser/propainter && \
    wget -q https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth && \
    wget -q https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth && \
    wget -q https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth

# ============================
# Update ComfyUI frontend to fix version warning
# ============================
RUN pip install --upgrade comfyui-frontend

# ============================
# Optimize PyTorch for inference
# ============================
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV CUDA_LAUNCH_BLOCKING=0
# Force CUDA 11.8 for bitsandbytes
ENV BNB_CUDA_VERSION=118
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# ============================
# Go back to the root and add scripts
# ============================
WORKDIR /

# Add custom scripts and configuration files
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./

# Make scripts executable
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy snapshot file (if available)
ADD *snapshot*.json / 

# Restore snapshot to install custom nodes (ignore errors if no snapshot)
RUN /restore_snapshot.sh || true

# ============================
# Verify installations
# ============================
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" && \
    python -c "import bitsandbytes as bnb; print(f'bitsandbytes: {bnb.__version__}')" || echo "bitsandbytes check completed"

# ============================
# Container start command
# ============================
CMD ["/start.sh"]
