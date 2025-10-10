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
# Fix bitsandbytes for CUDA 11.8
# ============================
RUN pip uninstall -y bitsandbytes && \
    pip install bitsandbytes --no-cache-dir

# ============================
# Fix opencv-contrib-python for LayerStyle nodes
# ============================
RUN pip uninstall -y opencv-python opencv-contrib-python && \
    pip install opencv-contrib-python --no-cache-dir

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
# Optimize PyTorch for inference
# ============================
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV CUDA_LAUNCH_BLOCKING=0

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
# Warm up: Pre-load models on build (optional but recommended)
# ============================
# This can significantly reduce first-run time
# Uncomment if you want to pre-warm the models during build
# RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || true

# ============================
# Container start command
# ============================
CMD ["/start.sh"]
