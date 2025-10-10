# ================================
# Stage 1: Base image with common dependencies
# ================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevent prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# ================================
# System setup
# ================================
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# ================================
# Python setup
# ================================
RUN pip install --upgrade pip setuptools wheel

# ✅ Install CUDA-compatible PyTorch stack
RUN pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ✅ Fix bitsandbytes GPU binding issue (supports CUDA 11.8)
RUN pip install bitsandbytes==0.43.1

# ✅ Install other essentials
RUN pip install runpod requests opencv-contrib-python==4.8.1.78

# ================================
# Install ComfyUI via comfy-cli
# ================================
RUN pip install comfy-cli && /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia

WORKDIR /comfyui

# ================================
# Add extra model paths
# ================================
ADD src/extra_model_paths.yaml ./

# ================================
# Add scripts and restore snapshot
# ================================
WORKDIR /
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh
ADD *snapshot*.json /  # optional snapshot for custom nodes
RUN /restore_snapshot.sh || true

# ================================
# Environment Fixes
# ================================
# Ensures CUDA libraries resolve properly
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Optional optimization flags
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TORCH_CUDNN_V8_API_ENABLED=1

# ================================
# Start container
# ================================
CMD ["/start.sh"]
