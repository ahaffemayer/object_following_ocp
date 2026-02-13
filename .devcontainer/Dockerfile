FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 1. Set CUDA architecture for RTX 5080 (Compute Capability 12.0)
# We include 12.0 and the PTX (forward compatible) instructions
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV FORCE_CUDA="1"

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    build-essential \
    wget \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# 2. Install PyTorch Nightly for CUDA 12.8 support
# Stable builds often lag behind new hardware; Nightly is required for SM 12.0
RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install common Python packages
RUN pip install ipykernel matplotlib pandas numpy scikit-learn tensorboard

# Install Ruff for linting and formatting
RUN pip install --no-cache-dir ruff

# 3. Clone and install cuRobo
# The --no-cache-dir ensures we don't use old, incorrectly compiled wheels
WORKDIR /workspace
RUN git clone https://github.com/NVlabs/curobo.git && \
    cd curobo && \
    pip install . --no-build-isolation --no-cache-dir

WORKDIR /workspace