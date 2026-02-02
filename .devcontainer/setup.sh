#!/bin/bash
set -e

echo "🚀 Setting up development environment..."

# Install system dependencies for robotics/visualization
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "📥 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Sync project dependencies
echo "🔧 Installing project dependencies with uv..."
uv sync

# Verify installation
echo "✅ Verifying installation..."
uv run python -c "import colmpc; import matplotlib; import meshcat; print('All core dependencies imported successfully!')"

echo "✨ Setup complete! You can now start developing."
echo "💡 Tip: Use 'uv run python your_script.py' to run scripts with the project environment."