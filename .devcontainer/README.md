# Development Container Setup

This directory contains configurations for developing the Object Following OCP project in a containerized environment.

## 📁 Files Included

- **`devcontainer.json`**: Basic setup using Microsoft's Python base image
- **`devcontainer-with-dockerfile.json`**: Alternative using custom Dockerfile (rename to `devcontainer.json` to use)
- **`Dockerfile`**: Custom image with pre-installed system dependencies
- **`setup.sh`**: Post-creation script that installs project dependencies with `uv`

## 🚀 Quick Start

1. Install [Docker](https://www.docker.com/products/docker-desktop) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open this project in VS Code
4. When prompted (or via Command Palette: "Reopen in Container"), select "Reopen in Container"
5. Wait for the container to build and dependencies to install

## 🎯 What's Included

### System Dependencies
- Build tools (gcc, cmake, etc.)
- OpenGL libraries for visualization
- Graphics libraries for meshcat/matplotlib

### Python Environment
- Python 3.11
- `uv` package manager (fast dependency management)
- All project dependencies from `pyproject.toml`

### VS Code Extensions
- Python language support
- Pylance (type checking)
- Ruff (linting/formatting)
- Jupyter notebook support
- Auto docstring generator

## 🔧 Configuration Options

### Display/Visualization Support
The container is configured to support GUI applications (meshcat viewer, matplotlib):
- X11 socket is mounted from host
- `DISPLAY` environment variable is forwarded
- Network host mode for easy access to web-based visualizations

### Using the Custom Dockerfile
If you want more control or need to modify system dependencies:
1. Rename `devcontainer-with-dockerfile.json` to `devcontainer.json`
2. Edit `Dockerfile` as needed
3. Rebuild the container

## 💡 Usage Tips

### Running Python Scripts
```bash
# Inside the container
uv run python your_script.py
```

### Adding Dependencies
Edit `pyproject.toml` and run:
```bash
uv sync
```

### Accessing Meshcat Viewer
Meshcat typically runs on `http://localhost:7000` - accessible from your host browser thanks to `--network=host`.

### Jupyter Notebooks
The Jupyter extension is pre-installed. Open any `.ipynb` file and select the project's virtual environment as the kernel.

## 🐛 Troubleshooting

### Display Issues
If visualization doesn't work:
1. On Linux, run `xhost +local:docker` on your host
2. Check that `DISPLAY` is set correctly in your host environment

### Permission Issues
The container runs as the `vscode` user. If you encounter permission issues:
```bash
sudo chown -R vscode:vscode /workspaces/object_following_ocp
```

### PyTorch CPU Index
The project uses PyTorch CPU-only builds. If you need GPU support, modify the `[[tool.uv.index]]` section in `pyproject.toml`.