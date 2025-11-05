# Snake Sim

A snake simulation environment with support for deep learning strategies.

## Installation

```bash
pip install -e .
```

This installs PyTorch with **CPU support** by default.

### For CUDA/GPU Support

If you want to use GPU acceleration for deep learning strategies:

```bash
# First, install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Then install snake-sim (will use the CUDA PyTorch you just installed)
pip install -e .
```

## Building from Source

If you need to build the C++ extensions manually:

```bash
python setup.py build
python setup.py build_ext --inplace
```

## Running on Headless Servers

To run the simulation on a headless server, `xvfb` needs to be installed:

```bash
sudo apt update && sudo apt upgrade
sudo apt install xvfb
```

## Usage

After installation, you can run the snake simulation:

```bash
snake-sim
```