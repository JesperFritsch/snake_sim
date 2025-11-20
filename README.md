# Snake Sim

A snake simulation environment with support for deep learning snakes.

## Installation (use -e if developing)

```bash
pip install -e .
```

This installs PyTorch with **CPU support** by default.

### For CUDA/GPU Support

If you want to use GPU acceleration for deep RL snake:


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


## Interactive Controls (Render Loop)

When running the simulation, you can control the playback using your keyboard. The controls work in both graphical window mode and terminal mode.

### Key Controls:

- **Right Arrow**: Step forward one frame, or hold to play continuously
- **Left Arrow**: Step backward one frame, or hold to rewind
- **Ctrl + Arrow**: Fast forward/rewind (20x speed)
- **Shift + Arrow**: Jump forward/backward 10 frames at a time
- **Spacebar**: Pause/unpause the automatic playback
- **Enter**: Save the current state (if supported)
- **q**, **Esc**, or **Ctrl + C**: Quit the simulation

Make sure the window is active (clicked on) in graphical mode for keys to register. In terminal mode, fast and multi-step controls depend on your keyboard's repeat rate.

If running without a display (headless), controls are disabled. Use command-line options or remote control instead.

Note: On some systems like Wayland, the controls work better with window focus.

## Example Commands

Here are some common ways to run the snake simulation:

```bash
# Play a previously recorded simulation file
snake-sim play-file runs/my_simulation.json

# Run a live simulation with terminal rendering
snake-sim compute --renderer terminal

# Run a live simulation with graphical window (default)
snake-sim compute --renderer window --fps 30

# Run with verbose output and custom grid size
snake-sim compute --verbose --grid-size 20

# Record a simulation to a specific file
snake-sim compute --record-file my_run.json
```
