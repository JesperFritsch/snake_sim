# Snake Sim

A snake simulation environment with support for deep learning snakes.

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


## Interactive Controls (Render Loop)

When using the `PygameRenderer`, the render loop listens to window-focused key
events (works on Wayland, X11, Windows, macOS). Ensure the Pygame window has
focus for input to be registered.

Key bindings:

    RIGHT Arrow      step forward / continuous while held
    LEFT Arrow       step backward / continuous while held
    CTRL (with arrow) fast mode (20x base FPS)
    SHIFT            increase frame step size to 10
    SPACE            toggle pause (pause auto-render; manual stepping OK)
    ENTER            save current state (if state builder attached)
    CTRL + C         quit render loop

Headless mode: Input is disabled (Dummy provider). Use command-line or IPC
mechanisms if you need remote control.

Wayland note: Previous global key capture (pynput) was replaced by window
events to avoid unreliable behavior under Wayland compositors.

### Terminal Renderer Input

When using the terminal renderer (no Pygame window), a terminal input provider
captures keys directly from stdin (TTY):

    RIGHT / LEFT     arrow keys (escape sequences) for forward/back stepping
    SPACE            pause/unpause
    ENTER            save state
    c (with CTRL)    quit (CTRL+C as usual)

Provider selection order:

1. Explicit provider passed to `RenderLoop` constructor (if given)
2. If renderer is `TerminalRenderer` -> Terminal provider
3. If a Pygame window environment exists -> Pygame provider
4. If stdin is a TTY -> Terminal provider
5. Fallback -> Dummy (no input)

Terminal specifics:

    RIGHT / LEFT    arrow keys to step forward/back
    SPACE           pause/unpause
    ENTER           save state
    q               quit render loop
    ESC             quit render loop

Fast (CTRL) and multi-step (SHIFT) modifiers rely on terminal key repeat; true
release events aren't available so holds are not fully simulated.
