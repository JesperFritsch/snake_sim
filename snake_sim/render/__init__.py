import os
# This variable forces the window to be x11 style and not handled in waylands tiling system like regular windows on wayland
if os.environ.get("WAYLAND_DISPLAY"):
    os.environ["SDL_VIDEODRIVER"] = "x11"