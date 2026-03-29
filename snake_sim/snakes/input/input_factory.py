import evdev
import platform
from enum import Enum
from dataclasses import dataclass

from snake_sim.snakes.input.input_provider_interface import IInputProvider
from snake_sim.snakes.input.EvdevPointerProvider import EvdevPointerProvider

class InputType(Enum):
    POINTER = "pointer"
    GAMEPAD = "gamepad"
    KEYBOARD = "keyboard"


@dataclass
class AvailableInput:
    name: str
    input_type: InputType


@dataclass
class LinuxInput(AvailableInput):
    device_path: str


@dataclass
class WindowsInput(AvailableInput):
    device_id: int


def _discover_linux_inputs() -> list[LinuxInput]:
    """
    Discover compatible linux input devices
    """
    paths = evdev.list_devices()
    devices = [evdev.InputDevice(path) for path in paths]
    compatible_devices = []
    for device in devices:
        capabilities = device.capabilities()
        if evdev.ecodes.EV_REL in capabilities:
            rel_events = capabilities[evdev.ecodes.EV_REL]
            if evdev.ecodes.REL_X in rel_events and evdev.ecodes.REL_Y in rel_events:
                compatible_devices.append(LinuxInput(
                    name=device.name,
                    input_type=InputType.POINTER,
                    device_path=device.path
                ))
    return compatible_devices


def _discover_windows_inputs() -> list[WindowsInput]:
    """
    Discover compatible windows input devices
    """
    raise NotImplementedError("Windows input discovery is not implemented yet")


def discover_inputs() -> list[AvailableInput]:
    """
    Discover compatible input devices for the current platform
    """
    system = platform.system()
    if system == "Linux":
        return _discover_linux_inputs()
    elif system == "Windows":
        return _discover_windows_inputs()
    else:
        raise NotImplementedError(f"Input discovery is not implemented for platform {system}")


def create_input_provider(input: AvailableInput) -> IInputProvider:
    """
    Create an input provider instance based on the provided input device information
    """
    if input.input_type == InputType.POINTER:
        if isinstance(input, LinuxInput):
            return EvdevPointerProvider(input.device_path)
        elif isinstance(input, WindowsInput):
            raise NotImplementedError("Windows pointer input provider is not implemented yet")
    elif input.input_type == InputType.GAMEPAD:
        raise NotImplementedError("Gamepad input provider is not implemented yet")
    elif input.input_type == InputType.KEYBOARD:
        raise NotImplementedError("Keyboard input provider is not implemented yet")
    else:
        raise ValueError(f"Unsupported input type: {input.input_type}")


def ask_input():
    inputs = discover_inputs()

    import questionary
    import math
    choices = [f"{input.name} ({input.input_type.value})" for input in inputs]
    selected = questionary.select("Select an input device:", choices=choices).ask()
    selected_input = inputs[choices.index(selected)]
    input_provider = create_input_provider(selected_input)
    return input_provider


if __name__ == "__main__":
    inputs = discover_inputs()

    import questionary
    import math
    choices = [f"{input.name} ({input.input_type.value})" for input in inputs]
    selected = questionary.select("Select an input device:", choices=choices).ask()
    selected_input = inputs[choices.index(selected)]
    input_provider = create_input_provider(selected_input)
    print(f"Selected input provider: {input_provider}")

    while True:
        angle = input_provider.get_angle()
        print(f"\rAngle degrees: {math.degrees(angle): <8.2f}, radians: {angle: <8.2f}", end=""   )
