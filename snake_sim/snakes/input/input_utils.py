import platform
import evdev
import evdev.ecodes as ec
import questionary

from enum import Enum
from dataclasses import dataclass, asdict


from snake_sim.snakes.input.input_provider_interface import IInputProvider
from snake_sim.snakes.input.evdev_pointer_provider import EvdevPointerProvider
from snake_sim.snakes.input.evdev_key_provider import EvdevKeyProvider
from snake_sim.snakes.input.evdev_gamepad_provider import EvdevGamepadProvider


PRECONFIGURED_KEY_MAPPINGS = {
    "Arrow keys": {
        ec.KEY_UP: (0, -1),
        ec.KEY_RIGHT: (1, 0),
        ec.KEY_DOWN: (0, 1),
        ec.KEY_LEFT: (-1, 0),
    },
    "WASD": {
        ec.KEY_W: (0, -1),
        ec.KEY_D: (1, 0),
        ec.KEY_S: (0, 1),
        ec.KEY_A: (-1, 0),
    },
    "IJKL": {
        ec.KEY_I: (0, -1),
        ec.KEY_L: (1, 0),
        ec.KEY_K: (0, 1),
        ec.KEY_J: (-1, 0),
    },
}


class InputType(Enum):
    POINTER = "pointer"
    GAMEPAD = "gamepad"
    KEYBOARD = "keyboard"


@dataclass
class AvailableInput:
    name: str
    input_type: InputType

    @classmethod
    def from_dict(cls: 'AvailableInput', dict):
        if "device_path" in dict:
            return LinuxInput(
                name=dict["name"],
                input_type=InputType(dict["input_type"]),
                device_path=dict["device_path"]
            )
        if "device_id" in dict:
            return WindowsInput(
                name=dict["name"],
                input_type=InputType(dict["input_type"]),
                device_id=dict["device_id"]
            )
        return cls(name=dict["name"], input_type=InputType(dict["input_type"]))


@dataclass
class LinuxInput(AvailableInput):
    device_path: str


@dataclass
class WindowsInput(AvailableInput):
    device_id: int


@dataclass
class InputConfig:
    input: AvailableInput
    params: dict

    def to_dict(self):
        d = asdict(self)
        d["input"]["input_type"] = self.input.input_type.value
        return d

def _discover_linux_inputs() -> list[LinuxInput]:
    """
    Discover compatible linux input devices
    """
    paths = evdev.list_devices()
    devices = [evdev.InputDevice(path) for path in paths]
    compatible_devices = []
    for device in devices:
        capabilities = device.capabilities(absinfo=False)
        if ec.EV_REL in capabilities:
            rel_events = capabilities[ec.EV_REL]
            if ec.REL_X in rel_events and ec.REL_Y in rel_events:
                compatible_devices.append(LinuxInput(
                    name=device.name,
                    input_type=InputType.POINTER,
                    device_path=device.path
                ))
        if ec.EV_KEY in capabilities:
            key_events = capabilities[ec.EV_KEY]
            if all(key in key_events for key in (ec.KEY_A, ec.KEY_UP)):
                compatible_devices.append(LinuxInput(
                    name=device.name,
                    input_type=InputType.KEYBOARD,
                    device_path=device.path
                ))
        if ec.EV_ABS in capabilities:
            abs_events = capabilities[ec.EV_ABS]
            key_events = capabilities.get(ec.EV_KEY, [])
            has_sticks = ec.ABS_X in abs_events and ec.ABS_Y in abs_events
            has_gamepad_btn = any(btn in key_events for btn in (ec.BTN_A, ec.BTN_SOUTH, ec.BTN_GAMEPAD))
            if has_sticks and has_gamepad_btn:
                compatible_devices.append(LinuxInput(
                    name=device.name,
                    input_type=InputType.GAMEPAD,
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


def create_input_provider(input_config: InputConfig) -> IInputProvider:
    """
    Create an input provider instance based on the provided input device information
    """
    if isinstance(input_config.input, LinuxInput):
        if input_config.input.input_type == InputType.POINTER:
            return EvdevPointerProvider(input_config.input.device_path)
        elif input_config.input.input_type == InputType.GAMEPAD:
            return EvdevGamepadProvider(input_config.input.device_path)
        elif input_config.input.input_type == InputType.KEYBOARD:
            return EvdevKeyProvider(input_config.input.device_path, input_config.params["key_mapping"])
    elif isinstance(input_config.input, WindowsInput):
        raise NotImplementedError("Windows pointer input provider is not implemented yet")
    else:
        raise ValueError(f"Unsupported input type: {input_config.input}")


def _pick_key_mapping(used_mappings: set) -> tuple[str, dict]:
    """Let the player pick a key mapping, excluding already-used ones."""
    available = {
        name: mapping
        for name, mapping in PRECONFIGURED_KEY_MAPPINGS.items()
        if name not in used_mappings
    }
    if not available:
        raise RuntimeError("No more key mappings available")

    choices = list(available.keys())
    selected = questionary.select(
        "Select key mapping:",
        choices=choices
    ).ask()
    return selected, available[selected]


def setup_player_input(num_players: int) -> list[InputConfig]:
    inputs = discover_inputs()
    player_configs = []
    claimed_devices = set()       # device paths that can't be reused (pointer/gamepad)
    used_key_mappings = set()     # key mapping names already assigned

    for i in range(num_players):
        print(f"\n--- Player {i + 1} setup ---")

        available = [
            inp for inp in inputs
            if inp.input_type == InputType.KEYBOARD
            or getattr(inp, 'device_path', None) not in claimed_devices
        ]
        if not available:
            raise RuntimeError(f"Not enough input devices for {num_players} players")

        choices = [f"{inp.name} ({inp.input_type.value})" for inp in available]
        selected_label = questionary.select(
            "Select an input device:",
            choices=choices
        ).ask()
        selected = available[choices.index(selected_label)]

        params = {}
        if selected.input_type == InputType.KEYBOARD:
            mapping_name, mapping = _pick_key_mapping(used_key_mappings)
            used_key_mappings.add(mapping_name)
            params["key_mapping"] = mapping
        else:
            # Pointer/gamepad — claim exclusively
            claimed_devices.add(selected.device_path)

        player_configs.append(InputConfig(input=selected, params=params))

    return player_configs



if __name__ == "__main__":
    input_config = setup_player_input(2)
    import json
    print(json.dumps(input_config, indent=2))
    print(input_config)