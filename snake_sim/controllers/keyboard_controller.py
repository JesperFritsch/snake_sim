from typing import Set

from snake_sim.utils import is_headless

# This is a workaround for running the code in a headless environment, pynput requires a display
if is_headless():
    from pyvirtualdisplay import Display
    display = Display(visible=False, size=(1024, 768))
    display.start()

from pynput import keyboard

from snake_sim.snakes.manual_snake import ManualSnake

DEFAULT_KEYS_EXPLICIT = (
    {
        'w': 'explicit_up',
        's': 'explicit_down',
        'a': 'explicit_left',
        'd': 'explicit_right',
    },
    {
        'i': 'explicit_up',
        'k': 'explicit_down',
        'j': 'explicit_left',
        'l': 'explicit_right',
    },
    {
        't': 'explicit_up',
        'g': 'explicit_down',
        'f': 'explicit_left',
        'h': 'explicit_right',
    },
)

DEFAULT_KEYS_IMPLICIT = (
    {
        'a': 'implicit_left',
        's': 'implicit_right',
    },
    {
        'k': 'implicit_left',
        'l': 'implicit_right',
    },
    {
        'f': 'implicit_left',
        'g': 'implicit_right',
    },
)


class SingleSnakeController:
    def __init__(self, snake: ManualSnake, keys: dict) -> None:
        self.snake = snake
        self.keys = keys

    def handle_key(self, key):
        actions = []
        for bound_key, bound_action in self.keys.items():
            if key == keyboard.KeyCode.from_char(bound_key):
                actions.append(bound_action)
        if len(actions) == 1:
            self.snake.set_direction(actions[0])

    def get_action(self):
        return None

    def reset(self):
        pass


class ControllerCollection:
    def __init__(self) -> None:
        self.controllers: Set[SingleSnakeController] = set()

    def bind_controller(self, snake, controller_type='explicit'):
        controller_index = len(self.controllers)
        if controller_type == 'explicit':
            DEFAULT_KEYS = DEFAULT_KEYS_EXPLICIT
        elif controller_type == 'implicit':
            DEFAULT_KEYS = DEFAULT_KEYS_IMPLICIT
        else:
            raise ValueError(f"Invalid controller type: {controller_type}")
        controller = SingleSnakeController(snake, DEFAULT_KEYS[controller_index])
        self.controllers.add(controller)

    def on_press(self, key):
        for controller in self.controllers:
            controller.handle_key(key)

    def handle_controllers(self):
        keyboard.Listener(on_press=self.on_press).start()