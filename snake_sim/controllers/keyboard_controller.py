from typing import Set
from pynput import keyboard

from snake_sim.snakes.manual_snake import ManualSnake

DEFAULT_KEYS = (
    {
        'a': 'left',
        's': 'right',
    },
    {
        'k': 'left',
        'l': 'right',
    },
    {
        'v': 'left',
        'b': 'right',
    },
)


class SingleSnakeController:
    def __init__(self, snake: ManualSnake, keys: dict):
        self.snake = snake
        self.keys = keys

    def handle_key(self, key):
        actions = []
        for bound_key, bound_action in self.keys.items():
            if key == keyboard.KeyCode.from_char(bound_key):
                actions.append(bound_action)
        if len(actions) == 1:
            self.snake.set_direction_choice(actions[0])

    def get_action(self):
        return None

    def reset(self):
        pass


class ControllerCollection:
    def __init__(self) -> None:
        self.controllers: Set[SingleSnakeController] = set()

    def bind_controller(self, snake):
        controller_index = len(self.controllers)
        controller = SingleSnakeController(snake, DEFAULT_KEYS[controller_index])
        self.controllers.add(controller)

    def on_press(self, key):
        for controller in self.controllers:
            controller.handle_key(key)

    def handle_controllers(self):
        keyboard.Listener(on_press=self.on_press).start()