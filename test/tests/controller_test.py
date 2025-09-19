from snake_sim.controllers.keyboard_controller import ControllerCollection
from snake_sim.snakes.manual_snake import ManualSnake


snake = ManualSnake("A", 3)
controller = ControllerCollection()
controller.bind_controller(snake)
controller.handle_controllers()