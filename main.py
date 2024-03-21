import sys
import json
import os
from snake_env import SnakeEnv, coord_op
from snakes.autoSnake import AutoSnake
import pygame

TILE_SIZE_PX = 40
GRID_WIDTH = 15
GRID_HEIGHT = 15
TILE_SIZE = 3

def drawGrid(surface):
    for y in range(0, int(GRID_HEIGHT)):
        for x in range(0, int(GRID_WIDTH)):
            rr = pygame.Rect((x*TILE_SIZE_PX, y*TILE_SIZE_PX), (TILE_SIZE_PX,TILE_SIZE_PX))
            pygame.draw.rect(surface, (120,120,120), rr)
            # if (x+y)%2 == 0:
            #     r = pygame.Rect((x*TILE_SIZE_PX, y*TILE_SIZE_PX), (TILE_SIZE_PX,TILE_SIZE_PX))
            #     pygame.draw.rect(surface,(93,216,228), r)
            # else:
            #     rr = pygame.Rect((x*TILE_SIZE_PX, y*TILE_SIZE_PX), (TILE_SIZE_PX,TILE_SIZE_PX))
            #     pygame.draw.rect(surface, (84,194,205), rr)

def draw_color_map(surface, color_map):
    for (x, y), color in color_map.items():
        r = pygame.Rect((x*TILE_SIZE_PX, y*TILE_SIZE_PX), (TILE_SIZE_PX,TILE_SIZE_PX))
        pygame.draw.rect(surface, color, r)

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

def print_saved_run(filename):
    run_dict = {}
    with open(filename) as run_json:
        run_dict = json.load(run_json)
    grid_height = run_dict['height']
    grid_width = run_dict['width']

    steps = run_dict['steps']
    for step in steps:
        step_data = steps[step]
        colors = step_data['colors']
        for i in range(0, len(colors), grid_width):
            print(colors[i:i+grid_width])

# def x3_frames_from_runfile(filename):
#     bg_color = (0,0,0)
#     run_dict = {}
#     frames = []
#     with open(filename) as run_json:
#         run_dict = json.load(run_json)
#     grid_height = run_dict['height']
#     grid_width = run_dict['width']
#     new_grid_h = grid_height * TILE_SIZE + 1
#     new_grid_w = grid_width * TILE_SIZE + 1
#     offset_x = 1
#     offset_y = 1
#     steps = run_dict['steps']
#     color_list = [bg_color] * ((new_grid_w) * (new_grid_h))
#     for step in steps:
#         step_data = steps[step]
#         state = step_data['state']
#         for i, color in enumerate(step_data['colors']):
#             if color in [list(SnakeEnv.COLOR_MAPPING[x]) for x in SnakeEnv.valid_tile_values]:
#                 o_coord = (i // grid_width, i % grid_width)
#                 x3_x, x3_y = coord_op(o_coord, (3, 3), '*')
#                 x3_x += offset_x
#                 x3_y += offset_y
#                 color_list[x3_y * (new_grid_w) + x3_x] = color
#         for i in range(3):
#             for snake_id in state:
#                 snake_data = state[snake_id]
#                 cur_coord = snake_data['current_coord']
#                 last_coord = snake_data['last_coord']
#                 if cur_coord != last_coord:
#                     s_dir = coord_op(cur_coord, last_coord, '-')
#                     l_x, l_y = last_coord
#                     s_x = l_x * TILE_SIZE + offset_x
#                     s_y = l_y * TILE_SIZE + offset_y
#                     add = coord_op(s_dir, (i, i), '*')
#                     fin_x, fin_y = coord_op((s_x, s_y), add, '+')
#                     print(add, (fin_x, fin_y))
#                     color_list[fin_y * (new_grid_w) + fin_x] = snake_data['b_color']
#             frames.append(color_list.copy())
#     return frames


def draw_frame(screen, width, height, frame):
    squaresize = TILE_SIZE_PX / TILE_SIZE
    for row in range(height):
        for col in range(width):
            color_index = row * width + col
            color = frame[color_index % len(frame)]
            x = col * squaresize
            y = row * squaresize
            pygame.draw.rect(screen, color, (x, y, squaresize, squaresize))

if __name__ == '__main__':
    DEFAULT_FILENAME = os.path.abspath(os.path.join(os.getcwd(), 'snake_run.json'))
    window_width = GRID_WIDTH * TILE_SIZE_PX
    window_height = GRID_HEIGHT * TILE_SIZE_PX
    snake_len = 5
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, 15)
    env.add_snake(AutoSnake('A', snake_len), (176, 27, 16), (125, 19, 11))
    # env.add_snake(AutoSnake('B', snake_len), (19, 44, 209), (8, 23, 120))
    # env.add_snake(AutoSnake('C', snake_len), (19, 212, 77), (10, 140, 49))
    # env.add_snake(AutoSnake('D', snake_len), (128, 3, 111), (199, 4, 173))
    VISUAL = False

    if VISUAL:
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((window_width, window_height), 0, 32)
        myfont = pygame.font.SysFont("monospace",16)
        surface = pygame.Surface(screen.get_size())
        surface = surface.convert()
        drawGrid(surface)
        ongoing = True
        while ongoing:
            handle_events()
            drawGrid(surface)
            if env.alive_snakes:
                env.update()
                color_map = env.get_color_map()
                draw_color_map(surface, color_map)
            else:
                print("GAME OVER")
                ongoing = False
            screen.blit(surface, (0,0))
            clock.tick(5)
            pygame.display.update()
    else:
        # window_width = GRID_WIDTH * TILE_SIZE_PX
        # window_height = GRID_HEIGHT * TILE_SIZE_PX
        env.generate_run(DEFAULT_FILENAME)
        # frames = x3_frames_from_runfile(DEFAULT_FILENAME)
        # pygame.init()
        # clock = pygame.time.Clock()
        # screen = pygame.display.set_mode((window_width, window_height), 0, 32)
        # myfont = pygame.font.SysFont("monospace",16)
        # surface = pygame.Surface(screen.get_size())
        # surface = surface.convert()
        # for frame in frames:
        #     handle_events()
        #     draw_frame(surface, GRID_WIDTH * TILE_SIZE + 1, GRID_HEIGHT * TILE_SIZE + 1, frame)
        #     screen.blit(surface, (0,0))
        #     clock.tick(5)
        #     pygame.display.update()

