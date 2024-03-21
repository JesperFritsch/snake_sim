import pygame
import json
import sys

from ..snake_env import SnakeEnv
from ..main import (
    GRID_HEIGHT,
    GRID_WIDTH,
    TILE_SIZE_PX,
    TILE_SIZE,
    coord_op
)

def x3_frames_from_runfile(filename):
    bg_color = (0,0,0)
    run_dict = {}
    frames = []
    with open(filename) as run_json:
        run_dict = json.load(run_json)
    grid_height = run_dict['height']
    grid_width = run_dict['width']
    new_grid_h = grid_height * TILE_SIZE + 1
    new_grid_w = grid_width * TILE_SIZE + 1
    offset_x = 1
    offset_y = 1
    steps = run_dict['steps']
    color_list = [bg_color] * ((new_grid_w) * (new_grid_h))
    for step in steps:
        step_data = steps[step]
        state = step_data['state']
        for i, color in enumerate(step_data['colors']):
            if color in [list(SnakeEnv.COLOR_MAPPING[x]) for x in SnakeEnv.valid_tile_values]:
                o_coord = (i // grid_width, i % grid_width)
                x3_x, x3_y = coord_op(o_coord, (3, 3), '*')
                x3_x += offset_x
                x3_y += offset_y
                color_list[x3_y * (new_grid_w) + x3_x] = color
        for i in range(3):
            for snake_id in state:
                snake_data = state[snake_id]
                cur_coord = snake_data['current_coord']
                last_coord = snake_data['last_coord']
                if cur_coord != last_coord:
                    s_dir = coord_op(cur_coord, last_coord, '-')
                    l_x, l_y = last_coord
                    s_x = l_x * TILE_SIZE + offset_x
                    s_y = l_y * TILE_SIZE + offset_y
                    add = coord_op(s_dir, (i, i), '*')
                    fin_x, fin_y = coord_op((s_x, s_y), add, '+')
                    print(add, (fin_x, fin_y))
                    color_list[fin_y * (new_grid_w) + fin_x] = snake_data['b_color']
            frames.append(color_list.copy())
    return frames

def draw_frame(screen, width, height, frame):
    squaresize = TILE_SIZE_PX / TILE_SIZE
    for row in range(height):
        for col in range(width):
            color_index = row * width + col
            color = frame[color_index % len(frame)]
            x = col * squaresize
            y = row * squaresize
            pygame.draw.rect(screen, color, (x, y, squaresize, squaresize))


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