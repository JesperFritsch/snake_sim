
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

import pygame
import json
import sys

from snake_env import coord_op, SnakeEnv

def frames_from_runfile(filename, expand_factor=2):
    bg_color = (0,0,0)
    run_dict = {}
    frames = []
    with open(filename) as run_json:
        run_dict = json.load(run_json)
    grid_height = run_dict['height']
    grid_width = run_dict['width']
    new_grid_h = grid_height * expand_factor
    new_grid_w = grid_width * expand_factor
    snake_data = run_dict['snake_data']
    snake_colors = {x.get('snake_id'): {'head_color': x.get('head_color'), 'body_color': x.get('body_color')} for x in snake_data}
    steps = run_dict['steps']
    color_list = [bg_color] * ((new_grid_w) * (new_grid_h))
    for step, step_data in steps.items():
        for food in step_data['food']:
            food_x, food_y = coord_op(food, (expand_factor, expand_factor), '*')
            color_list[food_y * (new_grid_w) + food_x] = SnakeEnv.COLOR_MAPPING[SnakeEnv.FOOD_TILE]
        for i in range(1, expand_factor+1):
            for snake in step_data['snakes']:
                snake_id = snake['snake_id']
                head_color = snake_colors[snake_id]['head_color']
                body_color = snake_colors[snake_id]['body_color']
                head_dir = snake['head_dir']
                tail_dir = snake['tail_dir']
                head_coord = snake['coords'][1] #fill in from prevoius head to current head.
                tail_coord = snake['coords'][-1]
                t_dir_mult = coord_op(tail_dir, (expand_factor, expand_factor), '*')
                head_coord_mult = coord_op(head_coord, (expand_factor, expand_factor), '*')
                tail_coord_mult = coord_op(tail_coord, (expand_factor, expand_factor), '*')
                old_tail = coord_op(tail_coord_mult, t_dir_mult, '-')
                h_dir_mult = coord_op(head_dir, (i, i), '*')
                h_x, h_y = coord_op(head_coord_mult, h_dir_mult, '+')
                color_list[h_y * (new_grid_w) + h_x] = body_color
                if any([x != 0 for x in tail_dir]):
                    t_dir_mult = coord_op(tail_dir, (i, i), '*')
                    t_x, t_y = coord_op(old_tail, t_dir_mult, '+')
                    color_list[t_y * (new_grid_w) + t_x] = bg_color
            frames.append(color_list.copy())
    return frames, new_grid_w, new_grid_h


def draw_frame(screen, width, height, frame):
    TILE_SIZE_PX = SCREEN_WIDTH / width
    for row in range(height):
        for col in range(width):
            color_index = row * width + col
            color = frame[color_index % len(frame)]
            x = col * TILE_SIZE_PX
            y = row * TILE_SIZE_PX
            pygame.draw.rect(screen, color, (x, y, TILE_SIZE_PX + 1, TILE_SIZE_PX + 1))


def drawGray(surface, grid_width, grid_height):
    TILE_SIZE_PX = SCREEN_WIDTH / grid_width
    for y in range(0, int(grid_height)):
        for x in range(0, int(grid_width)):
            rr = pygame.Rect((x*TILE_SIZE_PX, y*TILE_SIZE_PX), (TILE_SIZE_PX,TILE_SIZE_PX))
            pygame.draw.rect(surface, (120,120,120), rr)


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def playback_runfile(filename):
    frames, grid_width, grid_height = frames_from_runfile(filename)
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()

    drawGray(surface, grid_width, grid_height)
    for frame in frames:
        clock.tick(10)
        handle_events()
        draw_frame(screen, grid_width, grid_height, frame)
        pygame.display.flip()