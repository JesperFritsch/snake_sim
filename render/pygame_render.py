
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

import pygame
import os
import json
import argparse
import sys
import math
import utils

import core

def frames_from_runfile(filepath, expand_factor=2):
    pixel_changes = core.pixel_changes_from_runfile(filepath, expand_factor)
    frames = []
    grid_height = pixel_changes['height']
    grid_width = pixel_changes['width']
    color_list = [pixel_changes['free_color']] * ((grid_width) * (grid_height))
    changes = pixel_changes['changes']
    for step_data in changes:
        for (x, y), color in step_data:
            color_list[y * (grid_width) + x] = color
        frames.append(color_list.copy())
    return frames, grid_width, grid_height


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
            raise KeyboardInterrupt


def playback_runfile(filename=None, frames=None, grid_width=None, grid_height=None, expand=2):
    if not ((frames is not None) != (filename is not None)):
        raise ValueError("Either filename or frames must be provided, not both.")
    else:
        if frames is None:
            frames, grid_width, grid_height = frames_from_runfile(filename, expand)
        elif grid_height is None or grid_width is None:
            grid_width = int(len(frames[0]) ** 0.5)
            grid_height = int(len(frames[0]) ** 0.5)
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()

    drawGray(surface, grid_width, grid_height)
    running = True
    default_fps = 10
    frame_counter = 0
    play_direction = 1
    while running:
        pause = False
        while running:
            step_nr = math.ceil(frame_counter / expand)
            fps = default_fps
            speed_up = 20
            play_direction = 1
            keys = pygame.key.get_pressed()
            new_frame = not pause
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pause = not pause
                    elif event.key == pygame.K_LEFT:
                        play_direction = -1
                        new_frame = True
                    elif event.key == pygame.K_RIGHT:
                        new_frame = True
                    elif event.key == pygame.K_RETURN:
                        print(utils.get_run_step(filename, step_nr))
            if keys[pygame.K_LCTRL]:
                if keys[pygame.K_LSHIFT]:
                    speed_up *= 20
                if keys[pygame.K_LEFT]:
                    play_direction = -1
                    fps = default_fps * speed_up
                    new_frame = True
                elif keys[pygame.K_RIGHT]:
                    fps = default_fps * speed_up
                    new_frame = True

            if new_frame:
                frame_counter = max(min(frame_counter + play_direction, len(frames) - 1), 0)
                if 0 <= frame_counter < len(frames):
                    frame = frames[frame_counter]
                    play_direction = 1
                    print(f"step: {frame_counter // expand}")
                    draw_frame(screen, grid_width, grid_height, frame)
            pygame.display.flip()
            clock.tick(fps)
            if not pause:
                break
    pygame.quit()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', type=str, required=False)
    args = ap.parse_args(sys.argv[1:])
    if not args.file:
        args.file = r"B:\pythonStuff\snake_sim\runs\grid_32x32\7_snakes_32x32_VHYGPX_1254.json"
    playback_runfile(filename=args.file, expand=2)
