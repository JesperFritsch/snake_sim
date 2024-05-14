
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

import pygame
import argparse
import sys
import math
import json
import utils
import time
import numpy as np
from pathlib import Path

import core

def frames_from_runfile(filepath, expand_factor=2):
    frames = []
    with open(Path(filepath)) as run_file:
        run_data = json.load(run_file)
        metadata = run_data.copy()
        del metadata['steps']
        frame_builder = core.FrameBuilder(metadata, expand_factor, offset=(1, 1))
        for step_nr, step_data in run_data['steps'].items():
            frames.extend(frame_builder.step_to_frames(step_data))
    return frames, metadata['width'], metadata['height']

def draw_frame(screen, frame_buffer):
    frame_buffer = np.rot90(np.fliplr(frame_buffer))
    buffer_surface = pygame.surfarray.make_surface(frame_buffer)
    scaled_surface = pygame.transform.scale(buffer_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.blit(scaled_surface, (0, 0))

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

def play_stream(stream_conn, expand=2):
    snakes = {}
    frames = []
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()

    # wait for init data
    while not stream_conn.poll():
        pass

    run_meta_data = stream_conn.recv()
    frame_builder = core.FrameBuilder(run_meta_data=run_meta_data, expand_factor=expand, offset=(1, 1))
    grid_width = run_meta_data['width']
    grid_height = run_meta_data['height']

    drawGray(surface, grid_width, grid_height)
    running = True
    default_fps = 10
    frame_counter = 0
    play_direction = 1
    pause = False
    while running:
        if stream_conn.poll():
            step_data = stream_conn.recv()
            new_frames = frame_builder.step_to_frames(step_data)
            frames.extend(new_frames)
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
                    pass
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
                # print(f"step: {frame_counter // expand}")
                draw_frame(screen, frame)
                pygame.display.flip()
        clock.tick(fps)
    pygame.quit()




def play_runfile(filename=None, frames=None, grid_width=None, grid_height=None, expand=2):
    if not ((frames is not None) != (filename is not None)):
        raise ValueError("Either filename or frames must be provided, not both.")
    else:
        if frames is None:
            frames, grid_width, grid_height = frames_from_runfile(filename, expand)
        elif grid_height is None or grid_width is None:
            grid_width = grid_height = len(frames[0])

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
                    draw_frame(screen, frame)
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
    play_runfile(filename=args.file, expand=2)

