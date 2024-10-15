
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

import pygame
import json
import numpy as np
import threading
from pathlib import Path

from snake_sim.render import core
from snake_sim.snake_env import RunData, StepData


def frames_from_runfile(filepath, expand_factor=2):
    frames = []
    with open(Path(filepath)) as run_file:
        run_data = json.load(run_file)
        metadata = run_data.copy()
        metadata['color_mapping'] = {int(k): tuple(v) for k, v in metadata['color_mapping'].items()}
        del metadata['steps']
        frame_builder = core.FrameBuilder(metadata, expand_factor)
        for step_nr, step_data in run_data['steps'].items():
            frames.extend(frame_builder.step_to_frames(step_data))
    return frames, metadata['width'], metadata['height']


def frames_sound_from_run_data(run_data, expand_factor=2):
    frames_buffer = []
    sound_buffer = []
    frame_builder = core.FrameBuilder(run_data.to_dict(), expand_factor)
    for step_nr, step_data in run_data.steps.items():
        new_frames = frame_builder.step_to_frames(step_data.to_dict())
        frames_buffer.extend(new_frames)
        turn_sounds = []
        eat_sounds = []
        for snake_data in step_data.snakes:
            if snake_data["did_eat"]:
                eat_sounds.append('eat')
            if snake_data["did_turn"] == 'left':
                turn_sounds.append('left')
            elif snake_data["did_turn"] == 'right':
                turn_sounds.append('right')
        sound_buffer.extend([turn_sounds, eat_sounds])
    return frames_buffer, sound_buffer


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


def handle_stream(stream_conn, frame_buffer: list, sound_buffer: list, run_data: RunData):

    while not stream_conn.poll(0.05):
        pass

    run_meta_data = stream_conn.recv()

    run_data.height = run_meta_data['height']
    run_data.width = run_meta_data['width']
    run_data.base_map = np.array(run_meta_data['base_map'], dtype=np.uint8)
    run_data.snake_data = run_meta_data['snake_data']

    frame_builder = core.FrameBuilder(run_meta_data=run_meta_data)

    while True:
        if stream_conn.poll(0.05):
            turn_sounds = []
            eat_sounds = []
            step_data_dict = stream_conn.recv()
            step_data = StepData.from_dict(step_data_dict)
            step_count = step_data.step
            run_data.add_step(step_count, step_data)
            new_frames = frame_builder.step_to_frames(step_data_dict)
            frame_buffer.extend(new_frames)
            for snake_data in step_data_dict["snakes"]:
                if snake_data['did_eat']:
                    eat_sounds.append('eat')
                if snake_data["did_turn"] == 'left':
                    turn_sounds.append('left')
                elif snake_data["did_turn"] == 'right':
                    turn_sounds.append('right')
            sound_buffer.extend([turn_sounds, eat_sounds])


def play_run(frame_buffer, sound_buffer, run_data: RunData, grid_width, grid_height, sound_on=True):

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    sound_mixer = pygame.mixer
    sound_mixer.init()
    eat_sound = sound_mixer.Sound("snake_sim/render/sounds/eat.wav")
    left_sound = sound_mixer.Sound("snake_sim/render/sounds/turn_left.wav")
    right_sound = sound_mixer.Sound("snake_sim/render/sounds/turn_right.wav")
    eat_sound.set_volume(1)
    left_sound.set_volume(1)
    right_sound.set_volume(1)

    drawGray(surface, grid_width, grid_height)
    running = True
    default_fps = 10
    frame_counter = 0
    sim_step = 0
    play_direction = 1
    pause = False
    while running:
        sim_step = (frame_counter // 2) + 1
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
                    print(run_data.get_coord_mapping(sim_step))
        if keys[pygame.K_LCTRL]:
            if keys[pygame.K_LSHIFT]:
                frame_counter += 5 * play_direction
            if keys[pygame.K_LEFT]:
                play_direction = -1
                fps = default_fps * speed_up
                new_frame = True
            elif keys[pygame.K_RIGHT]:
                fps = default_fps * speed_up
                new_frame = True

        if new_frame:
            frame_counter = max(min(frame_counter + play_direction, len(frame_buffer) - 1), 0)
            if 0 <= frame_counter < len(frame_buffer):
                frame = frame_buffer[frame_counter]
                if sound_on:
                    for sound in sound_buffer[frame_counter]:
                        if sound == 'eat':
                            eat_sound.play()
                        elif sound == 'left':
                            left_sound.play()
                        elif sound == 'right':
                            right_sound.play()
                play_direction = 1
                draw_frame(screen, frame)
                pygame.display.flip()
            if frame_counter == len(frame_buffer) - 1:
                pause = True
        clock.tick(fps)
    pygame.quit()


def play_stream(stream_conn, sound_on=True):
    sound_buffer = []
    frame_buffer = []
    run_data = RunData(0, 0, [], np.array([])) # create this here so that the stream thread and the play thread can share the same object
    stream_thread = threading.Thread(target=handle_stream, args=(stream_conn, frame_buffer, sound_buffer, run_data))
    stream_thread.daemon = True
    stream_thread.start()
    # wait for the stream thread to initialize the run data
    while run_data.width == 0 and run_data.height == 0:
        pass
    play_run(frame_buffer, sound_buffer, run_data, run_data.width, run_data.height, sound_on=sound_on)


def play_runfile(filepath=None, frames=None, grid_height=None, grid_width=None, sound_on=True):
    if filepath:
        run_data = RunData.from_json_file(filepath)
        grid_height = run_data.height
        grid_width = run_data.width
        frame_buffer, sound_buffer = frames_sound_from_run_data(run_data)
    elif all([frames, grid_height, grid_width]):
        frame_buffer = frames
        sound_buffer = [[] * len(frame_buffer)]
    play_run(frame_buffer, sound_buffer, run_data, grid_width, grid_height, sound_on=sound_on)

