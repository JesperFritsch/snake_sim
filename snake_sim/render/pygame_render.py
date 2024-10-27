import pygame
import json
import numpy as np
import threading
from pathlib import Path
import asyncio

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

from snake_sim.render import core
from snake_sim.snake_env import RunData, StepData

queue = asyncio.Queue()

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


async def handle_stream(stream_conn, sound_on=True):

    while not stream_conn.poll(0.05):
        pass

    run_meta_data = stream_conn.recv()

    await queue.put(run_meta_data)
    run_data = RunData(0, 0, [], np.array([]))
    run_data.height = run_meta_data['height']
    run_data.width = run_meta_data['width']
    run_data.base_map = np.array(run_meta_data['base_map'], dtype=np.uint8)
    run_data.snake_data = run_meta_data['snake_data']
    
    await asyncio.sleep(0.001)
    
    step_count = 0
    while True:
        if stream_conn.poll(0.05):
            turn_sounds = []
            eat_sounds = []
            step_data_dict = stream_conn.recv()
            step_data = StepData.from_dict(step_data_dict)
            step_count = step_data.step
            run_data.add_step(step_count, step_data)
            await queue.put(step_data_dict)
            if sound_on:
                for snake_data in step_data_dict["snakes"]:
                    if snake_data['did_eat']:
                        eat_sounds.append('eat')
                    if snake_data["did_turn"] == 'left':
                        turn_sounds.append('left')
                    elif snake_data["did_turn"] == 'right':
                        turn_sounds.append('right')
        await asyncio.sleep(0.001)


async def play_run(sound_on=True, fps_playback=10):
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    sound_mixer = pygame.mixer
    sound_mixer.init()
    if sound_on:
        eat_sound = sound_mixer.Sound("snake_sim/render/sounds/eat.wav")
        left_sound = sound_mixer.Sound("snake_sim/render/sounds/turn_left.wav")
        right_sound = sound_mixer.Sound("snake_sim/render/sounds/turn_right.wav")
        eat_sound.set_volume(1)
        left_sound.set_volume(1)
        right_sound.set_volume(1)
    run_meta_data = await queue.get()
    frame_builder = core.FrameBuilder(run_meta_data=run_meta_data)
    run_data = RunData(0, 0, [], np.array([]))
    run_data.height = run_meta_data['height']
    run_data.width = run_meta_data['width']
    drawGray(surface, run_data.width, run_data.height)
    queue.task_done()
    running = True
    frame_counter = 0
    sim_step = 0
    play_direction = 1
    pause = False
    while running:
        sim_step = (frame_counter // 2) + 1
        fps = fps_playback
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
            if keys[pygame.K_LEFT]:
                play_direction = -1
                fps = fps_playback * speed_up
                new_frame = True
            elif keys[pygame.K_RIGHT]:
                fps = fps_playback * speed_up
                new_frame = True
            if keys[pygame.K_LSHIFT]:
                frame_counter += 5 * play_direction
        
        await asyncio.sleep(0.001)
        step_data_dict = await queue.get()
        frames = frame_builder.step_to_frames(step_data_dict)
        queue.task_done()
        for frame in frames:
            draw_frame(screen, frame)
        pygame.display.flip()
        await asyncio.sleep(0.001)
        clock.tick(fps)
        print(f"Frame: {frame_counter}, Step: {sim_step}")
        
    pygame.quit()


def play_stream(stream_conn, fps=10, sound_on=True):
    asyncio.run(play_stream_async(stream_conn, fps, sound_on))

async def play_stream_async(stream_conn, fps=10, sound_on=True):
    sound_buffer = []
    frame_buffer = []
    run_data = RunData(0, 0, [], np.array([])) # create this here so that the stream thread and the play thread can share the same object
    # wait for the stream thread to initialize the run data
    handle_stream_task = asyncio.create_task(handle_stream(stream_conn))
    play_run_task = asyncio.create_task(play_run(sound_on=sound_on, fps_playback=fps))
    await handle_stream_task
    await play_run_task

def play_runfile(filepath=None, frames=None, grid_height=None, grid_width=None, sound_on=True, fps=10):
    if filepath:
        run_data = RunData.from_json_file(filepath)
        grid_height = run_data.height
        grid_width = run_data.width
        frame_buffer, sound_buffer = frames_sound_from_run_data(run_data)
    elif all([frames, grid_height, grid_width]):
        frame_buffer = frames
        sound_buffer = [[None]] * len(frame_buffer)
        run_data = RunData(grid_width, grid_height, [], np.array([]))
    play_run(frame_buffer, sound_buffer, run_data, grid_width, grid_height, sound_on=sound_on, fps_playback=fps)

