SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

import pygame
import json
import numpy as np
import threading
import time
from pathlib import Path

from importlib import resources

from snake_sim.render import core
# from snake_sim.snake_env import RunData, StepData
from snake_sim.run_data.run_data import RunData, StepData

STREAM_IS_LIVE = False


def catch_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            pygame.quit()
    return wrapper


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
    nr_steps = len(run_data.steps)
    for step_nr in range(nr_steps):
        step_data = run_data.steps[step_nr]
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

@catch_exceptions
def handle_stream(stream_conn, frame_buffer: list, sound_buffer: list, run_data: RunData):
    global STREAM_IS_LIVE
    while not stream_conn.poll(0.05):
        pass

    STREAM_IS_LIVE = True
    run_meta_data = stream_conn.recv()
    run_data.height = run_meta_data['height']
    run_data.width = run_meta_data['width']
    run_data.base_map = np.array(run_meta_data['base_map'], dtype=np.uint8)
    run_data.snake_ids = run_meta_data['snake_ids']
    run_data.food_value = run_meta_data['food_value']
    run_data.free_value = run_meta_data['free_value']
    run_data.blocked_value = run_meta_data['blocked_value']
    run_data.color_mapping = {int(k): tuple(v) for k, v in run_meta_data['color_mapping'].items()}
    run_data.snake_values = run_meta_data["snake_values"]

    frame_builder = core.FrameBuilder(run_meta_data=run_meta_data)

    while STREAM_IS_LIVE:
        turn_sounds = []
        eat_sounds = []
        try:
            payload = stream_conn.recv()
        except EOFError:
            break
        if payload == 'stopped':
            break
        step_data_dict = payload
        step_data = StepData.from_dict(step_data_dict)
        run_data.add_step(step_data)
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
    STREAM_IS_LIVE = False

@catch_exceptions
def play_run(frame_buffer, sound_buffer, run_data: RunData, grid_width, grid_height, fps_playback, sound_on=True, keep_up=False):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    if sound_on:
        sound_mixer = pygame.mixer
        sound_mixer.init()
        eat_sound = sound_mixer.Sound(str(Path(__file__).parent / "sounds/eat.wav"))
        left_sound = sound_mixer.Sound(str(Path(__file__).parent / "sounds/turn_left.wav"))
        right_sound = sound_mixer.Sound(str(Path(__file__).parent / "sounds/turn_right.wav"))
        eat_sound.set_volume(1)
        left_sound.set_volume(1)
        right_sound.set_volume(1)

    drawGray(surface, grid_width, grid_height)
    running = True
    frame_counter = 0
    sim_step = 0
    play_direction = 1
    pause = False
    last_frame = None
    time_start = time.time()
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
                    with resources.path('snake_sim', '__init__.py') as init_path:
                        p_root = init_path.parent
                    state_files_folder = p_root / "test_bench" / "state_files"
                    state_file = state_files_folder / f"state_{sim_step}.json"
                    if not state_file.parent.exists():
                        state_file.parent.mkdir(parents=True)
                    with open(state_file, 'w') as f:
                        f.write(json.dumps(run_data.get_state_dict(sim_step)))
                    print(f"State saved to {state_file}")
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

        if new_frame:
            frame_counter = max(min(frame_counter + play_direction, len(frame_buffer) - 1), 0)
            if 0 <= frame_counter < len(frame_buffer):
                frame = frame_buffer[frame_counter]
                if frame is not last_frame:
                    if sound_on:
                        for sound in sound_buffer[frame_counter]:
                            if sound == 'eat':
                                eat_sound.play()
                            elif sound == 'left':
                                left_sound.play()
                            elif sound == 'right':
                                right_sound.play()
                    draw_frame(screen, frame)
                last_frame = frame
                pygame.display.flip()
            while (frame_counter >= len(frame_buffer) - 1) and STREAM_IS_LIVE:
                time.sleep(0.05)
            if keep_up and frame_counter < len(frame_buffer) - 2:
                fps = fps_playback * 10
        time_end = time.time()
        time_elapsed = time_end - time_start
        sleep_time = (1 / fps) - time_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        time_start = time.time()
    pygame.quit()


def play_stream(stream_conn, fps=10, sound_on=True):
    sound_buffer = []
    frame_buffer = []
    run_data = RunData(0, 0, [], np.array([]),0,0,0,{},{}) # create this here so that the stream thread and the play thread can share the same object
    stream_thread = threading.Thread(target=handle_stream, args=(stream_conn, frame_buffer, sound_buffer, run_data))
    stream_thread.daemon = True
    stream_thread.start()
    # wait for the stream thread to initialize the run data
    while run_data.width == 0 and run_data.height == 0:
        time.sleep(0.1)
    play_run(frame_buffer, sound_buffer, run_data, run_data.width, run_data.height, sound_on=sound_on, fps_playback=fps)


def play_frame_buffer(frame_buffer, grid_width, grid_height, fps=10):
    sound_buffer = [[None]] * len(frame_buffer)
    run_data = RunData(grid_width, grid_height, [], np.array([]),0,0,0,{},{})
    run_data.steps = {i: StepData([], i) for i in range(1, int(len(frame_buffer) / 2) + 1)}
    play_run(frame_buffer, sound_buffer, run_data, grid_width, grid_height, fps_playback=fps)


def play_runfile(filepath=None, sound_on=True, fps=10):
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix == '.json':
        run_data = RunData.from_json_file(filepath)
    if file_path.suffix == '.pb':
        run_data = RunData.from_protobuf_file(filepath)
    grid_height = run_data.height
    grid_width = run_data.width
    frame_buffer, sound_buffer = frames_sound_from_run_data(run_data)
    play_run(frame_buffer, sound_buffer, run_data, grid_width, grid_height, sound_on=sound_on, fps_playback=fps)


def play_game(conn, spm, sound_on=True):
    fps = spm / 60 * 2
    sound_buffer = []
    frame_buffer = []
    run_data = RunData(0, 0, [], np.array([]),0,0,0,{},{}) # create this here so that the stream thread and the play thread can share the same object
    stream_thread = threading.Thread(target=handle_stream, args=(conn, frame_buffer, sound_buffer, run_data))
    stream_thread.daemon = True
    stream_thread.start()
    # wait for the stream thread to initialize the run data
    while run_data.width == 0 and run_data.height == 0:
        time.sleep(0.1)
    play_run(frame_buffer, sound_buffer, run_data, run_data.width, run_data.height, sound_on=sound_on, keep_up=True, fps_playback=fps)