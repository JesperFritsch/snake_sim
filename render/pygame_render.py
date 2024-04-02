
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

import pygame

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
    while True:
        handle_events()
        pygame.display.flip()

# if __name__ == '__main__':
#     playback_runfile(r'B:\pythonStuff\snake_sim\runs\batch\grid_32x32\10_snakes_32x32_69D8WQ_184__ABORTED.json')