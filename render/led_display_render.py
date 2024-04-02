import core

def pixel_changes_from_runfile(filepath, expand_factor=2):
    grid_changes = core.grid_changes_from_runfile(filepath, expand_factor)
    color_changes = grid_changes['changes']
    pixel_changes = []
    for step_data in color_changes:
        step_changes = []
        food_changes = step_data['food_changes']
        for snake_change in step_data['snake_changes']:
            step_changes.append(food_changes + snake_change)
            food_changes = []
        pixel_changes.append(step_changes)
    return pixel_changes