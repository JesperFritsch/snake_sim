import json

from utils import coord_op


def pixel_changes_from_runfile(filepath, expand_factor=2):
    grid_changes = grid_changes_from_runfile(filepath, expand_factor)
    pixel_changes = {}
    pixel_changes.update(grid_changes)
    pixel_changes['changes'] = []
    color_changes = grid_changes['changes']
    for step_data in color_changes:
        food_changes = step_data['food_changes']
        for snake_change in step_data['snake_changes']:
            pixel_changes['changes'].append(food_changes + snake_change)
            food_changes = []
    return pixel_changes

def grid_changes_from_runfile(filename, expand_factor=2):
    grid_changes = {}
    run_dict = {}
    with open(filename) as run_json:
        run_dict = json.load(run_json)
    grid_height = run_dict['height']
    grid_width = run_dict['width']
    new_grid_h = grid_height * expand_factor
    new_grid_w = grid_width * expand_factor
    snake_data = run_dict['snake_data']
    snake_colors = {x.get('snake_id'): {'head_color': x.get('head_color'), 'body_color': x.get('body_color')} for x in snake_data}
    food_color = run_dict['food_color']
    free_color = run_dict['free_color']
    steps = run_dict['steps']
    grid_changes['height'] = new_grid_h
    grid_changes['width'] = new_grid_w
    grid_changes['food_color'] = food_color
    grid_changes['free_color'] = free_color
    grid_changes['snake_colors'] = snake_colors
    grid_changes['nr_of_steps'] = len(steps)
    grid_changes['expand_factor'] = expand_factor
    grid_changes['changes'] = []
    old_food = []
    color_list = [free_color] * ((new_grid_w) * (new_grid_h)) #The color list just makes it easier to keep track of the grid changes.
    for step, step_data in steps.items():
        food_changes = []
        removed_food = [food for food in old_food if food not in step_data['food']]
        new_food = [food for food in step_data['food'] if food not in old_food]
        for food in removed_food:
            food_x, food_y = coord_op(food, (expand_factor, expand_factor), '*')
            if color_list[food_y * (new_grid_w) + food_x] == food_color:
                food_changes.append(((food_x, food_y), free_color))
                color_list[food_y * (new_grid_w) + food_x] = free_color
        for food in new_food:
            food_x, food_y = coord_op(food, (expand_factor, expand_factor), '*')
            food_changes.append(((food_x, food_y), food_color))
            color_list[food_y * (new_grid_w) + food_x] = food_color
        old_food = step_data['food']
        step_grid_change = {'food_changes': food_changes, 'snake_changes': []}
        for i in range(1, expand_factor+1):
            snake_changes = []
            for snake in step_data['snakes']:
                snake_id = snake['snake_id']
                head_color = snake_colors[snake_id]['head_color']
                body_color = snake_colors[snake_id]['body_color']
                head_dir = snake['head_dir']
                tail_dir = snake['tail_dir']
                head_coord = snake['prev_head'] #fill in from prevoius head to current head.
                tail_coord = snake['curr_tail']
                t_dir_mult = coord_op(tail_dir, (expand_factor, expand_factor), '*')
                h_dir_mult = coord_op(head_dir, (i, i), '*')
                head_coord_mult = coord_op(head_coord, (expand_factor, expand_factor), '*')
                tail_coord_mult = coord_op(tail_coord, (expand_factor, expand_factor), '*')
                old_tail = coord_op(tail_coord_mult, t_dir_mult, '-')
                h_x, h_y = coord_op(head_coord_mult, h_dir_mult, '+')
                snake_changes.append(((h_x, h_y), body_color))
                color_list[h_y * (new_grid_w) + h_x] = body_color
                if any([x != 0 for x in tail_dir]):
                    t_dir_mult = coord_op(tail_dir, (i-1, i-1), '*')
                    t_x, t_y = coord_op(old_tail, t_dir_mult, '+')
                    color_list[t_y * (new_grid_w) + t_x] = free_color
                    snake_changes.append(((t_x, t_y), free_color))
            step_grid_change['snake_changes'].append(snake_changes)
        grid_changes['changes'].append(step_grid_change)
    return grid_changes

# if __name__ == '__main__':
#     print(json.dumps(grid_changes_from_runfile(r'B:\pythonStuff\snake_sim\runs\batch\grid_32x32\1_snakes_32x32_U8VAUE_9__ABORTED.json'), indent=2))