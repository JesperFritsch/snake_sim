#include "path_utils.hpp"


bool is_inside(int x, int y, int width, int height)
{
    return !(x < 0 || y < 0 || x >= width || y >= height);
}


Coord get_dir_to_tile(
    uint8_t *s_map, 
    int width, 
    int height, 
    Coord from_coord, 
    int tile_value,
    std::vector<int> visitable_values,
    bool clockwise
){
    std::vector<bool> visited(width * height, false);
    std::deque<Coord> coords_to_visit;
    std::unordered_map<Coord, Coord> came_from;
    came_from.reserve(width * height);
    coords_to_visit.push_back(from_coord);
    visited[from_coord.y * width + from_coord.x] = true;

    while (!coords_to_visit.empty()){
        Coord current = coords_to_visit.front();
        coords_to_visit.pop_front();
        int current_index = current.y * width + current.x;
        if (s_map[current_index] == tile_value){
            // Reconstruct path
            Coord step = current;
            while (came_from.find(step) != came_from.end() && !(came_from[step] == from_coord)){
                step = came_from[step];
            }
            return step - from_coord;
        }
        auto visitable_tiles = get_visitable_tiles(s_map, width, height, current, visitable_values);
        if (!clockwise){
            std::reverse(visitable_tiles.begin(), visitable_tiles.end());
        }

        for (const auto& valid_coord : visitable_tiles){
            int neighbor_index = valid_coord.y * width + valid_coord.x;
            if (!visited[neighbor_index]){
                visited[neighbor_index] = true;
                coords_to_visit.push_back(valid_coord);
                came_from[valid_coord] = current;
            }
        }
    }

    return Coord(0,0); // Return (0,0) if no path found
}


std::vector<Coord> get_visitable_tiles(
    uint8_t *s_map, 
    int width, 
    int height, 
    Coord center_coord, 
    std::vector<int> visitable_values
){
    std::array<Coord, 4> directions = {Coord(1,0), Coord(0,1), Coord(-1,0), Coord(0,-1)};
    std::vector<Coord> result;
    for (const auto& dir : directions){
        Coord candidate_coord = center_coord + dir;
        if (is_inside(candidate_coord.x, candidate_coord.y, width, height)){
            int index = candidate_coord.y * width + candidate_coord.x;
            if (std::find(visitable_values.begin(), visitable_values.end(), s_map[index]) != visitable_values.end()){
                result.push_back(candidate_coord);
            }
        }
    }
    return result;
}