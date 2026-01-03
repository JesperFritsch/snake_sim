#include "area_utils.hpp"


std::vector<Coord> get_locations_with_value(uint8_t *s_map, int width, int height, uint8_t value) {
    std::vector<Coord> locations;
    locations.reserve(width * height);
    int size = width * height;
    for (int idx = 0; idx < size; ++idx) {
        if (s_map[idx] == value) {
            int y = idx / width;
            int x = idx % width;
            locations.push_back(Coord(x, y));
        }
    }
    return locations;
}

void print_map(uint8_t *s_map, int width, int height, int head_value, int body_value, int food_value){
    // Calculate digit widths for formatting
    int max_nr_digits_width = std::to_string(width).length();
    int max_nr_digits_height = std::to_string(height).length();

    // Prepare column number strings
    std::vector<std::string> col_nr_strings(width);
    for (int j = 0; j < width; ++j) {
        col_nr_strings[j] = std::string(max_nr_digits_width - std::to_string(j).length(), ' ') + std::to_string(j);
    }
    // Print column numbers (top)
    for (int digit = 0; digit < max_nr_digits_width; ++digit) {
        std::cout << std::string(max_nr_digits_height + 1, ' ');
        for (int j = 0; j < width; ++j) {
            std::cout << col_nr_strings[j][digit] << " ";
        }
        std::cout << std::endl;
    }
    // Print map rows with row numbers
    for (int i = 0; i < height; i++) {
        std::string row_nr = std::string(max_nr_digits_height - std::to_string(i).length(), ' ') + std::to_string(i);
        std::cout << row_nr << " ";
        for (int j = 0; j < width; j++) {
            char c;
            uint8_t v = s_map[i * width + j];
            if (v == head_value) {
                c = 'A';
            } else if (v == body_value) {
                c = 'a';
            } else if (v == food_value) {
                c = 'F';
            } else if (v == 1) {
                c = '.';
            } else if (v == 0) {
                c = '#';
            } else if (v % 2 == 0) {
                c = 'X';
            } else {
                c = 'x';
            }
            std::cout << c << " ";
        }
        std::cout << row_nr << std::endl;
    }
    // Print column numbers (bottom)
    for (int digit = 0; digit < max_nr_digits_width; ++digit) {
        std::cout << std::string(max_nr_digits_height + 1, ' ');
        for (int j = 0; j < width; ++j) {
            std::cout << col_nr_strings[j][digit] << " ";
        }
        std::cout << std::endl;
    }
    std::cout.flush();
}

bool can_make_area_inaccessible(
    uint8_t *s_map, 
    int width, 
    int height,
    int free_value,
    Coord head_pos,
    Coord neck_pos
)
{
    //. # # # # # .
    //. # # # # # .
    //. . . H . . .
    //. . . N . . .
    // simply check if any of the hash are blocked tiles
    // if so return true

    if (head_pos.x < 2 || head_pos.x >= width - 2 || head_pos.y < 2 || head_pos.y >= height - 2) {
        // head is too close to edge to make area inaccessible
        return true;
    }   
    Coord direction = head_pos - neck_pos;
    auto rot_head_coord = [&](Coord rel_pos) {
        // transorm the coord to be relative to head_pos and rotated so that direction is "up"
        if (direction.x == 1 && direction.y == 0) { 
            // right
            return Coord(head_pos.x - rel_pos.y, head_pos.y + rel_pos.x);
        } else if (direction.x == -1 && direction.y == 0) { 
            // left
            return Coord(head_pos.x + rel_pos.y, head_pos.y - rel_pos.x);
        } else if (direction.x == 0 && direction.y == 1) { 
            // down
            return Coord(head_pos.x - rel_pos.x, head_pos.y - rel_pos.y);
        } else if (direction.x == 0 && direction.y == -1) { 
            // up
            return Coord(head_pos.x + rel_pos.x, head_pos.y + rel_pos.y);
        } else {
            throw std::invalid_argument("direction must be one of the four cardinal directions");
        }
    };

    std::vector<Coord> check_coords = {
        rot_head_coord(Coord(-2, -2)),
        rot_head_coord(Coord(-1, -2)),
        rot_head_coord(Coord(0, -2)),
        rot_head_coord(Coord(1, -2)),
        rot_head_coord(Coord(2, -2)),
        rot_head_coord(Coord(-2, -1)),
        rot_head_coord(Coord(-1, -1)),
        rot_head_coord(Coord(1, -1)),
        rot_head_coord(Coord(2, -1)),
    };

    for (const auto& coord : check_coords) {
        if (s_map[coord.y * width + coord.x] > free_value) {
            return true;
        }
    }
    return false;
}


unsigned int* dist_heat_map(
    uint8_t *s_map,
    int width,
    int height,
    int free_value,
    int blocked_value,
    int target_value
){
    unsigned int* heat_map = new unsigned int[width * height];
    std::fill(heat_map, heat_map + width * height, 255); // Initialize all distances to "infinity" (255)

    std::queue<Coord> to_visit;

    std::unordered_set<Coord> visited;

    // Initialize the queue with all target positions
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (s_map[y * width + x] == target_value) {
                heat_map[y * width + x] = 0;
                to_visit.push(Coord(x, y));
            }
        }
    }

    std::vector<Coord> directions = {Coord(1,0), Coord(-1,0), Coord(0,1), Coord(0,-1)};

    while (!to_visit.empty()) {
        Coord current = to_visit.front();
        to_visit.pop();
        unsigned int current_dist = heat_map[current.y * width + current.x];
        for (const auto& dir : directions) {
            Coord neighbor(current.x + dir.x, current.y + dir.y);
            if (
                neighbor.x >= 0 && 
                neighbor.x < width && 
                neighbor.y >= 0 && 
                neighbor.y < height && 
                visited.find(neighbor) == visited.end()
            ) {
                visited.insert(neighbor);
                if (s_map[neighbor.y * width + neighbor.x] != blocked_value) {
                    unsigned int& neighbor_dist = heat_map[neighbor.y * width + neighbor.x];
                    if (current_dist + 1 < neighbor_dist) {
                        neighbor_dist = current_dist + 1;
                        to_visit.push(neighbor);
                    }
                }
            }
        }
    }

    return heat_map;
}


std::vector<int> area_boundary_tiles(
    uint8_t *s_map,
    int width,
    int height,
    int free_value,
    Coord area_start
) {
    std::unordered_set<int> boundary_tiles;
    std::queue<Coord> to_visit;
    std::unordered_set<Coord> visited;

    to_visit.push(area_start);
    visited.insert(area_start);

    std::vector<Coord> directions = {Coord(1,0), Coord(-1,0), Coord(0,1), Coord(0,-1)};

    while (!to_visit.empty()) {
        Coord current = to_visit.front();
        to_visit.pop();

        for (const auto& dir : directions) {
            Coord neighbor(current.x + dir.x, current.y + dir.y);
            if (
                neighbor.x >= 0 && 
                neighbor.x < width && 
                neighbor.y >= 0 && 
                neighbor.y < height
            ) {
                if (s_map[neighbor.y * width + neighbor.x] > free_value) {
                    int idx = neighbor.y * width + neighbor.x;
                    boundary_tiles.insert(s_map[idx]);
                } else {
                    auto [it, inserted] = visited.insert(neighbor);
                    if (inserted) {
                        to_visit.push(neighbor);
                    }
                }
            }
            else {
                boundary_tiles.insert(-1); 
            }
        }
    }

    return std::vector<int>(boundary_tiles.begin(), boundary_tiles.end());
}