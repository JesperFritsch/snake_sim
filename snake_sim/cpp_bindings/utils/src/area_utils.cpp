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