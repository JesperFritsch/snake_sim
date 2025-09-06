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