#include "area_checker.hpp"


void AreaChecker::print_map(uint8_t *s_map)
{
    int rows = this->height;
    int cols = this->width;
    char c;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (s_map[i * cols + j] == 1)
            {
                c = '.';
            }
            else if (s_map[i * cols + j] == 0)
            {
                c = 'F';
            }
            else if (s_map[i * cols + j] == 2)
            {
                c = '#';
            }
            else
            {
                c = (char)s_map[i * cols + j];
            }
            if (Coord(j, i) == this->print_mark)
            {
                c = '+';
            }
            std::cout << " " << c << " ";
        }
        std::cout << std::endl;
    }
    this->print_mark = Coord();
}

bool AreaChecker::_is_bad_gateway(uint8_t *s_map, Coord coord1, Coord coord2)
// Checking if going through a 'gateway' creates an unvisitable tile.
// below is a diagram of the situation we are checking for
/*
X X 2 X X
X . 1 . X
*/
{
    const int cols = this->width;
    Coord delta = coord1 - coord2;
    if(std::abs(delta.x) + std::abs(delta.y) != 1)
    {
        throw std::invalid_argument("coord1 and coord2 must be adjacent");
    }
    
    const Coord n_r1 = Coord(coord1.x + delta.y, coord1.y + delta.x);
    const Coord n_r2 = Coord(coord1.x + (delta.y * 2), coord1.y + (delta.x * 2));
    const Coord n_l1 = Coord(coord1.x - delta.y, coord1.y - delta.x);
    const Coord n_l2 = Coord(coord1.x - (delta.y * 2), coord1.y - (delta.x * 2));

    if (this->is_inside(n_r1.x, n_r1.y) && s_map[n_r1.y * cols + n_r1.x] <= this->free_value)
    {
        if (!this->is_inside(n_r2.x, n_r2.y) || s_map[n_r2.y * cols + n_r2.x] > this->free_value)
        {
            if (this->is_inside(n_l1.x, n_l1.y) && s_map[n_l1.y * cols + n_l1.x] <= this->free_value)
            {
                if (!this->is_inside(n_l2.x, n_l2.y) || s_map[n_l2.y * cols + n_l2.x] > this->free_value)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

int AreaChecker::is_single_entrance(uint8_t *s_map, Coord coord, Coord check_coord)
{
    // return code 2 is for a passage like:
    // x . .
    // . . .
    // . . x
    // or flipped
    const int cols = this->width;
    static constexpr int offmap_value = -1;
    const Coord delta = check_coord - coord;
    int value;

    const std::array<Coord, 8> ch_neighbours = {
        Coord(check_coord.x - 1, check_coord.y - 1),
        Coord(check_coord.x, check_coord.y - 1),
        Coord(check_coord.x + 1, check_coord.y - 1),
        Coord(check_coord.x + 1, check_coord.y),
        Coord(check_coord.x + 1, check_coord.y + 1),
        Coord(check_coord.x, check_coord.y + 1),
        Coord(check_coord.x - 1, check_coord.y + 1),
        Coord(check_coord.x - 1, check_coord.y)};
    {
        bool all_free = true;
        for (const auto &c : ch_neighbours)
        {
            if (!this->is_inside(c.x, c.y) ||
                s_map[c.y * cols + c.x] > this->free_value)
            {
                all_free = false;
                break;
            }
        }
        if (all_free)
        {
            return 0;
        }
    }

    // 2) Prepare neighbours
    const std::array<Coord, 8> neighbours = {
        Coord(coord.x - 1, coord.y - 1),
        Coord(coord.x, coord.y - 1),
        Coord(coord.x + 1, coord.y - 1),
        Coord(coord.x + 1, coord.y),
        Coord(coord.x + 1, coord.y + 1),
        Coord(coord.x, coord.y + 1),
        Coord(coord.x - 1, coord.y + 1),
        Coord(coord.x - 1, coord.y)};

    // 3) Fill corner_values & neighbour_values in fewer branches
    std::array<int, 4> corner_values{};
    std::array<int, 4> neighbour_values{};

    for (unsigned int i = 0; i < 8; i++)
    {
        const Coord &nc = neighbours[i];
        const bool inside = this->is_inside(nc.x, nc.y);
        if (i % 2 == 0)
        {
            corner_values[i / 2] =
                inside ? s_map[nc.y * cols + nc.x] : offmap_value;
        }
        else
        {
            neighbour_values[i / 2] =
                inside ? s_map[nc.y * cols + nc.x] : offmap_value;
        }
    }

    if (corner_values[0] > this->free_value && corner_values[0] != offmap_value &&
        corner_values[2] > this->free_value && corner_values[2] != offmap_value)
    {
        bool is_diagonal = true;
        for (unsigned int i = 0; i < neighbour_values.size(); i++)
        {
            if (neighbour_values[i] > this->free_value)
            {
                is_diagonal = false;
                break;
            }
            if (i != 0 && i != 2 && corner_values[i] > this->free_value)
            {
                is_diagonal = false;
                break;
            }
        }
        if (is_diagonal)
        {
            return 2;
        }
    }
    if (corner_values[1] > this->free_value && corner_values[1] != offmap_value &&
        corner_values[3] > this->free_value && corner_values[3] != offmap_value)
    {
        bool is_diagonal = true;
        for (unsigned int i = 0; i < neighbour_values.size(); i++)
        {
            if (neighbour_values[i] > this->free_value)
            {
                is_diagonal = false;
                break;
            }
            if (i != 1 && i != 3 && corner_values[i] > this->free_value)
            {
                is_diagonal = false;
                break;
            }
        }
        if (is_diagonal)
        {
            return 2;
        }
    }

    int x = check_coord.x + delta.y;
    int y = check_coord.y + delta.x;
    if (this->is_inside(x, y))
    {
        value = s_map[y * cols + x];
        if (value <= this->free_value)
        {
            x = coord.x + delta.y;
            y = coord.y + delta.x;
            if (this->is_inside(x, y))
            {
                value = s_map[y * cols + x];
                if (value <= this->free_value)
                {
                    return 0;
                }
            }
        }
    }

    x = check_coord.x - delta.y;
    y = check_coord.y - delta.x;
    if (this->is_inside(x, y))
    {
        value = s_map[y * cols + x];
        if (value <= this->free_value)
        {
            x = coord.x - delta.y;
            y = coord.y - delta.x;
            if (this->is_inside(x, y))
            {
                value = s_map[y * cols + x];
                if (value <= this->free_value)
                {
                    return 0;
                }
            }
        }
    }
    return 1;
}

py::dict AreaChecker::py_area_check(
    py::array_t<uint8_t> s_map,
    py::list body_coords_py,
    py::tuple start_coord_py,
    int target_margin,
    int max_food,
    bool food_check,
    bool exhaustive)
{
    try {
        auto s_map_buf = s_map.request();
        uint8_t *s_map_ptr = static_cast<uint8_t *>(s_map_buf.ptr);
        Coord start_coord = Coord(start_coord_py[0].cast<int>(), start_coord_py[1].cast<int>());
        std::vector<Coord> body_coords;
        for (auto item : body_coords_py)
        {
            auto coord = item.cast<py::tuple>();
            body_coords.push_back(Coord(coord[0].cast<int>(), coord[1].cast<int>()));
        }

        AreaCheckResult result = area_check(
            s_map_ptr,
            body_coords,
            start_coord,
            target_margin,
            max_food,
            food_check,
            exhaustive);
        return py::dict(
            py::arg("is_clear") = result.is_clear,
            py::arg("tile_count") = result.tile_count,
            py::arg("total_steps") = result.total_steps,
            py::arg("food_count") = result.food_count,
            py::arg("has_tail") = result.has_tail,
            py::arg("margin") = result.margin);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception in area_check");
        throw py::error_already_set();
    }
}

ExploreResults AreaChecker::explore_area(
    uint8_t *s_map,
    std::vector<Coord> &body_coords,
    Coord &start_coord,
    int area_id,
    std::vector<int> &checked,
    bool early_exit,
    int snake_length,
    int target_margin,
    int total_food_count)
{
    int tile_count = 1;
    int food_count = 0;
    int max_index = area_id == 0 ? 0 : -1;
    Coord max_index_coord = Coord();
    int coord_parity_diff = 0;
    bool has_tail = false;
    bool did_early_exit = false;
    std::vector<Coord> to_explore;
    std::vector<std::pair<int, Coord>> body_tiles;
    std::vector<ConnectedAreaInfo> connected_areas;
    std::deque<Coord> current_coords;
    body_tiles.reserve(body_coords.size());
    coord_parity_diff += get_coord_mod_parity(start_coord) ? 1 : -1;
    current_coords.push_back(start_coord);
    checked[start_coord.y * width + start_coord.x] = area_id;
    while (!current_coords.empty())
    {
        auto curr_coord = current_coords.front();
        current_coords.pop_front();
        int c_x, c_y;
        c_x = curr_coord.x;
        c_y = curr_coord.y;
        if (s_map[c_y * width + c_x] == food_value)
        {
            // food_coords.insert(curr_coord);
            food_count += 1;
        }

        std::array<Coord, 4> neighbours = {
            Coord(c_x, c_y - 1),
            Coord(c_x + 1, c_y),
            Coord(c_x, c_y + 1),
            Coord(c_x - 1, c_y)};

        for (auto &n_coord : neighbours)
        {
            int n_x, n_y;

            n_x = n_coord.x;
            n_y = n_coord.y;
            if (!this->is_inside(n_x, n_y))
            {
                continue;
            }
            int checked_val = checked[n_y * width + n_x];
            if (checked_val != unexplored_area_id)
            {
                if (checked_val != area_id)
                {
                    if (std::find_if(
                            connected_areas.begin(),
                            connected_areas.end(),
                            [checked_val](const ConnectedAreaInfo& p) {
                                return p.id == checked_val;
                            }) == connected_areas.end())
                    {
                        const bool is_bad_gateway_from_here = _is_bad_gateway(s_map, curr_coord, n_coord);
                        const bool is_bad_gateway_to_here = _is_bad_gateway(s_map, n_coord, curr_coord);
                        // check if the gateway is diagonal
                        const Coord delta = n_coord - curr_coord;
                        const Coord check_coord = n_coord + delta;
                        const int check_result = is_single_entrance(s_map, n_coord, check_coord);
                        const bool is_diag_gateway = (check_result == 2);
                        connected_areas.push_back(
                            ConnectedAreaInfo(
                                checked_val, 
                                curr_coord, 
                                n_coord, 
                                is_bad_gateway_from_here, 
                                is_bad_gateway_to_here,
                                is_diag_gateway
                            )
                        );
                    }
                }
                continue;
            }
            int n_coord_val = s_map[n_y * width + n_x];
            if (n_coord_val == free_value || n_coord_val == food_value)
            {
                int entrance_code = is_single_entrance(s_map, curr_coord, n_coord);
                // int entrance_code = 0;
                if (entrance_code == 0)
                {
                    checked[n_y * width + n_x] = area_id; // this used to be above this if statement, dont know if this will cause a bug, but i think it should be fine.
                    tile_count += 1;
                    coord_parity_diff += get_coord_mod_parity(n_coord) ? 1 : -1;
                    current_coords.push_back(n_coord);
                }
                else
                {
                    if (std::find(to_explore.begin(), to_explore.end(), n_coord) == to_explore.end())
                    {
                        to_explore.push_back(n_coord);
                    }
                    if (entrance_code == 2)
                    {
                        break;
                    }
                }
            }
            else if (n_coord_val == body_value || n_coord_val == head_value)
            {
                auto it = std::find(body_coords.begin(), body_coords.end(), n_coord);
                if (it != body_coords.end())
                {
                    int body_index = static_cast<int>(std::distance(body_coords.begin(), it)); // Cast to int
                    if(body_index > max_index){
                        max_index = std::max(max_index, body_index);
                        max_index_coord = n_coord;
                        if (body_index == static_cast<int>(body_coords.size()) - 1)
                        {
                            has_tail = true;
                        }
                    }
                    body_tiles.push_back(std::make_pair(body_index, curr_coord));
                }
            }
        }
        int calc_target_margin = std::max(std::max(target_margin, food_count + total_food_count), 1);
        int total_steps = tile_count - (food_count + total_food_count) - std::abs(coord_parity_diff);
        int needed_steps = (max_index > 0) ? snake_length - max_index : snake_length + 1;
        int margin = total_steps - needed_steps;
        if (early_exit && margin > calc_target_margin * 2 && max_index_coord != start_coord)
        {
            // std::cout << "Early exit" << std::endl;
            // std::cout << "Margin: " << margin << std::endl;
            // std::cout << "Target margin: " << calc_target_margin << std::endl;
            // std::cout << "Tile count: " << tile_count << std::endl;
            // std::cout << "Food count: " << food_count << std::endl;
            // std::cout << "prev_food count: " << total_food_count << std::endl;

            did_early_exit = true;
            break;
        }
    }
    return ExploreResults(
        tile_count, 
        food_count, 
        coord_parity_diff,
        did_early_exit, 
        has_tail,
        body_tiles,
        connected_areas, 
        to_explore
    );
}

AreaCheckResult AreaChecker::area_check(
    uint8_t *s_map,
    std::vector<Coord> &body_coords,
    Coord &start_coord,
    int target_margin,
    int max_food,
    bool food_check,
    bool exhaustive)
{
    std::vector<int> checked;
    checked.resize(height * width);
    std::fill(checked.begin(), checked.end(), unexplored_area_id);

    AreaNode *prev_node = nullptr;
    AreaNode *current_node;
    AreaGraph graph = AreaGraph();
    std::deque<ExploreData> areas_to_explore;
    areas_to_explore.push_back(ExploreData(start_coord, graph.next_id++, prev_node));
    // in the while loop we will explore the area and add new nodes to the graph
    int total_food_count = 0;
    while (!areas_to_explore.empty())
    {
        auto current_area_data = areas_to_explore.front();
        areas_to_explore.pop_front();
        Coord current_coord = current_area_data.start_coord;
        int current_id = current_area_data.area_id;
        prev_node = current_area_data.prev_node;

        if (checked[current_coord.y * width + current_coord.x] != unexplored_area_id)
        {
            continue;
        }

        auto result = explore_area(
            s_map,
            body_coords,
            current_coord,
            current_id,
            checked,
            !(food_check || exhaustive),
            body_coords.size(),
            target_margin,
            total_food_count
        );
        total_food_count += result.food_count;

        // If an area has just one tile, no max index and only one area to explore, then we can just add the tile to the previous node
        if (
            prev_node != nullptr && 
            prev_node->is_one_dim && 
            result.tile_count == 1 && 
            result.body_tiles.size() == 0 &&
            (result.connected_areas.size() + result.to_explore.size() == 2)
        )
        {
            current_node = prev_node;
            graph.add_id_for_node(prev_node->id, current_id);
            current_node->tile_count += result.tile_count;
            current_node->food_count += result.food_count;
            current_node->coord_parity_diff += result.coord_parity_diff;
            current_node->has_tail = result.has_tail;
        }
        else
        {
            current_node = graph.add_node_with_id(current_coord, current_id);
            current_node->tile_count = result.tile_count;
            current_node->food_count = result.food_count;
            current_node->coord_parity_diff = result.coord_parity_diff;
            current_node->has_tail = result.has_tail;
            current_node->body_tiles = result.body_tiles;
            if (
                current_node->tile_count == 1 && 
                (result.body_tiles.size() == 1 ? result.body_tiles[0].first == 0 : result.body_tiles.size() == 0) &&
                (result.connected_areas.size() + result.to_explore.size() <= 2)
            )
            { // a node cant really have 2 or 3 tiles, next step after 1 is 4, but anyways...
                current_node->is_one_dim = true;
            }
        }

        if (
            prev_node != nullptr &&
            (prev_node->tile_count == 1 || prev_node->is_one_dim) &&
            !current_node->has_body &&
            prev_node->has_only_head &&
            (prev_node->get_nr_connections() <= 2)
        )
        { 
            // this is to propagate connection to head from the start to the first node with more than two connections
            // it is needed if the start is in a 1 tile narrow passage and leads to a juntion.
            current_node->has_only_head = true;
        }

        for (auto connected_area : result.connected_areas)
        {
            if (graph.get_node(connected_area.id) != nullptr)
            {
                graph.connect_nodes(current_node->id, connected_area.id, connected_area);
            }
        }
        current_node->finalize_node();
        for (auto &area_start_coord : result.to_explore)
        {
            areas_to_explore.push_back(ExploreData(area_start_coord, graph.next_id++, current_node));
        }
    }
    
    DEBUG_PRINT(graph.print_nodes_debug());

    return graph.search_best2(body_coords.size(), s_map, food_value, width, target_margin, max_food, food_check, exhaustive);
}