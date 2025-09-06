#include "area_checker.hpp"


void AreaChecker::print_map(uint8_t *s_map)
{
    ::print_map(s_map, width, height, head_value, body_value, food_value);
}

bool AreaChecker::_is_bad_gateway(uint8_t *s_map, Coord coord1, Coord coord2)
// Checking if going through a 'gateway' creates an unvisitable tile.
// below is a diagram of the situation we are checking for
/*
X X 2 X X
X . 1 . X
*/
{
    const int cols = width;
    Coord delta = coord1 - coord2;
    if(std::abs(delta.x) + std::abs(delta.y) != 1)
    {
        throw std::invalid_argument("coord1 and coord2 must be adjacent");
    }
    
    const Coord n_r1 = Coord(coord1.x + delta.y, coord1.y + delta.x);
    const Coord n_r2 = Coord(coord1.x + (delta.y * 2), coord1.y + (delta.x * 2));
    const Coord n_l1 = Coord(coord1.x - delta.y, coord1.y - delta.x);
    const Coord n_l2 = Coord(coord1.x - (delta.y * 2), coord1.y - (delta.x * 2));

    if (is_inside(n_r1.x, n_r1.y) && s_map[n_r1.y * cols + n_r1.x] <= free_value)
    {
        if (!is_inside(n_r2.x, n_r2.y) || s_map[n_r2.y * cols + n_r2.x] > free_value)
        {
            if (is_inside(n_l1.x, n_l1.y) && s_map[n_l1.y * cols + n_l1.x] <= free_value)
            {
                if (!is_inside(n_l2.x, n_l2.y) || s_map[n_l2.y * cols + n_l2.x] > free_value)
                {
                    return true;
                }
            }
        }
    }
    return false;
}


int AreaChecker::is_gate_way(uint8_t __restrict *s_map, Coord coord, Coord check_coord)
{
    // return code 2 is for a passage like:
    // x . .
    // . . .
    // . . x
    // or flipped
    const unsigned int cols = static_cast<unsigned int>(this->width);
    const unsigned int rows = static_cast<unsigned int>(this->height);
    const uint8_t free_value = static_cast<uint8_t>(this->free_value);
    const uint8_t offmap_value = free_value + 1; // offmap is considered blocked
    const unsigned int ch_x = check_coord.x, ch_y = check_coord.y;
    const unsigned int c_x = coord.x, c_y = coord.y;
    const Coord delta = check_coord - coord;
    const int delta_x = delta.x;
    const int delta_y = delta.y;
    const bool ch_interior = (1 <= ch_x && ch_x < cols-1 && 1 <= ch_y && ch_y < rows-1);
    const bool c_interior  = (1 <= c_x  && c_x  < cols-1 && 1 <= c_y  && c_y  < rows-1);

    if (
        ch_interior && c_interior
    )
    {
        // 1) If all 8 neighbors around check_coord are free, return 0 quickly

        const uint8_t* rN = s_map + (ch_y - 1) * cols + (ch_x - 1);
        const uint8_t* rC = rN + cols;
        const uint8_t* rS = rC + cols;

        const uint8_t ch_NW = rN[0], ch_N = rN[1], ch_NE = rN[2];
        const uint8_t ch_W  = rC[0],               ch_E = rC[2];
        const uint8_t ch_SW = rS[0], ch_S = rS[1], ch_SE = rS[2];

        if ((ch_NW <= free_value) && (ch_N <= free_value) &&
            (ch_NE <= free_value) && (ch_E <= free_value) &&
            (ch_SE <= free_value) && (ch_S <= free_value) &&
            (ch_SW <= free_value) && (ch_W <= free_value))
        {
            return 0;
        }

        rN = s_map + (c_y - 1) * cols + (c_x - 1);
        rC = rN + cols;
        rS = rC + cols;

        const uint8_t c_NW = rN[0], c_N = rN[1], c_NE = rN[2];
        const uint8_t c_W  = rC[0],               c_E = rC[2];
        const uint8_t c_SW = rS[0], c_S = rS[1], c_SE = rS[2];

        if ((!(c_NW <= free_value) && !(c_SE <= free_value) &&
            (c_N <= free_value) && (c_E <= free_value) &&
            (c_S <= free_value) && (c_W <= free_value) &&
            (c_NE <= free_value) && (c_SW <= free_value)) || 
            (!(c_NE <= free_value) && !(c_SW <= free_value) &&
            (c_N <= free_value) && (c_E <= free_value) &&
            (c_S <= free_value) && (c_W <= free_value) &&
            (c_NW <= free_value) && (c_SE <= free_value)))
        {
            return 2;
        }

        const uint8_t dir_c_NW = s_map[(ch_y - delta_x) * cols + (ch_x + delta_y)];
        const uint8_t dir_c_NE = s_map[(ch_y + delta_x) * cols + (ch_x - delta_y)];
        const uint8_t dir_c_W = s_map[(c_y - delta_x) * cols + (c_x + delta_y)];
        const uint8_t dir_c_E = s_map[(c_y + delta_x) * cols + (c_x - delta_y)];

        if ((!(dir_c_NW <= free_value) || !(dir_c_W <= free_value)) &&
            (!(dir_c_E <= free_value) || !(dir_c_NE <= free_value)))
        {
            return 1;
        }

    }
    else {
        // 1) If all 8 neighbors around check_coord are free, return 0 quickly

        auto on_map = [&](unsigned int x, unsigned int y) -> bool {
            return (0 <= x && x < cols && 0 <= y && y < rows);
        };

        auto safe_get = [&](unsigned int x, unsigned int y) -> uint8_t {
            if (!on_map(x,y)) return offmap_value;
            size_t idx = static_cast<size_t>(y) * cols + static_cast<size_t>(x);
            return s_map[idx];
        };

        const uint8_t ch_NW = safe_get(ch_x - 1, ch_y - 1);
        const uint8_t ch_N = safe_get(ch_x, ch_y - 1);
        const uint8_t ch_NE = safe_get(ch_x + 1, ch_y - 1);
        const uint8_t ch_E = safe_get(ch_x + 1, ch_y);
        const uint8_t ch_SE = safe_get(ch_x + 1, ch_y + 1);
        const uint8_t ch_S = safe_get(ch_x, ch_y + 1);
        const uint8_t ch_SW = safe_get(ch_x - 1, ch_y + 1);
        const uint8_t ch_W = safe_get(ch_x - 1, ch_y);

        if ((ch_NW <= free_value) && (ch_N <= free_value) &&
            (ch_NE <= free_value) && (ch_E <= free_value) &&
            (ch_SE <= free_value) && (ch_S <= free_value) &&
            (ch_SW <= free_value) && (ch_W <= free_value))
        {
            return 0;
        }

        const uint8_t c_NW = safe_get(c_x - 1, c_y - 1);
        const uint8_t c_N = safe_get(c_x, c_y - 1);
        const uint8_t c_NE = safe_get(c_x + 1, c_y - 1);
        const uint8_t c_E = safe_get(c_x + 1, c_y);
        const uint8_t c_SE = safe_get(c_x + 1, c_y + 1);
        const uint8_t c_S = safe_get(c_x, c_y + 1);
        const uint8_t c_SW = safe_get(c_x - 1, c_y + 1);
        const uint8_t c_W = safe_get(c_x - 1, c_y);

        if ((!(c_NW <= free_value) && !(c_SE <= free_value) &&
            (c_N <= free_value) && (c_E <= free_value) &&
            (c_S <= free_value) && (c_W <= free_value) &&
            (c_NE <= free_value) && (c_SW <= free_value)) || 
            (!(c_NE <= free_value) && !(c_SW <= free_value) &&
            (c_N <= free_value) && (c_E <= free_value) &&
            (c_S <= free_value) && (c_W <= free_value) &&
            (c_NW <= free_value) && (c_SE <= free_value)))
        {
            return 2;
        }

        const uint8_t dir_c_NW = safe_get(ch_x + delta_y, ch_y - delta_x);
        const uint8_t dir_c_NE = safe_get(ch_x - delta_y, ch_y + delta_x);
        const uint8_t dir_c_W = safe_get(c_x + delta_y, c_y - delta_x);
        const uint8_t dir_c_E = safe_get(c_x - delta_y, c_y + delta_x);

        if ((!(dir_c_NW <= free_value) || !(dir_c_W <= free_value)) &&
            (!(dir_c_E <= free_value) || !(dir_c_NE <= free_value)))
        {
            return 1;
        }

    }

    return 0;
}



py::dict AreaChecker::py_area_check(
    py::array_t<uint8_t> s_map,
    py::list body_coords_py,
    py::tuple start_coord_py,
    int target_margin,
    bool food_check,
    bool complete_area,
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
            food_check,
            complete_area,
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
    int tile_count = 0;
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
    current_coords.push_back(start_coord);
    checked[start_coord.y * width + start_coord.x] = area_id;

    if (s_map[start_coord.y * width + start_coord.x] > free_value)
    {
        return ExploreResults();
    }


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
        tile_count += 1;
        coord_parity_diff += get_coord_mod_parity(curr_coord) ? 1 : -1;

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
                        const int check_result = is_gate_way(s_map, n_coord, check_coord);
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
                int entrance_code = is_gate_way(s_map, curr_coord, n_coord);
                if (entrance_code == 0)
                {
                    checked[n_y * width + n_x] = area_id; // this used to be above this if statement, dont know if this will cause a bug, but i think it should be fine.
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
    bool food_check,
    bool complete_area,
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
            !(food_check || exhaustive || complete_area),
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
            (prev_node->id != 0 ? 
                result.connected_areas.size() + result.to_explore.size() == 2
                : result.connected_areas.size() + result.to_explore.size() == 1
            )
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

    // return AreaCheckResult();

    return graph.search_best2(body_coords.size(), s_map, food_value, width, target_margin, food_check, exhaustive);
}

// best first search area check
// returns a map of base coords to a map of depth to best margin fraction found at that depth
// eg. {Coord(5,5): {0: 0.1, 1: 0.05}, Coord(6,5): {0: 0.2, 1: 0.15}}
// max_depth is the maximum frames allowed in the stack -1 (-1 because the first frame is not a move)
// will perform one area_check per visitable tile at each frame.

RecurseCheckResult AreaChecker::recurse_area_check(
    uint8_t *s_map,
    std::deque<Coord> &body_coords,
    Coord search_first,
    unsigned int snake_length,
    unsigned int max_depth,
    float safe_margin_frac
){  

    auto get_margin_frac = [&] (AreaCheckResult &result) -> float {
        if (result.total_steps == 0) return 0.0f;
        if (result.has_tail) return std::numeric_limits<float>::infinity();
        return static_cast<float>(result.margin) / static_cast<float>(result.total_steps);
    };

    auto move_snake_forward = [&] (std::deque<Coord> &body_coords, Coord &new_head, bool do_grow) -> void {
        body_coords.push_front(new_head);
        if (!do_grow){
            body_coords.pop_back();
        }
    };

    auto move_snake_backward = [&] (std::deque<Coord> &body_coords, Coord &old_tail, bool do_shrink) -> void {
        body_coords.pop_front();
        if (!do_shrink){
            body_coords.push_back(old_tail);
        }
    };

    auto apply_snake_move_on_map = [&] (std::vector<uint8_t> &s_map_vec, Coord &current_head, Coord &current_tail, Coord &new_head) -> void {
        s_map_vec[current_head.y * width + current_head.x] = body_value;
        s_map_vec[new_head.y * width + new_head.x] = head_value;
        s_map_vec[current_tail.y * width + current_tail.x] = free_value;
    };

    auto revert_snake_move_on_map = [&] (std::vector<uint8_t> &s_map_vec, Coord &current_head, Coord &current_tail, Coord &new_head) -> void {
        s_map_vec[current_head.y * width + current_head.x] = head_value;
        s_map_vec[new_head.y * width + new_head.x] = free_value;
        s_map_vec[current_tail.y * width + current_tail.x] = body_value;
    };

    auto has_food = [&] (std::vector<uint8_t> &s_map_vec, Coord &coord) -> bool {
        return s_map_vec[coord.y * width + coord.x] == food_value;
    };

    auto s_map_vec = std::vector<uint8_t>(s_map, s_map + (width * height));
    RecurseCheckResult result;
    std::vector<RecurseCheckFrame> stack;
    unsigned int depth = 0;
    int target_margin = 10;
    Coord head_coord = body_coords.front();
    RecurseCheckFrame first_frame(
        head_coord,
        get_visitable_tiles(s_map, width, height, head_coord, {free_value, food_value})
    );
    stack.reserve(max_depth + 1);
    //sort to_visit in reverse order by distance to search_first
    if (search_first != Coord()){
        std::sort(first_frame.to_visit.begin(), first_frame.to_visit.end(), [&] (const Coord &a, const Coord &b) {
            return a.distance(search_first) > b.distance(search_first);
        });
    }
    
    stack.push_back(first_frame);
    
    while(!stack.empty()){
        // std::cout << "----------------------------------------" << std::endl << std::endl;
        auto& current_frame = stack.back();
        depth = stack.size() - 1;
        // the base coords are the next options from the head of the first frame
        
        if (
            depth >= max_depth && 
            current_frame.best_margin_frac >= safe_margin_frac
        ){
            // we can stop searching this base coord, as we have found a safe margin fraction deep enough
            stack.clear();
            break;
        }
    
        if (!current_frame.setup_done && depth > 0){ 
            // apply the move of the current frame
            Coord curr_head = body_coords.front();
            Coord curr_tail = body_coords.back();
            bool do_grow = has_food(s_map_vec, current_frame.head);
            move_snake_forward(body_coords, current_frame.head, do_grow);
            apply_snake_move_on_map(s_map_vec, curr_head, curr_tail, current_frame.head);
            current_frame.setup_done = true;
            current_frame.old_tail = curr_tail;
            current_frame.did_grow = do_grow;
        }

        // print_map(s_map_vec.data());
        // std::cout << "Current base coord: (" << current_frame.base_coord.x << "," << current_frame.base_coord.y << ")" << std::endl;
        // std::cout << "Depth: " << depth << ", stack size: " << stack.size() << ", to visit: " << current_frame.to_visit.size() << ", head: (" << current_frame.head.x << "," << current_frame.head.y << ")" << std::endl;
        // std::cout << "best margin frac for this frame: " << current_frame.best_margin_frac << std::endl;
        // std::cout << "results so far: " << std::endl;
        // for (const auto& kv : result.best_margin_fracs_at_depth) {
        //     std::cout << "  Base coord (" << kv.first.x << "," << kv.first.y << "): ";
        //     for (const auto& depth_kv : kv.second) {
        //         std::cout << "  Depth " << depth_kv.first << ": " << depth_kv.second << " ";
        //     }
        //     std::cout << std::endl;
        // }   
        // std::cout << "visitable tiles: ";
        // for (const auto& tile : current_frame.to_visit) {
        //     std::cout << "(" << tile.x << "," << tile.y << ") ";
        // }
        // std::cout << std::endl;
        
        
        if (!current_frame.to_visit.empty()){
            Coord &next_tile = current_frame.to_visit.back();
            current_frame.to_visit.pop_back();
            Coord base_coord = (depth == 0) ? next_tile : current_frame.base_coord;
            auto body_coords_vec = std::vector<Coord>(body_coords.begin(), body_coords.end());
            // std::cout << "checking next tile (" << next_tile.x << "," << next_tile.y << ")" << std::endl;
            auto check_result = area_check(
                s_map_vec.data(),
                body_coords_vec,
                next_tile,
                target_margin,
                false,
                false,
                false
            );
            // std::cout << "  area check result - is_clear: " << check_result.is_clear << ", tile_count: " << check_result.tile_count << ", total_steps: " << check_result.total_steps << ", food_count: " << check_result.food_count << ", has_tail: " << check_result.has_tail << ", margin: " << check_result.margin << std::endl;
            auto& best_margin_fracs_for_base = result.best_margin_fracs_at_depth[base_coord];
            auto margin_frac_it = best_margin_fracs_for_base.find(depth);
            auto margin_frac = get_margin_frac(check_result);
            // std::cout << "calculated margin frac for next tile (" << next_tile.x << "," << next_tile.y << "): " << margin_frac << std::endl;
            // std::cout << "safe margin frac: " << safe_margin_frac << std::endl;

            if (margin_frac_it == best_margin_fracs_for_base.end()){
                best_margin_fracs_for_base[depth] = margin_frac;
            }
            else if (margin_frac > margin_frac_it->second){
                // std::cout << "updating best margin frac for base coord (" << base_coord.x << "," << base_coord.y << ") at depth " << depth << " from " << margin_frac_it->second << " to " << margin_frac << std::endl;
                margin_frac_it->second = margin_frac;
            }

            if (margin_frac > current_frame.best_margin_frac){
                current_frame.best_margin_frac = margin_frac;
            }

            if (margin_frac >= safe_margin_frac && (depth < max_depth)){

                auto current_direction = next_tile - current_frame.head;
                auto next_visitables = get_visitable_tiles(s_map_vec.data(), width, height, next_tile, {free_value, food_value});
                auto preferred_tile = next_tile + current_direction;
                std::sort(next_visitables.begin(), next_visitables.end(), [&] (const Coord &a, const Coord &b) {
                    return a.distance(preferred_tile) > b.distance(preferred_tile);
                });

                RecurseCheckFrame next_frame(
                    next_tile,
                    next_visitables,
                    base_coord
                );
                // std::cout << "  pushing new frame for next tile (" << next_tile.x << "," << next_tile.y << ") with " << next_visitables.size() << " visitable tiles" << std::endl;
                stack.push_back(next_frame);
            }
            // always pop the tile after we checked it.
        } else {
            if (current_frame.setup_done && depth > 0){
                // revert the move of the current frame
                Coord old_head = body_coords[1];
                Coord old_tail = current_frame.old_tail;
                move_snake_backward(body_coords, current_frame.old_tail, current_frame.did_grow);
                revert_snake_move_on_map(s_map_vec, old_head, old_tail, current_frame.head);
            }
            stack.pop_back();
        }
    }
    return result;
}

py::dict AreaChecker::py_recurse_area_check(
    py::array_t<uint8_t> s_map,
    py::list body_coords_py,
    py::tuple search_first_py,
    unsigned int snake_length,
    unsigned int max_depth,
    float safe_margin_frac)
{
    auto buf = s_map.request();
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
    std::deque<Coord> body_coords;
    for (auto item : body_coords_py) {
        py::tuple t = item.cast<py::tuple>();
        body_coords.emplace_back(t[0].cast<int>(), t[1].cast<int>());
    }
    Coord search_first(search_first_py[0].cast<int>(), search_first_py[1].cast<int>());
    RecurseCheckResult result = recurse_area_check(
        ptr, body_coords, search_first, snake_length, max_depth, safe_margin_frac);
    py::dict margin_fracs_dict;
    for (const auto& kv : result.best_margin_fracs_at_depth) {
        py::tuple coord_tuple = py::make_tuple(kv.first.x, kv.first.y);
        py::dict depth_dict;
        for (const auto& depth_kv : kv.second) {
            depth_dict[py::int_(depth_kv.first)] = py::float_(depth_kv.second);
        }
        margin_fracs_dict[coord_tuple] = depth_dict;
    }
    return margin_fracs_dict;
}