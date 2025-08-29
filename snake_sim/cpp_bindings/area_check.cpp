#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For automatic conversion of C++ STL containers to Python
#include <atomic>
#include <chrono>
#include <deque>
#include <queue>
#include <thread>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream> // Include iostream for std::cout
#include <typeinfo> // Include typeinfo for typeid
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <functional>


#ifdef DEBUG
#define DEBUG_PRINT(x) do { x; } while (0)
#else
#define DEBUG_PRINT(x) do {} while (0)
#endif


namespace py = pybind11;

unsigned int cantor_pairing(int k1, int k2)
{
    return (k1 + k2) * (k1 + k2 + 1) / 2 + k2;
}


struct Coord
{
    int x;
    int y;

    Coord() : x(-1), y(-1) {}

    Coord(int x, int y) : x(x), y(y) {}
    // Overload operator[] to access x and y like an array
    int &operator[](size_t index)
    {
        if (index == 0)
            return x;
        if (index == 1)
            return y;
        throw std::out_of_range("Index out of range");
    }

    const int &operator[](size_t index) const
    {
        if (index == 0)
            return x;
        if (index == 1)
            return y;
        throw std::out_of_range("Index out of range");
    }

    bool operator==(const Coord &other) const
    {
        return x == other.x && y == other.y;
    }
    
    bool operator!=(const Coord &other) const
    {
        return x != other.x || y != other.y;
    }

    std::size_t hash() const
    {
        return x * 1000 + y;
    }
};

bool tile_has_food(uint8_t *s_map, int width, Coord coord, uint8_t food_value){
    return s_map[coord.y * width + coord.x] == food_value;
}

bool get_coord_mod_parity(Coord coord)
// returns true if the tile is even and false if odd, think black and white on a chessboard.
{
    return coord.x % 2 == coord.y % 2;
}


namespace std
{
    template <>
    struct hash<Coord>
    {
        std::size_t operator()(const Coord &coord) const noexcept
        {
            return coord.hash();
        }
    };
}

struct AreaCheckResult
{
    bool is_clear;
    int tile_count;
    int total_steps;
    int food_count;
    bool has_tail;
    int margin;
    int needed_steps;
    AreaCheckResult() : is_clear(false),
                        tile_count(0),
                        total_steps(0),
                        food_count(0),
                        has_tail(false),
                        margin(INT_MIN),
                        needed_steps(0) {}
    AreaCheckResult(
        bool is_clear,
        int tile_count,
        int total_steps,
        int food_count,
        bool has_tail,
        int margin,
        int needed_steps
    ) : 
        is_clear(is_clear),
        tile_count(tile_count),
        total_steps(total_steps),
        food_count(food_count),
        has_tail(has_tail),
        margin(margin),
        needed_steps(needed_steps) {}
};


struct ConnectedAreaInfo
{
    int id;
    Coord self_coord;
    Coord other_coord;
    bool is_bad_gateway_from_here;
    bool is_bad_gateway_to_here;

    ConnectedAreaInfo() : 
        id(-1),
        self_coord(Coord()),
        other_coord(Coord()),
        is_bad_gateway_from_here(false),
        is_bad_gateway_to_here(false) {}

    ConnectedAreaInfo(
        int id, 
        Coord self_coord, 
        Coord other_coord, 
        bool is_bad_gateway_from_here,
        bool is_bad_gateway_to_here
    ) : id(id),
        self_coord(self_coord),
        other_coord(other_coord),
        is_bad_gateway_from_here(is_bad_gateway_from_here),
        is_bad_gateway_to_here(is_bad_gateway_to_here) {}
};


class AreaNode
{
public:
    Coord start_coord;
    Coord end_coord = Coord();
    int id;
    int max_index = -1; // max_index is the index of the snake in this area, if it is not in this area, then it is -1
    int tile_count = 0;
    int food_count = 0;
    int coord_parity_diff = 0;
    Coord max_index_coord = Coord();
    std::unordered_map<int, ConnectedAreaInfo> neighbour_connections; // map of area_id to pair of coords that connect the areas

    // one_dim is true if the area is a line that the snake can not turn around in.
    bool is_one_dim = false;
    bool has_tail = false;
    std::vector<std::pair<AreaNode *, unsigned int>> edge_nodes;

    AreaNode(Coord start_coord, int id) : start_coord(start_coord),
                                          id(id)
    {
        edge_nodes.reserve(6);
        
    }

    void remove_connection(AreaNode *other_node)
    {
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [other_node](std::pair<AreaNode *, unsigned int> &pair)
                               { return pair.first == other_node; });
        if (it != edge_nodes.end())
        {
            edge_nodes.erase(it);
            neighbour_connections.erase(other_node->id);
        }
    }

    void remove_connection(unsigned int edge)
    {
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [edge](std::pair<AreaNode *, unsigned int> &pair)
                               { return pair.second == edge; });
        if (it != edge_nodes.end())
        {
            edge_nodes.erase(it);
            neighbour_connections.erase(it->first->id);
        }
    }

    void add_connection(AreaNode *new_node, unsigned int edge, ConnectedAreaInfo info)
    {
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [new_node](std::pair<AreaNode *, unsigned int> &pair)
                               { return pair.first == new_node; });
        if (it == edge_nodes.end())
        {
            edge_nodes.push_back(std::make_pair(new_node, edge));
            neighbour_connections[new_node->id] = info;
        }
    }

    int get_nr_connections()
    {
        return edge_nodes.size();
    }

    ConnectedAreaInfo get_connection_info(int area_id)
    {
        if (neighbour_connections.find(area_id) != neighbour_connections.end())
        {
            return neighbour_connections[area_id];
        }
        throw std::out_of_range("Area ID not found in neighbour connections");
    }

    int get_visitable_tiles(){
        if (is_one_dim || tile_count == 1){
            return tile_count;
        }
        else{
            return tile_count - std::abs(coord_parity_diff);
        }
    }

    unsigned int get_edge(AreaNode *node)
    {
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [node](std::pair<AreaNode *, unsigned int> &pair)
                               { return pair.first == node; });
        if (it != edge_nodes.end())
        {
            return it->second;
        }
        return 0;
    }

    ~AreaNode() = default;
};

struct ExploreData
{
    Coord start_coord;
    int area_id;
    AreaNode *prev_node;

    ExploreData() : start_coord(Coord()),
                    area_id(-1),
                    prev_node(nullptr) {}

    ExploreData(Coord start_coord, int area_id, AreaNode *prev_node) : start_coord(start_coord),
                                                                       area_id(area_id),
                                                                       prev_node(prev_node) {}
};

struct ExploreResults
{
    int tile_count = 0;
    int food_count = 0;
    int coord_parity_diff = 0; // coord_parity_diff is positive if there are more even tiles (black), negative if more odd tiles (white)
    int max_index = -1; // max_index is the index of the snake in this area, if it is not in this area, then it is -1
    Coord max_index_coord = Coord();
    bool has_tail = false;
    bool early_exit = false;
    std::vector<ConnectedAreaInfo> connected_areas;
    std::vector<Coord> to_explore;

    ExploreResults()
    {
        connected_areas.reserve(10);
        to_explore.reserve(100);
    }

    ExploreResults(
        int tile_count,
        int food_count,
        int coord_parity_diff,
        int max_index,
        Coord max_index_coord,
        bool has_tail,
        bool early_exit,
        std::vector<ConnectedAreaInfo> connected_areas,
        std::vector<Coord> to_explore
    ) : 
        tile_count(tile_count),
        food_count(food_count),
        coord_parity_diff(coord_parity_diff),
        max_index(max_index),
        max_index_coord(max_index_coord),
        has_tail(has_tail),
        early_exit(early_exit),
        connected_areas(connected_areas),
        to_explore(to_explore) {}
};

struct TileCounts
{
    int total_tiles;
    int total_food;
    int new_tiles;

    TileCounts() : 
        total_tiles(0),
        total_food(0),
        new_tiles(0) {}

    TileCounts(int total_tiles, int total_food, int new_tiles) : 
        total_tiles(total_tiles),
        total_food(total_food),
        new_tiles(new_tiles) {}
};

struct SearchNode
{
    AreaNode *node = nullptr;
    int nr_visits = 0;
    int tiles_until_here = 0;
    int food_until_here = 0;
    
    std::vector<std::vector<unsigned int>> searched_edges;
    std::vector<unsigned int> used_edges;
    std::vector<Coord> used_coords;
    std::vector<int> entered_from_nodes;

    SearchNode() = default;

    SearchNode(AreaNode *node) : node(node) {}

    bool first_visit()
    {
        return nr_visits == 1;
    }

    bool is_visited()
    {
        return nr_visits > 0;
    }

    bool is_used_edge(unsigned int edge)
    {
        return std::find(used_edges.begin(), used_edges.end(), edge) != used_edges.end();
    }

    bool is_searched_edge(unsigned int edge)
    {
        auto searched = searched_edges.back();
        if (std::find(searched.begin(), searched.end(), edge) != searched.end())
        {
            return true;
        }
        return false;
    }

    void add_searched_edge(unsigned int edge)
    {
        searched_edges.back().push_back(edge);
    }

    void add_used_edge(unsigned int edge)
    {
        used_edges.push_back(edge);
    }

    void enter_from(AreaNode *prev_node, int tiles, int food)
    {
        auto edge = node->get_edge(prev_node);
        if (edge != 0)
        {
            add_used_edge(edge);
            entered_from_nodes.push_back(prev_node->id);
        }
        if (prev_node != nullptr){
            auto connection_info = node->get_connection_info(prev_node->id);
            used_coords.push_back(connection_info.self_coord);
        }
        nr_visits++;
        tiles_until_here = tiles;
        food_until_here = food;
        searched_edges.resize(nr_visits);
    }

    void enter_unwind(int tiles, int food)
    {
        tiles_until_here = tiles;
        food_until_here = food;
        used_edges.pop_back();
        used_coords.pop_back();
    }   

    void exit_to(AreaNode *next_node){
        auto connection_info = node->get_connection_info(next_node->id);
        used_coords.push_back(connection_info.self_coord);
    }

    void exit_unwind()
    {
        searched_edges.pop_back();
        used_edges.pop_back();
        used_coords.pop_back();
        entered_from_nodes.pop_back();
        nr_visits--;
        if (nr_visits == 0)
        {
            reset();
        }
    }

    std::pair<AreaNode *, unsigned int> get_next_node_and_edge()
    {
        for (auto &edge_node : node->edge_nodes)
        {
            if (!is_used_edge(edge_node.second) && !is_searched_edge(edge_node.second))
            {
                return edge_node;
            }
        }
        return std::make_pair(nullptr, 0);
    }

    void reset()
    {
        nr_visits = 0;
        tiles_until_here = 0;
        food_until_here = 0;
        searched_edges.clear();
        used_edges.clear();
    }

    TileCounts tile_count_on_enter()
    {
        TileCounts tile_counts;
        if (first_visit())
        {
            int new_tiles = node->get_visitable_tiles();
            tile_counts.new_tiles = new_tiles;
            tile_counts.total_food = food_until_here + node->food_count;
            tile_counts.total_tiles = tiles_until_here + new_tiles;
        }
        else
        {
            tile_counts.new_tiles = 0;
            tile_counts.total_food = food_until_here;
            tile_counts.total_tiles = tiles_until_here;
        }
        if (!entered_from_nodes.empty()){
            auto connection_info = node->get_connection_info(entered_from_nodes.back());
            if (connection_info.is_bad_gateway_from_here)
            // we are actually coming in to the current node, but the naming is from the perspective of insdide the node.
            {
                tile_counts.new_tiles -= 1;
            }

        }
        return tile_counts;
    }

    TileCounts tile_count_on_exit(AreaNode *next_node, uint8_t *s_map, int width, uint8_t food_value)
    {
        auto tile_counts = tile_count_on_enter();
        if (is_exit_coord_start_cord(next_node))
        {
            // if we are exiting through the start coord of node 0 we can ever only count one tile.
            tile_counts.total_tiles = 1;
            tile_counts.total_food = tile_has_food(s_map, width, node->start_coord, food_value) ? 1 : 0;
        }
        else if (is_exit_coord_used(next_node))
        {
            Coord tile_coord = get_entry_coord();
            tile_counts.total_tiles = tiles_until_here + 1;
            tile_counts.total_food = food_until_here + tile_has_food(s_map, width, tile_coord, food_value) ? 1 : 0;
        }
        else {
            tile_counts.total_tiles += path_tile_adjustment(next_node);
        }
        return tile_counts;
    }

    bool is_exit_coord_start_cord(AreaNode *next_node)
    {
        if (node->tile_count == 1 || node->is_one_dim){
            return false;
        }
        auto next_connection_info = node->get_connection_info(next_node->id);
        Coord exit_coord = next_connection_info.self_coord;
        DEBUG_PRINT(std::cout << "is start coord blocking exit Start coord: (" << node->start_coord.x << ", " << node->start_coord.y << "), Exit coord: (" << exit_coord.x << ", " << exit_coord.y << ")" << std::endl;);
        if (node->start_coord == exit_coord && node->id == 0){
            return true;
        }
        return false;
    }

    bool is_exit_coord_used(AreaNode *next_node)
    // returns true if the entry coord is the same as the exit coord for the next node and the area has more than 1 tile
    {
        if (node->tile_count == 1 || node->is_one_dim){
            return false;
        }
        auto next_connection_info = node->get_connection_info(next_node->id);
        Coord exit_coord = next_connection_info.self_coord;
        DEBUG_PRINT(std::cout << "Used coords: "; for (const auto& coord : used_coords) { std::cout << "(" << coord.x << ", " << coord.y << ") "; } std::cout << std::endl;);
        if (std::find(used_coords.begin(), used_coords.end(), exit_coord) != used_coords.end()){
            return true;
        }
        return false;
    }

    Coord get_entry_coord()
    {
        if (!entered_from_nodes.empty()) {
            auto previous_connection_info = node->get_connection_info(entered_from_nodes.back());
            return previous_connection_info.self_coord;
        }
        else {
            return node->start_coord;
        }
    }

    int path_parity_tile_adjustment(Coord start, Coord end){
        if (start == end || node->is_one_dim){
            return 0;
        }

        bool start_parity = get_coord_mod_parity(start);
        bool end_parity = get_coord_mod_parity(end);
        if(node->coord_parity_diff == 0){
            if (start_parity != end_parity){
                return 0;
            }
            else{
                return -1;
            }
        }
        else if(node->coord_parity_diff > 0){
            // more even tiles
            if (!start_parity && !end_parity){
                return -1;
            }
            else if (start_parity && end_parity){
                return 1;
            }
            else{
                return 0;
            }
        }
        else{
            // more odd tiles
            if (start_parity && end_parity){
                return -1;
            }
            else if (!start_parity && !end_parity){
                return 1;
            }
            else{
                return 0;
            }
        }
    }

    int max_index_tile_adjustment()
    {
        Coord entry_coord = get_entry_coord();
        if (node->is_one_dim || 
            node->max_index == -1 || 
            entry_coord == node->max_index_coord ||
            node->max_index_coord == Coord()
            )
        {
            return 0;
        }
        return path_parity_tile_adjustment(entry_coord, node->max_index_coord);

    }

    int path_tile_adjustment(AreaNode *next_node)
    // depending of the path in nodes the amount of visitable tiles may change
    // more complex logic could possibly be implemented here sp that the exact number of visitable tiles
    // is more accurate, now it might be a bit lower than reality.
    {
        auto next_connection_info = node->get_connection_info(next_node->id);
        Coord entry_coord = get_entry_coord();
        Coord exit_coord = next_connection_info.self_coord;
        DEBUG_PRINT(std::cout << "Entry coord: (" << entry_coord.x << ", " << entry_coord.y << "), Exit coord: (" << exit_coord.x << ", " << exit_coord.y << ")" << std::endl;);
        if (node->is_one_dim || entry_coord == exit_coord)
        {
            DEBUG_PRINT(std::cout << "Path tile adjustment from node " << node->id << " to node " << next_node->id << " is 0 (one_dim or same coord)" << std::endl;);
            return 0;
        }
        int adjustment = 0;
        if (next_connection_info.is_bad_gateway_from_here)
        {
            adjustment -= 1;
        }
        adjustment += path_parity_tile_adjustment(entry_coord, exit_coord);
        DEBUG_PRINT(std::cout << "Path tile adjustment from node " << node->id << " to node " << next_node->id << " is " << adjustment << std::endl;);
        return adjustment;
    }

};

class AreaGraph
{
public:
    int next_id = 0;
    int map_width = 0;
    int map_height = 0;
    AreaNode *root = nullptr;
    std::unordered_map<int, std::shared_ptr<AreaNode>> nodes;

    AreaGraph()
    {
        nodes.reserve(200);
    }

    AreaGraph(int width, int height) : map_width(width),
                                       map_height(height)
    {
        nodes.reserve(200);
    }

    void connect_nodes(int id1, int id2, ConnectedAreaInfo conn_info)
    {
        AreaNode *node1 = get_node(id1);
        AreaNode *node2 = get_node(id2);
        if (node1 == nullptr || node2 == nullptr || node1->id == node2->id)
        {
            return;
        }
        // Create ConnectedAreaInfo for the other node, so coords and bad_gateway are swapped.
        ConnectedAreaInfo conn_info2 = ConnectedAreaInfo(
            id1,
            conn_info.other_coord,
            conn_info.self_coord,
            conn_info.is_bad_gateway_to_here,
            conn_info.is_bad_gateway_from_here
        );
        this->connect_nodes(node1, node2, conn_info, conn_info2);
    }

    void connect_nodes(AreaNode *node1, AreaNode *node2, ConnectedAreaInfo conn_info1, ConnectedAreaInfo conn_info2)
    // conn_info1 has info about the connection from node1 to node2 and conn_info2 vice versa
    {
        if (node1 == nullptr || node2 == nullptr || node1->id == node2->id)
        {
            return;
        }
        if (node1->id > node2->id)
        {
            std::swap(node1, node2);
            std::swap(conn_info1, conn_info2);
        }
        auto edge_id = cantor_pairing(node1->id, node2->id);
        node1->add_connection(node2, edge_id, conn_info1);
        node2->add_connection(node1, edge_id, conn_info2);
    }

    AreaNode *get_node(int id)
    {
        if (nodes.find(id) == nodes.end())
        {
            return nullptr;
        }
        return nodes[id].get();
    }

    AreaNode *add_node(Coord start_coord)
    {
        return add_node_with_id(start_coord, next_id++);
    }

    AreaNode *add_node_with_id(Coord start_coord, int id)
    {
        auto new_node = std::make_shared<AreaNode>(start_coord, id);
        auto new_node_ptr = new_node.get();
        nodes[id] = new_node;
        if (id == 0)
        {
            root = new_node_ptr;
        }
        return new_node_ptr;
    }

    void add_id_for_node(int original_id, int linked_id)
    {
        auto original_node_it = nodes.find(original_id);
        if (original_node_it == nodes.end())
        {
            throw std::out_of_range("Original node ID not found");
        }
        // Map the new id to the same AreaNode object (no copy)
        nodes[linked_id] = original_node_it->second;
    }

    void remove_node(int id)
    {
        auto node = nodes[id].get();
        for (auto &edge_node : node->edge_nodes)
        {
            edge_node.first->remove_connection(node);
        }
        nodes.erase(id);
    }

    void print_nodes_debug() const {
        for (const auto& node_pair : nodes) {
            const AreaNode* node = node_pair.second.get();
            std::cout 
                << "Node ID: " << node->id  << std::endl
                << "Tile Count: " << node->tile_count  << std::endl
                << "Food Count: " << node->food_count  << std::endl
                << "Even or Odd balance: " << node->coord_parity_diff << std::endl
                << "Max Index: " << node->max_index  << std::endl
                << "Has Tail: " << node->has_tail  << std::endl
                << "Is One Dim: " << node->is_one_dim  << std::endl
                << "Start Coord: (" << node->start_coord.x << ", " << node->start_coord.y << ")" << std::endl
                << "End Coord: (" << node->end_coord.x << ", " << node->end_coord.y << ")" << std::endl
                << "Max Index Coord: (" << node->max_index_coord.x << ", " << node->max_index_coord.y << ")" << std::endl
                << "Connections: " << std::endl;
            for (const auto& conn_pair : node->neighbour_connections) {
                int connected_area_id = conn_pair.first;
                const ConnectedAreaInfo& info = conn_pair.second;
                std::cout << "    Connected to area: " << connected_area_id
                        << " [Self: (" << info.self_coord.x << ", " << info.self_coord.y << ")"
                        << " Other: (" << info.other_coord.x << ", " << info.other_coord.y << ")]"
                        << " Bad gateway: " << info.is_bad_gateway_from_here
                        << std::endl;
            }
            std::cout << std::endl;
        }
    }

    AreaCheckResult search_best2(int snake_length, uint8_t *s_map, uint8_t food_value, int width, int target_margin, bool food_check, bool exhaustive)
    {
        bool forward = true;
        bool skipped_one = false;
        // Map to keep track of visited nodes
        std::unordered_map<AreaNode *, SearchNode> search_nodes_data;
        for (auto &node : nodes)
        {
            search_nodes_data[node.second.get()] = SearchNode(node.second.get());
        }
        // pair(cantor_pairing of from_node - to_node, needed_steps), are the elements to be cached
        std::vector<SearchNode *> search_stack;
        std::vector<int> total_tile_count_stack;
        std::vector<int> total_food_count_stack;
        total_food_count_stack.reserve(100);
        total_tile_count_stack.reserve(100);
        search_stack.reserve(100);
        search_stack.push_back(&search_nodes_data[root]);
        AreaCheckResult best_result;
        AreaNode *current_node = nullptr;
        AreaNode *prev_node = nullptr;

        while (!search_stack.empty())
        {
            AreaCheckResult current_result;
            SearchNode *step_data = search_stack.back();
            current_node = step_data->node;
            int tiles_before = total_food_count_stack.empty() ? 0 : total_tile_count_stack.back();
            int food_before = total_food_count_stack.empty() ? 0 : total_food_count_stack.back();
            if (forward)
            {
                if (!skipped_one)
                {
                    step_data->enter_from(prev_node, tiles_before, food_before);
                }
                skipped_one = false;
            }
            else
            {
                step_data->enter_unwind(tiles_before, food_before);
            }

            // if this is not the first visit to this node, then countable tiles are 0
            auto curr_tile_counts = step_data->tile_count_on_enter();

            int needed_steps;
            int margin;
            int total_steps;
            int calc_tiles;
            int calc_food;
            if (step_data->node->has_tail && !food_check)
            {
                best_result.has_tail = true;
                best_result.margin = INT_MAX;
                best_result.is_clear = true;
                best_result.tile_count = curr_tile_counts.total_tiles;
                best_result.food_count = curr_tile_counts.total_food;
                break;
            }

            if (step_data->node->max_index >= 0)
            {
                calc_tiles = curr_tile_counts.total_tiles + step_data->max_index_tile_adjustment();
                calc_food = curr_tile_counts.total_food;
                total_steps = calc_tiles - calc_food;
                needed_steps = snake_length - step_data->node->max_index;
                margin = total_steps - needed_steps;
            }
            else
            {
                calc_tiles = curr_tile_counts.new_tiles;
                calc_food = curr_tile_counts.total_food;
                total_steps = calc_tiles - calc_food;
                needed_steps = snake_length + 1;
                margin = total_steps - needed_steps;
            }
            current_result.margin = margin;
            current_result.total_steps = total_steps;
            current_result.tile_count = calc_tiles;
            current_result.food_count = calc_food;
            current_result.needed_steps = needed_steps;
            if (current_result.margin >= 0)
            {
                current_result.is_clear = true;
            }
            
            if (food_check)
            {
                if (current_result.margin > current_result.food_count && (current_result.food_count >= best_result.food_count))
                {
                    best_result = current_result;
                }
            }
            else
            {
                if (current_result.margin > best_result.margin)
                {
                    best_result = current_result;
                }
                if ((best_result.margin >= target_margin && best_result.margin > best_result.food_count) && !exhaustive)
                {
                    break;
                }
            }
            
            DEBUG_PRINT(std::cout << "\n####### NODE #######" << std::endl;);
            DEBUG_PRINT(std::cout << (forward ? "--> Forward" : "<-- Backward") << std::endl;);
            DEBUG_PRINT(std::cout << "nr_visits: " << step_data->nr_visits << std::endl;);
            DEBUG_PRINT(std::cout << "max_index: " << step_data->node->max_index << std::endl;);
            DEBUG_PRINT(std::cout << "Current node: " << current_node->id << std::endl;);
            DEBUG_PRINT(std::cout << "start coord: (" << current_node->start_coord.x << ", " << current_node->start_coord.y << ")" << std::endl;);
            DEBUG_PRINT(std::cout << "end coord: (" << current_node->end_coord.x << ", " << current_node->end_coord.y << ")" << std::endl;);
            DEBUG_PRINT(std::cout << "max index coord: (" << current_node->max_index_coord.x << ", " << current_node->max_index_coord.y << ")" << std::endl;);
            DEBUG_PRINT(std::cout << "node tile count: " << current_node->tile_count << std::endl;);
            DEBUG_PRINT(std::cout << "node food count: " << current_node->food_count << std::endl;);
            DEBUG_PRINT(std::cout << "coord parity diff: " << current_node->coord_parity_diff << std::endl;);
            DEBUG_PRINT(std::cout << "is one dim: " << current_node->is_one_dim << std::endl;);
            DEBUG_PRINT(std::cout << "has tail: " << current_node->has_tail << std::endl;);
            DEBUG_PRINT(std::cout << "Tiles before: " << tiles_before << std::endl;);
            DEBUG_PRINT(std::cout << "Food before: " << food_before << std::endl;);
            DEBUG_PRINT(std::cout << "current node new tiles: " << curr_tile_counts.new_tiles << std::endl;);
            DEBUG_PRINT(std::cout << "tiles until here: " << step_data->tiles_until_here << std::endl;);
            DEBUG_PRINT(std::cout << "food until here: " << step_data->food_until_here << std::endl;);
            DEBUG_PRINT(std::cout << "needed steps: " << needed_steps << std::endl;);
            DEBUG_PRINT(std::cout << "total steps: " << total_steps << std::endl;);
            DEBUG_PRINT(std::cout << "margin: " << margin << std::endl;);
            DEBUG_PRINT(std::cout << "searched edges now: ";);
            DEBUG_PRINT(for(auto edge : step_data->searched_edges.back()){ std::cout << edge << ", "; });
            DEBUG_PRINT(std::cout << std::endl;);
            DEBUG_PRINT(std::cout << "used edges: ";);
            DEBUG_PRINT(for(auto edge : step_data->used_edges){ std::cout << edge << ", "; });
            DEBUG_PRINT(std::cout << std::endl;);
            DEBUG_PRINT(std::cout << "edge nodes: ";);
            DEBUG_PRINT(for(auto edge_node : step_data->node->edge_nodes){ std::cout << "(" << edge_node.first->id << ", " << edge_node.second << "), "; });
            DEBUG_PRINT(std::cout << std::endl;);
            DEBUG_PRINT(std::cout << "search stack: (";);
            DEBUG_PRINT(for(auto node : search_stack){ std::cout << node->node->id << ", "; });
            DEBUG_PRINT(std::cout << ")" << std::endl;);
            DEBUG_PRINT(std::cout << "total tile count stack: (";);
            DEBUG_PRINT(for(auto count : total_tile_count_stack){ std::cout << count << ", "; });
            DEBUG_PRINT(std::cout << ")" << std::endl;);
            DEBUG_PRINT(std::cout << std::endl;);
            DEBUG_PRINT(std::cout << "current result: \n";);
            DEBUG_PRINT(std::cout << "  is clear: " << current_result.is_clear << std::endl;);
            DEBUG_PRINT(std::cout << "  tile count: " << current_result.tile_count << std::endl;);
            DEBUG_PRINT(std::cout << "  food count: " << current_result.food_count << std::endl;);
            DEBUG_PRINT(std::cout << "  needed steps: " << current_result.needed_steps << std::endl;);
            DEBUG_PRINT(std::cout << "  margin: " << current_result.margin << std::endl;);
            DEBUG_PRINT(std::cout << "  total steps: " << current_result.total_steps << std::endl;);
            DEBUG_PRINT(std::cout << "  has tail: " << current_result.has_tail << std::endl;);
            DEBUG_PRINT(std::cout << "best result: \n";);
            DEBUG_PRINT(std::cout << "  is clear: " << best_result.is_clear << std::endl;);
            DEBUG_PRINT(std::cout << "  tile count: " << best_result.tile_count << std::endl;);
            DEBUG_PRINT(std::cout << "  food count: " << best_result.food_count << std::endl;);
            DEBUG_PRINT(std::cout << "  needed steps: " << best_result.needed_steps << std::endl;);
            DEBUG_PRINT(std::cout << "  margin: " << best_result.margin << std::endl;);
            DEBUG_PRINT(std::cout << "  total steps: " << best_result.total_steps << std::endl;);
            DEBUG_PRINT(std::cout << "  has tail: " << best_result.has_tail << std::endl;);


            auto node_edge_pair = step_data->get_next_node_and_edge();
            if (node_edge_pair.first != nullptr)
            {

                forward = true;
                auto next_node = node_edge_pair.first;
                auto next_step_data = &search_nodes_data[next_node];
                // this check is important, if we have visited a node with only 1 tile, then we can not visit it again
                if (next_step_data->is_visited())
                {
                    if (next_step_data->node->tile_count == 1)
                    {
                        step_data->add_searched_edge(node_edge_pair.second);
                        skipped_one = true;
                        continue;
                    }
                }
                // std::cout << "Next node: " << next_node->id << std::endl;
                step_data->add_searched_edge(node_edge_pair.second);
                step_data->add_used_edge(node_edge_pair.second);
                search_stack.push_back(next_step_data);
                TileCounts tile_counts_on_exit = step_data->tile_count_on_exit(next_node, s_map, width, food_value);
                total_tile_count_stack.push_back(tile_counts_on_exit.total_tiles);
                total_food_count_stack.push_back(tile_counts_on_exit.total_food);
                step_data->exit_to(next_node);
            }
            else
            {
                forward = false;
                search_stack.pop_back();
                total_food_count_stack.pop_back();
                total_tile_count_stack.pop_back();
                if (search_stack.empty())
                {
                    break;
                }
                step_data->exit_unwind();
            }
            prev_node = current_node;
        }
        return best_result;
    }

    ~AreaGraph() = default;
};

class AreaChecker
{
public:
    // Constructor to initialize food_value, free_value, and body_value
    AreaChecker(
        uint8_t food_value, 
        uint8_t free_value, 
        uint8_t body_value, 
        uint8_t head_value, 
        int width, 
        int height
    ) : 
        food_value(food_value),
        free_value(free_value),
        body_value(body_value),
        head_value(head_value),
        width(width),
        height(height) {}

    bool is_inside(int x, int y)
    {
        return !(x < 0 || y < 0 || x >= this->width || y >= this->height);
    }

    void print_map(uint8_t *s_map)
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

    bool _is_bad_gateway(uint8_t *s_map, Coord coord1, Coord coord2)
    // Checking if going through a 'gateway' creates an unvisitable tile.
    // below is a diagram of the situation we are checking for
    /*
    X X 2 X X
    X . 1 . X
    */
    {
        const int cols = this->width;
        const int coord1_x = coord1.x;
        const int coord1_y = coord1.y;
        const int coord2_x = coord2.x;
        const int coord2_y = coord2.y;

        const int delta_y = coord1_y - coord2_y;
        const int delta_x = coord1_x - coord2_x;
        if(std::abs(delta_x) + std::abs(delta_y) != 1)
        {
            throw std::invalid_argument("coord1 and coord2 must be adjacent");
        }
        
        const Coord n_r1 = Coord(coord1_x + delta_y, coord1_y + delta_x);
        const Coord n_r2 = Coord(coord1_x + (delta_y * 2), coord1_y + (delta_x * 2));
        const Coord n_l1 = Coord(coord1_x - delta_y, coord1_y - delta_x);
        const Coord n_l2 = Coord(coord1_x - (delta_y * 2), coord1_y - (delta_x * 2));

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

    bool is_single_entrance(py::array_t<uint8_t> s_map, py::tuple coord, py::tuple check_coord)
    {
        auto buf = s_map.request();
        uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
        return _is_single_entrance(
            ptr,
            Coord(coord[0].cast<int>(), coord[1].cast<int>()),
            Coord(check_coord[0].cast<int>(), check_coord[1].cast<int>()));
    }

    int _is_single_entrance(uint8_t *s_map, Coord coord, Coord check_coord)
    {
        // return code 2 is for a passage like:
        // x . .
        // . . .
        // . . x
        // or flipped
        const int cols = this->width;
        const int c_x = coord.x;
        const int c_y = coord.y;
        const int ch_x = check_coord.x;
        const int ch_y = check_coord.y;
        static constexpr int offmap_value = -1;

        const int delta_y = ch_y - c_y;
        const int delta_x = ch_x - c_x;
        int value;

        const std::array<Coord, 8> ch_neighbours = {
            Coord(ch_x - 1, ch_y - 1),
            Coord(ch_x, ch_y - 1),
            Coord(ch_x + 1, ch_y - 1),
            Coord(ch_x + 1, ch_y),
            Coord(ch_x + 1, ch_y + 1),
            Coord(ch_x, ch_y + 1),
            Coord(ch_x - 1, ch_y + 1),
            Coord(ch_x - 1, ch_y)};
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
            Coord(c_x - 1, c_y - 1),
            Coord(c_x, c_y - 1),
            Coord(c_x + 1, c_y - 1),
            Coord(c_x + 1, c_y),
            Coord(c_x + 1, c_y + 1),
            Coord(c_x, c_y + 1),
            Coord(c_x - 1, c_y + 1),
            Coord(c_x - 1, c_y)};

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

        int x = ch_x + delta_y;
        int y = ch_y + delta_x;
        if (this->is_inside(x, y))
        {
            value = s_map[y * cols + x];
            if (value <= this->free_value)
            {
                x = c_x + delta_y;
                y = c_y + delta_x;
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

        x = ch_x - delta_y;
        y = ch_y - delta_x;
        if (this->is_inside(x, y))
        {
            value = s_map[y * cols + x];
            if (value <= this->free_value)
            {
                x = c_x - delta_y;
                y = c_y - delta_x;
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

    py::dict area_check(
        py::array_t<uint8_t> s_map,
        py::list body_coords_py,
        py::tuple start_coord_py,
        int target_margin,
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

            AreaCheckResult result = _area_check2(
                s_map_ptr,
                body_coords,
                start_coord,
                target_margin,
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

    ExploreResults explore_area(
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
        Coord max_index_coord;
        int coord_parity_diff = 0;
        bool has_tail = false;
        bool did_early_exit = false;
        std::vector<Coord> to_explore;
        std::vector<ConnectedAreaInfo> connected_areas;
        int checked_value = checked[start_coord.y * width + start_coord.x];
        if (checked_value != unexplored_area_id && checked_value != area_id)
        {
            return ExploreResults(
                tile_count, 
                food_count, 
                coord_parity_diff,
                max_index, 
                max_index_coord,
                has_tail, 
                did_early_exit, 
                connected_areas, 
                to_explore
            );
        }
        tile_count += 1;
        coord_parity_diff += get_coord_mod_parity(start_coord) ? 1 : -1;
        size_t body_len = body_coords.size();
        auto tail_coord = body_coords[body_len - 1];
        std::deque<Coord> current_coords;
        current_coords.push_back(start_coord);
        checked[start_coord.y * width + start_coord.x] = area_id;
        // std::cout << "Start coord: (" << start_coord.x << ", " << start_coord.y << ")" << std::endl;
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
                            connected_areas.push_back(
                                ConnectedAreaInfo(
                                    checked_val, 
                                    curr_coord, 
                                    n_coord, 
                                    is_bad_gateway_from_here, 
                                    is_bad_gateway_to_here
                                )
                            );
                        }
                    }
                    continue;
                }
                int n_coord_val = s_map[n_y * width + n_x];
                if (n_coord_val == free_value || n_coord_val == food_value)
                {
                    int entrance_code = _is_single_entrance(s_map, curr_coord, n_coord);
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
                else if ((n_coord_val == body_value || n_coord_val == head_value) && !(curr_coord == start_coord && area_id == 0))
                {
                    auto it = std::find(body_coords.begin(), body_coords.end(), n_coord);
                    if (it != body_coords.end())
                    {
                        int body_index = static_cast<int>(std::distance(body_coords.begin(), it)); // Cast to int
                        if (body_index > max_index)
                        {
                            max_index = body_index;
                            max_index_coord = curr_coord;
                        }
                    }
                    if (n_coord == tail_coord)
                    {
                        max_index = body_len - 1;
                        max_index_coord = curr_coord;
                        has_tail = true;
                    }
                }
            }
            int calc_target_margin = std::max(std::max(target_margin, food_count + total_food_count), 1);
            int total_steps = tile_count - (food_count + total_food_count) - std::abs(coord_parity_diff);
            int needed_steps = (max_index > 0) ? snake_length - max_index : snake_length + 1;
            int margin = total_steps - needed_steps;
            if (early_exit && margin > calc_target_margin * 2)
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
            max_index, 
            max_index_coord,
            has_tail, 
            did_early_exit, 
            connected_areas, 
            to_explore
        );

    }

    AreaCheckResult _area_check2(
        uint8_t *s_map,
        std::vector<Coord> &body_coords,
        Coord &start_coord,
        int target_margin,
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
            auto result = explore_area(
                s_map,
                body_coords,
                current_coord,
                current_id,
                checked,
                !(food_check || exhaustive),
                body_coords.size(),
                target_margin,
                total_food_count);
            total_food_count += result.food_count;

            if (result.tile_count == 0)
            {
                continue;
            }
            // If an area has just one tile, no max index and only one area to explore, then we can just add the tile to the previous node
            if (
                prev_node != nullptr && 
                prev_node->is_one_dim && 
                result.tile_count == 1 && 
                result.max_index < 0 &&
                (result.connected_areas.size() + result.to_explore.size() == 2)
            )
            {
                current_node = prev_node;
                graph.add_id_for_node(prev_node->id, current_id);
                current_node->tile_count += result.tile_count;
                current_node->food_count += result.food_count;
                current_node->coord_parity_diff += result.coord_parity_diff;
                current_node->max_index = result.max_index;
                current_node->has_tail = result.has_tail;
                current_node->end_coord = current_coord;
            }
            else
            {
                current_node = graph.add_node_with_id(current_coord, current_id);
                current_node->tile_count = result.tile_count;
                current_node->food_count = result.food_count;
                current_node->coord_parity_diff = result.coord_parity_diff;
                current_node->max_index = result.max_index;
                current_node->has_tail = result.has_tail;
                current_node->max_index_coord = result.max_index_coord;
                if (current_node->tile_count == 1 && result.max_index <= 0 && (result.connected_areas.size() + result.to_explore.size() <= 2))
                { // a node cant really have 2 or 3 tiles, next step after 1 is 4, but anyways...
                    current_node->is_one_dim = true;
                }
            }

            if (
                prev_node != nullptr &&
                (prev_node->tile_count == 1 || prev_node->is_one_dim) &&
                current_node->max_index < 0 &&
                prev_node->max_index == 0 &&
                (prev_node->get_nr_connections() <= 2)
            )
            { 
                // this is to propagate max_index from the start to the first node with more than two connections
                // it is needed if the start is in a 1 tile narrow passage and leads to a juntion.
                current_node->max_index = 0;
            }

            for (auto connected_area : result.connected_areas)
            {
                if (graph.get_node(connected_area.id) != nullptr)
                {
                    graph.connect_nodes(current_node->id, connected_area.id, connected_area);
                }
            }
            for (auto &area_start_coord : result.to_explore)
            {
                areas_to_explore.push_back(ExploreData(area_start_coord, graph.next_id++, current_node));
            }
        }
        
        DEBUG_PRINT(graph.print_nodes_debug());

        return graph.search_best2(body_coords.size(), s_map, food_value, width, target_margin, food_check, exhaustive);
    }

private:
    uint8_t food_value;
    uint8_t free_value;
    uint8_t body_value;
    uint8_t head_value;
    int width;
    int height;
    Coord print_mark;
    int unexplored_area_id = -1;
    // ThreadPool thread_pool = ThreadPool(std::thread::hardware_concurrency());
};

PYBIND11_MODULE(area_check, m)
{
    py::class_<AreaChecker>(m, "AreaChecker")
        .def(py::init<int, int, int, int, int, int>())
        .def("is_single_entrance", &AreaChecker::is_single_entrance)
        .def("area_check", &AreaChecker::area_check,
             py::arg("s_map"),
             py::arg("body_coords_py"),
             py::arg("start_coord_py"),
             py::arg("target_margin"),
             py::arg("food_check"),
             py::arg("exhaustive"));

    py::class_<Coord>(m, "Coord")
        .def(py::init<int, int>())
        .def_readwrite("x", &Coord::x)
        .def_readwrite("y", &Coord::y)
        .def("__getitem__", [](const Coord &c, size_t i)
             {
            if (i >= 2) throw std::out_of_range("Index out of range");
            return c[i]; })
        .def("__setitem__", [](Coord &c, size_t i, int v)
             {
            if (i >= 2) throw std::out_of_range("Index out of range");
            c[i] = v; });
}
