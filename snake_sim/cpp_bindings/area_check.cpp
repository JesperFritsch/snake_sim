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
    int edge_tile_count;
    bool has_tail;
    int margin;
    int needed_steps;
    float margin_over_edge;
    AreaCheckResult() : is_clear(false),
                        tile_count(0),
                        total_steps(0),
                        food_count(0),
                        edge_tile_count(0),
                        has_tail(false),
                        margin(INT_MIN),
                        needed_steps(0),
                        margin_over_edge(0) {}
    AreaCheckResult(bool is_clear,
                    int tile_count,
                    int total_steps,
                    int food_count,
                    int edge_tile_count,
                    bool has_tail,
                    int margin,
                    int needed_steps,
                    float margin_over_edge) : is_clear(is_clear),
                                              tile_count(tile_count),
                                              total_steps(total_steps),
                                              food_count(food_count),
                                              edge_tile_count(edge_tile_count),
                                              has_tail(has_tail),
                                              margin(margin),
                                              needed_steps(needed_steps),
                                              margin_over_edge(margin_over_edge) {}
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
    int edge_tile_count = 0;
    int coord_parity_diff = 0;
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

    void remove_connection(AreaNode *node)
    {
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [node](std::pair<AreaNode *, unsigned int> &pair)
                               { return pair.first == node; });
        if (it != edge_nodes.end())
        {
            edge_nodes.erase(it);
            neighbour_connections.erase(node->id);
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

    void add_connection(AreaNode *node, unsigned int edge, ConnectedAreaInfo info)
    {
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [node](std::pair<AreaNode *, unsigned int> &pair)
                               { return pair.first == node; });
        if (it == edge_nodes.end())
        {
            edge_nodes.push_back(std::make_pair(node, edge));
            neighbour_connections[node->id] = info;
        }
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
        if (is_one_dim){
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
    int edge_tile_count = 0;
    int coord_parity_diff = 0; // coord_parity_diff is positive if there are more even tiles (black), negative if more odd tiles (white)
    int max_index = -1; // max_index is the index of the snake in this area, if it is not in this area, then it is -1
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
        int edge_tile_count,
        int coord_parity_diff,
        int max_index,
        bool has_tail,
        bool early_exit,
        std::vector<ConnectedAreaInfo> connected_areas,
        std::vector<Coord> to_explore) : tile_count(tile_count),
                                         food_count(food_count),
                                         edge_tile_count(edge_tile_count),
                                         coord_parity_diff(coord_parity_diff),
                                         max_index(max_index),
                                         has_tail(has_tail),
                                         early_exit(early_exit),
                                         connected_areas(connected_areas),
                                         to_explore(to_explore) {}
};

struct Additionals
{
    int tile_count;
    int food_count;
    int edge_tile_count;

    Additionals() : tile_count(0),
                    food_count(0),
                    edge_tile_count(0) {}

    Additionals(int tile_count, int food_count, int edge_tile_count) : tile_count(tile_count),
                                                                       food_count(food_count),
                                                                       edge_tile_count(edge_tile_count) {}
};

struct SearchNode
{
    AreaNode *node = nullptr;
    int nr_visits = 0;
    int tiles_until_here = 0;
    int food_until_here = 0;
    int edge_tiles_until_here = 0;
    
    std::vector<std::vector<unsigned int>> searched_edges;
    std::vector<unsigned int> used_edges;
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

    void enter_from(AreaNode *node, int tiles, int food, int edge_tiles)
    {
        auto edge = this->node->get_edge(node);
        if (edge != 0)
        {
            add_used_edge(edge);
            entered_from_nodes.push_back(node->id);
        }
        nr_visits++;
        if (first_visit())
        {
            tiles_until_here = tiles;
            food_until_here = food;
            edge_tiles_until_here = edge_tiles;
        }
        searched_edges.resize(nr_visits);
    }

    void exit()
    {
        searched_edges.pop_back();
        used_edges.pop_back();
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

    Additionals additional_tiles()
    {
        int tiles;
        int food;
        int edge_tiles;
        if (first_visit())
        {
            tiles = node->get_visitable_tiles();
            food = node->food_count;
            edge_tiles = node->edge_tile_count;
        }
        else
        {
            tiles = 0;
            food = 0;
            edge_tiles = 0;
        }
        if (!entered_from_nodes.empty()){
            auto connection_info = node->get_connection_info(entered_from_nodes.back());
            if (connection_info.is_bad_gateway_from_here)
            // we are actually coming in to the current node, but the naming is from the perspective of insdide the node.
            {
                tiles -= 1;
            }
        }
        return Additionals(tiles, food, edge_tiles);
    }

    int path_tile_adjustment(AreaNode *next_node)
    // depending of the path in nodes the amount of visitable tiles may change
    // more complex logic could possibly be implemented here sp that the exact number of visitable tiles
    // is more accurate, now it might be a bit lower than reality.
    {
        if (!entered_from_nodes.empty()) {
            auto next_connection_info = node->get_connection_info(next_node->id);
            auto previous_connection_info = node->get_connection_info(entered_from_nodes.back());
            // bool exit_coord_parity = get_coord_mod_parity(next_connection_info.self_coord);
            // bool entry_coord_parity = get_coord_mod_parity(previous_connection_info.self_coord);
            if (node->is_one_dim)
            {
               return 0;
            }
            int adjustment = 0;
            if (next_connection_info.is_bad_gateway_from_here)
            {
                adjustment -= 1;
            }
            return adjustment;
        } else {
            // No previous node, so no adjustment possible
            return 0;
        }
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

    AreaCheckResult search_best2(int snake_length, uint8_t *s_map, uint8_t food_value, int width, int target_margin, bool food_check, bool exhaustive, float safe_margin_factor)
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
        std::vector<int> total_edge_tile_count_stack;
        total_food_count_stack.reserve(100);
        total_tile_count_stack.reserve(100);
        total_edge_tile_count_stack.reserve(100);
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
            int edge_tiles_before = total_edge_tile_count_stack.empty() ? 0 : total_edge_tile_count_stack.back();
            if (forward)
            {
                if (!skipped_one)
                {
                    step_data->enter_from(prev_node, tiles_before, food_before, edge_tiles_before);
                }
                skipped_one = false;
            }
            else
            {
                step_data->used_edges.pop_back();
            }

            // if this is not the first visit to this node, then countable tiles are 0
            auto current_additional = step_data->additional_tiles();
            int current_countable_tiles = current_additional.tile_count;
            int current_countable_food = current_additional.food_count;
            int current_edge_tile_count = current_additional.edge_tile_count;

            int total_tile_count_here = current_countable_tiles + tiles_before;
            int total_food_count_here = current_countable_food + food_before;
            int total_edge_tile_count_here = current_edge_tile_count + edge_tiles_before;
            int edge_tiles_here;
            int tiles_here;
            int needed_steps;
            int margin;
            int total_steps;
            int calc_tiles;
            int calc_edge_tiles;
            int calc_food;
            if (step_data->node->has_tail && !food_check)
            {
                best_result.has_tail = true;
                best_result.margin = INT_MAX;
                best_result.is_clear = true;
                best_result.tile_count = total_tile_count_here;
                best_result.food_count = total_food_count_here;
                best_result.edge_tile_count = total_edge_tile_count_here;
                best_result.margin_over_edge = 1;
                break;
            }

            // this is how loops in the graph are handled
            if (step_data->first_visit())
            {
                tiles_here = current_node->tile_count;
                edge_tiles_here = current_node->edge_tile_count;
            }
            else
            {
                // tiles_until_here and food_until_here are only set at the first visit
                tiles_here = total_tile_count_here - step_data->tiles_until_here;
                edge_tiles_here = total_edge_tile_count_here - step_data->edge_tiles_until_here;
            }

            if (step_data->node->max_index >= 0)
            {
                calc_tiles = total_tile_count_here;
                calc_food = total_food_count_here;
                calc_edge_tiles = total_edge_tile_count_here;
                total_steps = calc_tiles - calc_food;
                needed_steps = snake_length - step_data->node->max_index;
                margin = total_steps - needed_steps;
            }
            else
            {
                calc_tiles = tiles_here;
                calc_food = total_food_count_here;
                calc_edge_tiles = edge_tiles_here;
                total_steps = calc_tiles - calc_food;
                needed_steps = snake_length + 1;
                margin = total_steps - needed_steps;
            }
            current_result.margin = margin;
            current_result.total_steps = total_steps;
            current_result.tile_count = calc_tiles;
            current_result.food_count = calc_food;
            current_result.edge_tile_count = calc_edge_tiles;
            current_result.needed_steps = needed_steps;
            current_result.margin_over_edge = (float)margin / (float)calc_edge_tiles;
            if (current_result.margin >= 0)
            {
                current_result.is_clear = true;
            }

            // std::cout << "\n####### ENTERING NODE #######" << std::endl;
            // std::cout << (forward ? "--> Forward" : "<-- Backward") << std::endl;
            // std::cout << "nr_visits: " << step_data->nr_visits << std::endl;
            // std::cout << "max_index: " << step_data->node->max_index << std::endl;
            // std::cout << "Current node: " << current_node->id << std::endl;
            // std::cout << "start coord: (" << current_node->start_coord.x << ", " << current_node->start_coord.y << ")" << std::endl;
            // std::cout << "end coord: (" << current_node->end_coord.x << ", " << current_node->end_coord.y << ")" << std::endl;
            // std::cout << "node tile count: " << current_node->tile_count << std::endl;
            // std::cout << "node food count: " << current_node->food_count << std::endl;
            // std::cout << "edge tile count: " << current_node->edge_tile_count << std::endl;
            // std::cout << "coord parity diff: " << current_node->coord_parity_diff << std::endl;
            // std::cout << "is one dim: " << current_node->is_one_dim << std::endl;
            // std::cout << "has tail: " << current_node->has_tail << std::endl;
            // std::cout << "Tiles before: " << tiles_before << std::endl;
            // std::cout << "Food before: " << food_before << std::endl;
            // std::cout << "Edge tiles before: " << edge_tiles_before << std::endl;
            // std::cout << "current node tile count: " << current_countable_tiles << std::endl;
            // std::cout << "current node food count: " << current_countable_food << std::endl;
            // std::cout << "tiles until here: " << step_data->tiles_until_here << std::endl;
            // std::cout << "food until here: " << step_data->food_until_here << std::endl;
            // std::cout << "edge tiles until here: " << step_data->edge_tiles_until_here << std::endl;
            // std::cout << "needed steps: " << needed_steps << std::endl;
            // std::cout << "total steps: " << total_steps << std::endl;
            // std::cout << "margin: " << margin << std::endl;
            // std::cout << "searched edges now: ";
            // for(auto edge : step_data->searched_edges.back()){
            //     std::cout << edge << ", ";
            // }
            // std::cout << std::endl;
            // std::cout << "used edges: ";
            // for(auto edge : step_data->used_edges){
            //     std::cout << edge << ", ";
            // }
            // std::cout << std::endl;
            // std::cout << "edge nodes: ";
            // for(auto edge_node : step_data->node->edge_nodes){
            //     std::cout << "(" << edge_node.first->id << ", " << edge_node.second << "), ";
            // }
            // std::cout << std::endl;
            // std::cout << "search stack: (";
            // for(auto node : search_stack){
            //     std::cout << node->node->id << ", ";
            // }
            // std::cout << ")" << std::endl;
            // std::cout << std::endl;
            // std::cout << "current result: \n";
            // std::cout << "  is clear: " << current_result.is_clear << std::endl;
            // std::cout << "  tile count: " << current_result.tile_count << std::endl;
            // std::cout << "  food count: " << current_result.food_count << std::endl;
            // std::cout << "  needed steps: " << current_result.needed_steps << std::endl;
            // std::cout << "  margin: " << current_result.margin << std::endl;
            // std::cout << "  total steps: " << current_result.total_steps << std::endl;
            // std::cout << "  edge tile count: " << current_result.edge_tile_count << std::endl;
            // std::cout << "  has tail: " << current_result.has_tail << std::endl;
            // std::cout << "best result: \n";
            // std::cout << "  is clear: " << best_result.is_clear << std::endl;
            // std::cout << "  tile count: " << best_result.tile_count << std::endl;
            // std::cout << "  food count: " << best_result.food_count << std::endl;
            // std::cout << "  needed steps: " << best_result.needed_steps << std::endl;
            // std::cout << "  margin: " << best_result.margin << std::endl;
            // std::cout << "  total steps: " << best_result.total_steps << std::endl;
            // std::cout << "  edge tile count: " << best_result.edge_tile_count << std::endl;
            // std::cout << "  has tail: " << best_result.has_tail << std::endl;

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
                // if we visit an area more than once we only want to add the additional tiles and food once,
                // additional_tiles() returns the additional tiles and food for the current node if it is the first visit
                // else 0 for both
                bool is_corner = false;
                bool corner_has_food = false;
                if (search_stack.size() >= 3 && current_node->tile_count > 1)
                {
                    // Check corner
                    int prev_x = search_stack[search_stack.size() - 3]->node->start_coord.x;
                    int prev_y = search_stack[search_stack.size() - 3]->node->start_coord.y;
                    int prev_x2 = search_stack[search_stack.size() - 3]->node->end_coord.x;
                    int prev_y2 = search_stack[search_stack.size() - 3]->node->end_coord.y;
                    int next_x = next_node->start_coord.x;
                    int next_y = next_node->start_coord.y;
                    int delta_x = next_x - prev_x;
                    int delta_y = next_y - prev_y;
                    int delta_x2 = next_x - prev_x2;
                    int delta_y2 = next_y - prev_y2;
                    // Check if path goes through a corner, if so then only extend the tile count stack by 1
                    // if we are currently on node 2
                    // going like this 1 -> 2 -> 3
                    /*
                        x 3 x
                        1 2 2
                        x 2 2
                    */
                    if (std::abs(delta_x) == 1 && std::abs(delta_y) == 1)
                    {
                        is_corner = true;
                        int pot_f_x = prev_x + delta_x;
                        int pot_f_y = prev_y + delta_y;
                        if (s_map[prev_y * width + pot_f_x] == food_value || s_map[pot_f_y * width + prev_x] == food_value)
                        {
                            corner_has_food = true;
                        }
                    }
                    if (prev_x2 != -1 && prev_y2 != -1 && (std::abs(delta_x2) == 1 && std::abs(delta_y2) == 1))
                    {
                        is_corner = true;
                        int pot_f_x = prev_x2 + delta_x2;
                        int pot_f_y = prev_y2 + delta_y2;
                        if (s_map[prev_y2 * width + pot_f_x] == food_value || s_map[pot_f_y * width + prev_x2] == food_value)
                        {
                            corner_has_food = true;
                        }
                    }
                    // else{
                    //     total_tile_count_stack.push_back(total_tile_count_here);
                    //     total_food_count_stack.push_back(total_food_count_here);
                    // }
                }
                else if (current_node->tile_count > 1 && !is_corner)
                {
                    int curr_x = current_node->start_coord.x;
                    int curr_y = current_node->start_coord.y;
                    int next_x = next_node->start_coord.x;
                    int next_y = next_node->start_coord.y;
                    int delta_x = next_x - curr_x;
                    int delta_y = next_y - curr_y;
                    if ((std::abs(delta_x) == 1 && std::abs(delta_y) == 0) || (std::abs(delta_x) == 0 && std::abs(delta_y) == 1))
                    {
                        is_corner = true;
                    }
                    if (s_map[curr_y * width + curr_x] == food_value)
                    {
                        corner_has_food = true;
                    }
                }
                if (is_corner)
                {
                    // std::cout << "Corner" << std::endl;
                    int tiles_before_actually = (current_node->id == 0 ? 0 : tiles_before);
                    int food_before_actually = (current_node->id == 0 ? 0 : food_before);
                    total_tile_count_stack.push_back(tiles_before_actually + 1);
                    total_food_count_stack.push_back(food_before_actually + (corner_has_food ? 1 : 0));
                    total_edge_tile_count_stack.push_back(edge_tiles_before + 1);
                }
                else
                {
                    // std::cout << "Not corner" << std::endl;
                    total_tile_count_stack.push_back(total_tile_count_here);
                    total_food_count_stack.push_back(total_food_count_here);
                    total_edge_tile_count_stack.push_back(total_edge_tile_count_here);
                }
                // int tile_adjustment = step_data->path_tile_adjustment(next_node);
                // if (tile_adjustment != 0)
                // {
                //     total_tile_count_stack.back() += tile_adjustment;
                // }
            }
            else
            {
                forward = false;
                search_stack.pop_back();
                total_food_count_stack.pop_back();
                total_tile_count_stack.pop_back();
                total_edge_tile_count_stack.pop_back();
                if (search_stack.empty())
                {
                    break;
                }
                step_data->exit();
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
    AreaChecker(uint8_t food_value, uint8_t free_value, uint8_t body_value, uint8_t head_value, int width, int height) : food_value(food_value),
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
        bool exhaustive,
        float safe_margin_factor)
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
                exhaustive,
                safe_margin_factor);
            return py::dict(
                py::arg("is_clear") = result.is_clear,
                py::arg("tile_count") = result.tile_count,
                py::arg("total_steps") = result.total_steps,
                py::arg("food_count") = result.food_count,
                py::arg("has_tail") = result.has_tail,
                py::arg("margin") = result.margin,
                py::arg("margin_over_edge") = result.margin_over_edge,
                py::arg("edge_tile_count") = result.edge_tile_count);
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
        float safe_margin_factor,
        int target_margin,
        int total_food_count)
    {
        int tile_count = 0;
        int food_count = 0;
        int edge_tile_count = 0;
        int max_index = -1;
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
                edge_tile_count, 
                coord_parity_diff,
                max_index, 
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
            bool is_edge = false;
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
                    is_edge = true;
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
                        is_edge = true;
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
                        }
                    }
                    if (n_coord == tail_coord)
                    {
                        max_index = body_len - 1;
                        has_tail = true;
                    }
                    is_edge = true;
                }
                else
                {
                    is_edge = true;
                }
            }
            if (is_edge)
            {
                edge_tile_count += 1;
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
            edge_tile_count, 
            coord_parity_diff,
            max_index, 
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
        bool exhaustive,
        float safe_margin_factor)
    {
        std::vector<int> checked;
        checked.resize(height * width);
        std::fill(checked.begin(), checked.end(), unexplored_area_id);

        AreaNode *prev_node = nullptr;
        AreaNode *current_node;
        AreaGraph graph = AreaGraph();
        std::vector<ExploreData> areas_to_explore;
        areas_to_explore.reserve(300);
        areas_to_explore.push_back(ExploreData(start_coord, graph.next_id++, prev_node));
        // in the while loop we will explore the area and add new nodes to the graph
        int total_food_count = 0;
        while (!areas_to_explore.empty())
        {
            auto current_area_data = areas_to_explore.back();
            areas_to_explore.pop_back();
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
                safe_margin_factor,
                target_margin,
                total_food_count);
            total_food_count += result.food_count;

            // std::cout << std::endl;
            // std::cout << "Explored area: " << current_id << std::endl;
            // std::cout << "Current coord: (" << current_coord.x << ", " << current_coord.y << ")" << std::endl;
            // std::cout << "Prev node: " << (prev_node == nullptr ? -1 : prev_node->id) << std::endl;
            // std::cout << "Tile count: " << result.tile_count << std::endl;
            // std::cout << "Food count: " << result.food_count << std::endl;
            // std::cout << "Edge tile count: " << result.edge_tile_count << std::endl;
            // std::cout << "Max index: " << result.max_index << std::endl;
            // std::cout << "Has tail: " << result.has_tail << std::endl;
            // std::cout << "prev is one dim: " << (prev_node == nullptr ? false : prev_node->is_one_dim) << std::endl;
            // for (auto& connected_area : result.connected_areas){
            //     std::cout << "Connected area: " << connected_area << std::endl;
            // }
            // for (auto& coord : result.to_explore){
            //     std::cout << "To explore: (" << coord.x << ", " << coord.y << ")" << std::endl;
            // }

            if (result.tile_count == 0)
            {
                continue;
            }
            // If an area has just one tile, no max index and only one area to explore, then we can just add the tile to the previous node
            if (prev_node != nullptr && prev_node->is_one_dim && result.tile_count == 1 && result.max_index < 0 && (result.connected_areas.size() + result.to_explore.size() == 2))
            {
                // std::cout << "Adding to previous node" << std::endl;
                current_node = prev_node;
                graph.add_id_for_node(prev_node->id, current_id);
                current_node->tile_count += result.tile_count;
                current_node->food_count += result.food_count;
                current_node->edge_tile_count += result.edge_tile_count;
                current_node->coord_parity_diff += result.coord_parity_diff;
                current_node->max_index = result.max_index;
                current_node->has_tail = result.has_tail;
                current_node->end_coord = current_coord;
            }
            else
            {
                // std::cout << "Adding node to graph" << std::endl;
                current_node = graph.add_node_with_id(current_coord, current_id);
                current_node->tile_count = result.tile_count;
                current_node->food_count = result.food_count;
                current_node->edge_tile_count = result.edge_tile_count;
                current_node->coord_parity_diff = result.coord_parity_diff;
                current_node->max_index = result.max_index;
                current_node->has_tail = result.has_tail;
                if (current_node->tile_count == 1)
                { // a node cant really have 2 or 3 tiles, next step after 1 is 4, but anyways...
                    current_node->is_one_dim = true;
                }
            }
            for (auto connected_area : result.connected_areas)
            {
                if (graph.get_node(connected_area.id) != nullptr)
                {
                    graph.connect_nodes(current_node->id, connected_area.id, connected_area);
                }
            }
            // std::cout << "Made connections" << std::endl;
            for (auto &area_start_coord : result.to_explore)
            {
                areas_to_explore.push_back(ExploreData(area_start_coord, graph.next_id++, current_node));
            }
        }
        
        // print the nodes and their attributes along with their connections
        // for (const auto& node_pair : graph.nodes) {
        //     const AreaNode* node = node_pair.second.get();
        //     std::cout 
        //     << "Node ID: " << node->id  << std::endl
        //     << "Tile Count: " << node->tile_count  << std::endl
        //     << "Food Count: " << node->food_count  << std::endl
        //     << "Edge Tile Count: " << node->edge_tile_count  << std::endl
        //     << "Even or Odd balance: " << node->coord_parity_diff << std::endl
        //     << "Max Index: " << node->max_index  << std::endl
        //     << "Has Tail: " << node->has_tail  << std::endl
        //     << "Is One Dim: " << node->is_one_dim  << std::endl
        //     << "Start Coord: (" << node->start_coord.x << ", " << node->start_coord.y << ")" << std::endl
        //     << "End Coord: (" << node->end_coord.x << ", " << node->end_coord.y << ")" << std::endl
        //     << "Connections: " << std::endl;
        //     for (const auto& conn_pair : node->neighbour_connections) {
        //         int connected_area_id = conn_pair.first;
        //         const ConnectedAreaInfo& info = conn_pair.second;
        //         std::cout << "    Connected to area: " << connected_area_id
        //                 << " [Self: (" << info.self_coord.x << ", " << info.self_coord.y << ")"
        //                 << " Other: (" << info.other_coord.x << ", " << info.other_coord.y << ")]"
        //                 << " Bad gateway: " << info.is_bad_gateway_from_here
        //                 << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        // std::cout << "Graph size: " << graph.nodes.size() << std::endl;
        return graph.search_best2(body_coords.size(), s_map, food_value, width, target_margin, food_check, exhaustive, safe_margin_factor);
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
             py::arg("exhaustive"),
             py::arg("safe_margin_factor"));

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
