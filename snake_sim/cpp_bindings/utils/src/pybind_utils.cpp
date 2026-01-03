#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept> // For std::out_of_range

#include "path_utils.hpp"
#include "util_types.hpp"
#include "area_utils.hpp"

namespace py = pybind11;


PYBIND11_MODULE(utils, m)
{
    m.def("get_dir_to_tile", &py_get_dir_to_tile,
        py::arg("s_map"),
        py::arg("width"),
        py::arg("height"),
        py::arg("from_coord"),
        py::arg("tile_value"),
        py::arg("visitable_values"),
        py::arg("clockwise") = true,
        py::return_value_policy::copy
    );

    m.def("distance_to_tile_with_value", &py_distance_to_tile_with_value,
        py::arg("s_map"),
        py::arg("width"),
        py::arg("height"),
        py::arg("from_coord"),
        py::arg("tile_value"),
        py::arg("visitable_values"),
        py::return_value_policy::copy
    );

    m.def("distance_to_coord", &py_distance_to_coord,
        py::arg("s_map"),
        py::arg("width"),
        py::arg("height"),
        py::arg("from_coord"),
        py::arg("to_coord"),
        py::arg("visitable_values"),
        py::return_value_policy::copy
    );

    m.def("get_visitable_tiles", &py_get_visitable_tiles,
        py::arg("s_map"),
        py::arg("width"),
        py::arg("height"),
        py::arg("center_coord"),
        py::arg("visitable_values"),
        py::return_value_policy::copy
    );

    m.def("get_locations_with_value", &py_get_locations_with_value,
        py::arg("s_map"),
        py::arg("width"),
        py::arg("height"),
        py::arg("value"),
        py::return_value_policy::copy
    );

    m.def("can_make_area_inaccessible", &py_can_make_area_inaccessible,
        py::arg("s_map"),
        py::arg("width"),
        py::arg("height"),
        py::arg("free_value"),
        py::arg("head_pos"),
        py::arg("direction"),
        py::return_value_policy::copy
    );

    m.def("dist_heat_map", &py_dist_heat_map,
        py::arg("s_map"),
        py::arg("width"),
        py::arg("height"),
        py::arg("free_value"),
        py::arg("blocked_value"),
        py::arg("target_value"),
        py::return_value_policy::take_ownership
    );

    m.def("area_boundary_tiles", &py_area_boundary_tiles,
        py::arg("s_map"),
        py::arg("width"),
        py::arg("height"),
        py::arg("free_value"),
        py::arg("area_start"),
        py::return_value_policy::copy
    );
} 