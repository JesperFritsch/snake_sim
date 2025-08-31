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
}
