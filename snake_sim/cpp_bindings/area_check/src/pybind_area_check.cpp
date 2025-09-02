#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept> // For std::out_of_range

#include "area_checker.hpp"
#include "area_types.hpp"


PYBIND11_MODULE(area_check, m)
{
    py::class_<AreaChecker>(m, "AreaChecker")
        .def(py::init<int, int, int, int, int, int>())
        .def("is_gate_way", &AreaChecker::py_is_gate_way)
        .def("area_check", &AreaChecker::py_area_check,
            py::arg("s_map"),
            py::arg("body_coords_py"),
            py::arg("start_coord_py"),
            py::arg("target_margin"),
            py::arg("max_food"),
            py::arg("food_check"),
            py::arg("complete_area"),
            py::arg("exhaustive")
        );
}
