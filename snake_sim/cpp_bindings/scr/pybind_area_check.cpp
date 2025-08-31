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
        .def("is_single_entrance", &AreaChecker::is_single_entrance)
        .def("area_check", &AreaChecker::area_check,
            py::arg("s_map"),
            py::arg("body_coords_py"),
            py::arg("start_coord_py"),
            py::arg("target_margin"),
            py::arg("max_food"),
            py::arg("food_check"),
            py::arg("exhaustive")
        );

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
