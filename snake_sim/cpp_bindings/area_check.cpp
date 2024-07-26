#include <pybind11/pybind11.h>

namespace py = pybind11;

class AreaChecker {
    public:
        AreaChecker(const std::string &name) : name(name) {}


};

PYBIND11_MODULE(my_module, m) {
    py::class_<Snake>(m, "Snake")
        .def(py::init<const std::string &>())
        .def("set_name", &Snake::set_name)
        .def("get_name", &Snake::get_name);
}