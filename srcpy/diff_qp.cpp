#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "diff_qp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ioc_mpc_cpp, m)
{
    m.doc() = "Differentiable QP";

    py::class_<diff_qp::DiffQP> kd (m, "DiffQP");
    kd.def(py::init<>());
    kd.def("test_func", &diff_qp::DiffQP::test_func);
}