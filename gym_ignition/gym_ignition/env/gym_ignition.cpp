#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace gym_ignition;

#ifndef ENVIRONMENT_NAME
    #define ENVIRONMENT_NAME GymIgnitionEnv
#endif

PYBIND11_MODULE(_gym_ignition, m) {
    py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, IGN_MAKE_STR(ENVIRONMENT_NAME))
        .def(py::init<std::string, std::string>())
        .def("init",                    &VectorizedEnvironment<ENVIRONMENT>::init)
        .def("reset",                   &VectorizedEnvironment<ENVIRONMENT>::reset)
        .def("observe",                 &VectorizedEnvironment<ENVIRONMENT>::observe)
        .def("step",                    &VectorizedEnvironment<ENVIRONMENT>::step)
        .def("close",                   &VectorizedEnvironment<ENVIRONMENT>::close)
        .def("getExtraInfo",            &VectorizedEnvironment<ENVIRONMENT>::getExtraInfo)
        .def("isTerminalState",         &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
        .def("setSimulationTimeStep",   &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
        .def("setControlTimeStep",      &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
        .def("getObsDim",               &VectorizedEnvironment<ENVIRONMENT>::getObsDim)
        .def("getExtraInfoDim",         &VectorizedEnvironment<ENVIRONMENT>::getExtraInfoDim)
        .def("getActionDim",            &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
        .def("getNumOfEnvs",            &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
}