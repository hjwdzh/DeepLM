#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "baproblem_manager.h"
#include "io.h"

namespace py = pybind11;

PYBIND11_MODULE(BACore, m) {
	// io
	m.def("LoadBALFromFile", &LoadBALFromFile);
}
