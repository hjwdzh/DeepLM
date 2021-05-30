#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "kernel.h"

namespace py = pybind11;

PYBIND11_MODULE(LMCore, m) {
	m.def("SquareDot", &SquareDot);
	m.def("JacobiBlockJtJ", &JacobiBlockJtJ);
	m.def("JacobiRightMultiply", &JacobiRightMultiply);
	m.def("JacobiLeftMultiply", &JacobiLeftMultiply);
	m.def("ListRightMultiply", &ListRightMultiply);
	m.def("JacobiColumnSquare", &JacobiColumnSquare);
	m.def("ColumnInverseSquare", &ColumnInverseSquare);
	m.def("JacobiNormalize", &JacobiNormalize);
}
