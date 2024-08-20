#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

//#include "cosseratrodmodel.h"

namespace py = pybind11;


void tendon_driven_robot_pyb(py::module_ &);
void testing(py::module_ &);
void cosserat_rod_model_pyb(py::module_ &);
void subsegmentcosserat_rod_model_pyb(py::module_ &);

PYBIND11_MODULE(tdrpyb, m) {
    m.doc() = "Binding for the TendonDrivenRobot class"; // optional module  docstring

    // TendonDrivenRobot class
   cosserat_rod_model_pyb(m);
   subsegmentcosserat_rod_model_pyb(m);
   tendon_driven_robot_pyb(m);

   testing(m);
}
