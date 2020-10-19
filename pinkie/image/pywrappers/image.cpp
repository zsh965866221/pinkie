#include "pinkie/image/csrc/frame.h"

#include <sstream>
#include <torch/extension.h>


namespace py = pybind11;

PYBIND11_MODULE(pinkie_image_python, m) {
  py::class_<pinkie::Frame>(m, "Frame")
    .def(py::init())
    .def(py::init<pinkie::Frame>())
    .def("origin", &pinkie::Frame::origin)
    .def("spacing", &pinkie::Frame::spacing)
    .def("axes", &pinkie::Frame::axes)
    .def("axis", &pinkie::Frame::axis)

    .def("set_origin", &pinkie::Frame::set_origin)
    .def("set_spacing", &pinkie::Frame::set_spacing)
    .def("set_axes", &pinkie::Frame::set_axes)
    .def("set_axis", &pinkie::Frame::set_axis)

    .def(
      "to", 
      py::overload_cast<const c10::Device&>(&pinkie::Frame::to, py::const_)
    )
    .def(
      "to", 
      py::overload_cast<const std::string&>(&pinkie::Frame::to, py::const_)
    )
    .def(
      "to_", 
      py::overload_cast<const c10::Device&>(&pinkie::Frame::to_)
    )
    .def(
      "to_", 
      py::overload_cast<const std::string&>(&pinkie::Frame::to_)
    )

    .def("device", &pinkie::Frame::device)
    
    .def(
      "__repr__",
      [] (const pinkie::Frame& frame) {
        std::stringstream stream;
        stream << 
          "<<< Frame Start >>>" << std::endl <<
          "origin: \n" << frame.origin() << std::endl <<
          "spacing:\n " << frame.spacing() << std::endl <<
          "axes: \n" << frame.axes() << std::endl <<
          "<<< Frame End >>>";
        return stream.str();
      }
    )
  ;

}

