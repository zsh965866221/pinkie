#include "pinkie/image/csrc/frame.h"
#include "pinkie/image/csrc/image.h"

#include <sstream>
#include <torch/extension.h>


namespace py = pybind11;

PYBIND11_MODULE(pinkie_image_python, m) {
  using namespace pybind11::literals;
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
      [] (pinkie::Frame& frame, py::object device) {
        return frame.to(torch::python::detail::py_object_to_device(device));
      }
    )
    .def(
      "to_", 
      [] (pinkie::Frame& frame, py::object device) {
        frame.to_(torch::python::detail::py_object_to_device(device));
      }
    )

    .def(
      "device", 
      [] (pinkie::Frame& frame) {
        auto device = frame.device();
        py::object pyobject;
        pyobject.ptr() = THPDevice_New(device);
        return pyobject;
      }
    )

    .def(
      "world_to_voxel",
      &pinkie::Frame::world_to_voxel
    )
    .def(
      "voxel_to_world",
      &pinkie::Frame::voxel_to_world
    )

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

  py::object pydefault_dtype;
  pydefault_dtype.ptr() = THPDtype_New(torch::kFloat, "Image");
  py::object pydefault_device;
  pydefault_device.ptr() = THPDevice_New(torch::Device(torch::kCPU));

  py::class_<pinkie::Image>(m, "Image")
    .def(py::init())
    .def(py::init<const pinkie::Image&>())
    .def(
      py::init(
        [] (
          const int& height,
          const int& width,
          const int& depth,
          py::object dtype,
          py::object device
        ) {
          return pinkie::Image(
            height,
            width,
            depth,
            torch::python::detail::py_object_to_dtype(dtype),
            torch::python::detail::py_object_to_device(device)
          );
        }
      ),
      "height"_a,
      "width"_a,
      "depth"_a,
      "dtype"_a=pydefault_dtype,
      "device"_a=pydefault_device
    )

    .def("frame", &pinkie::Image::frame)
    .def("set_frame", &pinkie::Image::set_frame)

    .def("data", &pinkie::Image::data)
    .def("set_data", &pinkie::Image::set_data)

    .def(
      "to", 
      [] (pinkie::Image& image, py::object device) {
        return image.to(torch::python::detail::py_object_to_device(device));
      }
    )
    .def(
      "to_", 
      [] (pinkie::Image& image, py::object device) {
        image.to_(torch::python::detail::py_object_to_device(device));
      }
    )

    .def(
      "device", 
      [] (pinkie::Image& image) {
        auto device = image.device();
        py::object pyobject;
        pyobject.ptr() = THPDevice_New(device);
        return pyobject;
      }
    )
    .def(
      "dtype", 
      [] (pinkie::Image& image) {
        auto dtype = image.dtype();
        py::object pyobject;
        pyobject.ptr() = THPDtype_New(
          dtype, 
          "Image"
        );
        return pyobject;
      }
    )

    .def(
      "cast", 
      [] (pinkie::Image& image, py::object dtype) {
        return image.cast(torch::python::detail::py_object_to_dtype(dtype));
      }
    )
    .def(
      "cast_", 
      [] (pinkie::Image& image, py::object dtype) {
        image.cast_(torch::python::detail::py_object_to_dtype(dtype));
      }
    )

    .def("origin", &pinkie::Image::origin)
    .def("spacing", &pinkie::Image::spacing)
    .def("axes", &pinkie::Image::axes)
    .def("axis", &pinkie::Image::axis)
    .def("size", &pinkie::Image::size)

    .def(
      "world_to_voxel",
      &pinkie::Image::world_to_voxel
    )
    .def(
      "voxel_to_world",
      &pinkie::Image::voxel_to_world
    )

    .def(
      "__repr__",
      [] (const pinkie::Image& image) {
        std::stringstream stream;
        stream << 
          "<<< Image Start >>>" << std::endl <<
          "size: \n" << image.size() << std::endl <<
          "origin: \n" << image.origin() << std::endl <<
          "spacing: \n" << image.spacing() << std::endl <<
          "axes: \n" << image.axes() << std::endl <<
          "<<< Image End >>>";
        return stream.str();
      }
    )
  ;
}

