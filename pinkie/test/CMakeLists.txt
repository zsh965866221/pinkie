# gflags and glog
add_executable(
  test_gflags_glog
  test_gflags_glog.cpp
)
target_link_libraries(
  test_gflags_glog
  gflags
  glog::glog
)

# cuda
cuda_add_library(
  mat_mul
  STATIC
  mat_mul.h
  mat_mul_cuda.cu
  mat_mul_cpu.cpp
)
cuda_add_executable(
  test_cuda
  test_cuda.cpp
)
target_link_libraries(
  test_cuda
  mat_mul
)

# pybind11
cuda_add_library(
  pinkie_python_test
  MODULE
  test_python_interface.cpp
  test_python_interface.cu
)
target_link_libraries(
  pinkie_python_test 
  pythonlibs
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
)
set_target_properties(
  pinkie_python_test
  PROPERTIES 
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# torch
add_executable(
  test_torch
  test_torch.cpp
)
target_link_libraries(
  test_torch
  gflags
  glog::glog
  ${TORCH_LIBRARIES}
)

add_library(
  pinkie_python_torch
  SHARED
  test_torch_library.cpp
)
target_link_libraries(
  pinkie_python_torch 
  gflags
  glog::glog
  ${TORCH_LIBRARIES}
)
