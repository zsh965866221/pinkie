# pytorch
cuda_add_library(
  pinkie_python_test
  MODULE
  test_python_interface.h
  test_python_interface.cpp
  test_python_interface.cu
)
target_link_libraries(
  pinkie_python_test 
  pythonlibs
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  ${EXTRA_LIBS}
)
set_target_properties(
  pinkie_python_test
  PROPERTIES 
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
