add_library(
  pinkie_transform_python
  MODULE
  transform.cpp
)
target_link_libraries(
  pinkie_transform_python 
  pythonlibs
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  ${EXTRA_LIBS}
  pinkie_transform
)
set_target_properties(
  pinkie_transform_python
  PROPERTIES 
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
