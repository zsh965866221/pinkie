add_library(
  pinkie_image_python
  MODULE
  image.cpp
)
target_link_libraries(
  pinkie_image_python 
  pythonlibs
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  ${EXTRA_LIBS}
  pinkie_image
)
set_target_properties(
  pinkie_image_python
  PROPERTIES 
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
