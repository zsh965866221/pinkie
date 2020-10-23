add_library(
  pinkie_pyframe
  MODULE
  frame.h
  frame.cpp
)
target_link_libraries(
  pinkie_pyframe
  pinkie_image
)

add_library(
  pinkie_pyimage
  MODULE
  image.h
  image.cpp
)
target_link_libraries(
  pinkie_pyimage
  pinkie_image
)

