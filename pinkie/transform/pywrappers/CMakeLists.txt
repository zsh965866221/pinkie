add_library(
  pinkie_pytransform
  MODULE
  transform.h
  transform.cpp
)
target_link_libraries(
  pinkie_pytransform
  pinkie_transform
)