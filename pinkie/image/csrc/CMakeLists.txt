file(
  GLOB
  PINKIE_IMAGE_SRC
  frame.h
  frame.cpp
  pixel_type.h
  pixel_type.cpp
  image.h
  image.cpp
)
source_group(
  pinkie_image 
  FILES
  ${PINKIE_IMAGE_SRC}
)
add_library(
  pinkie_image
  STATIC
  ${PINKIE_IMAGE_SRC}
)
target_link_libraries(
  pinkie_image
  eigen
  ${EXTRA_LIBS}
)
