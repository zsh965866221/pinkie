cuda_add_library(
  pinkie_cuda_image
  STATIC
  image.h
  image.cpp
)
target_link_libraries(
  pinkie_cuda_image
  pinkie_image
)

cuda_add_executable(
  test_cuda
  test_cuda.cpp
)
