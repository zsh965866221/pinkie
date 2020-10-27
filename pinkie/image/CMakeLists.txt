add_subdirectory(csrc)
add_subdirectory(pywrappers)

if (USE_CUDA)
  add_subdirectory(cuda)
endif()

if (BUILD_TEST)
  add_subdirectory(test)
endif()