add_subdirectory(csrc)
add_subdirectory(pywrappers)

if (BUILD_TEST)
  add_subdirectory(test)
endif()