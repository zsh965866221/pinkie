option(BUILD_TESTING "" OFF)
option(WITH_GFLAGS "" OFF)
option(WITH_UNWIND "" OFF)
set(GFLAGS_IS_SUBPROJECT TRUE)

option(
  USE_GLOG
  "if use GLOG, OFF default"
  OFF
)
option(
  USE_GFLAGS
  "if use GFLAGS, OFF default"
  OFF
)
if (USE_GLOG)
  set (EXTRA_LIBS ${EXTRA_LIBS} glog::glog)
  add_subdirectory(glog)
endif()
if (USE_GFLAGS)
  set (EXTRA_LIBS ${EXTRA_LIBS} gflags)
  add_subdirectory(gflags)
endif()

set(EIGEN_DIRS "eigen")
add_library(eigen INTERFACE)
target_include_directories(
  eigen INTERFACE
  ${EIGEN_DIRS}
)
