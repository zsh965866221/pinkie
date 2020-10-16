option(BUILD_TESTING "" OFF)
option(WITH_GFLAGS "" OFF)
option(WITH_UNWIND "" OFF)
set(GFLAGS_IS_SUBPROJECT TRUE)

option(
  USE_GLOG
  "if use GLOG, OFF default"
  ON
)
option(
  USE_GFLAGS
  "if use GFLAGS, OFF default"
  ON
)
if (USE_GLOG)
  set (EXTRA_LIBS ${EXTRA_LIBS} glog::glog)
  add_subdirectory(glog)
endif()
if (USE_GFLAGS)
  set (EXTRA_LIBS ${EXTRA_LIBS} gflags)
  add_subdirectory(gflags)
endif()

