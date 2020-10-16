option(BUILD_TESTING "" OFF)
option(WITH_GFLAGS "" OFF)
option(WITH_UNWIND "" OFF)
set(GFLAGS_IS_SUBPROJECT TRUE)

add_subdirectory(gflags)
add_subdirectory(glog)

