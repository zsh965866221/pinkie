#include <iostream>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(
  param,
  "test param",
  "param"
);

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  LOG(INFO) << FLAGS_param << std::endl;

  return 0;
}