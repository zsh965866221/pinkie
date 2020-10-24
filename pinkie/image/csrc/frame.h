#ifndef PINKIE_IMAGE_CSRC_FRAME_H
#define PINKIE_IMAGE_CSRC_FRAME_H

#include <string>
#include <Eigen/Eigen>

namespace pinkie {

class Frame {
public:
  Frame();
  Frame(const Frame& frame);
  virtual ~Frame() {}

public:
  void set_origin(const Eigen::Vector3f& origin);
  void set_spacing(const Eigen::Vector3f& spacing);
  void set_axes(const Eigen::Matrix3f& axes);
  void set_axis(const Eigen::Vector3f& axis, const size_t index);

public:
  const Eigen::Vector3f& origin() const;
  const Eigen::Vector3f& spacing() const;
  const Eigen::Matrix3f& axes() const;
  Eigen::Vector3f axis(const size_t index) const;

public:
  void reset();

public:
  Eigen::Vector3f world_to_voxel(const Eigen::Vector3f& world) const;
  Eigen::Vector3f voxel_to_world(const Eigen::Vector3f& voxel) const;

private:
  Eigen::Vector3f origin_;
  Eigen::Vector3f spacing_;
  Eigen::Matrix3f axes_;
};

} // namespace pinkie

#endif // PINKIE_IMAGE_CSRC_FRAME_H