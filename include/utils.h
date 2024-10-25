#include "types.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

inline float pointDistance(PointType p) {
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

inline float pointDistance(PointType p1, PointType p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
              (p1.z - p2.z) * (p1.z - p2.z));
}

// TODO: Test all of these to make sure they all use the same convention w/
// Euler angles!
template <typename T>
void quat2rpy(Eigen::Quaterniond thisImuMsg, T *rosRoll, T *rosPitch,
              T *rosYaw) {
  // https://stackoverflow.com/a/45577965
  double imuRoll, imuPitch, imuYaw;
  Eigen::Vector3d angles = Eigen::Matrix3d(thisImuMsg).eulerAngles(2, 1, 0);

  *rosRoll = angles[2];
  *rosPitch = angles[1];
  *rosYaw = angles[0];
}

template <typename T> Eigen::Quaternion<T> rpy2quat(T roll, T pitch, T yaw) {
  return Eigen::AngleAxis<T>(yaw, Eigen::Matrix<T, 3, 1>::UnitZ()) *
         Eigen::AngleAxis<T>(pitch, Eigen::Matrix<T, 3, 1>::UnitY()) *
         Eigen::AngleAxis<T>(roll, Eigen::Matrix<T, 3, 1>::UnitX());
}

inline Eigen::Affine3f trans2Affine3f(float transformIn[]) {
  return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5],
                                transformIn[0], transformIn[1], transformIn[2]);
}