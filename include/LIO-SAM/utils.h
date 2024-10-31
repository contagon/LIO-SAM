#pragma once

#include "types.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>

namespace lio_sam {

inline float pointDistance(PointType p) {
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

inline float pointDistance(PointType p1, PointType p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
              (p1.z - p2.z) * (p1.z - p2.z));
}

// ---------------- Conversion between SO3 types ---------------- //
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

// ---------------- Conversion between SE3 types ---------------- //
inline Eigen::Affine3f trans2Affine3f(float transformIn[]) {
  return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5],
                                transformIn[0], transformIn[1], transformIn[2]);
}

inline gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
  return gtsam::Pose3(
      gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
      gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

inline gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll),
                                          double(thisPoint.pitch),
                                          double(thisPoint.yaw)),
                      gtsam::Point3(double(thisPoint.x), double(thisPoint.y),
                                    double(thisPoint.z)));
}

inline Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
  return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

inline PointTypePose trans2PointTypePose(float transformIn[]) {
  PointTypePose thisPose6D;
  thisPose6D.x = transformIn[3];
  thisPose6D.y = transformIn[4];
  thisPose6D.z = transformIn[5];
  thisPose6D.roll = transformIn[0];
  thisPose6D.pitch = transformIn[1];
  thisPose6D.yaw = transformIn[2];
  return thisPose6D;
}

inline Odometry trans2Odometry(double stamp, float transformIn[]) {
  return Odometry{
      stamp,
      rpy2quat((double)transformIn[0], (double)transformIn[1],
               (double)transformIn[2]),
      Eigen::Vector3d(transformIn[3], transformIn[4], transformIn[5])};
}

} // namespace lio_sam