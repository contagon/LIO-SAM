#include <Eigen/Core>
#include <Eigen/Geometry>

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

template <typename T> Eigen::Quaterniond rpy2quat(T roll, T pitch, T yaw) {
  return AngleAxisf(yaw, Eigen::Matrix<T, 3, 1>::UnitZ()) *
         AngleAxisf(pitch, Eigen::Matrix<T, 3, 1>::UnitY()) *
         AngleAxisf(roll, Eigen::Matrix<T, 3, 1>::UnitX());
}