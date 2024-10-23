#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <optional>
#include <pcl/point_cloud.h>
#include <types.h>

template <typename T>
void imuRPY2rosRPY(Eigen::Quaterniond thisImuMsg, T *rosRoll, T *rosPitch,
                   T *rosYaw) {
  // https://stackoverflow.com/a/45577965
  double imuRoll, imuPitch, imuYaw;
  Eigen::Vector3d angles = Eigen::Matrix3d(thisImuMsg).eulerAngles(2, 1, 0);

  *rosRoll = angles[2];
  *rosPitch = angles[1];
  *rosYaw = angles[0];
}

const int queueLength = 2000;

class ImageProjection {
private:
  std::mutex imuLock;
  std::mutex odoLock;

  LioSamParams params_;

  std::deque<Imu> imuQueue;
  std::deque<Odometry> odomQueue;

  std::deque<std::pair<double, pcl::PointCloud<PointXYZIRT>::Ptr>> cloudQueue;
  std::pair<double, pcl::PointCloud<PointXYZIRT>::Ptr> currentCloudMsg;

  double *imuTime = new double[queueLength];
  double *imuRotX = new double[queueLength];
  double *imuRotY = new double[queueLength];
  double *imuRotZ = new double[queueLength];

  int imuPointerCur;
  bool firstPointFlag;
  Eigen::Affine3f transStartInverse;

  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
  pcl::PointCloud<PointType>::Ptr fullCloud;
  pcl::PointCloud<PointType>::Ptr extractedCloud;

  int deskewFlag;
  cv::Mat rangeMat;

  bool odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  CloudInfo<PointType> cloudInfo;
  double timeScanCur;
  double timeScanEnd;

  std::vector<int> columnIdnCountVec;

  void allocateMemory();

  void resetParameters();
  bool cachePointCloud(double stamp,
                       const pcl::PointCloud<PointXYZIRT>::Ptr laserCloudMsg);
  bool deskewInfo();
  void imuDeskewInfo();
  void odomDeskewInfo();
  void findRotation(double pointTime, float *rotXCur, float *rotYCur,
                    float *rotZCur);
  void findPosition(double relTime, float *posXCur, float *posYCur,
                    float *posZCur);
  PointType deskewPoint(PointType *point, double relTime);
  void projectPointCloud();
  void cloudExtraction();

public:
  ImageProjection(const LioSamParams &params);
  ~ImageProjection() {}

  void imuHandler(const Imu &imuMsg);
  // TODO: What does this do? It's published from the IMU node.
  void odometryHandler(const Odometry &odometryMsg);
  std::optional<CloudInfo<PointType>>
  cloudHandler(double stamp,
               const pcl::PointCloud<PointXYZIRT>::Ptr laserCloudMsg);
};