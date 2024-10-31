#pragma once

#include "types.h"
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <optional>
#include <pcl/point_cloud.h>

namespace lio_sam {

const int queueLength = 2000;

class ImageProjection {
private:
  std::mutex imuLock;
  std::mutex odoLock;

  LioSamParams params_;

  std::deque<std::pair<double, Eigen::Vector3d>> gyroQueue;
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

  CloudInfo cloudInfo;
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

  void odometryHandler(const Odometry &odometryMsg);
  std::optional<CloudInfo>
  cloudHandler(double stamp,
               const pcl::PointCloud<PointXYZIRT>::Ptr laserCloudMsg);
};

} // namespace lio_sam