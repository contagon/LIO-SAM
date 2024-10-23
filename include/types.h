#pragma once

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl/search/impl/search.hpp>

#include <pcl/point_cloud.h>
#include <stddef.h>
enum class SensorType { VELODYNE, OUSTER, LIVOX };

struct LioSamParams {
  SensorType sensor;
  int N_SCAN;
  int Horizon_SCAN;
  int downsampleRate;
  float lidarMinRange;
  float lidarMaxRange;

  // IMU
  // TODO: This can probably be simplified a bunch
  float imuAccNoise;
  float imuGyrNoise;
  float imuAccBiasN;
  float imuGyrBiasN;
  float imuGravity;
  float imuRPYWeight;
  std::vector<double> extRotV;
  std::vector<double> extRPYV;
  std::vector<double> extTransV;
  Eigen::Matrix3d extRot;
  Eigen::Matrix3d extRPY;
  Eigen::Vector3d extTrans;
  Eigen::Quaterniond extQRPY;

  // LOAM
  float edgeThreshold;
  float surfThreshold;
  int edgeFeatureMinValidNum;
  int surfFeatureMinValidNum;

  // voxel filter paprams
  float odometrySurfLeafSize;
  float mappingCornerLeafSize;
  float mappingSurfLeafSize;

  float z_tollerance;
  float rotation_tollerance;

  // CPU Params
  int numberOfCores;
  // double mappingProcessInterval;

  // TODO: Decide what to do with mapping stuff
  // Surrounding map
  // float surroundingkeyframeAddingDistThreshold;
  // float surroundingkeyframeAddingAngleThreshold;
  // float surroundingKeyframeDensity;
  // float surroundingKeyframeSearchRadius;

  // Loop closure
  // bool loopClosureEnableFlag;
  // float loopClosureFrequency;
  // int surroundingKeyframeSize;
  // float historyKeyframeSearchRadius;
  // float historyKeyframeSearchTimeDiff;
  // int historyKeyframeSearchNum;
  // float historyKeyframeFitnessScore;
};

template <typename PointT> struct CloudInfo {
  double timestamp;

  std::vector<std::int32_t> startRingIndex;
  std::vector<std::int32_t> endRingIndex;

  std::vector<std::int32_t> pointColInd; // point column index in range image
  std::vector<float> pointRange;         // point range

  std::int64_t imuAvailable;
  std::int64_t odomAvailable;

  // Attitude for LOAM initialization
  float imuRollInit;
  float imuPitchInit;
  float imuYawInit;

  // Initial guess from imu pre-integration
  float initialGuessX;
  float initialGuessY;
  float initialGuessZ;
  float initialGuessRoll;
  float initialGuessPitch;
  float initialGuessYaw;

  // Point cloud messages
  typename pcl::PointCloud<PointT>::Ptr
      cloud_deskewed; // original cloud deskewed
  typename pcl::PointCloud<PointT>::Ptr
      cloud_corner; // extracted corner feature
  typename pcl::PointCloud<PointT>::Ptr
      cloud_surface; // extracted surface feature

  // 3rd party messages
  typename pcl::PointCloud<PointT>::Ptr key_frame_cloud;
  typename pcl::PointCloud<PointT>::Ptr key_frame_color;
  typename pcl::PointCloud<PointT>::Ptr key_frame_poses;
  typename pcl::PointCloud<PointT>::Ptr key_frame_map;
};

typedef pcl::PointXYZI PointType;
