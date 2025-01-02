#pragma once
#define PCL_NO_PRECOMPILE
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/src/Geometry/Quaternion.h>
#include <vector>

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

namespace lio_sam {

struct Imu {
  double stamp;
  Eigen::Vector3d gyro;
  Eigen::Vector3d acc;
};

struct Odometry {
  double stamp;
  Eigen::Quaternion<double> orientation;
  Eigen::Vector3d position;
};

struct LioSamParams {
  // REQUIRED PARAMS
  // Lidar
  int N_SCAN;
  int Horizon_SCAN;
  int downsampleRate;
  float lidarMinRange;
  float lidarMaxRange;

  // imu
  float imuAccNoise;
  float imuGyrNoise;
  float imuAccBiasN;
  float imuGyrBiasN;
  float imuGravity;
  // rotation matrix of lidar -> imu
  Eigen::Quaterniond lidar_R_imu;
  Eigen::Vector3d lidar_P_imu;

  // Optionally tuned params
  // LOAM
  float edgeThreshold = 1.0;
  float surfThreshold = 0.1;
  int edgeFeatureMinValidNum = 10;
  int surfFeatureMinValidNum = 100;

  // voxel filter params
  float odometrySurfLeafSize = 0.4;
  float mappingCornerLeafSize = 0.2;
  float mappingSurfLeafSize = 0.4;

  float z_tolerance = 1000;
  float rotation_tolerance = 1000;

  // CPU Params
  int numberOfCores = 4;
  double mappingProcessInterval = 0.15;

  // Surrounding map
  float surroundingkeyframeAddingDistThreshold = 1.0;
  float surroundingkeyframeAddingAngleThreshold = 0.2;
  float surroundingKeyframeDensity = 2.0;
  float surroundingKeyframeSearchRadius = 50.0;

  // global map visualization radius
  float globalMapVisualizationSearchRadius = 1000.0;
  float globalMapVisualizationPoseDensity = 10.0;
  float globalMapVisualizationLeafSize = 1.0;

  // Intensity params
  std::string intensity_metric = "norm0";
  std::function<double(double, double)> intensity_residual = 0;

  // Save pcd
  bool savePCD = false;
  std::string savePCDDirectory = "/tmp";
  float resolution = 0.2;
};

typedef pcl::PointXYZI PointType;

struct CloudInfo {
  double stamp;

  std::vector<std::int32_t> startRingIndex;
  std::vector<std::int32_t> endRingIndex;

  std::vector<std::int32_t> pointColInd; // point column index in range image
  std::vector<float> pointRange;         // point range

  std::int64_t imuAvailable;
  std::int64_t odomAvailable;

  // Initial guess from imu pre-integration
  float initialGuessX;
  float initialGuessY;
  float initialGuessZ;
  float initialGuessRoll;
  float initialGuessPitch;
  float initialGuessYaw;

  // Point cloud messages
  typename pcl::PointCloud<PointType> cloud_deskewed; // original cloud deskewed
  typename pcl::PointCloud<PointType> cloud_corner; // extracted corner feature
  typename pcl::PointCloud<PointType>
      cloud_surface; // extracted surface feature
};

struct PointXYZIRT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointTypeIndexed {
  PCL_ADD_POINT4D
  float index;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointTypeResidual {
  PCL_ADD_POINT4D
  float intensity_diff;
  float residual;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointTypePose {
  PCL_ADD_POINT4D
  float index;
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace lio_sam

POINT_CLOUD_REGISTER_POINT_STRUCT(
    lio_sam::PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                            intensity)(std::uint16_t, ring,
                                                       ring)(float, time, time))

POINT_CLOUD_REGISTER_POINT_STRUCT(lio_sam::PointTypeIndexed,
                                  (float, x, x)(float, y, y)(float, z,
                                                             z)(float, index,
                                                                index))

POINT_CLOUD_REGISTER_POINT_STRUCT(
    lio_sam::PointTypeResidual,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity_diff,
                                            intensity_diff)(float, residual,
                                                            residual))

POINT_CLOUD_REGISTER_POINT_STRUCT(
    lio_sam::PointTypePose,
    (float, x, x)(float, y, y)(float, z, z)(float, index, index)(
        float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time,
                                                                 time))