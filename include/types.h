#pragma once
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

struct Imu {
  double stamp;
  Eigen::Vector3d gyro;
  Eigen::Vector3d acc;
  // TODO: Be rid
  Eigen::Quaternion<double> orientation;
};

struct Odometry {
  double stamp;
  Eigen::Vector3d position;
  Eigen::Quaternion<double> orientation;
};

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
  // TODO: Be rid of both of these soon enough here
  // rotation matrix of accel -> gyro ??
  Eigen::Matrix3d extRot;
  Eigen::Quaterniond extQRPY;
  // rotation matrix of lidar -> accel
  Eigen::Matrix3d extRPY;
  Eigen::Vector3d extTrans;

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
  double mappingProcessInterval;

  // TODO: Decide what to do with mapping stuff
  // Surrounding map
  float surroundingkeyframeAddingDistThreshold;
  float surroundingkeyframeAddingAngleThreshold;
  float surroundingKeyframeDensity;
  float surroundingKeyframeSearchRadius;

  // Save pcd
  bool savePCD;
  std::string savePCDDirectory;
  float resolution;

  // global map visualization radius
  float globalMapVisualizationSearchRadius;
  float globalMapVisualizationPoseDensity;
  float globalMapVisualizationLeafSize;

  // TODO: I'm rather unclear what this is doing... changing frames of some
  // sort, but I'm not 100% sure from and to what
  Imu imuConverter(const Imu &imu_in) {
    // rotate acceleration to lidar frame
    Eigen::Vector3d acc = extRot * imu_in.acc;

    // rotate gyroscope to lidar frame
    Eigen::Vector3d gyro = extRot * imu_in.gyro;

    // rotate roll pitch yaw
    Eigen::Quaterniond q_final = imu_in.orientation * extQRPY;

    if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() +
             q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1) {
      std::cout << "Invalid quaternion, please use a 9-axis IMU!" << std::endl;
      throw -1;
    }

    return Imu{.gyro = gyro, .acc = acc, .orientation = q_final};
  }
};

// TODO: Should all the pointclouds here have the same type?
template <typename PointT> struct CloudInfo {
  double stamp;

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

struct PointXYZIRT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                            intensity)(std::uint16_t, ring,
                                                       ring)(float, time, time))

struct PointXYZIRPYT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16; // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRPYT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time,
                                                                 time))

typedef PointXYZIRPYT PointTypePose;