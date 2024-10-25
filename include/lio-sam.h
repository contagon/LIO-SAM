// TODO: Eventually reintroduce ROS wrapper
#include "featureExtraction.h"
#include "imageProjection.h"
#include "imuPreintegration.h"
#include "mapOptimization.h"
#include "types.h"
#include <Eigen/src/Geometry/Transform.h>

class LIOSAM {
private:
  Odometry pose;

  LioSamParams params_;

  IMUPreintegration imuPreintegrator;
  ImageProjection imageProjector;
  FeatureExtraction featureExtractor;
  MapOptimization mapOptimizer;

public:
  LIOSAM(LioSamParams params)
      : params_(params), imuPreintegrator(params), featureExtractor(params),
        imageProjector(params), mapOptimizer(params) {}
  ~LIOSAM();

  void addImuMeasurement(const Imu imuMsg) {
    // Simulate sending imu measurement to all of the nodes
    auto odometry = imuPreintegrator.imuHandler(imuMsg);
    imageProjector.imuHandler(imuMsg);

    // Everywhere the imu node sends imu_incremental to
    if (odometry.has_value()) {
      imageProjector.odometryHandler(odometry.value());
    }
  }

  Odometry
  addLidarMeasurement(double stamp,
                      const pcl::PointCloud<PointXYZIRT>::Ptr laserCloudMsg) {
    // Deskew
    auto maybe_cloudinfo = imageProjector.cloudHandler(stamp, laserCloudMsg);

    if (maybe_cloudinfo.has_value()) {
      // Extract features
      auto cloud_info = maybe_cloudinfo.value();
      featureExtractor.processCloud(cloud_info);

      // Add scan to map
      auto new_pose = mapOptimizer.laserCloudInfoHandler(cloud_info);

      if (new_pose.has_value()) {
        pose = new_pose.value();
        imuPreintegrator.odometryHandler(pose);
      }
      // Simulate sending odometry to the imu
      // /lio_sam/mapping/odometry
      // tf.lidarOdometryHandler(odom, stamp);
    }

    return pose;
  }
};