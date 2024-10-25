// TODO: Eventually reintroduce ROS wrapper
#include "featureExtraction.h"
#include "imageProjection.h"
#include "mapOptimization.h"
#include "types.h"
#include <Eigen/src/Geometry/Transform.h>

class LIOSAM {
private:
  Eigen::Affine3f pose;

  // TransformFusion tf;
  ImageProjection imageProjector;
  FeatureExtraction featureExtractor;
  MapOptimization mapOptimizer;

public:
  LIOSAM(LioSamParams params)
      : featureExtractor(params), imageProjector(params), mapOptimizer(params) {
  }
  ~LIOSAM();

  void addImuMeasurement() {

    // Simulate sending imu measurement to all of the nodes
    // odom_topic + _incremental
    // tf.imuOdometryHandler(odom, stamp);
    // imageProjector.imuHandler(const Imu &imuMsg);

    // Everywhere the imu node sends imu_incremental to
    // imageProjector.odometryHandler(const Odometry &odometryMsg);
  }

  void
  addLidarMeasurement(double stamp,
                      const pcl::PointCloud<PointXYZIRT>::Ptr laserCloudMsg) {
    // Deskew
    auto maybe_cloudinfo = imageProjector.cloudHandler(stamp, laserCloudMsg);

    if (maybe_cloudinfo.has_value()) {
      // Extract features
      auto cloud_info = maybe_cloudinfo.value();
      featureExtractor.processCloud(cloud_info);

      // Add scan to map
      pose = mapOptimizer.laserCloudInfoHandler(cloud_info);

      // Simulate sending odometry to the imu
      // /lio_sam/mapping/odometry
      // tf.lidarOdometryHandler(odom, stamp);
    }
  }
};