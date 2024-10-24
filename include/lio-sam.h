// TODO: Eventually reintroduce ROS wrapper
#include "featureExtraction.h"
#include "imageProjection.h"
#include "types.h"

class LIOSAM {
private:
  // TransformFusion tf;
  ImageProjection imageProjector;
  FeatureExtraction featureExtractor;

public:
  LIOSAM(LioSamParams params)
      : featureExtractor(params), imageProjector(params) {}
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

      // mapOptimization
      // mapOptimization.laserCloudInfoHandler(cloud_info);
      // mapOptimization.somethingsomething

      // Simulate sending odometry to all of the nodes
      // /lio_sam/mapping/odometry
      // tf.lidarOdometryHandler(odom, stamp);
    }
  }
};