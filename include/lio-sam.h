// TODO: Eventually reintroduce ROS wrapper
#include "featureExtraction.h"
#include "types.h"

class LIOSAM {
private:
  // TransformFusion tf;
  FeatureExtraction featureExtractor;

public:
  LIOSAM(LioSamParams params) : featureExtractor(params) {}
  ~LIOSAM();

  void addImuMeasurement() {

    // Simulate sending imu measurement to all of the nodes
    // odom_topic + _incremental
    // tf.imuOdometryHandler(odom, stamp);
  }

  void addLidarMeasurement() {
    // Process point cloud
    // deskew
    // featureExtractor.processCloud(cloudInfo);
    // map

    // Simulate sending odometry to all of the nodes
    // /lio_sam/mapping/odometry
    // tf.lidarOdometryHandler(odom, stamp);
  }
};