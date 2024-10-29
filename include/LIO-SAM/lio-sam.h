// TODO: Eventually reintroduce ROS wrapper
#include "featureExtraction.h"
#include "imageProjection.h"
#include "imuPreintegration.h"
#include "mapOptimization.h"
#include "types.h"
#include <Eigen/src/Geometry/Transform.h>
#include <pcl/point_cloud.h>

namespace lio_sam {

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
  ~LIOSAM(){};

  Odometry getPose() { return pose; }

  pcl::PointCloud<PointType>::Ptr getMap() {
    return mapOptimizer.getGlobalMap();
  }

  pcl::PointCloud<PointType>::Ptr getMostRecentFrame() {
    return mapOptimizer.getMostRecentFrame();
  }

  void addImuMeasurement(const Imu imuMsg) {
    // Simulate sending imu measurement to all of the nodes
    std::cout << "adding to preingrator" << std::endl;
    auto odometry = imuPreintegrator.imuHandler(imuMsg);
    std::cout << "adding to projector" << std::endl;
    imageProjector.imuHandler(imuMsg);

    // Everywhere the imu node sends odometry to
    std::cout << "adding odometry to imageProjector" << std::endl;
    imageProjector.odometryHandler(odometry);
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

      // If we successfully update the pose, send it to the imuPreintegrator
      if (new_pose.has_value()) {
        pose = new_pose.value();
        imuPreintegrator.odometryHandler(pose);
      }
    }

    return pose;
  }
};

} // namespace lio_sam