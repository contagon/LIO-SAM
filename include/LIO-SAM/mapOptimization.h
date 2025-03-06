#pragma once

#include "types.h"

#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/Transform.h>
#include <opencv2/opencv.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <optional>
#include <pcl/point_cloud.h>
#include <vector>

using namespace gtsam;

using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

namespace lio_sam {

class MapOptimization {

private:
  LioSamParams params_;

  // gtsam
  NonlinearFactorGraph gtSAMgraph;
  Values initialEstimate;
  ISAM2 *isam;
  Values isamCurrentEstimate;

  // Intermediate computations that need to be cached
  double timeLastProcessing = -1;
  bool lastImuPreTransAvailable = false;
  Eigen::Affine3f lastImuTransformation = Eigen::Affine3f();
  Eigen::Affine3f lastImuPreTransformation = Eigen::Affine3f();

  std::vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

  pcl::PointCloud<PointTypeIndexed>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

  // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
  // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
  // downsampled corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
  // downsampled surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointTypeResidual>::Ptr coeffSel;

  // corner point holder for parallel computation
  std::vector<PointType> laserCloudOriCornerVec;
  std::vector<PointTypeResidual> coeffSelCornerVec;
  std::vector<bool> laserCloudOriCornerFlag;
  // surf point holder for parallel computation
  std::vector<PointType> laserCloudOriSurfVec;
  std::vector<PointTypeResidual> coeffSelSurfVec;
  std::vector<bool> laserCloudOriSurfFlag;

  std::map<int,
           std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>>
      laserCloudMapContainer;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

  pcl::KdTreeFLANN<PointTypeIndexed>::Ptr kdtreeSurroundingKeyPoses;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  // for surrounding key poses of scan-to-map optimization
  pcl::VoxelGrid<PointTypeIndexed> downSizeFilterSurroundingKeyPoses;

  double timeLaserInfoCur;

  float transformTobeMapped[6];

  std::mutex mtx;

  bool isDegenerate = false;
  cv::Mat matP;

  int laserCloudCornerLastDSNum = 0;
  int laserCloudSurfLastDSNum = 0;

  Eigen::Affine3f transPointAssociateToMap;

  void allocateMemory();

  void pointAssociateToMap(PointType const *const pi, PointType *const po);

  pcl::PointCloud<PointType>::Ptr
  transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                      PointTypePose *transformIn);

  void updateInitialGuess(const CloudInfo &cloudInfo);

  void extractNearby();

  void saveKeyFramesAndFactor();

  void extractCloud(pcl::PointCloud<PointTypeIndexed>::Ptr cloudToExtract);

  void extractSurroundingKeyFrames();

  void downsampleCurrentScan();

  void updatePointAssociateToMap();

  void cornerOptimization();

  void surfOptimization();

  void combineOptimizationCoeffs();

  bool LMOptimization(int iterCount);

  void scan2MapOptimization(const CloudInfo &cloudInfo);

  void transformUpdate(const CloudInfo &cloudInfo);

  float constraintTransformation(float value, float limit);

  void addOdomFactor();

public:
  MapOptimization(LioSamParams &params);

  std::optional<Odometry> laserCloudInfoHandler(const CloudInfo &cloudInfo);

  // Getters
  // Return the most recent features in the body pose
  pcl::PointCloud<PointType>::Ptr getMostRecentFrame();

  pcl::PointCloud<PointType>::Ptr getGlobalMap();
};

} // namespace lio_sam