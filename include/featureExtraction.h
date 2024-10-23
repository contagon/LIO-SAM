#include "types.h"

struct smoothness_t {
  float value;
  size_t ind;
};

struct by_value {
  bool operator()(smoothness_t const &left, smoothness_t const &right) {
    return left.value < right.value;
  }
};

class FeatureExtraction {
private:
  LioSamParams params_;

  pcl::PointCloud<PointType>::Ptr cornerCloud;
  pcl::PointCloud<PointType>::Ptr surfaceCloud;

  pcl::VoxelGrid<PointType> downSizeFilter;
  std::vector<smoothness_t> cloudSmoothness;
  float *cloudCurvature;
  int *cloudNeighborPicked;
  int *cloudLabel;

  void calculateSmoothness(const CloudInfo<PointType> &cloudInfo);
  void markOccludedPoints(const CloudInfo<PointType> &cloudInfo);
  void extractFeatures(const CloudInfo<PointType> &cloudInfo);

public:
  FeatureExtraction(const LioSamParams &params);
  void processCloud(CloudInfo<PointType> &cloudInfo);
};