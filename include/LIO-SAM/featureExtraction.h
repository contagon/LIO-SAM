#include "types.h"

namespace lio_sam {

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

  void calculateSmoothness(const CloudInfo &cloudInfo);
  void markOccludedPoints(const CloudInfo &cloudInfo);
  void extractFeatures(const CloudInfo &cloudInfo);

public:
  FeatureExtraction(const LioSamParams &params);
  void processCloud(CloudInfo &cloudInfo);
};

} // namespace lio_sam