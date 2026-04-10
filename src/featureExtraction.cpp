#include "LIO-SAM/featureExtraction.h"
#include <pcl/common/io.h>

namespace lio_sam {

FeatureExtraction::FeatureExtraction(const LioSamParams &params)
    : params_(params) {
  cloudSmoothness.resize(params_.N_SCAN * params_.Horizon_SCAN);

  downSizeFilter.setLeafSize(params_.odometrySurfLeafSize,
                             params_.odometrySurfLeafSize,
                             params_.odometrySurfLeafSize);

  cornerCloud.reset(new pcl::PointCloud<PointType>());
  surfaceCloud.reset(new pcl::PointCloud<PointType>());

  cloudCurvature = new float[params_.N_SCAN * params_.Horizon_SCAN];
  cloudNeighborPicked = new int[params_.N_SCAN * params_.Horizon_SCAN];
  cloudLabel = new int[params_.N_SCAN * params_.Horizon_SCAN];
}

void FeatureExtraction::processCloud(CloudInfo &cloudInfo) {
  // Compute features
  calculateSmoothness(cloudInfo);
  markOccludedPoints(cloudInfo);
  extractFeatures(cloudInfo);

  // Modify things in place
  // delete data we don't need anymore
  cloudInfo.startRingIndex.clear();
  cloudInfo.endRingIndex.clear();
  cloudInfo.pointColInd.clear();
  cloudInfo.pointRange.clear();
  // Add in feature clouds
  pcl::copyPointCloud(*cornerCloud, cloudInfo.cloud_corner);
  pcl::copyPointCloud(*surfaceCloud, cloudInfo.cloud_surface);
}

void FeatureExtraction::calculateSmoothness(const CloudInfo &cloudInfo) {
  int cloudSize = cloudInfo.cloud_deskewed.points.size();
  for (int i = 5; i < cloudSize - 5; i++) {
    float diffRange =
        cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4] +
        cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2] +
        cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10 +
        cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2] +
        cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4] +
        cloudInfo.pointRange[i + 5];

    cloudCurvature[i] =
        diffRange * diffRange; // diffX * diffX + diffY * diffY + diffZ * diffZ;

    cloudNeighborPicked[i] = 0;
    cloudLabel[i] = 0;
    // cloudSmoothness for sorting
    cloudSmoothness[i].value = cloudCurvature[i];
    cloudSmoothness[i].ind = i;
  }
}

void FeatureExtraction::markOccludedPoints(const CloudInfo &cloudInfo) {
  int cloudSize = cloudInfo.cloud_deskewed.points.size();

  auto mark_if_in_bounds = [this, cloudSize](const int idx) {
    if (idx >= 0 && idx < cloudSize)
      cloudNeighborPicked[idx] = 1;
  };

  // mark occluded points and parallel beam points
  for (int i = 5; i < cloudSize - 6; ++i) {
    // occluded points
    float depth1 = cloudInfo.pointRange[i];
    float depth2 = cloudInfo.pointRange[i + 1];
    int columnDiff =
        std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));

    if (columnDiff < 10) {
      // 10 pixel diff in range image
      if (depth1 - depth2 > 0.3) {
        mark_if_in_bounds(i - 5);
        mark_if_in_bounds(i - 4);
        mark_if_in_bounds(i - 3);
        mark_if_in_bounds(i - 2);
        mark_if_in_bounds(i - 1);
        mark_if_in_bounds(i);
      } else if (depth2 - depth1 > 0.3) {
        mark_if_in_bounds(i + 1);
        mark_if_in_bounds(i + 2);
        mark_if_in_bounds(i + 3);
        mark_if_in_bounds(i + 4);
        mark_if_in_bounds(i + 5);
        mark_if_in_bounds(i + 6);
      }
    }
    // parallel beam
    float diff1 =
        std::abs(float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
    float diff2 =
        std::abs(float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

    if (diff1 > 0.02 * cloudInfo.pointRange[i] &&
        diff2 > 0.02 * cloudInfo.pointRange[i])
      cloudNeighborPicked[i] = 1;
  }
}

void FeatureExtraction::extractFeatures(const CloudInfo &cloudInfo) {
  cornerCloud->clear();
  surfaceCloud->clear();

  pcl::PointCloud<PointType>::Ptr surfaceCloudScan(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(
      new pcl::PointCloud<PointType>());

  const int maxSize = params_.N_SCAN * params_.Horizon_SCAN;

  for (int i = 0; i < params_.N_SCAN; i++) {
    surfaceCloudScan->clear();

    for (int j = 0; j < 6; j++) {

      int sp = (cloudInfo.startRingIndex[i] * (6 - j) +
                cloudInfo.endRingIndex[i] * j) /
               6;
      int ep = (cloudInfo.startRingIndex[i] * (5 - j) +
                cloudInfo.endRingIndex[i] * (j + 1)) /
                   6 -
               1;

      sp = std::max(5, std::min(sp, maxSize - 6));
      ep = std::max(5, std::min(ep, maxSize - 6));

      if (sp >= ep)
        continue;

      std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep,
                by_value());

      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] > params_.edgeThreshold) {
          largestPickedNum++;
          if (largestPickedNum <= 20) {
            cloudLabel[ind] = 1;
            cornerCloud->push_back(cloudInfo.cloud_deskewed.points[ind]);
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {
            if (ind + l >= maxSize || ind + l - 1 >= maxSize)
              break;

            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] -
                                          cloudInfo.pointColInd[ind + l - 1]));
            if (columnDiff > 10)
              break;
            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            if (ind + l < 0 || ind + l + 1 < 0)
              break;

            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] -
                                          cloudInfo.pointColInd[ind + l + 1]));
            if (columnDiff > 10)
              break;
            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++) {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < params_.surfThreshold) {

          cloudLabel[ind] = -1;
          cloudNeighborPicked[ind] = 1;

          for (int l = 1; l <= 5; l++) {
            if (ind + l >= maxSize || ind + l - 1 >= maxSize)
              break;

            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] -
                                          cloudInfo.pointColInd[ind + l - 1]));
            if (columnDiff > 10)
              break;

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            if (ind + l < 0 || ind + l + 1 < 0)
              break;

            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] -
                                          cloudInfo.pointColInd[ind + l + 1]));
            if (columnDiff > 10)
              break;

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfaceCloudScan->push_back(cloudInfo.cloud_deskewed.points[k]);
        }
      }
    }

    surfaceCloudScanDS->clear();
    downSizeFilter.setInputCloud(surfaceCloudScan);
    downSizeFilter.filter(*surfaceCloudScanDS);

    *surfaceCloud += *surfaceCloudScanDS;
  }
}

} // namespace lio_sam
