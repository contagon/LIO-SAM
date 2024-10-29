#include "LIO-SAM/mapOptimization.h"
#include "LIO-SAM/utils.h"

namespace lio_sam {

MapOptimization::MapOptimization(LioSamParams &params) : params_(params) {
  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.1;
  parameters.relinearizeSkip = 1;
  isam = new ISAM2(parameters);

  downSizeFilterCorner.setLeafSize(params_.mappingCornerLeafSize,
                                   params_.mappingCornerLeafSize,
                                   params_.mappingCornerLeafSize);
  downSizeFilterSurf.setLeafSize(params_.mappingSurfLeafSize,
                                 params_.mappingSurfLeafSize,
                                 params_.mappingSurfLeafSize);
  downSizeFilterSurroundingKeyPoses.setLeafSize(
      params_.surroundingKeyframeDensity, params_.surroundingKeyframeDensity,
      params_.surroundingKeyframeDensity); // for surrounding key poses of
                                           // scan-to-map optimization

  allocateMemory();
}

// Return the most recent features in the body pose
pcl::PointCloud<PointType>::Ptr MapOptimization::getMostRecentFrame() {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
  // PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
  // *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
  // *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
  *cloudOut += *laserCloudCornerLastDS;
  *cloudOut += *laserCloudSurfLastDS;
  return cloudOut;
}

void MapOptimization::allocateMemory() {
  cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
  cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

  kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
  kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

  laserCloudCornerLast.reset(
      new pcl::PointCloud<PointType>()); // corner feature set from
                                         // odoOptimization
  laserCloudSurfLast.reset(
      new pcl::PointCloud<PointType>()); // surf feature set from
                                         // odoOptimization
  laserCloudCornerLastDS.reset(
      new pcl::PointCloud<PointType>()); // downsampled corner featuer set
                                         // from odoOptimization
  laserCloudSurfLastDS.reset(
      new pcl::PointCloud<PointType>()); // downsampled surf featuer set from
                                         // odoOptimization

  laserCloudOri.reset(new pcl::PointCloud<PointType>());
  coeffSel.reset(new pcl::PointCloud<PointType>());

  laserCloudOriCornerVec.resize(params_.N_SCAN * params_.Horizon_SCAN);
  coeffSelCornerVec.resize(params_.N_SCAN * params_.Horizon_SCAN);
  laserCloudOriCornerFlag.resize(params_.N_SCAN * params_.Horizon_SCAN);
  laserCloudOriSurfVec.resize(params_.N_SCAN * params_.Horizon_SCAN);
  coeffSelSurfVec.resize(params_.N_SCAN * params_.Horizon_SCAN);
  laserCloudOriSurfFlag.resize(params_.N_SCAN * params_.Horizon_SCAN);

  std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(),
            false);
  std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

  laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
  laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

  kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
  kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

  for (int i = 0; i < 6; ++i) {
    transformTobeMapped[i] = 0;
  }

  matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
}

std::optional<Odometry>
MapOptimization::laserCloudInfoHandler(const CloudInfo<PointType> &cloudInfo) {
  // extract time stamp
  timeLaserInfoCur = cloudInfo.stamp;

  // extract info and feature cloud
  // TODO: Can probably remove these variables and pass them throughout
  // eventually
  laserCloudCornerLast = cloudInfo.cloud_corner;
  laserCloudSurfLast = cloudInfo.cloud_surface;

  std::lock_guard<std::mutex> lock(mtx);

  static double timeLastProcessing = -1;
  if (timeLaserInfoCur - timeLastProcessing >= params_.mappingProcessInterval) {
    timeLastProcessing = timeLaserInfoCur;
    updateInitialGuess(cloudInfo);
    extractSurroundingKeyFrames();
    downsampleCurrentScan();
    scan2MapOptimization(cloudInfo);
    return trans2Odometry(timeLaserInfoCur, transformTobeMapped);
  }

  return {};
}

void MapOptimization::pointAssociateToMap(PointType const *const pi,
                                          PointType *const po) {
  po->x = transPointAssociateToMap(0, 0) * pi->x +
          transPointAssociateToMap(0, 1) * pi->y +
          transPointAssociateToMap(0, 2) * pi->z +
          transPointAssociateToMap(0, 3);
  po->y = transPointAssociateToMap(1, 0) * pi->x +
          transPointAssociateToMap(1, 1) * pi->y +
          transPointAssociateToMap(1, 2) * pi->z +
          transPointAssociateToMap(1, 3);
  po->z = transPointAssociateToMap(2, 0) * pi->x +
          transPointAssociateToMap(2, 1) * pi->y +
          transPointAssociateToMap(2, 2) * pi->z +
          transPointAssociateToMap(2, 3);
  po->intensity = pi->intensity;
}

pcl::PointCloud<PointType>::Ptr
MapOptimization::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                                     PointTypePose *transformIn) {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(
      transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
      transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(params_.numberOfCores)
  for (int i = 0; i < cloudSize; ++i) {
    const auto &pointFrom = cloudIn->points[i];
    cloudOut->points[i].x = transCur(0, 0) * pointFrom.x +
                            transCur(0, 1) * pointFrom.y +
                            transCur(0, 2) * pointFrom.z + transCur(0, 3);
    cloudOut->points[i].y = transCur(1, 0) * pointFrom.x +
                            transCur(1, 1) * pointFrom.y +
                            transCur(1, 2) * pointFrom.z + transCur(1, 3);
    cloudOut->points[i].z = transCur(2, 0) * pointFrom.x +
                            transCur(2, 1) * pointFrom.y +
                            transCur(2, 2) * pointFrom.z + transCur(2, 3);
    cloudOut->points[i].intensity = pointFrom.intensity;
  }
  return cloudOut;
}

bool MapOptimization::saveMap() {
  std::string saveMapDirectory;

  std::cout << "****************************************************"
            << std::endl;
  std::cout << "Saving map to pcd files ..." << std::endl;
  saveMapDirectory = params_.savePCDDirectory;

  std::cout << "Save destination: " << saveMapDirectory << std::endl;
  // create directory and remove old files;
  int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
  unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
  // save key frame transformations
  pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd",
                             *cloudKeyPoses3D);
  pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd",
                             *cloudKeyPoses6D);
  // extract global point cloud map
  pcl::PointCloud<PointType>::Ptr globalCornerCloud(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr globalSurfCloud(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr globalMapCloud(
      new pcl::PointCloud<PointType>());
  for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
    *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],
                                               &cloudKeyPoses6D->points[i]);
    *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i],
                                             &cloudKeyPoses6D->points[i]);
    std::cout << "\r" << std::flush << "Processing feature cloud " << i
              << " of " << cloudKeyPoses6D->size() << " ...";
  }

  if (params_.resolution != 0) {
    std::cout << "\n\nSave resolution: " << params_.resolution << std::endl;

    // down-sample and save corner cloud
    downSizeFilterCorner.setInputCloud(globalCornerCloud);
    downSizeFilterCorner.setLeafSize(params_.resolution, params_.resolution,
                                     params_.resolution);
    downSizeFilterCorner.filter(*globalCornerCloudDS);
    pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd",
                               *globalCornerCloudDS);
    // down-sample and save surf cloud
    downSizeFilterSurf.setInputCloud(globalSurfCloud);
    downSizeFilterSurf.setLeafSize(params_.resolution, params_.resolution,
                                   params_.resolution);
    downSizeFilterSurf.filter(*globalSurfCloudDS);
    pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd",
                               *globalSurfCloudDS);
  } else {
    // save corner cloud
    pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd",
                               *globalCornerCloud);
    // save surf cloud
    pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd",
                               *globalSurfCloud);
  }

  // save global point cloud map
  *globalMapCloud += *globalCornerCloud;
  *globalMapCloud += *globalSurfCloud;

  int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd",
                                       *globalMapCloud);

  downSizeFilterCorner.setLeafSize(params_.mappingCornerLeafSize,
                                   params_.mappingCornerLeafSize,
                                   params_.mappingCornerLeafSize);
  downSizeFilterSurf.setLeafSize(params_.mappingSurfLeafSize,
                                 params_.mappingSurfLeafSize,
                                 params_.mappingSurfLeafSize);

  std::cout << "****************************************************"
            << std::endl;
  std::cout << "Saving map to pcd files completed\n" << std::endl;

  return true;
}

pcl::PointCloud<PointType>::Ptr MapOptimization::getGlobalMap() {

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(
      new pcl::KdTreeFLANN<PointType>());
  ;
  pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(
      new pcl::PointCloud<PointType>());

  if (cloudKeyPoses3D->points.empty() == true)
    return globalMapKeyPosesDS;

  // kd-tree to find near key frames to visualize
  std::vector<int> pointSearchIndGlobalMap;
  std::vector<float> pointSearchSqDisGlobalMap;
  // search near key frames to visualize
  mtx.lock();
  kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
  kdtreeGlobalMap->radiusSearch(
      cloudKeyPoses3D->back(), params_.globalMapVisualizationSearchRadius,
      pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
  mtx.unlock();

  for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
    globalMapKeyPoses->push_back(
        cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
  // downsample near selected key frames
  pcl::VoxelGrid<PointType>
      downSizeFilterGlobalMapKeyPoses; // for global map visualization
  downSizeFilterGlobalMapKeyPoses.setLeafSize(
      params_.globalMapVisualizationPoseDensity,
      params_.globalMapVisualizationPoseDensity,
      params_
          .globalMapVisualizationPoseDensity); // for global map visualization
  downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
  downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
  for (auto &pt : globalMapKeyPosesDS->points) {
    kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap,
                                    pointSearchSqDisGlobalMap);
    pt.intensity =
        cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
  }

  // extract visualized and downsampled key frames
  for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i) {
    if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) >
        params_.globalMapVisualizationSearchRadius)
      continue;
    int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
    *globalMapKeyFrames += *transformPointCloud(
        cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    *globalMapKeyFrames += *transformPointCloud(
        surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
  }
  // downsample visualized points
  pcl::VoxelGrid<PointType>
      downSizeFilterGlobalMapKeyFrames; // for global map visualization
  downSizeFilterGlobalMapKeyFrames.setLeafSize(
      params_.globalMapVisualizationLeafSize,
      params_.globalMapVisualizationLeafSize,
      params_.globalMapVisualizationLeafSize); // for global map visualization
  downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
  downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

  return globalMapKeyPosesDS;
}

void MapOptimization::updateInitialGuess(
    const CloudInfo<PointType> &cloudInfo) {
  static Eigen::Affine3f lastImuTransformation;
  // initialization
  if (cloudKeyPoses3D->points.empty()) {
    transformTobeMapped[0] = cloudInfo.initialGuessRoll;
    transformTobeMapped[1] = cloudInfo.initialGuessPitch;
    transformTobeMapped[2] = cloudInfo.initialGuessYaw;

    lastImuTransformation = pcl::getTransformation(
        0, 0, 0, cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch,
        cloudInfo.initialGuessYaw); // save imu before return;
    return;
  }

  // use imu pre-integration estimation for pose guess
  static bool lastImuPreTransAvailable = false;
  static Eigen::Affine3f lastImuPreTransformation;
  if (cloudInfo.odomAvailable == true) {
    Eigen::Affine3f transBack = pcl::getTransformation(
        cloudInfo.initialGuessX, cloudInfo.initialGuessY,
        cloudInfo.initialGuessZ, cloudInfo.initialGuessRoll,
        cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
    if (lastImuPreTransAvailable == false) {
      lastImuPreTransformation = transBack;
      lastImuPreTransAvailable = true;
    } else {
      Eigen::Affine3f transIncre =
          lastImuPreTransformation.inverse() * transBack;
      Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
      Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(
          transFinal, transformTobeMapped[3], transformTobeMapped[4],
          transformTobeMapped[5], transformTobeMapped[0],
          transformTobeMapped[1], transformTobeMapped[2]);

      lastImuPreTransformation = transBack;

      lastImuTransformation = pcl::getTransformation(
          0, 0, 0, cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch,
          cloudInfo.initialGuessYaw); // save imu before return;
      return;
    }
  }
}

void MapOptimization::extractNearby() {
  pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(
      new pcl::PointCloud<PointType>());
  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  // extract all the nearby key poses and downsample them
  kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
  kdtreeSurroundingKeyPoses->radiusSearch(
      cloudKeyPoses3D->back(), (double)params_.surroundingKeyframeSearchRadius,
      pointSearchInd, pointSearchSqDis);
  for (int i = 0; i < (int)pointSearchInd.size(); ++i) {
    int id = pointSearchInd[i];
    surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
  }

  downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
  downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
  for (auto &pt : surroundingKeyPosesDS->points) {
    kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd,
                                              pointSearchSqDis);
    pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
  }

  // also extract some latest key frames in case the robot rotates in one
  // position
  int numPoses = cloudKeyPoses3D->size();
  for (int i = numPoses - 1; i >= 0; --i) {
    if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
      surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
    else
      break;
  }

  extractCloud(surroundingKeyPosesDS);
}

void MapOptimization::extractCloud(
    pcl::PointCloud<PointType>::Ptr cloudToExtract) {
  // fuse the map
  laserCloudCornerFromMap->clear();
  laserCloudSurfFromMap->clear();
  for (int i = 0; i < (int)cloudToExtract->size(); ++i) {
    if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) >
        params_.surroundingKeyframeSearchRadius)
      continue;

    int thisKeyInd = (int)cloudToExtract->points[i].intensity;
    if (laserCloudMapContainer.find(thisKeyInd) !=
        laserCloudMapContainer.end()) {
      // transformed cloud available
      *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
      *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
    } else {
      // transformed cloud not available
      pcl::PointCloud<PointType> laserCloudCornerTemp =
          *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                               &cloudKeyPoses6D->points[thisKeyInd]);
      pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(
          surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      *laserCloudCornerFromMap += laserCloudCornerTemp;
      *laserCloudSurfFromMap += laserCloudSurfTemp;
      laserCloudMapContainer[thisKeyInd] =
          std::make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
    }
  }

  // Downsample the surrounding corner key frames (or map)
  downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
  downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
  // Downsample the surrounding surf key frames (or map)
  downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
  downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);

  // clear map cache if too large
  if (laserCloudMapContainer.size() > 1000)
    laserCloudMapContainer.clear();
}

void MapOptimization::extractSurroundingKeyFrames() {
  if (cloudKeyPoses3D->points.empty() == true)
    return;

  extractNearby();
}

void MapOptimization::downsampleCurrentScan() {
  // Downsample cloud from current scan
  laserCloudCornerLastDS->clear();
  downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
  downSizeFilterCorner.filter(*laserCloudCornerLastDS);
  laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

  laserCloudSurfLastDS->clear();
  downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
  downSizeFilterSurf.filter(*laserCloudSurfLastDS);
  laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
}

void MapOptimization::updatePointAssociateToMap() {
  transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
}

void MapOptimization::cornerOptimization() {
  updatePointAssociateToMap();

#pragma omp parallel for num_threads(params_.numberOfCores)
  for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
    PointType pointOri, pointSel, coeff;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pointOri = laserCloudCornerLastDS->points[i];
    pointAssociateToMap(&pointOri, &pointSel);
    kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                        pointSearchSqDis);

    cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

    if (pointSearchSqDis[4] < 1.0) {
      float cx = 0, cy = 0, cz = 0;
      for (int j = 0; j < 5; j++) {
        cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
        cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
        cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
      }
      cx /= 5;
      cy /= 5;
      cz /= 5;

      float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
      for (int j = 0; j < 5; j++) {
        float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
        float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
        float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

        a11 += ax * ax;
        a12 += ax * ay;
        a13 += ax * az;
        a22 += ay * ay;
        a23 += ay * az;
        a33 += az * az;
      }
      a11 /= 5;
      a12 /= 5;
      a13 /= 5;
      a22 /= 5;
      a23 /= 5;
      a33 /= 5;

      matA1.at<float>(0, 0) = a11;
      matA1.at<float>(0, 1) = a12;
      matA1.at<float>(0, 2) = a13;
      matA1.at<float>(1, 0) = a12;
      matA1.at<float>(1, 1) = a22;
      matA1.at<float>(1, 2) = a23;
      matA1.at<float>(2, 0) = a13;
      matA1.at<float>(2, 1) = a23;
      matA1.at<float>(2, 2) = a33;

      cv::eigen(matA1, matD1, matV1);

      if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

        float x0 = pointSel.x;
        float y0 = pointSel.y;
        float z0 = pointSel.z;
        float x1 = cx + 0.1 * matV1.at<float>(0, 0);
        float y1 = cy + 0.1 * matV1.at<float>(0, 1);
        float z1 = cz + 0.1 * matV1.at<float>(0, 2);
        float x2 = cx - 0.1 * matV1.at<float>(0, 0);
        float y2 = cy - 0.1 * matV1.at<float>(0, 1);
        float z2 = cz - 0.1 * matV1.at<float>(0, 2);

        float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                              ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                          ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                              ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                          ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                              ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

        float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                         (z1 - z2) * (z1 - z2));

        float la =
            ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
             (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
            a012 / l12;

        float lb =
            -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
              (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
            a012 / l12;

        float lc =
            -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
              (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
            a012 / l12;

        float ld2 = a012 / l12;

        float s = 1 - 0.9 * fabs(ld2);

        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
        coeff.intensity = s * ld2;

        if (s > 0.1) {
          laserCloudOriCornerVec[i] = pointOri;
          coeffSelCornerVec[i] = coeff;
          laserCloudOriCornerFlag[i] = true;
        }
      }
    }
  }
}

void MapOptimization::surfOptimization() {
  updatePointAssociateToMap();

#pragma omp parallel for num_threads(params_.numberOfCores)
  for (int i = 0; i < laserCloudSurfLastDSNum; i++) {
    PointType pointOri, pointSel, coeff;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pointOri = laserCloudSurfLastDS->points[i];
    pointAssociateToMap(&pointOri, &pointSel);
    kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                      pointSearchSqDis);

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    Eigen::Vector3f matX0;

    matA0.setZero();
    matB0.fill(-1);
    matX0.setZero();

    if (pointSearchSqDis[4] < 1.0) {
      for (int j = 0; j < 5; j++) {
        matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
        matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
        matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
      }

      matX0 = matA0.colPivHouseholderQr().solve(matB0);

      float pa = matX0(0, 0);
      float pb = matX0(1, 0);
      float pc = matX0(2, 0);
      float pd = 1;

      float ps = sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                 pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                 pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z +
                 pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

        float s = 1 - 0.9 * fabs(pd2) /
                          sqrt(sqrt(pointOri.x * pointOri.x +
                                    pointOri.y * pointOri.y +
                                    pointOri.z * pointOri.z));

        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        if (s > 0.1) {
          laserCloudOriSurfVec[i] = pointOri;
          coeffSelSurfVec[i] = coeff;
          laserCloudOriSurfFlag[i] = true;
        }
      }
    }
  }
}

void MapOptimization::combineOptimizationCoeffs() {
  // combine corner coeffs
  for (int i = 0; i < laserCloudCornerLastDSNum; ++i) {
    if (laserCloudOriCornerFlag[i] == true) {
      laserCloudOri->push_back(laserCloudOriCornerVec[i]);
      coeffSel->push_back(coeffSelCornerVec[i]);
    }
  }
  // combine surf coeffs
  for (int i = 0; i < laserCloudSurfLastDSNum; ++i) {
    if (laserCloudOriSurfFlag[i] == true) {
      laserCloudOri->push_back(laserCloudOriSurfVec[i]);
      coeffSel->push_back(coeffSelSurfVec[i]);
    }
  }
  // reset flag for next iteration
  std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(),
            false);
  std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
}

bool MapOptimization::LMOptimization(int iterCount) {
  // This optimization is from the original loam_velodyne by Ji Zhang, need to
  // cope with coordinate transformation lidar <- camera      ---     camera
  // <- lidar x = z                ---     x = y y = x                --- y =
  // z z = y                ---     z = x roll = yaw           ---     roll =
  // pitch pitch = roll         ---     pitch = yaw yaw = pitch          ---
  // yaw = roll

  // lidar -> camera
  float srx = sin(transformTobeMapped[1]);
  float crx = cos(transformTobeMapped[1]);
  float sry = sin(transformTobeMapped[2]);
  float cry = cos(transformTobeMapped[2]);
  float srz = sin(transformTobeMapped[0]);
  float crz = cos(transformTobeMapped[0]);

  int laserCloudSelNum = laserCloudOri->size();
  if (laserCloudSelNum < 50) {
    return false;
  }

  cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
  cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
  cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
  cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
  cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
  cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

  PointType pointOri, coeff;

  for (int i = 0; i < laserCloudSelNum; i++) {
    // lidar -> camera
    pointOri.x = laserCloudOri->points[i].y;
    pointOri.y = laserCloudOri->points[i].z;
    pointOri.z = laserCloudOri->points[i].x;
    // lidar -> camera
    coeff.x = coeffSel->points[i].y;
    coeff.y = coeffSel->points[i].z;
    coeff.z = coeffSel->points[i].x;
    coeff.intensity = coeffSel->points[i].intensity;
    // in camera
    float arx =
        (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y -
         srx * sry * pointOri.z) *
            coeff.x +
        (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) *
            coeff.y +
        (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y -
         cry * srx * pointOri.z) *
            coeff.z;

    float ary =
        ((cry * srx * srz - crz * sry) * pointOri.x +
         (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) *
            coeff.x +
        ((-cry * crz - srx * sry * srz) * pointOri.x +
         (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) *
            coeff.z;

    float arz = ((crz * srx * sry - cry * srz) * pointOri.x +
                 (-cry * crz - srx * sry * srz) * pointOri.y) *
                    coeff.x +
                (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                ((sry * srz + cry * crz * srx) * pointOri.x +
                 (crz * sry - cry * srx * srz) * pointOri.y) *
                    coeff.z;
    // camera -> lidar
    matA.at<float>(i, 0) = arz;
    matA.at<float>(i, 1) = arx;
    matA.at<float>(i, 2) = ary;
    matA.at<float>(i, 3) = coeff.z;
    matA.at<float>(i, 4) = coeff.x;
    matA.at<float>(i, 5) = coeff.y;
    matB.at<float>(i, 0) = -coeff.intensity;
  }

  cv::transpose(matA, matAt);
  matAtA = matAt * matA;
  matAtB = matAt * matB;
  cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

  if (iterCount == 0) {

    cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

    cv::eigen(matAtA, matE, matV);
    matV.copyTo(matV2);

    isDegenerate = false;
    float eignThre[6] = {100, 100, 100, 100, 100, 100};
    for (int i = 5; i >= 0; i--) {
      if (matE.at<float>(0, i) < eignThre[i]) {
        for (int j = 0; j < 6; j++) {
          matV2.at<float>(i, j) = 0;
        }
        isDegenerate = true;
      } else {
        break;
      }
    }
    matP = matV.inv() * matV2;
  }

  if (isDegenerate) {
    cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
    matX.copyTo(matX2);
    matX = matP * matX2;
  }

  transformTobeMapped[0] += matX.at<float>(0, 0);
  transformTobeMapped[1] += matX.at<float>(1, 0);
  transformTobeMapped[2] += matX.at<float>(2, 0);
  transformTobeMapped[3] += matX.at<float>(3, 0);
  transformTobeMapped[4] += matX.at<float>(4, 0);
  transformTobeMapped[5] += matX.at<float>(5, 0);

  float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                      pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                      pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
  float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                      pow(matX.at<float>(4, 0) * 100, 2) +
                      pow(matX.at<float>(5, 0) * 100, 2));

  if (deltaR < 0.05 && deltaT < 0.05) {
    return true; // converged
  }
  return false; // keep optimizing
}

void MapOptimization::scan2MapOptimization(
    const CloudInfo<PointType> &cloudInfo) {
  if (cloudKeyPoses3D->points.empty())
    return;

  if (laserCloudCornerLastDSNum > params_.edgeFeatureMinValidNum &&
      laserCloudSurfLastDSNum > params_.surfFeatureMinValidNum) {
    kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
    kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

    for (int iterCount = 0; iterCount < 30; iterCount++) {
      laserCloudOri->clear();
      coeffSel->clear();

      cornerOptimization();
      surfOptimization();

      combineOptimizationCoeffs();

      if (LMOptimization(iterCount) == true)
        break;
    }

    transformUpdate(cloudInfo);
  } else {
    std::cout << "Not enough features! Only " << laserCloudCornerLastDSNum
              << " edge and " << laserCloudSurfLastDSNum
              << " planar features available." << std::endl;
  }
}

void MapOptimization::transformUpdate(const CloudInfo<PointType> &cloudInfo) {
  // Weight actual transform with result from IMU orientation - disable
  // if (cloudInfo.imuAvailable == true) {
  //   if (std::abs(cloudInfo.imuPitchInit) < 1.4) {
  //     double imuWeight = params_.imuRPYWeight;
  //     Eigen::Quaterniond imuQuaternion;
  //     Eigen::Quaterniond transformQuaternion;
  //     double rollMid, pitchMid, yawMid;

  //     // slerp roll
  //     transformQuaternion = rpy2quat(double(transformTobeMapped[0]), 0.0,
  //     0.0); imuQuaternion = rpy2quat(double(cloudInfo.imuRollInit), 0.0,
  //     0.0); quat2rpy(transformQuaternion.slerp(imuWeight, imuQuaternion),
  //     &rollMid,
  //              &pitchMid, &yawMid);
  //     transformTobeMapped[0] = rollMid;

  //     // slerp pitch
  //     transformQuaternion = rpy2quat(0.0, double(transformTobeMapped[0]),
  //     0.0); imuQuaternion = rpy2quat(0.0, double(cloudInfo.imuRollInit),
  //     0.0); quat2rpy(transformQuaternion.slerp(imuWeight, imuQuaternion),
  //     &rollMid,
  //              &pitchMid, &yawMid);
  //     transformTobeMapped[1] = pitchMid;
  //   }
  // }

  transformTobeMapped[0] = constraintTransformation(
      transformTobeMapped[0], params_.rotation_tollerance);
  transformTobeMapped[1] = constraintTransformation(
      transformTobeMapped[1], params_.rotation_tollerance);
  transformTobeMapped[5] =
      constraintTransformation(transformTobeMapped[5], params_.z_tollerance);
}

float MapOptimization::constraintTransformation(float value, float limit) {
  if (value < -limit)
    value = -limit;
  if (value > limit)
    value = limit;

  return value;
}

void MapOptimization::addOdomFactor() {
  if (cloudKeyPoses3D->points.empty()) {
    noiseModel::Diagonal::shared_ptr priorNoise =
        noiseModel::Diagonal::Variances(
            (Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8)
                .finished()); // rad*rad, meter*meter
    gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped),
                                      priorNoise));
    initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
  } else {
    noiseModel::Diagonal::shared_ptr odometryNoise =
        noiseModel::Diagonal::Variances(
            (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    gtsam::Pose3 poseFrom =
        pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
    gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
    gtSAMgraph.add(BetweenFactor<Pose3>(
        cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(),
        poseFrom.between(poseTo), odometryNoise));
    initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
  }
}

} // namespace lio_sam