#include "LIO-SAM/imageProjection.h"
#include "LIO-SAM/utils.h"

namespace lio_sam {

// -------------------------- Public Methods -------------------------- //
ImageProjection::ImageProjection(const LioSamParams &params)
    : deskewFlag(0), params_(params) {
  allocateMemory();
  resetParameters();

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
}

// Store gyro for rotation deskewing
void ImageProjection::imuHandler(const Imu &imu) {
  auto gyro = params_.lidar_R_imu * imu.gyro;
  std::lock_guard<std::mutex> lock1(imuLock);
  gyroQueue.push_back(std::make_pair(imu.stamp, gyro));

  // debug IMU data
  // cout << std::setprecision(6);
  // cout << "IMU acc: " << endl;
  // cout << "x: " << thisImu.linear_acceleration.x <<
  //       ", y: " << thisImu.linear_acceleration.y <<
  //       ", z: " << thisImu.linear_acceleration.z << endl;
  // cout << "IMU gyro: " << endl;
  // cout << "x: " << thisImu.angular_velocity.x <<
  //       ", y: " << thisImu.angular_velocity.y <<
  //       ", z: " << thisImu.angular_velocity.z << endl;
  // double imuRoll, imuPitch, imuYaw;
  // tf::Quaternion orientation;
  // tf::quaternionMsgToTF(thisImu.orientation, orientation);
  // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
  // cout << "IMU roll pitch yaw: " << endl;
  // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " <<
  // imuYaw << endl << endl;
}

// Store odometry from IMU Preintegration to use for initial guesses and for
// translation deskewing
void ImageProjection::odometryHandler(const Odometry &odometryMsg) {
  std::lock_guard<std::mutex> lock2(odoLock);
  odomQueue.push_back(odometryMsg);
}

std::optional<CloudInfo<PointType>> ImageProjection::cloudHandler(
    double stamp, const pcl::PointCloud<PointXYZIRT>::Ptr laserCloudMsg) {
  if (!cachePointCloud(stamp, laserCloudMsg))
    return {};

  if (!deskewInfo())
    return {};

  projectPointCloud();
  cloudExtraction();

  // Save our deskewed cloud!
  cloudInfo.stamp = stamp;
  cloudInfo.cloud_deskewed = extractedCloud;

  resetParameters();

  return cloudInfo;
}

// -------------------------- Private Methods -------------------------- //

void ImageProjection::allocateMemory() {
  laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
  fullCloud.reset(new pcl::PointCloud<PointType>());
  extractedCloud.reset(new pcl::PointCloud<PointType>());

  fullCloud->points.resize(params_.N_SCAN * params_.Horizon_SCAN);

  cloudInfo.startRingIndex.assign(params_.N_SCAN, 0);
  cloudInfo.endRingIndex.assign(params_.N_SCAN, 0);

  cloudInfo.pointColInd.assign(params_.N_SCAN * params_.Horizon_SCAN, 0);
  cloudInfo.pointRange.assign(params_.N_SCAN * params_.Horizon_SCAN, 0);

  resetParameters();
}

void ImageProjection::resetParameters() {
  laserCloudIn->clear();
  extractedCloud->clear();
  // reset range matrix for range image projection
  rangeMat = cv::Mat(params_.N_SCAN, params_.Horizon_SCAN, CV_32F,
                     cv::Scalar::all(FLT_MAX));

  imuPointerCur = 0;
  firstPointFlag = true;
  odomDeskewFlag = false;

  for (int i = 0; i < queueLength; ++i) {
    imuTime[i] = 0;
    imuRotX[i] = 0;
    imuRotY[i] = 0;
    imuRotZ[i] = 0;
  }

  columnIdnCountVec.assign(params_.N_SCAN, 0);
}

bool ImageProjection::cachePointCloud(
    double stamp, const pcl::PointCloud<PointXYZIRT>::Ptr laserCloudMsg) {
  // cache point cloud
  cloudQueue.push_back(std::make_pair(stamp, laserCloudMsg));
  if (cloudQueue.size() <= 2)
    return false;

  // convert cloud
  currentCloudMsg = cloudQueue.front();
  laserCloudIn = currentCloudMsg.second;
  cloudQueue.pop_front();

  // get timestamp
  timeScanCur = currentCloudMsg.first;
  timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

  return true;
}

bool ImageProjection::deskewInfo() {
  std::lock_guard<std::mutex> lock1(imuLock);
  std::lock_guard<std::mutex> lock2(odoLock);

  // make sure IMU data available for the scan
  if (gyroQueue.empty() || gyroQueue.front().first > timeScanCur ||
      gyroQueue.back().first < timeScanEnd) {
    std::cout << "Waiting for IMU data ..." << std::endl;
    return false;
  }

  imuDeskewInfo();

  odomDeskewInfo();

  return true;
}

// IMU is used for deskewing the rotation
void ImageProjection::imuDeskewInfo() {
  cloudInfo.imuAvailable = false;

  while (!gyroQueue.empty()) {
    if (gyroQueue.front().first < timeScanCur - 0.01)
      gyroQueue.pop_front();
    else
      break;
  }

  if (gyroQueue.empty())
    return;

  imuPointerCur = 0;

  for (int i = 0; i < (int)gyroQueue.size(); ++i) {
    auto thisImuMsg = gyroQueue[i];
    double currentImuTime = thisImuMsg.first;

    if (currentImuTime > timeScanEnd + 0.01)
      break;

    if (imuPointerCur == 0) {
      imuRotX[0] = 0;
      imuRotY[0] = 0;
      imuRotZ[0] = 0;
      imuTime[0] = currentImuTime;
      ++imuPointerCur;
      continue;
    }

    // get angular velocity
    double angular_x, angular_y, angular_z;
    angular_x = thisImuMsg.second.x();
    angular_y = thisImuMsg.second.y();
    angular_z = thisImuMsg.second.z();

    // integrate rotation
    double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
    imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
    imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
    imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
    imuTime[imuPointerCur] = currentImuTime;
    ++imuPointerCur;
  }

  --imuPointerCur;

  if (imuPointerCur <= 0)
    return;

  cloudInfo.imuAvailable = true;
}

// Odometry is used for deskewing the position
void ImageProjection::odomDeskewInfo() {
  cloudInfo.odomAvailable = false;

  while (!odomQueue.empty()) {
    if (odomQueue.front().stamp < timeScanCur - 0.01)
      odomQueue.pop_front();
    else
      break;
  }

  if (odomQueue.empty())
    return;

  if (odomQueue.front().stamp > timeScanCur)
    return;

  // get start odometry at the beginning of the scan
  Odometry startOdomMsg;

  for (int i = 0; i < (int)odomQueue.size(); ++i) {
    startOdomMsg = odomQueue[i];

    if (startOdomMsg.stamp < timeScanCur)
      continue;
    else
      break;
  }

  double roll, pitch, yaw;
  quat2rpy(startOdomMsg.orientation, &roll, &pitch, &yaw);

  // Initial guess used in mapOptimization
  cloudInfo.initialGuessX = startOdomMsg.position.x();
  cloudInfo.initialGuessY = startOdomMsg.position.y();
  cloudInfo.initialGuessZ = startOdomMsg.position.z();
  cloudInfo.initialGuessRoll = roll;
  cloudInfo.initialGuessPitch = pitch;
  cloudInfo.initialGuessYaw = yaw;

  cloudInfo.odomAvailable = true;

  // get end odometry at the end of the scan
  odomDeskewFlag = false;

  if (odomQueue.back().stamp < timeScanEnd)
    return;

  Odometry endOdomMsg;

  for (int i = 0; i < (int)odomQueue.size(); ++i) {
    endOdomMsg = odomQueue[i];

    if (endOdomMsg.stamp < timeScanEnd)
      continue;
    else
      break;
  }

  Eigen::Affine3f transBegin = pcl::getTransformation(
      startOdomMsg.position.x(), startOdomMsg.position.y(),
      startOdomMsg.position.z(), roll, pitch, yaw);

  quat2rpy(endOdomMsg.orientation, &roll, &pitch, &yaw);
  Eigen::Affine3f transEnd =
      pcl::getTransformation(endOdomMsg.position.x(), endOdomMsg.position.y(),
                             endOdomMsg.position.z(), roll, pitch, yaw);

  Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

  float rollIncre, pitchIncre, yawIncre;
  pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ,
                                    rollIncre, pitchIncre, yawIncre);

  odomDeskewFlag = true;
}

void ImageProjection::findRotation(double pointTime, float *rotXCur,
                                   float *rotYCur, float *rotZCur) {
  *rotXCur = 0;
  *rotYCur = 0;
  *rotZCur = 0;

  int imuPointerFront = 0;
  while (imuPointerFront < imuPointerCur) {
    if (pointTime < imuTime[imuPointerFront])
      break;
    ++imuPointerFront;
  }

  if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) {
    *rotXCur = imuRotX[imuPointerFront];
    *rotYCur = imuRotY[imuPointerFront];
    *rotZCur = imuRotZ[imuPointerFront];
  } else {
    int imuPointerBack = imuPointerFront - 1;
    double ratioFront = (pointTime - imuTime[imuPointerBack]) /
                        (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
    double ratioBack = (imuTime[imuPointerFront] - pointTime) /
                       (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
    *rotXCur = imuRotX[imuPointerFront] * ratioFront +
               imuRotX[imuPointerBack] * ratioBack;
    *rotYCur = imuRotY[imuPointerFront] * ratioFront +
               imuRotY[imuPointerBack] * ratioBack;
    *rotZCur = imuRotZ[imuPointerFront] * ratioFront +
               imuRotZ[imuPointerBack] * ratioBack;
  }
}

void ImageProjection::findPosition(double relTime, float *posXCur,
                                   float *posYCur, float *posZCur) {
  *posXCur = 0;
  *posYCur = 0;
  *posZCur = 0;

  // If the sensor moves relatively slow, like walking speed, positional
  // deskew seems to have little benefits. Thus code below is commented.

  // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
  //     return;

  // float ratio = relTime / (timeScanEnd - timeScanCur);

  // *posXCur = ratio * odomIncreX;
  // *posYCur = ratio * odomIncreY;
  // *posZCur = ratio * odomIncreZ;
}

PointType ImageProjection::deskewPoint(PointType *point, double relTime) {
  if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
    return *point;

  double pointTime = timeScanCur + relTime;

  float rotXCur, rotYCur, rotZCur;
  findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

  float posXCur, posYCur, posZCur;
  findPosition(relTime, &posXCur, &posYCur, &posZCur);

  if (firstPointFlag == true) {
    transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur,
                                                rotXCur, rotYCur, rotZCur))
                            .inverse();
    firstPointFlag = false;
  }

  // transform points to start
  Eigen::Affine3f transFinal = pcl::getTransformation(
      posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
  Eigen::Affine3f transBt = transStartInverse * transFinal;

  PointType newPoint;
  newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y +
               transBt(0, 2) * point->z + transBt(0, 3);
  newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y +
               transBt(1, 2) * point->z + transBt(1, 3);
  newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y +
               transBt(2, 2) * point->z + transBt(2, 3);
  newPoint.intensity = point->intensity;

  return newPoint;
}

void ImageProjection::projectPointCloud() {
  int cloudSize = laserCloudIn->points.size();
  // range image projection
  for (int i = 0; i < cloudSize; ++i) {
    PointType thisPoint;
    thisPoint.x = laserCloudIn->points[i].x;
    thisPoint.y = laserCloudIn->points[i].y;
    thisPoint.z = laserCloudIn->points[i].z;
    thisPoint.intensity = laserCloudIn->points[i].intensity;

    float range = pointDistance(thisPoint);
    if (range < params_.lidarMinRange || range > params_.lidarMaxRange)
      continue;

    int rowIdn = laserCloudIn->points[i].ring;
    if (rowIdn < 0 || rowIdn >= params_.N_SCAN)
      continue;

    if (rowIdn % params_.downsampleRate != 0)
      continue;

    // TODO: Need to handle this myself somehow
    int columnIdn = -1;
    if (params_.sensor == SensorType::VELODYNE ||
        params_.sensor == SensorType::OUSTER) {
      float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
      static float ang_res_x = 360.0 / float(params_.Horizon_SCAN);
      columnIdn = -round((horizonAngle - 90.0) / ang_res_x) +
                  float(params_.Horizon_SCAN) / 2;
      if (columnIdn >= params_.Horizon_SCAN)
        columnIdn -= params_.Horizon_SCAN;
    } else if (params_.sensor == SensorType::LIVOX) {
      columnIdn = columnIdnCountVec[rowIdn];
      columnIdnCountVec[rowIdn] += 1;
    }

    if (columnIdn < 0 || columnIdn >= params_.Horizon_SCAN)
      continue;

    if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
      continue;

    thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

    rangeMat.at<float>(rowIdn, columnIdn) = range;

    int index = columnIdn + rowIdn * params_.Horizon_SCAN;
    fullCloud->points[index] = thisPoint;
  }
}

void ImageProjection::cloudExtraction() {
  int count = 0;
  // extract segmented cloud for lidar odometry
  for (int i = 0; i < params_.N_SCAN; ++i) {
    cloudInfo.startRingIndex[i] = count - 1 + 5;

    for (int j = 0; j < params_.Horizon_SCAN; ++j) {
      if (rangeMat.at<float>(i, j) != FLT_MAX) {
        // mark the points' column index for marking occlusion later
        cloudInfo.pointColInd[count] = j;
        // save range info
        cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
        // save extracted cloud
        extractedCloud->push_back(
            fullCloud->points[j + i * params_.Horizon_SCAN]);
        // size of extracted cloud
        ++count;
      }
    }
    cloudInfo.endRingIndex[i] = count - 1 - 5;
  }
}

} // namespace lio_sam