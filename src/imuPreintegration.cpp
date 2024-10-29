#include "LIO-SAM/imuPreintegration.h"

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

namespace lio_sam {

IMUPreintegration::IMUPreintegration(const LioSamParams &params)
    : params_(params) {
  imu2Lidar =
      gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
                   gtsam::Point3(-params_.extTrans.x(), -params_.extTrans.y(),
                                 -params_.extTrans.z()));
  lidar2Imu =
      gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
                   gtsam::Point3(params_.extTrans.x(), params_.extTrans.y(),
                                 params_.extTrans.z()));

  boost::shared_ptr<gtsam::PreintegrationParams> p =
      gtsam::PreintegrationParams::MakeSharedU(params_.imuGravity);
  p->accelerometerCovariance =
      gtsam::Matrix33::Identity(3, 3) *
      pow(params_.imuAccNoise, 2); // acc white noise in continuous
  p->gyroscopeCovariance =
      gtsam::Matrix33::Identity(3, 3) *
      pow(params_.imuGyrNoise, 2); // gyro white noise in continuous
  p->integrationCovariance =
      gtsam::Matrix33::Identity(3, 3) *
      pow(1e-4, 2); // error committed in integrating position from velocities
  gtsam::imuBias::ConstantBias prior_imu_bias(
      (gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
  ; // assume zero initial bias

  priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2)
          .finished()); // rad,rad,rad,m, m, m
  priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
  priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(
      6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
  correctionNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1)
          .finished()); // rad,rad,rad,m, m, m
  noiseModelBetweenBias =
      (gtsam::Vector(6) << params_.imuAccBiasN, params_.imuAccBiasN,
       params_.imuAccBiasN, params_.imuGyrBiasN, params_.imuGyrBiasN,
       params_.imuGyrBiasN)
          .finished();

  imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(
      p, prior_imu_bias); // setting up the IMU integration for IMU message
                          // thread
  imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(
      p, prior_imu_bias); // setting up the IMU integration for optimization
}

void IMUPreintegration::resetOptimization() {
  gtsam::ISAM2Params optParameters;
  optParameters.relinearizeThreshold = 0.1;
  optParameters.relinearizeSkip = 1;
  optimizer = gtsam::ISAM2(optParameters);

  gtsam::NonlinearFactorGraph newGraphFactors;
  graphFactors = newGraphFactors;

  gtsam::Values NewGraphValues;
  graphValues = NewGraphValues;
}

void IMUPreintegration::resetParams() {
  lastImuT_imu = -1;
  doneFirstOpt = false;
  systemInitialized = false;
}

void IMUPreintegration::odometryHandler(const Odometry &odomMsg) {
  std::lock_guard<std::mutex> lock(mtx);

  double currentCorrectionTime = odomMsg.stamp;

  // make sure we have imu data to integrate
  if (imuQueOpt.empty())
    return;

  float p_x = odomMsg.position.x();
  float p_y = odomMsg.position.y();
  float p_z = odomMsg.position.z();
  float r_x = odomMsg.orientation.x();
  float r_y = odomMsg.orientation.y();
  float r_z = odomMsg.orientation.z();
  float r_w = odomMsg.orientation.w();
  gtsam::Pose3 lidarPose =
      gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z),
                   gtsam::Point3(p_x, p_y, p_z));

  // 0. initialize system
  if (systemInitialized == false) {
    resetOptimization();

    // pop old IMU message
    while (!imuQueOpt.empty()) {
      if (imuQueOpt.front().stamp < currentCorrectionTime - delta_t) {
        lastImuT_opt = imuQueOpt.front().stamp;
        imuQueOpt.pop_front();
      } else
        break;
    }
    // initial pose
    prevPose_ = lidarPose.compose(lidar2Imu);
    gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
    graphFactors.add(priorPose);
    // initial velocity
    prevVel_ = gtsam::Vector3(0, 0, 0);
    gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
    graphFactors.add(priorVel);
    // initial bias
    prevBias_ = gtsam::imuBias::ConstantBias();
    gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_,
                                                               priorBiasNoise);
    graphFactors.add(priorBias);
    // add values
    graphValues.insert(X(0), prevPose_);
    graphValues.insert(V(0), prevVel_);
    graphValues.insert(B(0), prevBias_);
    // optimize once
    optimizer.update(graphFactors, graphValues);
    graphFactors.resize(0);
    graphValues.clear();

    imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

    key = 1;
    systemInitialized = true;
    return;
  }

  // reset graph for speed
  if (key == 100) {
    // get updated noise before reset
    gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise =
        gtsam::noiseModel::Gaussian::Covariance(
            optimizer.marginalCovariance(X(key - 1)));
    gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise =
        gtsam::noiseModel::Gaussian::Covariance(
            optimizer.marginalCovariance(V(key - 1)));
    gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise =
        gtsam::noiseModel::Gaussian::Covariance(
            optimizer.marginalCovariance(B(key - 1)));
    // reset graph
    resetOptimization();
    // add pose
    gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_,
                                               updatedPoseNoise);
    graphFactors.add(priorPose);
    // add velocity
    gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_,
                                                updatedVelNoise);
    graphFactors.add(priorVel);
    // add bias
    gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(
        B(0), prevBias_, updatedBiasNoise);
    graphFactors.add(priorBias);
    // add values
    graphValues.insert(X(0), prevPose_);
    graphValues.insert(V(0), prevVel_);
    graphValues.insert(B(0), prevBias_);
    // optimize once
    optimizer.update(graphFactors, graphValues);
    graphFactors.resize(0);
    graphValues.clear();

    key = 1;
  }

  // 1. integrate imu data and optimize
  while (!imuQueOpt.empty()) {
    // pop and integrate imu data that is between two optimizations
    Imu thisImu = imuQueOpt.front();
    double imuTime = thisImu.stamp;
    if (imuTime < currentCorrectionTime - delta_t) {
      double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
      imuIntegratorOpt_->integrateMeasurement(
          gtsam::Vector3(thisImu.acc.x(), thisImu.acc.y(), thisImu.acc.z()),
          gtsam::Vector3(thisImu.gyro.x(), thisImu.gyro.y(), thisImu.gyro.z()),
          dt);

      lastImuT_opt = imuTime;
      imuQueOpt.pop_front();
    } else
      break;
  }
  // add imu factor to graph
  const gtsam::PreintegratedImuMeasurements &preint_imu =
      dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(
          *imuIntegratorOpt_);
  gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key),
                              B(key - 1), preint_imu);
  graphFactors.add(imu_factor);
  // add imu bias between factor
  graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
      B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
      gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) *
                                          noiseModelBetweenBias)));
  // add pose factor
  gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
  gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose,
                                               correctionNoise);
  graphFactors.add(pose_factor);
  // insert predicted values
  gtsam::NavState propState_ =
      imuIntegratorOpt_->predict(prevState_, prevBias_);
  graphValues.insert(X(key), propState_.pose());
  graphValues.insert(V(key), propState_.v());
  graphValues.insert(B(key), prevBias_);
  // optimize
  optimizer.update(graphFactors, graphValues);
  optimizer.update();
  graphFactors.resize(0);
  graphValues.clear();
  // Overwrite the beginning of the preintegration for the next step.
  gtsam::Values result = optimizer.calculateEstimate();
  prevPose_ = result.at<gtsam::Pose3>(X(key));
  prevVel_ = result.at<gtsam::Vector3>(V(key));
  prevState_ = gtsam::NavState(prevPose_, prevVel_);
  prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
  // Reset the optimization preintegration object.
  imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
  // check optimization
  if (failureDetection(prevVel_, prevBias_)) {
    resetParams();
    return;
  }

  // 2. after optiization, re-propagate imu odometry preintegration
  prevStateOdom = prevState_;
  prevBiasOdom = prevBias_;
  // first pop imu message older than current correction data
  double lastImuQT = -1;
  while (!imuQueImu.empty() &&
         imuQueImu.front().stamp < currentCorrectionTime - delta_t) {
    lastImuQT = imuQueImu.front().stamp;
    imuQueImu.pop_front();
  }
  // repropogate
  if (!imuQueImu.empty()) {
    // reset bias use the newly optimized bias
    imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
    // integrate imu message from the beginning of this optimization
    for (int i = 0; i < (int)imuQueImu.size(); ++i) {
      Imu thisImu = imuQueImu[i];
      double imuTime = thisImu.stamp;
      double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);

      imuIntegratorImu_->integrateMeasurement(
          gtsam::Vector3(thisImu.acc.x(), thisImu.acc.y(), thisImu.acc.z()),
          gtsam::Vector3(thisImu.gyro.x(), thisImu.gyro.y(), thisImu.gyro.z()),
          dt);
      lastImuQT = imuTime;
    }
  }

  ++key;
  doneFirstOpt = true;
}

bool IMUPreintegration::failureDetection(
    const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur) {
  Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
  if (vel.norm() > 30) {
    std::cout << "Large velocity, reset IMU-preintegration!" << std::endl;
    return true;
  }

  Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(),
                     biasCur.accelerometer().z());
  Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(),
                     biasCur.gyroscope().z());
  if (ba.norm() > 1.0 || bg.norm() > 1.0) {
    std::cout << "Large bias, reset IMU-preintegration!" << std::endl;
    return true;
  }

  return false;
}

std::optional<Odometry> IMUPreintegration::imuHandler(const Imu &imu_raw) {
  std::lock_guard<std::mutex> lock(mtx);

  Imu thisImu = params_.imuConverter(imu_raw);

  imuQueOpt.push_back(thisImu);
  imuQueImu.push_back(thisImu);

  if (doneFirstOpt == false)
    return {};

  double imuTime = thisImu.stamp;
  double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
  lastImuT_imu = imuTime;

  // integrate this single imu message
  imuIntegratorImu_->integrateMeasurement(
      gtsam::Vector3(thisImu.acc.x(), thisImu.acc.y(), thisImu.acc.z()),
      gtsam::Vector3(thisImu.gyro.x(), thisImu.gyro.y(), thisImu.gyro.z()), dt);

  // predict odometry
  gtsam::NavState currentState =
      imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

  // publish odometry
  // transform imu pose to lidar
  gtsam::Pose3 imuPose =
      gtsam::Pose3(currentState.quaternion(), currentState.position());
  gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);
  Odometry odometry{
      .stamp = thisImu.stamp,
      .orientation = lidarPose.rotation().toQuaternion(),
      .position = lidarPose.translation(),
  };

  return odometry;
}

} // namespace lio_sam