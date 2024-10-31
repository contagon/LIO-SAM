# LIO-SAM
This is a fork of the original LIO-SAM to enable usage without ROS, specifically in an offline setting for evaluation purposes in [evalio](TODO). 

The following changes have been made,
- Removal of ROS dependencies
- Removal of loop closure and GPS code. The only bits remaining are the Lidar-Inertial Odometry code.
- Removal of dependence on a 9-axis IMU. The vehicle is assumed to be at rest for the first tenth of a second to initialize bias and orientation using the accelerometer.
- Clean up some of the cruft of the original LIO-SAM implementation, mostly related to dealing with ROS types.

Everything appears to function appropriately. If you'd like to run this, I recommend you check out [evalio](TODO) for a simple python wrapper and connection to existing datasets.


## TODOs
There's still a lot that could be done to improve things IMO, such as,
- Deskewing using only IMU integrated poses and not both integrated poses and the accelerometer (adds unnecessary complexity).
- The `CloudInfo` type should be broken apart into it's various pieces for the different steps it's used in.
- Anywhere orientation is stored as RPY should be ~~burned~~ switched to quaternions.
- Currently the `mapOptimization` class stores pose history as a `pcl::PointCloud<PointTypePose>` and should really be either a `std::vector<gtsam::Pose3>` or something similar.
- Add some tests to make sure RPY conversions are all correct in `types.h` (Or remove all the RPY code). 
- *Maybe* reintroduce an optional ROS wrapper around this improved code, it should be extremely straightforward to do now (IE an IMU/LIDAR callback and conversion from `ros2::PointCloud2` -> `pcl::PointCloud` is all you need)

I may eventually get to these changes, but frankly, things somehow *work* right now so I'm going to leave it as is for the forseeable future.

## Building

Dependencies:
- Eigen3
- gtsam
- pcl
- OpenCV

Building is done in the usual CMake fashion.

## Contributions

If you'd like to contribute, please do! I'm happy to accept pull requests (especially see the list above), issues, or questions. The goal here is to ease running baselines for LIO papers. 

## Acknowledgements

Obviously, this code is based on [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) and I appreciate them making it open-source so we can build on it here.