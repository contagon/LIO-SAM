cmake_minimum_required(VERSION 3.16)
project(lio_sam VERSION 0.1.0 LANGUAGES CXX)

find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)

set(GTSAM_DEPENDENCY "find_dependency (GTSAM ${GTSAM_VERSION} QUIET)")
set(OPENCV_DEPENDENCY "find_dependency (OpenCV ${OpenCV_VERSION} QUIET)")
set(PCL_DEPENDENCY "find_dependency (PCL ${PCL_VERSION} QUIET)")
# TODO: OpenMP_VERSION is blank, not sure why, but still works
set(OpenMP_DEPENDENCY "find_dependency (OpenMP ${OpenMP_VERSION} QUIET)")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------- Build ------------------------- #
add_library(${PROJECT_NAME})
target_sources(
    ${PROJECT_NAME} 
    PRIVATE
        src/imuPreintegration.cpp
        src/imageProjection.cpp 
        src/featureExtraction.cpp 
        src/mapOptimization.cpp
)
target_link_libraries(
    ${PROJECT_NAME} PUBLIC
        ${PCL_LIBRARIES} 
        ${OpenCV_LIBRARIES} 
        ${OpenMP_CXX_FLAGS} 
        gtsam
)
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)