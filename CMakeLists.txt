cmake_minimum_required(VERSION 3.16)
project(lio_sam)

find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenMP REQUIRED)


###########
## Build ##
###########
add_library(
    ${PROJECT_NAME} 
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
target_include_directories(
    ${PROJECT_NAME} PUBLIC 
        "${CMAKE_CURRENT_SOURCE_DIR}/include" 
        ${PCL_INCLUDE_DIRS} 
        ${OpenCV_INCLUDE_DIRS} 
        ${GTSAM_INCLUDE_DIR}
)
target_compile_features(
    ${PROJECT_NAME} PRIVATE cxx_std_17)