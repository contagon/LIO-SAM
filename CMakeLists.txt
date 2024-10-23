cmake_minimum_required(VERSION 3.16)
project(lio_sam)

find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer)


###########
## Build ##
###########

# Range Image Projection
# add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
# target_link_libraries(${PROJECT_NAME}_imageProjection ${PCL_LIBRARIES})
# target_include_directories(${PROJECT_NAME}_imageProjection PUBLIC include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIR})

# # Feature Association
# add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
# target_link_libraries(${PROJECT_NAME}_featureExtraction ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
# target_include_directories(${PROJECT_NAME}_featureExtraction PUBLIC include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIR})

# # Mapping Optimization
# add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp)
# target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(${PROJECT_NAME}_mapOptmization Boost::timer ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)
# target_include_directories(${PROJECT_NAME}_mapOptmization PUBLIC include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIR})

# IMU Preintegration
add_library(${PROJECT_NAME} src/featureExtraction.cpp)
target_link_libraries(${PROJECT_NAME} Boost::timer ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIR})
