cmake_minimum_required(VERSION 3.16)
project(lio_sam VERSION 0.1.0)

find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED core)
find_package(PCL REQUIRED COMPONENTS common kdtree filters)
find_package(OpenMP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###########
## Build ##
###########
add_library(${PROJECT_NAME})
target_sources(
    ${PROJECT_NAME}
    PUBLIC
        src/imuPreintegration.cpp
        src/imageProjection.cpp
        src/featureExtraction.cpp
        src/mapOptimization.cpp
)
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam
)

# handle optional openmp
if(OPENMP_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_FLAGS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC OPENMP_AVAILABLE)
else()
    message(STATUS "OpenMP not found, disabling OpenMP support")
endif()

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
