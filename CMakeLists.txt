cmake_minimum_required(VERSION 3.8)
project(simple_3d_localization)

cmake_policy(SET CMP0074 NEW)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)

  # 最適化
  # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -march=native")

  # Link Time Optimization (LTO)
  # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
  # set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
  # set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -flto")

  # Debug用
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

  if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_definitions(NDEBUG)
  endif()

endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  message(STATUS "OpenMP found and enabled.")
else()
  message(WARNING "OpenMP not found. Parallel processing will be disabled.")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen_kdl REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pcl_ros REQUIRED)

# External libraries
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(small_gicp REQUIRED)
# find_package(pclomp REQUIRED)
find_package(fast_gicp REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

set(msg_files
  "msg/ScanMatchingStatus.msg"
)

set(srv_files
  "srv/QueryGlobalLocalization.srv"
  "srv/SetGlobalMap.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# Globalmap Server Component
add_library(globalmap_server_component SHARED
  src/simple_3d_localization/globalmap_server_component.cpp
)
rclcpp_components_register_nodes(globalmap_server_component
  "s3l::GlobalmapServerComponent"
)
target_include_directories(globalmap_server_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
ament_target_dependencies(globalmap_server_component
  "rclcpp"
  "rclcpp_components"
  "std_msgs"
  "sensor_msgs"
  "pcl_conversions"
  "pcl_ros"
)
target_link_libraries(globalmap_server_component
  ${PCL_LIBRARIES}
)
install(TARGETS globalmap_server_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# HDL Localization Component
add_library(hdl_localization_component SHARED
  src/simple_3d_localization/hdl_localization_component.cpp
  src/simple_3d_localization/pose_estimator.cpp
)
rosidl_target_interfaces(hdl_localization_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
rclcpp_components_register_nodes(hdl_localization_component
  "s3l::HdlLocalizationComponent"
)
target_include_directories(hdl_localization_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${small_gicp_INCLUDE_DIRS} # 各ライブラリのインクルードパス (find_package で設定される想定)
  # ${pclomp_INCLUDE_DIRS}
  ${fast_gicp_INCLUDE_DIRS}
)
ament_target_dependencies(hdl_localization_component
  "rclcpp"
  "rclcpp_components"
  "std_srvs"
  "sensor_msgs"
  "nav_msgs"
  "geometry_msgs"
  "pcl_conversions"
  "tf2_eigen"
  "tf2_geometry_msgs"
  "pcl_ros"
  "tf2_eigen_kdl"
)
target_link_libraries(hdl_localization_component
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  Boost::system Boost::thread
  ${small_gicp_LIBRARIES}
  # ${pclomp_LIBRARIES}
  ${fast_gicp_LIBRARIES}
)
install(TARGETS hdl_localization_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install launch files 
# install(
#   DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}
# )

ament_package()
