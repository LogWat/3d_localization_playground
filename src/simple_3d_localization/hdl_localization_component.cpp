#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <pcl_ros/point_cloud.hpp>
#include <pcl_ros/transforms.hpp>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <std_srvs/srv/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <small_gicp/util/downsampling_omp.hpp>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <simple_3d_localization/pose_estimator.hpp>
#include <simple_3d_localization/delta_estimater.hpp>

namespace s3l {

class HdlLocalizationComponent : public rclcpp::Node {

public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    HdlLocalizationComponent(const rclcpp::NodeOptions & options) {

    }

    virtual HdlLocalizationComponent(const rclcpp::NodeOptions & options, const std::string & node_name);
};
} // namespace simple_3d_localization

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(s3l::HdlLocalizationComponent)
