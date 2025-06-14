#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
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

#include <simple_3d_localization/msg/scan_matching_status.hpp>
#include <simple_3d_localization/srv/set_global_map.hpp>
#include <simple_3d_localization/srv/query_global_localization.hpp>

namespace s3l {

class HdlLocalizationComponent : public rclcpp::Node {

public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    HdlLocalizationComponent(const rclcpp::NodeOptions & options)
    : rclcpp::Node("s3l_hdl_localization", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        robot_odom_frame_id_ = this->declare_parameter<std::string>("robot_odom_frame_id", "odom");
        odom_child_frame_id_ = this->declare_parameter<std::string>("odom_child_frame_id", "base_link");
        reg_method_ = this->declare_parameter<std::string>("registration_method", "ndt"); // "ndt_cuda", "ndt_omp", "gicp", "vgicp"
        ndt_neighbor_search_method_ = this->declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");
        use_imu_ = this->declare_parameter<bool>("use_imu", false);
        invert_acc_ = this->declare_parameter<bool>("invert_acc", false);
        invert_gyro_ = this->declare_parameter<bool>("invert_gyro", false);
        use_omp_ = this->declare_parameter<bool>("use_omp", true);
        odometry_based_prediction_ = this->declare_parameter<bool>("odometry_based_prediction", true);
        downsample_leaf_size_ = this->declare_parameter<double>("downsample_leaf_size", 0.1);

        ndt_neightbor_search_radius_ = this->declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
        ndt_resolution_ = this->declare_parameter<double>("ndt_resolution", 1.0);

        use_globlal_localization_ = this->declare_parameter<bool>("use_global_localization", false);
        if (use_globlal_localization_) {
            RCLCPP_INFO(this->get_logger(), "Global localization is enabled.");
            RCLCPP_INFO(this->get_logger(), "Wait for global lcoalization services");
            // TODO 実装
        }

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(
            std::shared_ptr<rclcpp::Node>(this, [](auto) {}));

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", rclcpp::SensorDataQoS(),
            std::bind(&HdlLocalizationComponent::imuCallback, this, std::placeholders::_1));
        points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "points_raw", rclcpp::SensorDataQoS(),
            std::bind(&HdlLocalizationComponent::pointsCallback, this, std::placeholders::_1));
            
        aligned_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "aligned_points", rclcpp::SensorDataQoS());


    }

    HdlLocalizationComponent(const rclcpp::NodeOptions & options, const std::string & node_name);


private:
    void imuCallback(const sensor_msgs::msg::Imu::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
        imu_buffer_.push_back(msg);
    }
    void pointsCallback(const sensor_msgs::msg::PointCloud2::ConstPtr& msg) {
        if(!globalmap_) {
            RCLCPP_WARN(this->get_logger(), "Global map is not set yet. Ignoring point cloud.");
            return;
        }

        const auto& stamp = msg->header.stamp;
        PointCloudT::Ptr pcl_cloud(new PointCloudT);
        pcl::fromROSMsg(*msg, *pcl_cloud);

        if (pcl_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud. Ignoring.");
            return;
        }

        // transform pointcloud into odom_child_frame_id
        PointCloudT::Ptr cloud(new PointCloudT);
        try {
            geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform(
                robot_odom_frame_id_, odom_child_frame_id_, stamp, rclcpp::Duration::from_seconds(0.1));
            pcl_ros::transformPointCloud(*pcl_cloud, *cloud, transform);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "Failed to transform point cloud: %s", ex.what());
            return;
        }

        // downsample point cloud
        PointCloudT::Ptr downsampled_cloud(new PointCloudT);
        if (use_omp_) {
            downsampled_cloud = small_gicp::voxelgrid_sampling_omp(*cloud, downsample_leaf_size_);
        } else if (downsampler_) {
            downsampler_->setInputCloud(cloud);
            downsampler_->filter(*downsampled_cloud);
        } else {
            RCLCPP_WARN(this->get_logger(), "Downsampler is not set. Using original point cloud.");
            downsampled_cloud = cloud;
        }
        last_scan_ = downsampled_cloud;

        if (relocalizing_) delta_estimater_->add_frame(downsampled_cloud);

        // pose estimation (kalman filter)
        std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
        if (!pose_estimator_) {
            RCLCPP_ERROR(this->get_logger(), "Waiting for initial pose input!");
            return;
        }
        Eigen::Matrix4f before_pose = pose_estimator_->matrix();
        if (!use_imu_) {
            pose_estimator_->predict(stamp);
        } else {
            std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
            auto imu_iter = imu_buffer_.begin();
            for (; imu_iter != imu_buffer_.end(); ++imu_iter) {
                const auto& imu_msg = *imu_iter;
                if (rclcpp::Time(imu_msg->header.stamp) > stamp) {
                    break; // imu messages are sorted by timestamp
                }
                Eigen::Vector3f acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                Eigen::Vector3f gyro(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                if (invert_acc_) acc = -acc;
                if (invert_gyro_) gyro = -gyro;
                pose_estimator_->predict(stamp, acc, gyro);
            }
            imu_buffer_.erase(imu_buffer_.begin(), imu_iter); // remove processed imu messages
        }

        // odometry-based prediction
        rclcpp::Time last_correction_time = pose_estimator_->last_correct_time();
        if (odometry_based_prediction_ && last_correction_time > rclcpp::Time(0)) {
            try {
                geometry_msgs::msg::TransformStamped odom_transform = tf_buffer_.lookupTransform(
                    robot_odom_frame_id_, odom_child_frame_id_, last_correction_time, rclcpp::Duration::from_seconds(0.1));
                Eigen::Matrix4f odom_delta = tf2::transformToEigen(odom_transform).matrix();
                pose_estimator_->predict_odom(odom_delta);
            } catch (const tf2::TransformException & ex) {
                RCLCPP_ERROR(this->get_logger(), "Failed to get odometry transform: %s", ex.what());
            }
        }

        // correct
        auto aligned = pose_estimator_->correct(stamp, downsampled_cloud);
    }

    // ---------------------------------------------------------------------------------------------
    // helper functions

    pcl::Registration<PointT, PointT>::Ptr createRegistration() const {
        if (reg_method_ == "ndt_omp") {
            RCLCPP_INFO(this->get_logger(), "NDT_OMP is selected");
            pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<PointT, PointT>());
            ndt_omp->setTransformationEpsilon(0.01);
            ndt_omp->setResolution(ndt_resolution_);
            if (ndt_neighbor_search_method_ == "DIRECT1") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT1);
            } else if (ndt_neighbor_search_method_ == "DIRECT7") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
            } else if (ndt_neighbor_search_method_ == "KDTREE") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::KDTREE);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Unknown NDT neighbor search method: %s", ndt_neighbor_search_method_.c_str());
                return nullptr;
            }
            return ndt_omp;
        } else if (reg_method_ == "ndt_cuda") {
            RCLCPP_INFO(this->get_logger(), "NDT_CUDA is selected");
            boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>());
        }
    }


    void globalmapCallback(const sensor_msgs::msg::PointCloud2::ConstPtr& msg) {
        RCLCPP_INFO(this->get_logger(), "Global map received!");
        globalmap_ = std::make_shared<pcl::PointCloud<PointT>>();
        pcl::fromROSMsg(*msg, *globalmap_);

    }


    // variables ------------------------------------------------------------------------------------
    std::string robot_odom_frame_id_, odom_child_frame_id_;
    std::string reg_method_, ndt_neighbor_search_method_;
    double ndt_neightbor_search_radius_;
    double ndt_resolution_;

    bool use_imu_;
    bool invert_acc_, invert_gyro_;
    bool odometry_based_prediction_;
    bool use_omp_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
    rclcpp::Publisher<simple_3d_localization::msg::ScanMatchingStatus>::SharedPtr status_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // 複数callbackで参照される可能性

    // imu input buffer
    std::mutex imu_buffer_mutex_;
    std::vector<sensor_msgs::msg::Imu::ConstPtr> imu_buffer_;

    // globalmap and registration method
    pcl::PointCloud<PointT>::Ptr globalmap_;
    pcl::Filter<PointT>::Ptr downsampler_;
    pcl::Registration<PointT, PointT>::Ptr registration_;
    double downsample_leaf_size_;

    // pose estimator
    std::mutex pose_estimator_mutex_;
    std::unique_ptr<s3l::PoseEstimator> pose_estimator_;

    // global localization
    bool use_globlal_localization_;
    std::atomic_bool relocalizing_;
    std::unique_ptr<s3l::DeltaEstimater> delta_estimater_;

    pcl::PointCloud<PointT>::ConstPtr last_scan_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr relocalize_srv_;
    rclcpp::Client<simple_3d_localization::srv::SetGlobalMap>::SharedPtr set_globalmap_client_;
    rclcpp::Client<simple_3d_localization::srv::QueryGlobalLocalization>::SharedPtr query_global_localization_client_;
};
} // namespace simple_3d_localization

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(s3l::HdlLocalizationComponent)
