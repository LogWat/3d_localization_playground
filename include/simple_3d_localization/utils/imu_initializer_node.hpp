#pragma once
#include <mutex>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <simple_3d_localization/type.hpp>
#include <simple_3d_localization/utils/imu_initializer.hpp>

namespace s3l::utils {

class ImuInitializerNode : public rclcpp::Node {
public:
    ImuInitializerNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("s3l_imu_initializer", options) {
        imu_topic_ = this->declare_parameter<std::string>("imu_topic", "imu/data");
        min_samples_ = this->declare_parameter<int>("min_samples", 200);
        gravity_magnitude_ = this->declare_parameter<double>("gravity_magnitude", 9.80665);
        auto_initialize_ = this->declare_parameter<bool>("auto_initialize", true);
        timeout_sec_ = this->declare_parameter<double>("timeout_sec", 5.0);

        this->declare_parameter<bool>("imu_initialized", false);
        this->declare_parameter<std::vector<double>>("gravity", {0.0, 0.0, -gravity_magnitude_});
        this->declare_parameter<std::vector<double>>("accel_bias", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>("gyro_bias", {0.0, 0.0, 0.0});

        gravity_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("imu_init/gravity", rclcpp::QoS(1).transient_local());
        accel_bias_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("imu_init/accel_bias", rclcpp::QoS(1).transient_local());
        gyro_bias_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("imu_init/gyro_bias", rclcpp::QoS(1).transient_local());

        init_srv_ = this->create_service<std_srvs::srv::Empty>(
            "initialize_imu",
            [this](const std_srvs::srv::Empty::Request::SharedPtr, std_srvs::srv::Empty::Response::SharedPtr) {
                this->initializeNow();
            });

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, rclcpp::SensorDataQoS(),
            std::bind(&ImuInitializerNode::imuCallback, this, std::placeholders::_1));

        start_time_ = now();
        init_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&ImuInitializerNode::checkAutoInitialize, this));

        RCLCPP_INFO(this->get_logger(), "ImuInitializerNode started. topic: %s, min_samples: %d, timeout: %.2fs",
                    imu_topic_.c_str(), min_samples_, timeout_sec_);
    }

private:
    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
        ImuData data;
        data.linear_acceleration = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        data.angular_velocity = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

        std::lock_guard<std::mutex> lock(mtx_);
        if (initialized_) return;
        buffer_.push_back(std::move(data));
    }

    void checkAutoInitialize() {
        if (!auto_initialize_) return;
        std::lock_guard<std::mutex> lock(mtx_);
        if (initialized_) return;

        const bool enough_samples = static_cast<int>(buffer_.size()) >= std::max(50, min_samples_);
        const bool timeout = (now() - start_time_) > rclcpp::Duration::from_seconds(timeout_sec_);
        if (enough_samples || (timeout && buffer_.size() >= 50)) {
            RCLCPP_INFO(this->get_logger(), "Trigger IMU initialization. samples: %zu, timeout: %d",
                        buffer_.size(), timeout ? 1 : 0);
            initializeNowLocked();
        }
    }

    void initializeNow() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (initialized_) {
            RCLCPP_INFO(this->get_logger(), "Already initialized. Re-initializing with current buffer.");
        }
        if (buffer_.size() < 50) {
            RCLCPP_WARN(this->get_logger(), "Not enough IMU data for initialization: %zu (< 50)", buffer_.size());
            return;
        }
        initializeNowLocked();
    }

    void initializeNowLocked() {
        ImuInitializer initializer(gravity_magnitude_);
        auto result = initializer.initialize(buffer_);
        if (!result.success) {
            RCLCPP_ERROR(this->get_logger(), "IMU initialization failed.");
            return;
        }

        // publish
        geometry_msgs::msg::Vector3::UniquePtr g_msg = std::make_unique<geometry_msgs::msg::Vector3>();
        geometry_msgs::msg::Vector3::UniquePtr ba_msg = std::make_unique<geometry_msgs::msg::Vector3>();
        geometry_msgs::msg::Vector3::UniquePtr bg_msg = std::make_unique<geometry_msgs::msg::Vector3>();
        g_msg->x = result.gravity_vec.x(); g_msg->y = result.gravity_vec.y(); g_msg->z = result.gravity_vec.z();
        ba_msg->x = result.accel_bias.x(); ba_msg->y = result.accel_bias.y(); ba_msg->z = result.accel_bias.z();
        bg_msg->x = result.gyro_bias.x();  bg_msg->y = result.gyro_bias.y();  bg_msg->z = result.gyro_bias.z();
        gravity_pub_->publish(std::move(g_msg));
        accel_bias_pub_->publish(std::move(ba_msg));
        gyro_bias_pub_->publish(std::move(bg_msg));

        // set parameters
        this->set_parameters({
            rclcpp::Parameter("imu_initialized", true),
            rclcpp::Parameter("gravity", std::vector<double>{result.gravity_vec.x(), result.gravity_vec.y(), result.gravity_vec.z()}),
            rclcpp::Parameter("accel_bias", std::vector<double>{result.accel_bias.x(), result.accel_bias.y(), result.accel_bias.z()}),
            rclcpp::Parameter("gyro_bias", std::vector<double>{result.gyro_bias.x(), result.gyro_bias.y(), result.gyro_bias.z()})
        });

        RCLCPP_INFO(this->get_logger(), "IMU initialized. |g|=%.6f, gravity=[%.6f %.6f %.6f], ba=[%.6f %.6f %.6f], bg=[%.6f %.6f %.6f]",
            result.gravity_vec.norm(),
            result.gravity_vec.x(), result.gravity_vec.y(), result.gravity_vec.z(),
            result.accel_bias.x(), result.accel_bias.y(), result.accel_bias.z(),
            result.gyro_bias.x(), result.gyro_bias.y(), result.gyro_bias.z());

        initialized_ = true;
        buffer_.clear();
    }

private:
    std::string imu_topic_;
    int min_samples_{200};
    double gravity_magnitude_{9.80665};
    bool auto_initialize_{true};
    double timeout_sec_{5.0};

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr gravity_pub_, accel_bias_pub_, gyro_bias_pub_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr init_srv_;
    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::Time start_time_;

    std::mutex mtx_;
    std::vector<ImuData> buffer_;
    bool initialized_{false};
};

} // namespace s3l::utils