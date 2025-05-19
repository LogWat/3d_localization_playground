#include <simple_3d_localization/pose_estimator.hpp>

#include <pcl/filters/voxel_grid.h>
#include <simple_3d_localization/pose_system.hpp>
#include <simple_3d_localization/odom_system.hpp>
#include <simple_3d_localization/ukf.hpp>

namespace s3l {

/**
 * @brief constructor
 * @param registration        registration method
 * @param pos                 initial position
 * @param quat                initial orientation
 * @param cool_time_duration  during "cool time", prediction is not performed
 */
PoseEstimator::PoseEstimator(
    pcl::Registration<PointT, PointT>::Ptr& registration,
    const Eigen::Vector3f& pos,
    const Eigen::Quaternionf& quat,
    double cool_time_duration
): 
    registration_(registration),
    cool_time_duration_(cool_time_duration) 
{
    last_observation_ = Eigen::Matrix4f::Identity();
    last_observation_.block<3, 3>(0, 0) = quat.toRotationMatrix();
    last_observation_.block<3, 1>(0, 3) = pos;

    // pose_system の stateベクトルの次元
    // 位置(3) + 速度(3) + 姿勢(4) + bias(3) + bias_gyro(3) = 16
    process_noise_ = Eigen::MatrixXf::Identity(16, 16);
    process_noise_.middleRows(0, 3) *= 1.0;
    process_noise_.middleRows(3, 3) *= 1.0;
    process_noise_.middleRows(6, 4) *= 0.5;
    process_noise_.middleRows(10, 3) *= 1e-6;

    // 位置(3) + 姿勢(4) = 7
    Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
    measurement_noise.middleRows(0, 3) *= 0.01;
    measurement_noise.middleRows(3, 4) *= 0.001;

    // 初期状態
    Eigen::VectorXf mean(16);
    mean.middleRows(0, 3) = pos;
    mean.middleRows(3, 3).setZero();
    mean.middleRows(6, 4) = Eigen::Vector4f(quat.x(), quat.y(), quat.z(), quat.w()).normalized();
    mean.middleRows(10, 3).setZero();
    mean.middleRows(13, 3).setZero();

    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

    PoseSystem system_model;
    ukf_.reset(new filter::UnscentedKalmanFilterX<float, PoseSystem>(
        system_model,
        16,
        6,
        7,
        process_noise_,
        measurement_noise,
        mean,
        cov
    ));
}

PoseEstimator::~PoseEstimator() {}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const rclcpp::Time& stamp) {
    if (init_stamp_ == rclcpp::Time()) {
        init_stamp_ = stamp;
    }
    if ((stamp - init_stamp_).seconds() < cool_time_duration_ || prev_stamp
}



}