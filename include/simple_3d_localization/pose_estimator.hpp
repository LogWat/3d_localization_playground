#pragma once

#include <memory>
#include <boost/optional.hpp>
#include <boost/make_shared.hpp>

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <simple_3d_localization/pose_system.hpp>

namespace s3l {

namespace filter {
template <typename T> class UnscentedKalmanFilterX;
} // namespace filter

class PoseSystem;
class OdomSystem;


/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
    using PointT = pcl::PointXYZI;

    /**
     * @brief Constructor
     * @param registration          registration method
     * @param pos                   initial pose
     * @param quat                  initial quaternion
     * @param cool_time_duration    during "cool time", prediction is not performed
     */
    PoseEstimator(
        boost::shared_ptr<pcl::Registration<PointT, PointT>>& registration,
        const Eigen::Vector3f& pos,
        const Eigen::Quaternionf& quat,
        double cool_time_duration = 1.0
    );

    ~PoseEstimator();

    /**
     * @brief predict
     * @param stamp    timestamp
     */
    void predict(const rclcpp::Time& stamp);

    /**
     * @brief predict
     * @param stamp    timestamp
     * @param acc      acceleration
     * @param gyro     angular velocity
     */
    void predict(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro);

    /**
     * @brief update the state of the odomety-based pose estimation
     */
    void predict_odom(const Eigen::Matrix4f& odom_delta);

    /**
     * @brief correct
     * @param cloud   input cloud
     * @return cloud aligned to the globalmap
     */
    pcl::PointCloud<PointT>::Ptr correct(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud);

    /* getters */
    rclcpp::Time last_correct_time() const;

    Eigen::Vector3f pos() const;
    Eigen::Vector3f vel() const;
    Eigen::Quaternionf quat() const;
    Eigen::Matrix4f matrix() const;

    Eigen::Vector3f odom_pos() const;
    Eigen::Quaternionf odom_quat() const;
    Eigen::Matrix4f odom_matrix() const;

    const boost::optional<Eigen::Matrix4f>& wo_prediction_error() const;
    const boost::optional<Eigen::Matrix4f>& imu_prediction_error() const;
    const boost::optional<Eigen::Matrix4f>& odom_prediction_error() const;

private:
    rclcpp::Time init_stamp_;             // when the estimator was initialized
    rclcpp::Time prev_stamp_;             // when the estimator was updated last time
    rclcpp::Time last_correct_stamp_;      // when the estimator performed the correct step
    double cool_time_duration_;

    boost::shared_ptr<pcl::Registration<PointT, PointT>> registration_;

    std::unique_ptr<PoseSystem> pose_system_model_;
    std::unique_ptr<OdomSystem> odom_system_model_;


    Eigen::MatrixXf process_noise_;
    std::unique_ptr<filter::UnscentedKalmanFilterX<float>> ukf_;
    std::unique_ptr<filter::UnscentedKalmanFilterX<float>> odom_ukf_;

    Eigen::Matrix4f last_observation_;
    boost::optional<Eigen::Matrix4f> wo_pred_error_;
    boost::optional<Eigen::Matrix4f> imu_pred_error_;
    boost::optional<Eigen::Matrix4f> odom_pred_error_;
};

} // namespace simple_3d_localization
