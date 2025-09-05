#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <simple_3d_localization/type.hpp>
#include <simple_3d_localization/filter/filter.hpp>
#include <simple_3d_localization/filter/ukf.hpp>
#include <simple_3d_localization/filter/ekf.hpp>
#include <simple_3d_localization/model/ukf_pose.hpp>
#include <simple_3d_localization/model/ekf_pose.hpp>

namespace s3l
{

namespace hdl_localization 
{

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
    explicit PoseEstimator(
        std::shared_ptr<pcl::Registration<PointT, PointT>>& registration,
        const Vector3t& pos,
        const Quaterniont& quat,
        FilterType filter_type,
        double cool_time_duration = 1.0
    ): 
        cool_time_duration_(cool_time_duration),
        registration_(registration),
        filter_type_(filter_type)
    {
        last_observation_ = Matrix4t::Identity();
        last_observation_.block<3, 3>(0, 0) = quat.toRotationMatrix();
        last_observation_.block<3, 1>(0, 3) = pos;

        // pose_system の stateベクトルの次元
        // 位置(3) + 速度(3) + 姿勢(4) + bias(3) + bias_gyro(3) + gravity(3) = 19
        process_noise_ = MatrixXt::Identity(19, 19);
        process_noise_.middleRows(0, 3) *= 1.0;
        process_noise_.middleRows(3, 3) *= 1.0;
        process_noise_.middleRows(6, 4) *= 0.5;
        process_noise_.middleRows(10, 3) *= 1e-5;
        process_noise_.middleRows(13, 3) *= 1e-5;
        process_noise_.middleRows(16, 3) *= 1e-5;

        // 位置(3) + 姿勢(4) = 7
        measurement_noise_ = MatrixXt::Identity(7, 7);
        measurement_noise_.middleRows(0, 3) *= 0.01;
        measurement_noise_.middleRows(3, 4) *= 0.001;

        // 初期状態
        VectorXt mean(19);
        mean.middleRows(0, 3) = pos;
        mean.middleRows(3, 3).setZero();
        mean.middleRows(6, 4) = Vector4t(quat.w(), quat.x(), quat.y(), quat.z()).normalized();
        mean.middleRows(10, 3).setZero();
        mean.middleRows(13, 3).setZero();
        mean.middleRows(16, 3) = Vector3t(0.0f, 0.0f, -9.81f); // 重力ベクトル

        MatrixXt cov = MatrixXt::Identity(19, 19) * 0.01;

        if (filter_type_ == FilterType::UKF) {
            ukf_system_model_ = std::make_unique<model::UKFPoseSystemModel>();
            filter_ = std::make_unique<filter::UnscentedKalmanFilterX>(
                *ukf_system_model_, 19, 6, 7, process_noise_, measurement_noise_, mean, cov
            );
        } else if (filter_type_ == FilterType::EKF) {
            ekf_system_model_ = std::make_unique<model::EKFPoseSystemModel>();
            filter_ = std::make_unique<filter::ExtendedKalmanFilterX>(
                *ekf_system_model_, 19, mean, cov, process_noise_
            );
            filter_->setMeasurementNoise(measurement_noise_);
        }
    }

    ~PoseEstimator() {}


    /**
     * @brief predict
     * @param stamp    timestamp
     * @param acc      acceleration
     * @param gyro     angular velocity
     */
    void predict(const rclcpp::Time& stamp, const Vector3t& acc, const Vector3t& gyro) {
        if (init_stamp_ == rclcpp::Time()) {
            init_stamp_ = stamp;
        }
        if ((stamp - init_stamp_).seconds() < cool_time_duration_ || prev_stamp_ == rclcpp::Time() || (stamp - prev_stamp_).seconds() < 0.01) {
            prev_stamp_ = stamp;
            return;
        }

        double dt = (stamp - prev_stamp_).seconds();
        prev_stamp_ = stamp;

        filter_->setDt(dt);
        filter_->setProcessNoise(process_noise_ * dt);

        VectorXt u(6); u.head<3>() = acc; u.tail<3>() = gyro;
        filter_->predict(u);
    }

    /**
     * @brief correct
     * @param cloud   input cloud
     * @return cloud aligned to the globalmap
     */
    pcl::PointCloud<PointT>::Ptr correct(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
        if (init_stamp_ == rclcpp::Time()) {
            init_stamp_ = stamp;
        }

        last_correct_stamp_ = stamp;

        Matrix4t no_guess = last_observation_;
        Matrix4t imu_guess;
        Matrix4t init_guess = Matrix4t::Identity();

        init_guess = imu_guess = matrix();

        pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
        registration_->setInputSource(cloud);
        registration_->align(*aligned, init_guess); // 事前に設定されているregistration方法でalign (NDT_CUDA, GICP, etc.)

        Matrix4t trans = registration_->getFinalTransformation();
        Vector3t p = trans.block<3, 1>(0, 3);
        Quaterniont q(trans.block<3, 3>(0, 0));

        if (quat().coeffs().dot(q.coeffs()) < 0.0f) q.coeffs() *= -1.0f; // quaternionの符号を合わせる

        VectorXt observation(7);
        observation.middleRows(0, 3) = p;
        observation.middleRows(3, 4) = Vector4t(q.w(), q.x(), q.y(), q.z()).normalized();
        last_observation_ = trans;

        wo_pred_error_ = no_guess.inverse() * registration_->getFinalTransformation();

        filter_->correct(observation);
        imu_pred_error_ = imu_guess.inverse() * registration_->getFinalTransformation();
                    
        return aligned;
    }

    /* getters */
    rclcpp::Time last_correct_time() const {
        return last_correct_stamp_;
    }
    Vector3t pos() const {
        const auto& s = filter_->getState();
        return { s[0], s[1], s[2] };
    }
    Vector3t vel() const {
        const auto& s = filter_->getState();
        return { s[3], s[4], s[5] };
    }
    Quaterniont quat() const {
        const auto& s = filter_->getState();
        return Quaterniont(s[6], s[7], s[8], s[9]).normalized();
    }
    Matrix4t matrix() const {
        Matrix4t mat = Matrix4t::Identity();
        mat.block<3, 3>(0, 0) = quat().toRotationMatrix();
        mat.block<3, 1>(0, 3) = pos();
        return mat;
    }
    const std::optional<Matrix4t>& wo_prediction_error() const {
        return wo_pred_error_;
    }
    const std::optional<Matrix4t>& imu_prediction_error() const {
        return imu_pred_error_;
    }


    /* utils */
    void initializeWithBiasAndGravity(const Vector3t& gravity, const Vector3t& accel_bias, const Vector3t& gyro_bias) {
        VectorXt mean = filter_->getState();
        mean.middleRows(10, 3) = accel_bias;
        mean.middleRows(13, 3) = gyro_bias;
        mean.middleRows(16, 3) = gravity;
        filter_->setMean(mean);
    }

private:
    rclcpp::Time init_stamp_;             // when the estimator was initialized
    rclcpp::Time prev_stamp_;             // when the estimator was updated last time
    rclcpp::Time last_correct_stamp_;      // when the estimator performed the correct step
    double cool_time_duration_;

    std::shared_ptr<pcl::Registration<PointT, PointT>> registration_;

    MatrixXt process_noise_;
    MatrixXt measurement_noise_;

    Matrix4t last_observation_;
    std::optional<Matrix4t> wo_pred_error_;
    std::optional<Matrix4t> imu_pred_error_;

    FilterType filter_type_;
    std::unique_ptr<model::UKFPoseSystemModel> ukf_system_model_;
    std::unique_ptr<model::EKFPoseSystemModel> ekf_system_model_;
    std::unique_ptr<filter::KalmanFilterX> filter_;
};

} // namespace hdl_localization
} // namespace s3l

