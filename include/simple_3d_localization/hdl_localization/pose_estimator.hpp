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
#include <simple_3d_localization/model/odom_system.hpp>
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
        const Eigen::Vector3f& pos,
        const Eigen::Quaternionf& quat,
        FilterType filter_type,
        double cool_time_duration = 1.0
    ): 
        cool_time_duration_(cool_time_duration),
        registration_(registration),
        filter_type_(filter_type)
    {
        last_observation_ = Eigen::Matrix4f::Identity();
        last_observation_.block<3, 3>(0, 0) = quat.toRotationMatrix();
        last_observation_.block<3, 1>(0, 3) = pos;

        // pose_system の stateベクトルの次元
        // 位置(3) + 速度(3) + 姿勢(4) + bias(3) + bias_gyro(3) + gravity(3) = 19
        process_noise_ = Eigen::MatrixXf::Identity(19, 19);
        process_noise_.middleRows(0, 3) *= 1.0;
        process_noise_.middleRows(3, 3) *= 1.0;
        process_noise_.middleRows(6, 4) *= 0.5;
        process_noise_.middleRows(10, 3) *= 1e-3;
        process_noise_.middleRows(13, 3) *= 1e-5;
        process_noise_.middleRows(16, 3) *= 1e-5;

        // 位置(3) + 姿勢(4) = 7
        measurement_noise_ = Eigen::MatrixXf::Identity(7, 7);
        measurement_noise_.middleRows(0, 3) *= 0.01;
        measurement_noise_.middleRows(3, 4) *= 0.001;

        // 初期状態
        Eigen::VectorXf mean(19);
        mean.middleRows(0, 3) = pos;
        mean.middleRows(3, 3).setZero();
        mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z()).normalized();
        mean.middleRows(10, 3).setZero();
        mean.middleRows(13, 3).setZero();
        mean.middleRows(16, 3) = Eigen::Vector3f(0.0f, 0.0f, -9.81f); // 重力ベクトル

        Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(19, 19) * 0.01;

        if (filter_type_ == FilterType::UKF) {
            ukf_system_model_ = std::make_unique<model::UKFPoseSystemModel>();
            filter_ = std::make_unique<filter::UnscentedKalmanFilterX<float>>(
                *ukf_system_model_, 19, 6, 7, process_noise_, measurement_noise_, mean, cov
            );
        } else if (filter_type_ == FilterType::EKF) {
            ekf_system_model_ = std::make_unique<model::EKFPoseSystemModel>();
            filter_ = std::make_unique<filter::ExtendedKalmanFilterX<float>>(
                *ekf_system_model_, 19, mean, cov, process_noise_
            );
            filter_->setMeasurementNoise(measurement_noise_);
        }
    }

    ~PoseEstimator() {}

    /**
     * @brief predict
     * @param stamp    timestamp
     */
    void predict(const rclcpp::Time& stamp) {
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
        filter_->predict();
    }

    /**
     * @brief predict
     * @param stamp    timestamp
     * @param acc      acceleration
     * @param gyro     angular velocity
     */
    void predict(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
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

        Eigen::VectorXf u(6); u.head<3>() = acc; u.tail<3>() = gyro;
        filter_->predict(u);
    }

    /**
     * @brief update the state of the odomety-based pose estimation
     */
    void predict_odom(const Eigen::Matrix4f& odom_delta) {
        if(!odom_ukf_) {
            // odom_ukf_ が初期化されていない場合は、初期化する
            Eigen::MatrixXf odom_process_noise = Eigen::MatrixXf::Identity(7, 7);
            Eigen::MatrixXf odom_measurement_noise = Eigen::MatrixXf::Identity(7, 7) * 1e-3;

            Eigen::VectorXf odom_mean(7);
            Eigen::VectorXf filter_mean = filter_->getState();
            odom_mean.block<3, 1>(0, 0) = Eigen::Vector3f(filter_mean[0], filter_mean[1], filter_mean[2]);
            odom_mean.block<4, 1>(3, 0) = Eigen::Vector4f(filter_mean[6], filter_mean[7], filter_mean[8], filter_mean[9]).normalized();
            Eigen::MatrixXf odom_cov = Eigen::MatrixXf::Identity(7, 7) * 1e-2;

            odom_system_model_ = std::make_unique<model::OdomSystemModel>();
            odom_ukf_.reset(new filter::UnscentedKalmanFilterX<float>(
                *odom_system_model_,
                7,  // state dimension
                7,  // input dimension
                7,  // measurement dimension
                odom_process_noise,
                odom_measurement_noise,
                odom_mean,
                odom_cov
            ));
        }

        // rotation axis が反転している場合は、符号を反転する
        Eigen::Quaternionf quat(odom_delta.block<3, 3>(0, 0));
        if (odom_quat().coeffs().dot(quat.coeffs()) < 0) {
            quat.coeffs() *= -1.0f; // quaternionの符号を合わせる
        }

        Eigen::VectorXf odom_control(7);
        odom_control.middleRows(0, 3) = odom_delta.block<3, 1>(0, 3); // translation
        odom_control.middleRows(3, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z()).normalized(); // rotation

        process_noise_.topLeftCorner(3, 3) = Eigen::Matrix3f::Identity() * odom_delta.block<3, 1>(0, 3).norm() + Eigen::Matrix3f::Identity() * 1e-3;
        process_noise_.bottomRightCorner(4, 4) = Eigen::Matrix4f::Identity() * (1 - std::abs(quat.w())) + Eigen::Matrix4f::Identity() * 1e-3;

        odom_ukf_->setProcessNoiseCov(process_noise_);
        odom_ukf_->predict(odom_control);
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

        Eigen::Matrix4f no_guess = last_observation_;
        Eigen::Matrix4f imu_guess;
        Eigen::Matrix4f odom_guess;
        Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

        if (!odom_ukf_) {
            init_guess = imu_guess = matrix();
        } else {
            imu_guess = matrix();
            odom_guess = odom_matrix();

            Eigen::VectorXf imu_mean(7);
            Eigen::MatrixXf imu_cov = Eigen::MatrixXf::Identity(7, 7);
            Eigen::VectorXf filter_mean = filter_->getState();
            Eigen::MatrixXf filter_cov = filter_->getCovariance();
            imu_mean.block<3, 1>(0, 0) = filter_mean.block<3, 1>(0, 0);
            imu_mean.block<4, 1>(3, 0) = filter_mean.block<4, 1>(6, 0).normalized();
            imu_cov.block<3, 3>(0, 0) = filter_cov.block<3, 3>(0, 0);
            imu_cov.block<3, 4>(0, 3) = filter_cov.block<3, 4>(0, 6);
            imu_cov.block<4, 3>(3, 0) = filter_cov.block<4, 3>(6, 0);
            imu_cov.block<4, 4>(3, 3) = filter_cov.block<4, 4>(6, 6);

            Eigen::VectorXf odom_mean = odom_ukf_->mean_;
            Eigen::MatrixXf odom_cov = odom_ukf_->cov_;

            if (imu_mean.tail<4>().dot(odom_mean.tail<4>()) < 0.0)  {
                odom_mean.tail<4>() *= -1.0; // quaternionの符号を合わせる
            }

            Eigen::MatrixXf inv_imu_cov = imu_cov.inverse();
            Eigen::MatrixXf inv_odom_cov = odom_cov.inverse();

            Eigen::MatrixXf fused_cov = (inv_imu_cov + inv_odom_cov).inverse();
            Eigen::VectorXf fused_mean = fused_cov * inv_imu_cov * imu_mean + fused_cov * inv_odom_cov * odom_mean;

            init_guess.block<3, 1>(0, 3) = Eigen::Vector3f(fused_mean[0], fused_mean[1], fused_mean[2]);
            init_guess.block<3, 3>(0, 0) = Eigen::Quaternionf(fused_mean[3], fused_mean[4], fused_mean[5], fused_mean[6]).normalized().toRotationMatrix();
        }

        pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
        registration_->setInputSource(cloud);
        registration_->align(*aligned, init_guess); // 事前に設定されているregistration方法でalign (NDT_CUDA, GICP, etc.)

        Eigen::Matrix4f trans = registration_->getFinalTransformation();
        Eigen::Vector3f p = trans.block<3, 1>(0, 3);
        Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

        if (quat().coeffs().dot(q.coeffs()) < 0.0f) q.coeffs() *= -1.0f; // quaternionの符号を合わせる

        Eigen::VectorXf observation(7);
        observation.middleRows(0, 3) = p;
        observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z()).normalized();
        last_observation_ = trans;

        wo_pred_error_ = no_guess.inverse() * registration_->getFinalTransformation();

        filter_->correct(observation);
        imu_pred_error_ = imu_guess.inverse() * registration_->getFinalTransformation();

        if (odom_ukf_) {
            if (observation.tail<4>().dot(odom_ukf_->mean_.tail<4>()) < 0.0) {
                odom_ukf_->mean_.tail<4>() *= -1.0; // quaternionの符号を合わせる
            }
            odom_ukf_->correct(observation);
            odom_pred_error_ = odom_guess.inverse() * registration_->getFinalTransformation();
        }
                    
        return aligned;
    }

    /* getters */
    rclcpp::Time last_correct_time() const {
        return last_correct_stamp_;
    }
    Eigen::Vector3f pos() const {
        const auto& s = filter_->getState();
        return { s[0], s[1], s[2] };
    }
    Eigen::Vector3f vel() const {
        const auto& s = filter_->getState();
        return { s[3], s[4], s[5] };
    }
    Eigen::Quaternionf quat() const {
        const auto& s = filter_->getState();
        return Eigen::Quaternionf(s[6], s[7], s[8], s[9]).normalized();
    }
    Eigen::Matrix4f matrix() const {
        Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
        mat.block<3, 3>(0, 0) = quat().toRotationMatrix();
        mat.block<3, 1>(0, 3) = pos();
        return mat;
    }
    Eigen::Vector3f odom_pos() const {
        return Eigen::Vector3f(odom_ukf_->mean_[0], odom_ukf_->mean_[1], odom_ukf_->mean_[2]);
    }
    Eigen::Quaternionf odom_quat() const {
        return Eigen::Quaternionf(odom_ukf_->mean_[3], odom_ukf_->mean_[4], odom_ukf_->mean_[5], odom_ukf_->mean_[6]).normalized();
    }
    Eigen::Matrix4f odom_matrix() const {
        Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
        mat.block<3, 3>(0, 0) = odom_quat().toRotationMatrix();
        mat.block<3, 1>(0, 3) = odom_pos();
        return mat;
    }
    const std::optional<Eigen::Matrix4f>& wo_prediction_error() const {
        return wo_pred_error_;
    }
    const std::optional<Eigen::Matrix4f>& imu_prediction_error() const {
        return imu_pred_error_;
    }
    const std::optional<Eigen::Matrix4f>& odom_prediction_error() const {
        return odom_pred_error_;
    }

private:
    rclcpp::Time init_stamp_;             // when the estimator was initialized
    rclcpp::Time prev_stamp_;             // when the estimator was updated last time
    rclcpp::Time last_correct_stamp_;      // when the estimator performed the correct step
    double cool_time_duration_;

    std::shared_ptr<pcl::Registration<PointT, PointT>> registration_;

    std::unique_ptr<model::OdomSystemModel> odom_system_model_;

    Eigen::MatrixXf process_noise_;
    Eigen::MatrixXf measurement_noise_;
    std::unique_ptr<filter::UnscentedKalmanFilterX<float>> odom_ukf_; // TODO: delete

    Eigen::Matrix4f last_observation_;
    std::optional<Eigen::Matrix4f> wo_pred_error_;
    std::optional<Eigen::Matrix4f> imu_pred_error_;
    std::optional<Eigen::Matrix4f> odom_pred_error_;

    FilterType filter_type_;
    std::unique_ptr<model::UKFPoseSystemModel> ukf_system_model_;
    std::unique_ptr<model::EKFPoseSystemModel> ekf_system_model_;
    std::unique_ptr<filter::KalmanFilterX<float>> filter_;
};

} // namespace hdl_localization
} // namespace s3l

