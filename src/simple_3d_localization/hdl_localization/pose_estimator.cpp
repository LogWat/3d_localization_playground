#include <pcl/filters/voxel_grid.h>
#include <simple_3d_localization/hdl_localization/pose_system.hpp>
#include <simple_3d_localization/filter/ukf.hpp>
#include <simple_3d_localization/hdl_localization/pose_estimator.hpp>
#include <simple_3d_localization/hdl_localization/odom_system.hpp>

namespace s3l::hdl_localization
{

/**
 * @brief constructor
 * @param registration        registration method
 * @param pos                 initial position
 * @param quat                initial orientation
 * @param cool_time_duration  during "cool time", prediction is not performed
 */
PoseEstimator::PoseEstimator(
    std::shared_ptr<pcl::Registration<PointT, PointT>>& registration,
    const Eigen::Vector3f& pos,
    const Eigen::Quaternionf& quat,
    double cool_time_duration
): 
    cool_time_duration_(cool_time_duration),
    registration_(registration)
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
    process_noise_.middleRows(10, 3) *= 1e-3;
    process_noise_.middleRows(13, 3) *= 1e-5;

    // 位置(3) + 姿勢(4) = 7
    Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
    measurement_noise.middleRows(0, 3) *= 0.01;
    measurement_noise.middleRows(3, 4) *= 0.001;

    // 初期状態
    Eigen::VectorXf mean(16);
    mean.middleRows(0, 3) = pos;
    mean.middleRows(3, 3).setZero();
    mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z()).normalized();
    mean.middleRows(10, 3).setZero();
    mean.middleRows(13, 3).setZero();

    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

    pose_system_model_ = std::make_unique<PoseSystem>();
    ukf_.reset(new filter::UnscentedKalmanFilterX<float>(
        *pose_system_model_,
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
    if ((stamp - init_stamp_).seconds() < cool_time_duration_ || prev_stamp_ == rclcpp::Time() || (stamp - prev_stamp_).seconds() < 0.01) {
        prev_stamp_ = stamp;
        return;
    }

    double dt = (stamp - prev_stamp_).seconds();
    prev_stamp_ = stamp;

    ukf_->setProcessNoiseCov(process_noise_ * dt);
    pose_system_model_->setDt(dt);
    ukf_->predict();
}


/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
    if (init_stamp_ == rclcpp::Time()) {
        init_stamp_ = stamp;
    }
    if ((stamp - init_stamp_).seconds() < cool_time_duration_ || prev_stamp_ == rclcpp::Time() || (stamp - prev_stamp_).seconds() < 0.01) {
        prev_stamp_ = stamp;
        return;
    }

    double dt = (stamp - prev_stamp_).seconds();
    prev_stamp_ = stamp;

    ukf_->setProcessNoiseCov(process_noise_ * dt);
    pose_system_model_->setDt(dt);
    Eigen::VectorXf control(6);
    control.head<3>() = acc; // acceleration
    control.tail<3>() = gyro; // angular velocity
    ukf_->predict(control);
}


/**
 * @brief update the state of the odomety-based pose estimation
 */
void PoseEstimator::predict_odom(const Eigen::Matrix4f& odom_delta) {
    if(!odom_ukf_) {
        // odom_ukf_ が初期化されていない場合は、初期化する
        Eigen::MatrixXf odom_process_noise = Eigen::MatrixXf::Identity(7, 7);
        Eigen::MatrixXf odom_measurement_noise = Eigen::MatrixXf::Identity(7, 7) * 1e-3;

        Eigen::VectorXf odom_mean(7);
        odom_mean.block<3, 1>(0, 0) = Eigen::Vector3f(ukf_->mean_[0], ukf_->mean_[1], ukf_->mean_[2]);
        odom_mean.block<4, 1>(3, 0) = Eigen::Vector4f(ukf_->mean_[6], ukf_->mean_[7], ukf_->mean_[8], ukf_->mean_[9]).normalized();
        Eigen::MatrixXf odom_cov = Eigen::MatrixXf::Identity(7, 7) * 1e-2;

        odom_system_model_ = std::make_unique<OdomSystem>();
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
pcl::PointCloud<PoseEstimator::PointT>::Ptr PoseEstimator::correct(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
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
        imu_mean.block<3, 1>(0, 0) = ukf_->mean_.block<3, 1>(0, 0);
        imu_mean.block<4, 1>(3, 0) = ukf_->mean_.block<4, 1>(6, 0).normalized();
        imu_cov.block<3, 3>(0, 0) = ukf_->cov_.block<3, 3>(0, 0);
        imu_cov.block<3, 4>(0, 3) = ukf_->cov_.block<3, 4>(0, 6);
        imu_cov.block<4, 3>(3, 0) = ukf_->cov_.block<4, 3>(6, 0);
        imu_cov.block<4, 4>(3, 3) = ukf_->cov_.block<4, 4>(6, 6);

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

    ukf_->correct(observation);
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
rclcpp::Time PoseEstimator::last_correct_time() const {
    return last_correct_stamp_;
}
Eigen::Vector3f PoseEstimator::pos() const {
    return Eigen::Vector3f(ukf_->mean_[0], ukf_->mean_[1], ukf_->mean_[2]);
}
Eigen::Vector3f PoseEstimator::vel() const {
    return Eigen::Vector3f(ukf_->mean_[3], ukf_->mean_[4], ukf_->mean_[5]);
}
Eigen::Quaternionf PoseEstimator::quat() const {
    return Eigen::Quaternionf(ukf_->mean_[6], ukf_->mean_[7], ukf_->mean_[8], ukf_->mean_[9]).normalized();
}
Eigen::Matrix4f PoseEstimator::matrix() const {
    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    mat.block<3, 3>(0, 0) = quat().toRotationMatrix();
    mat.block<3, 1>(0, 3) = pos();
    return mat;
}
Eigen::Vector3f PoseEstimator::odom_pos() const {
    return Eigen::Vector3f(odom_ukf_->mean_[0], odom_ukf_->mean_[1], odom_ukf_->mean_[2]);
}
Eigen::Quaternionf PoseEstimator::odom_quat() const {
    return Eigen::Quaternionf(odom_ukf_->mean_[3], odom_ukf_->mean_[4], odom_ukf_->mean_[5], odom_ukf_->mean_[6]).normalized();
}
Eigen::Matrix4f PoseEstimator::odom_matrix() const {
    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    mat.block<3, 3>(0, 0) = odom_quat().toRotationMatrix();
    mat.block<3, 1>(0, 3) = odom_pos();
    return mat;
}
const std::optional<Eigen::Matrix4f>& PoseEstimator::wo_prediction_error() const {
    return wo_pred_error_;
}
const std::optional<Eigen::Matrix4f>& PoseEstimator::imu_prediction_error() const {
    return imu_pred_error_;
}
const std::optional<Eigen::Matrix4f>& PoseEstimator::odom_prediction_error() const {
    return odom_pred_error_;
}


} // namespace s3l::hdl_localization
