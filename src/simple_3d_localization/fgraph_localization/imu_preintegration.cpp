#include <simple_3d_localization/imu_preintegration.hpp>

namespace s3l {

IMUPreintegration::IMUPreintegration(const IMUIntegrationParams& params) {
    auto imu_params = gtsam::PreintegrationParams::MakeSharedU();
    if (!params.upright_) imu_params = gtsam::PreintegrationParams::MakeSharedD();

    imu_params->accelerometerCovariance = Eigen::Matrix3d::Identity() * std::pow(params.acc_noise_, 2);
    imu_params->gyroscopeCovariance = Eigen::Matrix3d::Identity() * std::pow(params.gyr_noise_, 2);
    imu_params->integrationCovariance = Eigen::Matrix3d::Identity() * std::pow(params.int_noise_, 2);
    imu_measurements_.reset(new gtsam::PreintegratedImuMeasurements(imu_params));
}

IMUPreintegration::~IMUPreintegration() {}

void IMUPreintegration::insert(double stamp, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
    Eigen::Matrix<double, 7, 1> imu;
    imu << stamp, acc, gyro;
    imu_queue_.push_back(imu);
}

int IMUPreintegration::integrate(double start_stamp, double end_stamp, const gtsam::imuBias::ConstantBias& bias, int* num_integrated) {
    *num_integrated = 0;
    if (imu_queue_.empty()) return 0;
    imu_measurements_->resetIntegrationAndSetBias(bias);

    int cursor = 0;
    auto imu_itr = imu_queue_.begin();
    double last_stamp = start_stamp;
    for (; imu_itr != imu_queue_.end(); imu_itr++, cursor++) {
        const auto& imu_frame = *imu_itr;
        const double imu_stamp = imu_frame(0);

        if (imu_stamp > end_stamp) break;

        const double dt = imu_stamp - last_stamp;
        if (dt <= 0.0) continue;

        const auto& acc = imu_frame.block<3, 1>(1, 0);
        const auto& gyro = imu_frame.block<3, 1>(4, 0);
        imu_measurements_->integrateMeasurement(acc, gyro, dt);
        last_stamp = imu_stamp;
        (*num_integrated)++;
    }

    const double dt = end_stamp - last_stamp;
    if (dt > 0.0) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = (imu_itr == imu_queue_.end()) ? *(imu_itr - 1) : *imu_itr;
        const auto& acc = last_imu_frame.block<3, 1>(1, 0);
        const auto& gyro = last_imu_frame.block<3, 1>(4, 0);
        imu_measurements_->integrateMeasurement(acc, gyro, dt);
    }

    return cursor;
}

int IMUPreintegration::integrate(double start_stamp, double end_stamp, const gtsam::NavState& state, const gtsam::imuBias::ConstantBias& bias,
                                 std::vector<double>& pred_times, std::vector<Eigen::Isometry3d>& pred_poses) {
    pred_times.emplace_back(start_stamp);
    pred_poses.emplace_back(Eigen::Isometry3d(state.pose().matrix()));

    if (imu_queue_.empty()) {
        pred_times.emplace_back(end_stamp);
        pred_poses.emplace_back(Eigen::Isometry3d(state.pose().matrix()));
        return 0;
    }

    imu_measurements_->resetIntegrationAndSetBias(bias);
    
    int cursor = 0;
    auto imu_itr = imu_queue_.begin();
    double last_stamp = start_stamp;

    for (; imu_itr != imu_queue_.end(); imu_itr++, cursor++) {
        const auto& imu_frame = *imu_itr;
        const double imu_stamp = imu_frame(0);

        if (imu_stamp > end_stamp) break;

        const double dt = imu_stamp - last_stamp;
        if (dt <= 0.0) continue;

        const auto& acc = imu_frame.block<3, 1>(1, 0);
        const auto& gyro = imu_frame.block<3, 1>(4, 0);
        imu_measurements_->integrateMeasurement(acc, gyro, dt);

        gtsam::NavState state_pred = imu_measurements_->predict(state, gtsam::imuBias::ConstantBias());
        pred_times.emplace_back(imu_stamp);
        pred_poses.emplace_back(Eigen::Isometry3d(state_pred.pose().matrix()));
        last_stamp = imu_stamp;
    }

    const double dt = end_stamp - last_stamp;
    if (dt > 0.0) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = (imu_itr == imu_queue_.end()) ? *(imu_itr - 1) : *imu_itr;
        const auto& acc = last_imu_frame.block<3, 1>(1, 0);
        const auto& gyro = last_imu_frame.block<3, 1>(4, 0);
        imu_measurements_->integrateMeasurement(acc, gyro, dt);
        gtsam::NavState state_pred = imu_measurements_->predict(state, gtsam::imuBias::ConstantBias());
        pred_times.emplace_back(end_stamp);
        pred_poses.emplace_back(Eigen::Isometry3d(state_pred.pose().matrix()));
    }

    return cursor;
}

int IMUPreintegration::find_imu_idx(double start_stamp, double end_stamp, std::vector<double>& delta_times,
                                    std::vector<Eigen::Matrix<double, 7, 1>>& imu_data) {
    if (imu_queue_.empty()) return 0;

    int cursor = 0;
    auto imu_itr = imu_queue_.begin();
    double last_stamp = start_stamp;

    for (; imu_itr != imu_queue_.end(); imu_itr++, cursor++) {
        const auto& imu_frame = *imu_itr;
        const double imu_stamp = imu_frame(0);

        if (imu_stamp > end_stamp) break;

        const double dt = imu_stamp - last_stamp;
        if (dt <= 0.0) continue;

        delta_times.emplace_back(dt);
        imu_data.push_back(imu_frame);
        last_stamp = imu_stamp;
    }

    const double dt = end_stamp - last_stamp;
    if (dt > 0.0) {
        Eigen::Matrix<double, 7, 1> last_imu_frame = (imu_itr == imu_queue_.end()) ? *(imu_itr - 1) : *imu_itr;
        delta_times.emplace_back(dt);
        imu_data.push_back(last_imu_frame);
    }

    return cursor;
}

void IMUPreintegration::erase_imu_data(int last) {
    imu_queue_.erase(imu_queue_.begin(), imu_queue_.begin() + last);
}

} // namespace s3l