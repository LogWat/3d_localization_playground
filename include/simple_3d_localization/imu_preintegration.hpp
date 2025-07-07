/**
 * @file imu_preintegration.hpp
 * @brief Header file for IMU preintegration functionality
 * @brief reference: https://github.com/koide3/glim/blob/master/include/glim/common/imu_integration.hpp
 */

#pragma once

#include <deque>
#include <vector>
#include <Eigen/Core>
#include <gtsam/navigation/ImuFactor.h>

namespace s3l {

struct IMUIntegrationParams {
    IMUIntegrationParams(const bool upright = true, 
                         const double acc_noise = 1e-3, 
                         const double gyr_noise = 1e-3, 
                         const double int_noise = 1e-5)
        : upright_(upright), acc_noise_(acc_noise), gyr_noise_(gyr_noise), int_noise_(int_noise) {}
    ~IMUIntegrationParams() {}

    bool upright_;  // If true, the IMU is assumed to be upright
    double acc_noise_;
    double gyr_noise_;
    double int_noise_;
};

class IMUPreintegration {
public:
    IMUPreintegration(const IMUIntegrationParams& params = IMUIntegrationParams());
    ~IMUPreintegration();

    void insert(double stamp, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro);
    int integrate(double start_stamp, double end_stamp, const gtsam::imuBias::ConstantBias& bias, int* num_integrated);
    int integrate(double start_stamp, double end_stamp, const gtsam::NavState& state, const gtsam::imuBias::ConstantBias& bias, 
                  std::vector<double>& pred_times, std::vector<Eigen::Isometry3d>& pred_poses);
    int find_imu_idx(double start_stamp, double end_stamp, std::vector<double>& delta_times,
                     std::vector<Eigen::Matrix<double, 7, 1>>& imu_data);
    void erase_imu_data(int last);

    const gtsam::PreintegratedImuMeasurements& integrated_measurements() const {
        return *imu_measurements_;
    }
    const std::deque<Eigen::Matrix<double, 7, 1>>& imu_queue() const {
        return imu_queue_;
    }
private:
    std::shared_ptr<gtsam::PreintegratedImuMeasurements> imu_measurements_;
    std::deque<Eigen::Matrix<double, 7, 1>> imu_queue_;
};
}