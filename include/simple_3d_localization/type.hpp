#pragma once

#include <Eigen/Core>

enum FilterType {
    UKF,
    EKF
};

struct ImuData {
    double timestamp;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d linear_acceleration;
};

struct ImuInitializationResult {
    bool success = false;
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity_vec = Eigen::Vector3d(0, 0, -9.80665);
};
