#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace s3l::filter 
{

template <typename T, class SystemModel>
class KalmanFilter {
protected:
    using VectorXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXt = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    const int state_dim_;
    const SystemModel& system_model_;
    VectorXt X_; // State vector
    MatrixXt P_; // State covariance matrix
    MatrixXt Q_; // Process noise covariance matrix

public:
    virtual ~KalmanFilter() = default;
};

} // namespace s3l::filter
