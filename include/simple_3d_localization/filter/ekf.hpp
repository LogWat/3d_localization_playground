#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace d2::tools::filter {

template <class SystemModel>
class ExtendedKalmanFilter {
public:
    explicit ExtendedKalmanFilter(
        const SystemModel& model,
        const int state_dim,
        const Eigen::VectorXd& initial_state,
        const Eigen::MatrixXd& initial_cov,
        const Eigen::MatrixXd& process_noise)
        : state_dim_(state_dim), model_(model), X_(initial_state), P_(initial_cov), Q_(process_noise) {}

    /**
     * @brief Predict the next state and update the state covariance
     * @param dt Time step for prediction
     */
    void predict(const double dt) {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6); // Clamp dt to avoid instability
        Eigen::VectorXd X_pred = model_.f(X_, dt_c);
        Eigen::MatrixXd F = model_.stateTransitionJacobian(X_, dt_c);
        Eigen::MatrixXd P_pred = F * P_ * F.transpose() + Q_ * dt_c;
        X_ = X_pred;
        P_ = P_pred;
    }

    /**
     * @brief Predict the next state with control input
     * @param dt Time step for prediction
     * @param control Control input vector
     */
    void predict(const double dt, const Eigen::VectorXd& control) {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6); // Clamp dt to avoid instability
        Eigen::VectorXd X_pred = model_.f(X_, control, dt_c);
        Eigen::MatrixXd F = model_.stateTransitionJacobian(X_, control, dt_c);
        Eigen::MatrixXd P_pred = F * P_ * F.transpose() + Q_ * dt_c;
        X_ = X_pred;
        P_ = P_pred;
    }

    /**
     * @brief Correct the state with a measurement
     * @param measurement The measurement vector
     * @param measurement_noise The measurement noise covariance matrix
     */
    void correct(const Eigen::VectorXd& measurement, const Eigen::MatrixXd& measurement_noise) {
        Eigen::VectorXd measurement_pred = model_.h(X_);
        Eigen::MatrixXd H = model_.measurementJacobian(X_);
        Eigen::VectorXd y = measurement - measurement_pred;
        Eigen::MatrixXd S = H * P_ * H.transpose() + measurement_noise;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse(); // Kalman gain
        X_ += K * y; // Correct state estimate
        // Josephson correct for covariance
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * measurement_noise * K.transpose();
    }

    /* Setter */
    void setProcessNoise(const Eigen::MatrixXd& process_noise) {
        Q_ = process_noise;
    }

    /* Getter */
    const Eigen::VectorXd& getState() const {
        return X_;
    }

private:
    const int state_dim_;
    SystemModel model_; // System model
    Eigen::VectorXd X_; // State vector
    Eigen::MatrixXd P_; // State covariance matrix
    Eigen::MatrixXd Q_; // Process noise covariance
};

} // namespace d2::tools::filter
