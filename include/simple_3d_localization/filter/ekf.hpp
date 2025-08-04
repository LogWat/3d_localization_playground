#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace s3l::filter {

template <class SystemModel>
class ExtendedKalmanFilter {
public:
    explicit ExtendedKalmanFilter(
        const SystemModel& model,
        const int state_dim,
        const Eigen::VectorXf& initial_state,
        const Eigen::MatrixXf& initial_cov,
        const Eigen::MatrixXf& process_noise)
        : state_dim_(state_dim), model_(model), X_(initial_state), P_(initial_cov), Q_(process_noise) {}

    /**
     * @brief Predict the next state and update the state covariance
     * @param dt Time step for prediction
     */
    void predict(const double dt) {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6); // Clamp dt to avoid instability
        Eigen::VectorXf X_pred = model_.f(X_, dt_c);
        Eigen::MatrixXf F = model_.stateTransitionJacobian(X_, dt_c);
        Eigen::MatrixXf P_pred = F * P_ * F.transpose() + Q_ * dt_c;
        X_ = X_pred;
        P_ = P_pred;
    }

    /**
     * @brief Predict the next state with control input
     * @param dt Time step for prediction
     * @param control Control input vector
     */
    void predict(const double dt, const Eigen::VectorXf& control) {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6); // Clamp dt to avoid instability
        Eigen::VectorXf X_pred = model_.f(X_, control, dt_c);
        Eigen::MatrixXf F = model_.stateTransitionJacobian(X_, control, dt_c);
        Eigen::MatrixXf P_pred = F * P_ * F.transpose() + Q_ * dt_c;
        X_ = X_pred;
        P_ = P_pred;
    }

    /**
     * @brief Correct the state with a measurement
     * @param measurement The measurement vector
     * @param measurement_noise The measurement noise covariance matrix
     */
    void correct(const Eigen::VectorXf& measurement, const Eigen::MatrixXf& measurement_noise) {
        Eigen::VectorXf measurement_pred = model_.h(X_);
        Eigen::MatrixXf H = model_.measurementJacobian(X_);
        Eigen::VectorXf y = measurement - measurement_pred;
        Eigen::MatrixXf S = H * P_ * H.transpose() + measurement_noise;
        Eigen::MatrixXf K = P_ * H.transpose() * S.inverse(); // Kalman gain
        X_ += K * y; // Correct state estimate
        // Josephson correct for covariance
        Eigen::MatrixXf I = Eigen::MatrixXf::Identity(state_dim_, state_dim_);
        P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * measurement_noise * K.transpose();
    }

    /* Setter */
    void setProcessNoise(const Eigen::MatrixXf& process_noise) {
        Q_ = process_noise;
    }

    /* Getter */
    const Eigen::VectorXf& getState() const {
        return X_;
    }

private:
    const int state_dim_;
    SystemModel model_; // System model
    Eigen::VectorXf X_; // State vector
    Eigen::MatrixXf P_; // State covariance matrix
    Eigen::MatrixXf Q_; // Process noise covariance
};

} // namespace s3l::filter
