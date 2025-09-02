/**
 * Unscented Kalman Filter (UKF) class
 * @ref https://github.com/koide3/hdl_localization/blob/master/include/kkl/alg/unscented_kalman_filter.hpp
 * @author LogWat
 * @date 2025/05/13
 */

#pragma once

#include <random>
#include <Eigen/Dense>
#include <simple_3d_localization/filter/filter.hpp>
#include <simple_3d_localization/model/system_model.hpp>

namespace s3l::filter
{

template <typename T>
class UnscentedKalmanFilterX : public KalmanFilterX<T> {
    using VectorXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXt = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

public:
    UnscentedKalmanFilterX(
        model::SystemModel& model,
        int state_dim,
        int input_dim,
        int measurement_dim,
        const MatrixXt& process_noise,
        const MatrixXt& measurement_noise,
        const VectorXt& mean,
        const MatrixXt& cov
    ) : state_dim_(state_dim),
        input_dim_(input_dim),
        measurement_dim_(measurement_dim),
        N_(state_dim),
        M_(input_dim),
        K_(measurement_dim),
        S_(2 * N_ + 1),
        mean_(mean),
        cov_(cov),
        system_model_(model),
        process_noise_(process_noise),
        measurement_noise_(measurement_noise),
        lambda_(1),
        normal_dist(0.0, 1.0)
    {
        weights_.resize(S_, 1);
        sigma_points_.resize(S_, N_);
        ext_weights_.resize(2 * (N_ + K_) + 1, 1);
        ext_sigma_points_.resize(2 * (N_ + K_) + 1, N_ + K_);
        expected_measurements_.resize(2 * (N_ + K_) + 1, K_);

        // Initialize weights
        int i = 1;
        for (weights_(0) = lambda_ / (N_ + lambda_); i < 2 * N_ + 1; ++i) {
            weights_(i) = 1.0 / (2 * (N_ + lambda_));
        }
        // 拡張状態空間の重みを計算
        i = 1;
        for (ext_weights_(0) = lambda_ / (N_ + K_ + lambda_); i < 2 * (N_ + K_) + 1; ++i) {
            ext_weights_(i) = 1.0 / (2 * (N_ + K_ + lambda_));
        }
    }

    /**
     * @brief Predict the state and covariance of the system
     */
    void predict() override {
        // 共分散行列が正定値であることを確認
        ensurePositiveFinite(cov_);
        // シグマ点を計算
        computeSigmaPoints(mean_, cov_, sigma_points_);
        // 各シグマ点に対して、状態遷移関数を適用
        for (int i = 0; i < S_; i++) {
            sigma_points_.row(i) = system_model_.f(sigma_points_.row(i));
        }

        const auto& R = process_noise_;

        // Unscented transformation
        VectorXt mean_pred(mean_.size());
        MatrixXt cov_pred(cov_.rows(), cov_.cols());
        mean_pred.setZero();
        cov_pred.setZero();
        for (int i = 0; i < S_; i++) {
            mean_pred += weights_(i) * sigma_points_.row(i);
        }
        for (int i = 0; i < S_; i++) {
            VectorXt diff = sigma_points_.row(i).transpose() - mean_pred;
            cov_pred += weights_(i) * diff * diff.transpose();
        }

        mean_ = mean_pred;
        cov_ = cov_pred + R;
    }

    /**
     * @brief Predict the measurement and covariance of the system
     * @param control input vector
     */
    void predict(const VectorXt& control) override {
        ensurePositiveFinite(cov_);
        computeSigmaPoints(mean_, cov_, sigma_points_);
        for (int i = 0; i < S_; i++) {
            sigma_points_.row(i) = system_model_.f(sigma_points_.row(i), control); // それぞれのシグマ点に制御入力を適用
        }

        const auto& R = process_noise_;

        // Unscented transformation
        VectorXt mean_pred(mean_.size());
        MatrixXt cov_pred(cov_.rows(), cov_.cols());
        mean_pred.setZero();
        cov_pred.setZero();
        for (int i = 0; i < S_; i++) {
            mean_pred += weights_(i) * sigma_points_.row(i);
        }
        for (int i = 0; i < S_; i++) {
            VectorXt diff = sigma_points_.row(i).transpose() - mean_pred;
            cov_pred += weights_(i) * diff * diff.transpose();
        }

        mean_ = mean_pred;
        cov_ = cov_pred + R;
    }


    /**
     * @brief Correct the state and covariance of the system
     * @param measurement measurement vector
     */
    void correct(const VectorXt& measurement) override {
        // error variancesを含む拡張状態空間を考慮
        VectorXt ext_mean_pred = VectorXt::Zero(N_ + K_, 1);
        MatrixXt ext_cov_pred = MatrixXt::Zero(N_ + K_, N_ + K_);
        ext_mean_pred.topLeftCorner(N_, 1) = VectorXt(mean_);
        ext_cov_pred.topLeftCorner(N_, N_) = MatrixXt(cov_);
        ext_cov_pred.bottomRightCorner(K_, K_) = measurement_noise_;

        ensurePositiveFinite(ext_cov_pred);
        computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points_);

        // 各シグマ点に対して、観測関数を適用 (事前分布を観測空間に変換)
        expected_measurements_.setZero();
        for (int i = 0; i < ext_sigma_points_.rows(); i++) {
            expected_measurements_.row(i) = system_model_.h(ext_sigma_points_.row(i).transpose().topLeftCorner(N_, 1)); 
            expected_measurements_.row(i) += VectorXt(ext_sigma_points_.row(i).transpose().bottomRightCorner(K_, 1));
        }
        // Unscented transformation
        VectorXt expected_measurement_mean = VectorXt::Zero(K_);
        for (int i = 0; i < ext_sigma_points_.rows(); i++) {
            expected_measurement_mean += ext_weights_(i) * expected_measurements_.row(i);
        }
        MatrixXt expected_measurement_cov = MatrixXt::Zero(K_, K_);
        for (int i = 0; i < ext_sigma_points_.rows(); i++) {
            VectorXt diff = expected_measurements_.row(i).transpose() - expected_measurement_mean;
            expected_measurement_cov += ext_weights_(i) * diff * diff.transpose();
        }

        // 相互共分散
        MatrixXt sigma = MatrixXt::Zero(N_ + K_, K_);
        for (int i = 0; i < ext_sigma_points_.rows(); i++) {
            auto diffA = (ext_sigma_points_.row(i).transpose() - ext_mean_pred);
            auto diffB = (expected_measurements_.row(i).transpose() - expected_measurement_mean);
            sigma += ext_weights_(i) * (diffA * diffB.transpose());
        }

        kalman_gain_ = sigma * expected_measurement_cov.inverse();
        const auto& K = kalman_gain_;

        VectorXt ext_mean = ext_mean_pred + K * (measurement - expected_measurement_mean);
        MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();

        // 拡張状態空間から元の状態空間に戻す
        mean_ = ext_mean.topLeftCorner(N_, 1);
        cov_ = ext_cov.topLeftCorner(N_, N_);
    }

    // KalmanFilterX<T> overrides
    // 追加のインターフェイス実装
    void setDt(double dt) override { system_model_.setDt(dt); }
    void setMean(const VectorXt& mean) override { setX(mean); }
    void setProcessNoise(const MatrixXt& q) override { setProcessNoiseCov(q); }
    void setMeasurementNoise(const MatrixXt& r) override { setMeasurementNoiseCov(r); }

    const VectorXt& getState() const override { return getMean(); }
    const MatrixXt& getCovariance() const override { return getCov(); }

    
    /*              getter              */
    const VectorXt& getMean() const { return mean_; }
    const MatrixXt& getCov() const { return cov_; }
    const MatrixXt& getSigmaPoints() const { return sigma_points_; }
    const model::SystemModel& getSystemModel() { return system_model_; }
    const MatrixXt& getProcessNoiseCov() const { return process_noise_; }
    const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise_; }
    const MatrixXt& getKalmanGain() const { return kalman_gain_; }

    /*              setter              */
    UnscentedKalmanFilterX& setX(const VectorXt& m) { mean_ = m; return *this; }
    UnscentedKalmanFilterX& setCov(const MatrixXt& c) { cov_ = c; return *this; }
    UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& q) { process_noise_ = q; return *this; }
    UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& r) { measurement_noise_ = r; return *this; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // For Eigen's dynamic size matrices
private:
    /**
     * @brief compute sigma points
     * @param mean Mean vector
     * @param cov Covariance matrix
     * @param sigma_points calculated sigma points
     */
    void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
        const int n = mean.size();
        assert(cov.rows() == n && cov.cols() == n);

        // 共分散行列のコレスキー分解を計算
        Eigen::LLT<MatrixXt> llt;
        llt.compute((n + lambda_) * cov);
        MatrixXt l = llt.matrixL();

        sigma_points.row(0) = mean;
        for (int i = 0; i < n; ++i) {
            sigma_points.row(1 + i * 2) = mean + l.col(i);
            sigma_points.row(2 + i * 2) = mean - l.col(i);
        }
    }

    /**
     * @brief Make covariance matrix positive definite
     * @param cov Covariance matrix
     */
    void ensurePositiveFinite(MatrixXt& cov) {
        const double epsilon = 1e-9;

        Eigen::EigenSolver<MatrixXt> solver(cov);
        MatrixXt D = solver.pseudoEigenvalueMatrix();
        MatrixXt V = solver.pseudoEigenvectors();
        for (int i = 0; i < D.rows(); i++) {
            if (D(i, i) < epsilon) D(i, i) = epsilon;
        }
        cov = V * D * V.inverse();
    }


    const int state_dim_;
    const int input_dim_;
    const int measurement_dim_;

    const int N_, M_, K_, S_;

public:
    VectorXt mean_;
    MatrixXt cov_;

    model::SystemModel& system_model_;
    MatrixXt process_noise_;
    MatrixXt measurement_noise_;

    T lambda_;
    VectorXt weights_;

    MatrixXt sigma_points_;

    VectorXt ext_weights_;
    MatrixXt ext_sigma_points_;
    MatrixXt expected_measurements_;

    MatrixXt kalman_gain_;

    std::mt19937 mt;                            // Random number generator
    std::normal_distribution<T> normal_dist;    // Normal distribution for generating random numbers
};

} // namespace s3l::filter

