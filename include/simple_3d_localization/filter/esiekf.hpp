/**
 * [TODO] 他のフィルタと合わせてリファクタ　これだけ独立になってる…
 */

#pragma once

#include "simple_3d_localization/type.hpp"
#include "simple_3d_localization/model/eskf_system_model.hpp"

namespace s3l::filter {

template <typename StateType, typename ControlType, int ErrorStateDim>
class ESIEKF {
public:
    using ErrorVector = Eigen::Matrix<EigenType, ErrorStateDim, 1>;
    using ErrorMatrix = Eigen::Matrix<EigenType, ErrorStateDim, ErrorStateDim>;

    StateType state_; // nominal state

private:
    ESKFSystemModel<StateType, ControlType, ErrorStateDim> &model_;
    ErrorMatrix P_; // 誤差状態の共分散

    /* Parameter */
    const int max_iterations_ = 10; // 最大イテレーション回数   
    const double epsilon_ = 1e-6;   // 収束判定閾値

public:
    ESIEKF(ESKFSystemModel<StateType, ControlType, ErrorStateDim> &model,
           const StateType &initial_state,
           const ErrorMatrix &initial_cov)
        : model_(model), state_(initial_state), P_(initial_cov) {}

    void predict(const ControlType &control, double dt) {
        // nominal state predict (非線形)
        state_ = model_.f(state_, control, dt);

        // error state predict (線形)
        ErrorMatrix Fx, FiQi;
        model_.getErrorStateDynamics(state_, control, dt, Fx, FiQi);
        P_ = Fx * P_ * Fx.transpose() + FiQi;
    }

    bool correct(const VectorXt &measurement, const MatrixXt &measurement_noise) {
        StateType state_prior = state_; // 事前状態を保存
        ErrorMatrix P_prior = P_;

        for (int i = 0; i < max_iterations_; ++i) {
            // 現在の推定値に基づく観測予測とヤコビアン
            VectorXt measurement_pred = model_.h(state_);

            // 観測値と予測値を扱いやすい型に変換
            Vector3t p_meas = measurement.head(3); // 観測
            Quaterniont q_meas(measurement(6), measurement(3), measurement(4), measurement(5));
            Sophus::SO3<EigenType> R_meas(q_meas);

            Vector3t p_pred = measurement_pred.head(3); // 予測
            Quaterniont q_pred(measurement_pred(6), measurement_pred(3), measurement_pred(4), measurement_pred(5));
            Sophus::SO3<EigenType> R_pred(q_pred);

            // 6次元 innovation (残差) を算出
            Eigen::Matrix<EigenType, 6, 1> y;
            y.head<3>() = p_meas - p_pred;
            y.tail<3>() = (R_meas * R_pred.inverse()).log(); // SO(3) -> so(3)

            // IEKF MAP推定
            MatrixXt H = model_.getMeasurementJacobian(state_);
            MatrixXt S = H * P_prior * H.transpose() + measurement_noise;
            MatrixXt K = P_prior * H.transpose() * S.inverse();

            ErrorVector prior_diff = state_ - state_prior;
            ErrorVector delta_x = K * (y - H * prior_diff) - prior_diff;

            state_ = state_prior + delta_x; // boxplus

            if (delta_x.norm() < epsilon_) break;
        }

        // 共分散更新
        MatrixXt H_final = model_.getMeasurementJacobian(state_);
        MatrixXt K_final = P_prior * H_final.transpose() * (H_final * P_prior * H_final.transpose() + measurement_noise).inverse();
        ErrorMatrix I = ErrorMatrix::Identity();
        ErrorMatrix IKH = I - K_final * H_final;
        P_ = IKH * P_prior * IKH.transpose() + K_final * measurement_noise * K_final.transpose(); // Joseph form

        return true;
    }
};

} // namespace s3l::filter
