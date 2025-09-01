/**
 * @file ieskf_system_model.hpp
 * @brief Error State Iterated Extended Kalman Filter (ES-IEKF) system model for 3D pose estimation
 * @date 2025-09-01
 */

#pragma once

#include "simple_3d_localization/type.hpp"
#include "simple_3d_localization/model/eskf_system_model.hpp"
#include "simple_3d_localization/model/pose_state.hpp"

namespace s3l::model {

using ImuControl = Eigen::Matrix<EigenType, 6, 1>; // [acc(3), gyro(3)]

class ESIEKFPoseSystemModel : public ESKFSystemModel<PoseState, ImuControl, POSE_ERROR_STATE_DIM>
{
public:
    // continuous white noise densities (discrete Qiは下でdtでスケール) (262~265)
    double na_ = 0.02;  // accel noise [m/s^2/sqrt(Hz)]
    double nw_ = 0.002; // gyro noise  [rad/s/sqrt(Hz)]
    double nba_ = 0.0002; // accel bias RW [m/s^2/sqrt(Hz)]
    double nbw_ = 0.00002;// gyro bias RW [rad/s/sqrt(Hz)]
    double ng_ = 0.0;    // gravity RW if estimating g

    // Nominal propagation (5.4.1) (260a ~ 260f)
    PoseState f(const PoseState& state, const ImuControl& control, double dt) const override {
        PoseState next = state;
        Vector3t acc = control.head<3>();
        Vector3t gyro = control.tail<3>();
        
        Vector3t am = acc - state.ba;
        Vector3t wm = gyro - state.bg;

        Vector3t a_world = state.R * am + state.g;
        next.p += state.v * dt + 0.5 * a_world * dt * dt;
        next.v += a_world * dt;
        next.R = state.R * Sophus::SO3<EigenType>::exp(wm * dt);

        return next;
    }

    // 誤差ダイナミクス (5.4.3)
    void getErrorStateDynamics(const PoseState& state, const ImuControl& control, double dt,
                                ErrorMatrix& Fx, ErrorMatrix& FiQi) const override {
        Vector3t acc = control.head<3>();
        Vector3t gyro = control.tail<3>();
        Vector3t am = acc - state.ba;
        Vector3t wm = gyro - state.bg;

        // Fx (270)
        Fx = ErrorMatrix::Identity();
        Fx.block<3, 3>(0, 3) = Matrix3t::Identity() * dt;
        Fx.block<3, 3>(3, 6) = -state.R.matrix() * Sophus::SO3<EigenType>::hat(am) * dt;
        Fx.block<3, 3>(3, 9) = -state.R.matrix() * dt;
        Fx.block<3, 3>(3, 15) = Matrix3t::Identity() * dt;
        Fx.block<3, 3>(6, 6) = Sophus::SO3<EigenType>::exp(-wm * dt).matrix(); // [?]
        Fx.block<3, 3>(6, 12) = -Matrix3t::Identity() * dt;

        // Fi * Qi * Fi^T (271)
        Eigen::Matrix<EigenType, 18, 12> Fi = Eigen::Matrix<EigenType, 18, 12>::Zero();
        Fi.block<3, 3>(3, 0)  = Matrix3t::Identity(); // acc noise to dv
        Fi.block<3, 3>(6, 3)  = Matrix3t::Identity(); // gyro noise to dtheta
        Fi.block<3, 3>(9, 6)  = Matrix3t::Identity(); // accel bias RW
        Fi.block<3, 3>(12, 9) = Matrix3t::Identity(); // gyro bias RW

        Eigen::Matrix<EigenType, 12, 12> Qi = Eigen::Matrix<EigenType, 12, 12>::Zero();
        Qi.block<3, 3>(0, 0) = Matrix3t::Identity() * na_ * na_ * dt;
        Qi.block<3, 3>(3, 3) = Matrix3t::Identity() * nw_ * nw_ * dt;
        Qi.block<3, 3>(6, 6) = Matrix3t::Identity() * nba_ * nba_ * dt;
        Qi.block<3, 3>(9, 9) = Matrix3t::Identity() * nbw_ * nbw_ * dt;

        FiQi = Fi * Qi * Fi.transpose();
    }

    // 観測
    VectorXt h(const PoseState& state) const override {
        VectorXt measurement(7);
        measurement.head<3>() = state.p;
        Quaterniont q(state.R.matrix());
        measurement.tail<4>() = q.coeffs();
        return measurement;
    }

    // 観測ヤコビアン
    MatrixXt measurementJacobian(const PoseState& state) const override {
        MatrixXt H = MatrixXt::Zero(measurementDimension(), POSE_ERROR_STATE_DIM);
        H.block<3, 3>(0, 0) = Matrix3t::Identity();
        H.block<4, 4>(3, 6) = MatrixXt::Identity(4, 4);
        return H;
    }

    int measurementDimension() const override {
        return 7; // position(3) + orientation(4)
    }
};

} // namespace s3l::model
