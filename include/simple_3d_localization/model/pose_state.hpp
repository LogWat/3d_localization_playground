#pragma once

#include <sophus/so3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

// 誤差状態ベクトルの次元
constexpr int POSE_ERROR_STATE_DIM = 18; // δp(3), δv(3), δθ(3), δba(3), δbg(3), δg(3)
constexpr int POSE_NOMINAL_STATE_DIM = 19; // p(3), v(3), q(4), ba(3), bg(3), g(3)
using PoseErrorVector = Eigen::Matrix<double, POSE_ERROR_STATE_DIM, 1>;
using PoseNominalVector = Eigen::Matrix<double, POSE_NOMINAL_STATE_DIM, 1>;

// 公称状態を保持する構造体
struct PoseState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using T = double;

    Sophus::SO3<T> R;         // body -> world
    Eigen::Matrix<T, 3, 1> p = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 1> v = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 1> ba = Eigen::Matrix<T, 3, 1>::Zero(); // accel bias
    Eigen::Matrix<T, 3, 1> bg = Eigen::Matrix<T, 3, 1>::Zero(); // gyro bias
    Eigen::Matrix<T, 3, 1> g{0.0, 0.0, -9.80665};

    // boxplus(⊕): 公称状態に誤差状態を適用して補正する (Injection)
    // 論文 (314) に相当
    PoseState operator+(const PoseErrorVector& delta) const {
        PoseState result;
        result.p = this->p + delta.segment<3>(0);
        result.v = this->v + delta.segment<3>(3);
        // 世界座標系での誤差注入 (left-multiplication)
        result.R = Sophus::SO3<T>::exp(delta.segment<3>(6)) * this->R;
        result.ba = this->ba + delta.segment<3>(9);
        result.bg = this->bg + delta.segment<3>(12);
        result.g = this->g + delta.segment<3>(15);
        return result;
    }

    // boxminus(⊖): 2つの状態間の誤差を計算する
    PoseErrorVector operator-(const PoseState& other) const {
        PoseErrorVector delta;
        delta.segment<3>(0) = this->p - other.p;
        delta.segment<3>(3) = this->v - other.v;
        delta.segment<3>(6) = (other.R.inverse() * this->R).log();
        delta.segment<3>(9) = this->ba - other.ba;
        delta.segment<3>(12) = this->bg - other.bg;
        delta.segment<3>(15) = this->g - other.g;
        return delta;
    }
};
