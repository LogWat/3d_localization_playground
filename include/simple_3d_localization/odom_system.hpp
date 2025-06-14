#pragma once

#include <simple_3d_localization/ukf.hpp>
#include <simple_3d_localization/system_model.hpp>

namespace s3l {

/**
 * @brief This class models the sensor pose estimation based on robot odometry
 * @note state = [px py, pz, qw, qx, qy, qz]
 *       observation = [px, py, pz, qw, qx, qy, qz]
 *       指数写像を用いた方法を導入したほうが良いかもしれない
 */
class OdomSystem : public SystemModel {
public:
    VectorXt f(const VectorXt& state, const VectorXt& control) const override {
        Matrix4t pt = Matrix4t::Identity();
        pt.block<3, 1>(0, 3) = Vector3t(state(0), state(1), state(2));
        pt.block<3, 3>(0, 0) = Quaterniont(state(3), state(4), state(5), state(6)).normalized().toRotationMatrix();

        Matrix4t delta = Matrix4t::Identity();
        delta.block<3, 1>(0, 3) = Vector3t(control(0), control(1), control(2));
        delta.block<3, 3>(0, 0) = Quaterniont(control(3), control(4), control(5), control(6)).normalized().toRotationMatrix();

        Matrix4t pt_ = pt * delta;
        Quaterniont quat_(pt_.block<3, 3>(0, 0));

        VectorXt next_state(7);
        next_state.head<3>() = pt_.block<3, 1>(0, 3);
        next_state.tail<4>() = Vector4t(quat_.w(), quat_.x(), quat_.y(), quat_.z());

        return next_state;
    }

    // 未定義
    VectorXt f(const VectorXt& state) const override {
        return f(state, VectorXt::Zero(7));
    }

    // 観測モデル
    VectorXt h(const VectorXt& state) const override {
        return state;
    }
};


} // namespace s3l
