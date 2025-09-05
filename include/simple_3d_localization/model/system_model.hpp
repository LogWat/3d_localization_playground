/**
 * @note システムモデルの抽象クラス
 */

#pragma once

#include <Eigen/Dense>
#include <simple_3d_localization/type.hpp>

namespace s3l::model
{

class SystemModel {
public:
    virtual ~SystemModel() = default;

    virtual VectorXt f(const VectorXt& state) const = 0;
    virtual VectorXt f(const VectorXt& state, const VectorXt& control) const = 0;
    virtual VectorXt h(const VectorXt& state) const = 0;

    void setDt(double dt) noexcept { dt_ = dt; }
protected:
    double dt_ = 0.1; // time step in seconds
};

} // namespace s3l::model

