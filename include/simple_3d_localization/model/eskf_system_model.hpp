#pragma once
#include <Eigen/Core>
#include "simple_3d_localization/type.hpp"

namespace s3l::model
{
template<typename StateType, typename ControlType, int ErrorStateDim>
class ESKFSystemModel
{
public:
    using ErrorVector = Eigen::Matrix<EigenType, ErrorStateDim, 1>;
    using ErrorMatrix = Eigen::Matrix<EigenType, ErrorStateDim, ErrorStateDim>;

    virtual ~ESKFSystemModel() = default;
    
    // nominal predict (非線形)
    virtual StateType f(const StateType& state, const ControlType& control, double dt) const = 0;

    // 誤差状態の状態遷移行列Fとノイズ行列Fi, Qi
    virtual void getErrorStateDynamics(const StateType& state, const ControlType& control, double dt,
                                        ErrorMatrix& Fx, ErrorMatrix& FiQi) const = 0;

    // 観測予測(非線形)
    virtual VectorXt h(const StateType& state) const = 0;

    // 観測ヤコビアン
    virtual MatrixXt measurementJacobian(const StateType& state) const = 0;

    // 観測ベクトル次元
    virtual int measurementDimension() const = 0;
};

} // namespace s3l::model
