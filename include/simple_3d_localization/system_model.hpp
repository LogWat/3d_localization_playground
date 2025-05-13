/**
 * @note システムモデルの抽象クラス
 */

#pragma once

#include <Eigen/Dense>

namespace s3l {

class SystemModel {
public:
    using T = float;
    using VectorXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXt = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector3t = Eigen::Matrix<T, 3, 1>;
    using Vector4t = Eigen::Matrix<T, 4, 1>;
    using Matrix4t = Eigen::Matrix<T, 4, 4>;
    using Quaterniont = Eigen::Quaternion<T>;

    virtual ~SystemModel() = default;

    virtual VectorXt f(const VectorXt& state) const = 0;
    virtual VectorXt f(const VectorXt& state, const VectorXt& control) const = 0;
    virtual VectorXt h(const VectorXt& state) const = 0;
};

} // namespace s3l

