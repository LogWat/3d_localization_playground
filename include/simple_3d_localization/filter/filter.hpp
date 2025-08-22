#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <simple_3d_localization/model/system_model.hpp>

namespace s3l::filter 
{

template <typename T>
class KalmanFilterX {
protected:
    using VectorXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXt = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

public:
    virtual ~KalmanFilterX() = default;

    virtual void setDt(double dt) = 0;
    virtual void setProcessNoise(const MatrixXt& Q) = 0;
    virtual void setMeasurementNoise(const MatrixXt& R) = 0;

    virtual void predict() = 0;
    virtual void predict(const VectorXt& control) = 0;
    virtual void correct(const VectorXt& measurement) = 0;

    // 共通取得
    virtual const VectorXt& getState() const = 0;
    virtual const MatrixXt& getCovariance() const = 0;
};

} // namespace s3l::filter
