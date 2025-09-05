#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <simple_3d_localization/type.hpp>

namespace s3l::filter 
{

class KalmanFilterX {
public:
    virtual ~KalmanFilterX() = default;

    virtual void setDt(double dt) = 0;
    virtual void setMean(const VectorXt& mean) = 0;
    virtual void setProcessNoise(const MatrixXt& Q) = 0;
    virtual void setMeasurementNoise(const MatrixXt& R) = 0;

    virtual void predict(const VectorXt& control) = 0;
    virtual void correct(const VectorXt& measurement) = 0;

    // 共通取得
    [[nodiscard]] virtual const VectorXt& getState() const = 0;
    [[nodiscard]] virtual const MatrixXt& getCovariance() const = 0;
};

} // namespace s3l::filter
