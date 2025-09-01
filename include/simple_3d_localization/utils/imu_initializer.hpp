#include <mutex>
#include <memory>
#include <iostream>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <simple_3d_localization/type.hpp>

namespace s3l::utils
{
/* 重力ベクトルを球面上に成約するためのManifold */
class SphereManifold : public ceres::Manifold {
public:
    explicit SphereManifold(double radius) : radius_(radius) {}

    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        Eigen::Map<const Eigen::Vector3d> x_vec(x);
        Eigen::Map<const Eigen::Vector2d> delta_vec(delta);

        // 接空間の基底を計算
        Eigen::Vector3d b1, b2;
        Eigen::Vector3d x_normalized = x_vec.normalized();
        if (std::abs(x_normalized.z()) > 0.5) {
            b1 = Eigen::Vector3d(1.0, 0.0, -x_normalized.x() / x_normalized.z()).normalized();
        } else {
            b1 = Eigen::Vector3d(-x_normalized.y() / x_normalized.x(), 1.0, 0.0).normalized();
        }
        b2 = x_normalized.cross(b1);

        // 接空間上の移動を3D空間に適用し、球面上に射影
        Eigen::Map<Eigen::Vector3d> x_plus_delta_vec(x_plus_delta);
        x_plus_delta_vec = x_vec + (b1 * delta_vec.x() + b2 * delta_vec.y());
        x_plus_delta_vec = x_plus_delta_vec.normalized() * radius_;

        return true;
    }

    bool PlusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<const Eigen::Vector3d> x_vec(x);
        Eigen::Vector3d x_normalized = x_vec.normalized();

        // 接空間の基底を計算
        Eigen::Vector3d b1, b2;
        if (std::abs(x_normalized.z()) > 0.5) {
            b1 = Eigen::Vector3d(1.0, 0.0, -x_normalized.x() / x_normalized.z()).normalized();
        } else {
            b1 = Eigen::Vector3d(-x_normalized.y() / x_normalized.x(), 1.0, 0.0).normalized();
        }
        b2 = x_normalized.cross(b1);

        // J = [b1 b2] (3x2), row-major
        Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> J(jacobian);
        J.col(0) = b1;
        J.col(1) = b2;
        return true;
    }

    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        Eigen::Map<const Eigen::Vector3d> y_vec(y);
        Eigen::Map<const Eigen::Vector3d> x_vec(x);
        Eigen::Vector3d x_normalized = x_vec.normalized();

        // 接空間の基底を計算（xの接空間）
        Eigen::Vector3d b1, b2;
        if (std::abs(x_normalized.z()) > 0.5) {
            b1 = Eigen::Vector3d(1.0, 0.0, -x_normalized.x() / x_normalized.z()).normalized();
        } else {
            b1 = Eigen::Vector3d(-x_normalized.y() / x_normalized.x(), 1.0, 0.0).normalized();
        }
        b2 = x_normalized.cross(b1);

        // 近傍での線形近似: delta = [b1^T; b2^T] * (y - x)
        Eigen::Vector3d diff = y_vec - x_vec;
        Eigen::Map<Eigen::Vector2d> delta(y_minus_x);
        delta << b1.dot(diff), b2.dot(diff);
        return true;
    }

    bool MinusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<const Eigen::Vector3d> x_vec(x);
        Eigen::Vector3d x_normalized = x_vec.normalized();

        // 接空間の基底を計算
        Eigen::Vector3d b1, b2;
        if (std::abs(x_normalized.z()) > 0.5) {
            b1 = Eigen::Vector3d(1.0, 0.0, -x_normalized.x() / x_normalized.z()).normalized();
        } else {
            b1 = Eigen::Vector3d(-x_normalized.y() / x_normalized.x(), 1.0, 0.0).normalized();
        }
        b2 = x_normalized.cross(b1);

        // J = [b1^T; b2^T] (2x3), row-major
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobian);
        J.row(0) = b1.transpose();
        J.row(1) = b2.transpose();
        return true;
    }

    int AmbientSize() const override { return 3; } // 3Dベクトル空間
    int TangentSize() const override { return 2; } // 球面の接空間は2D
private:
    double radius_;
};


// 加速度と重力・バイアス誤差(残差)を計算するためのCeresコストファンクタ
struct GravityAccelFunctor {
    GravityAccelFunctor(const Eigen::Vector3d& acc_measurement) 
        : acc_m_(acc_measurement) {}

    template <typename T>
    bool operator()(const T* const gravity, const T* const accel_bias, T* residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> g(gravity);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> b_a(accel_bias);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> r(residual);

        r = acc_m_.template cast<T>() - (g + b_a);
        return true;
    }

private:
    const Eigen::Vector3d acc_m_;
};



class ImuInitializer {
public:
    explicit ImuInitializer(double gravity_magnitude = 9.80665)
        : gravity_magnitude_(gravity_magnitude) {}

    ImuInitializationResult initialize(const std::vector<ImuData>& imu_data) {
        ImuInitializationResult result;
        if (imu_data.size() < 50) {
            std::cerr << "Not enough IMU data for initialization." << std::endl;
            return result;
        }

        // gyro bias計算 (平均とるだけ)
        Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
        for (const auto& data : imu_data) {
            gyro_sum += data.angular_velocity;
        }
        result.gyro_bias = gyro_sum / static_cast<double>(imu_data.size());

        // 重力・加速度バイアス最適化
        Eigen::Vector3d acc_sum = Eigen::Vector3d::Zero();
        for (const auto& data : imu_data) {
            acc_sum += data.linear_acceleration;
        }
        Eigen::Vector3d acc_mean = acc_sum / static_cast<double>(imu_data.size());
        // 最適化の初期値設定
        result.gravity_vec = -acc_mean.normalized() * gravity_magnitude_;
        result.accel_bias.setZero();
        // Ceres Problem 構築
        ceres::Problem problem;
        problem.AddParameterBlock(result.gravity_vec.data(), 3, new SphereManifold(gravity_magnitude_));
        problem.AddParameterBlock(result.accel_bias.data(), 3);
        for (const auto &data : imu_data) {
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<GravityAccelFunctor, 3, 3, 3>(
                new GravityAccelFunctor(data.linear_acceleration)
            );
            problem.AddResidualBlock(cost_function, nullptr, result.gravity_vec.data(), result.accel_bias.data());
        }
        // solver設定＆実行
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR; // QR分解を使用
        options.minimizer_progress_to_stdout = true; // 進捗を標準出力に表示

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;
        
        if (summary.IsSolutionUsable()) {
            result.success = true;
        } else {
            std::cerr << "IMU initialization failed." << std::endl;
        }

        return result;
    }
private:
    const double gravity_magnitude_;
};

} // namespace s3l::utils