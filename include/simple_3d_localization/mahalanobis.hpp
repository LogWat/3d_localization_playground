#pragma once

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <cmath>

inline double squaredMahalanobis(const Eigen::VectorXd& x,
                                 const Eigen::VectorXd& y,
                                 const Eigen::MatrixXd& covariance) {
    Eigen::VectorXd d = x - y;
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if(llt.info()!=Eigen::Success) {
        Eigen::MatrixXd C = covariance;
        C.diagonal().array() += 1e-9;
        llt.compute(C);
    }
    Eigen::VectorXd sol = llt.solve(d);
    return d.dot(sol);
}
