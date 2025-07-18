#pragma once

#include <map>
#include <memory>
#include <random>
#include <vector>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <simple_3d_localization/imu_preintegration.hpp>
#include <simple_3d_localization/cloud_deskewing.hpp>
#include <simple_3d_localization/cloud_covariance_estimation.hpp>
#include <simple_3d_localization/cloud_preprocessor.hpp>
#include <simple_3d_localization/estimation_frame.hpp>

namespace s3l {
/**
 * @brief Parameters for OdometryEstimationIMU
 */
struct GLOdomEstimationParams {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GLOdomEstimationParams();
    ~GLOdomEstimationParams();

public:
    // Sensor params
    bool fix_imu_bias;
    double imu_bias_noise;
    Eigen::Isometry3d T_lidar_imu;
    Eigen::Matrix<double, 6, 1> imu_bias;

    // Init state
    std::string initialization_mode;
    bool estimate_init_state;
    Eigen::Isometry3d init_T_world_imu;
    Eigen::Vector3d init_v_world_imu;
    double init_pose_damping_scale;

    // Optimization params
    double smoother_lag;
    bool use_isam2_dogleg;
    double isam2_relinearize_skip;
    double isam2_relinearize_thresh;

    int num_threads;                  // Number of threads for preprocessing and per-factor parallelism
    int num_smoother_update_threads;  // Number of threads for TBB parallelism in smoother update (should be kept 1)
}

class GLOdomEstimation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GLOdomEstimation(std::unique_ptr<GLOdomEstimationParams>&& params);
    ~GLOdomEstimation();

    void insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel);
    EstimationFrame::ConstPtr insert_frame(const PreprocessFrame::Ptr& frame, std::vector<EstimationFrame::ConstPtr>& marginalized_frames);
    std::vector<EstimationFrame::ConstPtr> get_remaining_frames();

    void create_frame(EstimationFrame::Ptr& frame);
    gtsam::NonlinearFactorGraph create_factors(const int current, const std::shared_ptr<gtsam::ImuFactor>& imu_factor, gtsam::Values& new_values);
    void fallback_smoother();
    void update_frames(const int current, const gtsam::NonlinearFactorGraph& new_factors);


private:

    
    

};

} // namespace s3l