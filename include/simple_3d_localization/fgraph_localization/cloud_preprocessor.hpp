/**
 * @file cloud_preprocessor.hpp
 * @brief ref: https://github.com/koide3/glim
 * @brief This file is modified from glim/preprocess/cloud_preprocessor.hpp and glim/preprocess/preprocesed_frame.hpp
 * @author LogWat
 */

#pragma once

#include <random>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <simple_3d_localization/raw_points.hpp>

namespace s3l {

struct PreprocessFrame {
public:
    using Ptr = std::shared_ptr<PreprocessFrame>;
    using ConstPtr = std::shared_ptr<const PreprocessFrame>;

    /// Number of points
    int size() const { return points_.size(); }

public:
    double stamp_;                         // Timestamp of the first point
    double scan_end_stamp_;                // Timestamp of the last point
    std::vector<double> times_;            // Per-point timestamps relative to the first point
    std::vector<double> intensities_;      // Point intensities
    std::vector<Eigen::Vector4d> points_;  // Point coordinates
    int k_neighbors_;                      // Number of neighbors used for estimation
    std::vector<int> neighbors_;           // Neighbor indices (must be N * k, where N is the number of points)
    RawPoints::ConstPtr raw_points_;       // Raw points data
};

// --------------------------------------------------------------------------------------------------------------

struct CloudPreprocessorParams {
public:
    CloudPreprocessorParams(
        bool global_shutter = false,
        double distance_near_thresh = 1.0,
        double distance_far_thresh = 100.0,
        bool use_random_grid_downsampling = false,
        double downsample_resolution = 0.15,
        int downsample_target = 0,
        double downsample_rate = 0.3,
        bool enable_outlier_removal = false,
        int outlier_removal_k = 10,
        double outlier_std_mul_factor = 2.0,
        bool enable_cropbox_filter = false,
        int k_correspondences = 8,
        int num_threads = 2,
        const Eigen::Isometry3d& T_lidar_imu = Eigen::Isometry3d::Identity(),
        const std::string& crop_bbox_frame = "lidar",
        const Eigen::Vector3d& crop_bbox_min = Eigen::Vector3d::Zero(),
        const Eigen::Vector3d& crop_bbox_max = Eigen::Vector3d::Zero()
    );

    ~CloudPreprocessorParams() {}

public:
    double distance_near_thresh_;        ///< Minimum distance threshold
    double distance_far_thresh_;         ///< Maximum distance threshold
    bool global_shutter_;                ///< Assume all points in a scan are takes at the same moment and replace per-point timestamps with zero (disable deskewing)
    bool use_random_grid_downsampling_;  ///< If true, use random grid downsampling, otherwise, use the conventional voxel grid
    double downsample_resolution_;       ///< Downsampling resolution
    int downsample_target_;              ///< Target number of points for downsampling
    double downsample_rate_;             ///< Downsamping rate (used for random grid downsampling)
    bool enable_outlier_removal_;        ///< If true, apply statistical outlier removal
    int outlier_removal_k_;              ///< Number of neighbors used for outlier removal
    double outlier_std_mul_factor_;      ///< Statistical outlier removal std dev threshold multiplication factor
    bool enable_cropbox_filter_;         ///< If true, filter points out points within box
    std::string crop_bbox_frame_;        ///< Bounding box reference frame
    Eigen::Vector3d crop_bbox_min_;      ///< Bounding box min point
    Eigen::Vector3d crop_bbox_max_;      ///< Bounding box max point
    Eigen::Isometry3d T_imu_lidar_;      ///< LiDAR-IMU transformation when cropbox is defined in IMU frame
    int k_correspondences_;              ///< Number of neighboring points

    int num_threads_;                    ///< Number of threads
};

class CloudPreprocessor {
public:
    using Points = std::vector<Eigen::Vector4d>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CloudPreprocessor(const CloudPreprocessorParams& params = CloudPreprocessorParams());
    virtual ~CloudPreprocessor();

    virtual PreprocessFrame::Ptr preprocess(const RawPoints::ConstPtr& raw_points);

private:
    PreprocessFrame::Ptr preprocess_impl(const RawPoints::ConstPtr& raw_points);
    std::vector<int> get_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const;

    using Params = CloudPreprocessorParams;
    Params params_;

    mutable std::mt19937 mt_;

    std::shared_ptr<void> tbb_task_arena_;
};

}