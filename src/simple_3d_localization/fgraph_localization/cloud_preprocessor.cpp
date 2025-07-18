#include <simple_3d_localization/cloud_preprocessor.hpp>

#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/parallelism.hpp>

#ifdef GTSAM_POINTS_HAVE_TBB
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#endif

namespace s3l {

CloudPreprocessorParams::CloudPreprocessorParams(
    bool global_shutter,
    double distance_near_thresh,
    double distance_far_thresh,
    bool use_random_grid_downsampling,
    double downsample_resolution,
    int downsample_target,
    double downsample_rate,
    bool enable_outlier_removal,
    int outlier_removal_k,
    double outlier_std_mul_factor,
    bool enable_cropbox_filter,
    int k_correspondences,
    int num_threads,
    const Eigen::Isometry3d& T_lidar_imu,
    const std::string& crop_bbox_frame,
    const Eigen::Vector3d& crop_bbox_min,
    const Eigen::Vector3d& crop_bbox_max
) : global_shutter_(global_shutter),
    distance_near_thresh_(distance_near_thresh),
    distance_far_thresh_(distance_far_thresh),
    use_random_grid_downsampling_(use_random_grid_downsampling),
    downsample_resolution_(downsample_resolution),
    downsample_target_(downsample_target),
    downsample_rate_(downsample_rate),
    enable_outlier_removal_(enable_outlier_removal),
    outlier_removal_k_(outlier_removal_k),
    outlier_std_mul_factor_(outlier_std_mul_factor),
    enable_cropbox_filter_(enable_cropbox_filter),
    k_correspondences_(k_correspondences),
    num_threads_(num_threads),
    T_imu_lidar_(T_lidar_imu.inverse())
{
    crop_bbox_frame_ = "lidar";
    crop_bbox_max_.setZero();
    crop_bbox_min_.setZero();

    if (enable_cropbox_filter_) {
        crop_bbox_frame_ = crop_bbox_frame;
        crop_bbox_min_ = crop_bbox_min;
        crop_bbox_max_ = crop_bbox_max;
        if (crop_bbox_frame_ != "lidar" && crop_bbox_frame_ != "imu") {
            throw std::runtime_error("Invalid crop bounding box frame: " + crop_bbox_frame_);
        } else if ((crop_bbox_min_.array() > crop_bbox_max_.array()).any()) {
            throw std::runtime_error("Crop bounding box min point must be less than max point.");
        }
    }
}

CloudPreprocessor::CloudPreprocessor(const CloudPreprocessorParams& params) : params_(params) {
    #ifdef GTSAM_POINTS_HAVE_TBB
    if (gtsam_points::is_tbb_default()) {
        tbb_task_arena_.reset(new tbb::task_arena(params.num_threads_));
    }
    #endif
}

CloudPreprocessor::~CloudPreprocessor() {}

PreprocessFrame::Ptr CloudPreprocessor::preprocess(const RawPoints::ConstPtr& raw_points) {
    if (gtsam_points::is_omp_default() || params_.num_threads_ == 1 || !tbb_task_arena_) {
        return preprocess_impl(raw_points);
    }
    PreprocessFrame::Ptr preprocessed;
    #ifdef GTSAM_POINTS_HAVE_TBB
        auto arena = static_cast<tbb::task_arena*>(tbb_task_arena_.get());
        arena->execute([&] { preprocessed = preprocess_impl(raw_points); });
    #else
        std::cerr << "TBB is not available, falling back to single-threaded execution." << std::endl;
        abort();
    #endif
    return preprocessed;
}

// Preprocess
PreprocessFrame::Ptr CloudPreprocessor::preprocess_impl(const RawPoints::ConstPtr& raw_points) {
    gtsam_points::PointCloud::Ptr frame(new gtsam_points::PointCloud);
    frame->num_points = raw_points->points.size();
    frame->times = const_cast<double*>(raw_points->times.data());
    frame->points = const_cast<Eigen::Vector4d*>(raw_points->points.data());
    if (raw_points->intensities.size()) frame->intensities = const_cast<double*>(raw_points->intensities.data());

    // Downsampling
    if (params_.use_random_grid_downsampling_) {
        const double rate = params_.downsample_target_ > 0 ? static_cast<double>(params_.downsample_target_) / frame->num_points : params_.downsample_rate_;
        frame = gtsam_points::randomgrid_sampling(frame, params_.downsample_resolution_, rate, mt_, params_.num_threads_);
    } else {
        frame = gtsam_points::voxelgrid_sampling(frame, params_.downsample_resolution_, params_.num_threads_);
    }

    // Distance filtering
    std::vector<int> indices;
    indices.reserve(frame->size());
    double squared_distance_near_thresh = params_.distance_near_thresh_ * params_.distance_near_thresh_;
    double squared_distance_far_thresh = params_.distance_far_thresh_ * params_.distance_far_thresh_;

    for (int i = 0; i < frame->size(); i++) {
        const bool is_finite = frame->points[i].allFinite();
        const double squared_dist = (Eigen::Vector4d() << frame->points[i].head<3>(), 0.0).finished().squaredNorm();
        if (squared_dist > squared_distance_near_thresh && squared_dist < squared_distance_far_thresh && is_finite) {
            indices.push_back(i);
        }
    }

    // Sort by timestamp
    std::sort(indices.begin(), indices.end(), [&](const int lhs, const int rhs) {
        return frame->times[lhs] < frame->times[rhs];
    });
    frame = gtsam_points::sample(frame, indices);

    if (params_.global_shutter_) std::fill(frame->times, frame->times + frame->size(), 0.0);

    // Cropbox filter
    if (params_.enable_cropbox_filter_) {
        if (params_.crop_bbox_frame_ == "lidar") {
            auto is_inside_bbox = [&](const Eigen::Vector3d& p_lidar) {
                return (p_lidar.array() >= params_.crop_bbox_min_.array()).all() &&
                       (p_lidar.array() <= params_.crop_bbox_max_.array()).all();
            };
            frame = gtsam_points::filter(frame, [&](const auto &pt) { return !is_inside_bbox(pt.template head<3>()); });
        } else {
            throw std::runtime_error("IMU frame cropbox filtering is not implemented yet.");
        }
    }

    // Outlier removal
    if (params_.enable_outlier_removal_) frame = gtsam_points::remove_outliers(
        frame, params_.outlier_removal_k_, params_.outlier_std_mul_factor_, params_.num_threads_);

    // Create a PreprocessFrame
    PreprocessFrame::Ptr preprocessed(new PreprocessFrame);
    preprocessed->stamp_ = raw_points->stamp;
    preprocessed->scan_end_stamp_ = frame->size() ? raw_points->stamp + frame->times[frame->size() - 1] : raw_points->stamp;
    preprocessed->times_.assign(frame->times, frame->times + frame->size());
    preprocessed->intensities_.assign(frame->points, frame->points + frame->size());
    if (raw_points->intensities.size()) {
        preprocessed->intensities_.assign(raw_points->intensities.begin(), raw_points->intensities.end());
    }
    preprocessed->k_neighbors_ = params_.k_correspondences_;
    preprocessed->neighbors_ = get_neighbors(frame->points, frame->size(), params_.k_correspondences_);

    return preprocessed;
}

// NN Search
std::vector<int> CloudPreprocessor::get_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const {
    gtsam_points::KdTree tree(points, num_points);
    std::vector<int> neighbors(num_points * k);
    
    const auto perpoint_task = [&](const int i) {
        std::vector<size_t> k_indices(k);
        std::vector<double> k_sq_dists(k);
        tree.knn_search(points[i].data(), k, k_indices.data(), k_sq_dists.data());
        std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
    };

    if (gtsam_points::is_omp_default()) {
        #pragma omp parallel for num_threads(params_.num_threads_) schedule(guided, 8)
        for (int i = 0; i < num_points; ++i) perpoint_task(i);
    } else {
        #ifdef GTSAM_POINTS_HAVE_TBB
        tbb::parallel_for(tbb::blocked_range<int>(0, num_points), [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); ++i) perpoint_task(i);
        });
        #else
        std::cerr << "TBB is not available, falling back to OpenMP." << std::endl;
        abort();
        #endif
    }

    return neighbors;
}


} // namespace s3l
