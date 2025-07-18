/**
 * @file gl_component.hpp
 * @brief Header file for the GL(Graph-based Localization) component
 * @brief This component implements a 3D localization system using point cloud data and IMU
 * @brief reference:
 * @author LogWat
 */

#pragma once

#include <map>
#include <memory>
#include <random>
#include <mutex>
#include <thread>
#include <atomic>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <simple_3d_localization/cloud_deskewing.hpp>
#include <simple_3d_localization/cloud_preprocessor.hpp>
#include <simple_3d_localization/cloud_covariance_estimation.hpp>

namespace s3l {
class GLSystem {

};

} // namespace s3l
