# Simple 3D Localization

[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](LICENSE)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

3D-LiDARとIMUを使用した比較検証用に使用できる簡易な3次元位置推定パッケージ

推定器としてEKF(Extended Kalman Filter)とUKF(Unscented Kalman Filter)を実装し、点群レジストレーションにはNDT(ndt_omp), GICP(small_gicp), VGICP(small_gicp)を使用可能

内容物
- [hdl_localization](github.com/koide3/hdl_localization)をROS2で使用できるようにしたもの + 実験用にカルマンフィルタをUKFとEKFで切替可能にした + 点群レジストレーションで NDT(ndt_omp), GICP(small_gicp), VGICP(small_gicp)を切替可能にした
- globalmap_server (hdl_localizationにふくまれてたやつ 要修正)
- グラフ最適化ベース (実装中)


UKF, EKFの設計に関するdocは準備中…

## Installation

### Prerequisites

- ROS2 Humble
- Ubuntu 22.04 (recommended)

### Dependencies

- **Point Cloud Library (PCL)** >= 1.12
- **Eigen3**
- **small_gicp**: Fast GICP implementation
- **ndt_omp**: OpenMP-accelerated NDT
- **fast_gicp**: CUDA-accelerated GICP
- **gtsam_points**: Factor graph optimization (for FGraph localization)

## How to use

1. hdl_localization
    - hdl_localization.launch.xml を確認
2. fgraph_localization
    - UNDER CONSTRUCTION

## Acknowledgments
- [hdl_localization](github.com/koide3/hdl_localization): original
- [GLIM](https://github.com/koide3/glim): Factor graph localization inspiration
- [small_gicp](https://github.com/koide3/small_gicp): Fast GICP implementation
- [ndt_omp(TIERIV)](https://github.com/tier4/ndt_omp): NDT implementation
- [fast_gicp](https://github.com/koide3/fast_gicp): NDT_CUDA implementation
- [gtsam_points](https://github.com/koide3/gtsam_points): Point cloud processing utilities