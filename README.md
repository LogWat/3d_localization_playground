# Simple 3D Localization

[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](LICENSE)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

3D-LiDARとIMUを使用した自己位置推定用パッケージ

内容物
- [hdl_localization](github.com/koide3/hdl_localization)をROS2で使用できるようにしたもの ukfベース (要修正)
- globalmap_server (hdl_localizationにふくまれてたやつ 要修正)
- グラフ最適化ベース (実装中)


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

## !NOTICE!
**hdl_localization ROS2 の `registration_method` を `ndt_omp` に設定するとプログラムがクラッシュするバグが存在します**
- おそらくPCLのバージョン周りが関係してそうですが修正中　原因わかる方がいれば教えてくださいm(_ _)m

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