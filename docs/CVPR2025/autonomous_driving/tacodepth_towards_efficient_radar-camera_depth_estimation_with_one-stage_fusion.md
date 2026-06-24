---
title: >-
  [Paper Note] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion
description: >-
  [CVPR 2025][Autonomous Driving][Depth Estimation] TacoDepth proposes the first one-stage radar-camera fusion depth estimation framework. By utilizing a graph-based radar structure extractor and a pyramid-based radar fusion module, it bypasses the need for intermediate quasi-dense depth maps, improving accuracy by 12.8% and speed by 91.8%, achieving real-time performance at 37+ FPS.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Depth Estimation"
  - "Radar-Camera Fusion"
  - "One-Stage Fusion"
  - "Graph Neural Networks"
  - "Real-Time Processing"
date: 2026-05-08
content_hash: 8dcb1368279ecacf
---

# TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion

**Conference**: CVPR 2025  
**arXiv**: [2504.11773](https://arxiv.org/abs/2504.11773)  
**Code**: [https://github.com/RaymondWang987/TacoDepth](https://github.com/RaymondWang987/TacoDepth)  
**Area**: Autonomous Driving  
**Keywords**: Depth Estimation, Radar-Camera Fusion, One-Stage Fusion, Graph Neural Networks, Real-Time Processing

## TL;DR

TacoDepth proposes the first one-stage radar-camera fusion depth estimation framework. By utilizing a graph-based radar structure extractor and a pyramid-based radar fusion module, it bypasses the need for intermediate quasi-dense depth maps, improving accuracy by 12.8% and speed by 91.8%, achieving real-time performance at 37+ FPS.

## Background & Motivation

**Background**: Radar-camera depth estimation is a critical task for 3D perception in autonomous driving and robotics. Compared to LiDAR, millimeter-wave radar is low-cost, low-power, and reliable under all weather conditions, but its point clouds are extremely sparse (1000 times sparser than LiDAR) and noisy.

**Limitations of Prior Work**: To overcome radar sparseness, mainstream methods (such as Singh et al., RC-PDA, RadarCam-Depth) adopt complex multi-stage frameworks—first predicting an intermediate "quasi-dense depth" map from sparse radar, and then constructing the final dense depth based on it. This yields three limitations: (1) multi-stage inference is highly inefficient; for instance, the four-stage pipeline of RadarCam-Depth takes 358ms; (2) the intermediate quasi-dense depth remains sparse and noisy, sometimes yielding almost zero valid depth values under harsh lighting; (3) flawed intermediate results lead to structural fractures, blurred details, and obvious artifacts in the final depth prediction.

**Key Challenge**: The extreme sparsity of radar point clouds necessitates an intermediate stepping stone, but multi-stage architectures compromise efficiency and robustness. Can a one-stage approach directly map sparse radar to dense depth?

**Goal**: Achieve one-stage radar-camera depth estimation, simultaneously improving efficiency and accuracy while supporting both independent and plug-in inference modes.

**Key Insight**: Previous methods only extract features from single-point coordinates of radar points, neglecting the geometric structure and topological information of the point cloud. The graph structure of a point cloud (inter-point distances/relationships) is more informative and robust to noise than single-point coordinates.

**Core Idea**: Utilize GNNs to extract graph structural features from radar point clouds, and efficiently integrate radar structural information with multi-scale image semantic features via hierarchical pyramid-based fusion to achieve one-stage fusion.

## Method

### Overall Architecture

The inputs are a single RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and a radar point cloud $P \in \mathbb{R}^{K \times 3}$ ($K$ points with 3D coordinates), and the output is a dense metric depth map $D \in \mathbb{R}^{H \times W}$. The framework consists of two main steps: (1) a graph-based radar structure extractor extracts hierarchical representations of node and edge features from the point cloud; (2) a pyramid-based radar fusion module merges the radar features with multi-scale image features extracted by ResNet-18 layer by layer; finally, a decoder outputs the dense depth map.

### Key Designs

1. **Graph-based Radar Structure Extractor (GE)**:

    - **Function**: Extract rich geometric structure and topological features from sparse radar point clouds, replacing the simple point-wise MLP feature extraction of prior methods.
    - **Mechanism**: Treat radar points as graph nodes and construct an adjacency matrix to represent edges. A lightweight GNN architecture (PCA-GM) is adopted to update node features $N_l$ and aggregate edge features $E_l$ layer by layer across $L=3$ layers. Shallow layers capture point coordinates, while deep layers capture global topological structures. The output is the node and edge features of each layer for subsequent fusion.
    - **Design Motivation**: Singh et al. use MLPs to extract 32,256-dimensional features from 3D coordinates, introducing significant redundancy and noise. GNNs can model geometric relationships (e.g., distance, direction) between points, which are more robust to noise and outliers, and provide more effective information than single-point coordinates for one-stage fusion.

2. **Pyramid-based Radar Fusion (PF)**:

    - **Function**: Hierarchically fuse radar graph structure features with multi-scale image features to achieve efficient cross-modal alignment.
    - **Mechanism**: Node features $N_l$ from the $l$-th layer of GNN are fused with image features $F_{2l-1}$, and edge features $E_l$ are fused with $F_{2l}$. Within each layer, "Radar-Centric Flash Attention" is used to establish cross-modal correspondences: for each radar point $p$, a window region $[x_p - a_l, x_p + a_l]$ is defined based on its horizontal coordinate $x_p$. Attention is computed only between image pixels and radar points within this region. Queries are derived from image features, and keys/values are derived from radar edge features: $F'_{2l}[m] = \text{softmax}\frac{W_q \hat{F}_{2l}[m] (W_k \hat{E}_l)^T}{\sqrt{C_l}} W_v \hat{E}_l$.
    - **Design Motivation**: Global image attention is computationally heavy and introduces noise from irrelevant pixels. Leveraging the prior that radar horizontal coordinates are relatively accurate (compared to elevation), localized Flash Attention is performed centering around horizontal positions, significantly reducing computational cost. Shallow layers fuse details and coordinates, while deep layers fuse semantics and structure, forming a complementary fusion.

3. **Flexible Inference Modes (Independent + Plug-in)**:

    - **Function**: Support both independent inference (requiring no external depth model, achieving real-time speed at 37+ FPS) and plug-in inference (utilizing pre-trained depth models to boost accuracy).
    - **Mechanism**: Add an optional input branch to process an initial relative depth map $D^*$. In independent mode, $D^* = 0$; in plug-in mode, it integrates relative depth outputs from DPT, MiDaS, Depth-Anything-v2, etc. During training, each epoch randomly feeds relative depth for half of the data and zero input for the other half, optimizing both modes simultaneously: $D = \mathcal{T}_\theta(I, P | D^*)$.
    - **Design Motivation**: Independent models are efficient but have limited accuracy, while plug-in models are accurate but introduce latency. A unified framework allows users to choose based on their needs and seamlessly benefit from stronger depth predictors.

### Loss & Training

The training loss is a combination of L1 losses: $\ell_{L_1} = \frac{1}{|\Omega_{gt}|}\sum|D - D_{gt}| + \frac{\lambda}{|\Omega_{acc}|}\sum|D - D_{acc}|$, where $D_{gt}$ is the LiDAR ground truth depth, $D_{acc}$ is the accumulated reprojection depth map, and $\lambda=1$. The model is trained using the Adam optimizer on two A6000 GPUs for 50 epochs, with an initial learning rate of 1e-4, decaying by 1e-5 every 10 epochs.

## Key Experimental Results

### Main Results

nuScenes 0-70m range (Independent Mode):

| Method | MAE↓ | RMSE↓ | Inference Time (ms)↓ |
|------|------|-------|-------------|
| **TacoDepth** | **1712.6** | **3960.5** | **26.7** |
| Li et al. (ECCV'24) | 1822.9 | 4303.6 | 67.6 |
| Singh et al. (CVPR'23) | 2073.2 | 4590.7 | 94.2 |
| CaFNet (IROS'24) | 2010.3 | 4493.1 | 103.9 |

nuScenes 0-70m range (Plug-in Mode, DPT-Hybrid):

| Method | MAE↓ | RMSE↓ | Inference Time (ms)↓ |
|------|------|-------|-------------|
| **TacoDepth** | **1347.1** | **3152.8** | **29.3** |
| RadarCam-Depth | 1587.9 | 3662.5 | 358.3 |

### Ablation Study

| Configuration | MAE↓ | RMSE↓ |
|------|------|-------|
| RGB Baseline (w/o Radar) | 2474.3 | 5402.1 |
| Singh et al. (MLP + Two-stage) | 2073.2 | 4590.7 |
| TacoDepth w/o GE (MLP + PF) | 1815.6 | 4189.8 |
| **TacoDepth (GE + PF)** | **1712.6** | **3960.5** |

Model efficiency comparison:

| Method | Params (M)↓ | FLOPs (G)↓ | Time (ms)↓ |
|------|-----------|-----------|----------|
| **TacoDepth Independent** | **13.47** | **139.30** | **26.7** |
| Singh et al. | 22.81 | 502.09 | 94.2 |
| RadarCam-Depth | 33.26 | 619.02 | 358.3 |

### Key Findings

- PF (Pyramid Fusion) makes the largest contribution: replacing the fusion mechanism with PF while keeping MLP (without GE) drops MAE from 2073 to 1816 (a 12.4% reduction), proving the intrinsic value of the one-stage fusion strategy itself.
- GE further reduces MAE by 5.7% on top of PF, verifying that graph structural features are more effective than point-wise features.
- GNN depth $L=3$ is optimal; deeper networks ($L=4$) fail to benefit due to the extreme sparsity of the radar data.
- Performance gains are more pronounced in night scenes (MAE reduced by 29.1% vs. 12.9% during the day), demonstrating that the one-stage approach is more robust in adverse conditions where the intermediate depth predictions of multi-stage methods almost entirely fail.
- The plug-in mode scales well with stronger depth predictors (such as Depth-Anything-v2): reducing MAE from 983 to 730, showcasing outstanding compatibility.

## Highlights & Insights

- **Paradigm shift of One-Stage vs. Multi-Stage**: Intuitively, mapping sparse radar to dense depth seems to require an intermediate stepping stone. However, this work demonstrates that with superior feature extraction and fusion strategies, the intermediate step can be bypassed. This "de-intermediation" philosophy is highly worth exploring in other sparse-to-dense tasks.
- **Radar-Centric Flash Attention**: Leveraging the domain prior that radar horizontal coordinates are accurate, this design replaces global attention with localized attention at an extremely low cost, which is highly efficient and mitigates interference from irrelevant regions. This design is easily transferable to other sparse-to-dense sensor fusion scenarios.
- **Dual-Mode Training Strategy**: Randomly feeding initial relative depth for 50% of the training data and zero for the rest allows a single model to master both inference modes simultaneously. The design is simple yet highly effective.

## Limitations & Future Work

- The authors acknowledge that this work only provides one initial efficient implementation; more advanced techniques remain to be explored.
- It has only been validated on nuScenes (using 3D radar) and ZJU-4DRadarCam (using 4D radar); its generalizability requires further testing across diverse scenarios.
- The image encoder uses ResNet-18; upgrading to stronger backbones like ViT might yield further performance gains.
- The GNN construction strategy (fully connected adjacency matrix) may encounter scalability issues when the number of points increases.

## Related Work & Insights

- **vs. Singh et al. (CVPR'23)**: A representative two-stage independent method employing MLP for radar feature extraction and gated fusion. Ours uses GNN + pyramid fusion, requiring 41% fewer parameters and 72% fewer FLOPs while delivering higher accuracy.
- **vs. RadarCam-Depth (ICRA'24)**: A four-stage plug-in method (initial depth $\rightarrow$ global scale alignment $\rightarrow$ quasi-dense depth $\rightarrow$ scale learner). Ours operates in a single stage, running 92% faster with 12% higher accuracy.
- **vs. Depth-Anything-v2**: A vision-only relative depth model. Ours can act as a plug-in to convert its relative depth predictions into accurate metric depth maps.

## Rating

- Novelty: ⭐⭐⭐⭐ The one-stage fusion paradigm and radar-centric attention designs are novel, though the overall components (GNN/Flash Attention/pyramid fusion) represent a clever combination of existing technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across two datasets, diverse baselines, thorough ablation studies, complete efficiency analyses, and separate assessments of day/night scenarios.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with informative charts, and the motivation and pain points are clearly articulated.
- Value: ⭐⭐⭐⭐ Real-time radar-camera fusion depth estimation carries strong practical value for autonomous driving. The open-source code enhances reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[CVPR 2025\] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)
- [\[CVPR 2025\] RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network](rc-autocalib_an_end-to-end_radar-camera_automatic_calibration_network.md)

</div>

<!-- RELATED:END -->
