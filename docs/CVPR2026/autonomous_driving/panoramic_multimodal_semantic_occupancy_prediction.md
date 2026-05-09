---
title: >-
  [Paper Note] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots
description: >-
  [CVPR 2026][Autonomous Driving][Panoramic occupancy prediction] This paper introduces PanoMMOcc, the first panoramic multimodal (RGB + thermal + polarization + LiDAR) semantic occupancy dataset for quadruped robots, and proposes VoxelHound, a framework achieving robust 3D occupancy prediction via Vertical Jitter Compensation (VJC) and Multimodal Information Prompt Fusion (MIPF) modules, attaining 23.34% mIoU (+4.16%).
tags:
  - CVPR 2026
  - Autonomous Driving
  - Panoramic occupancy prediction
  - quadruped robots
  - multimodal fusion
  - vertical jitter compensation
  - BEV perception
date: 2026-05-08
content_hash: b4865d1f5f411e32
---

# Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots

**Conference**: CVPR 2026
**arXiv**: [2603.13108](https://arxiv.org/abs/2603.13108)
**Code**: [https://github.com/SXDR/PanoMMOcc](https://github.com/SXDR/PanoMMOcc)
**Area**: Autonomous Driving
**Keywords**: Panoramic occupancy prediction, quadruped robots, multimodal fusion, vertical jitter compensation, BEV perception

## TL;DR
This paper introduces PanoMMOcc, the first panoramic multimodal (RGB + thermal + polarization + LiDAR) semantic occupancy dataset for quadruped robots, and proposes VoxelHound, a framework achieving robust 3D occupancy prediction via Vertical Jitter Compensation (VJC) and Multimodal Information Prompt Fusion (MIPF) modules, attaining 23.34% mIoU (+4.16%).

## Background & Motivation

### State of the Field

**Background**: 3D semantic occupancy prediction serves as a critical intermediate representation bridging perception and motion planning, enabling unified modeling of free space, occupied space, and unknown space. Panoramic cameras provide 360° blind-spot-free visual coverage, making them well-suited for mobile robots. However, existing occupancy prediction methods and datasets are almost exclusively designed for wheeled autonomous driving scenarios, relying on multi-view pinhole cameras and vehicle-mounted LiDAR. Quadruped robots face three unique challenges: (1) low sensor viewpoints with severe self-occlusion; (2) gait-induced vertical body oscillation causing image blur and feature misalignment; (3) insufficient robustness of RGB-only perception under varying illumination, low-texture regions, and long-range scenes. A joint panoramic imaging and multimodal sensing solution is thus required, yet no such dataset or method previously existed.

### Limitations of Prior Work

**Goal**: How to achieve accurate 3D semantic occupancy prediction on a quadruped robot platform by leveraging panoramic cameras and complementary sensors (thermal, polarization, LiDAR) while overcoming gait-induced jitter and single-modality limitations? This encompasses three sub-problems: (1) the absence of a panoramic multimodal occupancy dataset for quadruped robots; (2) gait-induced vertical jitter disrupting the spatial consistency of BEV transformations; (3) effective fusion strategies for heterogeneous modalities.

## Method

### Overall Architecture
VoxelHound accepts four modality inputs: panoramic RGB images (PAL camera, 360°×70° FoV), thermal images, polarization images, and LiDAR point clouds. The camera branch extracts multi-scale features from each image modality using ResNet-18, aggregates them via FPN, and projects them into bird's-eye view (BEV) space through a 2D-to-BEV transformation. The LiDAR branch voxelizes the point cloud and extracts features via sparse 3D convolution, then compresses them onto the BEV plane. BEV features from all four modalities are fused and passed to a BEV encoder (SECOND-FPN architecture) for contextual modeling. An occupancy head then reshapes the channel dimension of the BEV features into the vertical dimension, producing a 64×64×16 3D semantic occupancy prediction (12 semantic classes + free class).

### Key Designs
1. **Vertical Jitter Compensation Module (VJC)**: Quadruped gait causes systematic vertical body oscillation, introducing vertical shifts in captured images. VJC is inserted between the image encoder and the BEV transformation. Concretely, a vertical structural feature $\mathbf{F}_v \in \mathbb{R}^{C \times H}$ is obtained by averaging the feature map along the width dimension, encoded by two 1D convolution layers with ReLU, and a global vertical offset $\Delta h$ is predicted via adaptive average pooling followed by a linear layer. A sampling grid with the predicted offset is then constructed for bilinear interpolation alignment. The module is extremely lightweight (negligible parameters and memory overhead) yet effectively compensates for gait-induced feature misalignment.
2. **Multimodal Information Prompt Fusion Module (MIPF)**: Conventional multimodal fusion (concatenation/addition) treats all modalities equally, ignoring the role distinction between LiDAR, which provides stable 3D geometric structure, and image modalities, which primarily contribute semantics. MIPF adopts an **asymmetric fusion principle** — geometry-dominant with semantic supplementation. Each modality is projected into a shared embedding space via 1×1 convolution. A compact **semantic prompt vector** $\mathbf{p}_m$ is generated for each image modality's BEV feature through global average pooling and an MLP. Attention interaction is performed with LiDAR BEV features as queries and semantic prompts as keys/values; the result modulates the LiDAR features via sigmoid gating as a residual — adaptively reweighting LiDAR features through prompts rather than overwriting the geometric structure. This design is substantially more efficient than dense spatial cross-attention (prompts consist of only 3 tokens).

### Loss & Training
A composite loss function is employed: cross-entropy loss $\mathcal{L}_{ce}$ + Lovász-Softmax loss $\mathcal{L}_{ls}$ (addressing class imbalance) + geometric affinity loss $\mathcal{L}_{scal}^{geo}$ + semantic affinity loss $\mathcal{L}_{scal}^{sem}$ (encouraging consistency among adjacent voxels). Training uses the AdamW optimizer with a learning rate of 4e-4, weight decay of 0.01, for 48 epochs on 4 RTX 3090 GPUs.

## Key Experimental Results

| Method | Modality | mIoU |
|--------|----------|------|
| MonoScene | C | 8.94 |
| EFFOcc-C | C | 4.47 |
| EFFOcc-L | L | 18.77 |
| EFFOcc-T (C+L) | C+L | 19.18 |
| **VoxelHound** | **C+L+T+P** | **23.34** |

| Lighting Condition | Modality | mIoU |
|--------------------|----------|------|
| Daytime | C+L | 22.56 |
| Daytime | C+L+T+P | 23.34 |
| Nighttime | C+L | 19.17 |
| Nighttime | C+L+T+P | 18.68 |

### Ablation Study
- Baseline (w/o VJC, w/o MIPF): 22.74 mIoU
- +VJC: 22.92 (+0.18), validating the effectiveness of jitter compensation
- +MIPF: 23.14 (+0.40), fusion module contributes more significantly
- Both modules: 23.34 (+0.60), demonstrating complementarity
- VJC hidden channel dimension: 64 is optimal (23.34), with negligible parameter overhead (0.04M)
- MIPF: prompt channel dimension 8 and 8 attention heads yield optimal performance (23.34)

## Highlights & Insights
- **Pioneering contribution**: The first panoramic multimodal occupancy dataset for quadruped robots, filling an important gap in the field.
- **Elegant and effective VJC design**: Global vertical offset estimation via 1D convolution compensates for gait-induced jitter with near-zero computational overhead.
- **Asymmetric fusion philosophy of MIPF**: Compressing image modalities into compact prompts rather than performing dense cross-attention preserves the LiDAR geometric backbone while introducing semantic enhancement. This "geometry-dominant, semantics-supplementary" paradigm is transferable to other multimodal fusion scenarios.
- **Four sensing modalities**: Thermal imaging enhances robustness under low illumination; polarization imaging reveals material properties and weak-target cues — the inclusion of these unconventional modalities merits attention.
- **Open-source calibration tools**: LiDAR-camera calibration tooling is publicly released.

## Limitations & Future Work
- Dataset scale is limited (21.6k frames), far smaller than large-scale autonomous driving datasets (nuScenes 40k, SemanticKITTI 43k), constraining the training of large models.
- Voxel resolution of 0.4 m is relatively coarse, unsuitable for manipulation tasks requiring fine-grained geometry such as grasping.
- Nighttime + full modality (18.68 mIoU) underperforms the daytime C+L configuration (22.56), indicating that the contributions of thermal and polarization imaging at night require improved fusion strategies.
- Coverage is limited to outdoor scenes; indoor environments are absent.
- VJC only compensates for global vertical translation; rotational and local deformations are not modeled.
- Validation is primarily conducted on the authors' own dataset, lacking generalization experiments on other occupancy benchmarks.

## Related Work & Insights
- **vs. EFFOcc**: The closest existing baseline. VoxelHound already surpasses EFFOcc-T by 4.16 mIoU in the camera+LiDAR configuration, with a further advantage upon adding thermal and polarization. The key differences lie in MIPF's asymmetric fusion strategy and VJC's jitter compensation.
- **vs. MonoScene**: MonoScene, a monocular camera occupancy prediction method, achieves only 8.94 mIoU in the panoramic setting, demonstrating the severe inadequacy of vision-only approaches on quadruped platforms (low viewpoint, jitter, illumination variation).
- **vs. QuadOcc**: Also targeting quadruped robots but using only panoramic RGB with fewer categories (6 classes); PanoMMOcc offers significant advantages in sensing modality richness and annotation completeness.

## Related Work & Insights
- The "prompt-based fusion" design in MIPF is generalizable to other multimodal tasks — replacing dense feature interaction with lightweight prompts.

## Rating
- Novelty: ⭐⭐⭐⭐ First panoramic multimodal occupancy dataset and framework for quadruped robots, filling a clear gap
- Experimental Thoroughness: ⭐⭐⭐ Validated only on the authors' own dataset; cross-dataset generalization experiments are absent
- Writing Quality: ⭐⭐⭐⭐ Clear structure, thorough dataset construction details, and comprehensive appendix
- Value: ⭐⭐⭐⭐ Open-source dataset and calibration tools provide significant value to the community

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)
- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)
- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_openvocabulary_occupancy_predi.md)
- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)

<!-- RELATED:END -->
