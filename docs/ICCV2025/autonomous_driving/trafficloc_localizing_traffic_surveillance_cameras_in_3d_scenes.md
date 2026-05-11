---
title: >-
  [Paper Note] TrafficLoc: Localizing Traffic Surveillance Cameras in 3D Scenes
description: >-
  [ICCV 2025][Autonomous Driving][Image-to-point cloud registration] This paper proposes TrafficLoc, a coarse-to-fine image-to-point cloud registration method that achieves high-accuracy localization of traffic surveillanc…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Image-to-point cloud registration"
  - "traffic camera localization"
  - "cross-modal feature fusion"
  - "contrastive learning"
  - "6-DoF pose estimation"
date: 2026-05-08
content_hash: 1682b500a1bd6ae7
---

# TrafficLoc: Localizing Traffic Surveillance Cameras in 3D Scenes

**Conference**: ICCV 2025
**arXiv**: [2412.10308](https://arxiv.org/abs/2412.10308)
**Code**: [GitHub](https://tum-luk.github.io/projects/trafficloc/)
**Area**: Autonomous Driving
**Keywords**: Image-to-point cloud registration, traffic camera localization, cross-modal feature fusion, contrastive learning, 6-DoF pose estimation

## TL;DR
This paper proposes TrafficLoc, a coarse-to-fine image-to-point cloud registration method that achieves high-accuracy localization of traffic surveillance cameras in 3D reference maps via Geometry-guided Attention Loss (GAL), Inter- and Intra-modal Contrastive Learning (ICL), and Dense Training Alignment (DTA). On the self-constructed Carla Intersection dataset, it outperforms the previous state of the art by up to 86%.

## Background & Motivation
Traffic surveillance cameras are cost-effective and easily deployable roadside sensors for collaborative perception, providing a broad global view of traffic. Fusing their data with onboard sensors enhances situational awareness and supports applications such as early obstacle detection and vehicle localization. However, traffic camera localization faces three major challenges:

**Large viewpoint discrepancy**: Images and 3D reference point clouds are captured at different times and from different viewpoints, making the accurate initial pose estimates required by conventional registration methods difficult to obtain.

**Cross-modal matching difficulty**: Directly projecting point clouds onto images introduces the "bleeding problem," and the feature gap between modalities is large.

**Varying intrinsics**: Traffic cameras typically use zoom lenses, causing frequent changes in intrinsic parameters.

Existing methods either require manual intervention (e.g., manual 2D–3D feature matching) or rely on additional panoramic or rendered images, increasing deployment complexity. Although existing I2P registration methods (e.g., CoFiI2P, CFI2P) perform well on vehicle-mounted cameras, their performance degrades sharply under the large viewpoint variations typical of traffic intersections.

**Core Idea**: Design a geometry-aware cross-modal feature fusion module combined with inter- and intra-modal contrastive learning and dense training alignment to achieve high-accuracy traffic camera localization within a single-stage training pipeline.

## Method

### Overall Architecture
TrafficLoc adopts a coarse-to-fine localization strategy. Given a traffic camera image and a 3D scene point cloud, the method first extracts 2D image patch features and 3D point group features separately, fuses them through a Geometry-guided Feature Fusion (GFF) module, performs coarse matching (patch–group level), then fine matching (pixel–point level), and finally estimates the 6-DoF camera pose using EPnP-RANSAC.

### Key Designs
1. **Dual-branch Feature Extraction**

    - **Function**: Extracts multi-scale features from the image and point cloud separately.
    - **Mechanism**: The image branch uses ResNet-18 for multi-level feature extraction and augments spatial relationships with a pretrained ViT encoder from DUSt3R, reducing dimensionality to 256 via $1 \times 1$ convolutions. The point cloud branch uses PointNet for per-point feature extraction, generates $M$ superpoints via FPS to form point groups, and enhances local geometric features with Point Transformer.
    - **Design Motivation**: DUSt3R possesses strong 3D coordinate regression capability and effectively encodes the spatial structure of images.

2. **Geometry-guided Feature Fusion (GFF) + Geometry-guided Attention Loss (GAL)**

    - **Function**: Enhances geometric awareness in Transformer-based cross-modal fusion.
    - **Mechanism**: Geometric supervision is applied to the last cross-attention layer of the Fusion Transformer. For I2P attention, an indicator function is constructed based on the angular radius $\text{Rad}(i,j)$ between the camera ray $OI_i$ and the point group center $P_j$:
    $$\mathbb{1}_{I2P}(i,j) = \begin{cases} 1, & \text{if } \text{Rad}(i,j) < \theta_{low} \\ 0, & \text{if } \text{Rad}(i,j) > \theta_{up} \end{cases}$$
    The raw cross-attention map is supervised with a BCE loss: $L_{I2P}(i,j) = \text{BCE}(\sigma(ATT_{I2P}(i,j)), \mathbb{1}_{I2P}(i,j))$. Similarly, P2I attention uses the distance from a point to the camera ray as the indicator function.
    - **Design Motivation**: Applying Transformer-based cross-modal fusion without geometric guidance yields poor robustness under large viewpoint variations. The intermediate threshold interval is left unsupervised to allow the network to learn flexibly.

3. **Inter- and Intra-modal Contrastive Learning (ICL) + Dense Training Alignment (DTA)**

    - **Function**: Enhances feature discriminability and global alignment in the coarse matching stage.
    - **Mechanism**: ICL not only pulls positive pairs (matched patch–group pairs) closer but also increases intra-modal feature distances between different patches or groups. The loss function is:
    $$L_{coarse}^S = \log\!\left[1 + \sum_j \exp^{\alpha_p(1 - s_p^j + m_p)} \sum_k \exp^{\alpha_n(s_n^k - m_n)}\right]$$
    where $\alpha_p, \alpha_n$ are adaptive weights. DTA performs dense position regression over all image patches via soft-argmax: $\hat{u}_x = \text{SoftArgmax}(S_x)$, propagating gradients to all patches.
    - **Design Motivation**: Conventional contrastive learning considers only cross-modal pairs and ignores intra-modal feature variance; sparsely sampled patch–group pairs during training overlook additional global features.

### Loss & Training
The total loss is a weighted sum of four components:
$$L = \lambda_1 L_{Att} + \lambda_2 L_{det} + \lambda_3 L_{coarse} + \lambda_4 L_{fine}$$
- $L_{Att}$: Geometry-guided attention loss (BCE)
- $L_{det}$: Detection loss within the viewing frustum (BCE)
- $L_{coarse} = L_{coarse}^S + L_{coarse}^D$: ICL + DTA coarse matching loss
- $L_{fine} = L_{fine}^S + L_{fine}^D$: CE + L2 fine matching loss

## Key Experimental Results

### Main Results

| Dataset | Metric | TrafficLoc | CoFiI2P (Prev. SOTA) | Gain |
|--------|------|-----------|-----------------|------|
| Carla Test$_{T1\text{-}T7}$ | RRE(°) / RTE(m) | **0.66 / 0.51** | 4.24 / 2.82 | RRE↓85%, RTE↓82% |
| Carla Test$_{T1\text{-}T7\text{hard}}$ | RRE(°) / RTE(m) | **2.64 / 1.13** | 7.87 / 5.34 | RRE↓66%, RTE↓78% |
| Carla Test$_{T10}$ (unseen) | RRE(°) / RTE(m) | **2.53 / 2.69** | 17.78 / 7.43 | RRE↓86%, RTE↓64% |
| KITTI Odometry | RRE(°) / RTE(m) | **0.87 / 0.19** | 1.14 / 0.29 | RTE↓34% |
| NuScenes | RRE(°) / RTE(m) / RR(%) | **1.38 / 0.78 / 99.45** | 1.48 / 0.87 / 98.67 | — |

### Ablation Study

| Configuration | RRE(°) | RTE(m) | Note |
|------|--------|--------|------|
| Baseline (NCL) | 1.53 | 0.82 | Standard contrastive learning only |
| + ICL | 1.27 | 0.74 | RTE↓9.8% |
| + ICL + DTA | 1.01 | 0.62 | RRE↓20.5%, RTE↓16.2% |
| + ICL + DTA + CM | 0.84 | 0.62 | With coarse matching |
| Full (+ FM + GAL) | **0.66** | **0.51** | GAL contributes RTE↓17.7% |

### Key Findings
- GAL focuses P2I attention on the projected regions of point groups and distributes I2P attention along camera rays, significantly improving performance under large viewpoint variations.
- ICL effectively increases intra-modal feature distances, yielding a clearer distribution in the similarity matrix.
- DTA eliminates the multi-peak problem caused by sparse supervision, retaining only a single peak in the similarity map.
- The model trained on Carla generalizes well to the real-world USTC intersection dataset with qualitatively good results, demonstrating Sim2Real transferability.

## Highlights & Insights
- **New dataset**: Carla Intersection covers 75 intersections across 8 worlds, filling a gap in traffic camera localization benchmarks.
- **Geometry-guided attention**: Using projective geometry as a supervision signal for cross-attention is a general and transferable strategy for enhancing cross-modal fusion.
- **ICL design philosophy**: Jointly optimizing cross-modal alignment and intra-modal discriminability is more effective than purely cross-modal contrastive learning.
- When intrinsics are unknown, DUSt3R can be used to predict and initialize them.

## Limitations & Future Work
- The method assumes the point cloud has been preprocessed (accumulated and downsampled), limiting real-time applicability.
- A domain gap remains between Carla simulation and real-world scenes; while Sim2Real qualitative results are promising, quantitative evaluation is absent.
- Inference speed (0.85 s with GT K) has room for further optimization.
- Robustness under extreme occlusion or low-texture scenarios has not been thoroughly evaluated.

## Related Work & Insights
- The coarse-to-fine strategy is similar to LoFTR but extended to cross-modal (image–point cloud) settings.
- The geometric indicator function design in GAL is generalizable to other attention mechanisms requiring geometric awareness.
- Extending ICL to other cross-modal matching tasks (e.g., image–text, image–audio) is a promising direction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ GAL and ICL are novel designs, though the overall framework represents incremental improvement.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-dataset validation, detailed ablation, visualization analysis, and Sim2Real testing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Traffic camera localization is a critical foundational capability for intelligent transportation systems with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception](unleashing_the_temporal_potential_of_stereo_event_cameras_for_continuous-time_3d.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](../../AAAI2026/autonomous_driving/tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[ICCV 2025\] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation](long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario.md)
- [\[CVPR 2026\] ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes](../../CVPR2026/autonomous_driving/rescene4d_temporally_consistent_semantic_instance_segmentation_of_evolving_indoo.md)
- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
