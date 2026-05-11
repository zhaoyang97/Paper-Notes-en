---
title: >-
  [Paper Note] HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation
description: >-
  [ICCV 2025][Human Understanding][pose estimation] This paper proposes simultaneously predicting the 3D coordinates of both the front and back surfaces of an object and densely sampling between the two surfaces to constru…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "pose estimation"
  - "2D-3D correspondences"
  - "front & back surface prediction"
  - "hierarchical continuous encoding"
  - "PnP"
date: 2026-05-08
content_hash: 356c98f42f15144a
---

# HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation

**Conference**: ICCV 2025  
**arXiv**: [2510.10177](https://arxiv.org/abs/2510.10177)  
**Code**: [https://github.com/WangYuLin-SEU/HCCEPose](https://github.com/WangYuLin-SEU/HCCEPose)  
**Area**: Human Understanding  
**Keywords**: pose estimation, 2D-3D correspondences, front & back surface prediction, hierarchical continuous encoding, PnP

## TL;DR

This paper proposes simultaneously predicting the 3D coordinates of both the front and back surfaces of an object and densely sampling between the two surfaces to construct ultra-dense 2D-3D correspondences. Combined with a novel Hierarchical Continuous Coordinate Encoding (HCCE), the method surpasses existing state-of-the-art approaches on all seven core BOP benchmark datasets.

## Background & Motivation

In instance-level 6D pose estimation, the dominant pipeline predicts 3D surface coordinates via a neural network, establishes 2D-3D correspondences, and solves for the pose using the PnP algorithm. Existing methods suffer from two key limitations:

**Reliance on front surface only**: Current methods predict 3D coordinates only for the visible front surface of the object, neglecting the potential information from the back surface and interior regions. Denser 2D-3D correspondences can help the RANSAC-PnP solver achieve more accurate pose estimates.

**Insufficient encoding precision**: In existing hierarchical binary encoding schemes (e.g., ZebraPose), neural networks struggle to accurately learn binary codes near stripe boundaries, limiting coordinate prediction accuracy.

## Method

### Overall Architecture

Given a cropped RGB image, a neural network simultaneously predicts the object mask, front surface coordinates, and back surface coordinates. Both sets of coordinates are encoded via HCCE into multi-level continuous codes, which the network predicts. At inference time, continuous codes are converted to binary codes and decoded to recover surface coordinates. 3D points are then densely sampled between the predicted front and back surface coordinates to construct ultra-dense 2D-3D correspondences, and the pose is finally solved via RANSAC-PnP.

### Key Designs

1. **Ultra-dense 2D-3D correspondence construction**: The network simultaneously predicts front surface coordinates $\tilde{Q}_f$ and back surface coordinates $\tilde{Q}_b$. For each 2D pixel, the corresponding front and back 3D coordinates $\tilde{q}_1$ and $\tilde{q}_2$ are used to compute a sampling count $n = \lfloor \|\tilde{q}_1 - \tilde{q}_2\|_2 / \bar{d} \rfloor$ (where $\bar{d}$ is the mean distance of the nearest point pairs), and intermediate points are uniformly interpolated as $s(\tilde{q}_1, \tilde{q}_2, a) = a \times \tilde{q}_1 + (1-a)\tilde{q}_2$. To avoid unreliable PnP solutions caused by multiple 3D points sharing the same 2D projection, at most one 3D point per 2D pixel is sampled in each RANSAC iteration.

2. **Hierarchical Continuous Coordinate Encoding (HCCE)**: Each coordinate component $x, y, z$ is independently encoded into multi-level continuous codes. The first level is $Cx_{1,k} = x_k$; higher levels are generated via a mirroring operation: when $x_k < 0.5$, the previous level code is copied; when $x_k \geq 0.5$, the previous level code is mirrored. The key advantage is the elimination of stripe boundaries present in binary encoding, making the representation easier for the network to learn. At decoding time, continuous codes are first converted to binary codes as $Bx_{i,k} = g(Cx_{i,k})$ or $1 - g(Cx_{i,k})$ (depending on whether the previous level was mirrored), and coordinates are recovered as $x_k \approx \sum_{i=1}^{8} 2^{-k} \times Bx_{i,k}$.

3. **Multi-histogram-based hierarchical learning**: Error histograms are computed separately for each coordinate component, recording the proportion of incorrect predictions at each level $r_{f,x,i}$. Loss weights are then computed as $h_{f,x,i} = \exp(\sigma \cdot \min\{r_{f,x,i}, 0.5 - r_{f,x,i}\})$ and normalized, enabling progressive learning from coarse to fine levels. Compared to ZebraPose's single-histogram approach, the multi-histogram strategy yields more stable training.

### Loss & Training

Total loss $L = L_M + \gamma \times (L_{xyz}^{Front} + L_{xyz}^{Back})$:
- **Mask loss** $L_M$: L1 norm between the predicted mask and ground truth.
- **Hierarchical loss**: Computed separately for the front and back surfaces. Taking the front surface $x$-component as an example: $L_x^{Front} = \sum_{i=1}^{8} (w_{f,x,i} \cdot \sum_{j=1}^{n} \|Cx_{f,i,j} - \widetilde{Cx}_{f,i,j}\|_1)$.

The network outputs 49 channels (front surface $3\times8$ + back surface $3\times8$ + mask $1$). ResNet34 (ablation) or EfficientNet-B4 (comparison experiments) is used as the backbone, with input/output resolution of $256\times256$ / $128\times128$. Each object is trained separately for approximately 24 hours on an NVIDIA RTX 4090.

## Key Experimental Results

### Main Results

Comparison with state-of-the-art methods on the seven core BOP benchmark datasets (BOP Score, %):

| Method | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | Core Avg. |
|--------|------|--------|-------|--------|-------|----|-------|-----------|
| GPose | 69.9 | 79.9 | 83.1 | 62.6 | 46.0 | 87.6 | 80.9 | 72.9 |
| ZebraPose | 72.9 | 82.1 | 85.0 | 59.2 | 50.4 | 92.2 | 82.8 | 74.9 |
| **Ours (RGB)** | **75.5** | **85.6** | **86.9** | **63.5** | **54.2** | 91.9 | **83.9** | **77.3** |
| GDRNPP (RGB-D) | 79.2 | 87.2 | 93.6 | 70.2 | 58.8 | 90.9 | 83.4 | 80.5 |
| **Ours (RGB→RGB-D)** | **80.5** | **87.9** | 94.4 | **72.4** | **73.4** | **93.1** | **91.1** | **84.7** |

In RGB mode, the proposed method outperforms ZebraPose by 2.4% in Core average BOP Score. When trained on RGB and tested with RGB-D, it surpasses GDRNPP by 4.2%.

### Ablation Study

Ablation of different encoding schemes on the IC-BIN dataset (ADD(-S) AR%):

| Method | AR of ADD(-S) | AR of ADD-S | AUC ADD(-S) | AUC ADD-S |
|--------|---------------|-------------|-------------|-----------|
| ZebraPose (surface encoding) | 55.85 | 61.82 | 72.91 | 76.40 |
| HBCE+f (coordinate encoding) | 56.82 | 63.50 | 73.15 | 76.56 |
| HCCE+f(h0) no weight adjustment | 61.35 | 67.00 | 76.35 | 79.82 |
| HCCE+f(h1) single histogram | 60.44 | 65.23 | 74.92 | 78.68 |
| **HCCE+f(h3) multi-histogram** | **61.95** | **68.30** | **77.67** | **81.11** |

Ablation of front/back surface information (average BOP AP% on LM-O/TUD-L/IC-BIN):

| Configuration | Avg. AP |
|---------------|---------|
| Front only (f) | 82.2 |
| Back only (b) | 81.3 |
| Front + Back (bf) | 82.6 |
| **Front + Back + Sampling (bfu)** | **83.3** |

### Key Findings

- Encoding coordinate components (HBCE) is more effective than encoding surface regions (ZebraPose), yielding a +0.97% improvement in ADD(-S).
- HCCE further improves over HBCE by 5.13%, demonstrating that continuous codes are significantly easier for the network to learn than binary codes.
- Multi-histogram weight adjustment improves over no adjustment by 0.6%, whereas the single-histogram variant decreases performance by 0.91%.
- Jointly exploiting front and back surfaces with dense sampling (bfu) improves over front surface only by 1.1% AP.
- The proposed method also outperforms ZebraPose by 3.7% on 2D segmentation, indicating that front-and-back surface prediction aids more precise object localization.

## Highlights & Insights

- **Front-and-back surface prediction is a simple yet effective idea**: All prior methods consider only the front surface, whereas geometrically, back surface information genuinely provides a more constrained solution space for PnP.
- **HCCE eliminates the stripe boundary problem of binary codes**, representing a significant improvement over the ZebraPose encoding scheme.
- The multi-histogram hierarchical learning strategy stabilizes training, with loss weight peaks gradually shifting from coarse to fine levels, realizing a natural curriculum learning effect.
- The method is architecturally general and compatible with different backbones (ResNet34 / EfficientNet-B4).

## Limitations & Future Work

- A separate network must be trained per object (approximately 24 hours each), making the approach unsuitable for large-scale scenarios with hundreds of objects.
- The method targets only seen objects and does not address pose estimation for unseen objects.
- Front and back surface definitions rely on depth-test-based rendering (GL_LESS / GL_GREATER), which may not apply well to transparent objects or objects with severe self-occlusion.
- Inference takes approximately 30 ms per object, which may become a bottleneck in multi-object scenarios.

## Related Work & Insights

- The hierarchical binary encoding proposed by ZebraPose is an important predecessor; HCCE presented in this paper is its natural evolution.
- StereoPose also predicts front and back surfaces but is limited to transparent objects using stereo images; this work generalizes the idea to general objects with monocular RGB.
- Future work could incorporate features from foundation models (e.g., SAM / DINOv2) to replace per-object training.
- The ultra-dense correspondence paradigm is broadly applicable to scenarios requiring precise poses, such as 6-DoF grasping and robotic manipulation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The two contributions—front-and-back surface prediction and HCCE encoding—are mutually complementary and naturally motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across all seven BOP core datasets with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic with complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐ Substantial empirical gains with a generalizable technical pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)
- [\[ICCV 2025\] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation](perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](../../NeurIPS2025/human_understanding/pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](../../NeurIPS2025/human_understanding/raptr_radar-based_3d_pose_estimation_using_transformer.md)

</div>

<!-- RELATED:END -->
