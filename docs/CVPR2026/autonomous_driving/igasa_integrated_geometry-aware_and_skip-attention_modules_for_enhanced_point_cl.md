---
title: >-
  [Paper Note] IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration
description: >-
  [CVPR2026][Autonomous Driving][Point Cloud Registration] This paper proposes the IGASA framework, which employs a three-stage pipeline consisting of a Hierarchical Pyramid Architecture (HPA)…
tags:
  - "CVPR2026"
  - "Autonomous Driving"
  - "Point Cloud Registration"
  - "Geometry-Aware"
  - "Skip-Attention"
  - "Hierarchical Pyramid"
  - "Coarse-to-Fine Matching"
date: 2026-05-08
content_hash: b4e190845f87fb6f
---

# IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration

**Conference**: CVPR2026  
**arXiv**: [2603.12719](https://arxiv.org/abs/2603.12719)  
**Code**: [DongXu-Zhang/IGASA](https://github.com/DongXu-Zhang/IGASA)  
**Area**: Autonomous Driving  
**Keywords**: Point Cloud Registration, Geometry-Aware, Skip-Attention, Hierarchical Pyramid, Coarse-to-Fine Matching, Autonomous Driving

## TL;DR

This paper proposes the IGASA framework, which employs a three-stage pipeline consisting of a Hierarchical Pyramid Architecture (HPA), Hierarchical Cross-Layer Attention (HCLA), and Iterative Geometry-Aware Refinement (IGAR) to bridge the semantic gap across multi-scale features and dynamically suppress outliers, achieving state-of-the-art performance on four benchmarks: 3DMatch, 3DLoMatch, KITTI, and nuScenes.

## Background & Motivation

**Point Cloud Registration (PCR)** is a foundational task in 3D vision with direct applications in autonomous driving, robotic navigation, and environmental modeling. However, it remains challenging in real-world scenarios involving noise, occlusion, and large-scale transformations, where both accuracy and robustness are insufficient.

**Traditional ICP-based methods** rely on nearest-neighbor iterative minimization, making them sensitive to initialization and prone to local minima, with significant performance degradation under large misalignment or sparse data.

**CNN-based methods** (e.g., FCGF, D3Feat) are constrained by fixed receptive fields and struggle to model long-range dependencies. While **Transformer-based methods** (e.g., GeoTransformer, RoITr) can capture global context, fine geometric details are progressively diluted by aggressive downsampling as the network deepens—a phenomenon referred to as the "semantic gap."

**Conventional skip connections** typically employ naive fusion strategies such as concatenation or element-wise summation, which fail to properly calibrate the resolution mismatch between low-level geometric cues and high-level semantic embeddings, causing critical geometric details to be attenuated during fusion.

In the **coarse-to-fine paradigm**, the fine matching stage typically relies on RANSAC or hard-threshold pruning to remove outliers, which is computationally expensive and prone to discarding correct correspondences in low-overlap regions.

Consequently, a new framework is needed that simultaneously addresses the two key bottlenecks of multi-scale semantic alignment and robust outlier suppression.

## Method

### Overall Architecture

IGASA adopts a three-stage pipeline:

1. **HPA (Hierarchical Pyramid Architecture)**: Extracts features at three resolution levels (ordinary / minor / primary) using KPConv, with voxel sizes $dl_0$, $2 \cdot dl_0$, and $4 \cdot dl_0$ respectively. The convolution radius is dynamically scaled to cover receptive fields ranging from local to global, producing $F_{\text{multi}} = \{F_{\text{ordinary}}, F_{\text{minor}}, F_{\text{primary}}\}$.
2. **HCLA (Hierarchical Cross-Layer Attention)**: Serves as the core of coarse matching and comprises two sub-modules—SGIRA (Skip-Guided Inter-Resolution Attention) and SAIGA (Skip-Augmented Intrinsic Geometry Attention)—which explicitly align global semantics with local geometry to output $F_{\text{minor}}^{++}$. Geometric consistency top-$k$ selection then generates the coarse correspondence set $\widetilde{C}^{(1)}$.
3. **IGAR (Iterative Geometry-Aware Refinement)**: During the fine matching stage, alternating optimization is performed via dynamic geometric consistency weighting, weighted centroid alignment, and SVD decomposition over $N=5$ iterations to output the high-precision pose $T^* = [R^*, t^*]$.

### Key Designs

**SGIRA Module**: Uses global semantic features from the primary level as Query/Key to guide weighted fusion of high-resolution features from the minor level. The attention scores integrate three components:

- Semantic similarity $S_{ij} = \frac{Q_i K_j^T}{\sqrt{d_a}}$
- Geometric distance compensation $R_{ij} = -\frac{\|P_i - M_j\|^2}{\sigma^2}$
- Skip residual $F_{\text{minor}}^{++} = F_{\text{minor}}^{+} + \gamma \cdot \text{SkipResidual}(F_{\text{minor}}^{+}, F_{\text{skip}})$

Fusion is realized through a Gated Fusion Mechanism: dual-branch convolution → adaptive gating weights → residual adjustment → weighted aggregation.

**SAIGA Module**: Applies self-attention on the SGIRA output $F_{\text{minor}}^{+}$, integrating semantic similarity $S_{\text{geo},ij}$ with learnable geometric distance weights $R_{\text{geo},ij} = -\alpha \|M_i - M_j\|^2$, and introducing a skip attention bias $\theta \cdot A_{\text{skip}}$. The output $F_{\text{minor}}^{++}$ is obtained by aggregating the value matrix after Softmax normalization.

**IGAR Module**: At each iteration, correspondence weights are dynamically updated as $w_{ij}^{(k)} = \exp\bigl(-\frac{\|p_{\text{tar}} - (R^{(k)} p_{\text{src}} + t^{(k)})\|^2}{\sigma^2}\bigr) \times \mathbb{I}[\cdot < \tau]$. Weighted centroids, a weighted cross-covariance matrix, and SVD decomposition are then used to solve for the optimal $R^*, t^*$, constituting a soft-suppression rather than hard-pruning strategy for outlier handling.

### Loss & Training

The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mat}} + \mathcal{L}_{\text{key}} + \mathcal{L}_{\text{den}}$, supervising three levels:

| Loss Term | Components | Role |
|---|---|---|
| $\mathcal{L}_{\text{mat}}$ | Hierarchical matching probability loss $\mathcal{L}_p$ + weighted cross-entropy $\mathcal{L}_c$ | Supervises coarse matching probabilities |
| $\mathcal{L}_{\text{key}}$ | InfoNCE descriptor loss $\mathcal{L}_f$ + keypoint position loss $\mathcal{L}_k$ + confidence BCE $\mathcal{L}_i$ | Supervises keypoint matching |
| $\mathcal{L}_{\text{den}}$ | Translation loss $\mathcal{L}_t$ + rotation orthogonality constraint $\mathcal{L}_r$ | Supervises global pose estimation |

## Key Experimental Results

### Indoor Benchmarks: 3DMatch & 3DLoMatch

| Method | 3DMatch RR(%) | 3DMatch IR(%) | 3DLoMatch RR(%) | 3DLoMatch IR(%) |
|---|---|---|---|---|
| GeoTransformer | 92.0 | 71.9 | 75.5 | 43.5 |
| RoITr | 91.9 | 82.6 | 74.7 | 54.3 |
| SIRA-PCR | 93.6 | 70.8 | 73.5 | 43.3 |
| **IGASA** | **94.6** | **87.9** | **76.5** | **61.6** |

- IGASA achieves the highest Registration Recall across all sampling rates (94.6%→94.3%), with minimal degradation as the number of samples decreases.
- The Inlier Ratio of 87.9% significantly outperforms RoITr (+5.3%) and SIRA-PCR (+17.1%).

### Outdoor Benchmarks: KITTI & nuScenes

| Method | KITTI RTE(cm) | KITTI RRE(°) | KITTI RR(%) | nuScenes RTE(m) | nuScenes RR(%) |
|---|---|---|---|---|---|
| GeoTransformer | 6.8 | 0.24 | 99.8 | - | - |
| HRegNet | 12 | 0.29 | 99.7 | 0.18 | 99.9 |
| **IGASA** | **4.6** | **0.24** | **100.0** | **0.12** | **99.9** |

- IGASA achieves a 100.0% registration success rate on KITTI with an RTE of only 4.6 cm, the lowest among all compared methods.
- On nuScenes, RTE = 0.12 m and RRE = 0.21°, both state-of-the-art.

### Ablation Study

| HPA | HCLA | IGAR | 3DMatch RR(%) | 3DMatch IR(%) |
|---|---|---|---|---|
| ✓ | - | - | 91.3 | 80.2 |
| ✓ | ✓ | - | 93.2 | 83.7 |
| ✓ | - | ✓ | 92.8 | 81.9 |
| ✓ | ✓ | ✓ | **94.6** | **87.9** |

**Key Findings**:

- The HCLA module contributes the largest RR improvement (+1.9%), validating the importance of cross-layer semantic alignment.
- The IGAR module contributes the most to IR improvement (83.7% → 87.9%), demonstrating that dynamic weight iteration effectively suppresses outliers.
- Joint training with all three loss terms is essential: using any single loss alone yields IR as low as 71–75%, while their combination achieves 87.9%.
- SGIRA and SAIGA exhibit synergistic effects: used independently, FMR is 96.2% and 96.7% respectively; combined, it reaches 98.2%.
- Inference speed is 2.763 s/frame, comparable to GeoTransformer (2.701 s) and CoFiNet (2.660 s).

## Highlights & Insights

- **Skip-Attention as a replacement for naive skip connections**: SGIRA and SAIGA provide two levels of attention to bridge the multi-scale semantic gap, rather than resorting to simple concatenation or summation.
- **Soft suppression instead of hard pruning**: IGAR replaces RANSAC with a combination of dynamic geometric consistency weights and indicator functions, avoiding high computational overhead while reducing false rejection of correct correspondences in low-overlap regions.
- **Comprehensive validation across four datasets**: State-of-the-art results on both indoor (3DMatch/3DLoMatch) and outdoor (KITTI/nuScenes) benchmarks, with a notable 100% RR on KITTI.
- **Clean modular design**: The three stages—HPA → HCLA → IGAR—each serve a distinct role, with ablation studies thoroughly validating the necessity of each component.

## Limitations & Future Work

- **FMR on 3DLoMatch is not optimal**: In low-overlap scenarios (10%–30%), the Feature Matching Recall (82.1%) falls below RoITr (89.6%) and GeoTransformer (88.3%), indicating that descriptor robustness under extremely low overlap remains improvable.
- **Slight increase in inference latency**: Approximately 0.1 s slower than CoFiNet, which may become a bottleneck in latency-critical applications.
- **IGAR iteration count is a manual hyperparameter**: $N=5$ is determined empirically, with no adaptive termination mechanism.
- **Evaluation limited to rigid registration**: Applicability to non-rigid or dynamic scenes has not been explored.
- **Training resource**: A single RTX 3090 was used; scalability to large-scale continual training has not been discussed.

## Related Work & Insights

- **Traditional methods**: ICP and its variants (sensitive to initialization, prone to local optima).
- **CNN-based features**: FCGF, D3Feat (fixed receptive fields, limited long-range dependency modeling).
- **Transformer-based methods**: GeoTransformer, RoITr, SIRA-PCR (strong global context but loss of fine-grained details).
- **Coarse-to-fine frameworks**: CoFiNet, PYRF-PCR (multi-scale fusion but reliance on RANSAC for fine matching).
- **Skip connections**: U-Net-style concatenation/summation (naive fusion leading to semantic gaps).
- The core distinction of IGASA lies in replacing naive skip fusion with attention mechanisms and substituting hard-threshold pruning with soft weighting.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of replacing naive skip connections with attention-based fusion is well-motivated; the soft-suppression mechanism in IGAR is soundly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, detailed ablations (modules / sub-modules / losses / efficiency), and qualitative visualizations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations; minor issues include symbol redundancy and an overly lengthy Related Work section.
- Value: ⭐⭐⭐⭐ — State-of-the-art on both indoor and outdoor benchmarks; 100% RR on KITTI demonstrates strong practical value; FMR on 3DLoMatch is not optimal.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](points-to-3d_structure-aware_3d_generation_with_point_cloud_priors.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)
- [\[CVPR 2026\] DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance](drivergaze360_omnidirectional_driver_attention_with_object-level_guidance.md)
- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](../../AAAI2026/autonomous_driving/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[CVPR 2026\] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
