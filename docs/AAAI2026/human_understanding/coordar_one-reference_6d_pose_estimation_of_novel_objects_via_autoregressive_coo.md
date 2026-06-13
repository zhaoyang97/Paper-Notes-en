---
title: >-
  [Paper Note] CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation
description: >-
  [AAAI 2026][Human Understanding][6D pose estimation] This paper proposes CoordAR, which formulates 3D-3D correspondence estimation in single-reference-view 6D pose estimation as an autoregressive generation problem over…
tags:
  - "AAAI 2026"
  - "Human Understanding"
  - "6D pose estimation"
  - "autoregressive"
  - "coordinate map"
  - "VQ-VAE"
  - "single reference view"
date: 2026-05-08
content_hash: ad5a29ee8ff469f6
---

# CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation

**Conference**: AAAI 2026
**arXiv**: [2511.12919](https://arxiv.org/abs/2511.12919)  
**Code**: [Project Page](https://sjtu-visys-team.github.io/CoordAR)  
**Area**: Human Understanding
**Keywords**: 6D pose estimation, autoregressive, coordinate map, VQ-VAE, single reference view

## TL;DR

This paper proposes CoordAR, which formulates 3D-3D correspondence estimation in single-reference-view 6D pose estimation as an autoregressive generation problem over discrete tokens. Through coordinate map tokenization, modality-decoupled encoding, and an autoregressive Transformer decoder, CoordAR substantially outperforms existing single-view methods on multiple benchmarks and demonstrates strong robustness to challenging scenarios such as symmetry and occlusion.

## Background & Motivation

6D object pose estimation (recovering the rotation and translation of a rigid object) is a fundamental task in CV and robotics, with broad applications in AR, robotic manipulation, and industrial automation. Core challenges:

**Reliance on 3D models**: Instance-level methods require training a dedicated network per object; category-level methods are restricted to known categories; and category-agnostic methods still require CAD models.

**One-reference setting**: Only a single reference image with a known pose is used to estimate the object pose in a novel view — more realistic but extremely challenging.

Limitations of existing single-reference methods:
- **One2Any** uses a convolutional decoder to regress continuous coordinate maps; the local receptive field of convolutions limits global consistency.
- Continuous regression cannot handle **inherent ambiguities** caused by symmetry and occlusion — symmetric objects admit multiple valid solutions, and direct regression produces erroneous averaged results.

## Method

### Overall Architecture

CoordAR consists of three main stages:

1. **Modality-decoupled encoding**: Separately encodes reference/query RGB images and the reference object coordinate (ROC) map.
2. **Feature fusion**: Stacked fusion blocks inject reference information into query features.
3. **Autoregressive decoding**: Conditioned on fused features, discrete coordinate tokens are autoregressively generated and then de-tokenized into a pixel-level coordinate map.

The final rigid transformation is recovered from coordinate correspondences via the Umeyama algorithm.

### Key Designs

**1. Reference Object Coordinate (ROC) Map Representation**

The ROC map of the reference view $\mathbf{X}^R$ is obtained via depth back-projection. The query ROC map $\mathbf{X}^Q = \mathbf{S}\mathbf{T}_{RQ}\mathbf{\Pi}^{-1}(\mathcal{D}_Q)[\mathcal{M}_Q=1]$ encodes the relative transformation between the reference and query views. Both ROC maps represent 3D points in the reference object coordinate frame, providing pixel-level 3D-3D correspondences. The pose estimation problem is thus reformulated as: **estimating the query ROC map $\mathbf{X}^Q$**.

**2. Modality-Decoupled Encoding**

Unlike One2Any's role-specific encoding (which concatenates RGB and ROC maps as input to a shared encoder), CoordAR assigns independent encoders to different modalities:
- **Shared RGB encoder**: Processes RGB images from both reference and query views.
- **ROC encoder**: Dedicated to encoding the reference coordinate map.

This avoids mixing two modalities with vastly different structural patterns. In the cross-attention of fusion blocks, affinity is computed separately per modality to reduce the RGB-ROC domain gap.

**3. Coordinate Map Tokenization (VQ-VAE)**

A VQ-VAE is used to encode continuous coordinate maps into discrete token sequences $\{s_1, \ldots, s_{h \cdot w}\}$, where each token $s_* \in \mathcal{V}$ (pretrained codebook). Core advantages:

- Establishes a **probability distribution** over the discrete space; each patch position outputs a categorical distribution over the codebook.
- Symmetric objects can be represented via multimodal distributions with multiple valid solutions, rather than being forced into an erroneous averaged regression result.
- Occluded regions can express uncertainty through high-entropy distributions.

**4. Autoregressive Transformer Decoder**

Token generation is modeled as a masked autoregressive process:

$$p(s) = \prod_{k=1}^K p(S_k | S_{<k}, C_{\mathbf{F}})$$

where $S_k$ is the set of tokens generated at step $k$ and $C_{\mathbf{F}}$ denotes position-aligned conditioning features. The training objective is to minimize the negative log-likelihood:

$$\mathcal{L}_{\text{AR}} = -\sum_k^K [\log(p(S_k | S_{<k}, C_{\mathbf{F}}))]$$

The decoder consists of multiple self-attention blocks that take learnable mask tokens as input and predict subsequent tokens conditioned on previously generated tokens and position-aligned conditioning features. At inference, multi-step generation (64/16/4/1 steps) is supported, allowing flexible trade-offs between accuracy and speed.

**5. Pose Recovery**

The generated ROC map $\hat{\mathbf{X}}^Q$ is inverse-normalized to obtain 3D points $\hat{\mathbf{P}}_R^Q$ in the reference coordinate frame. These are pixel-aligned with 3D points $\mathbf{P}_Q^Q$ back-projected from the query depth map, and the optimal rigid transformation is solved via the Umeyama algorithm.

### Loss & Training

- Primary loss: autoregressive negative log-likelihood $\mathcal{L}_{\text{AR}}$
- VQ-VAE tokenizer is pretrained and frozen during training
- Trained on the FoundationPose dataset and a subset of OO3D-9D
- Ground-truth masks are assumed to be provided

## Key Experimental Results

### Main Results

**Table 1: Real275 & Toyota-Light (Generalization to Real Novel Objects)**

| Method | Modality | # Refs | Real275 AR | Real275 ADD(-S) | Toyota AR | Toyota ADD(-S) |
|---|---|---|---|---|---|---|
| One2Any | RGBD | 1 | 54.9 | 41.0 | 42.0 | 34.6 |
| **CoordAR** | RGBD | 1 | **71.0** | **82.2** | **62.5** | **82.6** |

AR gains of +16.1/+20.5 and ADD(-S) gains of +41.2/+48.0 represent substantial improvements.

**Table 3: LINEMOD (Large Viewpoint Variation)**

| Method | # Refs | Mean ADD(-S) |
|---|---|---|
| One2Any | 1 | 52.6 |
| **CoordAR** | 1 | **75.0** |
| OnePose++ | 200 | 76.9 |

Using only 1 reference image, CoordAR nearly matches OnePose++, which uses 200 reference images.

### Ablation Study

**Table 4: Ablation of Key Design Choices (LINEMOD)**

| Component / Variant | ADD(-S) | AR |
|---|---|---|
| Convolutional decoder | 70.9 | 59.7 |
| Autoregressive decoder | **73.1** | **61.6** |
| w/o autoregression (parallel tokens) | 60.7 | 52.1 |
| w/o tokenization (continuous regression) | 56.4 | 48.7 |
| Role-specific encoding | 61.6 | 49.8 |
| Modality-decoupled encoding | **73.6** | **61.9** |

All three core designs (tokenization, autoregressive generation, modality-decoupled encoding) contribute significantly; modality-decoupled encoding has the largest impact (+12.0 AR).

**Table 5: Inference Speed–Accuracy Trade-off**

| Steps | Time (s) | ADD(-S) |
|---|---|---|
| 64 | 0.63 | 75.0 |
| 16 | 0.25 | 75.0 |
| 4 | 0.13 | 74.7 |
| 1 | 0.10 | 74.3 |

Single-step inference incurs only a 0.7% accuracy drop while achieving near-real-time speed (0.10 s/frame).

### Key Findings

- Coordinate maps generated autoregressively exhibit substantially better spatial consistency than those generated in parallel (non-autoregressive methods show color discontinuities in visualizations).
- Discrete tokenization is particularly effective for symmetric objects (bowls) and occluded objects (cups), successfully avoiding ambiguity averaging.
- On the YCB-V occlusion benchmark, ADD-S AUC significantly outperforms One2Any; ADD AUC is slightly lower due to evaluation bias on texture-symmetric objects.

## Highlights & Insights

1. **Discretization + Autoregression = Resolving Ambiguity**: The design intuition of converting continuous coordinate regression into discrete probabilistic prediction is insightful and directly addresses the core challenge of symmetry and occlusion.
2. **Modality Decoupling > Role Decoupling**: Assigning encoders based on data modality (RGB vs. coordinates) rather than functional role (reference vs. query) yields substantially larger performance gains.
3. **Flexible Speed–Accuracy Trade-off**: The 1-to-64-step inference range incurs only a 0.7% accuracy gap, with single-step inference approaching real-time.
4. **Minimal Pose Recovery**: The Umeyama algorithm directly solves pose from dense correspondences without iterative optimization.

## Limitations & Future Work

1. The method relies on ground-truth mask inputs; practical deployment requires integration with a segmentation network.
2. The VQ-VAE tokenizer requires pretraining, and the choice of codebook size and patch resolution affects the performance ceiling.
3. On objects with rich texture but geometric symmetry (e.g., soup cans in YCB-V), the ADD metric is affected by evaluation protocol bias.
4. Multi-step autoregressive generation is slower than One2Any's parallel approach (0.63 s vs. 0.09 s).

## Related Work & Insights

- **One2Any**: The most direct baseline, using convolutional regression of continuous coordinate maps; CoordAR builds upon it by introducing tokenization and autoregressive generation.
- **VQ-VAE / MAGVIT**: Successful experience with discrete tokenization in image/video generation is cleverly transferred to the coordinate map domain.
- **Insights**: The autoregressive modeling of dense correspondences can be extended to optical flow estimation, scene flow, and related tasks; the conclusion on modality-decoupled encoding is worth referencing in broader multimodal fusion research.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of autoregressive coordinate map generation to 6D pose estimation; paradigm-level innovation.
- Technical Depth: ⭐⭐⭐⭐ — The combined design of tokenization, autoregression, and modality-decoupled encoding is well-integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, detailed ablations (5 groups), and speed analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is formally clear; ablation analysis is thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](../../ICCV2025/human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[AAAI 2026\] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)
- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](../../CVPR2026/human_understanding/next-scale_autoregressive_models_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->
