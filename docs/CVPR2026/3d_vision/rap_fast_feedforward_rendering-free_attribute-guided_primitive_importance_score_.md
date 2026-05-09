---
title: >-
  [Paper Note] RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes RAP, a rendering-free feedforward method for Gaussian primitive importance scoring. It extracts 15-dimensional features from intrinsic attributes and local neighborhood statistics, employs a lightweight MLP to predict importance scores, and generalizes to unseen scenes after a single training run.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - importance estimation
  - rendering-free inference
  - feedforward prediction
  - pruning
date: 2026-05-08
content_hash: 5ed9719dbfebea92
---

# RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing

**Conference**: CVPR 2026
**arXiv**: [2602.19753](https://arxiv.org/abs/2602.19753)
**Authors**: Kaifa Yang, Qi Yang, Yiling Xu, Zhu Li (Shanghai Jiao Tong University, UMKC)
**Code**: [GitHub](https://github.com/yyyykf/RAP)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, importance estimation, rendering-free inference, feedforward prediction, pruning

## TL;DR

This paper proposes RAP, a rendering-free feedforward method for Gaussian primitive importance scoring. It extracts 15-dimensional features from intrinsic attributes and local neighborhood statistics, employs a lightweight MLP to predict importance scores, and generalizes to unseen scenes after a single training run.

## Background & Motivation

Among the large number of primitives produced by 3DGS, contribution levels vary drastically. Estimating primitive importance is critical for pruning, compression, and transmission. Existing methods exhibit significant limitations:

- **Attribute-based heuristics** (e.g., opacity thresholding): overly simplistic, ignoring inter-primitive occlusion and interaction.
- **Rendering-based methods** (e.g., LightGaussian, EAGLES): rely on multi-view rendering, with runtime scaling linearly with the number of views; sensitive to view selection and requiring dedicated rasterizers.
- **Learning-based methods** (e.g., learnable masks): tightly coupled to specific reconstruction frameworks, requiring retraining upon scene modification.

## Core Problem

Can importance be predicted directly from the intrinsic attributes of Gaussian primitives, bypassing rendering entirely, to enable fast, plug-and-play, and generalizable importance estimation?

## Method

### 4.1 Importance-Aware Feature Extraction

A 15-dimensional feature vector is constructed for each Gaussian primitive.

**Raw feature computation** (8 dimensions):

$$\mathbf{F}_i^{raw} = \{d_i, A_i, s_{0,i}, s_{1,i}, s_{2,i}, V_i, o_i, C_i\}$$

- $d_i$: KNN mean distance — measures spatial isolation.
- $A_i$: color anisotropy — standard deviation of RGB computed over $M$ randomly sampled directions.
- $s_{0,i} \leq s_{1,i} \leq s_{2,i}$: sorted scales (rotation-invariant).
- $V_i = s_0 \times s_1 \times s_2$: Gaussian volume.
- $o_i$: opacity.
- $C_i$: DC color (zeroth-order SH mean).

**Feature normalization**:

$$f_i^{(G)} = \frac{f_i - \mu^{(G)}}{\sigma^{(G)}}, \quad f_i^{(L)} = \frac{f_i - \mu_i^{(L)}}{\sigma_i^{(L)}}$$

- Global z-score: provides scene-level reference.
- Local KNN z-score: emphasizes local contrast.
- DC color is normalized using local statistics only.
- Values are clipped to a percentile range and linearly rescaled to $[0,1]$.

**Final feature**: 7 global dimensions + 8 local dimensions = 15 dimensions, compact yet discriminative.

### 4.2 Learning Framework

A lightweight MLP (3 hidden layers with widths 32/32/16) maps the 15-dimensional features to importance scores $S_i \in [0,1]$.

**Tripartite loss function**:

**Rendering loss**: renders with softly reweighted opacity and scale to ensure high-importance primitives contribute substantially to the image:

$$\tilde{o}_i = o_i S_i, \quad \tilde{\mathbf{s}}_i = \mathbf{s}_i S_i$$

$$\mathcal{L}_{\text{render}} = (1-\lambda_{\text{dssim}}) \mathcal{L}_1 + \lambda_{\text{dssim}} \mathcal{L}_{\text{D-SSIM}}$$

**Pruning-aware loss**: prevents the trivial solution of assigning uniformly high scores to all primitives:

$$\mathcal{L}_{\text{prune}} = (\text{mean}(S_i) - S_{\text{target}})^2$$

**Distribution regularization**: maximizes the entropy of the score distribution to ensure smooth coverage of $[0,1]$:

$$\mathcal{L}_{\text{entropy}} = 1 - \text{EntropyNorm}(S)$$

A differentiable soft histogram with $B=250$ bins is used to approximate entropy.

**Training**: conducted on 10 scenes from DL3DV-10K for 15,000 iterations, sampling one random view from one scene per epoch. **No rendering is required at inference time.**

## Key Experimental Results

| Method | Mip-Outdoor BD-Rate | Mip-Indoor | Deep Blending | Tanks&Temples |
|--------|-------------------|------------|---------------|---------------|
| LightGS | -35.21% | -31.15% | -30.72% | -37.98% |
| MesonGS | -34.89% | -30.34% | -28.84% | -36.12% |
| EAGLES | -41.28% | -24.98% | -29.87% | -30.01% |
| PUP-3DGS | -22.54% | -8.70% | -19.46% | +7.06% |
| **RAP** | **-42.63%** | **-33.90%** | **-36.76%** | **-40.11%** |

*BD-Rate relative to opacity baseline; lower is better.*

| Dataset | Opacity | LightGS | MesonGS | C3DGS | PUP-3DGS | **RAP** |
|---------|---------|---------|---------|-------|----------|---------|
| Mip-Indoor computation time (s) | 1.27 | 22.71 | 14.84 | 15.96 | 21.22 | **5.72** |
| Tanks&Temples computation time (s) | 2.53 | 18.62 | 17.53 | 9.22 | 20.50 | **6.66** |

## Highlights & Insights

- **Rendering-free inference**: importance score computation requires no rendering after training, achieving 3–7× speedup over rendering-based methods.
- **Plug-and-play**: decoupled from specific reconstruction or compression frameworks, enabling direct integration into arbitrary 3DGS pipelines.
- **Strong generalizability**: trained on 10 scenes, yet performs consistently well across 13 scenes from Mip-NeRF 360, Deep Blending, and Tanks&Temples.
- **Elegant tripartite loss design**: rendering loss for fidelity, pruning loss to prevent trivial solutions, and entropy regularization to enforce distribution coverage — the three terms are mutually complementary.
- **Observation-driven feature design**: features are systematically derived from spatial, appearance, and scale cues, grounded in observations of attribute anomalies in redundant primitives.

## Limitations & Future Work

- Training is conducted on only 10 scenes; generalization to larger-scale or more diverse environments (e.g., city-scale scenes) remains unverified.
- Higher-order spherical harmonic coefficients are not incorporated into the 15-dimensional feature set.
- KNN distance computation introduces non-trivial overhead ($K=128$).
- Applicability to dynamic scenes or post-editing Gaussian representations remains to be evaluated.

## Related Work & Insights

- vs. **LightGaussian**: LightGS estimates importance via 2D projected area × absolute opacity and depends on rendering; RAP is attribute-driven and rendering-free.
- vs. **PUP-3DGS**: PUP analyzes importance via Hessian-based gradient information, yet underperforms even the opacity baseline on Tanks&Temples (+7.06% BD-Rate); RAP achieves consistently superior results.
- vs. **EAGLES/MesonGS**: these methods combine opacity and volume cues with competitive quality, but incur 3–7× higher computation time compared to RAP.
- vs. **Compact-3DGS**: learnable mask approaches are coupled to specific frameworks and require retraining; RAP is trained once and applied across scenes.

The idea of predicting importance directly from intrinsic attributes is generalizable to other 3D representations (e.g., NeRF voxels, point clouds). The "anti-trivial + distribution regularization" design pattern within the tripartite loss framework has broader applicability. The use of both local and global normalization in feature design offers a useful reference for handling cross-scene variation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Rendering-free importance prediction is a practically motivated and novel direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated across multiple datasets and tasks (post-hoc pruning, training-time pruning, and compression).
- Writing Quality: ⭐⭐⭐⭐ — Observation-driven feature design motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ — Offers tangible benefits for practical deployment of 3DGS systems.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction](fast_scenescript_fast_and_accurate_language-based_3d_scene_understanding_via_mul.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)

<!-- RELATED:END -->
