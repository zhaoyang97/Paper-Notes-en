---
title: >-
  [Paper Note] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching
description: >-
  [3D Vision] This paper systematically analyzes three key problems in monocular prior fusion—affine-invariant vs. absolute depth alignment, local optima in iterative updates, and noisy disparity interference—and proposes a Binary Local Ranking Map and a Global Registration Module. On SceneFlow→Middlebury/Booster generalization benchmarks, bad2 error is reduced by half or more with negligible additional computational cost.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 5029e8b6a3c9dae9
---

# Diving into the Fusion of Monocular Priors for Generalized Stereo Matching

## Paper Information

- **Conference**: ICCV 2025
- **arXiv**: 2505.14414
- **Code**: Not released (based on RAFT-Stereo architecture)
- **Area**: 3D Vision / Stereo Matching
- **Keywords**: stereo matching, monocular prior, vision foundation model, depth estimation, generalization

## TL;DR

This paper systematically analyzes three key problems in monocular prior fusion—affine-invariant vs. absolute depth alignment, local optima in iterative updates, and noisy disparity interference—and proposes a Binary Local Ranking Map and a Global Registration Module. On SceneFlow→Middlebury/Booster generalization benchmarks, bad2 error is reduced by half or more with negligible additional computational cost.

## Background & Motivation

Stereo matching generalizes poorly in ill-posed regions such as occlusions, textureless surfaces, and non-Lambertian materials (reflective/transparent). Incorporating monocular priors can correct matching errors in these regions, but a fundamental tension exists:

**Biased priors**: Monocular priors learned on small-scale stereo datasets suffer from domain bias and fail to generalize.

**Availability of unbiased priors**: Vision foundation models such as DepthAnything V2 can provide unbiased monocular priors, yet how to fuse them effectively remains an open question.

The authors conduct an in-depth analysis of the fusion process and identify three limiting factors:

- **Alignment gap**: Monocular depth is affine-invariant relative depth, which is fundamentally incompatible with the absolute disparity produced by stereo matching.
- **Local optima**: Over-confident disparity updates in iterative architectures cause the optimization to become trapped in local minima.
- **Noise interference**: Noisy disparity estimates from early iterations mislead direct depth fusion.

## Method

### Overall Architecture

The network consists of three modules:
1. **Monocular Encoder**: A frozen DepthAnything V2 that extracts unbiased monocular priors (depth maps + contextual features).
2. **Iterative Local Fusion**: A Binary Local Ranking Map guides disparity updates at each iteration.
3. **Global Fusion**: Monocular depth is registered to the refined disparity map.

### Key Design 1: Binary Local Ranking Map

Depth values are converted into local pairwise ordinal relations, unifying the representation of monocular depth and stereo disparity:

$$M_O(u,v) = \{\sigma(D(u', v') - D(u, v))\}$$

where $(u', v') \in \mathcal{N}_{(u,v)}$ and $\sigma$ denotes the sigmoid function.

Core advantages:
- **Robustness**: Converting absolute values to ordinal relations reduces sensitivity to outlier noise.
- **Unification**: Affine-invariant monocular depth and absolute disparity are naturally compatible in ordinal space.
- **Interpretability**: Visualization shows that the disparity ranking map progressively converges toward the monocular depth ranking map across iterations.

An LBP-like (Local Binary Pattern) convolutional block with fixed rather than learnable weights is used to compute the ranking maps (learnable weights are unreliable given limited training data). The monocular and disparity ranking maps are concatenated to predict a guidance map $G$ modeled as a Beta distribution.

Disparity updates are reweighted by the guidance map, with its influence gradually released over iterations:

$$\tilde{\Delta}_d = \Delta_d (1 + G \cdot r \cdot t / T)$$

$$D_d^t = D_d^{t-1} + \tilde{\Delta}_d$$

### Key Design 2: Global Fusion Module

The refined disparity map is modeled as a noisy registered version of monocular depth, and pixel-wise linear regression parameters are learned:

$$\tilde{D}_m = a \cdot D_m + b$$

$$a, b = \mathcal{F}(D_m, D_d^T)$$

where $\mathcal{F}$ is a convolutional network. The registered monocular depth and the refined disparity are fused via a confidence map:

$$D_f = c \cdot D_d^T + (1 - c) \cdot \tilde{D}_m$$

The confidence $c$ is jointly predicted from the cost volume, the GRU hidden state, and the guidance map from the final iteration.

### Loss & Training

$$\mathcal{L} = \sum_{t=1}^T \gamma^{T+2-t} \|D_d^t - D_G\|_1 + \gamma \|\tilde{D}_m - D_G\|_1 + \|D_f - D_G\|_1$$

Training proceeds in three stages: (1) train the main network without global fusion; (2) freeze the main network and train registration parameters; (3) freeze the main network and train the complete global fusion module.

## Key Experimental Results

### Main Results: SceneFlow → Real-World Generalization

| Method | Extra Data | Middlebury EPE | Middlebury bad2 | ETH3D EPE | ETH3D bad2 |
|--------|-----------|----------------|-----------------|-----------|------------|
| RAFTStereo | ✗ | 1.92 | 12.60 | 0.36 | 3.30 |
| IGEV | ✗ | 2.63 | 11.93 | 0.33 | 4.00 |
| Mocha-Stereo | ✗ | 2.66 | 10.18 | 0.28 | 3.47 |
| Selective-IGEV | ✗ | 2.59 | 11.79 | 0.33 | 4.05 |
| NerfStereo | ✓ | 1.42 | 9.67 | 0.29 | 2.94 |
| **Ours** | **✗** | **1.15** | **8.39** | **0.25** | **1.88** |

- On Middlebury, bad2 is reduced from 12.60 (RAFT-Stereo) to 8.39 (~33% improvement).
- On ETH3D, bad2 is reduced from 3.30 to 1.88 (~43% improvement).
- Without any extra data, the proposed method outperforms NerfStereo, which relies on additional NeRF-based data.

### Booster Dataset (Transparent/Reflective Regions)

| Method | All EPE | All bad2 | Trans bad2 | Trans bad5 | NonTrans bad2 |
|--------|---------|----------|-----------|------------|---------------|
| RAFTStereo | 4.18 | 17.64 | 67.69 | 47.40 | 13.13 |
| Mocha-Stereo | 3.88 | 16.82 | 66.44 | 45.73 | 12.31 |
| RAFT+ME | 2.40 | 11.44 | 64.84 | 43.95 | 6.96 |
| **Ours** | **2.26** | **11.02** | **59.83** | **38.44** | **6.98** |

Transparent region bad5 is reduced from ~47% to 38.4%, an improvement of approximately 10 percentage points.

### Ablation Study

| Configuration | Middlebury EPE | Middlebury bad2 |
|--------------|----------------|-----------------|
| Baseline (RAFT-Stereo) | 2.11±0.16 | 14.12±0.64 |
| Baseline w/o mono feature | 1.83±0.11 | 12.45±0.86 |
| + Monocular Encoder (ME) | 1.42±0.01 | 9.81±0.18 |
| + ME + Iterative Direct Fusion | 1.41±0.04 | 10.34±0.19 |
| + ME + Iterative Local Fusion (ILF) | 1.20±0.08 | 9.06±0.70 |
| + ME + ILF + Global Fusion (GF) | **1.15±0.01** | **8.35±0.04** |

**Key Findings**:
- Removing the biased monocular features in RAFT-Stereo actually improves performance (1.83 vs. 2.11), confirming that biased priors are harmful.
- The Monocular Encoder (ME) contributes the largest single improvement (EPE: 2.11→1.42).
- Direct Iterative Fusion (IDF) performs worse than no fusion (bad2: 10.34 vs. 9.81), validating the noise interference hypothesis.
- ILF (ranking map guidance) combined with GF (global registration) achieves the best overall performance.
- Fixed LBP-like weights outperform learnable convolutional weights, consistent with the hypothesis that learnable weights are unreliable under limited data.
- Runtime: Baseline 0.32s → Ours 0.40s, only a 25% increase.

## Highlights & Insights

1. **Rigorous problem analysis**: Rather than simply proposing a method, this work systematically identifies three bottlenecks in monocular prior fusion, with each component directly addressing a specific problem.
2. **Elegant design of the local ranking map**: Ordinal relations unify affine-invariant monocular depth and absolute disparity, simultaneously resolving the alignment gap and improving noise robustness.
3. **Fixed weights outperform learnable weights**: In generalization settings with limited data, hand-crafted priors (LBP-style convolutions) prove more reliable than learned parameters.
4. **Efficiency**: Despite incorporating a VFM (DepthAnything V2), freezing it and processing at an appropriate resolution keeps the additional runtime to only 0.08s.
5. **Comprehensive metric analysis**: The paper highlights implicit assumptions in prior evaluations (restricted disparity ranges, non-occluded regions only) and provides a more unified and fair comparison protocol.

## Limitations & Future Work

- The three-stage training pipeline is relatively complex, requiring sequential freezing of different modules.
- Performance depends on the quality of DepthAnything V2; fusion is ineffective if the monocular model fails in specific scenes.
- The linear registration assumption ($a \cdot D_m + b$) in global fusion may be insufficient for modeling nonlinear depth distributions.
- Improvements on KITTI are relatively modest, with larger gains concentrated on Middlebury/Booster.

## Related Work & Insights

- **Comparison with StereoAnywhere**: A concurrent work that also fuses VFM priors into stereo matching, but employs a different fusion mechanism.
- The registration approach (scale+shift alignment) is analogous to techniques used in MiDaS/ZoeDepth, but its application within iterative stereo matching is novel.
- The Binary Local Ranking Map draws inspiration from classical LBP texture descriptors to address depth ordering problems.
- The progressive guidance release schedule ($t/T$ annealing) is a generalizable design principle applicable to other noise-guided iterative optimization scenarios.

## Rating

⭐⭐⭐⭐⭐ — A methodologically rigorous work with strong analytical depth. The three-problem–three-solution correspondence is clearly articulated. Experiments are conducted with statistical rigor (mean ± std reported), achieving significant breakthroughs in the most challenging transparent/reflective regions with negligible additional computational cost. An important reference for generalization research in stereo matching.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)

</div>

<!-- RELATED:END -->
