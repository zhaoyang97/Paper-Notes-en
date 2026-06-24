---
title: >-
  [Paper Note] Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training
description: >-
  [ECCV 2024][3D Vision][Self-Supervised Monocular Depth Estimation] Proposed the SCAT framework, which reduces the sensitivity of UNet skip connections to perturbations via the Scale Depth Network (SDN) and introduces Conflict Gradient Surgery (CGS) to resolve the dual optimization conflict caused by adversarial augmentation, successfully applying adversarial data augmentation to self-supervised monocular depth estimation for the first time to enhance cross-domain generalizati…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Self-Supervised Monocular Depth Estimation"
  - "Domain Generalization"
  - "Adversarial Training"
  - "Gradient Conflict"
  - "UNet"
date: 2026-05-08
content_hash: f30d864477bb9f40
---

# Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training

**Conference**: ECCV 2024  
**arXiv**: [2411.02149](https://arxiv.org/abs/2411.02149)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Self-Supervised Monocular Depth Estimation, Domain Generalization, Adversarial Training, Gradient Conflict, UNet

## TL;DR

Proposed the SCAT framework, which reduces the sensitivity of UNet skip connections to perturbations via the Scale Depth Network (SDN) and introduces Conflict Gradient Surgery (CGS) to resolve the dual optimization conflict caused by adversarial augmentation, successfully applying adversarial data augmentation to self-supervised monocular depth estimation for the first time to enhance cross-domain generalization.

## Background & Motivation

**Background**: Self-supervised monocular depth estimation (MDE) learns depth through image reconstruction losses without requiring labeled data, and has achieved significant progress in driving scenes such as KITTI. Representative methods include MonoDepth2, CADepth, and MonoVit.

**Limitations of Prior Work**: A large number of domain shifts (e.g., foggy, rainy, nighttime) exist in the real world, causing trained models to suffer from a severe lack of generalization capability in unseen scenes. Existing offline data augmentation methods (such as Robust-Depth) require a preset target distribution and cannot cope with diverse real-world scenes.

**Key Challenge**: Adversarial Data Augmentation (ADA) has been proven effective in enhancing generalization in supervised tasks, but directly applying it to self-supervised MDE leads to severe performance degradation or even training collapse. The root of this conflict has not been fully analyzed.

**Goal**: To analyze the reasons why adversarial augmentation fails in self-supervised MDE, and to design a general adversarial training framework to make it stable and effective.

**Key Insight**: Approaching from two angles—(i) the intrinsic sensitivity of UNet Long Skip Connections (LSCs) to noise; and (ii) the conflict in gradient directions (dual optimization conflict) caused by adversarial augmentation.

**Core Idea**: Achieving stable adversarial training by scaling skip connection coefficients to suppress perturbation amplification and utilizing sequential conflict gradient surgery.

## Method

### Overall Architecture

SCAT (Stabilized Conflict-optimization Adversarial Training) is a model-agnostic adversarial training framework containing three core components:

- **Adversarial Noise Generator** $g_\phi$: learns to generate noise $\delta = g_\phi(z)$ that maximizes confusion for the depth network.
- **Scale Depth Network (SDN)**: adjusts the scaling coefficients $\kappa_i$ of the LSCs in UNet to stabilize training.
- **Conflict Gradient Surgery (CGS)**: samples multiple generators from a historical adversarial generator buffer to progressively merge adversarial gradients.

Training is formulated as a min-max game:

$$\min_\theta \max_\phi \mathbb{E}_{\tilde{I}_{t'}, I_t} \mathbb{E}_{\delta \sim p_\phi(\delta)} \left[ \mathcal{L}(f_\theta(\tilde{I}_{t'}), I_t) \right]$$

The key design is that adversarial augmentation is only applied to the input images, while the reconstruction target remains the unaugmented original frame $I_t$ to avoid misleading the training.

### Key Designs

1. **Scale Depth Network (SDN)**: Standard UNet LSC is formulated as $f_i(x) = b_{i+1} \circ [a_{i+1} \circ x + f_{i+1}(a_{i+1} \circ x)]$. SDN introduces a scaling coefficient $\kappa_i$:

$$f_i(x) = b_{i+1} \circ [\kappa_{i+1} \cdot a_{i+1} \circ x + f_{i+1}(a_{i+1} \circ x)]$$

Theoretical analysis proves that for a perturbation $\epsilon_\delta$, the upper bound of the depth network's output error is:

$$\|f_\theta(I_t^{\epsilon_\delta}) - f_\theta(I_t)\|_2 \leq \epsilon_\delta \left[\sum_{i=1}^{N} \kappa_i M_0^i + c_0\right]$$

The upper bound for standard UNet ($\kappa_i = 1$) is $\mathcal{O}(NM_0^N)$, which grows sharply when $N$ is large. By setting $\kappa < 1$ (default 0.7), perturbation sensitivity can be effectively reduced while maintaining depth estimation accuracy.

Design Motivation: Although skip connections fuse multi-scale features and preserve low-level details, they also provide direct transmission channels for adversarial noise. The scaling coefficient suppresses this amplification effect.

2. **Conflict Gradient Surgery (CGS)**: Define the angle $\theta_i$ between the original data gradient $g_i$ and the adversarial data gradient $g_{mix_i}$. Gradient conflict occurs when $\cos\theta_{ij} < 0$. The core objective of CGS is:

$$\mathbb{E}[\cos(\theta)] > 0, \quad \cos(\theta_i) = \frac{g_i \cdot g_{mix_i}}{|g_i| |g_{mix_i}|}$$

Implementation details: Maintain a historical adversarial generator buffer $\mathcal{B}$. In each training iteration, $j$ historical generators are randomly sampled to perform progressive gradient fusion using diverse adversarial examples generated by them, instead of relying solely on the current strongest adversary.

Design Motivation: A single adversarial generator continuously reinforced leads to over-regularization, causing the adversarial gradient to align in the opposite direction of the original gradient. By mixing generators from multiple historical stages, the adversarial intensity is diluted, and the cosine distribution of gradients shifts from negative skew to positive skew.

3. **Adversarial Training Pipeline**: Each epoch contains: (1) sampling historical generators from the buffer to generate adversarial noise $\delta^{1:j}$; (2) SDN processing clean and adversarial images separately to generate depth maps; (3) constraining both branches via reprojection loss using the clean target $I_t$; (4) saving the current generator to the buffer.

### Loss & Training

The overall loss consists of two parts:

- **Self-Supervised Constraint** $\mathcal{L}_p$: reprojection errors for clean and adversarial images:

$$\mathcal{L}_p = \sum_{t'} pe(I_t, I_{t' \to t}) + pe(I_t, \tilde{I}_{t' \to t})$$

- **Adversarial Loss** $\mathcal{L}_{AD}$: used to optimize the adversarial generator:

$$\mathcal{L}_{AD} = \sum_{t'} pe(I_t, \tilde{I}_{t' \to t})$$

where $pe$ is the weighted combination of L1 and SSIM: $pe(I_a, I_b) = \frac{\alpha}{2}(1 - \text{SSIM}(I_a, I_b)) + (1 - \alpha)\|I_a - I_b\|_1$.

The perturbation size defaults to $\epsilon_m = 135.0$, and the default for $\kappa$ is 0.7.

## Key Experimental Results

### Main Results — KITTI-C Cross-Domain Generalization

| Method | mCE(%)↓ | mRR(%)↑ | Abs Rel↓ | δ<1.25↑ |
|------|---------|---------|----------|---------|
| MonoDepth2 | 101.04 | 84.08 | 0.248 | 0.698 |
| + SCAT | **86.32** | **90.13** | **0.165** | **0.762** |
| MonoVit | 80.54 | 88.98 | 0.191 | 0.771 |
| + SCAT | **62.74** | **95.38** | **0.127** | **0.846** |
| Robust-Depth | 55.72 | 96.46 | 0.121 | 0.854 |
| + SCAT | **53.37** | **98.19** | **0.117** | **0.861** |

### Main Results — KITTI In-Domain Accuracy Maintenance

| Method | Abs Rel↓ | RMSE↓ | δ<1.25↑ |
|------|----------|-------|---------|
| MonoDepth2 | 0.115 | 4.863 | 0.877 |
| + ADA (Direct) | 0.121 | 4.992 | 0.862 |
| + SCAT | **0.116** | **4.877** | **0.877** |
| MonoVit | 0.099 | 4.372 | 0.900 |
| + ADA (Direct) | 0.106 | 4.591 | 0.897 |
| + SCAT | **0.100** | **4.389** | **0.899** |

### Ablation Study

| CGS | SDN | KITTI Abs Rel↓ | KITTI δ<1.25↑ | KITTI-C Abs Rel↓ | KITTI-C δ<1.25↑ |
|-----|-----|----------------|---------------|------------------|-----------------|
| ✗ | ✗ | 0.121 | 0.862 | 0.193 | 0.734 |
| ✓ | ✗ | 0.117 | 0.869 | 0.174 | 0.752 |
| ✗ | ✓ | 0.118 | 0.865 | 0.179 | 0.748 |
| ✓ | ✓ | **0.116** | **0.877** | **0.165** | **0.762** |

### Key Findings

- Direct ADA leads to performance degradation on all KITTI in-domain baselines, whereas SCAT almost fully maintains in-domain accuracy.
- The individual contributions of CGS and SDN are independent and complementary, achieving the best results when combined.
- $\kappa = 0.7$ is the optimal balance point; $\kappa = 1.0$ (standard UNet) exhibits the highest sensitivity.
- The perturbation size of $\epsilon_m = 135$ is optimal; setting it too large (180) slightly degrades generalization.
- SCAT is effective across 5 different baseline models, demonstrating its model-agnostic nature.

## Highlights & Insights

- Conducts the first systematic analysis of the two main reasons why adversarial training fails in self-supervised MDE (LSC sensitivity + gradient conflict), backed by in-depth theoretical analysis.
- The proposed SDN requires modifying only one hyperparameter $\kappa$ and is extremely lightweight.
- The gradient surgery strategy utilizing a historical buffer and progressive fusion outperforms simple gradient clipping or projection.
- Validated on realistic cross-domain scenarios such as NuScenes Night, Foggy Cityscapes, and DrivingStereo, demonstrating high practicality.

## Limitations & Future Work

- Only explored fixed $\kappa$ values, without attempting learnable adaptive scaling.
- The adversarial generator employs pixel-level perturbations, without considering semantic-aware local perturbations.
- Training cost increases due to adversarial training (dual-branch forward pass + generator optimization).
- The impact of LSCs was not systematically ablated on Transformer-based depth networks.

## Related Work & Insights

- **Robust-Depth**: A representative method of offline augmentation, upon which SCAT still yields consistent improvements.
- **PCGrad / GradNorm**: Classical methods for dealing with gradient conflicts. CGS draws inspiration from similar concepts but targets adversarial training scenarios.
- Inspiring ideas: Applying the scaling strategy of SDN to other encoder-decoder architectures that rely on skip connections.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Successfully introduces adversarial training into self-supervised MDE for the first time, backed by in-depth theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 5 baselines × 5 datasets, complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem analysis and well-motivated methods.
- **Value**: ⭐⭐⭐⭐ — A general framework with a sustained contribution to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)
- [\[ECCV 2024\] ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion](prodepth_boosting_self-supervised_multi-frame_monocular_depth_with_probabilistic.md)
- [\[ECCV 2024\] Camera Height Doesn't Change: Unsupervised Training for Metric Monocular Road-Scene Depth Estimation](camera_height_doesnapost_change_unsupervised_training_for_metric_monocular_road-.md)
- [\[ECCV 2024\] SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
