---
title: >-
  [Paper Note] Learning Single Index Models with Diffusion Priors
description: >-
  [ICML2025][Image Generation][Diffusion Models] An efficient method utilizing diffusion model priors to recover signals from nonlinear observations of Semi-parametric Single Index Models (SIM) is proposed. It requires only one round of unconditional sampling and partial inversion without knowing the link function, significantly outperforming existing methods on 1-bit and cubic measurements with minimal NFE.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Signal Recovery"
  - "Single Index Models"
  - "Nonlinear Measurements"
  - "Inverse Problems"
  - "Compressed Sensing"
date: 2026-05-08
content_hash: 833bb1ad60a3f18a
---

# Learning Single Index Models with Diffusion Priors

**Conference**: ICML2025  
**arXiv**: [2505.21135](https://arxiv.org/abs/2505.21135)  
**Code**: Pending  
**Area**: Diffusion Model Theory  
**Keywords**: Diffusion Models, Signal Recovery, Single Index Models, Nonlinear Measurements, Inverse Problems, Compressed Sensing

## TL;DR

An efficient method utilizing diffusion model priors to recover signals from nonlinear observations of Semi-parametric Single Index Models (SIM) is proposed. It requires only one round of unconditional sampling and partial inversion without knowing the link function, significantly outperforming existing methods on 1-bit and cubic measurements with minimal NFE.

## Background & Motivation

Traditional compressed sensing assumes a linear measurement model $\boldsymbol{y} = \mathbf{A}\boldsymbol{x}^* + \boldsymbol{e}$, but the measurement process is nonlinear in many practical problems. **Single Index Models (SIM)** represent one of the most popular nonlinear measurement models:

$$\boldsymbol{y} = f(\mathbf{A}\boldsymbol{x}^*)$$

where $f$ is an unknown and potentially discontinuous element-wise nonlinear link function. The objective is to reconstruct the signal $\boldsymbol{x}^*$ using only the measurement matrix $\mathbf{A}$ and observations $\boldsymbol{y}$, without knowledge of $f$.

Existing signal recovery work based on diffusion models (DMs) suffers from the following limitations:

- **Methods like DPS, DAPS**: Assume the link function $f$ is known and differentiable, making them unable to process discontinuous functions (such as $\text{sign}(\cdot)$).
- **QCS-SGM**: Restricted to quantized compressed sensing and suffers from extremely slow reconstruction speeds (requiring tens of thousands of NFEs).
- **DDRM, MCG, etc.**: Mainly target linear settings.

The core motivation of this paper is: Can an efficient diffusion model method be designed to solve signal recovery under SIM **without relying on knowledge of the link function**?

## Method

### Core Idea: Treating $\mathbf{A}^T\boldsymbol{y}/m$ as a Noisy Signal

The key observation of the paper stems from the following lemma: under mild conditions of SIM,

$$\left\|\frac{1}{m}\mathbf{A}^T\boldsymbol{y} - \mu\boldsymbol{x}^*\right\|_\infty \leq \frac{C'\sqrt{\log(2n)}}{\sqrt{m}}$$

where $\mu = \mathbb{E}[f(\boldsymbol{a}^T\boldsymbol{x}^*)\boldsymbol{a}^T\boldsymbol{x}^*]$. This indicates that $\mathbf{A}^T\boldsymbol{y}/m$ is essentially a noisy version of $\mu\boldsymbol{x}^*$, with the noise level proportional to $1/\sqrt{m}$.

### Comparison of Three Methods

The paper proposes three strategies, where the key difference lies in how they utilize the sampling $G$ and inversion $G^\dagger$ of the diffusion model:

| Method | Formula | Operation |
|------|------|------|
| **SIM-DMFIS** | $\hat{\boldsymbol{x}} = G \circ G^\dagger(\mathbf{A}^T\boldsymbol{y}/m)$ | Complete inversion from $\epsilon$ followed by complete sampling |
| **SIM-DMS** | $\hat{\boldsymbol{x}} = G_{t^*}(\alpha_{t^*}C_s'\mathbf{A}^T\boldsymbol{y}/m)$ | Partial sampling (denoising) starting only from $t^*$ |
| **SIM-DMIS** ⭐ | $\hat{\boldsymbol{x}} = G \circ G^\dagger_{t^*}(\alpha_{t^*}C_s'\mathbf{A}^T\boldsymbol{y}/m)$ | Partial inversion from $t^*$ to $T$, followed by complete sampling |

### Determination of the Intermediate Step $t^*$

By matching the noise level of $\mathbf{A}^T\boldsymbol{y}/m$ with the noise schedule of the diffusion forward process, the intermediate step $t^*$ is selected to satisfy:

$$\frac{\sigma_{t^*}}{\alpha_{t^*}} = \frac{C_s}{\sqrt{m}}$$

where $C_s$ is a tunable parameter. This is a theoretically-driven design: greater noise (smaller $m$) shifts the inversion starting point closer to $T$.

### Algorithmic Flow (SIM-DMIS)

1. **Input**: Measurement matrix $\mathbf{A}$, observations $\boldsymbol{y}$, data prediction network $\boldsymbol{x}_\theta$ of the pre-trained DM.
2. Calculate the intermediate step $t^*$ based on $C_s/\sqrt{m}$.
3. Construct the initial vector $\alpha_{t^*}C_s'\mathbf{A}^T\boldsymbol{y}/m$.
4. Execute partial inversion $G^\dagger_{t^*}$ from $t^*$ to $T$ (using the DM2M second-order inversion method).
5. Execute complete sampling $G$ from $T$ to $\epsilon$ (using DDIM sampling).
6. **Output**: Reconstructed signal $\hat{\boldsymbol{x}}$.

### Theoretical Analysis

| Theorem/Lemma | Content | Significance |
|-----------|------|------|
| Lemma 2 | $\|\mathbf{A}^T\boldsymbol{y}/m - \mu\boldsymbol{x}^*\|_\infty = O(\sqrt{\log n/m})$ | Establishes the noise level estimate to guide the selection of $t^*$ |
| Lemma 3 | Generator $G$ is $L$-Lipschitz continuous under Lipschitz conditions | Ensures that errors are not amplified by the sampling process |
| Theorem 3 | $\|\bar{\boldsymbol{x}}_\epsilon - G \circ G^\dagger_t(\bar{\boldsymbol{x}}_t)\|_2 = O(\sqrt{n}(h_{\max}^{k_2} + Lh_{\max}^{k_1}))$ | Error upper bound of SIM-DMIS, related to step size $h_{\max}$ and numerical orders $k_1, k_2$ |
| Assumption 1 | The data prediction network $\boldsymbol{x}_\theta(\cdot, t)$ is $L_t$-Lipschitz with respect to the first parameter | Standard assumption adopted by many theoretical works on DMs |

The theory demonstrates that utilizing high-order numerical methods ($k_1, k_2 \geq 2$) can significantly reduce reconstruction errors.

## Key Experimental Results

### FFHQ 256×256, 1-bit Measurements ($m = n/8$)

| Method | NFE | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|-----|--------|--------|---------|
| QCS-SGM | 11555 | 12.91 | 0.51 | 0.50 |
| DPS-N | 1000 | 11.14 | 0.37 | 0.69 |
| **SIM-DMS** | **50** | — | — | — |
| **SIM-DMIS** | **150** | **Best** | **Best** | **Best** |

### Key Findings

- SIM-DMIS outperforms QCS-SGM (which requires **11555 NFE**) with only **150 NFE**, achieving a **77x speedup**.
- In 1-bit measurements, SIM-DMIS remains superior even though DPS-N and DAPS-N exploit knowledge of the link function $f$.
- Partial inversion (SIM-DMIS) significantly outperforms complete inversion (SIM-DMFIS), validating the theoretical intuition of starting the inversion from the intermediate step $t^*$.
- Consistent performance is achieved across FFHQ and ImageNet (CIFAR-10 is shown in the Appendix).

## Highlights & Insights

1. **No Knowledge of Link Function Required**: This is the core advantage. In reality, the link functions of nonlinear measurement models are often unknown or non-differentiable; this method bypasses this limitation entirely.
2. **Theoretically-Driven Intermediate Step Selection**: By aligning the noise level of $\mathbf{A}^T\boldsymbol{y}/m$ with the diffusion noise schedule $\sigma_t/\alpha_t$ via Lemma 2, the inversion starting point $t^*$ is elegantly determined.
3. **Extremely High Computational Efficiency**: Requires only a single round of sampling + partial inversion (150 NFE), without iterative optimization or gradient computation.
4. **Counter-Intuitive Finding that Partial Inversion Outperforms Complete Inversion**: Executing complete inversion starting from $\epsilon$ incorrectly assumes that the input complies with the data distribution $q_0$, whereas starting from $t^*$ matched with the noise level is more reasonable.
5. **A Unified Framework** to handle different nonlinear measurements (1-bit, cubic, quantization, etc.) without requiring individual designs for each measurement type.

## Limitations & Future Work

1. **Not Applicable to Phase Retrieval**: The condition $\mu \neq 0$ excludes cases where $f(x) = x^2$ or $f(x) = |x|$.
2. **Tuning Dependencies**: $C_s$ and $C_s'$ require tuning for different measurement models and datasets.
3. **Gap Between Theory and Practice**: The error bound in Theorem 3 depends on the Lipschitz constant $L$, while the actual $L$ of DMs can be very large.
4. **Matrix Storage Overhead**: Requires explicit storage of the $m \times n$ measurement matrix $\mathbf{A}$, which is unfriendly to high-resolution images.
5. **Unexplored Structured Measurement Matrices**: Only i.i.d. Gaussian measurements are considered, whereas practical measurement matrices are usually structured.

## Related Work & Insights

- **DPS** (Chung et al., 2023): Signal recovery based on posterior sampling, requiring a known forward model.
- **DAPS** (Zhang et al., 2024): Extends DPS to nonlinear settings, but still requires $f$ to be differentiable.
- **QCS-SGM** (Meng & Kabashima, 2022): Uses SGM for quantized compressed sensing, but requires tens of thousands of NFEs.
- **CSGM** (Bora et al., 2017): Pioneered signal recovery using generative model priors.
- The proposed method can inspire other inverse problems: as long as observations can be expressed as a noisy version of the signal, the partial inversion + sampling framework of diffusion models can be utilized.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of determining the inversion starting point from the perspective of noise level matching is novel and theoretically sound.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons across multiple datasets, measurement models, and baselines, with extensive ablations included in the Appendix.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, notations are standard, and comparisons among the three methods are visually intuitive.
- Value: ⭐⭐⭐⭐ — Provides an efficient and general diffusion model solution for nonlinear inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](../../CVPR2025/image_generation/difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[CVPR 2025\] Learning Visual Generative Priors without Text](../../CVPR2025/image_generation/learning_visual_generative_priors_without_text.md)
- [\[CVPR 2025\] Nested Diffusion Models Using Hierarchical Latent Priors](../../CVPR2025/image_generation/nested_diffusion_models_using_hierarchical_latent_priors.md)
- [\[CVPR 2025\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](../../CVPR2025/image_generation/ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)

</div>

<!-- RELATED:END -->
