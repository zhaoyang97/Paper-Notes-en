---
title: >-
  [Paper Note] Reversing Flow for Image Restoration
description: >-
  [CVPR 2025][Image Generation][Continuous Normalizing Flows] ResFlow proposes modeling the image degradation process as a deterministic continuous normalizing flow (rather than a stochastic diffusion process). By resolving the irreversibility of degradation using auxiliary variables, it achieves reversible modeling. Implementing an entropy-preserving schedule, it completes high-quality image restoration in only 4 sampling steps, achieving SOTA on tasks such as desnowing…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Continuous Normalizing Flows"
  - "Image Restoration"
  - "Deterministic Degradation Path"
  - "Auxiliary Process"
  - "Entropy Conservation"
date: 2026-05-08
content_hash: db2dab803bb9a8c3
---

# Reversing Flow for Image Restoration

**Conference**: CVPR 2025  
**arXiv**: [2506.16961](https://arxiv.org/abs/2506.16961)  
**Code**: None (Project Page available)  
**Area**: Image Restoration / Generative Models  
**Keywords**: Continuous Normalizing Flows, Image Restoration, Deterministic Degradation Path, Auxiliary Process, Entropy Conservation

## TL;DR

ResFlow proposes modeling the image degradation process as a deterministic continuous normalizing flow (rather than a stochastic diffusion process). By resolving the irreversibility of degradation using auxiliary variables, it achieves reversible modeling. Implementing an entropy-preserving schedule, it completes high-quality image restoration in only 4 sampling steps, achieving SOTA on tasks such as desnowing, deraining, dehazing, denoising, and deblocking.

## Background & Motivation

**Background**: Image restoration aims to recover high-quality (HQ) images from degraded low-quality (LQ) images. Score- and diffusion-based generative models are currently dominant, including DDRM, IR-SDE, I2SB, ResShift, RDDM, etc. They typically model degradation as a stochastic forward process and learn a reverse process to restore images.

**Limitations of Prior Work**: (1) Starting the reverse process from Gaussian noise is unnecessary and inefficient, as LQ images already contain abundant structural information; (2) Even if models like IR-SDE and I2SB incorporate LQ images into the forward process, they still treat degradation as a stochastic process of progressive diffusion, introducing extra complexity; (3) Stochasticity leads to low training and inference efficiency, typically requiring dozens to hundreds of sampling steps.

**Key Challenge**: Image degradation is fundamentally **irreversible** (information is erased), as manifested by declining mutual information—the data processing inequality (DPI) indicates that the mutual information between intermediate states and the HQ image monotonically decreases during degradation. However, a deterministic ODE describes a reversible process (preserving constant mutual information), creating a contradiction. Hence, one cannot directly model degradation using an ODE.

**Goal**: Model the degradation process as a deterministic path rather than a stochastic path to achieve fast and high-quality image restoration.

**Key Insight**: Analyze the irreversibility of degradation from an information-theoretic perspective, and introduce an auxiliary variable process $\{\mathbf{y}_t\}$ to encode the information lost during degradation, making the augmented joint process $\mathbf{z}_t = [\mathbf{x}_t; \mathbf{y}_t]$ a reversible ODE.

**Core Idea**: Compensate for the information loss caused by degradation using auxiliary variables—as $\mathbf{x}_t$ approaches LQ and mutual information decreases, $\mathbf{y}_t$ compensates for this loss to keep the total mutual information constant, thereby achieving reversible degradation modeling with deterministic flows.

## Method

### Overall Architecture

ResFlow defines an augmented state $\mathbf{z}_t^T = [\mathbf{x}_t^T; \mathbf{y}_t^T]$, where $\mathbf{x}_0 = \mathbf{x}_{HQ}$, $\mathbf{x}_1 = \mathbf{x}_{LQ}$, $\mathbf{y}_0 = 0$, and $\mathbf{y}_1 \sim \mathcal{N}(0, I)$. A velocity field network $\mathbf{v}_\theta(\mathbf{x}_t, \mathbf{y}_t, t)$ is trained via velocity matching. During inference, integration is performed from $t=1$ (LQ + random $\mathbf{y}_1$) to $t=0$, taking $\hat{\mathbf{x}}_0$ as the restored result in only 4 steps.

### Key Designs

1. **Augmented Degradation Flow**:

    - **Function**: Transforms the irreversible degradation process into a reversible ODE using auxiliary variables.
    - **Mechanism**: Observes that a deterministic ODE $\partial \mathbf{z}_t / \partial t = \mathbf{v}(\mathbf{z}_t, t)$ preserves constant mutual information (Proposition 1), whereas the degradation process has decreasing mutual information. An auxiliary variable $\mathbf{y}_t$ is introduced and coupled with the "uncertainty range"—as $\mathbf{x}_t$ loses mutual information with HQ, $\mathbf{y}_t$ carries the compensating information. $\mathbf{y}_1$ is Gaussian noise (maximum entropy) and $\mathbf{y}_0 = 0$ (no auxiliary information is needed when restoration is complete). During training, $\mathbf{y}_t$ is independently coupled with $\mathbf{x}_0$, but the trained velocity network produces a deterministic coupling during inference.
    - **Design Motivation**: Direct ODEs cannot model irreversible degradation; augmenting the state space allows the conservation of mutual information in ODEs to be correctly exploited.

2. **Entropy-Preserving Schedule**:

    - **Function**: Defines individual interpolation paths for the data and auxiliary components.
    - **Mechanism**: The data component follows a straight path $\alpha_t^x = 1 - t$, $\sigma_t^x = t$ (a geodesic in Euclidean space). The auxiliary component employs a non-linear schedule $\sigma_t^y = \beta \cdot (1 - t + \beta)^{-1}$ ($\beta = 10$), keeping the total entropy constant throughout the process. This is based on the intuition that entropy should be conserved in a reversible process.
    - **Design Motivation**: Compared to parameterized estimation that directly predicts $\mathbb{E}[\mathbf{z}_0 | \mathbf{z}_t]$, velocity matching avoids high discretization errors near $t \to 0$.

3. **Adaptive Loss Weighting**:

    - **Function**: Balances training gradients across different timesteps.
    - **Mechanism**: As time approaches $t=1$ (the LQ end), $\mathbf{x}_t$ shares less mutual information with HQ, making velocity prediction harder. A loss weighting function $\lambda(t) = (\cos(\frac{\pi}{2}(t-2)) + 1)^\gamma$ ($\gamma = 1.75$) increases the weight as $t$ approaches 1, ensuring the model focuses more learning effort on difficult regions.
    - **Design Motivation**: Velocity estimation near the LQ end is more critical and difficult, requiring larger loss weights to compensate.

### Loss & Training

Velocity matching loss: $\min_\theta \mathbb{E}[\int_0^1 \lambda(t) \|\mathbf{v}_\theta(\mathbf{x}_t, \mathbf{y}_t, t) - \dot{\mathbf{z}}_t\|^2 dt]$. It matches target velocities directly without simulating the ODE, leading to efficient training. It uses the DDPM U-Net architecture, injecting timesteps via adaptive layer norm. It is trained on $256 \times 256$ crops and tested at full resolution. It uses a uniform time schedule with 4-step sampling.

## Key Experimental Results

### Main Results

**Synthetic Datasets**:

| Task | Metric | Prev. SOTA | Ours |
|------|------|----------|---------|
| Desnowing (Snow100K) | PSNR/SSIM/LPIPS | 30.92/0.917/0.034 | **31.86**/0.917/**0.030** |
| Deraining (Outdoor-Rain) | PSNR/SSIM | 30.99/0.934 | **32.82**/**0.936** |
| Dehazing (Dense-Haze) | PSNR/SSIM | 17.07/0.63 | **17.12**/0.59 |

**Real-world Datasets**:

| Task | Metric | Prev. SOTA | Ours |
|------|------|----------|---------|
| Denoising (SIDD) | PSNR/SSIM | 40.02/0.960 | **42.26**/**0.962** |
| Dehazing (NH-HAZE) | PSNR | 20.66 | **21.44** |
| Deraining (LHP) | PSNR/SSIM | 34.33/0.946 | **34.54**/0.939 |

### Ablation Study

Defocus Deblurring (DPDD Combined): Ours achieves the highest PSNR, outperforming Restormer (25.98) and FocalNet (26.18).

### Key Findings

- Only 4 sampling steps are required to reach SOTA, which is more than an order of magnitude faster than diffusion methods (typically 50-100 steps).
- Deterministic paths are more suitable for image restoration than stochastic paths—since degradation is known, stochasticity is unnecessary.
- The entropy-preserving schedule performs better than the straight-line schedule.
- Auxiliary variables conceptually encode the information lost during degradation. Different sampling of $\mathbf{y}_1$ during inference yields diverse restoration results, preserving the diversity of generative models.

## Highlights & Insights

- The **information-theoretic perspective** is elegant—explaining why ODEs cannot directly model degradation and why auxiliary variables are needed through mutual information conservation/decay.
- The proof of **Proposition 1** bridges ODEs and information theory, providing a solid theoretical foundation for ResFlow.
- Adapting the flow matching framework to image restoration is highly natural, completely avoiding the redundant stochasticity of diffusion models.
- The extreme efficiency of 4-step sampling is highly friendly to practical deployment.

## Limitations & Future Work

- Random sampling of the auxiliary variable $\mathbf{y}_1$ means restoration results can vary slightly each time, which may require extra handling for applications demanding deterministic outputs.
- The paper does not explore other low-level vision tasks such as super-resolution.
- For extreme degradation (where information is almost entirely lost), the compensation capability of the auxiliary variable might be insufficient.
- Integration with pre-trained diffusion models is unexplored—can this be adapted from a pre-trained Stable Diffusion model?

## Related Work & Insights

- The success of **Flow Matching / Continuous Normalizing Flow** in generative modeling is transferred to image restoration.
- Compared to InDI (which incrementally estimates HQ), ResFlow's velocity matching avoids error sensitivity near the HQ target.
- The concept of using auxiliary variables to resolve irreversibility can be generalized to other ill-posed inverse problems.

## Rating

- **Novelty**: 9/10 — The theory behind the information-theory-driven augmented flow design is beautiful and novel.
- **Experimental Thoroughness**: 8/10 — Covers 5 tasks across multiple datasets, though performance gains on some tasks are small.
- **Writing Quality**: 9/10 — Theoretical derivations are clear, and diagrams are intuitive.
- **Value**: 8/10 — Extreme efficiency of 4-step high-quality restoration is highly valuable for real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Navigating Image Restoration with VAR's Distribution Alignment Prior](navigating_image_restoration_with_vars_distribution_alignment_prior.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [\[CVPR 2025\] SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model](sir-diff_sparse_image_sets_restoration_with_multi-view_diffusion_model.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)

</div>

<!-- RELATED:END -->
