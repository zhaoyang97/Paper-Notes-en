---
title: >-
  [Paper Note] Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution
description: >-
  [ICML 2026][Image Restoration][Image Super-Resolution] ASASR achieves the optimal balance between perceptual quality and structural fidelity in super-resolution by replacing the isotropic Gaussian noise prior of Flow Mat…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Image Super-Resolution"
  - "Sobolev Space"
  - "DPO Alignment"
  - "Adversarial Learning"
  - "Spectral Consistency"
date: 2026-05-08
content_hash: 054d26eff3d0c615
---

# Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution

**Conference**: ICML 2026  
**arXiv**: [2605.23264](https://arxiv.org/abs/2605.23264)  
**Code**: https://github.com/wafer-bob/ASASR  
**Area**: Image Restoration  
**Keywords**: Image Super-Resolution, Sobolev Space, DPO Alignment, Adversarial Learning, Spectral Consistency  

## TL;DR

ASASR achieves the optimal balance between perceptual quality and structural fidelity in super-resolution by replacing the isotropic Gaussian noise prior of Flow Matching with Sobolev spectral colored noise, combined with an AS-DPO framework driven by adversarial manifold-guided hard negative samples.

## Background & Motivation

**Background**: Image super-resolution (SR) methods based on large-scale generative priors (Diffusion Models / Flow Matching) can synthesize realistic textures, but a fundamental conflict remains between generative quality and faithful restoration—models tend to "hallucinate" visually plausible but structurally incorrect high-frequency details.

**Limitations of Prior Work**: Existing methods (e.g., StableSR, SeeSR, DiffBIR) rely on supervised training paradigms with pixel-level alignment on synthetic degradation data. This leads to two issues: (1) models overfit to artificial degradation assumptions and fail to generalize to real-world degradation; (2) optimization based on the $\ell_2$ norm assigns uniform weight to all frequencies, failing to distinguish between genuine high-frequency details and artifacts. Introducing DPO to the SR domain is a potential alignment path, but standard DPO uses isotropic Gaussian parameterization, whose flat spectral prior is severely mismatched with the inherent spectral decay characteristics of natural images.

**Key Challenge**: The $\ell_2$ objective of standard DPO assigns unit weight to all frequency components in the Fourier domain $\|\boldsymbol{\gamma}_\theta\|_2^2 = \frac{1}{MN}\sum_{\boldsymbol{k}} 1 \cdot \|\hat{\boldsymbol{\gamma}}_\theta[\boldsymbol{k}]\|^2$, completely ignoring the statistical property where the power spectral density (PSD) of natural images decays with frequency. This prevents the model from effectively penalizing artifacts in high-frequency regions.

**Goal**: To design a theory-driven framework that performs super-resolution optimization in a geometric space aligned with the spectral characteristics of natural images, while providing informative negative samples to drive alignment learning.

**Key Insight**: The author observes that the PSD of natural images follows a $1/f$ decay law, and Sobolev spaces $H^s(\Omega)$ precisely encode this decay constraint through frequency-dependent weighting $(1+\|\boldsymbol{\omega}\|^2)^s$. Replacing the noise covariance matrix from $\mathbf{I}$ with a Sobolev spectral operator $\boldsymbol{\Sigma}_s$ naturally elevates the optimization metric from an $\ell_2$ norm to a Sobolev norm.

**Core Idea**: Reshape the DPO optimization geometry by substituting isotropic Gaussian noise with Sobolev spectral colored noise, and generate worst-case Sobolev gradients as hard negative samples using a parameterized adversary based on the Riesz Representation Theorem, achieving frequency-aware preference alignment.

## Method

### Overall Architecture

ASASR is built on the Flow Matching framework. It takes a low-resolution image $\boldsymbol{c}$ ($256 \times 256$) as input and outputs a $4\times$ super-resolution result ($1024 \times 1024$). The overall pipeline consists of two stages: (1) training the velocity network $\boldsymbol{v}_\theta$ under Sobolev spectral colored noise; (2) performing preference alignment via the AS-DPO loss, where negative samples are generated online by an adversarial network $\mathcal{A}_\phi$. The backbone employs FLUX.1-dev, fine-tuned via LoRA ($r=16, \alpha=16$).

### Key Designs

1.  **Sobolev Spectral Rectification (SSR)**:
    - **Function**: Lifts DPO optimization from flat Euclidean space to a Sobolev Riemannian manifold, allowing the optimization metric to automatically impose higher penalties on high-frequency errors.
    - **Mechanism**: Replaces the covariance matrix of the Flow Matching transition kernel from the identity matrix $\mathbf{I}$ with a structured spectral operator $\boldsymbol{\Sigma}_s = \mathcal{F}^{-1}\mathrm{diag}((1+\|\boldsymbol{\omega}\|_2^2)^{-s})\mathcal{F}$. Since the Gaussian likelihood is governed by the Mahalanobis distance, the precision matrix $\boldsymbol{\Sigma}_s^{-1}$ amplifies high-frequency components, transforming the log-likelihood ratio from an $\ell_2$ norm difference to a Sobolev norm difference: $\log \frac{p_\theta}{p_{\mathrm{ref}}} \propto -(\|\boldsymbol{\gamma}_\theta\|_{H^s}^2 - \|\boldsymbol{\gamma}_{\mathrm{ref}}\|_{H^s}^2)$.
    - **Design Motivation**: Directly addresses the spectral defect of standard DPO—where the $\ell_2$ norm treats all frequencies equally, leading to unpunished high-frequency artifacts—while the Sobolev norm naturally encodes the spectral decay prior via $(1+\|\boldsymbol{\omega}\|^2)^s$ weighting.

2.  **Adversarial Manifold Guidance (AMG)**:
    - **Function**: Synthesizes hard negative samples online that are semantically aligned with positive samples, driving S-DPO to learn the distinction between real details and structural artifacts.
    - **Mechanism**: Trains a parameterized adversary $\mathcal{A}_\phi$ to learn the model's typical reconstruction failure modes. Starting from an intermediate state $\boldsymbol{x}_t^w$ of a positive sample, the adversary predicts a velocity field leading the trajectory toward a degraded estimate $\widehat{\boldsymbol{x}}_1^a = \boldsymbol{x}_t^w + (1-t)\cdot\boldsymbol{v}_\phi(\boldsymbol{x}_t^w, t, \boldsymbol{c})$, and then projects back to the flow state using the same noise realization $\boldsymbol{x}_0$ to ensure semantic alignment. The optimal perturbation is proven to be equivalent to the Sobolev gradient direction: $\boldsymbol{\delta}_t^* = -\varepsilon_t \frac{\boldsymbol{\Sigma}_s \nabla_{\boldsymbol{x}}\mathcal{J}_{L^2}}{\sqrt{\langle\nabla_{\boldsymbol{x}}\mathcal{J}_{L^2}, \boldsymbol{\Sigma}_s\nabla_{\boldsymbol{x}}\mathcal{J}_{L^2}\rangle}}$.
    - **Design Motivation**: Standard DPO lacks informative negative samples—static pairs cannot capture subtle structural distortions in SR, and T2I preference datasets cannot provide spatially aligned negative samples due to semantic layout differences. AMG constructs counterfactual negative samples via shared noise realizations, specifically exposing the model's "falsely confident" blind spots.

3.  **AS-DPO Loss Integration**:
    - **Function**: Unifies SSR and AMG into a frequency-aware adversarial preference optimization objective.
    - **Mechanism**: Integrates the adversarial negative samples $\boldsymbol{x}_t^a$ generated by AMG with the Sobolev energy gap $\Delta\mathcal{E}_{H^s}$ from SSR into the final loss $\mathcal{L}_{\text{AS-DPO}}(\theta) = -\mathbb{E}[\log\sigma(\beta[\Delta\mathcal{E}_{H^s}(\boldsymbol{x}_t^a, \boldsymbol{c}) - \Delta\mathcal{E}_{H^s}(\boldsymbol{x}_t^w, \boldsymbol{c})])]$. Based on the Riesz Representation Theorem, the adversarial perturbation is proven equivalent to the worst-case Sobolev gradient, driving optimization along the tangent space of plausible perceptual failures.
    - **Design Motivation**: SSR and AMG are mutually coupled—AMG provides realism-oriented alignment signals, while SSR provides spectral-aware geometric constraints to ensure alignment does not come at the cost of reconstruction fidelity.

### Loss & Training

Training data for the adversarial network $\mathcal{A}_\phi$ comes from the inference outputs of Real-ESRGAN, SeeSR, and SUPSR on 25% random subsets of DIV2K/LSDIR/RealSR/DRealSR. The ASASR main model uses the AdamW optimizer with a learning rate of $1\times10^{-5}$; the adversarial network uses a learning rate of $5\times10^{-5}$. The Sobolev index is empirically set to $s=1.5$, achieving an optimal balance between structural fidelity and textural realism.

## Key Experimental Results

### Main Results

Compared with 11 SOTA methods on synthetic datasets (DIV2K-Val, LSDIR-Val) and real-world datasets (RealSR, DrealSR). Taking DIV2K-Val as an example:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | MANIQA↑ | MUSIQ↑ | CLIPIQA+↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BSRGAN | 20.36 | 0.5637 | 0.3899 | 0.3097 | 54.51 | 0.5641 |
| Real-ESRGAN | 21.11 | 0.5870 | 0.3147 | 0.3726 | 61.24 | 0.6126 |
| StableSR | 19.93 | 0.5528 | 0.3016 | 0.4224 | 66.30 | 0.6702 |
| SeeSR | 20.46 | 0.5411 | 0.3325 | 0.5187 | 70.59 | 0.7222 |
| DreamClear | 19.79 | 0.5137 | 0.3206 | 0.4878 | 60.66 | 0.6356 |
| DP2O-SR | 19.60 | 0.5064 | 0.3130 | 0.5810 | 70.58 | 0.7142 |
| **Ours (ASASR)** | **20.60** | **0.6171** | **0.2784** | **0.6519** | **71.40** | **0.7521** |

Downstream task evaluations (OCR / Object Detection / Instance Segmentation / Semantic Segmentation):

| Task | Metric | GT | LQ | SeeSR | DreamClear | DP2O-SR | **Ours** |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| OCR | Recall↑ | 50.32 | 3.81 | 29.33 | 36.78 | 40.03 | **45.91** |
| Detection | AP_b↑ | 48.32 | 12.53 | 28.06 | 28.04 | 33.51 | **35.62** |
| Instance Seg. | AP_m↑ | 42.52 | 11.03 | 23.98 | 24.20 | 29.22 | **30.98** |
| Semantic Seg. | mIoU↑ | 49.39 | 25.18 | 39.11 | 39.16 | 41.54 | **43.33** |

### Ablation Study

| Configuration | LPIPS↓ | MANIQA↑ | MUSIQ↑ | Recall↑ | AP_b↑ | mIoU↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full (Sobolev + Adversarial DPO) | **0.2784** | **0.6519** | **71.40** | **45.91** | **35.62** | **43.33** |
| Euclidean Guidance (w/o SSR) | 0.3115 | 0.6184 | 67.25 | 41.64 | 34.18 | 42.15 |
| DPO w/ Supervised Data (w/o AMG) | 0.3109 | 0.6047 | 68.82 | 40.16 | 32.49 | 40.26 |
| Only Supervised Learning | 0.3135 | 0.6012 | 69.15 | 40.33 | 31.95 | 39.85 |

### Key Findings

- SSR contributes the most: Removing SSR worsens LPIPS from 0.2784 to 0.3115 and MANIQA from 0.6519 to 0.6184, proving that spectral-aware geometry is the core of performance gains.
- Significant coupling between AMG and SSR: Using DPO with supervised data alone has limited effect, but combined with SSR, mIoU jumps from 40.26 to 43.33, indicating that AMG requires spectral-aware geometric constraints to work effectively.
- Sobolev index $s=1.5$ is the optimal choice: $s \geq 2$ causes over-smoothing and degrades perceptual quality, while $s < 1$ provides insufficient spectral constraints.
- User studies show a 91.1% Top-1 selection rate (50 participants, 64 test images).

## Highlights & Insights

- **Noise Coloring Concept**: The operation of replacing Flow Matching's isotropic Gaussian noise with spectral colored noise is remarkably elegant—it moves $\ell_2$ optimization to Sobolev norm optimization without changing the network architecture, merely by modifying the covariance matrix.
- **Adversary Philosophy**: AMG does not simply maximize loss; instead, it minimizes $\ell_2$ residual energy (mimicking model confidence) while deviating from the ground truth trajectory. This "blind spot exposure" strategy generates more informative hard negatives than random perturbations.
- **Transferable Combination**: The idea of reshaping optimization geometry into a spectral-aware space via SSR can be directly applied to other inverse problems (denoising, deblurring, compression artifact removal). Similarly, AMG’s counterfactual negative sample generation strategy is transferable to any generative task requiring preference alignment.

## Limitations & Future Work

- Training requires 8 H800 GPUs, entailing high computational costs; inference still requires multi-step ODE solvers, lacking real-time capability.
- Training the adversarial network relies on outputs from existing SR baseline models as artifact proxies; if the failure modes of these baselines are not diverse enough, it may limit the generalization of AMG.
- The Sobolev index $s$ requires manual tuning; future work could explore adaptive spectral weighting strategies.
- Validated only on $4\times$ SR; arbitrary scales or extreme degradation scenarios remain unexplored.

## Related Work & Insights

- **DP2O-SR** (Wu et al., 2025): A heuristic DPO based on aggregated IQA metrics, lacking theoretical foundation; ASASR provides a rigorous mathematical framework via Sobolev geometry.
- **FaithDiff** (Chen et al., 2024): Emphasizes fidelity but does not resolve the spectral alignment issue.
- **SeeSR / SUPSR**: Utilize semantic/text guidance to enhance controllability but are still limited by the spectral bias of the $\ell_2$ training paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Spectral Super-Resolution via Adversarial Unfolding and Data-Driven Spectrum Regularization](../../CVPR2026/image_restoration/spectral_super-resolution_via_adversarial_unfolding_and_data-driven_spectrum_reg.md)
- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [\[ICML 2026\] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)
- [\[ICML 2026\] Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)
- [\[ICML 2026\] PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)

</div>

<!-- RELATED:END -->
