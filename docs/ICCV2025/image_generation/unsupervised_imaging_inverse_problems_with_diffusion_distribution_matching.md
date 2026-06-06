---
title: >-
  [Paper Note] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching
description: >-
  [ICCV 2025][Image Generation][Unsupervised Image Restoration] DDM4IP proposes an unsupervised framework that models the degradation distribution via Conditional Flow Matching…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Unsupervised Image Restoration"
  - "Inverse Problems"
  - "Conditional Flow Matching"
  - "Distribution Matching"
  - "Forward Model Learning"
date: 2026-05-08
content_hash: d28efe16278d53fc
---

# Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching

**Conference**: ICCV 2025
**arXiv**: [2506.14605](https://arxiv.org/abs/2506.14605)  
**Code**: [https://github.com/inria-thoth/ddm4ip](https://github.com/inria-thoth/ddm4ip)  
**Area**: Diffusion Models
**Keywords**: Unsupervised Image Restoration, Inverse Problems, Conditional Flow Matching, Distribution Matching, Forward Model Learning

## TL;DR

DDM4IP proposes an unsupervised framework that models the degradation distribution via Conditional Flow Matching, while simultaneously learning an unknown forward degradation model through a distribution matching loss. Using only a small number of unpaired images, the method achieves competitive or superior performance on deblurring, spatially-varying PSF calibration, and blind super-resolution tasks.

## Background & Motivation

**Background**: Image restoration is typically formulated as an inverse problem: given a degraded observation $y$, recover the original image $x$. Classical approaches assume the forward model $y = A(x) + \epsilon$ is fully known (e.g., the blur kernel or downsampling operator is given), and solve the problem via priors and optimization. Recent work has extensively employed diffusion models as powerful image priors for inverse problems (e.g., DPS, DiffPIR, GSPnP).

**Limitations of Prior Work**: Existing methods face two critical constraints. First, most methods assume the forward degradation model is known (non-blind setting), whereas in practice, blur kernels and noise distributions are often unknown or misspecified. Second, even blind methods typically require paired degraded–clean image pairs for training, which are prohibitively expensive to collect in real-world scenarios—e.g., microscope lens calibration demands specialized experimental setups.

**Key Challenge**: Powerful diffusion priors can yield high-quality image restoration, but existing frameworks depend on known forward models or paired training data, assumptions that frequently fail in practice. The central challenge is: how can one simultaneously learn a forward model and leverage diffusion priors for restoration using only a small number of unpaired images?

**Goal**: To design an unsupervised framework that (1) requires only a small number of unpaired degraded and clean images; (2) automatically learns the forward degradation model; and (3) exploits the learned forward model to perform high-quality image restoration.

**Key Insight**: The authors observe that Conditional Flow Matching (CFM) efficiently models data distributions, and its training objective naturally gives rise to a distribution matching loss. If a learnable forward operator is used to degrade clean images, and the resulting degraded distribution is required to match that of real degraded images, the forward model can be learned without any paired data.

**Core Idea**: A flow matching model is first fitted to the degraded image distribution as a reference; an auxiliary flow matching model is then trained to model the distribution of degraded images produced by the learnable forward operator. The forward operator is learned by matching the velocity fields of the two flow models—constituting distribution-level rather than sample-level matching.

## Method

### Overall Architecture

DDM4IP follows a three-stage pipeline:

**Stage 1** (Learning the Degradation Distribution): A conditional flow matching model $v_\theta$ is trained on a dataset of degraded images to learn the degraded image distribution $p(y)$. This model maps Gaussian noise to the degraded image distribution, with training guided by the standard flow matching loss.

**Stage 2** (Forward Model Learning via Distribution Matching): Two networks are trained jointly—(1) a learnable forward operator $A_\phi$ (kernel network) that degrades clean images, and (2) an auxiliary flow matching model $v_\psi$ (auxiliary flow network) that models the distribution of degraded images produced by $A_\phi$. The core loss aligns the velocity field of the auxiliary model with that of the reference model from Stage 1, indirectly constraining $A_\phi$ so that the degraded distribution it generates matches the true degradation distribution.

**Stage 3** (Model Inversion): The learned forward model $A_\phi$ from Stage 2 is plugged into standard inverse problem solvers (GSPnP, DPS, DiffPIR, etc.) to restore degraded images.

### Key Designs

1. **Conditional Flow Matching for Degradation Distribution Modeling**

    - **Function**: Models the degraded image distribution $p(y)$, serving as the reference target for subsequent distribution matching.
    - **Mechanism**: Flow matching defines a linear interpolation path from noise $x_0 \sim \mathcal{N}(0,I)$ to data $x_1 \sim p(y)$ via $x_t = (1-t)x_0 + tx_1$, training a velocity field network $v_\theta$ to predict the direction $x_1 - x_0$. The loss is $L_{FM} = \mathbb{E}_{t, x_0, x_1}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$. The network architecture employs a precondition-free UNet (RFNoPrecond) with conditional input support for modeling conditional dependencies.
    - **Design Motivation**: Compared to traditional diffusion models, the linear interpolation path of flow matching is simpler and yields more stable training. Moreover, velocity field matching naturally provides an interface for distribution matching—agreement between two velocity fields is equivalent to agreement between the corresponding distributions.

2. **Distribution Matching Loss for Forward Model Learning (DiffInstruct-on-Y)**

    - **Function**: Learns the forward degradation operator $A_\phi$ without requiring paired data.
    - **Mechanism**: Given a clean image $x$, the learnable kernel network $A_\phi$ generates a degraded image $\hat{y} = A_\phi(x)$. An auxiliary flow model $v_\psi$ is trained on $\hat{y}$, while the distribution matching loss $L_{DI} = \mathbb{E}[(v_\psi(y_t, t) - v_\theta^{ref}(y_t, t)) \cdot x_1]$ is computed, where $v_\theta^{ref}$ is the fixed reference model from Stage 1. This loss is backpropagated into $A_\phi$, driving the degraded distribution it generates to approximate the true degradation distribution. $A_\phi$ is further regularized by multiple constraints (sparsity, Gaussianity, centrality, normalization) to ensure physical plausibility of the learned kernel.
    - **Design Motivation**: Conventional paired learning requires $(x, y)$ pairs; distribution matching elevates the constraint from the sample level to the distribution level—the generated degraded images need only resemble the true degraded distribution, without requiring knowledge of which clean image corresponds to each degraded observation. This substantially reduces data requirements.

3. **Standard Inverse Problem Solvers for Model Inversion**

    - **Function**: Applies the learned forward model $A_\phi$ within established inverse problem solvers for image restoration.
    - **Mechanism**: The $A_\phi$ learned in Stage 2 can be embedded into any standard inverse problem framework. The paper employs GSPnP (Gradient-Step Plug-and-Play with RED prior), DPS (Diffusion Posterior Sampling), and DiffPIR from the DeepInv library, with DRUNet or DiffUNet as the denoising prior. Given $A_\phi$, these solvers iteratively recover the clean image.
    - **Design Motivation**: Decoupling forward model learning from inverse problem solving makes the framework highly modular—any new inverse problem solver can be substituted as a drop-in replacement for Stage 3.

### Loss & Training

- **Stage 1**: Standard flow matching loss $L_{FM}$, trained on 1,000 degraded images.
- **Stage 2**: Distribution matching loss $L_{DI}$ combined with kernel regularization (sparsity, Gaussianity, centrality, normalization); the auxiliary flow model and kernel network are optimized alternately.
- **Stage 3**: Inference using GSPnP/DPS/DiffPIR with the learned forward model; no additional training is required.

## Key Experimental Results

### Main Results

**FFHQ Motion Deblurring** (256×256; training set: 1,000 degraded + 100 unpaired clean images):

| Method | Type | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Wiener (known kernel) | Non-blind | 27.5 | 0.82 | 0.22 |
| GSPnP (known kernel) | Non-blind | 30.2 | 0.88 | 0.12 |
| DPS (known kernel) | Non-blind | 29.8 | 0.87 | 0.13 |
| BlindDPS | Blind (single) | 25.3 | 0.74 | 0.31 |
| GibbsDDRM | Blind (single) | 26.1 | 0.76 | 0.28 |
| **DDM4IP + GSPnP** | **Unsupervised** | **29.5** | **0.87** | **0.14** |

**DIV2K Blind Super-Resolution** (DIV2KRK benchmark, ×2 and ×4):

| Method | Type | PSNR↑ (×2) | PSNR↑ (×4) |
|--------|------|-----------|-----------|
| KernelGAN + ZSSR | Blind (single) | 31.2 | 27.8 |
| DCLS | Supervised | 32.1 | 28.5 |
| Real-ESRGAN | Supervised | 30.8 | 27.4 |
| **DDM4IP + ESRGAN** | **Unsupervised** | **31.9** | **28.3** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Notes |
|---------------|-------|-------|-------|
| Full DDM4IP (Stages 1+2+3) | 29.5 | 0.87 | Complete three-stage framework |
| w/o kernel regularization | 27.8 | 0.81 | Kernel degenerates to arbitrary convolution |
| w/o auxiliary flow model | 26.4 | 0.77 | Distribution matching infeasible without auxiliary flow |
| Stage 1 only (direct DPS) | 25.1 | 0.73 | Models degraded distribution without learning forward model |
| w/ paired data (oracle) | 30.1 | 0.88 | Upper bound reference using paired data |

### Key Findings

- DDM4IP closely approaches non-blind methods with known kernels on deblurring (PSNR gap of only 0.7 dB), while substantially outperforming single-image blind methods (by 3–4 dB), validating the effectiveness of distribution matching for forward model learning.
- On blind super-resolution, performance is competitive with state-of-the-art supervised methods (gap of 0.2 dB), demonstrating the generalizability of the framework.
- Kernel regularization has a significant impact on performance (removing it causes a 1.7 dB PSNR drop), confirming that physical prior constraints are essential for recovering a correct degradation kernel from distribution matching.
- The real-world parking lot lens calibration experiment is a notable highlight: whereas conventional methods require specialized equipment, DDM4IP estimates spatially-varying PSFs from a small set of unpaired photographs, demonstrating strong practical applicability.
- The auxiliary flow model serves as the critical bridge for distribution matching—removing it causes a 3.1 dB performance drop, indicating that directly imposing distributional constraints on the forward model alone is insufficient.

## Highlights & Insights

- **Distribution matching as a replacement for paired learning** is the central innovation: data requirements for inverse problems are reduced from paired samples to unpaired distributions, substantially extending the applicability of diffusion models to real-world image restoration. This principle transfers naturally to any task requiring learning of unknown degradations or transformations (e.g., domain adaptation, style transfer).
- **The three-stage decoupled design** is elegant: pretrain flow model → learn forward model via distribution matching → restore via standard solver. Each stage is independently replaceable; for instance, Stage 3 can seamlessly incorporate future, more powerful inverse problem solvers.
- **The insight that velocity field matching implies distribution matching** is particularly profound: two flow models whose velocity fields agree model identical distributions, providing a differentiable and tractable surrogate loss for distribution matching.
- The method operates with approximately 100 clean images and 1,000 degraded images, exhibiting high data efficiency.

## Limitations & Future Work

- The three-stage pipeline is relatively complex, resulting in substantial total training time—Stage 1 requires approximately 4.2 million iterations on FFHQ, with Stage 2 adding a further 1 million iterations.
- The forward model is currently limited to convolution-kernel-based degradations (blur, downsampling); more complex degradations (e.g., JPEG compression, weather corruption) require different parameterizations.
- The Stage 3 solvers (GSPnP, DPS, etc.) have slow inference speeds, limiting real-time applicability.
- The non-uniform PSF experiment fixes the grid size at 8×8; finer spatial variations may require adjustments to this resolution.
- Experiments are conducted primarily on face (FFHQ) and natural image (DIV2K) datasets; performance on specialized domains such as medical imaging has not been validated.

## Related Work & Insights

- **vs. DPS/DiffPIR**: These methods require a known forward model; the core contribution of DDM4IP is relaxing this assumption to an unknown model that is first learned and then applied. The learned forward model can be directly fed into DPS/DiffPIR.
- **vs. BlindDPS/GibbsDDRM**: These blind methods jointly estimate the forward model and restored image from a single image, and are limited by insufficient single-image information, resulting in notably lower quality compared to DDM4IP's distribution-level approach.
- **vs. CycleDiffusion/Unpaired IR**: Conventional unpaired methods typically rely on CycleGAN-style cycle consistency; the distribution matching mechanism in DDM4IP is more principled and stable.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of learning a forward model via distribution matching is original, natural, and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers deblurring, super-resolution, and real-world lens calibration, though validation across a broader range of degradation types is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous, the framework is described clearly, and code is publicly available.
- **Value**: ⭐⭐⭐⭐⭐ — Unsupervised inverse problem solving has high practical value; the real-world lens calibration application is convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](../../NeurIPS2025/image_generation/npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](../../ICML2026/image_generation/saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICCV 2025\] Pretrained Reversible Generation as Unsupervised Visual Representation Learning](pretrained_reversible_generation_as_unsupervised_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
