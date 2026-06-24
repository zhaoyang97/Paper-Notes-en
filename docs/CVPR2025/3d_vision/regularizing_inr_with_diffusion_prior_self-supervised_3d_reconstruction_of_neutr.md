---
title: >-
  [Paper Note] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data
description: >-
  [CVPR 2025][3D Vision][Implicit Neural Representation] This paper proposes DINR (Diffusive INR), which combines implicit neural representation (INR/SIREN) with a pretrained diffusion model prior. By regularizing the INR reconstruction with the diffusion denoising output using a proximal loss at each DDIM timestep, DINR outperforms FBP, pure INR, DD3IP, and classical MBIR (qGGMRF) methods on sparse-view neutron CT (down to 4-5 views).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Implicit Neural Representation"
  - "Diffusion Prior"
  - "Neutron CT"
  - "Sparse-view Reconstruction"
  - "Self-supervised 3D Reconstruction"
date: 2026-05-08
content_hash: eab0b38452d73c4e
---

# Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data

**Conference**: CVPR 2025  
**arXiv**: [2603.10947](https://arxiv.org/abs/2603.10947)  
**Code**: To be released  
**Area**: 3D Vision / CT Reconstruction  
**Keywords**: Implicit Neural Representation, Diffusion Prior, Neutron CT, Sparse-view Reconstruction, Self-supervised 3D Reconstruction

## TL;DR
This paper proposes DINR (Diffusive INR), which combines implicit neural representation (INR/SIREN) with a pretrained diffusion model prior. By regularizing the INR reconstruction with the diffusion denoising output using a proximal loss at each DDIM timestep, DINR outperforms FBP, pure INR, DD3IP, and classical MBIR (qGGMRF) methods on sparse-view neutron CT (down to 4-5 views).

## Background & Motivation
**Background**: Neutron CT is an important imaging modality for characterizing volume based on hydrogen distribution (e.g., fuel cells, lithium-ion batteries, concrete structures). However, the low neutron flux leads to long exposure times, making sparse-view reconstruction urgently needed to accelerate acquisition.

**Limitations of Prior Work**: FBP produces severe artifacts under sub-Nyquist sampling; MBIR with handcrafted priors (e.g., TV/qGGMRF) requires extensive parameter tuning and has limited expressive power; pure INR (SIREN) lacks strong image priors, resulting in unstable high-frequency reconstruction.

**Key Challenge**: Diffusion models can model complex image priors, but directly applying them to posterior sampling in inverse problems (e.g., DD3IP/SCD) does not fully exploit data consistency. Conversely, INR can flexibly integrate forward models but lacks learned priors.

**Goal**: How to combine the strong generative power of diffusion priors with the data consistency advantages of INR to achieve high-quality sparse-view neutron CT reconstruction?

**Key Insight**: Based on the modular design of the DD3IP framework, INR is used to replace the original data-driven inverse problem solver (DIS) within each DDIM timestep, incorporating the diffusion denoising estimate via a proximal loss.

**Core Idea**: Embedding INR as a differentiable inverse solver within the DD3IP diffusion framework, achieving online guidance of INR by the diffusion prior through proximal regularization.

## Method

### Overall Architecture
DINR operates within the DD3IP framework: it initializes the INR parameters $\phi_T$ (using pure data consistency); in the reverse diffusion loop from $t=T$ to $t=1$, it first adapts the diffusion model weights $\theta_{t-1}$ (SCD) at each step to obtain the denoised estimate $\hat{x}_t$. Then, it optimizes the INR parameters $\phi_{t-1}$ using a proximal loss, and finally advances to the next step via DDIM sampling.

### Key Designs

1. **Proximal INR Loss Function**:

    - Function: Adds a proximal regularization term of the diffusion denoising output to the standard data consistency (projection-domain MSE).
    - Formula: $\mathcal{L}_\phi(S, y, \hat{x}_{0|t}, \rho) = \text{MSE}(A F_\phi(S, A^*y), y) + \rho \cdot \text{MSE}(\hat{x}_t, F_\phi(S, A^*y))$
    - Design Motivation: $\rho$ controls the influence strength of the diffusion prior. At initialization, $\rho=0$ (pure data fitting), and in subsequent timesteps, the diffusion estimate provides increasingly cleaner prior guidance.

2. **INR Architecture (SIREN + FBP Input)**:

    - Function: Maps 3D coordinates to attenuation coefficients using SIREN (sine-activated MLP).
    - Mechanism: Accepts the FBP reconstruction $A^*y$ as an additional input channel to provide an initial estimate and accelerate convergence.
    - Design Motivation: Dual inputs of coordinates and FBP allow the INR to obtain both precise coordinate localization and coarse structural information.

3. **Noise Injection Scaling $\omega$**:

    - Function: Controls the relative scale of the FBP reconstruction $A^*y$ and noise $\epsilon$ during reverse diffusion initialization.
    - Formula: $x_T \leftarrow \sqrt{\alpha_T} A^*y + \sqrt{1-\alpha_T} \epsilon * \omega$
    - Design Motivation: $\omega$ acts as a tunable parameter to balance the low-frequency initial estimate with random exploration.

### Loss & Training
- The diffusion model is pretrained on synthetic ellipsoid data (without needing real neutron CT data).
- SCD adapts the diffusion model weights at each timestep by minimizing $\text{MSE}(A D_\theta(x_t|y), y)$.
- INR is re-optimized at each step using the proximal loss, employing Tomosipo to implement a distance-driven parallel-beam projector.

## Key Experimental Results

### Synthetic Data (256×256, 2 slices)

| Number of Views | FBP | INR (SIREN) | DD3IP | DINR |
|---|---|---|---|---|
| 4 views | 19.31/0.08 | 14.76/0.18 | 26.17/0.25 | **26.27/0.24** |
| 8 views | 21.67/0.18 | 28.15/0.35 | 28.37/0.34 | **28.56/0.38** |
| 16 views | 25.27/0.30 | 30.34/0.54 | 31.21/0.61 | **31.30/0.63** |
| 32 views | 29.62/0.43 | 32.85/0.66 | 32.91/0.74 | **33.43/0.76** |

### Real Neutron CT Data

| Number of Views | FBP | MBIR(qGGMRF) | INR | DD3IP | DINR |
|---|---|---|---|---|---|
| 5 views | 19.9/0.10 | 21.02/0.04 | 20.18/0.03 | 20.89/0.06 | **21.27/0.05** |
| 9 views | 22.9/0.33 | **26.0/0.38** | 24.08/0.27 | 25.41/0.34 | 25.22/0.35 |
| 17 views | 25.91/0.55 | **28.1/0.58** | 27.3/0.54 | 28.04/0.62 | 27.56/0.62 |
| 33 views | 30.11/0.73 | 31.0/0.77 | 29.7/0.71 | 31.19/0.79 | **31.37/0.77** |

### Ablation Study / ROI Analysis

| ROI Size | Observation |
|---|---|
| 8×8 ~ 32×32 (Microstructure region) | DINR significantly outperforms other methods |
| 48×48 ~ 64×96 (Including background) | MBIR is close to or outperforms DINR |

- DINR achieves the optimal reconstruction in microstructural details (pores/boundaries), but its advantage diminishes in large homogeneous background areas.
- This aligns with the inherent advantage of MBIR's qGGMRF prior in smooth regions.

### Key Findings
- The diffusion model pretrained only on synthetic ellipsoids can effectively guide the reconstruction of real concrete microstructures, demonstrating OOD (out-of-distribution) adaptability.
- The advantage of DINR is most pronounced in the ultra-sparse range (4-5 views), where data constraints are extremely weak and a strong prior is most critical.
- The proximal regularization of INR is more flexible than the conjugate gradient DIS of DD3IP, allowing for seamless integration into the forward physical model.
- Better quantitative metrics are needed—PSNR/SSIM lack sufficient discriminative power in evaluating microstructure reconstruction quality.

## Highlights & Insights
- **Modular Diffusion-INR Fusion**: The modular design of the DIS within the DD3IP framework allows INR to replace other solvers in a plug-and-play manner; the scalability of this framework is noteworthy.
- **Synthetic Pretraining + OOD Inference**: The diffusion model pretrained solely on synthetic data can guide real data reconstruction, reducing the reliance on in-domain training data.
- **Insight from ROI Analysis**: Conventional full-image PSNR can obscure the true advantages of a method in crucial regions (microstructures), highlighting the need for task-oriented evaluation.

## Limitations & Future Work
- High computational overhead: both the INR parameters and the diffusion model weights must be optimized at each DDIM timestep.
- On real data, DINR fails to outperform MBIR across all view numbers (MBIR is superior at 9 and 17 views).
- $\rho$ and $\omega$ require meticulous parameter search; the authors admit that a more comprehensive search could yield better results.
- Only parallel-beam geometry was validated; it has not been extended to cone-beam or helical CT.
- Lack of comparison with other learning-based CT reconstruction methods (e.g., end-to-end U-Net).

## Related Work & Insights
- **vs DD3IP**: DINR replaces the CG iterations in DD3IP with INR as the DIS, achieving better performance under ultra-sparse views.
- **vs MBIR+qGGMRF**: MBIR remains competitive at moderate view counts but requires exhaustive grid search for regularization parameters; DINR is superior under ultra-sparse views and in microstructural regions.
- **vs Pure SIREN/INR**: Lacking strong priors leads to high-frequency instabilities; the diffusion regularization in DINR effectively addresses this issue.
- This holds direct reference value for researchers working on industrial/scientific CT that requires low-scan-count acquisitions.

## Rating
- Novelty: ⭐⭐⭐⭐ Embedding INR as DIS within DD3IP is a novel fusion approach.
- Experimental Thoroughness: ⭐⭐⭐ The data scale is small (2 slices), and there is a lack of ablation studies and more baselines.
- Writing Quality: ⭐⭐⭐ The method is described clearly, but the experimental analysis could be deeper.
- Value: ⭐⭐⭐⭐ Holds practical value for the field of scientific CT reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Decompositional Neural Scene Reconstruction with Generative Diffusion Prior](decompositional_neural_scene_reconstruction_with_generative_diffusion_prior.md)
- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)
- [\[CVPR 2025\] Sonata: Self-Supervised Learning of Reliable Point Representations](sonata_self-supervised_learning_of_reliable_point_representations.md)
- [\[CVPR 2025\] SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input](spatialdreamer_self-supervised_stereo_video_synthesis_from_monocular_input.md)

</div>

<!-- RELATED:END -->
