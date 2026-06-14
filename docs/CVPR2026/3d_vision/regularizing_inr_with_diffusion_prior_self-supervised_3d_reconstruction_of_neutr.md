---
title: >-
  [Paper Note] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data
description: >-
  [CVPR 2026][3D Vision][Neutron CT] This paper proposes Diffusive INR (DINR), a framework that replaces the conventional DIS in the DD3IP diffusion reconstruction pipeline with an INR…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Neutron CT"
  - "Implicit Neural Representation"
  - "Diffusion Prior"
  - "Sparse-view Reconstruction"
  - "Inverse Problem"
date: 2026-05-08
content_hash: 9bca2c2ce07b1e21
---

# Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data

**Conference**: CVPR 2026
**arXiv**: [2603.10947](https://arxiv.org/abs/2603.10947)  
**Code**: Coming soon  
**Area**: 3D Vision
**Keywords**: Neutron CT, Implicit Neural Representation, Diffusion Prior, Sparse-view Reconstruction, Inverse Problem

## TL;DR

This paper proposes Diffusive INR (DINR), a framework that replaces the conventional DIS in the DD3IP diffusion reconstruction pipeline with an INR, and injects the diffusion model's denoising estimate as a regularization prior into the INR optimization via a proximal loss function. Under extremely sparse neutron CT conditions with only 4–5 views, DINR surpasses MBIR (qGGMRF), DD3IP, and vanilla INR in reconstruction quality.

## Background & Motivation

Neutron computed tomography (Neutron CT) plays an irreplaceable role in fuel cell manufacturing, lithium battery research, plant water transport, and concrete structural monitoring due to its unique capability to characterize hydrogen distributions. However, neutron beam flux is far lower than that of X-rays, requiring significantly longer exposure times per acquisition, making sparse-view acquisition a practical necessity.

Conventional Filtered Back Projection (FBP) produces severe artifacts when the number of projection views falls below the Nyquist requirement. Model-Based Iterative Reconstruction (MBIR) alleviates this issue by incorporating handcrafted priors (e.g., TV, qGGMRF), but the modeling capacity of such priors remains limited. Two technical lines have recently shown promise:

- **Implicit Neural Representation (INR)**: Maps coordinates to attenuation coefficients via an MLP, providing a continuous and memory-efficient volumetric representation that readily integrates with physical forward models. However, INR suffers from spectral bias—a tendency toward low-frequency components—yielding poor reconstruction of high-frequency structures under sparse supervision.
- **Diffusion Model Prior**: The DD3IP framework adapts a pretrained diffusion model to out-of-distribution (OOD) inference data via Steerable Conditional Diffusion (SCD), enabling consistent 3D reconstruction across orthogonal spatial dimensions. A key finding of DD3IP is that its framework is agnostic to the choice of DIS (diffusion inverse problem solver), allowing any state-of-the-art solver to be plugged in.

The motivation of this paper follows naturally: **use INR as the DIS within the DD3IP framework**, combining the continuous representational capacity of INR with the strong generative prior of diffusion models, while using the diffusion estimate to regularize the INR and compensate for its spectral bias.

## Method

### Overall Architecture

DINR operates within the DD3IP framework. Let the observation model be $y = Ax + n$, where $x$ is the 3D attenuation coefficient volume, $A$ is the parallel-beam CT projection matrix, and $n$ is additive noise. The complete DINR pipeline proceeds as follows:

1. **Initialization**: Pretrain INR weights $\phi_T$ using a pure data fidelity loss; load pretrained diffusion model weights $\theta_T$ trained on synthetic data; initialize the diffusion starting point $x_T$ by adding noise to the FBP reconstruction $A^*y$.
2. **Diffusion Iterations** ($t = T \to 1$):
    - Update diffusion model weights $\theta_{t-1}$ to adapt to the current OOD data by minimizing $\text{MSE}(A D_\theta(x_t|y), y)$.
    - Obtain the current estimate via diffusion denoising: $\hat{x}_t = D_{\theta_{t-1}}(x_t|y)$.
    - Update INR weights $\phi_{t-1}$ using the proximal loss, with $\hat{x}_t$ serving as the regularization target.
    - Generate the next-step estimate $x_{t-1}$ via DDIM sampling, where the posterior mean is provided by the INR output $F_{\phi_{t-1}}(S, A^*y)$.
3. **Final Step** ($t=1$): Directly output the INR reconstruction without further noise addition.

### Key Designs

1. **Proximal Regularization INR Loss (Proximal INR Loss)**
    - **Function**: Injects the diffusion model's denoising estimate into the INR optimization to compensate for INR's low-frequency bias.
    - **Mechanism**: The loss consists of two terms—a data fidelity term $\text{MSE}(A F_\phi(S, A^*y), y)$ ensuring consistency with the projection data, and a proximal term $\rho \cdot \text{MSE}(\hat{x}_t, F_\phi(S, A^*y))$ that pulls the INR output toward the current diffusion estimate. The full formula is $\mathcal{L}_\phi = \text{MSE}(AF_\phi, y) + \rho \cdot \text{MSE}(\hat{x}_t, F_\phi)$.
    - **Design Motivation**: Diffusion priors excel at modeling high-frequency structures, while INR excels at enforcing data consistency constraints. The proximal formulation enables the complementary strengths of both. A single hyperparameter $\rho$ controls the balance.

2. **FBP-Augmented SIREN INR Architecture**
    - **Function**: Provides a continuous 3D volumetric representation while leveraging the coarse FBP reconstruction to accelerate convergence.
    - **Mechanism**: The INR adopts the SIREN architecture (an MLP with periodic sinusoidal activation functions), taking as input the concatenation of the 3D coordinate grid $S$ and the FBP reconstruction $A^*y$. The periodic activations of SIREN are naturally suited to capturing high-frequency signals.
    - **Design Motivation**: Although low in quality, FBP provides a meaningful initial estimate that helps the INR converge more rapidly to a reasonable solution; coordinate-based input makes reconstruction resolution-independent.

3. **Noise Scaling Parameter $\omega$ and Initialization Strategy**
    - **Function**: Controls the relative ratio of signal to noise at the diffusion starting point.
    - **Mechanism**: $x_T = \sqrt{\alpha_T} A^*y + \sqrt{1 - \alpha_T} \cdot \epsilon \cdot \omega$. A larger $\omega$ introduces stronger noise, amplifying the influence of the diffusion prior; a smaller $\omega$ places greater trust in the FBP estimate.
    - **Design Motivation**: The quality of FBP varies substantially across different sparsity levels, and $\omega$ provides flexibility for adaptive adjustment. Experiments show that the optimal $\omega$ differs across view counts (0.2 for 4 views; 0.02 or 0.002 for 8–32 views).

### Loss & Training

- **INR Initialization**: $\rho = 0$; pure data fidelity loss with no diffusion prior.
- **During Diffusion Iterations**: $\rho$ is set such that the ratio of the proximal term to the data fidelity term is approximately $10^{-5}$ (synthetic data) or $10^{-6}$ (real data).
- **Diffusion Model Pretraining**: A UNet DDPM is trained exclusively on synthetic ellipsoid data; at inference time, SCD weight updates adapt it to OOD concrete microstructure data.
- **DDIM Sampling**: A deterministic DDIM reverse process is used; $\eta$ controls stochasticity, and noise is interpolated using spherical interpolation.

## Key Experimental Results

### Main Results

**Synthetic data** (2×256×256 volume, parallel-beam projection):

| Views | FBP (PSNR/SSIM) | INR (SIREN) | DD3IP | **DINR** |
|-------|-----------------|-------------|-------|----------|
| 4 | 19.31 / 0.08 | 14.76 / 0.18 | 26.17 / 0.25 | **26.27 / 0.24** |
| 8 | 21.67 / 0.18 | 28.15 / 0.35 | 28.37 / 0.34 | **28.56 / 0.38** |
| 16 | 25.27 / 0.30 | 30.34 / 0.54 | 31.21 / 0.61 | **31.30 / 0.63** |
| 32 | 29.62 / 0.43 | 32.85 / 0.66 | 32.91 / 0.74 | **33.43 / 0.76** |

DINR achieves the highest PSNR at all sparsity levels. At 32 views it outperforms DD3IP by 0.52 dB; at 4 views it surpasses vanilla INR by 11.51 dB.

**Real neutron CT data** (concrete microstructure; 1091-view 360° scan subsampled):

| Views | FBP | MBIR (qGGMRF) | INR | DD3IP | **DINR** |
|-------|-----|---------------|-----|-------|----------|
| 5 | 19.90 / 0.10 | 21.02 / 0.04 | 20.18 / 0.03 | 20.89 / 0.06 | **21.27 / 0.05** |
| 9 | 22.90 / 0.33 | **26.00 / 0.38** | 24.08 / 0.27 | 25.41 / 0.34 | 25.22 / 0.35 |
| 17 | 25.91 / 0.55 | **28.10 / 0.58** | 27.30 / 0.54 | 28.04 / 0.62 | 27.56 / 0.62 |
| 33 | 30.11 / 0.73 | 31.00 / 0.77 | 29.70 / 0.71 | 31.19 / 0.79 | **31.37 / 0.77** |

At 5 views, DINR (21.27 dB) outperforms carefully tuned MBIR (21.02 dB); at 33 views, DINR (31.37 dB) also surpasses MBIR (31.00 dB). At 9 and 17 views, MBIR achieves higher overall PSNR, primarily due to advantages in smooth background regions.

### Ablation Study

The paper employs ROI-scale analysis in place of conventional ablation experiments:

- **Relationship between ROI scale and method advantage**: In progressively cropped ROIs ranging from 64×96 to 8×8 pixels, DINR begins to outperform other methods when ROI < 48×48, with the advantage becoming pronounced below 32×32. This demonstrates that DINR achieves higher reconstruction fidelity in microstructure regions rather than flat backgrounds.
- **Effect of $\omega$**: The optimal $\omega = 0.2$ for 4 views (requiring a stronger diffusion prior), $\omega = 0.02$ for 8 views, and $\omega = 0.02$ for 32 views (relying more on data fidelity).
- **Setting of $\rho$**: The proximal-to-fidelity term ratio is $10^{-5}$ for synthetic data and $10^{-6}$ for real data, indicating that a weaker diffusion constraint is needed for real data.

### Key Findings

- Vanilla INR achieves only 14.76 dB PSNR at 4 views (lower than FBP), demonstrating that INR without prior knowledge completely fails under extreme sparsity.
- DINR reaches 26.27 dB at 4 views; the diffusion prior improves INR performance by 11.51 dB.
- MBIR achieves higher PSNR at moderate sparsity (9–17 views), but this advantage stems from background smoothing rather than microstructure fidelity—ROI analysis reveals the misleading nature of global metrics.
- The diffusion model, trained solely on synthetic ellipsoids, successfully transfers to structurally distinct concrete microstructures, demonstrating the OOD generalization capability of the SCD weight adaptation mechanism.

## Highlights & Insights

- **Modular design philosophy**: DINR exploits DD3IP's DIS-agnostic property to seamlessly integrate INR into the diffusion framework. This "building-block" design allows future substitution with improved INR architectures or diffusion models.
- **Elegance of the proximal loss**: A single parameter $\rho$ balances physics-based data constraints against learned priors, yielding a concise formulation with straightforward implementation.
- **ROI-scale analysis methodology**: This analysis exposes the limitations of global PSNR/SSIM in scientific imaging—when microstructural detail is the focus, evaluation must be conducted across different spatial scales. This analytical approach is itself a methodological contribution.
- **Synthetic-to-real transfer**: A diffusion model trained on simple geometric primitives (ellipsoids) can be adapted to complex microstructures via SCD weight fine-tuning, reducing training data requirements.

## Limitations & Future Work

- **Limited volume scale**: Only 2×256×256 volumes (2 slices) are evaluated; the feasibility of large-scale 3D volume reconstruction is not verified.
- **Lack of systematic ablation**: The authors acknowledge the absence of ablation experiments on FBP input, network architecture, and number of diffusion steps.
- **Manual hyperparameter tuning**: Both $\rho$ and $\omega$ require manual search tailored to data characteristics, with no adaptive strategy provided.
- **Computational efficiency not reported**: Total runtime for INR optimization, diffusion iterations, and SCD weight updates is not provided and may be substantially slower than FBP or MBIR.
- **Limited advantage at moderate sparsity**: DINR does not surpass well-tuned MBIR at 9–17 views, constraining its practical applicability.
- **Narrow evaluation metrics**: Only PSNR/SSIM are reported; application-oriented metrics such as segmentation accuracy and boundary fidelity are absent.

## Related Work & Insights

- **DD3IP [Chung & Ye, ECCV 2024]**: The direct predecessor of DINR, providing the framework and the key finding that DIS is replaceable. DINR achieves a PSNR improvement of 0.1–0.5 dB on synthetic data.
- **SIREN [Sitzmann et al., NeurIPS 2020]**: The INR backbone adopted by DINR; periodic activation functions alleviate but do not fully resolve spectral bias.
- **SCD [Barbano et al., 2023]**: Provides the theoretical foundation and weight update mechanism for adapting diffusion priors to OOD data.
- **Insights**: The proximal regularization coupling strategy is generalizable to other INR-based reconstruction tasks such as NeRF and 3D Gaussian Splatting; the OOD adaptation strategy is a valuable reference for data-scarce scientific imaging modalities (electron microscopy, cryo-EM).

## Rating

- **Novelty**: ⭐⭐⭐⭐ Primarily an incremental improvement to the DD3IP framework, replacing the CG solver with INR and a proximal regularizer—a natural but non-breakthrough contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Limited to 2-slice volumes; no systematic ablation, no runtime analysis, and no downstream task evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Method derivation is clear and complete; the paper is concise overall, with ROI analysis being a notable highlight.
- **Value**: ⭐⭐⭐⭐ Directly applicable to sparse-view neutron CT reconstruction, but generalizability and scalability remain insufficiently validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [\[CVPR 2026\] PointINS: Instance-Aware Self-Supervised Learning for Point Clouds](pointins_instance-aware_self-supervised_learning_for_point_clouds.md)
- [\[CVPR 2026\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weaklysupervised_s.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)

</div>

<!-- RELATED:END -->
