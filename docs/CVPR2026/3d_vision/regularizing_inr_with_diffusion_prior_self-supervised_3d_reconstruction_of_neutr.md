---
title: >-
  [Paper Note] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data
description: >-
  [CVPR 2026][3D Vision][Neutron CT] This paper proposes DINR (Diffusive INR), which replaces the conventional inversion solver within the DD3IP diffusion framework with an INR, injecting diffusion denoising estimates into the INR optimization via a proximal loss. DINR surpasses existing SOTA methods for neutron CT reconstruction under extremely sparse-view conditions (as few as 4–5 views).
tags:
  - CVPR 2026
  - 3D Vision
  - Neutron CT
  - Sparse-view Reconstruction
  - Implicit Neural Representation
  - Diffusion Prior
  - Inverse Problem
date: 2026-05-08
content_hash: d30314d69a96271c
---

# Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data

**Conference**: CVPR 2026
**arXiv**: [2603.10947](https://arxiv.org/abs/2603.10947)
**Code**: Coming soon
**Area**: 3D Vision
**Keywords**: Neutron CT, Sparse-view Reconstruction, Implicit Neural Representation, Diffusion Prior, Inverse Problem

## TL;DR

This paper proposes DINR (Diffusive INR), which replaces the conventional inversion solver within the DD3IP diffusion framework with an INR, injecting diffusion denoising estimates into the INR optimization via a proximal loss. DINR surpasses existing SOTA methods for neutron CT reconstruction under extremely sparse-view conditions (as few as 4–5 views).

## Background & Motivation

**Background**: Neutron CT is an important imaging modality that characterizes volumetric internal structures through hydrogen distribution, with broad applications in hydrogen fuel cell manufacturing, lithium-ion battery research, plant/soil moisture transport monitoring, and concrete radiation shielding safety inspection.

**Limitations of Prior Work**: Neutron beam flux is extremely low, requiring long exposure times per view, resulting in far fewer projections than Nyquist sampling demands. Conventional FBP produces severe artifacts under sparse views (only 19.31 dB PSNR at 4 views). MBIR methods using handcrafted priors (TV, qGGMRF) offer improvements but require time-consuming parameter searches for each sparsity level and exhibit limited fidelity to microstructural details.

**Key Challenge**: INRs (e.g., SIREN) offer advantages such as resolution independence, memory efficiency, and easy integration with physical forward models, but suffer from severe low-frequency spectral bias, leading to poor high-frequency structure recovery under sparse supervision (only 14.76 dB PSNR at 4 views). Diffusion models (e.g., DD3IP/SCD) can provide powerful generative priors and adapt to out-of-distribution (OOD) data, but their inversion steps typically rely on CG solvers, failing to exploit the continuous representation advantages of INRs.

**Goal**: The paper seeks to effectively inject the strong generative prior of diffusion models into the INR framework without modifying the diffusion model architecture, enabling high-fidelity 3D CT reconstruction under extremely sparse-view conditions.

**Key Insight**: A key insight from the DD3IP framework is that the posterior mean estimation method can be freely substituted. This paper exploits this modularity by replacing the CG solver with an INR as the posterior mean estimator in the diffusion inversion step, feeding diffusion denoising outputs back to the INR via a proximal loss.

**Core Idea**: At each timestep of the diffusion reverse process, the INR weights are optimized using a loss that includes a proximal term anchored to the diffusion denoising estimate, enabling the INR to simultaneously satisfy measurement data consistency and diffusion prior constraints.

## Method

### Overall Architecture

DINR is built upon the DD3IP (3D Deep Diffusion Image Prior) framework. The problem is modeled as $y = Ax + n$, where $x$ is the 3D attenuation coefficient volume, $y$ is the projection measurement, and $A$ is the parallel-beam projection matrix. At each timestep $t$ of the diffusion reverse process, the framework performs three operations: (1) updating diffusion model weights to adapt to OOD data; (2) generating a denoising estimate $\hat{x}_t$; and (3) optimizing INR weights via proximal loss and generating the next-step estimate via DDIM sampling. The diffusion model is pretrained solely on synthetic ellipsoid data and adapts to real neutron CT data at inference time through the SCD weight update mechanism.

### Key Designs

1. **INR as Posterior Mean Estimator (INR as DIS)**:
    - Function: Replaces the CG solver in DD3IP as the posterior mean estimation method within the diffusion reverse process.
    - Mechanism: The INR model $F_\phi$ uses a SIREN architecture, mapping a 3D coordinate grid $S$ to attenuation coefficient values, with the FBP reconstruction $A^*y$ provided as auxiliary input to accelerate convergence.
    - Design Motivation: DD3IP demonstrates that posterior sampling is agnostic to the choice of DIS. The continuous representation of INRs naturally supports resolution-independent reconstruction, and their differentiability facilitates joint optimization with the CT forward model and diffusion prior.

2. **Proximal Loss**:
    - Function: Simultaneously enforces data fidelity and diffusion prior constraints during INR optimization.
    - Mechanism: The loss comprises two terms — a data fidelity term $\text{MSE}(AF_\phi(S, A^*y), y)$ ensuring projection consistency, and a proximal term $\rho \cdot \text{MSE}(\hat{x}_t, F_\phi(S, A^*y))$ pulling the INR output toward the diffusion denoising estimate $\hat{x}_t$.
    - Design Motivation: The proximal term introduces diffusion-learned image priors into the INR, compensating for the lack of strong priors in pure INR methods. The parameter $\rho$ controls the prior influence; during initialization, $\rho=0$ is set for data fitting only.

3. **Noise Scaling Initialization**:
    - Function: Controls the relative ratio of signal to noise at the starting point of the diffusion reverse process.
    - Mechanism: $x_T = \sqrt{\alpha_T} A^*y + \sqrt{1-\alpha_T} \epsilon \cdot \omega$, introducing a tunable parameter $\omega > 0$ to scale the injected noise.
    - Design Motivation: $\omega$ controls the proportion of FBP low-frequency components relative to noise, indirectly adjusting the regularization strength of the DD3IP framework. Different sparsity levels require different $\omega$ values for optimal reconstruction.

### Loss & Training

**Complete Algorithm**:

1. Initialize INR weights $\phi_T$: fit projection data using standard MSE loss ($\rho=0$).
2. Load pretrained diffusion model weights $\theta_T$ (trained on synthetic ellipsoid data).
3. Initialize $x_T = \sqrt{\alpha_T} A^*y + \sqrt{1-\alpha_T} \epsilon \cdot \omega$.
4. For each timestep $t = T \to 1$:
    - SCD step: update $\theta_{t-1} = \arg\min_\theta \text{MSE}(AD_\theta(x_t|y), y)$.
    - Denoising: $\hat{x}_t = D_{\theta_{t-1}}(x_t|y)$.
    - INR update: $\phi_{t-1} = \arg\min_\phi \mathcal{L}_\phi(S, y, \hat{x}_t, \rho)$.
    - Sampling: if $t>1$, $x_{t-1} = \text{DDIM}_{\theta_{t-1}}(F_{\phi_{t-1}}(S, A^*y), \eta)$; if $t=1$, output $x_0 = F_{\phi_0}(S, A^*y)$ directly.

**Key Hyperparameter Settings**: For synthetic data, $\rho$ is set such that the ratio of the proximal term to the data term is $1 \times 10^{-5}$; for real data, this ratio is $1 \times 10^{-6}$. $\omega$ is determined via parameter search (0.02–0.2 for synthetic data; 0.002 for real data).

## Key Experimental Results

### Main Results

**Synthetic data** ($2 \times 256 \times 256$ concrete microstructure phantom):

| Views | FBP | INR (SIREN) | DD3IP | **DINR** |
|:---:|:---:|:---:|:---:|:---:|
| 4 | 19.31 / 0.08 | 14.76 / 0.18 | 26.17 / 0.25 | **26.27 / 0.24** |
| 8 | 21.67 / 0.18 | 28.15 / 0.35 | 28.37 / 0.34 | **28.56 / 0.38** |
| 16 | 25.27 / 0.30 | 30.34 / 0.54 | 31.21 / 0.61 | **31.30 / 0.63** |
| 32 | 29.62 / 0.43 | 32.85 / 0.66 | 32.91 / 0.74 | **33.43 / 0.76** |

*Metrics: PSNR (dB) / SSIM*

**Real neutron CT data** (1091 views/360° neutron scanner, downsampled to 256 resolution):

| Views | FBP | MBIR (qGGMRF) | INR | DD3IP | **DINR** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 19.90 / 0.10 | 21.02 / 0.04 | 20.18 / 0.03 | 20.89 / 0.06 | **21.27 / 0.05** |
| 9 | 22.90 / 0.33 | **26.00 / 0.38** | 24.08 / 0.27 | 25.41 / 0.34 | 25.22 / 0.35 |
| 17 | 25.91 / 0.55 | 28.10 / 0.58 | 27.30 / 0.54 | **28.04 / 0.62** | 27.56 / 0.62 |
| 33 | 30.11 / 0.73 | 31.00 / 0.77 | 29.70 / 0.71 | 31.19 / 0.79 | **31.37 / 0.77** |

*Metrics: PSNR (dB) / SSIM; MBIR uses exhaustive parameter search ($10^{-4}$ to $10^6$) per sparsity level to obtain best performance.*

### Ablation Study

**ROI Scale Analysis** (real data, data-driven unbiased ROI selection):

Rather than conventional ablation experiments, the paper provides a systematic ROI scale analysis. Sub-regions ranging from $64 \times 64$ to $8 \times 8$ are cropped from a $64 \times 96$ region for PSNR computation:

| ROI Scale | DINR vs. DD3IP Trend | DINR vs. MBIR Trend |
|:---:|:---:|:---:|
| $> 48 \times 48$ | Comparable or marginally better | MBIR superior at moderate sparsity |
| $32 \times 32$ | DINR clearly superior | DINR begins to surpass MBIR |
| $< 32 \times 32$ | DINR significantly superior | DINR comprehensively outperforms |

### Key Findings

- **Consistent gains on synthetic data**: DINR achieves the highest PSNR and SSIM across all four sparsity levels (4/8/16/32 views), outperforming pure INR by 11.51 dB at 4 views.
- **Strong advantage in ultra-sparse regimes**: At 5 views on real data, DINR (21.27 dB) surpasses exhaustively tuned MBIR (21.02 dB) and DD3IP (20.89 dB).
- **Superior microstructure fidelity**: DINR's advantage grows as ROI shrinks below $32 \times 32$, indicating superior preservation of high-frequency details such as pores and microstructures.
- **MBIR remains competitive at moderate sparsity**: At 9 views on real data, MBIR (26.00 dB) outperforms DINR (25.22 dB), suggesting handcrafted priors remain competitive when data constraints are less extreme.
- **Global metrics may underestimate DINR's advantage**: MBIR achieves relatively high full-image PSNR at 5/9 views due to smooth background regions despite poor overall visual quality; ROI analysis exposes this metric bias.

## Highlights & Insights

- **Modular prior injection**: By exploiting the DIS-agnostic property of the DD3IP framework, DINR integrates diffusion priors into INR via a proximal loss in a plug-and-play manner — an elegant and principled design.
- **Synthetic pretraining → real inference**: A diffusion model trained solely on synthetic ellipsoid data effectively guides real concrete microstructure reconstruction, validating the OOD adaptation capability of SCD.
- **ROI analysis methodology**: The paper proposes a data-driven multi-scale ROI evaluation analogous to SNR growth curves in CT image quality assessment, revealing local advantages masked by global PSNR.
- **Practical value for scientific imaging**: Neutron CT is a representative low-flux imaging modality; DINR's success demonstrates the potential of the diffusion prior + INR paradigm for scientific imaging inverse problems.

## Limitations & Future Work

- **Hyperparameter sensitivity**: $\rho$ and $\omega$ require manual tuning or parameter search, with different configurations needed for different datasets and sparsity levels.
- **Underperforms MBIR at moderate sparsity**: At 9 views on real data, MBIR still leads by 0.78 dB, suggesting the diffusion prior may introduce bias when data constraints are not extreme.
- **Limited validation scale**: Experiments are conducted only on $2 \times 256 \times 256$ volumes, with no extension to large-scale 3D reconstruction.
- **Absence of ablation experiments**: The contributions of FBP input to INR, the effect of the proximal term versus its removal, and the impact of different INR architectures are not quantified.
- **Inconsistency between SSIM and PSNR**: In some experiments, PSNR is highest but SSIM is not (e.g., at 5 views on real data, DINR SSIM=0.05 vs. DD3IP SSIM=0.06).
- **Computational cost not discussed**: The cost of jointly optimizing diffusion weights and INR weights at each timestep is not quantified.
- **Limited to parallel-beam geometry**: The method is not extended to cone-beam or helical CT acquisition geometries.

## Related Work & Insights

- **vs. DD3IP**: DINR replaces the CG solver with an INR, gaining continuous representation and resolution independence, consistently outperforming DD3IP across all four synthetic-data sparsity levels.
- **vs. pure INR (SIREN)**: Pure INR collapses at 4 views (14.76 dB); the diffusion prior proximal term effectively compensates for INR's low-frequency bias.
- **vs. MBIR (qGGMRF)**: MBIR requires exhaustive regularization parameter search ($10^{-4} \sim 10^6$) per sparsity level; DINR is more robust and outperforms MBIR in ultra-sparse scenarios.
- **Generality of proximal loss**: The paradigm of injecting generative priors into physics-driven optimization via proximal terms is transferable to other tomographic modalities such as X-ray CT and electron CT.
- **Insight**: ROI scale analysis reveals that global metrics may severely underestimate algorithmic advantages in critical regions; scientific imaging evaluation should develop segmentation-based, task-driven metrics.

## Rating

- Novelty: ⭐⭐⭐⭐ — The proximal fusion of INR and diffusion prior is a meaningful contribution, though the core idea builds on modular substitution within the DD3IP framework.
- Experimental Thoroughness: ⭐⭐⭐ — Validated on both synthetic and real data, but volume scale is limited, ablation studies are absent, and performance is suboptimal at certain sparsity levels.
- Writing Quality: ⭐⭐⭐ — Method description is clear and derivations are complete, though the paper is relatively short and figures are compact.
- Value: ⭐⭐⭐⭐ — Offers tangible value for extremely sparse reconstruction in scientific imaging; the modular design has good extensibility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [\[CVPR 2026\] Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery](reparameterized_tensor_ring_functional_decomposition_for_multi-dimensional_data_.md)
- [\[CVPR 2026\] PointINS: Instance-Aware Self-Supervised Learning for Point Clouds](pointins_instance-aware_self-supervised_learning_for_point_clouds.md)
- [\[CVPR 2026\] MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation](mvggt_multimodal_visual_geometry_grounded_transformer_for_multiview_3d_referring.md)
- [\[CVPR 2026\] SASNet: Spatially-Adaptive Sinusoidal Networks for INRs](sasnet_spatially_adaptive_sinusoidal_networks_for_inrs.md)

<!-- RELATED:END -->
