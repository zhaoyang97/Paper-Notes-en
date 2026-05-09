---
title: >-
  [Paper Note] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models (EDA)
description: >-
  [CVPR2026][Medical Imaging][Diffusion Models] This paper proposes the EDA framework, which extends the EDM design space from Gaussian noise to arbitrary noise patterns by parameterizing a covariance matrix via a multivariate Gaussian distribution. EDA enables flexible noise diffusion and achieves performance at or above 100-step EDM methods and task-specific approaches using only 5 sampling steps across three tasks: MRI bias field correction, CT metal artifact reduction, and natural image shadow removal.
tags:
  - CVPR2026
  - Medical Imaging
  - Diffusion Models
  - Arbitrary Noise
  - Design Space
  - Image Restoration
  - MRI Bias Field Correction
  - CT Metal Artifact Reduction
  - Shadow Removal
date: 2026-05-08
content_hash: bfc8ba978ebeba43
---

# Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models (EDA)

**Conference**: CVPR2026
**arXiv**: [2507.18534](https://arxiv.org/abs/2507.18534)
**Code**: [PerceptionComputingLab/EDA](https://github.com/PerceptionComputingLab/EDA)
**Area**: Medical Imaging / Image Restoration
**Keywords**: Diffusion Models, Arbitrary Noise, Design Space, Image Restoration, MRI Bias Field Correction, CT Metal Artifact Reduction, Shadow Removal

## TL;DR

This paper proposes the EDA framework, which extends the EDM design space from Gaussian noise to arbitrary noise patterns by parameterizing a covariance matrix via a multivariate Gaussian distribution. EDA enables flexible noise diffusion and achieves performance at or above 100-step EDM methods and task-specific approaches using only 5 sampling steps across three tasks: MRI bias field correction, CT metal artifact reduction, and natural image shadow removal.

## Background & Motivation

**EDM is limited to Gaussian noise**: EDM unifies the design space of most diffusion models, but its forward process only supports pixel-wise independent Gaussian noise (covariance $\sigma^2(t)\mathbf{I}$), and cannot accommodate newer methods such as Flow Matching that support arbitrary noise diffusion.

**Forced injection of Gaussian noise degrades image restoration**: In restoration tasks, EDM-based methods must additionally superimpose Gaussian noise onto degraded images to initiate the reverse process, which destroys task-specific information present in the degraded input.

**Restoration distance is artificially lengthened**: The injection of Gaussian noise shifts the starting point of the reverse process away from the degraded image distribution, increasing the restoration distance and task complexity and requiring more sampling steps.

**SDE frameworks outperform ODEs**: Although Flow Matching provides a flexible ODE-based diffusion framework that breaks the Gaussian noise constraint, SDE-based methods demonstrate superior diversity and quality of results.

**Lack of a unified SDE design space**: No unified design space currently exists that both supports flexible noise patterns and retains the advantages of SDEs, which hinders the theoretical development of diffusion models.

**Need to initialize directly from degraded images**: Ideally, a noise pattern can be customized so that the reverse process starts directly from the known degraded image, shortening the restoration distance and reducing task complexity.

## Method

### Overall Architecture

EDA (Elucidating the Design space of Arbitrary-noise diffusion models) characterizes the diffusion process via a multivariate Gaussian distribution, generalizing the diagonal covariance $\sigma^2(t)\mathbf{I}$ in EDM to a covariance matrix $\boldsymbol{\Sigma}_{x_0} = H_{x_0}H_{x_0}^\top$ defined by an arbitrary set of basis functions, thereby supporting the diffusion and removal of arbitrary noise patterns.

### Key Designs

**Generalized forward process**: The diffusion noise in EDA is defined as

$$N = \sum_{m=1}^{M} \frac{\eta + \epsilon_m}{\eta + 1} h_{m, x_0}$$

where $H_{x_0} = [h_{1,x_0}, \ldots, h_{M,x_0}]$ is the set of basis functions governing the noise pattern, $\epsilon_m \sim \mathcal{N}(0,1)$ are independent Gaussian variables, and $\eta \geq 0$ controls the stochasticity of the noise ($\eta=0$ for maximum stochasticity; $\eta \to \infty$ approaches determinism).

**Multi-Wiener-process SDE**: The forward process is driven by multiple independent Wiener processes:

$$\mathrm{d}\boldsymbol{x} = [f(t)\boldsymbol{x} + \phi_{x_0}(t)]\mathrm{d}t + g(t)\sum_{m=1}^{M} h_{m,x_0} \mathrm{d}\omega_t^{(m)}$$

**Key theoretical results**:
- **Proposition 1**: EDA supports the diffusion and removal of arbitrary noise via three configurations covering all scenarios — a unified basis set (optimal case), sample-dependent basis functions (general case), and discrete sampling of non-Gaussian noise.
- **Proposition 2**: Generalizing from simple Gaussian noise to complex arbitrary patterns introduces no additional computational overhead — extra terms arising after solving the PFODE can be analytically simplified and eliminated, yielding a deterministic sampling formula identical to that of EDM.
- **Proposition 3**: EDM is a special case of EDA ($\eta=0$, with the basis set taken as the pixel-level identity matrix).

### Loss & Training

The same denoiser training objective as EDM is adopted:

$$\mathcal{L} = \mathbb{E}_{x_0 \sim P_{\text{data}}} \mathbb{E}_{x \sim P(x_t | y)} \| D_\theta(x; \sigma) - x_0 \|^2$$

The denoiser retains the skip-connection form of EDM: $D_\theta(x; \sigma) = c_{\text{skip}}(\sigma)x + c_{\text{out}}(\sigma)F_\theta(c_{\text{in}}(\sigma)x; c_{\text{noise}}(\sigma))$, where the network $F_\theta$ predicts the diffusion noise. Sampling employs deterministic first-order Euler integration.

## Key Experimental Results

### Experimental Setup

- **Framework**: PyTorch, single NVIDIA RTX 3090 GPU
- **Parameters**: $s(t)=1$, $\sigma = \sqrt{1-\bar{\alpha_t}}$, total training steps $T=100$
- **Three tasks**: MRI bias field correction (HCP dataset, 2206/1000 train/test slices), CT metal artifact reduction (DeepLesion, 1000/200 train/test images), natural image shadow removal (ISTD, 1330/540 train/test images)

### Main Results

#### MRI Bias Field Correction

| Method | SSIM ↑ | PSNR ↑ | COCO ↑ | CV(WM) ↓ |
|--------|--------|--------|--------|----------|
| N4 | 0.95 | 25.62 | 0.95 | 7.95 |
| ABCNet | 0.98 | 29.58 | 0.97 | 7.69 |
| Refusion (100 steps) | 0.98 | 34.67 | 0.98 | 7.72 |
| **EDA (5 steps)** | **0.99** | **38.02** | **0.99** | **7.40** |

#### Shadow Removal (ISTD)

| Method | ALL PSNR ↑ | ALL SSIM ↑ | NS PSNR ↑ | NS RMSE ↓ |
|--------|-----------|-----------|----------|----------|
| ShadowFormer | 31.81 | 0.967 | 33.89 | 3.90 |
| Refusion | 27.23 | 0.882 | 28.64 | 6.99 |
| **EDA** | **32.01** | **0.968** | **34.31** | **3.77** |

### Ablation Study

- **Sampling efficiency**: EDA with only 5 steps matches or exceeds Refusion at 100 steps, achieving approximately **53×** speedup (BFC task: 0.182 vs. 9.665 sec/slice).
- **ODE vs. SDE**: MeanFlow (ODE) performs significantly worse on all three restoration tasks, as ODE trajectories produce averaged solutions rather than high-fidelity restorations — CV(GM) is highest in BFC (15.49), artifact regions appear blurred in MAR, and ALL RMSE reaches 9.77 in shadow removal.
- **Image domain only vs. dual domain**: In CT MAR, EDA using only image-domain information surpasses several dual-domain methods (LI, CNNMAR, DSCMAR, etc.), though a gap remains with state-of-the-art dual-domain methods (InDuDoNet+, DICDNet).
- **Non-shadow region fidelity**: In shadow removal, EDA achieves a non-shadow-region PSNR of 34.31 dB and RMSE of 3.77, outperforming all competing methods, indicating the framework can precisely delineate shadow boundaries.

## Highlights & Insights

- Solid theoretical contributions: the paper rigorously proves that diffusion with arbitrary noise incurs no additional sampling computation, and that EDM is a special case of EDA.
- The reverse process is initialized directly from the degraded image, avoiding the information loss and increased distance caused by Gaussian noise injection.
- 5-step sampling achieves state-of-the-art performance, offering a 53× speedup over 100-step Refusion, with strong potential for clinical deployment.
- A unified treatment of noise flexibility and structural parameter flexibility within the SDE framework is achieved.

## Limitations & Future Work

- An inherent trade-off exists between stochasticity and applicability within the SDE framework: Case 1 (maximum stochasticity) applies only when the noise can be decomposed into a fixed basis set, while Cases 2–3 (more general but less stochastic) are closer to deterministic methods.
- Using only image-domain information in CT MAR leaves a performance gap relative to state-of-the-art dual-domain methods.
- Validation is limited to specific medical and natural image restoration tasks; experiments on other degradation types (e.g., super-resolution, deblurring) are absent.
- The selection of basis set $H_{x_0}$ relies on task-specific prior knowledge; an automated basis learning mechanism is lacking.

## Related Work & Insights

- **EDM** [Karras et al.]: Provides a unified design space but is restricted to Gaussian noise; the direct predecessor that EDA generalizes.
- **Flow Matching** [Lipman et al.]: An ODE framework supporting arbitrary distribution transport, but lacking the stochasticity advantages of SDEs.
- **MeanFlow** [Geng et al., 2025]: State-of-the-art single-step generation, but averaged trajectories lead to poor performance on restoration tasks.
- **Cold Diffusion** [Bansal et al.]: Replaces Gaussian noise with deterministic degradation operators but lacks a rigorous theoretical foundation.
- **Refusion** [CVPR NTIRE]: A representative Gaussian diffusion restoration method; its 100-step performance is surpassed by EDA in 5 steps.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Unifying the SDE design space for arbitrary-noise diffusion from a multivariate Gaussian perspective represents a valuable theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three restoration tasks spanning distinct noise types, covering both medical and natural images, with sufficient ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and figures are intuitive, though notation is somewhat dense in places.
- **Value**: ⭐⭐⭐⭐ — Provides a more efficient unified framework for diffusion-based restoration; the 53× speedup carries practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](eda_arbitrary_noise_diffusion_design_space.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[CVPR 2026\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](../../ICLR2026/medical_imaging/dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)

<!-- RELATED:END -->
