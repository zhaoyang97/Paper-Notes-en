---
title: >-
  [Paper Note] DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration
description: >-
  [ICCV 2025][Human Understanding][Blind face restoration] This paper proposes DynFaceRestore, which reformulates blind degradation as a Gaussian deblurring problem via Dynamic Blur Level Mapping (DBLM)…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "Blind face restoration"
  - "diffusion model guidance"
  - "dynamic blur mapping"
  - "fidelity-quality balance"
  - "region-adaptive guidance"
date: 2026-05-08
content_hash: b963b0f276c207ca
---

# DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration

**Conference**: ICCV 2025
**arXiv**: [2507.13797](https://arxiv.org/abs/2507.13797)
**Code**: [Project Page](https://github.com/)
**Area**: Diffusion Models / Face Restoration
**Keywords**: Blind face restoration, diffusion model guidance, dynamic blur mapping, fidelity-quality balance, region-adaptive guidance

## TL;DR

This paper proposes DynFaceRestore, which reformulates blind degradation as a Gaussian deblurring problem via Dynamic Blur Level Mapping (DBLM), and achieves an optimal fidelity-quality trade-off during diffusion model sampling through a Dynamic Starting Step lookup table (DSST) and a Dynamic Guidance Scaling Adjuster (DGSA).

## Background & Motivation

Blind Face Restoration (BFR) aims to recover high-fidelity, detail-rich facial images from low-quality inputs with unknown degradation sources. The core challenge lies in simultaneously enhancing facial details and preserving identity consistency. While pre-trained diffusion models have been widely adopted as image priors for generating fine-grained details, existing methods exhibit three critical limitations:

**Fixed diffusion starting step**: Methods such as DifFace assume uniform degradation severity across all low-quality inputs and apply a fixed diffusion sampling starting timestep. This leads to "under-diffusion" (insufficient detail) for severely degraded images and "over-diffusion" (artifact introduction) for mildly degraded ones, as clearly demonstrated by the t-SNE visualization in Fig. 2 of the paper.

**Kernel mismatch**: Under blind settings, degradation kernel estimation is inherently imprecise, and real-world degradation kernels are highly complex (combining blur, downsampling, noise, JPEG compression, etc.). Modeling them as a single kernel introduces guidance bias during diffusion sampling and degrades restoration fidelity.

**Global guidance scaling**: Existing guidance-based methods (e.g., DPS, PGDiff) apply a uniform guidance scaling factor to all pixels. However, high-frequency regions (hair, wrinkles) benefit from stronger diffusion model influence for perceptual quality enhancement, whereas low-frequency regions (facial contours) require stronger observation guidance to preserve structural fidelity. A global scaling factor cannot reconcile this spatial conflict.

## Method

### Overall Architecture

The DynFaceRestore framework consists of three core components that collectively reformulate BFR as a Gaussian deblurring problem:
1. **DBLM** (Dynamic Blur Level Mapping): transforms unknown degradation inputs into Gaussian-blurred images
2. **DSST** (Dynamic Starting Step lookup table): determines the optimal diffusion starting step based on blur level
3. **DGSA** (Dynamic Guidance Scaling Adjuster): adaptively adjusts guidance intensity in a region-aware manner

The guided diffusion sampling process is formulated as:
$$x_{t-1} = x'_{t-1} - A_t \times \nabla_{x_t} \| \acute{y} - k_t \otimes x^0_t \|^2$$

### Key Designs

1. **Dynamic Blur Level Mapping (DBLM)**: transforms the blind degradation input $y$ into the Gaussian-blurred form $\acute{y} = k^{\hat{std^*}}_y \otimes RM(y)$, where $RM$ can be any pre-trained restoration model (SwinIR is used in this work). The core idea is not to perfectly estimate the complex degradation kernel, but to convert it into a known Gaussian kernel form, thereby providing reliable guidance during diffusion sampling. The optimal standard deviation $std^*$ is estimated by an SE network comprising two sub-modules: a Transfer Model (TM) and a Standard Deviation Estimator (SDE). A key advantage of DBLM is that even when $RM$ produces imperfect restorations, re-applying Gaussian blur confines the residual error within a controllable range, effectively mitigating the kernel mismatch problem.

2. **Dynamic Starting Step Lookup Table (DSST)**: based on the key observation that a high-quality image $x_0$ and its blurred counterpart $\tilde{y}^{std}_0$ statistically converge at some timestep $t$ during forward diffusion. This convergence point serves as the optimal guidance insertion step for the blurred observation $\acute{y}$. The formulation is:

    $t_{std} = \underset{t}{argmin} \left( log(\mathbf{X}_t) - log(\tilde{\mathbf{Y}}^{std}_t) \leq tol \right)$

   The optimal starting step for each $std$ value is pre-computed and stored as a lookup table. At inference, $t_{start}$ is retrieved directly using the SE-estimated $\hat{std^*}$, avoiding both under- and over-diffusion. Experiments show that the sampling range is reduced from a fixed 1000 steps to [690, 925].

3. **Dynamic Guidance Scaling Adjuster (DGSA)**: a lightweight CNN (3 convolutional layers) that outputs a region-wise guidance scaling map $A_t \in [0,1]$. Inputs include the current measurement $\acute{y}$, the high-quality prediction $x^0_t$, and timestep $t$. In high-frequency texture regions (hair, wrinkles), smaller $A_t$ values are produced (weakening guidance to allow the diffusion model to freely generate details), whereas in low-frequency structural regions (contours, skin), larger $A_t$ values are produced (strengthening guidance to preserve fidelity). Training employs stationary wavelet transform (SWT) sub-band supervision and DISTS perceptual loss:

    $L_{DGSA} = \sum_i \gamma_i \mathbb{D}(SWT(x^0_{t-1})_i, SWT(x_0)_i) + DISTS(x^0_{t-1}, x_0)$

### Loss & Training

- **SE network**: SDE is first pre-trained to estimate Gaussian blur levels, followed by end-to-end training of SE = TM + SDE
- **DGSA**: timestep $t$ is randomly sampled; SWT sub-band L1 loss + DISTS perceptual loss is applied
- **Kernel adaptive update**: the estimated kernel is updated during sampling as $std_{t-1} = std_t - s \nabla_{std_t} \| \acute{y} - k_t \otimes x^0_t \|^2$
- **Multi-guidance extension**: optionally generates 3 guided outputs at different blur levels ($\hat{std^*}$, $\hat{std^*}-1$, $\hat{std^*}-2$) and combines them with weighted aggregation to balance DM quality and RM fidelity
- The pre-trained diffusion model is shared with DifFace/PGDiff; training is conducted on the FFHQ dataset

## Key Experimental Results

### Main Results

**Quantitative comparison on CelebA-Test**:

| Type | Method | PSNR↑ | SSIM↑ | FID↓ | IDA↓ | LMD↓ |
|------|--------|-------|-------|------|------|------|
| GAN | GPEN | 23.77 | 0.659 | 30.25 | 0.837 | 6.377 |
| GAN | GFP-GAN | 22.84 | 0.620 | 23.86 | 0.822 | 4.793 |
| Codebook | CodeFormer | 23.83 | 0.637 | 18.08 | 0.775 | 3.509 |
| DM | DifFace | 23.95 | 0.659 | 15.03 | 0.867 | 3.781 |
| DM | DiffBIR | 24.13 | 0.647 | 19.19 | 0.767 | 3.535 |
| DM | 3Diffusion | 23.39 | 0.651 | 15.45 | 0.943 | 3.781 |
| **DM** | **DynFaceRestore** | **24.35** | **0.664** | **14.78** | **0.748** | **3.419** |

**FID comparison on real-world datasets**:

| Method | LFW↓ | WebPhoto↓ | Wider↓ |
|--------|------|----------|--------|
| CodeFormer | 52.35 | 83.19 | 38.80 |
| DAEFR | 47.53 | 75.45 | 36.72 |
| DiffBIR | 43.45 | 91.20 | 36.72 |
| **DynFaceRestore** | **42.52** | 95.32 | **36.05** |

### Ablation Study

**Component-wise ablation on CelebA-Test**:

| Setting | DBLM | Multi-guide | DSST | DGSA | PSNR↑ | FID↓ | IDA↓ | Sampling Range |
|---------|------|------------|------|------|-------|------|------|---------------|
| Baseline | | | | | 11.13 | 55.78 | 1.461 | 1000 |
| A | ✓ | | | | 24.99 | 18.30 | 0.725 | 1000 |
| C | ✓ | ✓ | ✓ | | 25.11 | 19.79 | 0.724 | [690,925] |
| E | ✓ | | ✓ | ✓ | 24.33 | **14.69** | 0.755 | [690,925] |
| **F** | **✓** | **✓** | **✓** | **✓** | **24.35** | 14.78 | **0.748** | [690,925] |

### Key Findings

- DBLM is the most critical component, lifting PSNR from 11.13 (baseline) to 24.99
- DGSA substantially improves perceptual quality (FID reduced from 19.79 to 14.78) at a marginal cost to PSNR
- DSST improves metrics while shortening the sampling range (1000 → [690, 925]), simultaneously boosting both quality and efficiency
- The method achieves state-of-the-art performance on multiple fidelity metrics (PSNR/SSIM/IDA/LMD) and perceptual quality (FID) simultaneously, successfully balancing the fidelity-quality trade-off

## Highlights & Insights

- **Elegance of problem reformulation**: The core insight of the approach is to transform the complex blind restoration problem into a tractable Gaussian deblurring problem. By introducing an intermediate Gaussian-blurred representation, reliable low-frequency information is preserved while a well-defined guidance form is provided to the diffusion model.
- **Pervasive dynamic vs. static design**: The three components (DBLM, DSST, DGSA) respectively achieve "dynamic" adaptation along three dimensions — degradation mapping, timestep selection, and guidance intensity — systematically addressing the one-size-fits-all limitations of prior methods.
- **Region-adaptive guidance**: The DGSA design captures the spatially non-uniform distribution of fidelity and quality requirements, representing an effective improvement over the DPS guidance formulation.

## Limitations & Future Work

- Inference time is substantially longer (91.82s) compared to CodeFormer (0.06s), limiting real-time applicability
- FID performance on the WebPhoto dataset is inferior to some competing methods
- Hyperparameters of multi-guidance (number of guides, $std$ spacing) require manual specification and lack adaptive mechanisms
- The generalizability of DGSA as a separately trained network remains to be further validated
- Jointly optimizing the kernel update process (Eq. 7) and DGSA is a promising direction for future work

## Related Work & Insights

- DPS (Diffusion Posterior Sampling) provides the theoretical foundation, upon which this work implements three key extensions
- The limitations of DifFace's fixed-step strategy directly motivated the design of DSST
- The choice of SwinIR as the restoration model reflects a pragmatic engineering philosophy of leveraging existing components rather than reinventing them

## Rating

- **Novelty**: ⭐⭐⭐⭐ The problem reformulation is elegant, and the three dynamic components are well-motivated and mutually complementary
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluations cover both synthetic and real-world datasets with comprehensive ablation studies
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear and mathematical derivations are rigorous
- **Value**: ⭐⭐⭐⭐ Makes a significant contribution to diffusion-guided restoration, though inference speed limits practical deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions](tridi_trilateral_diffusion_of_3d_humans_objects_and_interactions.md)
- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[ICCV 2025\] IDFace: Face Template Protection for Efficient and Secure Identification](idface_face_template_protection_for_efficient_and_secure_identification.md)
- [\[ICCV 2025\] PHD: Personalized 3D Human Body Fitting with Point Diffusion](phd_personalized_3d_human_body_fitting_with_point_diffusion.md)
- [\[ICCV 2025\] AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](adahuman_animatable_detailed_3d_human_generation_with_compositional_multiview_di.md)

</div>

<!-- RELATED:END -->
