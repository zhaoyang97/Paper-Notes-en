---
title: >-
  [Paper Note] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction
description: >-
  [ICCV 2025][3D Vision][Single-view human reconstruction] MoGA is proposed to reconstruct high-fidelity 3D Gaussian avatars from a single image by learning a generative 3D avatar prior and leveraging it as a strong constr…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Single-view human reconstruction"
  - "3D Gaussian Avatar"
  - "generative prior"
  - "model inversion"
  - "SMPL-X"
date: 2026-05-08
content_hash: 3f421997dc559f1a
---

# MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2507.23597](https://arxiv.org/abs/2507.23597)  
**Code**: [Project Page](https://zj-dong.github.io/MoGA/)  
**Area**: 3D Vision
**Keywords**: Single-view human reconstruction, 3D Gaussian Avatar, generative prior, model inversion, SMPL-X

## TL;DR

MoGA is proposed to reconstruct high-fidelity 3D Gaussian avatars from a single image by learning a generative 3D avatar prior and leveraging it as a strong constraint for initialization, regularization, and pose optimization, substantially outperforming existing methods.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Creating animatable, photorealistic avatars from a single image poses fundamental challenges:

**Multi-view diffusion inconsistency**: Existing methods rely on 2D diffusion models to synthesize unseen views, but the generated views are sparse and 3D-inconsistent, resulting in blurriness and artifacts.

**Limitations of SMPL prior**: Parametric body models provide only minimal clothed body geometry without appearance priors, failing to model complex clothing and hairstyles.

**Self-occlusion**: Self-occlusion in arm and hand regions leads to incomplete reconstruction.

The core insight of MoGA is to replace SMPL with a generative 3D avatar model as the human prior, providing simultaneous geometric and appearance constraints.

## Method

### Generative Avatar Prior Training

**Canonical-space Gaussian representation**: 2D Gaussians are parameterized on the UV map of SMPL-X:
- Gaussian centers as residuals over SMPL-X: $\mu_k = \hat{\mu}_k + \delta_{\mu k}$
- Each identity is represented by a latent code $X_i \in \mathbb{R}^{64 \times 64 \times 32}$
- A shared CNN decoder maps latent codes to UV maps

**Joint training**: Following the single-stage pipeline of SSDNeRF, the auto-decoder and latent diffusion model are optimized simultaneously:

$$\mathcal{L} = \lambda_{\text{rend}} \mathcal{L}_{\text{rend}}(\{X_i\}, \psi) + \lambda_{\text{diff}} \mathcal{L}_{\text{diff}}(\{X_i\}, \phi)$$

The rendering loss comprises L2, perceptual, and regularization losses computed on both RGB and normal images.

### Model Fitting (Test Time)

**Multi-view generation**: A pretrained multi-view diffusion model is used to generate 6 synthetic views from the single input image.

**Triple role of the generative prior**:

1. **Initialization**: A meaningful latent code is sampled from the prior via image-guided sampling:
    - At each denoising step $t$, rendering gradients $g$ are computed and added to the denoised output as a correction.

2. **Regularization**: Diffusion and rendering losses are jointly optimized with the diffusion model and decoder weights frozen:
   $$\min_X \lambda_{\text{rend}} \mathcal{L}'_{\text{rend}}(X) + \lambda'_{\text{diff}} \mathcal{L}_{\text{diff}}(X)$$

3. **Pose optimization**: SMPL parameters and camera pose are optimized via photometric rendering loss:
   $$\mathcal{L}_{\text{pose}} = \lambda_{l2} \mathcal{L}_{l2} + \lambda_{vgg} \mathcal{L}_{vgg} + \lambda_{mask} \mathcal{L}_{mask}$$

Latent code optimization and pose optimization are performed alternately to avoid local minima.

## Key Experimental Results

### Quantitative Comparison (THuman2.1 and CustomHuman)


### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ | P2S↓ | NC↑ |
|--------|-------|-------|--------|-----|------|-----|
| SIFU | 17.53 | 0.922 | 0.102 | 2.62 | 2.45 | 0.787 |
| SiTH | 19.40 | 0.934 | 0.080 | 2.24 | 1.85 | 0.808 |
| PSHuman | 19.96 | 0.935 | 0.078 | 1.41 | 1.23 | 0.837 |
| **MoGA** | **24.09** | **0.946** | **0.073** | **1.36** | **1.22** | **0.850** |

MoGA substantially leads in appearance metrics (PSNR gain of 4.1 dB) and also achieves notably superior geometric quality.

### CustomHuman Dataset


### Ablation Study

| Method | PSNR↑ | CD↓ | NC↑ |
|--------|-------|-----|-----|
| PSHuman | 18.67 | 1.92 | 0.828 |
| **MoGA** | **23.44** | **1.81** | **0.834** |

A PSNR improvement of approximately 4.8 dB demonstrates the generalization capability of the proposed method.

## Highlights & Insights

1. **Triple role of the generative prior**: Initialization avoids local minima; regularization ensures 3D consistency; pose optimization improves alignment accuracy.
2. **Handling self-occlusion**: The 3D appearance prior effectively completes occluded regions such as arms and hands.
3. **Topological flexibility**: Being Gaussian-based rather than template-based, the method can reconstruct complex structures (e.g., ponytails) that deviate from the SMPL topology.
4. **Animatability**: The SMPL-X binding enables direct animation of the reconstructed avatar without post-processing.

## Limitations & Future Work

- The method depends on the quality of the pretrained multi-view diffusion model.
- Training data consist of 3D scans, which are limited in quantity.
- Generalization to extreme clothing and rare poses remains challenging.
- Inference requires an optimization process and is not feed-forward.

## Related Work & Insights

- PIFu, SIFU: Data-driven single-view reconstruction
- PSHuman, SiTH: Multi-view diffusion-assisted reconstruction
- GGHead, AG3D: 3D human GANs

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (generative prior as multi-role constraint for model fitting)
- Technical Depth: ⭐⭐⭐⭐⭐ (complete design integrating initialization, regularization, and pose optimization)
- Experimental Thoroughness: ⭐⭐⭐⭐ (quantitative + qualitative + ablation + in-the-wild images)
- Value: ⭐⭐⭐⭐ (significant PSNR improvement with practical applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GAS: Generative Avatar Synthesis from a Single Image](gas_generative_avatar_synthesis_from_a_single_image.md)
- [\[ICCV 2025\] GUAVA: Generalizable Upper Body 3D Gaussian Avatar](guava_generalizable_upper_body_3d_gaussian_avatar.md)
- [\[ICCV 2025\] DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior](driving_view_synthesis_on_free-form_trajectories_with_generative_prior.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)

</div>

<!-- RELATED:END -->
