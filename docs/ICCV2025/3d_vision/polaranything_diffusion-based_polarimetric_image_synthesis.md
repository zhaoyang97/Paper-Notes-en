---
title: >-
  [Paper Note] PolarAnything: Diffusion-based Polarimetric Image Synthesis
description: >-
  [ICCV 2025][3D Vision][Polarimetric image synthesis] This paper proposes PolarAnything, the first diffusion-based framework for generating polarimetric images from a single RGB image. By performing denoising diffusion over encoded AoLP and DoLP representations, the method achieves physically accurate and photorealistic polarimetric attribute synthesis without requiring 3D assets or polarization cameras.
tags:
  - ICCV 2025
  - 3D Vision
  - Polarimetric image synthesis
  - diffusion model
  - AoLP
  - DoLP
  - Shape from Polarization
  - Stable Diffusion
date: 2026-05-08
content_hash: 21f3aeef20bfcac1
---

# PolarAnything: Diffusion-based Polarimetric Image Synthesis

**Conference**: ICCV 2025
**arXiv**: [2507.17268](https://arxiv.org/abs/2507.17268)
**Code**: [Project Page](https://flzt11.github.io/PA_project/)
**Area**: 3D Vision / Polarimetric Imaging
**Keywords**: Polarimetric image synthesis, diffusion model, AoLP, DoLP, Shape from Polarization, Stable Diffusion

## TL;DR

This paper proposes PolarAnything, the first diffusion-based framework for generating polarimetric images from a single RGB image. By performing denoising diffusion over encoded AoLP and DoLP representations, the method achieves physically accurate and photorealistic polarimetric attribute synthesis without requiring 3D assets or polarization cameras.

## Background & Motivation

**Value and Pain Points of Polarimetric Imaging**: Polarimetric images provide unique physical cues for tasks such as shape-from-polarization (SfP), dehazing, and reflection removal—AoLP encodes surface normal information while DoLP reflects material properties. However, polarization cameras are expensive and rarely accessible, leading to:

**Dataset Scarcity**: Existing real polarimetric datasets are either extremely small (Morimatsu: 40 samples; Qiu: 38 samples) or scene-limited (HAMMER/HouseCat6D focus on indoor 6D pose estimation).

**Simulator Limitations**: Physics-based renderers such as Mitsuba require complete 3D assets (meshes, PBR materials, and environment lighting), and the gap between parametric pBRDF models and real polarimetric attributes prevents large-scale synthesis of realistic polarimetric images.

**Learning-based Method Constraints**: Learning-based approaches such as DeepSfP are trained on only 236 images, severely limiting generalization.

**Core Insight**: RGB images are easy to acquire and naturally cover diverse scenes; pretrained diffusion models (Stable Diffusion) have learned strong image priors from large-scale data and can be transferred zero-shot to discriminative tasks. Can diffusion priors be leveraged to directly generate polarimetric attributes from a single RGB image?

## Method

### Overall Architecture

PolarAnything is built on Stable Diffusion v1.5 via fine-tuning and consists of four modules:

- **Image Condition Encoder** $\mathcal{E}_{\text{img}}$: Convolutional layers with SiLU activations forming a multi-scale feature encoder isomorphic to the U-Net encoder, extracting conditional features from RGB images.
- **VAE Encoder** $\mathcal{E}_{\text{vae}}$: Maps encoded AoLP and DoLP to the latent space (weights frozen).
- **Denoising U-Net** $\mu_\theta$: Performs hierarchical ControlNet-style conditioning with the extracted features to denoise polarimetric attributes in the latent space.
- **VAE Decoder** $\mathcal{D}_{\text{vae}}$: Reconstructs AoLP and DoLP maps from the denoised latent codes (weights frozen).

Inference pipeline: RGB image → $\mathcal{E}_{\text{img}}$ extracts conditions → iterative denoising from Gaussian noise → decoding to AoLP and DoLP → synthesis of polarimetric images at arbitrary polarization angles via Eq. (1).

### Key Designs: Polarimetric Information Encoding Strategy

This constitutes the primary technical contribution of the paper. The authors compare three diffusion targets:

| Strategy | Diffusion Target | Issue |
|----------|-----------------|-------|
| (a) Direct polarimetric image generation | $\mathbf{I}_{0°}, \mathbf{I}_{45°}, \mathbf{I}_{90°}, \mathbf{I}_{135°}$ | Retains RGB radiance information but fails to recover polarimetric attributes; lacks physical constraints. |
| (b) Direct AoLP+DoLP generation | $\Phi, \mathbf{P}$ | AoLP has a domain of $[-90°, 90°]$ with $\pi$-periodicity; direct regression disrupts periodic structure. |
| (c) Encoded AoLP+DoLP ✓ | $[\cos 2\Phi; \sin 2\Phi; \mathbf{P}]$ | Sinusoidal encoding preserves periodicity and continuity; DoLP normalized to $[-1,1]$, compatible with VAE. |

**Why Encoding Works**: The $\pi$-periodicity of AoLP means that $-90°$ and $90°$ represent the same polarization direction; direct regression causes large errors at the boundary. The sinusoidal encoding $(\cos 2\Phi, \sin 2\Phi)$ maps angles to a continuous representation on the unit circle, naturally handling the periodicity.

### Loss & Training

Standard denoising diffusion loss:

$$\min_\theta \mathbb{E}_{x \sim \mathcal{E}_{\text{vae}}, t, \epsilon \sim \mathcal{N}(0,1)} \|\epsilon_t - \mu_\theta(z_t, t, c, \mathcal{E}_{\text{img}}(\mathbf{I}_{\text{RGB}}))\|_2^2$$

where $c$ denotes the CLIP text embedding and $z_t$ is the noised latent code. Notably, all U-Net weights are trainable—unlike ControlNet, which freezes the original weights—and experiments confirm this yields better performance.

### Dataset Construction

The authors collect 1,148 high-quality polarimetric images (1224×1024) using a polarization camera, covering:
- **100+ object categories**: transparent/opaque, conductor/insulator, diffuse/specular.
- **19 lighting environments**: 8 outdoor and 11 indoor.
- Combined with the Morimatsu dataset: 1,115 images for training and 33 for testing.

## Key Experimental Results

### Main Results: Polarimetric Attribute Synthesis Quality (Ablation)

| Diffusion Target | PSNR↑ | SSIM↑ | MAngE↓ | MAbsE↓ |
|-----------------|-------|-------|--------|--------|
| Polarimetric images (a) | 23.23 | 0.9165 | 45.67 | 0.1233 |
| AoLP+DoLP (b) | 40.57 | 0.9904 | 29.46 | 0.1100 |
| **Encoded AoLP+DoLP (c)** | **41.74** | **0.9927** | **25.33** | **0.1075** |

The encoding strategy achieves the best performance across all metrics: PSNR improves by nearly 19 dB over strategy (a), and MAngE decreases by 44% over strategy (a).

### Downstream Task: Monocular SfP (DeepSfP Enhancement)

| Test Set | Training Set | Mean↓ | Median↓ | RMSE↓ | ≤10°↑ | ≤20°↑ | ≤30°↑ |
|----------|-------------|-------|---------|-------|-------|-------|-------|
| DP+PN | DeepSfP original | 22.21 | 18.13 | 26.88 | 24.68 | 57.39 | 76.07 |
| DP+PN | +Mitsuba (MSO) | 20.42 | 17.19 | 24.31 | 23.23 | 61.32 | 80.67 |
| DP+PN | **+PolarAnything (PSO)** | **20.13** | **16.81** | **24.15** | **24.47** | **62.00** | **81.92** |

Augmenting the training set with 300 PolarAnything-synthesized images (PolarStanford-ORB) outperforms Mitsuba-rendered data of the same scale. Gains are more pronounced on the newly collected real test set PN (Mean: 30.69 → 22.93).

### Multi-view SfP (PISR Evaluation)

| Input Data | MAngE (Normal)↓ | CD (Mesh)↓ | MAngE (AoLP) | MAbsE (DoLP) |
|-----------|----------------|-----------|-------------|-------------|
| Real polarimetric images | 15.45 | 0.6765 | N/A | N/A |
| PolarAnything generated | 15.17 | 0.6564 | 33.68 | 0.1563 |

3D reconstruction quality using generated images is on par with, and in some metrics slightly superior to, real polarimetric images.

### Key Findings

1. **Encoding strategy is central**: Directly generating polarimetric images loses physical polarimetric information; sinusoidal AoLP encoding is essential for handling periodicity.
2. **Full fine-tuning outperforms ControlNet-style freezing**: Unlocking all U-Net weights yields better results than freezing them.
3. **Diffusion model vs. Transformer**: Restormer performs noticeably worse in low-data regimes; the zero-shot prior of diffusion models offers a significant advantage when training data is scarce.
4. **Strong generalization**: The model generates reasonable polarimetric attributes on out-of-distribution datasets (NeRSP, PANDORA), covering diffuse, metallic, and transparent materials.

## Highlights & Insights

1. **Precise problem formulation**: The paper reframes polarimetric image synthesis from a "rendering problem requiring 3D assets" to a "single-image conditional generation problem," substantially lowering the barrier to data acquisition.
2. **Elegant encoding design**: The sinusoidal encoding $(\cos 2\Phi, \sin 2\Phi)$ simultaneously resolves periodicity and is naturally compatible with the VAE's value range—an elegant integration of physical priors and deep learning.
3. **End-to-end practicality**: Beyond generating polarimetric images, the paper constructs the PolarStanford-ORB dataset and validates tangible gains on downstream tasks.
4. **Training efficiency**: Only 1,155 training images and 600 fine-tuning steps (8×A100, ~10 hours) demonstrate the high efficiency of diffusion prior transfer.
5. **RGB-to-polarization paradigm**: This work opens a new pathway for indirectly augmenting polarimetric training sets using large-scale RGB datasets, with broad value for all polarimetry-related computer vision tasks.

## Limitations & Future Work

1. **Grayscale limitation**: The current model converts RGB to grayscale before computing polarimetric attributes, discarding inter-channel polarization difference information.
2. **Training data scale**: 1,155 training images remain limited and may constrain generalization to complex scenes (large-scale outdoor environments, extreme lighting conditions).
3. **No explicit physical consistency constraints**: The loss function is the standard denoising objective without explicit enforcement of polarimetric physics (e.g., $\text{DoLP} \in [0,1]$, the Fresnel relationship between AoLP and surface normals), relying instead on the diffusion prior to implicitly learn these constraints.
4. **Evaluation limitations**: Quantitative comparison with Mitsuba is infeasible due to camera calibration errors; only qualitative comparisons are provided.
5. **Inherent ambiguity of single-image input**: Inferring polarimetric attributes from a single RGB image is an ill-posed problem, and a trade-off exists between diversity and accuracy.

## Related Work & Insights

- **Marigold / GeoWizard / StableNormal**: These works share the paradigm of "diffusion models for geometric prediction"; PolarAnything extends this to the polarimetric domain, demonstrating that the transfer potential of diffusion priors remains largely untapped.
- **ControlNet**: PolarAnything's conditional injection mechanism directly draws from ControlNet's hierarchical feature fusion, yet opts for full fine-tuning rather than freezing—a design choice that warrants further investigation into when freezing versus unlocking weights is preferable.
- **DeepSfP / PISR**: The data bottleneck of downstream SfP methods provides a clear application scenario for PolarAnything and suggests a "synthetic data + real fine-tuning" training paradigm.
- **Inspiration**: The approach of using pretrained generative models to synthesize data in scarce modalities is generalizable to thermal infrared, SAR, hyperspectral imaging, and other sensing modalities that similarly suffer from data scarcity.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|------------|-------|
| Novelty | 4 | First diffusion-based RGB→polarization generation framework; both problem formulation and encoding strategy are highly original. |
| Technical Depth | 3.5 | The polarimetric encoding design is clever, but the core architecture is primarily SD fine-tuning with limited architectural novelty. |
| Experimental Thoroughness | 4 | Ablation studies are clear; downstream task validation is comprehensive; qualitative results span diverse materials and scenes. |
| Value | 4.5 | Directly lowers the barrier to polarimetric data acquisition; PolarStanford-ORB dataset makes a tangible community contribution. |
| Writing Quality | 4 | Well-structured, with a smooth motivation narrative and carefully designed figures and tables. |
| **Overall** | **4.0** | A solid application-driven work; the encoding strategy is the core contribution, establishing a new paradigm for polarimetric data synthesis. |

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [\[ICCV 2025\] GAS: Generative Avatar Synthesis from a Single Image](gas_generative_avatar_synthesis_from_a_single_image.md)
- [\[ICCV 2025\] From Image to Video: An Empirical Study of Diffusion Representations](from_image_to_video_an_empirical_study_of_diffusion_representations.md)
- [\[CVPR 2026\] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](../../CVPR2026/3d_vision/dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)
- [\[ICCV 2025\] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)

<!-- RELATED:END -->
