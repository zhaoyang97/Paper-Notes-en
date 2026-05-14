---
title: >-
  [Paper Note] OmniVTON: Training-Free Universal Virtual Try-On
description: >-
  [ICCV2025][Image Generation][virtual try-on] OmniVTON proposes the first training-free universal virtual try-on framework. By decoupling garment texture and pose conditions…
tags:
  - "ICCV2025"
  - "Image Generation"
  - "virtual try-on"
  - "training-free"
  - "diffusion model"
  - "garment warping"
  - "pose alignment"
date: 2026-05-08
content_hash: b8aba788ee6b488b
---

# OmniVTON: Training-Free Universal Virtual Try-On

**Conference**: ICCV2025
**arXiv**: [2507.15037](https://arxiv.org/abs/2507.15037)
**Code**: [GitHub](https://github.com/Jerome-Young/OmniVTON)
**Area**: Image Generation
**Keywords**: virtual try-on, training-free, diffusion model, garment warping, pose alignment

## TL;DR

OmniVTON proposes the first training-free universal virtual try-on framework. By decoupling garment texture and pose conditions, the method employs three core modules—Structured Garment Morphing (SGM), Continuous Boundary Stitching (CBS), and Spectral Pose Injection (SPI)—to achieve high-fidelity try-on in both in-shop and in-the-wild settings, while also supporting multi-person try-on for the first time.

## Background & Motivation

- **Problem Definition**: Image-based virtual try-on (VTON) requires seamlessly transferring a garment image onto a target human body while preserving texture consistency and pose fidelity.
- **Limitations of Prior Work**:
    - Supervised in-shop methods (GP-VTON, IDM-VTON, etc.) rely on paired training data and generalize poorly across domains.
    - Unsupervised in-the-wild methods (StreetTryOn) are constrained by data distribution bias and lack general applicability.
    - Both paradigms require training dedicated models for specific conditions; constructing large-scale datasets spanning diverse categories and poses is impractical.
- **Key Challenge**:
  1. **Fine-grained texture consistency**: Without a training phase, establishing garment–body alignment while preserving texture details is inherently difficult.
  2. **Human pose alignment**: Existing methods rely on keypoint or DensePose conditioning and require retraining for cross-modal feature fusion.
- **Goal**: To develop a unified, training-free VTON framework capable of generalizing across different domains and scenarios.

## Method

### Overall Architecture

OmniVTON adopts a two-stage pipeline built on off-the-shelf diffusion models without any additional training:
1. **Stage 1**: Warp the target garment to align with the human body via SGM to produce a garment prior.
2. **Stage 2**: Progressively inpaint and complete the final try-on image using the garment prior and pose-encoded noise through the CBS mechanism.

### Key Design 1: Structured Garment Morphing (SGM)

SGM leverages skeleton information and parsing maps to constrain garment deformation without retraining, making it applicable across different domains.

**Pseudo-person image generation**: For Shop-to-X scenarios (garment image only), a pseudo-person image is generated via attention modulation. Specifically, garment-conditioned noise and person-conditioned noise are denoised in parallel, with the K and V of the latter injected into the self-attention layers of the former:

$$f_c = \text{Softmax}\left(\frac{Q_c \cdot [K_c \| K_p]^\top}{\sqrt{d}}\right)[V_c \| V_p]$$

**Multi-part semantic correspondence**: Using the 25 keypoints from OpenPose, the human body is divided into $N$ semantic regions (e.g., for upper-body garments: torso, left/right upper arm, and left/right forearm—5 regions in total), establishing multi-part semantic correspondences between the target garment and the source person image. TAPPS is used to generate segmentation maps to isolate pixel regions.

**Localized transformation**: For each bounding-box corner pair, a homography matrix $\mathcal{H}_{o \to p}^i \in \mathbb{R}^{3 \times 3}$ is optimized using the Levenberg–Marquardt algorithm, followed by a piecewise perspective transformation:

$$\begin{bmatrix} x_o' \\ y_o' \\ 1 \end{bmatrix} = \sum_{i=1}^{5} \mathbb{I}_{\text{Region}_i}(x_o, y_o) H_{o \to p}^i \begin{bmatrix} x_o \\ y_o \\ 1 \end{bmatrix}$$

### Key Design 2: Spectral Pose Injection (SPI)

DDIM inversion preserves structural information from the source person but introduces source garment texture contamination. SPI addresses this via frequency-domain analysis:

1. Apply FFT and center-shift to the inverted noise $z_T^{inv}$ and random noise $z_T$.
2. Perform frequency-domain weighted fusion using a Gaussian low-pass mask $G_\tau$:
$$\hat{f}_T = G_\tau \odot f_T^{inv} + (1 - G_\tau) \odot f_T$$
3. Apply inverse FFT to obtain the blended initial noise $\hat{z}_T$.

**Core Idea**: Low frequencies retain the pose and structural information from the inverted noise, while high frequencies are replaced by random noise to eliminate texture residuals and enhance generation flexibility.

### Key Design 3: Continuous Boundary Stitching (CBS)

Multi-region stitching in SGM introduces texture discontinuities at boundaries. CBS improves boundary continuity through bidirectional semantic context:

- From the $I_c$ path to the $I_p'$ path: query $Q_p'$ matches target garment textures to bridge discontinuities.
- From the $I_p'$ path to the $I_c$ path: the similarity between attention maps of the two paths is enhanced while dissimilar values are suppressed.

### Loss & Training

OmniVTON is a training-free method and involves no loss function design. All components are realized through the inference pipeline of pretrained diffusion models.

## Key Experimental Results

### Main Results

**Quantitative comparison on VITON-HD** (all VTON methods use DressCode-pretrained models to evaluate cross-dataset generalization):

| Method | Year | FID_u ↓ | FID_p ↓ | SSIM_p ↑ | LPIPS_p ↓ |
|--------|------|---------|---------|----------|-----------|
| PBE | 2023 | 19.230 | 17.649 | 0.784 | 0.227 |
| AnyDoor | 2024 | 14.830 | 9.922 | 0.796 | 0.164 |
| GP-VTON | 2023 | 51.566 | 49.196 | 0.810 | 0.249 |
| IDM-VTON | 2024 | 23.035 | 20.460 | 0.812 | 0.147 |
| **OmniVTON** | — | **9.621** | **7.758** | **0.832** | **0.145** |

**Quantitative comparison on DressCode** (cross-garment-category adaptability):

| Method | FID_u ↓ | FID_p ↓ | SSIM_p ↑ | LPIPS_p ↓ |
|--------|---------|---------|----------|-----------|
| CAT-DM | 13.678 | 12.028 | 0.858 | 0.125 |
| IDM-VTON | 9.685 | 8.377 | 0.842 | 0.138 |
| **OmniVTON** | **6.450** | **5.335** | **0.865** | **0.119** |

### Ablation Study

| Variant | SGM | CBS | SPI | FID_u ↓ | FID_p ↓ | SSIM_p ↑ | LPIPS_p ↓ |
|---------|-----|-----|-----|---------|---------|----------|-----------|
| Base | — | — | — | 18.445 | 16.878 | 0.773 | 0.222 |
| (A) | ✓ | — | — | 13.303 | 11.475 | 0.809 | 0.177 |
| (B) | ✓ | ✓ | — | 9.799 | 7.993 | 0.824 | 0.158 |
| (C) | ✓ | — | ✓ | 13.148 | 10.767 | 0.813 | 0.180 |
| **OmniVTON** | ✓ | ✓ | ✓ | **9.621** | **7.758** | **0.832** | **0.145** |

### Key Findings

1. SGM alone reduces FID_u from 18.445 to 13.303, validating the effectiveness of training-free garment alignment.
2. CBS further improves LPIPS by 0.019 over SGM, enhancing perceptual quality.
3. SPI provides substantial SSIM and FID gains, effectively suppressing noise contamination while maintaining structural consistency.
4. FID_u on DressCode improves by 33.4% over the strongest baseline, demonstrating cross-garment-category adaptability.
5. OmniVTON achieves leading performance across all four cross-scenario settings on the StreetTryOn benchmark, even surpassing StreetTryOn trained on in-domain data.

## Highlights & Insights

1. **First training-free universal VTON framework**: Unifies in-shop and in-the-wild scenarios, eliminating the need to train dedicated models for specific conditions.
2. **Elegant decoupling strategy**: Garment texture preservation and pose alignment are decoupled into independent modules, avoiding the bias that arises when diffusion models simultaneously handle multiple conditions.
3. **Novel application of frequency-domain analysis**: SPI exploits the spectral properties of the latent space—low frequencies retain pose structure while high frequencies enhance generation flexibility—resulting in a conceptually elegant design.
4. **Multi-person try-on**: Enables try-on in multi-person scenes for the first time by concatenating multiple garments along the spatial dimension to generate pseudo-person images simultaneously.
5. **Strong cross-domain generalization**: Evaluated on VITON-HD using a DressCode-pretrained model, OmniVTON reduces FID by 5.209, demonstrating that the method is not reliant on any specific domain.

## Limitations & Future Work

1. Performance degrades in extreme cases, such as densely crowded scenes or scenarios where the target body region is very small, leading to garment alignment failures.
2. Multi-region stitching may still introduce artifacts at boundaries; CBS does not fully eliminate all discontinuities.
3. The method depends on multiple pretrained models (OpenPose, TAPPS, diffusion models), resulting in a relatively long inference chain.
4. The training-free inference paradigm may incur slower inference speeds due to iterative denoising and repeated attention modulation.

## Related Work & Insights

- **Garment warping**: The trend progresses from TPS (VITON) → optical flow (GP-VTON) → training-free skeleton-guided warping (this work), reflecting a shift toward reduced reliance on paired data.
- **Implicit warping VTON**: IDM-VTON and StableVITON model deformation implicitly via attention mechanisms but lack explicit geometric constraints.
- **Exemplar-guided inpainting**: PBE and AnyDoor offer generality but lack try-on-specific designs.
- **Inspiration from frequency-domain methods**: The frequency-domain modulation concept in SPI is potentially transferable to other image generation tasks requiring the disentanglement of structural and texture information.

## Rating ⭐⭐⭐⭐

The paper presents a highly innovative approach as the first training-free universal VTON framework, with a well-motivated design in which each module is validated through clear ablation studies. The decoupling strategy and frequency-domain analysis are conceptually elegant. Experiments span multiple datasets and scenarios, with strong quantitative and qualitative results. Multi-person try-on represents a meaningful extension. Limitations are primarily confined to extreme scenarios, and the overall work is of high quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](../../CVPR2026/image_generation/promo_promptable_virtual_tryon_efficient.md)
- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [\[CVPR 2026\] Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](../../CVPR2026/image_generation/garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features](introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)

</div>

<!-- RELATED:END -->
