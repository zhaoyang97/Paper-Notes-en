---
title: >-
  [Paper Note] FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads
description: >-
  [ICCV 2025][3D Vision][3D face reconstruction] This paper presents FaceLift, a single-image 360° high-quality 3D human head reconstruction method trained exclusively on synthetic data yet generalizing well to real-world images. It generates identity-consistent multi-view images via a multi-view latent diffusion model, then feeds them into a Transformer-based reconstructor to produce pixel-aligned 3D Gaussian representations.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D face reconstruction
  - 3D Gaussian splatting
  - synthetic data
  - multi-view diffusion
  - single-image reconstruction
  - identity preservation
date: 2026-05-08
content_hash: 96959ea788791448
---

# FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads

**Conference**: ICCV 2025
**arXiv**: [2412.17812](https://arxiv.org/abs/2412.17812)
**Code**: [weijielyu.github.io/FaceLift](https://weijielyu.github.io/FaceLift)
**Area**: 3D Vision
**Keywords**: 3D face reconstruction, 3D Gaussian splatting, synthetic data, multi-view diffusion, single-image reconstruction, identity preservation

## TL;DR

This paper presents FaceLift, a single-image 360° high-quality 3D human head reconstruction method trained exclusively on synthetic data yet generalizing well to real-world images. It generates identity-consistent multi-view images via a multi-view latent diffusion model, then feeds them into a Transformer-based reconstructor to produce pixel-aligned 3D Gaussian representations.

## Background & Motivation

Single-image 3D face reconstruction has been a central problem in computer vision and graphics for decades, with critical applications in VR/AR, VFX, games, digital entertainment, and telepresence systems. Achieving high-quality reconstruction from a single image poses a dual challenge: monocular reconstruction is highly ill-posed (a single 2D image corresponds to infinitely many 3D face shapes), and the human visual system is extremely sensitive to facial details, perceiving even subtle artifacts.

**Limitations of traditional methods**: 3DMM-based approaches (Blanz & Vetter 1999) rely on parametric textured mesh models, producing results that lack fine-grained geometric detail, realistic texture, and plausible hair. GAN-based methods (EG3D, PanoHead) improve generation quality, but EG3D supports only near-frontal view synthesis, while PanoHead, though capable of 360° views, does not provide a consistent 3D representation for multi-view rendering.

**Problems with RodinHD**: RodinHD attempts to train a triplane diffusion model on synthetic data to directly output 3D neural representations, but training exclusively on synthetic data leads to severe identity loss, with generated results on real images differing substantially from the input facial identity.

**Core insight of FaceLift**: Decouple identity preservation from 3D reconstruction into a two-stage pipeline — the first stage preserves identity in image space via a conditional diffusion model, and the second stage leverages a Transformer reconstructor pretrained on general objects to acquire rich geometric priors. Two key techniques bridge the synthetic-to-real domain gap: (1) input-view reconstruction strategy; (2) two-stage reconstructor training.

## Method

### Overall Architecture

FaceLift is a feed-forward pipeline consisting of two core modules:

1. **Multi-view latent diffusion model $f_D$**: conditioned on a single frontal face image $y$ and CLIP text embeddings, it generates 6 identity-consistent views $\{X_0^1, X_0^2, \dots, X_0^N\}$ covering 360°.
2. **Transformer-based Gaussian reconstructor $f_G$**: fuses the 6-view images and corresponding Plücker ray coordinates $P^{1:N}$ into a set of pixel-aligned 3D Gaussians $\{G_i\}_{i=1}^{NHW}$, with each pixel encoding one 3D Gaussian.

The choice of 3D Gaussians over NeRF or meshes is motivated by the fact that Gaussians provide explicit volumetric primitives that better capture subtle facial geometry and fine details; their semi-transparent kernels naturally model hair strands and translucent effects.

### Key Designs

#### 1. Synthetic Human Head Dataset

High-quality 3D head assets are constructed using Blender: base high-quality artist-created 3D head meshes are augmented with fine-grained components including eyes, teeth, gums, facial hair, and scalp hair; skeletal rigging enables pose variation; blendshape deformation supports diverse expressions; PBR texture maps (albedo, normal, roughness, specular, SSS) are applied; and clothing assets are paired with each head model.

- Rendering settings: 512×512 resolution; 200 unique identities × 50 appearance variants (different hairstyles, skin tones, expressions, clothing, poses) = 10,000 combinations.
- Lighting conditions: ambient lighting and randomized HDR environment lighting (ablation studies confirm that diverse lighting is critical for handling shadows and highlights).
- Diffusion model training: 6 views rendered per subject.
- Reconstructor fine-tuning: 32 random viewpoints rendered per subject.

#### 2. View Selection

Given the input image azimuth $\alpha$, 6 views are generated: $\{\alpha, \alpha \pm 45°, \alpha \pm 90°, \alpha + 180°\}$, all at zero elevation. Six views represent the optimal trade-off — 4 views lose forehead information, while 8 views yield no significant visual improvement at increased computational cost.

#### 3. Multi-View Attention Mechanism

Standard 2D self-attention is extended to 3D cross-view attention. The input tensor of shape $B \times V \times H \times W \times C$ is reshaped to $B \times VHW \times C$, treating all spatial positions across views as a unified token sequence for self-attention. This enables the model to learn inter-view correlations and ensure consistent multi-view RGB generation.

#### 4. Input-View Reconstruction (Core Technique)

During training, the first generated view is constrained to share the same camera as the input image, i.e., the model reconstructs the input view. This seemingly simple design, combined with multi-view attention, substantially outperforms generating only novel views:

- **Without input-view reconstruction**: the model overfits to the identity distribution of synthetic training data, resulting in severe identity, expression, and facial detail loss on real images.
- **With input-view reconstruction**: the diffusion model is forced to faithfully preserve input identity features, significantly improving cross-domain generalization.

In essence, input-view reconstruction transforms the diffusion model from "imagining from scratch" to "faithfully extending from the input."

#### 5. Transformer Reconstructor and Two-Stage Training

The reconstructor is based on the GS-LRM architecture: multi-view images concatenated with Plücker ray coordinates are patchified into non-overlapping patches, mapped to tokens via a linear layer, processed through Transformer blocks (Pre-LayerNorm + multi-head self-attention + MLP + residual connections), and decoded into Gaussian parameters via a linear layer followed by unpatchification.

**Two-stage training**:
- **Pretraining**: trained on Objaverse general-object data to learn diverse geometry and texture priors → resolves unclear texture in fine facial regions (eyes, nose, ears).
- **Fine-tuning**: fine-tuned on synthetic head data to inject head-specific geometric structure knowledge → achieves smoother and more realistic facial reconstruction.

Training solely on synthetic head data yields insufficient geometric diversity and poor texture detail; training solely on general-object data lacks fine-grained understanding of facial structure.

### Loss & Training

The reconstructor is trained with a combination of **MSE loss and perceptual loss**. During training, 4 input views are randomly selected to reconstruct a total of 8 views (4 input + 4 novel). The diffusion model is trained with the standard DDPM noise prediction objective.

### Real-Image Inference

- A fixed FOV of 50° is adopted to match the training setup.
- An MTCNN face detector estimates face size and center.
- Images are scaled and cropped/padded to match the mean face size and center distribution of the training data.
- Total inference time: approximately **8 seconds** (1.5s preprocessing + 5.5s multi-view generation + <1s 3D Gaussian reconstruction).

## Key Experimental Results

### Main Results

**Cafca synthetic dataset (40 subjects, held-out from training set)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DreamSim↓ | ArcFace↓ |
|--------|-------|-------|--------|-----------|----------|
| GGHead | 10.35 | 0.7406 | 0.3636 | 0.3252 | 0.2681 |
| PanoHead | 10.72 | 0.7594 | 0.3351 | 0.2048 | 0.2183 |
| Dual Encoder | 10.78 | 0.7385 | 0.3922 | 0.2785 | 0.2421 |
| Era3D | 13.69 | 0.7230 | 0.3662 | 0.2892 | 0.2978 |
| LGM | 16.52 | 0.7933 | 0.3060 | 0.1552 | 0.2557 |
| Our MV+LGM | 14.13 | 0.7812 | 0.2956 | 0.1282 | 0.1767 |
| **FaceLift** | **16.61** | **0.7968** | **0.2694** | **0.1096** | **0.1573** |

**Ava-256 real-capture dataset (10 subjects, studio-captured real humans)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DreamSim↓ | ArcFace↓ |
|--------|-------|-------|--------|-----------|----------|
| Era3D | 14.77 | 0.7963 | 0.2538 | 0.2515 | 0.3721 |
| LGM | 14.05 | 0.8136 | 0.2476 | 0.1496 | 0.3142 |
| Our MV+LGM | 15.24 | 0.8213 | 0.2292 | 0.1093 | 0.2264 |
| **FaceLift** | **16.52** | **0.8271** | **0.2277** | **0.1065** | **0.1871** |

**Comparison with mesh-based methods (Cafca dataset)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DreamSim↓ | ArcFace↓ |
|--------|-------|-------|--------|-----------|----------|
| TRELLIS | 12.74 | 0.7412 | 0.3746 | 0.2170 | 0.4001 |
| Unique3D | 14.27 | 0.7643 | 0.3188 | 0.1277 | 0.2088 |
| InstantMesh | 16.44 | 0.7815 | 0.2792 | 0.1504 | 0.2741 |
| **FaceLift** | **16.61** | **0.7968** | **0.2694** | **0.1096** | **0.1573** |

### Ablation Study

| Variant | PSNR↑ | SSIM↑ | LPIPS↓ | DreamSim↓ | ArcFace↓ |
|---------|-------|-------|--------|-----------|----------|
| w/o input-view reconstruction | 16.02 | 0.7884 | 0.2893 | 0.1438 | 0.2367 |
| w/o multi-view attention | 16.29 | 0.7885 | 0.2861 | 0.1552 | 0.2126 |
| **Full Model** | **16.61** | **0.7968** | **0.2694** | **0.1096** | **0.1573** |

### Key Findings

1. **Comprehensive state-of-the-art**: FaceLift surpasses GAN-based (PanoHead, GGHead) and LRM-based (LGM, Era3D) methods across all metrics.
2. **Substantial lead in identity preservation**: ArcFace distance on Cafca is 0.1573 vs. LGM's 0.2557 (38% reduction); on Ava-256, 0.1871 vs. LGM's 0.3142 (40% reduction).
3. **Synthetic training, real generalization**: trained exclusively on synthetic data, FaceLift achieves state-of-the-art results on the real-world Ava-256 dataset.
4. **Modular compatibility**: combining FaceLift's multi-view diffusion with the existing LGM reconstructor (Our MV+LGM) also improves performance, demonstrating methodological flexibility.
5. **Input-view reconstruction is critical**: removing it degrades ArcFace distance from 0.1573 to 0.2367 (+50%), with qualitative severe identity loss.
6. **Mesh representations are suboptimal**: the semi-transparent kernels of 3D Gaussians significantly outperform mesh representations for hair and skin wrinkles.

## Highlights & Insights

- **A successful paradigm for synthetic-to-real generalization**: the key lies not in synthetic data per se, but in two carefully designed components — input-view reconstruction forces the model to be faithful to the input rather than hallucinating, while two-stage training compensates for the geometric diversity limitations of synthetic heads via general-object priors.
- **Underlying principle of input-view reconstruction**: without this constraint, the model only learns a mapping from "synthetic identity → multi-view outputs," overfitting to the training distribution. With input-view reconstruction enforced, the model must learn "arbitrary identity → faithful multi-view outputs," functioning as a form of self-supervised regularization.
- **Design rationale for 3D Gaussians over meshes**: for head reconstruction, Gaussians offer two key advantages: (1) semi-transparent kernels naturally represent hair and translucent effects; (2) pixel alignment allows each pixel to encode a Gaussian, yielding higher spatial resolution.
- **Potential for 4D extension**: supplementary material demonstrates the applicability of FaceLift to video sequences via an autoregressive deformation network for 4D novel-view synthesis, showcasing the method's extensibility even through simple post-processing.

## Limitations & Future Work

- **Limited data scale**: only 200 synthetic identities, augmented to 10,000 variants through 50 appearance variations, resulting in constrained identity diversity.
- **Fixed FOV assumption**: inference assumes FOV = 50°, which may fail for extreme perspective distortion, wide-angle, or fisheye lenses.
- **Static reconstruction only**: dynamic expression driving or pose variation (animatable avatars) is not supported.
- **Back-of-head hallucination**: rear views are entirely generated by the diffusion model, which may be unreliable for highly complex hairstyles (e.g., intricate braids or headwear).
- **Inference latency**: the 8-second total inference time, with 5.5 seconds spent on diffusion sampling, remains far from real-time deployment.
- **Insufficient lighting alignment**: complex real-world illumination (e.g., strong side lighting, backlighting) may adversely affect reconstruction fidelity.

## Related Work & Insights

- **Two-stage pipelines have become mainstream**: the paradigm of multi-view diffusion followed by 3D reconstruction has been widely adopted (Zero-1-to-3, Era3D, Wonder3D, etc.); FaceLift specializes this paradigm for the human head domain and demonstrates the necessity of domain-specific adaptation.
- **GS-LRM** provides the efficient reconstructor backbone for FaceLift, validating the transfer value of large-scale pretrained reconstruction models in domain-specific tasks.
- **Implications of synthetic-data-to-real-generalization**: this approach can be extended to other domain-specific 3D reconstruction tasks such as hands and full bodies; the key is identifying analogous techniques to input-view reconstruction for reducing the domain gap.
- Complementary to animatable methods such as HeadGAP and Morphable Diffusion: FaceLift focuses on static high-fidelity reconstruction, and future work could integrate expression-driving capabilities.

## Rating

- **Novelty**: ⭐⭐⭐ — The overall pipeline is a well-engineered combination of existing components (DDPM + GS-LRM + synthetic data); the input-view reconstruction strategy is a distinctive contribution but represents a single-point innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two quantitative benchmarks, multiple baselines spanning three categories (GAN/LRM/Mesh), in-the-wild qualitative evaluation, comprehensive ablations (view count, input reconstruction, two-stage training, lighting conditions, multi-view attention), and 4D extension experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, and rich visualizations.
- **Value**: ⭐⭐⭐⭐ — A successful demonstration of synthetic-to-real generalization with high practical value; 8-second reconstruction approaching usability.
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] A Recipe for Generating 3D Worlds from a Single Image](a_recipe_for_generating_3d_worlds_from_a_single_image.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)

</div>

<!-- RELATED:END -->
