---
title: >-
  [Paper Note] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models
description: >-
  [ICCV2025][3D Vision][multi-view generation] This paper presents SpinMeRound, an identity-conditioned multi-view diffusion model that generates 360° full-head portraits with consistent identity and corresponding normal maps from a single or few face images, surpassing existing multi-view diffusion methods on face novel view synthesis benchmarks.
tags:
  - "ICCV2025"
  - "3D Vision"
  - "multi-view generation"
  - "diffusion models"
  - "face novel view synthesis"
  - "identity preservation"
  - "normal estimation"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: b960c8479ca66f61
---

# SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models

**Conference**: ICCV2025  
**arXiv**: [2504.10716](https://arxiv.org/abs/2504.10716)  
**Code**: Not yet released  
**Area**: 3D Vision  
**Keywords**: multi-view generation, diffusion models, face novel view synthesis, identity preservation, normal estimation, 3D Gaussian Splatting

## TL;DR

This paper presents SpinMeRound, an identity-conditioned multi-view diffusion model that generates 360° full-head portraits with consistent identity and corresponding normal maps from a single or few face images, surpassing existing multi-view diffusion methods on face novel view synthesis benchmarks.

## Background & Motivation

Generating high-quality head portraits from arbitrary viewpoints given a single face image is a long-standing challenge in computer vision. The main difficulties arise from the following aspects:

**Scarcity of 3D face data**: Large-scale multi-view full-head datasets are extremely limited, constraining model training.

**Limitations of traditional methods**: 3DMM-based methods can only model the facial region and fail to handle complex structures such as hair; NeRF-based methods (e.g., PanoHead) produce poor back-of-head synthesis and suffer from difficult inversion on in-the-wild images.

**Deficiencies of existing diffusion models**:
   - General-purpose multi-view diffusion models (e.g., Cat3D) are not optimized for faces and exhibit uncanny valley effects.
   - Zero123-series models produce low-quality outputs with poor multi-view consistency.
   - DiffPortrait3D can only synthesize near-frontal views.
   - Era3D/Morphable Diffusion support only fixed viewpoints.
   - Video diffusion models (SV3D) incur high computational costs and are constrained to specific camera trajectories.

The authors argue that a face-specific multi-view diffusion approach is needed—one that maintains identity consistency while generating high-fidelity views covering the entire head.

## Method

### Overall Architecture

SpinMeRound is built upon a latent-space multi-view UNet, with three key components: an identity conditioning mechanism, a multi-view diffusion model, and a novel view sampling strategy.

### 1. Identity Conditioning Mechanism

- A pretrained ArcFace network is used to extract an identity embedding $\mathbf{w} \in \mathbb{R}^{512}$.
- The Arc2Face injection scheme is adopted: the text prompt "a photo of \<\<id\>\> person" is constructed, with the identity embedding replacing the corresponding token.
- After processing through the CLIP text encoder, a conditioning vector $\mathbf{c} \in \mathbb{R}^{N \times 768}$ is obtained.

### 2. Multi-View Diffusion Model

**Input representation**: The model jointly processes $P = (M + K) = 8$ pairs of face images and their normal maps, where $M \in \{1, 3\}$ denotes the number of conditioning views and $K$ the number of target views.

**Encoding pipeline**:
- The SD1.5 pretrained AutoEncoder encodes images and normal maps into the latent space: $\mathbf{z} \in \mathbb{R}^{4 \times 64 \times 64}$.
- Image latents and normal latents are concatenated channel-wise.
- Each view's latent is further concatenated with a ray coordinate map $\mathbf{r} \in \mathbb{R}^{149 \times 64 \times 64}$ (encoding ray origins and directions).
- A binary mask $\mathbf{m} \in \{0, 1\}^{1 \times 64 \times 64}$ is appended to distinguish conditioning views from target views.

**Network architecture**:
- Initialized from Arc2Face.
- 3D attention layers (following Cat3D) are inserted between the original 2D self-attention layers to enable cross-view information sharing.
- Input/output convolutional layer channels are expanded to accommodate normal maps and camera information.

### 3. Training Strategy

- Training follows the EDM framework in two stages:
    - **Stage 1** (600k iterations): single conditioning view.
    - **Stage 2** (additional 1M iterations): 0, 1, or 3 conditioning views are randomly selected, each with probability 1/3.
- White backgrounds are replaced with random colors with 50% probability to enhance data diversity.
- CFG training: the identity vector is replaced with an empty string and conditioning images with zero images with probability $P_{uncond} = 0.15$.
- Log-SNR shift of $\log(N)$, where $N = 7$ is the number of target views.

### 4. Three-Step Sampling Strategy (Single Image Input)

Given a single in-the-wild image, a three-step strategy is employed to generate consistent views covering the full head:

**Step A — Alignment and Normal Generation**:
- The input image is cropped and aligned using the PanoHead alignment procedure.
- Normal map generation is formulated as a channel-wise inpainting task, using conditioning-guided sampling to obtain the corresponding normal map.

**Step B — Anchor View Generation**:
- Seven anchor images are generated, covering $\pm 45°, \pm 90°, \pm 135°, 180°$ for full 360° coverage.

**Step C — Intermediate View Generation**:
- Using the input image and the two nearest anchor images as a conditioning triplet, arbitrary intermediate views are synthesized.
- 48, 88, or more views can be generated depending on the angular step size.

Sampling uses the EDM sampler with 50 steps and a guidance scale of 3.

### 5. Training Dataset

- Approximately 7k synthetic identities are generated using PanoHead (manually filtered from ~10k to remove samples with back-of-head artifacts).
- 125 viewpoints of images and normal maps are rendered per identity.
- Shapes are extracted from tri-plane feature maps via marching cubes, and normals are rendered using PyTorch3D.

## Key Experimental Results

### Quantitative Comparison (NeRSemble Dataset, 222 identities, 16 angles)

| Method | L2↓ | LPIPS↓ | SSIM↑ | ID Sim (ArcFace)↑ | ID Sim (VGGFace)↑ |
|------|------|--------|-------|---------------------|---------------------|
| EG3D (NeRF) | 0.025 | 0.4 | 0.55 | 0.31 | 0.89 |
| PanoHead (NeRF) | 0.012 | 0.32 | 0.65 | 0.27 | 0.88 |
| Zero123 | 0.195 | 0.515 | 0.55 | 0.169 | 0.44 |
| Zero123-XL | 0.198 | 0.51 | 0.563 | 0.118 | 0.442 |
| SV3D | 0.087 | 0.41 | 0.660 | 0.36 | 0.881 |
| DiffPortrait3D | 0.1 | 0.5 | 0.35 | 0.55 | 0.887 |
| **SpinMeRound** | **0.033** | **0.3** | **0.73** | **0.61** | **0.911** |

SpinMeRound achieves state-of-the-art performance on LPIPS, SSIM, and both identity similarity metrics, with L2 distance comparable to NeRF-based methods.

### Ablation Study

| Variant | L2↓ | LPIPS↓ | SSIM↑ |
|------|------|--------|-------|
| No input image (ID embedding only) | 0.1246 | 0.4299 | 0.568 |
| No identity embedding | 0.028 | 0.26 | 0.70 |
| No normal generation | 0.056 | 0.32 | 0.65 |
| **Full SpinMeRound** | **0.018** | **0.22** | **0.75** |

All three components—input image, identity embedding, and normal generation—contribute significantly to the final performance.

### Additional Capabilities
- **Unconditional sampling**: By setting an empty identity embedding under the CFG training scheme, multi-view images of entirely new identities can be generated.
- **3D reconstruction**: Feeding 48 generated views into 3DGS yields consistent 3D head reconstructions.
- **Identity interpolation**: Linear interpolation in the identity embedding space enables smooth identity morphing.

## Highlights & Insights

1. **Elegant combination of identity embedding and multi-view diffusion**: Injecting ArcFace identity features into the diffusion process leverages the generalization capacity of face recognition models while ensuring cross-view identity consistency.
2. **Joint normal map generation**: Simultaneously generating RGB images and normal maps provides a 3D shape prior; ablation studies confirm that this design significantly improves consistency and detail quality.
3. **Three-step anchor sampling strategy**: This approach elegantly addresses the model's limitation of generating only a finite number of views per inference, achieving arbitrary-density 360° coverage by first generating anchor views and then interpolating intermediate ones.
4. **Generalization to in-the-wild images despite purely synthetic training**: Trained on only ~7k synthetic identities generated by PanoHead, the model achieves state-of-the-art results on the real NeRSemble dataset and in-the-wild images.
5. **Support for unconditional generation and identity interpolation**: The CFG training scheme naturally enables additional generative flexibility.

## Limitations & Future Work

1. **Dependence on PanoHead for training data**: The quality ceiling of the synthetic data is bounded by PanoHead, particularly for back-of-head regions; access to real multi-view data or more advanced synthesis tools could further improve performance.
2. **Static head only**: Expression variation and dynamic modeling are not supported, precluding the generation of animatable avatars (concurrent works such as Pippo/DiffPortrait360 have partially explored this direction).
3. **Resolution constrained by SD1.5**: The 512×512 resolution of SD1.5 limits output quality; upgrading to a stronger base model (e.g., SDXL/SD3) may yield substantial quality improvements.
4. **Sampling efficiency**: Generating 48 views requires multiple rounds of 50-step sampling, leaving room for optimization.
5. **Limited number of training identities**: Approximately 7k identities may constrain generalization; scaling up the synthetic data is worth exploring.
6. **No comparison with the latest closed-source methods**: Cat3D is closed-source, precluding a fair direct comparison.

## Related Work & Insights

- **Cat3D [Gao et al., 2024]**: A general-purpose multi-view diffusion framework; this work adopts its 3D attention layers and camera encoding scheme, but specializes them for faces by adding identity conditioning and normal map generation.
- **Arc2Face [Deng et al., 2019]**: Provides an elegant mechanism for injecting identity embeddings into diffusion models, used here for model initialization.
- **PanoHead [An et al., 2023]**: A 360° full-head GAN used for training data generation and as the source of the alignment algorithm.
- **Zero123 / SV3D**: Representative multi-view/video diffusion baselines, comprehensively surpassed on face reconstruction metrics.
- **3DGS [Kerbl et al., 2023]**: Used to validate multi-view consistency, demonstrating that the generated views can directly support high-quality 3D reconstruction.

**Implications for future research**: The paradigm of integrating domain-specific priors (identity embeddings) into general-purpose multi-view diffusion frameworks can be extended to other object categories (e.g., vehicles, buildings). Joint generation of auxiliary modalities (normal maps) is also an effective strategy for improving cross-view consistency.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of identity embedding, joint normal generation, and anchor-based sampling is effective and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive quantitative and qualitative comparisons with thorough ablations, though a comparison with the closed-source Cat3D is absent.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete methodological exposition.
- Value: ⭐⭐⭐⭐ — Advances the state of the art in full-head novel view synthesis with direct applicability to 3D avatar construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[ICCV 2025\] MV-Adapter: Multi-view Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)
- [\[ICCV 2025\] FlexGen: Flexible Multi-View Generation from Text and Image Inputs](flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](auto-regressively_generating_multi-view_consistent_images.md)

</div>

<!-- RELATED:END -->
