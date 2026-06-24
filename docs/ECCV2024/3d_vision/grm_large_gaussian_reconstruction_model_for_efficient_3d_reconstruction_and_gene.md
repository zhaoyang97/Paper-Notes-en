---
title: >-
  [Paper Note] GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation
description: >-
  [ECCV 2024][3D Vision][3D Gaussian] This paper proposes GRM, a feed-forward 3D reconstruction model based on a pure Transformer architecture. It converts sparse-view (4 images) pixels into dense 3D Gaussian representations via pixel-aligned Gaussians, completing the reconstruction in approximately 0.1 seconds. Combined with multi-view diffusion models, it enables text-to-3D and image-to-3D generation.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian"
  - "sparse-view reconstruction"
  - "3D generation"
  - "Transformer"
  - "feed-forward reconstruction"
date: 2026-05-08
content_hash: a7004f87551f7b9c
---

# GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation

**Conference**: ECCV 2024  
**arXiv**: [2403.14621](https://arxiv.org/abs/2403.14621)  
**Code**: [https://justimyhxu.github.io/projects/grm/](https://justimyhxu.github.io/projects/grm/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian, sparse-view reconstruction, 3D generation, Transformer, feed-forward reconstruction

## TL;DR

This paper proposes GRM, a feed-forward 3D reconstruction model based on a pure Transformer architecture. It converts sparse-view (4 images) pixels into dense 3D Gaussian representations via pixel-aligned Gaussians, completing the reconstruction in approximately 0.1 seconds. Combined with multi-view diffusion models, it enables text-to-3D and image-to-3D generation.

## Background & Motivation

**Background**: 3D asset creation is in high demand in robotics, gaming, architecture, and other fields. Optimization-based 3D generation methods (such as SDS) yield high quality but take hours. Feed-forward methods are fast but mostly rely on the triplane-NeRF representation, which is limited by volume rendering efficiency and resolution.

**Limitations of Prior Work**: 
   - Optimization methods (SDS-based): A single 3D asset requires hours of optimization.
   - Feed-forward methods (e.g., LRM): Rely on triplane representation + volume rendering, suffering from low efficiency and limited resolution.
   - Concurrent works (LGM, Splatter Image): Use convolutional U-Net architectures, which generate a limited number of Gaussians.

**Key Challenge**: How to generate a sufficient number of high-quality 3D Gaussians for high-fidelity reconstruction while maintaining extremely fast inference?

**Goal**: Design a scalable feed-forward framework capable of efficiently generating a large number of pixel-aligned 3D Gaussians from sparse views, achieving high-quality, fast 3D reconstruction and generation.

**Key Insight**: Replace triplane-NeRF with 3D Gaussians, substitute CNN-based models with a pure Transformer architecture, and utilize window attention for efficient upsampling.

**Core Idea**: A Transformer encoder aggregates multi-view information, a Transformer upsampler generates high-resolution features, and pixel-aligned Gaussians constrain Gaussian positions along view rays.

## Method

### Overall Architecture

The pipeline of GRM:
1. **Input**: 4 sparse-view images + camera poses (can be derived from multi-view diffusion models).
2. **Transformer Encoder**: Extracts and fuses multi-view features.
3. **Transformer Upsampler**: Progressively upsamples features to the original resolution.
4. **Gaussian Property Prediction**: Linear heads predict depth, rotation, scaling, opacity, and SH coefficients.
5. **Back-projection**: Pixel-aligned 3D Gaussians are placed in 3D space along the view rays.
6. **Rendering**: Real-time rendering of arbitrary views via Gaussian Splatting.

### Key Designs

1. **Pixel-aligned Gaussians**:

    - **Function**: Constrain the positions of the 3D Gaussians along the input camera rays instead of freely predicting 3D coordinates.
    - **Mechanism**: The 3D position of each Gaussian is determined by the camera center and the ray direction:
    $\boldsymbol{\mu} = \mathbf{c}_o + \tau \mathbf{r}$
   where $\mathbf{c}_o$ is the camera center, $\mathbf{r}$ is the ray direction, and $\tau$ is the predicted depth value. Each view predicts an $H \times W \times 12$ Gaussian attribute map (depth, rotation, scale, opacity, and SH DC term), generating a total of $V \times H \times W$ 3D Gaussians.
    - **Design Motivation**: Directly predicting unconstrained 3D coordinates makes optimization difficult (as multiple configurations can yield the same visual result). Ray constraints establish a direct mapping from pixels to 3D space, lowering the learning difficulty.

2. **Transformer Encoder**:

    - **Function**: Extracts and fuses global features from multi-view images, enabling cross-view information exchange.
    - **Mechanism**: 
        - Plücker embeddings are utilized to inject camera information into each pixel.
        - A convolutional tokenizer (kernel/stride=16) extracts $H/16 \times W/16$ local features.
        - All view features are concatenated into a sequence of length $(V \times H/16 \times W/16)$.
        - 24 self-attention layers perform cross-view information exchange:
    $\mathbf{F} = E_{\theta, \phi}(\mathcal{I}, \mathcal{C})$
    - **Design Motivation**: Global self-attention is equivalent to cross-view feature matching, ensuring consistent predictions for the same 3D point across different views.

3. **Transformer Upsampler**:

    - **Function**: Progressively upsamples low-resolution feature maps to the original input resolution to recover high-frequency details.
    - **Mechanism**: Each upsampling block contains:
        - A Linear layer to expand the channel dimension by 4 times.
        - PixelShuffle for 2x spatial upsampling.
        - Window self-attention + shifted window self-attention (similar to Swin Transformer):
    $\mathbf{F} = \text{PixelShuffle}(\text{Linear}(\mathbf{F}), 2)$
    $\mathbf{F} = \text{SelfAttn}(\mathbf{F}, W)$
    $\mathbf{F} = \text{Shift}(\text{SelfAttn}(\text{Shift}(\mathbf{F}, W/2), W), -W/2)$
    - **Design Motivation**: The encoder's patchification process loses high-frequency details; CNN-based upsamplers fail to capture multi-view correspondences; window attention strikes a balance between computational efficiency and non-local information transfer.

### Loss & Training

- **Image Reconstruction Loss**: $\mathcal{L}_\text{img} = L_2(\mathbf{I}, \hat{\mathbf{I}}) + 0.5 L_p(\mathbf{I}, \hat{\mathbf{I}})$ (L2 + perceptual loss).
- **Opacity Mask Loss**: $\mathcal{L}_\text{mask} = L_2(\mathbf{M}, \hat{\mathbf{M}})$, to eliminate floaters.
- **Scale Activation Function**: Restrains the Gaussian scale within $[s_{min}, s_{max}]$:
$$\mathbf{s} = s_{min} \cdot \sigma(\mathbf{s}_o) + s_{max} \cdot (1 - \sigma(\mathbf{s}_o))$$
- **Training Data**: Objaverse 100K high-quality objects, with 32 random views rendered.
- **Training Configuration**: 32 × A100 GPUs, 4 days, $512 \times 512$ resolution, AdamW + cosine annealing.
- Deferred Back-propagation is used to optimize GPU memory.

## Key Experimental Results

### Main Results — Sparse-View Reconstruction (GSO Dataset, 100 Objects, 64 Views)

| Method | Input Views | PSNR↑ | SSIM↑ | LPIPS↓ | Time |
|------|-----------|-------|-------|--------|---------|
| GS (Optim.) | 4 | 21.22 | 0.854 | 0.140 | 9 min |
| IBRNet | 16 | 21.50 | 0.877 | 0.155 | 21 sec |
| SparseNeuS | 16 | 22.60 | 0.873 | 0.132 | 6 sec |
| LGM | 4 | 23.79 | 0.882 | 0.097 | 0.07 sec |
| MV-LRM | 4 | 25.38 | 0.897 | 0.068 | 0.25 sec |
| **GRM (Ours)** | **4** | **30.05** | **0.906** | **0.052** | **0.11 sec** |

### Main Results — Single Image to 3D (GSO Dataset, 250 Objects)

| Method | PSNR↑ | LPIPS↓ | CLIP↑ | FID↓ | Time |
|------|-------|--------|-------|------|-----|
| DreamGaussian | 19.19 | 0.171 | 0.862 | 57.6 | 2 min |
| One-2-3-45++ | 17.79 | 0.219 | 0.886 | 42.1 | 1 min |
| LGM | 16.90 | 0.235 | 0.855 | 42.1 | 5 sec |
| **GRM (Ours)** | **20.10** | **0.136** | **0.932** | **27.4** | **5 sec** |

### Ablation Study (Trained at 256 Resolution)

| Configuration | PSNR | SSIM | LPIPS | Description |
|------|------|------|-------|------|
| W/o Sigmoid scale activation | 24.43 | 0.638 | 0.133 | Exponential activation causes instability |
| + Sigmoid scale activation | 27.51 | 0.900 | 0.044 | +3dB, significant improvement |
| + 1 upsampling block | 29.11 | 0.922 | 0.037 | Upsampling improves details |
| + 3 upsampling blocks | 29.38 | 0.917 | 0.036 | More blocks further improve |
| + Alpha regularization | 29.48 | 0.920 | 0.031 | Eliminates floaters |
| Conv upsampler (alternative) | 27.23 | 0.894 | 0.063 | Much worse than Transformer upsampler |
| XYZ prediction (alternative) | 28.61 | 0.910 | 0.037 | Inferior to depth prediction |

### Key Findings

- GRM achieves a PSNR of up to 30.05 on sparse-view reconstruction, surpassing the runner-up MV-LRM by nearly **5 dB**, using only 4 input views (compared to 16 views required by competitors, which still yield inferior results).
- The inference time is only 0.11 seconds, generating **16 times** more Gaussians than LGM, dramatically leading in reconstruction fidelity.
- The Transformer upsampler outperforms the CNN upsampler by 2.25 dB in PSNR, demonstrating that cross-view attention is critical for detail reconstruction.
- The pixel-aligned design (depth prediction) outperforms free XYZ coordinate prediction by 0.87 dB, indicating that ray constraints effectively simplify learning.
- In user preference studies, text-to-3D generation surpasses the optimization method MVDream (29.5% vs. 25.9%), while being 450 times faster.

## Highlights & Insights

- **The Choice of 3D Representation is Crucial**: Replacing triplane-NeRF with 3D Gaussians simultaneously solves efficiency and quality bottleneck, offering real-time rendering and higher fidelity.
- **Advantages of Pure Transformer Architectures**: Self-attention is inherently suited for cross-view feature matching, whereas CNNs fail to capture multi-view correspondences.
- **The Simple Wisdom of Pixel-Aligned Constraints**: Constraining the coordinates to camera rays reduces the unconstrained 3D learning problem to 1D depth estimation, greatly easing the learning process.
- **Effect of Model Scaling**: The large-scale model consisting of a 24-layer encoder and 4 upsampling blocks demonstrates robust generalization when trained on large-scale data (100K objects).
- **Modular Design**: The reconstructor is plug-and-play and can be integrated with any multi-view diffusion model for text/image-to-3D generation.

## Limitations & Future Work

- Reconstruction quality degrades when input views are inconsistent (e.g., multi-view images generated by diffusion models may lack perfect consistency).
- The reconstructor is deterministic; future work could integrate probabilistic frameworks to handle multi-modal uncertainty.
- Only applicable to object-centric scenarios and cannot handle large-scale scenes.
- High training cost (32 × A100 GPUs for 4 days), making reproduction difficult.
- Only the DC term of SH (0-th order) is utilized, yielding limited view-dependent appearance effects.

## Related Work & Insights

- **Comparison with the LRM Series**: LRM uses triplane-NeRF representation, requiring expensive volume rendering, whereas GRM employs 3D Gaussians to achieve real-time rendering and higher fidelity.
- **Comparison with LGM**: LGM also uses 3D Gaussians but relies on a CNN U-Net architecture, leading to a smaller number of generated Gaussians (GRM generates 16 times more) and a significant quality gap.
- **Transfer of Swin Transformer Concepts**: Shifted window attention is used in 3D reconstruction for the first time as an efficient upsampler, balancing computation efficiency with non-local information.
- **Insights**: For structured output prediction problems, constraining the outputs to a geometrically reasonable space (e.g., ray directions) can significantly ease learning.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of pixel-aligned Gaussians and a pure Transformer upsampler is simple yet highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering sparse-view reconstruction, single image-to-3D, text-to-3D, user studies, and detailed ablation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, fair comparisons, and sufficient visualizations.
- Value: ⭐⭐⭐⭐⭐ Significantly outperforms existing methods in both speed and quality, acting as a milestone work in feed-forward 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [\[ECCV 2024\] AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation](animatabledreamer_text-guided_non-rigid_3d_model_generation_and_reconstruction_w.md)
- [\[ECCV 2024\] SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction](splatfields_neural_gaussian_splats_for_sparse_3d_and_4d_reconstruction.md)
- [\[ECCV 2024\] LaRa: Efficient Large-Baseline Radiance Fields](lara_efficient_large-baseline_radiance_fields.md)

</div>

<!-- RELATED:END -->
