---
title: >-
  [Paper Note] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][Large Reconstruction Model] This paper proposes GS-LRM, an extremely simple Transformer-based large reconstruction model that patchifies multi-view images and directly regresses per-pixel 3D Gaussian parameters through self-attention. It significantly outperforms SOTA in both object-level (surpassing Triplane-LRM by 4dB PSNR) and scene-level (surpassing pixelSplat by 2.2dB PSNR) reconstruction, completing inference in 0.23 seconds on a single A100 GPU.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Large Reconstruction Model"
  - "3D Gaussian Splatting"
  - "Transformer"
  - "Sparse-view Reconstruction"
  - "Feed-forward 3D Reconstruction"
date: 2026-05-08
content_hash: 61001e7b001f777a
---

# GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2404.19702](https://arxiv.org/abs/2404.19702)  
**Code**: [https://sai-bi.github.io/project/gs-lrm/](https://sai-bi.github.io/project/gs-lrm/)  
**Area**: 3D Vision  
**Keywords**: Large Reconstruction Model, 3D Gaussian Splatting, Transformer, Sparse-view Reconstruction, Feed-forward 3D Reconstruction

## TL;DR

This paper proposes GS-LRM, an extremely simple Transformer-based large reconstruction model that patchifies multi-view images and directly regresses per-pixel 3D Gaussian parameters through self-attention. It significantly outperforms SOTA in both object-level (surpassing Triplane-LRM by 4dB PSNR) and scene-level (surpassing pixelSplat by 2.2dB PSNR) reconstruction, completing inference in 0.23 seconds on a single A100 GPU.

## Background & Motivation

**Background**: 3D reconstruction is a core problem in computer vision. Traditional methods rely on dense multi-view inputs and complex SfM/MVS pipelines. Recently, NeRF and 3D Gaussian Splatting (3DGS) have achieved outstanding rendering quality under the per-scene optimization paradigm, but they are slow and require a large number of input views. Transformer-based Large Reconstruction Models (LRMs) achieve feed-forward sparse-view 3D reconstruction by training on large-scale 3D data.

**Limitations of Prior Work**: (1) Existing LRMs use Triplane NeRF representations, which are constrained by fixed low-resolution triplanes (typically 32) and struggle to retain high-frequency details; (2) Volume rendering is computationally expensive, limiting the training resolution; (3) Triplane NeRF is naturally suited only for object-level reconstruction and is difficult to scale to large-scale scenes; (4) The model designs are complex, requiring extra triplane tokens and meticulously designed decoders.

**Key Challenge**: Implementing a general, scalable, and efficient 3D reconstruction model requires simultaneously satisfying: high-quality detail preservation, fast rendering, and scale adaptability from objects to scenes. Triplane NeRF exhibits bottlenecks in all three dimensions.

**Goal** (1) How to design a simpler and more efficient LRM architecture? (2) How to enable the reconstruction model to handle both objects and scenes? (3) How to preserve high-frequency details while achieving fast rendering?

**Key Insight**: Replace Triplane NeRF with 3D Gaussian Splatting as the output representation, and design a pure Transformer model to directly predict per-pixel 3D Gaussian parameters. Pixel-aligned Gaussian prediction aligns 2D inputs and 3D outputs in the same space, simplifying the architecture while naturally supporting high-resolution inputs and large-scale scenes.

**Core Idea**: Use a pure Transformer to tokenized multi-view images and directly regress per-pixel 3D Gaussian primitives, achieving an extremely simple, scalable, and object/scene-agnostic feed-forward high-quality 3D reconstruction.

## Method

### Overall Architecture

The architecture of GS-LRM is extremely simple: input $N$ multi-view images and camera parameters $\to$ concatenate each image with Plücker ray coordinates (9 channels) $\to$ patchify into non-overlapping patch tokens $\to$ concatenate multi-view tokens and feed them into an $L$-layer Transformer block (self-attention + MLP) $\to$ decode each output token via a linear layer into $p^2$ 3D Gaussian parameters $\to$ unpatchify to obtain per-pixel Gaussians $\to$ merge Gaussians from all views as the final output.

### Key Designs

1. **Per-pixel Gaussian Prediction**:
    - **Function**: Map 2D image pixels to 3D Gaussians one-to-one, simplifying the network and preserving high-frequency details.
    - **Mechanism**: Each 2D pixel corresponds to one 3D Gaussian, with parameters including 3-channel RGB, 3-channel scale, 4-channel rotation quaternion, 1-channel opacity, and 1-channel ray distance (12 dimensions in total). The Gaussian center is computed from the ray distance and camera parameters via $xyz = ray_o + t \cdot ray_d$. The total output is $N \times H \times W$ Gaussians.
    - **Design Motivation**: Pixel alignment creates a shortcut from input RGB to output color, making it easier for the network to learn accurate per-Gaussian colors; it automatically scales the number of Gaussians with the input resolution to adapt to high resolution and large scenes.

2. **Plücker Ray Positional Encoding**:
    - **Function**: Provide unique spatial and viewpoint information for each patch, replacing traditional positional encodings.
    - **Mechanism**: Concatenate the Plücker ray coordinates (6 dimensions) of each pixel with RGB (3 dimensions) into a 9-channel input, and map it into $d$-dimensional tokens via a linear layer after patchifying. Since Plücker coordinates naturally contain pixel position and viewpoint information, no extra positional encoding or viewpoint encoding is required.
    - **Design Motivation**: Eliminate the complexity of designing positional encodings and viewpoint encodings, while enabling self-attention to perform multi-view correspondence reasoning based on geometric ray information.

3. **Pure Self-Attention Multi-View Fusion**:
    - **Function**: Establish correspondences across all patches of all views via global self-attention.
    - **Mechanism**: Concatenate all patch tokens from all views into a single long sequence and feed it into a standard Transformer block. Each self-attention layer globally computes across and within all viewpoints, naturally learning multi-view correspondences and reconstruction priors.
    - **Design Motivation**: Compared to methods like pixelSplat that require specifically designed epipolar feature aggregation, global self-attention is simpler and utilizes information from all pixels (rather than just a subset on the epipolar lines), yielding better performance when trained on large-scale data.

### Loss & Training

The loss is a weighted sum of MSE and perceptual loss: $\mathcal{L} = \frac{1}{M}\sum_{i'} (MSE(\hat{I}_{i'}^*, I_{i'}^*) + \lambda \cdot Perceptual(\hat{I}_{i'}^*, I_{i'}^*))$ with $\lambda=0.5$. The perceptual loss is based on VGG-19 (which is more stable to train than LPIPS). The model consists of 24 Transformer layers with a hidden dimension of 1024, 16 attention heads, an MLP dimension of 4096, and a patch size of 8×8, totaling about 300M parameters. It is pre-trained at 256 resolution for 2 days and fine-tuned at 512 resolution for 1 day, using 64× A100 (40G) GPUs.

## Key Experimental Results

### Main Results

**Object-level Reconstruction (GSO Dataset):**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Input Resolution |
|------|-------|-------|--------|-----------|
| Triplane-LRM | 26.54 | 0.893 | 0.064 | 512 |
| GS-LRM (Res-512) | **30.52** | **0.952** | **0.050** | 512 |
| LGM | 21.44 | 0.832 | 0.122 | 256 |
| GS-LRM (Res-256) | **29.59** | **0.944** | **0.051** | 256 |

**Scene-level Reconstruction (RealEstate10K):**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| pixelNeRF | 20.43 | 0.589 | 0.550 |
| GPNR | 24.11 | 0.793 | 0.255 |
| pixelSplat | 25.89 | 0.858 | 0.142 |
| GS-LRM | **28.10** | **0.892** | **0.114** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| vs Triplane-LRM (GSO) | +3.98dB PSNR | Pixel-aligned Gaussians are far superior to triplane |
| vs LGM (GSO, 256) | +8.15dB PSNR | Pure Transformer is far superior to U-Net |
| vs pixelSplat (RE10K) | +2.21dB PSNR | Global self-attention is superior to epipolar sampling |
| 4-view input object | 30.52 PSNR | Standard configuration |
| 2-view input scene | 28.10 PSNR | Equally outstanding at scene level |

### Key Findings

- Per-pixel Gaussian prediction is the core of the quality improvement—establishing a direct mapping from input RGB to output color.
- While Triplane-LRM struggles to reconstruct high-frequency details and thin structures like text, GS-LRM can faithfully reproduce them.
- The U-Net architecture of LGM is significantly inferior to the Transformer under the same computational budget (an 8dB difference).
- Pure self-attention does not require 3D inductive biases such as epipolar geometry, as it can automatically learn multi-view correspondences when trained on large-scale data.
- The inference speed is approximately 0.23 seconds (on a single A100), supporting interactive applications.

## Highlights & Insights

- **Elegant Architectural Simplicity**: Only three steps: patchify $\to$ Transformer $\to$ unpatchify, without triplane tokens, extra decoders, or 3D inductive biases.
- **Pixel Alignment is the Key Insight**: Aligning the input and output in the same pixel space creates a shortcut for color learning.
- **Object-Scene Agnostic**: The same architecture can switch tasks simply by changing the training data, demonstrating extreme versatility.
- **Strong Scalability**: Automatically scales the number of Gaussians and token lengths with resolution, supporting 512+ resolution.

## Limitations & Future Work

- The current maximum resolution is about 512×904; scaling up to 1K-2K resolution can be explored.
- Camera parameters are required, which might not be available in practical applications (e.g., taking 4 photos with a smartphone).
- The pixel-aligned representation can only model surfaces within the frustum; invisible regions cannot be reconstructed.
- Higher-order spherical harmonics are not used, limiting view-dependent quality.
- Integration with DUSt3R can be explored to achieve pose-free input.

## Related Work & Insights

- **LRM/Instant3D**: The first Transformer-based large reconstruction model, utilizing Triplane NeRF.
- **pixelSplat**: Concurrent work that also predicts pixel-aligned Gaussians but uses epipolar feature aggregation instead of global self-attention.
- **LGM**: Concurrent work that uses a U-Net architecture to predict multi-view Gaussians, focusing on object generation.
- **Insights**: The simple route of pure Transformer + big data is also effective in the 3D domain, echoing the scaling laws of NLP and 2D vision.

## Rating

- Novelty: ⭐⭐⭐⭐ The architecture is extremely simple yet highly effective; combining LRM with pixel-aligned Gaussians is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual evaluation on both objects and scenes + extensive baselines + downstream generation applications + high-resolution demonstrations.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear and concise with precise method descriptions.
- Value: ⭐⭐⭐⭐⭐ Establishes a baseline for GS-based large-scale feed-forward reconstruction, with broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation](grm_large_gaussian_reconstruction_model_for_efficient_3d_reconstruction_and_gene.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [\[CVPR 2026\] iLRM: An Iterative Large 3D Reconstruction Model](../../CVPR2026/3d_vision/ilrm_an_iterative_large_3d_reconstruction_model.md)
- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
