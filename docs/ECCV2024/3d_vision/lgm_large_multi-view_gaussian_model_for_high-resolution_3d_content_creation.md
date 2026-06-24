---
title: >-
  [Paper Note] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation
description: >-
  [ECCV 2024][3D Vision][3D Generation] This paper proposes LGM, a multi-view 3D Gaussian reconstruction model based on an asymmetric U-Net architecture. It predicts 65,536 3D Gaussian primitives from 4 orthogonal view images, achieving text/image-to-high-resolution 3D model generation within 5 seconds at a 512 resolution. The model bridges the training-inference domain gap through data augmentation strategies.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Generation"
  - "Gaussian Splatting"
  - "Multi-view Reconstruction"
  - "High-resolution"
  - "U-Net"
date: 2026-05-08
content_hash: 54ff8341dc7994bf
---

# LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation

**Conference**: ECCV 2024  
**arXiv**: [2402.05054](https://arxiv.org/abs/2402.05054)  
**Code**: [https://github.com/3DTopia/LGM](https://github.com/3DTopia/LGM)  
**Area**: 3D Vision  
**Keywords**: 3D Generation, Gaussian Splatting, Multi-view Reconstruction, High-resolution, U-Net

## TL;DR

This paper proposes LGM, a multi-view 3D Gaussian reconstruction model based on an asymmetric U-Net architecture. It predicts 65,536 3D Gaussian primitives from 4 orthogonal view images, achieving text/image-to-high-resolution 3D model generation within 5 seconds at a 512 resolution. The model bridges the training-inference domain gap through data augmentation strategies.

## Background & Motivation

**Background**: 3D content creation is in high demand in gaming, VR, and film production. Existing methods are divided into two categories: (1) SDS optimization methods (such as DreamFusion, Magic3D) lift 2D diffusion priors to 3D via score distillation; they yield high-quality results but take minutes to hours. (2) Feed-forward methods (such as LRM) achieve second-level inference via large-scale training, but are limited by the low resolution of Triplane NeRF and the high computational cost of volume rendering.

**Limitations of Prior Work**: (1) Triplane-based methods like LRM limit the Triplane resolution to 32 and rendering resolution to 128, leading to a severe lack of details. (2) Transformer backbones have a large number of parameters, which restricts the training resolution. (3) Although SDS methods preserve good details, their execution speed is too slow (on the order of minutes), and they suffer from the multi-face Janus problem and a lack of diversity.

**Key Challenge**: Achieving high-resolution 3D generation requires both an expressive, rendering-efficient 3D representation and a backbone network capable of efficient training at high resolutions. The combination of Triplane NeRF and Transformer poses bottlenecks in both dimensions.

**Goal**: (1) How to design an efficient feed-forward model for high-resolution 3D generation? (2) How to bridge the domain gap between the 3D-rendered images used during training and the diffusion-generated images used during inference?

**Key Insight**: Choose 3D Gaussian Splatting as the representation (rendering-efficient and highly expressive) and U-Net as the backbone (lighter than Transformers and supporting higher-resolution training). Each output pixel is interpreted as a 3D Gaussian, allowing the fusion and generation of a sufficient number of Gaussians (65,536) from 4 multi-view images.

**Core Idea**: Asymmetric U-Net paired with multi-view pixel-level 3D Gaussian prediction, enabling high-resolution 3D content generation within 5 seconds under a 512-resolution training setup.

## Method

### Overall Architecture

A two-step generation pipeline: (1) Utilize off-the-shelf multi-view diffusion models (such as MVDream/ImageDream) to generate 4 orthogonal multi-view images from text or a single image. (2) Feed the 4 images into an asymmetric U-Net to output 4 feature maps, where each pixel is decoded into 3D Gaussian parameters. These are fused into the final 3D Gaussian set. Optional step: Convert the Gaussians into a smooth textured mesh via a NeRF-based intermediate proxy.

### Key Designs

1. **Asymmetric U-Net Architecture**:
    - **Function**: Efficiently predict a sufficient number of 3D Gaussians from multi-view images.
    - **Mechanism**: The U-Net takes an input resolution of 256×256 and produces an output resolution of 128×128 (asymmetric design). It consists of 6 downsampling blocks, 1 middle block, and 5 upsampling blocks, with channel configurations of [64,128,256,512,1024,1024] $\rightarrow$ [1024] $\rightarrow$ [1024,1024,512,256,128]. Cross-view self-attention is inserted in the deep blocks (the last 3 downsampling blocks, the middle block, and the first 3 upsampling blocks), where features of the 4 images are flattened and concatenated to perform self-attention for multi-view information exchange. Finally, a 1×1 convolution outputs 14-channel pixel-by-pixel Gaussian features.
    - **Design Motivation**: Compared with the large Transformer backbone in LRM, U-Net significantly reduces the parameter size and computational cost while preserving high-resolution capability. The asymmetric design allows high-resolution input while keeping the number of output Gaussians within a reasonable range (65,536).

2. **Data Augmentation - Grid Distortion and Camera Jitter**:
    - **Function**: Bridge the domain gap between training (realistic rendered 3D images) and inference (synthetic images generated by diffusion models).
    - **Mechanism**: Grid Distortion — Except for the first reference view, the other 3 input images are randomly subjected to grid distortion during training to simulate the subtle inconsistencies among multi-view images generated by diffusion models. Orbital Camera Jitter — The camera poses of the last 3 input views are randomly rotated to achieve tolerance against inaccurate camera poses output by diffusion models. Both augmentations are applied with a probability of 50%.
    - **Design Motivation**: Multi-view images generated by diffusion models lack an underlying 3D representation, resulting in cross-view inconsistencies and camera pose shifts. Models trained without augmentation achieve lower training loss but produce more floating artifacts and poorer geometry during inference.

3. **Gaussian-to-Mesh Conversion Pipeline**:
    - **Function**: Convert the generated 3D Gaussians into polygonal meshes commonly used in downstream applications.
    - **Mechanism**: Instead of directly extracting occupancy fields from Gaussian opacity (as in DreamGaussian), which is unsuitable due to the sparseness of feed-forward generated Gaussians, an alternative is adopted. An efficient NeRF (using a hash grid) is trained on images rendered from the Gaussians, and a coarse mesh is extracted via Marching Cubes, which is then iteratively refined and baked with textures. The whole pipeline takes about 1 minute.
    - **Design Motivation**: Feed-forward generated Gaussians are sparsely distributed and do not satisfy the densification assumptions of DreamGaussian. Utilizing NeRF as an intermediate representation yields smoother surfaces.

### Loss & Training

RGB loss: $\mathcal{L}_{rgb} = \mathcal{L}_{MSE}(I_{rgb}, I_{rgb}^{GT}) + \lambda \mathcal{L}_{LPIPS}(I_{rgb}, I_{rgb}^{GT})$. Alpha loss: $\mathcal{L}_\alpha = \mathcal{L}_{MSE}(I_\alpha, I_\alpha^{GT})$. Rendering 8 views per step (4 input + 4 novel views), utilizing 512×512 resolution MSE + 256×256 resolution LPIPS. Trained for 4 days on 32×A100 (80G) with a batch size of 256 (bf16), using AdamW ($lr=4\times 10^{-4}$, weight decay 0.05). Positional initialization is clamped to $[-1,1]^3$.

## Key Experimental Results

### Main Results

**User study (1-5 scale, higher is better):**

| Method | Image Consistency | Overall Quality |
|------|-----------|---------|
| DreamGaussian | 2.30 | 1.98 |
| TriplaneGaussian | 3.02 | 2.67 |
| LGM (Ours) | **4.18** | **3.95** |

**Qualitative comparison with LRM:**
- LRM single-view input $\rightarrow$ blurry back side, flat geometry
- LGM multi-view input $\rightarrow$ sharp back side, accurate geometry

**Generation speed comparison:**

| Method | Generation Time | Resolution |
|------|---------|--------|
| DreamGaussian (SDS) | Minutes | Low |
| LRM | ~5 seconds | 128 |
| LGM | ~5 seconds | 512 |

### Ablation Study

| Configuration | Result/Metric | Description |
|------|---------|------|
| Single-view input | Good front view, blurry back side | U-Net regression model struggles with heavy occlusions |
| Without data augmentation | More floating artifacts, poor geometry | Domain gap causes degradation during inference |
| With data augmentation | Better 3D consistency correction | Augmentation strategies are effective |
| Output 64×64 (16K Gaussians) | Poorer details | Insufficient number of Gaussians |
| Output 128×128 (65K Gaussians) | Rich details | Standard configuration |
| Training resolution 256 | Weaker details than 512 | Resolution scaling is effective |
| Training resolution 512 | Best details | Default configuration |

### Key Findings

- 4-view input significantly improves back-side quality compared to single-view input, highlighting that the additional information provided by multi-view diffusion models is crucial for reconstruction.
- Data augmentation is the key to bridging the training-inference domain gap. Although it increases training loss, it substantially improves generalization during inference.
- 65,536 Gaussians are sufficient to represent most single objects, and training at 512 resolution effectively captures fine details.
- The entire pipeline (diffusion + reconstruction) requires only about 10GB of VRAM, making it deployment-friendly.
- The quality of the multi-view diffusion model is the bottleneck for LGM — 3D inconsistencies lead to floating artifacts, and low resolution limits the upper bound of detail.

## Highlights & Insights

- **Pragmatic Choice of U-Net vs. Transformer**: In 3D generation scenarios, the high-resolution training capability of U-Net is more critical than the expressive power of a Transformer.
- **Elegant Data Augmentation Design**: Grid distortion simulates geometric inconsistency and camera jitter simulates pose drift, directly addressing the two core issues of diffusion-generated outputs.
- **Complete Ecosystem**: Text $\rightarrow$ multi-view images $\rightarrow$ 3D Gaussians $\rightarrow$ meshes, enabling end-to-end deployment.
- **Unprecedented Efficiency**: 5 seconds + 10GB VRAM = democratization of high-resolution 3D generation.

## Limitations & Future Work

- Heavily reliant on the quality of multi-view diffusion models — 3D inconsistencies in the diffusion models are the primary source of failure.
- The resolution of multi-view diffusion models is limited to 256×256, which constrains the upper bound of LGM's detail quality.
- ImageDream cannot handle input images with high elevation angles.
- The model does not employ high-order spherical harmonics, resulting in limited viewpoint-dependent effects.
- Better multi-view generation models (e.g., the 6-view version of Zero123++) can be explored to further improve performance.

## Related Work & Insights

- **Splatter Image**: A pioneer in using single-view U-Net to predict pixel-level Gaussians, which inspired LGM's pixel-by-pixel Gaussian design.
- **LRM/Instant3D**: Large-scale reconstruction models following the Triplane NeRF + Transformer paradigm.
- **MVDream/ImageDream**: Multi-view diffusion models serving as the upstream generation models for LGM.
- **GS-LRM**: A concurrent work following a pure Transformer route, delivering higher quality but at a much higher computational cost.
- Insight: In feed-forward 3D generation, a combination of a lightweight backbone and an efficient representation might be more suitable for practical deployment than large models paired with complex representations.

## Rating

- Novelty: ⭐⭐⭐ The combination of U-Net and multi-view Gaussians is of high practical value but limited novelty.
- Experimental Thoroughness: ⭐⭐⭐ The paper employs user studies instead of quantitative metrics, and ablation studies cover key designs.
- Writing Quality: ⭐⭐⭐⭐ Clear and elegant, with a complete pipeline description.
- Value: ⭐⭐⭐⭐ A practical route featuring high resolution and fast generation, contributing to the democratization of 3D generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)
- [\[ECCV 2024\] GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation](grm_large_gaussian_reconstruction_model_for_efficient_3d_reconstruction_and_gene.md)
- [\[ECCV 2024\] SuperGaussian: Repurposing Video Models for 3D Super Resolution](supergaussian_repurposing_video_models_for_3d_super_resolution.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)

</div>

<!-- RELATED:END -->
