---
title: >-
  [Paper Note] Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model
description: >-
  [ECCV 2024][3D Vision][NeRF copyright protection] NeRFProtector is proposed, which utilizes a pre-trained watermarking base model (message extractor) to embed binary watermarks in a plug-and-play manner during the NeRF creation process. By employing Progressive Global Rendering (PGR), watermarking knowledge is distilled into the NeRF representation, achieving high bit-accuracy copyright protection without modifying the NeRF architecture.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "NeRF copyright protection"
  - "digital watermarking"
  - "plug-and-play"
  - "progressive global rendering"
  - "knowledge distillation"
date: 2026-05-08
content_hash: 939001ef06226206
---

# Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model

**Conference**: ECCV 2024  
**arXiv**: [2407.07735](https://arxiv.org/abs/2407.07735)  
**Code**: [https://qsong2001.github.io/NeRFProtector](https://qsong2001.github.io/NeRFProtector)  
**Area**: 3D Vision  
**Keywords**: NeRF copyright protection, digital watermarking, plug-and-play, progressive global rendering, knowledge distillation

## TL;DR

NeRFProtector is proposed, which utilizes a pre-trained watermarking base model (message extractor) to embed binary watermarks in a plug-and-play manner during the NeRF creation process. By employing Progressive Global Rendering (PGR), watermarking knowledge is distilled into the NeRF representation, achieving high bit-accuracy copyright protection without modifying the NeRF architecture.

## Background & Motivation

**Background**: NeRF has become a key technology for 3D scene representation. As its influence expands, protecting the intellectual property of NeRF models is increasingly important. Existing methods, such as CopyRNeRF, protect copyright by embedding binary watermarks in NeRF models.

**Limitations of Prior Work**: CopyRNeRF has two significant drawbacks: first, watermark embedding occurs during the model fine-tuning stage after NeRF creation is complete, creating a time window during which malicious users can obtain the unprotected model; second, NeRF creators need to jointly train a message extractor when embedding the watermark, making the entire process extremely time-consuming and complex (about 30 hours), which may deter creators from using watermark protection.

**Key Challenge**: The contradiction between the practicality and ease of use of copyright protection—existing methods are either not timely (embedding only after creation) or have a high barrier to use (requiring modifications to the NeRF architecture or joint training of extra modules), leading to low adoption by creators.

**Goal**: (1) How to synchronize watermark embedding with the NeRF creation process to eliminate the protection time window; (2) how to make the watermarking scheme compatible with various NeRF variants without architectural modifications; (3) how to achieve high-accuracy watermark extraction while maintaining rendering quality.

**Key Insight**: The authors observe that pre-trained message extractors already exist in traditional 2D image watermarking frameworks (such as HiDDeN), and these extractors have already learned the knowledge of watermark patterns. If this knowledge can be "distilled" into NeRF, there is no need to modify the architecture of NeRF itself.

**Core Idea**: Utilizing a pre-trained 2D watermark extractor as a plug-and-play base model, the watermark knowledge is distilled into the NeRF representation via progressive global rendering, enabling simultaneous creation and protection.

## Method

### Overall Architecture

NeRFProtector consists of three stages: (1) constructing the watermarking base model—obtaining the pre-trained message extractor $\mathcal{F}$ from the HiDDeN framework; (2) during NeRF creation, freezing the base model weights and distilling the watermarking knowledge into the NeRF representation via progressive global rendering (PGR); (3) after creation, extracting the binary watermark from the rendered images using the same base model for copyright declaration. The inputs are multi-view images of the 3D scene and the binary message to be embedded, and the output is the watermarked NeRF model.

### Key Designs

1. **Watermarking Base Model**:

    - **Function**: Provides plug-and-play capabilities for watermark embedding and extraction.
    - **Mechanism**: Employs the HiDDeN framework to jointly train an encoder $\mathcal{E}$ and an extractor $\mathcal{F}$. The encoder embeds a 48-bit binary message into a cover image to generate a watermarked image, while the extractor recovers the message from the (potentially degraded) watermarked image. Once training is complete, the encoder is discarded, and only the extractor is retained as the base model. Random transformation layers $T$ are introduced during training to enhance robustness against common image distortions.
    - **Design Motivation**: Leverages existing mature 2D watermarking frameworks to avoid redesigning, and the extractor has already learned the knowledge of message patterns, facilitating subsequent distillation into NeRF.

2. **Progressive Global Rendering (PGR)**:

    - **Function**: Replaces the random local rendering of NeRF to achieve global watermark embedding.
    - **Mechanism**: Standard NeRF training renders only a small random subset of pixels at a time (local rendering), causing watermark patterns to be embedded only in random positions and preventing the formation of an effective global pattern. PGR renders all pixels across multiple resolution scales, generating a cascade of $N_k=3$ views $\hat{I}_{set}$, where each layer has a resolution of $\frac{W}{2^n} \times \frac{H}{2^n}$. Computational cost is kept manageable due to the reduced resolution of global rendering.
    - **Design Motivation**: Global rendering ensures that message patterns are deeply integrated into the scene representation. Multi-scale rendering exploits the different characteristics of 3D information under various 2D projection resolutions, which assists in message distillation.

3. **Message Distillation**:

    - **Function**: Transfers watermarking knowledge from the base model to the NeRF representation.
    - **Mechanism**: For the multi-scale rendered images generated by PGR, the base model is used to extract messages $\hat{m}_{set} = \mathcal{F}(\hat{I}_{set})$. Distillation is performed by minimizing the BCE loss between the extracted messages and the target message: $\mathcal{L}_{dis} = \sum_{i=1}^{N_k} \alpha_i \cdot BCE(m, \hat{m}_i)$. Concurrently, an invisibility loss $\mathcal{L}_{inv}$ is applied to constrain the rendering quality.
    - **Design Motivation**: Does not modify NeRF's underlying representation structure; knowledge transfer is achieved solely through changes to the rendering scheme, maintaining the plug-and-play characteristic.

### Loss & Training

The total loss is a weighted sum of three components: $\mathcal{L} = \lambda_1 \mathcal{L}_{local} + \lambda_2 \mathcal{L}_{inv} + \lambda_3 \mathcal{L}_{dis}$, where $\lambda_1=0.01$ and $\lambda_3=0.001$. $\mathcal{L}_{local}$ is the standard NeRF reconstruction loss, $\mathcal{L}_{inv}$ is the MSE loss between the highest-resolution rendering and the ground truth, and $\mathcal{L}_{dis}$ is the multi-scale distillation loss. The weights of the base model are frozen and not updated.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Bit Acc.(None) | Bit Acc.(Crop) | Bit Acc.(Resize) | Bit Acc.(JPEG) |
|--------|------|-------|-------|--------|----------------|----------------|------------------|----------------|
| Blender | NeRF w/o wm | 30.62 | 0.9579 | 0.0343 | N/A | N/A | N/A | N/A |
| Blender | CopyRNeRF | 25.50 | 0.9073 | 0.0885 | 62.15% | 56.63% | 57.32% | 58.41% |
| Blender | **NeRFProtector** | **29.26** | **0.9393** | **0.0483** | **92.69%** | **92.95%** | **91.87%** | **78.62%** |
| LLFF | NeRF w/o wm | 26.37 | 0.8352 | 0.1013 | N/A | N/A | N/A | N/A |
| LLFF | CopyRNeRF | 25.80 | 0.8302 | 0.1035 | 63.72% | 60.45% | 55.34% | 54.11% |
| LLFF | **NeRFProtector** | **26.82** | **0.8569** | **0.0834** | **96.99%** | **93.57%** | **80.53%** | **76.26%** |

### Ablation Study

| Rendering Strategy | PSNR↑ | SSIM↑ | LPIPS↓ | Bit Accuracy |
|---------|-------|-------|--------|-------------|
| Local rendering only | 30.38 | 0.9521 | 0.0360 | 45.99% |
| Single-scale global | 29.57 | 0.9402 | 0.0449 | 87.27% |
| Progressive (Ours) | 29.26 | 0.9394 | 0.0483 | **92.69%** |

| NeRF Variant + Base Model | PSNR↑ | Bit Accuracy |
|-------------------|-------|-------------|
| Instant-NGP + HiDDeN | 32.92 | 91.96% |
| TensorRF + HiDDeN | 32.73 | 89.35% |
| Plenoxels + HiDDeN | 34.19 | 97.92% |
| Instant-NGP + MBRS | 31.71 | 89.13% |

### Key Findings

- PGR is the most critical design: bit accuracy jumps from 45.99% with local rendering to 92.69% with progressive global rendering.
- The method is compatible with various NeRF variants (Instant-NGP, TensorRF, Plenoxels) and multiple base models (HiDDeN, MBRS), validating the plug-and-play capability.
- The training time is only about 50 minutes, whereas CopyRNeRF requires around 30 hours, achieving a 36x speedup.
- Under common image distortions (cropping, scaling), bit accuracy remains above 80%+.

## Highlights & Insights

- **Plug-and-play design philosophy**: Encapsulates the watermarking capability into an independent base model and decouples it from the NeRF architecture. This modular concept can be transferred to the copyright protection of other 3D representations (such as 3D Gaussian Splatting).
- **Discovery of the link between rendering strategies and watermark embedding**: Reveals that NeRF's random local rendering cannot effectively embed global watermark patterns. This observation is inspiring—any task that depends on global patterns could potentially benefit from global rendering strategies.
- **Cross-dimensional transfer via knowledge distillation**: Transfers 2D watermark extraction knowledge into 3D scene representations without needing to design specialized 3D watermarking schemes, embodying the approach of solving problems by dimensionality reduction.

## Limitations & Future Work

- White-box attack threats: If an attacker obtains the base model, they can remove the watermark with minimal distortion via PGD attacks; the confidentiality of the base model is a prerequisite for security.
- If an attacker obtains the raw training images, the watermark can be removed via fine-tuning without watermarking loss.
- The embedding capability for longer messages remains unexplored, as it was only validated with a 48-bit message length.
- More recent 3D representation methods, such as 3D Gaussian Splatting, were not considered.
- Copyright protection requires a comprehensive strategy beyond technical solutions, including support from legal frameworks.

## Related Work & Insights

- **vs CopyRNeRF**: CopyRNeRF embeds watermarks via fine-tuning after NeRF creation and requires joint training of the extractor, taking about 30 hours and leaving a vulnerability window. NeRFProtector embeds watermarks simultaneously during creation in only 50 minutes, eliminating this security window.
- **vs StegaNeRF**: StegaNeRF hides data in NeRF but requires structural modifications; NeRFProtector keeps the NeRF architecture unchanged, offering better compatibility.
- **vs HiDDeN/MBRS**: These 2D watermarking methods directly process images and then train NeRF, but watermark information fails to remain consistent across 3D rendering (achieving only ~50% bit accuracy). NeRFProtector achieves cross-view consistency via distillation.

## Rating

- Novelty: ⭐⭐⭐⭐ The plug-and-play watermarking base model approach is novel, but core components (HiDDeN, distillation) are existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Excellent ablation studies, cross-variant validation, and comprehensive attack analyses, but only evaluated on two datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation, detailed methodology description, and well-designed figures/tables.
- Value: ⭐⭐⭐⭐ Solves the practical issues of NeRF copyright protection, but the application scenario is somewhat narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SCFlow2: Plug-and-Play Object Pose Refiner with Shape-Constraint Scene Flow](../../CVPR2025/3d_vision/scflow2_plug-and-play_object_pose_refiner_with_shape-constraint_scene_flow.md)
- [\[AAAI 2026\] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?](../../AAAI2026/3d_vision/can_protective_watermarking_safeguard_the_copyright_of_3d_gaussian_splatting.md)
- [\[ECCV 2024\] DATENeRF: Depth-Aware Text-based Editing of NeRFs](datenerf_depth-aware_text-based_editing_of_nerfs.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[CVPR 2026\] Plug-and-Play PDE Optimization for 3D Gaussian Splatting: Toward High-Quality Rendering and Reconstruction](../../CVPR2026/3d_vision/plug-and-play_pde_optimization_for_3d_gaussian_splatting_toward_high-quality_ren.md)

</div>

<!-- RELATED:END -->
