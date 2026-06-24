---
title: >-
  [Paper Note] FeatSharp: Your Vision Model Features, Sharper
description: >-
  [ICML 2025][Segmentation][Feature Upsampling] This paper proposes FeatSharp, which coherently upsamples feature maps of low-resolution vision encoders to high resolution at an extremely low cost by taking FeatUp's Joint Bilateral Upsampling (JBU) and attentively fusing it with image tiling features, while capturing fine-grained details lost at the original resolution.
tags:
  - "ICML 2025"
  - "Segmentation"
  - "Feature Upsampling"
  - "Vision Transformer"
  - "Multi-View Consistency"
  - "Joint Bilateral Upsampling"
  - "Tiling Fusion"
date: 2026-05-08
content_hash: 608504a363d48a76
---

# FeatSharp: Your Vision Model Features, Sharper

**Conference**: ICML 2025  
**arXiv**: [2502.16025](https://arxiv.org/abs/2502.16025)  
**Code**: [https://github.com/NVlabs/FeatSharp](https://github.com/NVlabs/FeatSharp)  
**Area**: Segmentation  
**Keywords**: Feature Upsampling, Vision Transformer, Multi-View Consistency, Joint Bilateral Upsampling, Tiling Fusion

## TL;DR

This paper proposes FeatSharp, which coherently upsamples feature maps of low-resolution vision encoders to high resolution at an extremely low cost by taking FeatUp's Joint Bilateral Upsampling (JBU) and attentively fusing it with image tiling features, while capturing fine-grained details lost at the original resolution.

## Background & Motivation

Currently, mainstream vision foundation models (VFMs) rely on Vision Transformer (ViT) backbones, typically trained via contrastive learning such as CLIP. These models suffer from a core limitation: **fixed and low resolution**. A typical CLIP model operates at a resolution of 224×224 or 336×336, resulting in a 14x downsampling rate for spatial features (224² → 16²). Due to the nature of learned positional encodings, ViTs are also inflexible to changes in input resolution.

Directly increasing the input resolution faces two challenges: (1) the computational cost of ViTs scales **quadratically** with resolution $O((w \cdot h)^2)$; (2) many models (e.g., CLIP, SigLIP) generalize poorly outside their training resolution.

Prior work FeatUp proposed learning an upsampler via multi-view consistency training. Although its JBU variant is fast, it has significant drawbacks: (1) it relies solely on RGB pixels as guidance, resulting in blurry features when semantic boundaries inside objects are lacking; (2) it cannot introduce new details finer than the original resolution; (3) while the implicit upsampler performs better, it is costly, taking 1-5 minutes per image.

The motivation of FeatSharp is to find an **optimal trade-off between computational efficiency and detail quality** between JBU (single-pass inference, fast) and implicit models (multi-pass inference, fine-grained).

## Method

### Overall Architecture

The workflow of FeatSharp is as follows:

1. **Global Low-Resolution Inference**: Feed the input image into the frozen vision encoder to obtain the low-resolution feature map $f(x)$.
2. **JBU Upsampling**: Use FeatUp's Joint Bilateral Upsampling to upsample the low-resolution feature map to the target resolution, using the RGB image as guidance.
3. **Tiling Inference**: Divide the input image into $n \times n$ tiles. Each tile is resized to the encoder's original input resolution, passed **independently** through the encoder, and then stitched back into a high-resolution feature map.
4. **FeatSharp Fusion Module**: Concatenate the JBU-upsampled features and tile features along the channel dimension, then fuse them using a Transformer block with sliding window attention.
5. **Output Slicing**: Slice the first $C$ channels of the output (corresponding to the residual path of JBU upsampling) as the final high-resolution features.

### Key Designs

#### 1. JBU Improvement: Prime Factorization Upsampling

The original FeatUp only supports stacked $2\times$ JBU. FeatSharp proposes performing **prime factorization** on any arbitrary integer upsampling factor $z$ and applying the corresponding JBU layer for each prime factor. For example, a 14x upsampling (corresponding to a backbone with patch size 14) is factorized into $\text{JBU}_{7\times} \circ \text{JBU}_{2\times}$.

#### 2. Tile-Guided Attention Fusion (FeatSharp Module)

This is the core innovation of this paper. The fundamental limitations of JBU are:
- It relies on RGB pixel guidance, causing semantic boundaries that are indistinct in color space to be blurred.
- It fails to capture details of small objects that are invisible at low resolution.

Tile features can provide: high-resolution semantic information and details of small regions invisible at original resolution. However, stitched tiles suffer from **severe boundary discontinuity** (inconsistent representations across different tiles).

Design of the FeatSharp module:
- Concatenate the JBU features $(H, W, C)$ and tile features $(H, W, C)$ along the channel dimension into $(H, W, 2C)$.
- Process through an **Attention + SwiGLU** Transformer block, employing **2D local sliding window attention** to avoid the quadratic overhead of global attention.
- Finally, slice the first $C$ channels as the output.

A key design insight is that the Transformer block utilizes residual connections; thus, a no-op is equivalent to directly returning JBU features, making the learning of useful information extraction from tiles progressive.

#### 3. Feature Denoising (Learnable Bias Buffer)

Inspired by ViT-Denoiser, ViT features contain fixed, position-dependent noise $g(E_{pos})$. FeatSharp introduces a **learnable bias buffer** $g$:

$$\hat{f}(x) = f(x) + g$$

This buffer is automatically learned via multi-view consistency training to **counteract position-fixed artifacts**. Since fixed patterns reduce multi-view consistency (as patterns are always local and lack global consistency), the training process naturally drives $g$ to eliminate these artifacts.

#### 4. PHI-S Feature Normalization

To avoid directly using raw features (which exhibit large distribution differences and cause unstable training) while avoiding LayerNorm (which disrupts the original feature space), PHI-S normalization is adopted: normalization is performed based on distribution statistics calculated over 100k samples from the training set, maintaining feature space compatibility.

### Loss & Training

- **Training Objective**: Pure **multi-view consistency** — the upsampled features, when subjected to an arbitrary affine transformation and then downsampled, should match the model's original low-resolution prediction on the transformed image.
- **Loss Function**: Uses only **MSE loss**, omitting the Total Variation (TV) and Conditional Random Field (CRF) losses used in FeatUp.
- **Frozen/Learnable**: The vision encoder is completely frozen; only the JBU parameters, the FeatSharp fusion module, and the denoising bias buffer are trained.

**Computational Complexity Advantage**: When using $x$-fold tiles, the inference cost of FeatSharp is $f(x) = c(1 + x^2)$, whereas the cost of directly running the model at high resolution is $g(x) = cx^4$. For any $x > 1$, FeatSharp is more efficient.

## Key Experimental Results

### Main Results

**Semantic Segmentation (ADE20K mIoU, Linear Probe)**

| Model | Method | Upsampling | Input Size | mIoU | Note |
|------|------|--------|----------|------|------|
| RADIOv2.5-L | Baseline | 1× | 1× | 51.47 | Published SOTA |
| RADIOv2.5-L | FeatSharp | 2× | 2× | **53.13** | **+1.66 mIoU** |
| RADIOv2.5-L | FeatUp | 2× | 2× | < Baseline 2× | Inferior to baseline |
| All models | FeatSharp | - | - | Best | Best across all models |

**Object Detection (COCO 2017)**

| Upsampling Method | Upsampling Factor | AP* | AP_Sm | AP_Md | AP_Lg |
|-----------|-----------|-----|-------|-------|-------|
| Baseline (RADIO) | 1× | 51.38 | 28.73 | 56.56 | 73.72 |
| Bilinear (RADIO) | 2× | 51.61 | 28.43 | 56.98 | 74.14 |
| FeatUp (RADIO) | 2× | 46.71 | 21.77 | 52.01 | 72.25 |
| **FeatSharp (RADIO)** | **2×** | **54.83** | **34.72** | **59.40** | **74.40** |
| Baseline (SigLIP2) | 1× | 52.66 | 30.31 | 57.94 | 74.31 |
| **FeatSharp (SigLIP2)** | **2×** | **55.93** | **36.85** | **61.00** | **74.62** |

The improvement of FeatSharp on small object detection is particularly significant: +6.0 pts on AP_Sm for RADIO, and +6.5 pts for SigLIP2.

### Ablation Study

**RADIO Aggregation Model Training (MTL Gain $\Delta_m\%$)**

| Teacher Upsampling Method | Classification | Dense | 3D Probe | Retrieval | Pascal | NYUDv2 | VILA | $\Delta_m\%$ |
|--------------|------|------|--------|------|--------|--------|------|-------------|
| RADIOv2.5-L | -0.47 | -0.09 | -1.05 | -0.45 | 0.62 | -2.26 | 2.24 | -0.21 |
| Baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Tile | -0.03 | 0.30 | -0.08 | -0.23 | -0.02 | 1.33 | -3.17 | -0.27 |
| S2 | -0.05 | 0.15 | -0.03 | -0.44 | 0.13 | 1.33 | -0.89 | 0.03 |
| FeatUp | -0.07 | 0.14 | 0.23 | -0.07 | 0.14 | 0.32 | -1.58 | -0.13 |
| **FeatSharp** | **0.06** | **0.16** | **0.83** | **0.13** | **0.17** | **0.93** | **0.43** | **+0.39** |

### Key Findings

1. **FeatSharp is the only method that achieves comprehensive gains across all tasks**: $\Delta_m = +0.39\%$, whereas Tile and FeatUp exhibit performance drops on certain tasks.
2. **CLIP-family models do not benefit from high-resolution inputs**: DFN CLIP, SigLIP, etc., show unchanged or even degraded segmentation performance when resolution is increased, whereas DINOv2 and RADIO natively benefit.
3. **Multi-View Consistency (Fidelity)**: FeatSharp consistently achieves the highest fidelity across all evaluated models, with particularly pronounced advantages on "clean" models (DINOv2-L, RADIOv2.5-L, SAM-H).
4. **FeatSharp is 57% faster than direct execution at high resolution**: In the RADIO + ADE20K experiments, 2× upsampling + 2× input is 57% faster than direct 2× input execution.

## Highlights & Insights

1. **Minimalist Design Philosophy**: The entire FeatSharp module consists of only a single Attention+SwiGLU block and a learnable bias buffer, completely avoiding complex multi-stage architectures, yet significantly outperforming FeatUp.
2. **Prime Factorization JBU**: Supports arbitrary integer upsampling factors (e.g., 14×) instead of being restricted to powers of 2, greatly improving practical usability.
3. **Unified Perspective of Denoising and Upsampling**: Reveals that FeatUp and ViT-Denoiser fundamentally leverage multi-view consistency to eliminate position-fixed noise, solving this elegantly with a simple learnable bias buffer.
4. **Tiling Fusion Solves Known VLM Issues**: Directly stitching tile features in VLMs causes boundary discontinuities and representation inconsistencies; FeatSharp's attention fusion provides a systematic solution.
5. **In-Depth Analysis of Model Resolution Robustness**: Uncovers the root cause of why the CLIP family does not benefit from high resolution, offering valuable references for selecting VFMs.

## Limitations & Future Work

1. **Requires Additional Tile Inference**: Although cheaper than direct high-resolution execution, $n \times n$ tiles still require $n^2$ additional encoder inference passes, which remains a bottleneck for real-time applications.
2. **Only Supports Integer Upsampling Factors**: The core training algorithm is restricted to integer upsampling factors, though RADIO's emulation can indirectly support arbitrary scaling.
3. **Unexplored Performance Drop at 3× Upsampling**: The paper acknowledges that 3× upsampling is typically slightly worse than 2× and 4×, but does not provide an explanation.
4. **Optimal Selection of Tile Counts**: Experiments only test final-layer tiles ($1 + x^2$ inference passes); the effect of progressive upsampling is left for future work.
5. **Limited Interpretability of the Learnable Bias**: Despite visualization efforts, an in-depth analysis of what the bias buffer actually learns is lacking.

## Related Work & Insights

- **FeatUp (Fu et al., 2024)**: The direct foundation of this work, which proposed the JBU multi-view consistency training framework. FeatSharp builds upon it by adding tile fusion and denoising.
- **ViT-Denoiser (Yang et al., 2024)**: Uncovers the source of ViT feature noise, inspiring the design of FeatSharp's learnable bias.
- **AM-RADIO / RADIOv2.5 (Ranzinger et al., 2024)**: An aggregation model framework, where FeatSharp is utilized during training to improve the quality of teacher features.
- **LLaVA 1.6 / InternVL (Liu et al., 2024; Chen et al., 2024)**: Tile-based strategies in VLMs inspired FeatSharp to introduce tiling to feature upsampling.
- **CARAFE / SAPA / LiFT**: The evolutionary path of pixel-adaptive upsampling methods.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4 | The JBU+Tiling fusion approach is clear and effective, and the unified perspective on denoising is insightful. |
| Experimental Thoroughness | 5 | Comprehensive coverage with 7 foundation models × multi-task × multiple upsampling factors. |
| Practicality | 4 | Minimalist module design allows plug-and-play, but still incurs additional inference overhead. |
| Writing Quality | 4 | Well-structured, rich with figures/tables, and motivation is well-articulated. |
| Overall Rating | **4.3** | Solid engineering and methodological contributions, providing valuable references for utilizing high-resolution inputs in vision foundation models. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Your ViT is Secretly an Image Segmentation Model](../../CVPR2025/segmentation/your_vit_is_secretly_an_image_segmentation_model.md)
- [\[ICLR 2026\] TRACE: Your Diffusion Model is Secretly an Instance Edge Detector](../../ICLR2026/segmentation/trace_your_diffusion_model_is_secretly_an_instance_edge_detector.md)
- [\[CVPR 2026\] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](../../CVPR2026/segmentation/videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)
- [\[ICCV 2025\] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](../../ICCV2025/segmentation/know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [\[CVPR 2025\] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)

</div>

<!-- RELATED:END -->
