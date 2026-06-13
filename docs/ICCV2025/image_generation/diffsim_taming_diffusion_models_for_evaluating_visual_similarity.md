---
title: >-
  [Paper Note] DiffSim: Taming Diffusion Models for Evaluating Visual Similarity
description: >-
  [ICCV 2025][Image Generation][Diffusion models] DiffSim is the first work to discover that attention-layer features of pretrained diffusion models (Stable Diffusion) can be used to measure visual similarity. It proposes…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Diffusion models"
  - "visual similarity"
  - "attention alignment"
  - "perceptual metrics"
  - "style similarity"
date: 2026-05-08
content_hash: 7b083fde970ae1a1
---

# DiffSim: Taming Diffusion Models for Evaluating Visual Similarity

**Conference**: ICCV 2025
**arXiv**: [2412.14580](https://arxiv.org/abs/2412.14580)  
**Code**: [https://github.com/showlab/DiffSim](https://github.com/showlab/DiffSim)  
**Area**: Image Generation / Visual Similarity
**Keywords**: Diffusion models, visual similarity, attention alignment, perceptual metrics, style similarity

## TL;DR

DiffSim is the first work to discover that attention-layer features of pretrained diffusion models (Stable Diffusion) can be used to measure visual similarity. It proposes the Aligned Attention Score (AAS), which aligns features from two images in the self-attention and cross-attention layers of the U-Net and computes cosine similarity, achieving state-of-the-art performance on multiple benchmarks covering human perceptual alignment, style similarity, and instance consistency.

## Background & Motivation

- The rise of **custom generation** has made evaluating the similarity between generated and reference images increasingly important.
- **Traditional perceptual metrics** (LPIPS, SSIM, etc.) compare low-level color and texture at the pixel/patch level and fail to capture mid- to high-level similarity such as layout, pose, and semantic content.
- **CLIP / DINO**, though commonly used for semantic similarity, heavily compress images into $1 \times 768$-dimensional feature vectors for cosine similarity computation, losing substantial appearance detail.
- Methods such as **DreamSim** train on human-annotated data to align with human preferences, but suffer from limited out-of-domain generalization.
- **Key insights**:
    - ReferenceNet and related methods demonstrate that U-Net self-attention layer features preserve appearance consistency.
    - Custom Diffusion shows that the `to_K` / `to_V` matrices of cross-attention are critical for concept learning.
    - IP-Adapter demonstrates that injecting IP tokens into cross-attention enables consistent generation.
    - → The features in these layers inherently encode visual appearance information and can be directly leveraged for similarity evaluation.

## Method

### Overall Architecture

DiffSim proposes two implementations:
1. **DiffSim-S**: Utilizes self-attention layers for feature alignment and similarity computation.
2. **DiffSim-C**: Utilizes IP-Adapter Plus in cross-attention layers for feature alignment.

Both share the same core component: the **Aligned Attention Score (AAS)**.

### Key Designs

1. **Aligned Attention Score (AAS)**:

    - Conventional methods assume pixel-level feature alignment, which breaks down under style or pose differences.
    - AAS leverages the attention mechanism to achieve implicit alignment:
    $\text{AAS}(L_A, L_B) = \cos(\text{attn}(Q_A, K_A, V_A), \text{attn}(Q_A, K_B, V_B))$
    - Specifically, the Query of image A attends to the Key/Value of image B, and the resulting output is compared via cosine similarity to the self-attention output of image A.
    - Bidirectional symmetric computation: $\text{Similarity}(L_A, L_B) = \text{AAS}(L_A, L_B) + \text{AAS}(L_B, L_A)$
    - Each image serves as both query and key, ensuring completeness of the metric.

2. **DiffSim-S (Self-Attention Variant)**:

    - Both images are noised to timestep $t$ and passed through the $n$-th self-attention layer of the U-Net.
    - The latent representations $z_{t,\text{self},n}^A$ and $z_{t,\text{self},n}^B$ are extracted from that layer.
    - AAS is computed within the self-attention of that layer.
    - **Shallow layers + high-noise timesteps** are suited for low-level/style similarity; **deep layers + low-noise timesteps** are suited for semantic similarity.
    - Different granularities of similarity can be obtained simply by adjusting the layer index and timestep.

3. **DiffSim-C (Cross-Attention Variant)**:

    - IP-Adapter Plus is used to extract patch-level features from the CLIP image encoder, which are injected into the U-Net cross-attention.
    - The roles of the two images in IP-Adapter and the U-Net are swapped.
    - AAS is computed in the cross-attention layers.
    - Particularly well-suited for instance-level similarity evaluation (e.g., the CUTE dataset).

4. **Extending AAS to CLIP / DINO**:

    - AAS is not limited to diffusion models; it can also be applied to the self-attention layers of CLIP and DINOv2.
    - This yields **CLIP AAS** and **DINO v2 AAS**, which significantly improve upon the original models on certain tasks.
    - Improvements are especially notable on low-level similarity (TID2013) and style similarity (Sref).

### Loss & Training

DiffSim is **fully unsupervised** and requires no fine-tuning or training. Pretrained Stable Diffusion 1.5 is used directly; adaptation to different tasks is achieved solely by selecting appropriate layer and timestep configurations.

## Key Experimental Results

### Main Results (Table)

| Model | NIGHTS | Dreambench++ | CUTE | IP | TID2013 | Sref | InstantStyle |
|------|--------|--------------|------|-----|---------|------|--------------|
| LPIPS | 71.13% | 62.33% | 63.17% | 84.01% | **94.50%** | 87.85% | 93.15% |
| CLIP | 82.26% | 70.54% | 72.71% | 91.70% | 90.33% | 84.60% | 82.90% |
| DINOv2 | 85.24% | **72.25%** | **77.27%** | 85.35% | 91.50% | 87.20% | 86.10% |
| CLIP AAS | 82.18% | 67.23% | 71.78% | 88.03% | 96.33% | 94.45% | 97.80% |
| DINO v2 AAS | 86.47% | 70.01% | 77.04% | 90.38% | 95.83% | 95.90% | 96.65% |
| **DiffSim** | **86.52%** | 71.50% | 76.17% | **91.84%** | 94.17% | **97.40%** | **99.05%** |
| Ensemble | 89.43% | 72.15% | 77.78% | 94.92% | 95.00% | 91.70% | 88.30% |

> DiffSim achieves best performance on human perceptual alignment (NIGHTS), instance similarity (IP), and style similarity (Sref, InstantStyle).

### Ablation Study (Table)

| Setting | Sref | IP | NIGHTS |
|------|------|-----|--------|
| **DiffSim (AAS)** | **97.40%** | **92.04%** | **86.82%** |
| Diffusion feature (w/o AAS) | 78.80% | 62.47% | 66.75% |
| CLIP AAS | 71.15% | 87.36% | 80.54% |
| CLIP features (w/o AAS) | 66.50% | 82.54% | 75.84% |
| DINO v2 AAS | 78.50% | 90.38% | 86.41% |
| DINO v2 feature (w/o AAS) | 76.90% | 87.56% | 81.00% |

> **AAS is the core component**: removing AAS and computing cosine similarity directly from diffusion model features causes Sref to drop sharply from 97.4% to 78.8%. AAS yields consistent and significant gains across all model families (CLIP, DINO, Diffusion).

### Key Findings

- **Effect of timestep**: High $t$ (~900) is suited for style evaluation; low $t$ (~500) is suited for semantic/instance evaluation.
- **Effect of layer selection**: Mid-level U-Net layers (Down2, Up0) attend to instance-level features; shallow layers (Down0, Up1) attend to style.
- **Effect of resolution**: Scaling from 512 to 768 yields marginal improvements on high-resolution inputs, but the default 512 is already sufficient.
- **Architecture comparison**: DiffSim-S (SD1.5) achieves the best overall performance; DiffSim-C performs slightly better on CUTE.
- **Ensemble is effective**: Hard voting over CLIP + DINOv2 + DiffSim improves performance across multiple benchmarks.

## Highlights & Insights

- **First work to apply diffusion models to image similarity evaluation**, opening a new research direction.
- **Elegant AAS design**: implicit alignment is achieved by swapping Q/K/V, eliminating the need for explicit feature alignment steps, and the approach generalizes to any attention-based architecture.
- **No training or annotation required** is a significant practical advantage, enabling plug-and-play deployment.
- **Multi-granularity similarity**: coverage from low-level to high-level similarity is achieved simply by adjusting layer and timestep.
- **AAS enhancement of CLIP/DINO** constitutes an additional contribution, demonstrating the generality of AAS.
- The newly introduced **Sref bench** (508 styles selected by human artists) and **IP bench** (299 IP characters) address existing gaps in evaluation coverage.

## Limitations & Future Work

- **Over-reliance on background**: DiffSim sometimes over-emphasizes background features and misses critical details of small foreground objects (e.g., retrieving dog images with similar backgrounds when querying a small cat). Cropping the foreground can partially mitigate this.
- The current implementation is based on SD1.5; transferring to more recent diffusion models (SDXL, SD3, Flux) may yield further improvements.
- Inference is relatively slow due to forward diffusion and U-Net computation, making it less lightweight than CLIP/DINO.
- Optimal layer and timestep selection requires grid search, which is not fully automated.
- For highly abstract semantic similarity tasks, CLIP may still hold an advantage.

## Related Work & Insights

- **ReferenceNet / IP-Adapter**: Demonstrate that self-attention/cross-attention layers encode appearance information, serving as the direct inspiration for DiffSim.
- **DreamSim**: Integrates CLIP+DINO features and trains on human annotations, representing the supervised approach.
- **CSD**: Extracts style descriptors via multi-label contrastive learning.
- **LPIPS**: A classic perceptual similarity metric that remains competitive on low-level similarity tasks.
- Insight: The internal representations of generative models are not only capable of generating images but can also capture relationships between images.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to discover that attention layers of diffusion models can be directly used for similarity evaluation; AAS design is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 benchmarks, diverse baselines, detailed ablations, new datasets, and extended applications.
- **Writing Quality**: ⭐⭐⭐⭐ — Method analysis is clear and experiments are thorough, though some sections are notation-dense.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new direction, code is open-sourced, two new benchmarks are contributed, and AAS generalizes to other models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SliderSpace: Decomposing the Visual Capabilities of Diffusion Models](sliderspace_decomposing_the_visual_capabilities_of_diffusion_models.md)
- [\[ICCV 2025\] VSC: Visual Search Compositional Text-to-Image Diffusion Model](vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](../../CVPR2026/image_generation/dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](../../CVPR2026/image_generation/erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[ICCV 2025\] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](csd-var_content-style_decomposition_in_visual_autoregressive_models.md)

</div>

<!-- RELATED:END -->
