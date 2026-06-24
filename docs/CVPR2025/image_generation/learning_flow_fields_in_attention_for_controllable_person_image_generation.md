---
title: >-
  [Paper Note] Learning Flow Fields in Attention for Controllable Person Image Generation
description: >-
  [CVPR 2025][Image Generation][virtual try-on] Proposes Leffa (Learning Flow Fields in Attention), which converts attention maps into flow fields within the attention layers of diffusion models and performs pixel-level regularization supervision. This explicitly guides the target query to attend to the correct reference key regions, successfully reducing fine-grained detail distortions (textures, text, logos) with **zero additional inference overhead**. It achieves state-of-th…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "virtual try-on"
  - "pose transfer"
  - "diffusion model"
  - "flow field"
  - "attention regularization"
date: 2026-05-08
content_hash: ca106a2fb263e7d2
---

# Learning Flow Fields in Attention for Controllable Person Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.08486](https://arxiv.org/abs/2412.08486)  
**Code**: TBD (Meta AI)  
**Area**: Image Generation  
**Keywords**: virtual try-on, pose transfer, diffusion model, flow field, attention regularization

## TL;DR

Proposes Leffa (Learning Flow Fields in Attention), which converts attention maps into flow fields within the attention layers of diffusion models and performs pixel-level regularization supervision. This explicitly guides the target query to attend to the correct reference key regions, successfully reducing fine-grained detail distortions (textures, text, logos) with **zero additional inference overhead**. It achieves state-of-the-art (SOTA) performance in both virtual try-on (VITON-HD, DressCode) and pose transfer (DeepFashion).

## Background & Motivation

**Background**: Controllable person image generation (virtual try-on, pose transfer) based on diffusion models has achieved high-quality results. However, fine-grained texture distortions (e.g., incorrect stripe directions, distorted text, incorrect numbers of buttons) still persist upon close observation.

**Limitations of Prior Work**:
1. **Auxiliary Model Solutions** (IDM-VTON, OOTDiffusion): Incorporate CLIP/DINOv2 features or warping models, which increases model complexity but lacks explicit visual consistency supervision.
2. **Multi-stage Inference** (Yang et al.): Increases computational cost during inference.
3. **Root Cause**: By **visualizing attention maps**, the authors find that the target queries in regions with detail distortions scatter their attention to incorrect areas instead of focusing on the corresponding locations in the reference.

**Key Findings**: Manually correcting the attention maps (by shifting the highest response to the correct region) significantly repairs texture distortions **without any additional training**. This inspires the research direction of guiding attention via explicit supervision.

## Method

### Overall Architecture

Baseline based on SD1.5:
- Duplicates the pretrained UNet into a **Generative UNet** (processing the target image) and a **Reference UNet** (processing the reference image).
- Removes the text encoder and text cross-attention (purely visual condition).
- Enables feature interaction between the two UNets via **Spatially Concatenated Self-Attention**.

Leffa loss is introduced as a regularization term during the fine-tuning stage, requiring **zero extra parameters and inference overhead**.

### Key Designs

#### 1. Attention Flow Fields (Flow Fields in Attention)

Mechanism — Interpreting the attention map as spatial correspondence:
- In the $l$-th attention layer, $Q = F_{gen}^l$ (target) and $K = F_{ref}^l$ (reference).
- Calculate the attention map $A^l = \text{softmax}(QK^\top / \sqrt{d} / \tau)$, and average over the head dimension to get $\hat{A^l}$.
- Construct a normalized coordinate grid $C^l \in \mathbb{R}^{n^l \times 2}$ (from top-left $[-1,-1]$ to bottom-right $[1,1]$).
- **Flow Field** $\mathcal{F}^l = \hat{A^l} \cdot C^l$: Each target token weighted-aggregates the reference coordinates to find the spatial location it "attends" to.

#### 2. Pixel-level Flow Field Supervision (Leffa Loss)

- Bilinearly upsample the flow field to the original image resolution $\mathcal{F}_{up}^l \in \mathbb{R}^{H \times W \times 2}$.
- Perform grid sampling using $\mathcal{F}_{up}^l$ to warp the reference image $I_{ref}$ to the target space, yielding $I_{warp}^l$.
- L2 Loss: $\mathcal{L}_{leffa} = \sum_{l=1}^{L} \| I_{tgt} * I_m - I_{warp}^l * I_m \|_2^2$

During training, $I_{src} = I_{tgt}$ (the same image), and the mask $I_m$ restricts the loss to clothing/human body regions only.

#### 3. Carefully Designed Application Conditions

- **Attention Layer Selection**: Only high-resolution attention layers with resolution $\ge 1/32$ of the original image are involved (low-resolution warping is imprecise).
- **Timestep Selection**: Leffa loss is computed only when $t < 500$ (out of $T=1000$) (when noise is too high, attention cannot align semantics correctly).
- **Temperature Coefficient**: A larger $\tau=2.0$ is used to make attention smoother and more fault-tolerant.
- **Progressive Training**: Baseline training at low resolution $\rightarrow$ training at high resolution $\rightarrow$ fine-tuning with Leffa loss in the **final stage**.

### Loss & Training

$\mathcal{L}_{finetune} = \mathcal{L}_{diffusion} + \lambda_{leffa} \mathcal{L}_{leffa}$

$\lambda_{leffa} = 10^{-3}$, using Leffa loss as a regularization term without interfering with the main generation quality.

## Key Experimental Results

### Main Results

**VITON-HD Virtual Try-On**:

| Method | Paired FID ↓ | SSIM ↑ | LPIPS ↓ | Unpaired FID ↓ |
|------|-------------|--------|---------|---------------|
| CatVTON | 5.42 | 0.870 | 0.057 | 9.02 |
| IDM-VTON | 5.76 | 0.850 | 0.063 | 9.84 |
| StableVITON | 8.23 | 0.888 | 0.073 | - |
| **Leffa** | **4.54** | **0.899** | **0.048** | **8.52** |

Paired FID drops from 5.42 to **4.54** (−16.2%), and LPIPS drops from 0.057 to **0.048** (−15.8%).

**DressCode Virtual Try-On (All Categories)**:

| Method | Paired FID ↓ | SSIM ↑ | Unpaired FID ↓ |
|------|-------------|--------|---------------|
| CatVTON | 3.99 | 0.892 | 6.14 |
| OOTDiffusion | 4.61 | 0.885 | 12.57 |
| **Leffa** | **2.06** | **0.924** | **4.48** |

Paired FID drops from 3.99 to **2.06** (−48.4%), showing a highly significant improvement.

**DeepFashion Pose Transfer (512×352)**:

| Method | FID ↓ | SSIM ↑ | LPIPS ↓ |
|------|-------|--------|---------|
| CFLD | 9.36 | 0.729 | 0.171 |
| PIDM | 9.81 | 0.684 | 0.192 |
| **Leffa** | **7.75** | 0.714 | **0.159** |

### Key Findings

- Leffa loss is **model-agnostic**: applying it to IDM-VTON reduces Paired FID from 5.76 $\rightarrow$ 5.20, and applying it to CatVTON reduces it from 5.42 $\rightarrow$ 5.11.
- Visualization validation: After incorporating Leffa, the attention maps transition from a scattered state to precisely aligning with the corresponding regions.
- Temperature $\tau=2.0$ is optimal: too small yields unstable gradients, while too large leads to blurry matches.
- Timestep threshold of 500 is optimal: below 200 is too strict (insufficient supervision signal), while above 700 introduces too much noise interference.

## Highlights & Insights

1. **Deep Insights**: Attributes the root cause of detail distortions to attention visualization, and validates the causal relationship via manual correction experiments—a textbook research methodology.
2. **Extremely Simple Implementation**: Leffa loss only requires calculating the flow field from existing attention maps plus an L2 loss, requiring zero extra parameters and zero additional inference overhead.
3. **Model Agnosticity**: Highly versatile, it can be plugged and played into any diffusion model that utilizes reference attention.
4. **Unified Framework**: A single baseline handles both virtual try-on and pose transfer tasks simultaneously, achieving a clean and concise architecture.
5. **Substantial Improvement on DressCode** (Paired FID 2.06): Indicates that Leffa's detail-preservation advantage is even more prominent on complex clothing categories.

## Limitations & Future Work

1. Built on SD1.5, transferring to stronger base models such as **SDXL/SD3** could potentially yield further improvements.
2. When severe **occlusions** or **large angle variations** exist between the reference and target images, the flow field assumption (one-to-one mapping) may break down.
3. Validated only on person images; whether controllable generation of **general objects/scenes** can benefit from Leffa remains unexplored.
4. The progressive training strategy increases the total training steps; whether **Leffa loss can be introduced from the early stages of training** warrants further investigation.

## Related Work & Insights

- **IDM-VTON** (Choi et al., CVPR): One of the current virtual try-on SOTAs, on which this work validates Leffa's generalization ability.
- **CatVTON** (Chong et al.): A representative of the concatenated self-attention paradigm; the baseline design of this paper is similar but cleaner.
- **CFLD** (Lu et al.): A pose transfer SOTA; this paper significantly outperforms it in FID while yielding slightly lower SSIM.
- **Insight**: The "attention $\rightarrow$ flow field $\rightarrow$ pixel-level supervision" paradigm can be generalized to any attention mechanism requiring spatial alignment, such as inpainting, image editing, and video generation.

## Rating

⭐⭐⭐⭐⭐ — Precise insights, elegant method (the transition from attention to flow field is highly natural), comprehensive experiments (3 datasets × 2 tasks × model-agnostic validation), and exceptional practicality (zero additional inference overhead). A top-tier CVPR work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)
- [\[CVPR 2025\] FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation](foundhand_large-scale_domain-specific_learning_for_controllable_hand_image_gener.md)
- [\[CVPR 2025\] BootComp: Controllable Human Image Generation with Personalized Multi-Garments](controllable_human_image_generation_with_personalized_multi-garments.md)
- [\[CVPR 2025\] Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
