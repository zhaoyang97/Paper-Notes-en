---
title: >-
  [Paper Note] Text2Place: Affordance-aware Text Guided Human Placement
description: >-
  [ECCV 2024][Image Generation] Proposes Text2Place—the first method for realistic human placement guided by text. It optimizes Gaussian-blob-parameterized semantic masks using Score Distillation Sampling (SDS) loss to learn scene affordances, followed by subject-conditioned inpainting for identity-preserving human placement.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 3143837dac8cc09e
---

# Text2Place: Affordance-aware Text Guided Human Placement

**Conference**: ECCV 2024  
**arXiv**: [2407.15446](https://arxiv.org/abs/2407.15446)  
**Area**: Image Generation

## TL;DR

Proposes Text2Place—the first method for realistic human placement guided by text. It optimizes Gaussian-blob-parameterized semantic masks using Score Distillation Sampling (SDS) loss to learn scene affordances, followed by subject-conditioned inpainting for identity-preserving human placement.

## Background & Motivation

- Given a background scene, humans can intuitively reason about where to place a person and what pose they should adopt—an ability referred to as "affordance."
- **Key Challenge**: Designing computational models to reason about these affordances is highly challenging, requiring the consideration of diverse backgrounds, human scales and poses, and identity preservation.
- Limitations of Prior Work:
    - Early methods were limited to specific datasets (such as sitcom scenes).
    - Kulal et al. only modeled local affordance within a given bounding box, failing to reason about global affordance (such as choosing whether to sit or stand).
    - Ramrakhya et al. required large-scale training data (creating paired datasets via inpainting).
- **Core Problem**: How to learn text-guided global and local human affordances without large-scale training?

## Method

### Overall Architecture

Divided into two stages:
1. **Semantic Mask Optimization**: Utilizes SDS loss to optimize blob-parameterized semantic masks, locating regions in the scene suitable for placing human figures.
2. **Subject-conditioned Inpainting**: Learns the text embedding of the subject via Textual Inversion, and uses the inpainting pipeline of a T2I model for identity-preserving human placement.

### Key Designs

**1. Gaussian-Blob-Based Semantic Mask Parameterization**

Directly learning semantic masks in pixel space leads to collapse (the mask covers the entire image). This work proposes parameterizing the mask with $K$ interconnected Gaussian blobs:

Each blob is defined by the following parameters:
- Center position $\mathbf{x} \in [0,1]^2$
- Scale $\mathbf{s}$, aspect ratio $\mathbf{a}$, rotation angle $\theta$
- Distance between consecutive blobs $\mathbf{r}$ (fixed)

Blobs are connected by a fixed distance, with the $i$-th blob center derived from the previous one:
$$\mathbf{x}_i = \mathbf{x}_{i-1} + [\mathbf{r}\cos(\alpha_i), \mathbf{r}\sin(\alpha_i)]^T$$

Each blob's mask is calculated using the Mahalanobis distance: $\mathcal{M}_i[\mathbf{x}_{grid}] = \exp(-0.5 \cdot D^m(\mathbf{x}_{grid}, \mathbf{x}_i))$

During training, only the first blob center $\mathbf{x}_1$, all rotation angles $\theta_i$, and relative angles $\alpha_i$ are optimized, while $\mathbf{s}$, $\mathbf{a}$, and $\mathbf{r}$ are **fixed** to prevent the mask from expanding infinitely.

**2. SDS Loss-Driven Mask Optimization**

- Create a learnable foreground human image $\mathcal{I}_p$ (initialized as a copy of the background image $\mathcal{I}_b$).
- Composite $\mathcal{I}_p$ and $\mathcal{I}_b$ into $\mathcal{I}_c$ using the mask $\mathcal{M}$.
- Optimize $\mathcal{I}_c$ using SDS loss (guidance scale = 200), and backpropagate gradients to update the parameters of $\mathcal{I}_p$ and $\mathcal{M}$.
- As training progresses, $\mathcal{I}_p$ generates a person, and $\mathcal{M}$ converges to the correct position and shape.

**3. Subject-Conditioned Inpainting**

- Learn the token embedding $\mathbf{V*}$ from a small number of subject images (3-5 images) via Textual Inversion.
- Utilize the inpainting prompt (e.g., "A $\mathbf{V*}$ person sitting on sofa") and the optimized semantic mask.
- Use the inpainting pipeline of Stable Diffusion XL for generation.
- The rich scene-human priors of the T2I model allow the output to automatically adapt to a plausible human pose.

### Loss & Training

**Mask Optimization Stage**: Uses Score Distillation Sampling (SDS) loss, with the guidance scale set to 200, calculated on the composited image $\mathcal{I}_c$.

**Inpainting Stage**: Standard noise prediction loss (inherent to the T2I inpainting pipeline).

## Key Experimental Results

### Main Results

Quantitative comparison with baseline methods:

| Method | LPIPS↓ | CLIP-sim↑ | %Person↑ |
|------|--------|----------|---------|
| GracoNet | 0.1090 | 0.2601 | 53.48 |
| TopNet | 0.1162 | 0.2617 | 67.3 |
| LLaVA | 0.1296 | 0.2501 | 20.91 |
| GPT4V | 0.1059 | 0.2615 | 64.18 |
| Ours (center) | 0.0845 | 0.2613 | 55.52 |
| **Ours** | **0.0934** | **0.2726** | **88.55** |

Text2Place significantly leads in human generation success rate (%Person): 88.55% versus 64.18% for the runner-up GPT4V (a 38% improvement), while also achieving the highest CLIP similarity.

### Ablation Study

**Blob Scale**:

| Scale $\mathbf{s}$ | LPIPS↓ | CLIP-sim↑ | %Person↑ |
|------|--------|----------|---------|
| 0.3 | 0.0537 | 0.2594 | 41.1 |
| 0.4 | 0.0806 | 0.2663 | 69.0 |
| 0.5 | 0.0858 | 0.2712 | 81.5 |
| **0.6** | **0.0904** | **0.2736** | **90.6** |
| 0.7 | 0.1074 | 0.2729 | 96.0 |

**Number of Blobs**:

| #blobs | LPIPS↓ | CLIP-sim↑ | %Person↑ |
|--------|--------|----------|---------|
| 1 | 0.1318 | 0.2780 | 93.0 |
| 3 | 0.1305 | 0.2797 | 94.9 |
| **5** | **0.0904** | **0.2736** | **90.6** |
| 7 | 0.0780 | 0.2749 | 75.0 |

Five blobs achieve the best balance between background preservation and human generation.

### Key Findings

1. **Pixel-accurate masks are counterproductive**: Experiments demonstrate that overly precise masks restrict the generation freedom of T2I inpainting models, whereas coarse blob masks actually yield better results.
2. **Blob connection is critical**: Unconnected, independent blobs tend to scatter across the image, failing to form a continuous mask suitable for human poses.
3. **Fixing some parameters is necessary**: If all blob parameters were learnable, the scale and aspect ratio would expand infinitely to minimize the SDS loss, resulting in excessively large mask regions.
4. **VLMs (LLaVA/GPT4V) have limited affordance reasoning capabilities**: LLaVA often places the bounding box in the bottom-left corner of the image, while GPT4V, though locating the box correctly, often produces incorrect sizes.
5. **Textual Inversion is sufficient**: A simple Textual Inversion with 5 images is enough to achieve high-quality identity preservation.

## Highlights & Insights

- The problem definition is highly elegant: Semantic Human Placement unifies global affordance reasoning, local pose adaptation, and identity preservation into a single framework.
- **No large-scale training required** is the core advantage: Affordances are learned solely by distilling the knowledge of the T2I model through SDS loss.
- The blob parameterization design is simple yet effective, requiring very few learnable parameters (around $5 \times 4 = 20$ parameters), which enables fast training.
- Rich downstream applications: Scene hallucination, multi-person placement, textual attribute editing, child placement, etc.
- The counter-intuitive finding that "coarse masks are better than precise masks" provides crucial guidance for practical applications.

## Limitations & Future Work

- **Difficulties with small objects**: Blob parameterization occupies relatively large image regions, making it less effective for placing small objects.
- **Lack of precise pose control**: The generated human pose is implicitly determined by the T2I inpainting model and cannot be precisely specified.
- The evaluation dataset is small (only 30 background images + 15 celebrity subjects), raising questions about statistical significance.
- The end-to-end pipeline is time-consuming, requiring 1000 iterations for SDS optimization in addition to the training time for Textual Inversion.
- The optimal configuration of blob parameters (scale, number) requires manual adjustment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to define the Semantic Human Placement task; the combination of blob parameterization and SDS distillation is novel.
- **Value**: ⭐⭐⭐⭐ — No large-scale training required, supporting diverse scenarios and downstream tasks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparison (including VLM baselines), and detailed ablations, though the dataset size is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, clear methodology explanation, and highly intuitive and abundant illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)

</div>

<!-- RELATED:END -->
