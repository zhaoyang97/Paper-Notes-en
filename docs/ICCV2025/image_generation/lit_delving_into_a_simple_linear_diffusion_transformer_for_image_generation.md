---
title: >-
  [Paper Note] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation
description: >-
  [Image Generation] > This paper systematically investigates how to safely and efficiently convert a pretrained DiT into a linear attention variant called LiT. It proposes five practical guidelines—depth-wise convolution augmentation, fewer heads, weight inheritance, selective parameter loading, and hybrid distillation—achieving comparable performance with only 20% of DiT's training steps.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 4323cd3ccc3ef301
---

# LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation

| Info | Content |
|------|------|
| Conference | ICCV 2025 |
| arXiv | [2501.12976](https://arxiv.org/abs/2501.12976) |
| Code | - |
| Area | Image Generation · Diffusion Models · Efficient Architecture |
| Keywords | linear attention, diffusion transformer, knowledge distillation, efficient generation, DiT |

## TL;DR

> This paper systematically investigates how to safely and efficiently convert a pretrained DiT into a linear attention variant called LiT. It proposes five practical guidelines—depth-wise convolution augmentation, fewer heads, weight inheritance, selective parameter loading, and hybrid distillation—achieving comparable performance with only 20% of DiT's training steps.

## Background & Motivation

### Efficiency Bottleneck of DiT

Diffusion Transformers (DiT) have demonstrated strong commercial potential in image generation, but the quadratic complexity $\mathcal{O}(N^2D)$ of their self-attention modules introduces severe latency and memory issues at high resolutions:
- At 2048px resolution, linear attention is nearly **9×** faster than softmax attention
- For DiT-S/2 at 2048px, GPU memory drops from ~14 GB to ~4 GB

### Limitations of Prior Work

Existing efficient attention methods (e.g., SANA, Mediators, CLEAR) introduce carefully engineered modifications to DiT but **sacrifice architectural simplicity**. The central question is:

> Can a purely simple linear attention be used to make LiT a safe and efficient drop-in replacement for DiT?

### Problem Decomposition

The problem is decomposed into two sub-questions:
1. What type of linear attention architecture is suitable for image generation?
2. What training strategy can effectively optimize a linear DiT?

## Method

### Overall Architecture

LiT preserves the macro- and micro-level architectural design of DiT, replacing all softmax self-attention with simple linear attention. Five practical guidelines ensure the quality of this substitution.

### Linear Attention Basics

Standard self-attention: $\text{Sim}(\mathbf{Q}, \mathbf{K}) = \exp(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})$, with complexity $\mathcal{O}(N^2D)$.

Linear attention substitutes a kernel function $\phi(\cdot)$:

$$\mathbf{O}_i = \frac{\phi(\mathbf{Q}_i) \left(\sum_{j=1}^{N} \phi(\mathbf{K}_j)^\top \mathbf{V}_j\right)}{\phi(\mathbf{Q}_i) \left(\sum_{j=1}^{N} \phi(\mathbf{K}_j)^\top\right)}$$

Complexity is reduced to $\mathcal{O}(ND^2/h)$.

### Guideline 1: Depth-wise Convolution Suffices

Directly replacing softmax attention with ReLU linear attention causes a large FID degradation (S/2: 68.40→88.46). Three remedies are evaluated:

| Method | DiT-S/2 FID↓ | DiT-B/2 FID↓ |
|------|:---:|:---:|
| Softmax baseline | 68.40 | 43.47 |
| ReLU linear baseline | 88.46 | 56.92 |
| + Depth-wise Conv (DWC) | **63.66** | **42.11** |
| + Focused Linear | 63.05 | 40.58 |
| + GELU kernel | 70.83 | 58.86 |

**Adding a single depth-wise convolution (kernel=5) already surpasses softmax attention.** The underlying reason is that noise prediction tends to rely on neighboring pixel information; DWC compensates for the locality that linear attention lacks. The Focused function (designed to sharpen distributions for classification) proves unnecessary for noise prediction.

### Guideline 2: Fewer Heads as a Free Lunch

Theoretically, the computational cost of linear attention is inversely proportional to the number of heads $h$: $\mathcal{O}(ND^2/h)$. Experiments reveal that—

**Reducing the number of heads increases GMACs but does not increase actual latency.**

| DiT | Heads | FID↓ | IS↑ |
|-----|:---:|:---:|:---:|
| S/2 | 2 | **63.24** | 22.07 |
| S/2 | 6 (default) | 63.66 | 22.16 |
| S/2 | 96 | 78.76 | 17.46 |
| XL/2 | 4 | **20.82** | 65.52 |
| XL/2 | 16 (default) | 21.69 | 63.06 |

The fewer-heads strategy increases the theoretical computational budget and, following scaling laws, raises the model's performance ceiling. Furthermore, attention maps across different heads exhibit high homogeneity (cosine similarity >0.5), suggesting that a small number of heads is sufficient to capture the principal information.

### Guideline 3: Initialize from a Converged DiT

| Pretraining Steps | FID↓ |
|:---:|:---:|
| None | 63.24 |
| 200K | 57.84 |
| 400K | 56.07 |
| 600K | 54.80 |
| 800K | **53.83** |

More thoroughly pretrained DiT weights provide better initialization for LiT, even when the two architectures are not fully identical. This is likely due to the functional decoupling of different modules in DiT—knowledge from shared components such as the FFN and adaLN can be transferred directly.

### Guideline 4: Do Not Load Attention Weights

| Loading Strategy | FID↓ |
|----------|:---:|
| Exclude attention weights | **54.80** |
| Load Q, K, V | 55.29 |
| Load K, V | 55.07 |
| Load V | 54.93 |
| Load Q | 54.82 |

Since linear and softmax attention operate under fundamentally different computational paradigms (linear attention directly computes $\mathbf{K}^\top \mathbf{V}$), forcibly loading pretrained attention weights interferes with optimization. It is recommended to load all pretrained parameters except those of the attention module.

### Guideline 5: Hybrid Knowledge Distillation

Distillation targets include not only the predicted noise but also the variance of the reverse diffusion process:

$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_1 \underbrace{\|\epsilon^{(\mathcal{T})} - \epsilon^{(\mathcal{S})}\|^2}_{\text{noise distillation}} + \lambda_2 \underbrace{\|\Sigma^{(\mathcal{T})} - \Sigma^{(\mathcal{S})}\|^2}_{\text{variance distillation}}$$

| $\lambda_1$ | $\lambda_2$ | FID↓ |
|:---:|:---:|:---:|
| 0.0 | 0.0 (no distillation) | 53.83 |
| 0.5 | 0.0 (noise only) | 51.13 |
| 0.0 | 0.05 (variance only) | 53.49 |
| **0.5** | **0.05** | **50.79** |

Variance distillation should be applied moderately ($\lambda_2=0.05$), as denoising capability remains the core objective of diffusion models.

## Key Experimental Results

### Class-Conditional ImageNet 256×256

| Model | Training Steps | FID↓ | IS↑ |
|------|:---:|:---:|:---:|
| DiT-XL/2 | 400K | 19.47 | - |
| DiG-XL/2 (GLA) | 400K | 18.53 | 68.53 |
| **LiT-XL/2** | **100K** | **12.90** | **95.80** |
| DiT-XL/2-G (cfg=1.50) | 7M | 2.27 | 278.24 |
| **LiT-XL/2-G (cfg=1.50)** | **1.4M (20%)** | **2.32** | **265.20** |

LiT achieves comparable FID (2.32 vs. 2.27) using only **20% of DiT's training steps** (1.4M vs. 7M).

### Class-Conditional ImageNet 512×512

| Model | FID↓ |
|------|:---:|
| DiT-XL/2-G (cfg=1.50) | 3.04 |
| **LiT-XL/2-G (cfg=1.50)** | **3.69** |

At 512px, LiT uses only ~23% of training steps (700K vs. 3M), with an FID gap of only 0.65.

### Text-to-Image Generation (GenEval)

| Model | Parameters | Overall↑ |
|------|:---:|:---:|
| PixArt-Σ | 0.6B | 0.52 |
| SDv2.1 | 0.9B | 0.50 |
| **LiT (1024px)** | **0.6B** | **0.48** |

LiT converted from PixArt-Σ maintains comparable GenEval scores, demonstrating that the proposed guidelines generalize to text-to-image generation.

## Highlights & Insights

1. **Actionable practical guidelines**: Five plug-and-play rules lower the barrier to deploying linear DiT in practice.
2. **"Free lunch" finding**: Fewer-head linear attention increases GMACs without increasing latency—an intriguing hardware-level insight.
3. **Remarkable efficiency**: LiT-XL/2 surpasses DiT-XL/2's 400K-step result with only 100K steps.
4. **Laptop deployment**: The authors deploy LiT-0.6B offline on a Windows 11 laptop, achieving 1K-resolution image generation.

## Limitations & Future Work

- At 512px, a non-trivial FID gap remains (3.69 vs. 3.04), indicating a small but persistent performance cost of linear attention at higher resolutions.
- GenEval overall score for text-to-image generation is slightly below PixArt-Σ (0.48 vs. 0.52), with gaps on multi-object, positional, and attribute binding tasks.
- The theoretical justification for variance distillation is insufficiently developed.
- Validation is limited to DiT and PixArt-Σ; generalization to other architectures (e.g., SD3, FLUX) remains unexplored.

## Related Work & Insights

- **Efficient attention**: SANA (Mix-FFN), Attention Mediators, CLEAR (recurrent windows)
- **Linear attention**: Flatten Transformer, EfficientViT, FLatten, etc.
- **SSM/GLA-based diffusion models**: DiG (GLA), DiM (SSM), DiffuSSM
- **Knowledge distillation**: noise distillation, sampling step distillation, etc.

## Rating

| Dimension | Score |
|------|:----:|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](../../CVPR2025/image_generation/dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
