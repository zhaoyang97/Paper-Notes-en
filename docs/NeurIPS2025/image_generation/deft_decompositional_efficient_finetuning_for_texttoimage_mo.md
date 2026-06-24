---
title: >-
  [Paper Note] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models
description: >-
  [NeurIPS 2025][Image Generation][Efficient Fine-Tuning] This paper proposes DEFT (Decompositional Efficient Fine-Tuning), which efficiently fine-tunes T2I models by decomposing weight updates into two components — subspace projection and low-rank adjustment — outperforming LoRA and PaRa on both personalized and general image generation tasks.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Efficient Fine-Tuning"
  - "Diffusion Models"
  - "Low-Rank Decomposition"
  - "Personalized Generation"
  - "General Image Generation"
date: 2026-05-08
content_hash: f465a48d14f3787f
---

# DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.22793](https://arxiv.org/abs/2509.22793)  
**Code**: [DEFT](https://github.com/DEFT)  
**Area**: Image Generation
**Keywords**: Efficient Fine-Tuning, Diffusion Models, Low-Rank Decomposition, Personalized Generation, General Image Generation

## TL;DR

This paper proposes DEFT (Decompositional Efficient Fine-Tuning), which efficiently fine-tunes T2I models by decomposing weight updates into two components — subspace projection and low-rank adjustment — outperforming LoRA and PaRa on both personalized and general image generation tasks.

## Background & Motivation

1. **Background**: Fine-tuning text-to-image (T2I) models faces challenges related to computational resources and overfitting. LoRA achieves parameter-efficient fine-tuning via low-rank updates, while PaRa reduces rank through orthogonal subspace projection.

2. **Limitations of Prior Work**: LoRA's low-rank updates lack structural constraints, making them prone to overfitting with limited control over pose and spatial layout. PaRa only reduces the rank of pretrained weights without introducing new directions. Concept interference and blending in multi-concept compositional generation remain challenging.

3. **Key Challenge**: Efficient fine-tuning requires balancing three objectives — learning the target distribution, preserving instruction-following capability, and maintaining editability (generation under diverse prompts or contexts) — which existing methods struggle to achieve simultaneously.

4. **Goal**: To design a more flexible weight update scheme that efficiently adapts to new concepts or tasks while preserving the generalization capability of the pretrained model.

5. **Key Insight**: Decompose weight updates into two complementary components — subspace projection (removing certain directions) and low-rank adjustment (injecting new directions) — jointly realized through two trainable matrices.

6. **Core Idea**: $W_{total} = (I - PP^T)W_0 + PR$, where $P$ defines the subspace projection and $R$ enables flexible adjustment within that subspace, expanding the column space of the weights.

## Method

### Overall Architecture

DEFT applies a decompositional update on the pretrained weight $W_0$: it first removes part of the subspace via $PP^T$, then injects new directions through $PR$. The approach is applicable to UNet layers in Stable Diffusion and Transformer linear layers in unified models such as OmniGen.

### Key Designs

**1. Decompositional Weight Update**

- **Function**: Injects new task-specific directions while removing old ones, balancing adaptation and retention.
- **Mechanism**: $W_{total} = (I - PP^T)W_0 + PR$. The term $(I - PP^T)W_0$ removes components in the column space of $P$, while $PR$ injects new information within that subspace. The total column space satisfies $\text{col}(W_{reduce}) + \text{col}(QR) \subseteq \text{col}(W_0) + \text{col}(Q)$.
- **Design Motivation**: Low-rank updates are most effective in directions orthogonal to the dominant singular vectors of $W_0$. PaRa only reduces rank without adding new directions; LoRA adds directions without removing old ones; DEFT achieves both simultaneously.

**2. Multiple Decomposition Strategies**

- **Function**: Provides plug-and-play options with different structural biases.
- **Mechanism**: Supports QR decomposition, truncated SVD, low-rank matrix factorization (LRMF), non-negative matrix factorization (NMF), and eigendecomposition. The non-negativity constraint in NMF yields sparser and more structured updates.
- **Design Motivation**: Different data regimes and downstream tasks may benefit from different structural biases.

**3. Differentiated Learning Rate Design**

- **Function**: Stabilizes optimization by balancing the update rates of the projection and adjustment matrices.
- **Mechanism**: A higher learning rate is used for $R$ and a lower one for $P$, analogous to the asymmetric learning rate design for $A$/$B$ in LoRA.
- **Design Motivation**: $P$ defines the subspace structure and should evolve slowly and stably, while $R$ handles content-specific adaptation and should update more rapidly.

### Loss & Training

Standard diffusion denoising loss is used. Rank=4 is adopted for DreamBench Plus, and rank=32 for general generation on VisualCloze. Personalization follows DreamBooth-style training.

## Key Experimental Results

### Main Results

Text-image alignment scores on DreamBench Plus (CLIP-T, 150 subjects × 8 prompts):

| Method | T2I Model | CLIP-T |
|--------|-----------|--------|
| Textual Inversion | SD v1.5 | 0.302 |
| DreamBooth | SD v1.5 | 0.323 |
| DreamBooth LoRA | SDXL v1.0 | 0.341 |
| PaRa | SDXL v1.0 | 0.354 |
| **DEFT (Ours)** | SDXL v1.0 | **0.361** |

General image generation performance on VisualCloze:

| Condition | Method | CLIP-Score | DINO-v1 | DINO-v2 |
|-----------|--------|-----------|---------|---------|
| Canny | OmniGen | 95.45 | 87.13 | 87.60 |
| Canny | **DEFT** | **95.78** | **90.37** | **90.65** |
| Depth | OmniGen | 92.02 | 85.16 | 77.39 |
| Depth | **DEFT** | **93.18** | **88.98** | **85.75** |

### Ablation Study

| Decomposition | CLIP-I | CLIP-T | Notes |
|---------------|--------|--------|-------|
| None (default) | baseline | baseline | Simplest; used as default |
| QR | competitive | competitive | Orthogonal constraint |
| NMF | high | highest | Non-negativity yields better prompt control |
| TSVD | competitive | competitive | SVD-based decomposition |
| Relaxing P | high | high | Learnable projection matrix |

### Key Findings

- DEFT outperforms LoRA on instruction-following (CLIP-T) by 2 percentage points, attributable to the low-rank injection expanding the fine-tuning subspace.
- In style transfer, DEFT achieves an Image Score of 0.69, substantially surpassing InstantStyle (0.60) and OmniGen (0.52).
- For multi-concept composition, DEFT does not require separate LoRA modules per concept and supports joint fine-tuning.
- Controllability metrics (Controllability F1) are on par with OmniGen, while quality metrics (SSIM) improve significantly.

## Highlights & Insights

- **Theoretical Elegance**: The "remove + inject" decomposition is conceptually clear and grounded in linear algebra theory (column space expansion proof).
- **High Flexibility**: Supports multiple decomposition methods in a plug-and-play manner.
- **Emergent Generalization**: Models fine-tuned on a small number of images exhibit combinatorial generalization to compositions not seen during training.
- **Unified Framework**: The same method applies to diverse tasks including personalization, style transfer, and conditional generation.

## Limitations & Future Work

- Decomposition is disabled by default for simplicity; automated guidance for selecting the optimal decomposition strategy is lacking.
- Validation is limited to SDXL and OmniGen; effectiveness on larger models (e.g., FLUX) remains to be verified.
- The upper limit on the number of concepts in multi-concept composition has not been explicitly tested.
- A deeper theoretical comparison with methods such as SVDiff could be explored.

## Related Work & Insights

- LoRA adds directions without removal; PaRa removes without adding; DEFT unifies both.
- Custom Diffusion fine-tunes a subset of parameters; DEFT provides a more flexible parameter adaptation space.
- Insight: Efficient fine-tuning is not merely about the number of parameters, but about the structure of updates (subspace selection).

## Rating

- Novelty: ⭐⭐⭐⭐ The decompositional update idea is theoretically grounded and clearly distinct from existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets and task types.
- Writing Quality: ⭐⭐⭐ Content is rich but could be organized more compactly.
- Value: ⭐⭐⭐⭐ Opens a new design space for efficient fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)
- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[CVPR 2025\] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](../../CVPR2025/image_generation/focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)
- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)

</div>

<!-- RELATED:END -->
