---
title: >-
  [Paper Note] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Concept Unlearning] This paper identifies that combining training-based concept unlearning with training-free safety guidance (negative prompt guidance) yields degraded performance, and proposes replacing explicit negative prompts with implicit concept embeddings obtained via Concept Inversion, effectively restoring the defensive capability of training-free methods on unlearned models.
tags:
  - NeurIPS 2025
  - Image Generation
  - Concept Unlearning
  - Safety Guidance
  - Negative Prompt
  - Concept Inversion
  - Text-to-Image Safety
date: 2026-05-08
content_hash: 9beec0fb9bca42ee
---

# Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.04834](https://arxiv.org/abs/2511.04834)
**Code**: None
**Area**: Image Generation
**Keywords**: Concept Unlearning, Safety Guidance, Negative Prompt, Concept Inversion, Text-to-Image Safety

## TL;DR

This paper identifies that combining training-based concept unlearning with training-free safety guidance (negative prompt guidance) yields degraded performance, and proposes replacing explicit negative prompts with implicit concept embeddings obtained via Concept Inversion, effectively restoring the defensive capability of training-free methods on unlearned models.

## Background & Motivation

Text-to-image diffusion models (e.g., Stable Diffusion) pose risks of generating harmful content. Two mainstream safety approaches have been proposed: (1) training-based concept unlearning, which fine-tunes model weights to "forget" harmful concepts (e.g., ESD, SPM, UCE, DUO); and (2) training-free guidance methods, which steer generation away from harmful content at inference time using negative prompts (e.g., SLD, SAFREE).

These two approaches are orthogonal by design and are theoretically composable for stronger safety guarantees. However, the authors identify a critical incompatibility: **models that have undergone unlearning fine-tuning are no longer responsive to explicit negative prompt words** (e.g., "Sexual Acts," "Nudity"), causing training-free methods applied directly to unlearned models to yield only marginal improvements or even performance degradation.

The core insight is that although unlearned models are insensitive to explicit text prompts, research on Concept Inversion demonstrates that **unlearned models can still generate harmful content via implicit embeddings**, indicating that vectors representing harmful concepts persist in the text embedding space but are simply inaccessible through manually chosen explicit words.

## Method

### Overall Architecture

The method is straightforward: the **manually selected negative prompt embeddings** $\mathbf{C}_n$ used in training-free safety methods such as SLD and SAFREE are replaced with **implicit concept embeddings** $\mathbf{C}_*$ obtained via Concept Inversion. This requires no modification to either the unlearning method or the training-free method, enabling plug-and-play integration.

### Key Designs

1. **Review of Training-Free Safety Methods**: SLD augments classifier-free guidance (CFG) with an additional guidance term based on the negative prompt embedding $\mathbf{c}_n$, adaptively subtracting scores in the harmful direction. SAFREE constructs a negative subspace and adjusts prompt token embeddings that are close to this subspace. Both methods rely on user-specified negative prompt words.

2. **Obtaining Implicit Embeddings via Concept Inversion**: Using the Textual Inversion framework, an embedding vector $\mathbf{c}_*$ is optimized by minimizing the LDM denoising loss on the unlearned model, such that it represents the forgotten harmful concept. Specifically, given a dataset of harmful images $\mathbf{x}$, the optimal concept embedding is found by solving $\mathbf{c}_* = \arg\min_{\mathbf{c}} \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, \mathbf{c}, t)\|_2^2]$. Optimization uses the Adam optimizer with learning rate $5 \times 10^{-3}$, batch size 1, and 3000 gradient steps.

3. **Embedding Substitution**: The obtained implicit concept embedding $\mathbf{C}_*$ directly replaces the original $\mathbf{C}_n$ in both SAFREE's embedding adjustment function and SLD's negative guidance term. Experiments set $K_* = 1$ (i.e., a single concept embedding).

### Loss & Training

The Concept Inversion stage employs the standard LDM denoising loss (L2 loss), optimizing only the embedding vector $\mathbf{c}$ while keeping all model parameters frozen. The entire Concept Inversion process takes approximately 15 minutes on an NVIDIA RTX 3090.

## Key Experimental Results

### Main Results

DUO is used as the base unlearning model, evaluated across 4 checkpoints with different hyperparameter values $\beta$.

**Violence Task — Ring-a-Bell Benchmark**:

| $\beta$ | Method | DSR ↑ | PP ↑ |
|---------|--------|-------|------|
| 1000 | DUO | 0.613 | 0.820 |
| 1000 | DUO+SLD | 0.760 | 0.784 |
| 1000 | DUO+SLD+Ours | **0.880** | 0.693 |
| 1000 | DUO+SAFREE | 0.793 | 0.733 |
| 1000 | DUO+SAFREE+Ours | **0.947** | 0.676 |

**Nudity Task — I2P Benchmark**:

| $\beta$ | Method | DSR ↑ | PP ↑ |
|---------|--------|-------|------|
| 2000 | DUO | 0.442 | 0.802 |
| 2000 | DUO+SLD | 0.484 | 0.777 |
| 2000 | DUO+SLD+Ours | **0.937** | 0.712 |
| 2000 | DUO+SAFREE | 0.568 | 0.678 |
| 2000 | DUO+SAFREE+Ours | **0.947** | 0.632 |

### Ablation Study (Transferability Validation)

Concept embeddings extracted from the $\beta=500$ checkpoint are directly applied to other checkpoints:

| $\beta$ | DUO+SLD+Ours (shared CE) DSR | DUO+SAFREE+Ours (shared CE) DSR |
|---------|-------------------------------|----------------------------------|
| 250 | 1.000 | 0.990 |
| 1000 | 0.990 | 0.990 |
| 2000 | 0.905 | 0.968 |

### Key Findings

- When SLD/SAFREE are applied directly to unlearned models, DSR on the nudity task does not improve and may even decrease.
- Replacing with implicit concept embeddings leads to substantial DSR improvements across all checkpoints (nudity task: maximum gain from 0.44 to 0.94).
- Concept embeddings transfer across checkpoints; embeddings extracted from one checkpoint generalize effectively to other checkpoints of the same base model.
- When the extracted concept embeddings are used to prompt the original SD v1.4, harmful images can still be generated, indicating that unlearned and original models share residual negative text embedding spaces.

## Highlights & Insights

- **Precisely Identified Problem**: This work is the first to systematically demonstrate a fundamental incompatibility between training-based unlearning and training-free guidance — unlearned models are effectively "immune" to explicit negative prompts.
- **Elegant Simplicity**: The proposed fix is a single embedding substitution that requires no structural modification to any existing method, embodying a "fighting fire with fire" philosophy.
- **Transferability Finding**: The transferability of implicit concept embeddings across checkpoints suggests that residual concept representations in unlearned models exhibit a degree of shared structure.

## Limitations & Future Work

- A separate concept embedding must be extracted for each unlearned model, limiting full plug-and-play applicability.
- A harmful image dataset is required as input for Concept Inversion, which may be restrictive in practical deployment settings.
- Validation is limited to a single unlearning method (DUO); other methods (ESD, UCE, etc.) remain untested.
- Only the simple case of $K_* = 1$ is considered; the effect of multiple concept embeddings warrants further exploration.
- Evaluation covers only nudity and violence tasks.

## Related Work & Insights

- DUO applies preference optimization for image-level unlearning and represents a strong current baseline.
- SLD and SAFREE represent two dominant paradigms for training-free safety guidance.
- Concept Inversion was originally proposed as an attack against unlearning (to recover forgotten concepts); this paper innovatively repurposes it for defense.
- This work highlights that **combining safety methods does not necessarily yield additive benefits**, and that a deeper understanding of each method's mechanism is essential before integration.

## Rating

- Novelty: ⭐⭐⭐⭐ (Valuable problem identification; the method itself is relatively simple)
- Experimental Thoroughness: ⭐⭐⭐ (Only one unlearning method and two tasks; limited in scale)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic and well-motivated)
- Value: ⭐⭐⭐⭐ (Reveals compatibility issues among safety methods; practically meaningful)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](../../CVPR2026/image_generation/when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)

</div>

<!-- RELATED:END -->
