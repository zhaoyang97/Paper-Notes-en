---
title: >-
  [Paper Note] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Text embedding optimization] This paper proposes DATE (Diffusion Adaptive Text Embedding), which dynamically updates text embeddings during diffusion model sampling based on the current denoising intermediate results, improving text-image semantic alignment without any additional training.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Text embedding optimization"
  - "diffusion model sampling"
  - "text-image alignment"
  - "test-time optimization"
  - "adaptive conditioning"
date: 2026-05-08
content_hash: 2a5b31fb802e5de2
---

# Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.23974](https://arxiv.org/abs/2510.23974)  
**Code**: [https://github.com/aailab-kaist/DATE](https://github.com/aailab-kaist/DATE)  
**Area**: Image Generation
**Keywords**: Text embedding optimization, diffusion model sampling, text-image alignment, test-time optimization, adaptive conditioning

## TL;DR

This paper proposes DATE (Diffusion Adaptive Text Embedding), which dynamically updates text embeddings during diffusion model sampling based on the current denoising intermediate results, improving text-image semantic alignment without any additional training.

## Background & Motivation

Text-to-image diffusion models rely on pretrained text encoders (CLIP, T5) to encode prompts into fixed embeddings that remain unchanged across all sampling steps. However, different timesteps of the diffusion process exert fundamentally different influences on the image: early steps determine global structure while later steps refine details. Using static embeddings cannot adapt to this semantic evolution, often leading to concept confusion (e.g., generating "two people" when prompted for "one person").

Existing improvement directions include fine-tuning model parameters (costly), data-space guidance (Universal Guidance, requiring careful tuning of the guidance scale), and prompt-level optimization (requiring training additional language models). These approaches overlook the optimization value of the text embeddings themselves—a critical variable that can be optimized at test time with zero training cost. EBCA attempts energy-based updates at cross-attention layers but lacks global semantic control and severely degrades FID. P2L directly optimizes text embeddings for inverse problems but is limited to specific tasks. DATE provides a general, training-free framework applicable to arbitrary text-to-image diffusion models.

## Method

### Overall Architecture

DATE replaces fixed text embeddings with timestep- and instance-adaptive dynamic embeddings. At each sampling step: (1) the Tweedie formula is used to estimate a mean-predicted image $\bar{\mathbf{x}}_0$ from the noisy data and the current embedding; (2) the gradient of an evaluation function $h(\bar{\mathbf{x}}_0; y)$ with respect to the text embedding is computed; (3) the embedding is updated along the normalized gradient direction; (4) the updated embedding is used to perform the standard denoising step. The entire process runs purely at inference time without modifying network parameters or architecture.

### Key Designs

1. **Timestep-level objective decomposition**: The original objective is to maximize the evaluation function value of the final generated image, which involves joint optimization over all sampling steps. DATE decomposes this into sequential per-step optimization: at timestep $t$, the embeddings for $\tau < t$ are fixed to $\mathbf{c}_t$, and the Tweedie formula (one forward pass) is used to estimate $\bar{\mathbf{x}}_0$ and compute $h$. A Taylor expansion then transforms the problem into a constrained optimization in embedding space. Theoretically, this decomposition is equivalent to the original joint optimization (Proposition 1), guaranteeing no degradation compared to fixed embeddings.

2. **Normalized gradient single-step update**: Using the Cauchy-Schwarz inequality, a closed-form solution to the constrained optimization is derived: $\hat{\mathbf{c}}_t = \mathbf{c}_{\text{org}} + \rho \cdot \nabla_{\mathbf{c}} h_t / \|\nabla_{\mathbf{c}} h_t\|_2$. Normalization ensures a constant update magnitude (controlled by $\rho$), inspired by SAM (Sharpness-Aware Minimization). Theoretical analysis (Theorem 2) shows that the updated embedding is equivalent to adding an embedding-domain guidance term to the original score function, balancing semantic alignment and preservation of the model distribution.

3. **Computational efficiency strategies**: Updates are applied at only a subset of sampling steps (e.g., 10%), with the most recently updated embedding reused at non-update steps. Experiments show that using the previous step's updated embedding as the starting point for the next update (rather than always returning to the initial embedding) enables broader exploration of the embedding space and improves CLIP score. Updates at middle-to-late timesteps contribute more to alignment performance, consistent with the detail refinement stage. FP16 inference is supported to further reduce overhead (time: 7.82min → 4.40min; VRAM: 61.5GB → 32.9GB).

### Loss & Training

DATE requires no training. The objective is to maximize an online evaluation function $h_t$ at each step, which can be any differentiable text-image alignment metric: CLIP Score (semantic alignment), ImageReward (human preference), PickScore, Aesthetic Score, or weighted combinations thereof. Combined optimization of multiple metrics can yield synergistic gains—for example, under the CS+IR combination, CLIP Score even surpasses optimizing CS alone.

## Key Experimental Results

### Main Results

Evaluated on 5,000 images from the COCO validation set using SD v1.5 + DDIM 50 steps:

| Method | FID↓ | CLIP Score↑ | ImageReward↑ |
|--------|------|-------------|--------------|
| Fixed embedding (50 steps) | 18.66 | 0.3204 | 0.2132 |
| Fixed embedding (70 steps) | 18.27 | 0.3199 | 0.2137 |
| EBCA | 25.85 | 0.2877 | -0.3128 |
| Universal Guidance | 18.56 | 0.3216 | 0.2221 |
| **DATE 10% update (CS)** | **17.90** | **0.3237** | 0.2364 |
| **DATE 10% update (IR)** | 18.61 | 0.3224 | **0.4792** |

Consistent improvements across backbones: SD3 (IR: 1.0018→1.0457), FLUX (CS: 0.3257→0.3283), SDXL (IR: 0.7284→0.9096).

### Ablation Study

| Variant | FID↓ | CS↑ | IR↑ |
|---------|------|-----|-----|
| Fixed embedding | 18.66 | 0.3204 | 0.2132 |
| Random direction update | 18.66 | 0.3204 | 0.2136 |
| Compute $h$ on noisy data | 18.80 | 0.3200 | 0.2121 |
| Non-normalized gradient | 18.46 | 0.3212 | 0.2225 |
| **DATE (normalized)** | **17.91** | **0.3220** | **0.2229** |

### Key Findings

- Random updates perform on par with fixed embeddings, demonstrating that the gradient direction—not mere perturbation—is the key factor.
- At 85% of timesteps, cosine similarity between update directions is below 0.1, confirming that different steps require different embedding directions.
- Update directions for different instances of the same prompt are nearly orthogonal (< 0.05), indicating that updates are instance-specific.
- Middle-to-late timestep updates are more effective than early ones—embedding adjustments are more valuable during the detail refinement stage.
- While DATE improves the target metric, other non-target metrics also improve simultaneously, reflecting overall quality gains rather than overfitting to a single metric.

## Highlights & Insights

- DATE fills the gap in the embedding dimension among the three main optimization axes of diffusion models (parameters, latents, and text embeddings).
- The embedding update is theoretically unified as a guidance term within the score function, drawing an analogy to Classifier Guidance.
- The plug-and-play nature is strong: no model modification is required, and the method is compatible with arbitrary backbones, samplers, and evaluation functions.
- Effectiveness is consistently validated on two downstream tasks: multi-concept generation (AnE dataset) and text-guided image editing (DDPM Inversion).

## Limitations & Future Work

- Each update step requires an additional score network forward pass and gradient computation; with 10% update frequency, runtime increases by approximately 39%.
- GPU memory consumption increases significantly (24GB → 61.5GB); FP16 mitigates this but remains higher than the baseline.
- The hyperparameter $\rho$ requires tuning; excessively large values degrade performance due to Taylor approximation errors.
- Performance depends on the quality of the evaluation function $h$—Aesthetic Score has low correlation with semantic alignment, and using it alone actually degrades other metrics.
- When the text encoder itself cannot effectively represent the target semantics, the gains from optimizing in the embedding space are limited.

## Related Work & Insights

- **Universal Guidance**: applies guidance in data space; DATE optimizes in embedding space—the latter achieves better FID (17.90 vs. 18.56).
- **Textual Inversion**: optimizes special token embeddings but requires multi-step training; DATE performs single-step updates over all embeddings.
- **SAM**: the normalized gradient idea is generalized from model parameter optimization to text embedding optimization, suggesting broader applicability across domains.
- The concept of timestep-adaptive embeddings in DATE can be extended to frame-level adaptive conditioning in video generation.

## Rating

⭐⭐⭐⭐ — Theoretically clear and methodologically elegant (training-free test-time optimization), with consistent effectiveness across multiple backbones and tasks. The primary limitations are increased computational overhead and memory consumption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[NeurIPS 2025\] Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models](mitigating_sexual_content_generation_via_embedding_distortion_in_text-conditione.md)
- [\[NeurIPS 2025\] Aligning Text to Image in Diffusion Models is Easier Than You Think](aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](../../ICCV2025/image_generation/text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](../../ICCV2025/image_generation/addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)

</div>

<!-- RELATED:END -->
