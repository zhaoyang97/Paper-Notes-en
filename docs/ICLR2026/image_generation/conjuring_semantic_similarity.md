---
title: >-
  [Paper Note] Conjuring Semantic Similarity
description: >-
  [ICLR 2026][Image Generation][semantic similarity] This paper proposes a vision-imagination-based measure of textual semantic similarity by computing the Jeffreys divergence between the reverse SDEs induced by a text-con…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "semantic similarity"
  - "diffusion model"
  - "Jeffreys divergence"
  - "SDE"
  - "text-to-image"
date: 2026-05-08
content_hash: 82dfa91f1929e791
---

# Conjuring Semantic Similarity

**Conference**: ICLR 2026
**arXiv**: [2410.16431](https://arxiv.org/abs/2410.16431)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: semantic similarity, diffusion model, Jeffreys divergence, SDE, text-to-image

## TL;DR
This paper proposes a vision-imagination-based measure of textual semantic similarity by computing the Jeffreys divergence between the reverse SDEs induced by a text-conditioned diffusion model under two text prompts. The metric is directly computable via Monte-Carlo sampling and, for the first time, quantifies the alignment between the semantic space learned by diffusion models and human annotations.

## Background & Motivation

**Background**: Semantic similarity has traditionally been measured in the text space (Word2Vec, BERT embeddings, CLIP, etc.). Liu et al. (2023) define the meaning space of autoregressive LLMs as the distribution over continuations.

**Limitations of Prior Work**: (a) Text embedding methods produce uninterpretable vector distances; (b) No existing method quantifies the quality of the semantic space learned by text-conditioned diffusion models; (c) Bender & Koller (2020) argue that language-only training is insufficient to capture semantics—grounding in the external world is required.

**Key Challenge**: Semantic similarity should be interpretable, yet existing methods yield only numerical scores without explanation. Humans compare meanings by "imagining" scenes, but systematic comparison of mental images is infeasible.

**Key Insight**: Use the diffusion model as an "imagination faculty"—the semantic distance between two texts equals the distance between the image distributions they induce.

**Core Idea**: Textual semantic similarity = Jeffreys divergence between the path measures of the reverse diffusion SDEs conditioned on two texts, computed via Monte-Carlo estimation.

## Method

### Overall Architecture
Given two texts $y_1, y_2$ and a pretrained diffusion model $s_\theta$: (1) Starting from the same noise, denoise separately under $y_1$ and $y_2$; (2) At each timestep, compute the squared difference between the two score functions $\|s_\theta(x_t, t|y_1) - s_\theta(x_t, t|y_2)\|_2^2$; (3) Sum over the denoising trajectory and average via Monte-Carlo.

### Key Designs

1. **SDE Derivation of Jeffreys Divergence**:

    - Function: Converts distribution comparison into a comparison of SDE path measures.
    - Core formula: $d_{\text{ours}}(y_1, y_2) = \mathbb{E}_{t, x \sim \frac{1}{2}p_t(\cdot|y_1) + \frac{1}{2}p_t(\cdot|y_2)} \|s_\theta(x, t|y_1) - s_\theta(x, t|y_2)\|_2^2$
    - KL divergence is derived via Girsanov's theorem and then symmetrized into the Jeffreys divergence.
    - Design Motivation: Direct comparison of image distributions (e.g., FID) requires a large number of samples. The SDE divergence can be computed incrementally during denoising, making it both efficient and theoretically rigorous.

2. **Monte-Carlo Sampling Algorithm**:

    - Sample noise from $\mathcal{N}(0,I)$ → denoise separately under $y_1$ and $y_2$ → compute the L2 norm of the score difference at each step → average. Repeat $k$ times. Setting $T=10$ steps is sufficient.

3. **Interpretability**:

    - As a byproduct, the denoising process produces visualizations—one can observe how the model morphs one concept into another (e.g., snow leopard → Bengal tiger: spots → stripes).

## Key Experimental Results

### Main Results (STS Benchmark, Spearman Correlation)

| Method | STS-B | STS12 | STS13 | STS14 | Avg |
|------|-------|-------|-------|-------|-----|
| BERT-CLS | 16.5 | 20.2 | 30.0 | 20.1 | 29.2 |
| BERT-mean | 45.4 | 38.8 | 58.0 | 58.0 | ~50 |
| SimCSE-BERT | 68.4 | 82.4 | 74.4 | 80.9 | **76.3** |
| CLIP-ViTL14 | 65.5 | 67.7 | 68.5 | 58.0 | 67.0 |
| **Ours (SD v1.4)** | ~55 | ~50 | ~55 | ~50 | ~53 |

### Ablation Study

| Configuration | Performance | Note |
|------|------|------|
| Early steps only | Weak | Low discriminability under high noise |
| Final steps only | Moderate | Informative but incomplete |
| Full trajectory (Ours) | Best | Accumulates semantic information across all scales |
| KL vs. Jeffreys | Jeffreys more stable | Symmetrization improves performance |
| $T$ step ablation | Saturates at $T=10$ | Computationally friendly |

### Key Findings
- **Zero-shot method surpasses BERT encoders**: Using Stable Diffusion alone achieves semantic similarity performance comparable to language models, demonstrating that diffusion models have learned meaningful semantic structure.
- **Interpretability as a unique advantage**: The method not only provides a numerical score but also visualizes the "morphing process" between two concepts—something text embedding methods cannot offer.
- **First quantification of semantic alignment in diffusion models**: Opens a new dimension for evaluating T2I models—assessing not only image quality but also semantic understanding.

## Highlights & Insights
- **"Meaning = the distribution of evoked images"**: Extends Wittgenstein's "meaning as use" from text to the visual domain—a compelling conceptual transfer.
- **Elegant application of Girsanov's theorem in AI**: Reduces the abstract path measure distance to a simple difference of score functions—theoretically elegant and practically useful.
- **Generalizable to any conditional generative model**: The method is not restricted to text-to-image; in principle it applies to audio-text, video-text, and other modalities.

## Limitations & Future Work
- **Underperforms dedicated embedding models**: SimCSE-BERT (76.3) vs. Ours (~53)—task-specific models retain a substantial advantage.
- **Computational cost**: Each pair requires multiple denoising passes (~2s/step × 10 steps × $k$ runs), several orders of magnitude slower than embedding-based distance.
- **Dependence on diffusion model quality**: The semantic space of SD v1.4 is limited; stronger models (e.g., DALL-E 3) may yield better results.

## Related Work & Insights
- **vs. Liu et al. (2023)**: They define semantics via the LLM continuation distribution. This paper uses the image distribution of a diffusion model—shifting from the text space to the visual space.
- **vs. CLIP score**: CLIP measures distance via aligned text-image embeddings. This paper measures distance directly within the diffusion process—more native and more interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The definition of "semantics = evoked image distribution" is highly creative; the SDE divergence derivation is mathematically elegant.
- Experimental Thoroughness: ⭐⭐⭐ Validation on the STS benchmark is solid, but the method does not surpass dedicated models and its application scope is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts are clearly articulated, derivations are rigorous, and visualizations are impressive.
- Value: ⭐⭐⭐⭐ Opens a new direction for evaluating the semantic space of diffusion models; the contribution is primarily conceptual rather than state-of-the-art performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)
- [\[ICLR 2026\] SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP](semobridge_semantic_modality_bridge_for_efficient_few-shot_adaptation_of_clip.md)
- [\[ICLR 2026\] Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study](does_semantic_noise_initialization_transfer_from_images_to_videos_a_paired_diagn.md)
- [\[ICCV 2025\] DiffSim: Taming Diffusion Models for Evaluating Visual Similarity](../../ICCV2025/image_generation/diffsim_taming_diffusion_models_for_evaluating_visual_similarity.md)
- [\[CVPR 2026\] Disentangling to Re-couple: Resolving the Similarity-Controllability Paradox in Subject-Driven Text-to-Image Generation](../../CVPR2026/image_generation/disentangling_to_re-couple_resolving_the_similarity-controllability_paradox_in_s.md)

</div>

<!-- RELATED:END -->
