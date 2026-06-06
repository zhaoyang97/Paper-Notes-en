---
title: >-
  [Paper Note] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models
description: >-
  [ICLR2026][Image Generation][Diffusion Models] This paper proposes S²-Guidance, which constructs a weak model by **randomly dropping transformer block activations** during denoising to perform self-guidance…
tags:
  - "ICLR2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Classifier-Free Guidance"
  - "Subnetwork"
  - "Stochastic Block-Dropping"
  - "Self-Guidance"
  - "Text-to-Image"
  - "Text-to-Video"
date: 2026-05-08
content_hash: df3669f5c10b2fec
---

# Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models

**Conference**: ICLR2026
**arXiv**: [2508.12880](https://arxiv.org/abs/2508.12880)  
**Code**: [Project Page](https://s2guidance.github.io/)  
**Area**: Image Generation
**Keywords**: Diffusion Models, Classifier-Free Guidance, Subnetwork, Stochastic Block-Dropping, Self-Guidance, Text-to-Image, Text-to-Video

## TL;DR

This paper proposes S²-Guidance, which constructs a weak model by **randomly dropping transformer block activations** during denoising to perform self-guidance, correcting the suboptimal predictions of CFG without additional training. The method consistently outperforms CFG and other advanced guidance strategies on text-to-image and text-to-video tasks.

## Background & Motivation

1. **CFG is the cornerstone of conditional generation**: Classifier-Free Guidance enhances generation quality by extrapolating between conditional and unconditional predictions, and has become the standard practice for diffusion models.
2. **CFG has inherent deficiencies**: Empirical analysis shows that CFG-generated results deviate from the true distribution, leading to semantic inconsistency and loss of detail.
3. **Weak model guidance is promising**: Works such as Autoguidance find that guiding with a degraded model improves over CFG, but require training a separate weak model, which is infeasible for large-scale pretrained models.
4. **Manual network modification generalizes poorly**: Methods like SEG simulate weak models by modifying attention regions, but rely on empirical hyperparameter tuning and are designed for specific tasks.
5. **Transformer blocks exhibit significant redundancy**: In mainstream architectures such as DiT, outputs across different blocks are highly similar, suggesting that subnetworks can substitute the full model for functional prediction.
6. **A universal training-free improvement is needed**: Existing methods either require training a weak model or depend on task-specific modifications, lacking a simple and general solution.

## Method

### Step 1: Analyzing the Suboptimality of CFG

The limitations of CFG are verified on a Gaussian mixture toy example — while conditional generation is improved, a mode shift occurs, and in the 2D case samples spread into non-target regions. t-SNE analysis on CIFAR-10 further confirms severe distributional collapse under CFG.

### Step 2: Naive S²-Guidance

The core idea is to use the model's own subnetwork as the weak model:

$$\tilde{D}_\theta^\lambda(x_t|c) = D_\theta(x_t|\phi) + \lambda(D_\theta(x_t|c) - D_\theta(x_t|\phi)) - \frac{\omega}{N}\sum_{i=1}^N(\hat{D}_\theta(x_t|c, \mathbf{m}_i) - D_\theta(x_t|c))$$

- A binary mask $\mathbf{m}$ randomly drops a subset of transformer blocks to construct subnetwork predictions $\hat{D}_\theta$.
- The deviation between the subnetwork prediction and the full model prediction serves as the self-guidance signal.
- $N$ different masks are sampled per step and the guidance signals are averaged.
- $\omega$ controls the self-guidance strength (S²Scale).

### Step 3: Simplification to S²-Guidance

A key finding is that within a reasonable drop range, dropping different blocks consistently steers the model toward the desired distribution. This motivates a simplification to **a single random block-dropping per timestep**:

$$\tilde{D}_\theta^\lambda(x_t|c) = D_\theta(x_t|\phi) + \lambda(D_\theta(x_t|c) - D_\theta(x_t|\phi)) - \omega(\hat{D}_\theta(x_t|c, \mathbf{m}_t) - D_\theta(x_t|c))$$

### Key Design Choices

- **Protecting critical blocks**: Structurally important blocks (e.g., the first block) are excluded; random dropping is applied only among non-critical blocks.
- **Drop ratio of ~10%**: Experiments confirm that dropping approximately 10% of blocks yields optimal performance.
- **Application interval**: The method is most effective when applied within the middle 80% of the noise level range during denoising.
- **Dynamic diversity**: Independently sampling masks at different timesteps is more robust than fixing the dropped block throughout denoising.

## Key Experimental Results

### Table 1: Text-to-Image HPSv2.1 and T2I-CompBench Comparison

| Model | Method | HPSv2.1 Avg↑ | Color↑ | Shape↑ | Texture↑ | Qalign(HPSv2.1)↑ |
|------|------|:---:|:---:|:---:|:---:|:---:|
| SD3 | CFG | 30.48 | 53.61 | 51.20 | 52.45 | 4.66 |
| SD3 | CFG-Zero | 30.78 | 52.70 | 52.84 | 53.37 | 4.66 |
| SD3 | SEG | 30.39 | 58.20 | 57.68 | 57.17 | 4.33 |
| SD3 | **S²-Guidance** | **31.09** | **59.63** | **58.71** | 56.77 | **4.65** |
| SD3.5 | CFG | 30.82 | 51.29 | 47.71 | 47.39 | 4.63 |
| SD3.5 | **S²-Guidance** | **31.56** | 57.57 | 51.23 | 50.13 | **4.70** |

S²-Guidance achieves the best results across all HPSv2.1 dimensions and leads substantially on Color and Shape in T2I-CompBench.

### Table 2: ImageNet 256×256 Class-Conditional Generation

| Method | IS↑ | FID↓ |
|------|:---:|:---:|
| Baseline | 125.13 | 9.41 |
| CFG | 258.09 | 2.15 |
| CFG-Zero | 258.87 | 2.10 |
| **S²-Guidance** | **259.12** | **2.03** |

### Table 3: VBench Text-to-Video Comparison (Wan Model)

| Model | Method | Total↑ | Quality↑ | Semantic↑ |
|------|------|:---:|:---:|:---:|
| Wan-1.3B | CFG | 80.29 | 84.32 | 64.16 |
| Wan-1.3B | CFG-Zero | 80.71 | 84.51 | 65.53 |
| Wan-1.3B | **S²-Guidance** | **80.93** | **84.74** | **65.70** |
| Wan-14B | CFG | 82.65 | 84.88 | 73.76 |
| Wan-14B | **S²-Guidance** | **82.84** | **84.89** | **74.65** |

The method achieves the highest overall score on both the 1.3B and 14B models, validating its generality.

### Computational Overhead

- Runtime: approximately 40% increase over CFG (29.2s → 40.2s).
- Peak memory: unchanged, as the subnetwork and full model run sequentially.
- S²-Guidance with 20 steps achieves a higher HPS Score than CFG with 60 steps, yielding a superior performance–efficiency frontier.

## Highlights & Insights

- **Training-free and plug-and-play**: No additional weak model training is required; the method directly exploits the inherent redundancy of the model's own subnetwork and adapts to any DiT architecture.
- **Clear theoretical intuition**: Starting from a closed-form analysis of Gaussian mixtures and progressively extending to real data, the argumentative chain is complete and rigorous.
- **Minimal and efficient design**: Only one additional forward pass with ~10% blocks dropped is needed per step, with no increase in memory usage.
- **Multi-task coverage**: Consistent improvements are demonstrated across class-conditional image generation, T2I, and T2V tasks, validated on multiple models including SD3, SD3.5, and Wan.
- **Dynamic diversity outperforms fixed strategies**: The time-varying diversity of stochastic dropping naturally avoids the limitation of using a fixed weak model throughout the entire denoising process.

## Limitations & Future Work

- **40% computational overhead**: Although memory usage is unchanged, the additional forward pass per step incurs non-trivial costs in large-scale deployment.
- **Manual tuning of $\omega$**: The optimal S²Scale may vary across models and tasks, and excessively large $\omega$ can cause over-correction.
- **Heuristic block-dropping design**: Excluding critical blocks and determining the drop range still relies on empirical analysis, lacking an automated selection mechanism.
- **Applicability to non-DiT architectures is unverified**: The method is primarily tested on Transformer-based diffusion models; its suitability for architectures such as UNet remains uncertain.
- **Diminishing gains on stronger models**: The improvement on Wan-14B is smaller than on the 1.3B model, suggesting diminishing marginal returns as the model approaches the state of the art.

## Related Work & Insights

| Method | Requires Training? | Generality | Core Mechanism | Comparison with S²-Guidance |
|------|:---:|--------|---------|------------------|
| CFG | × | High | Conditional–unconditional extrapolation | Suffers from mode shift and distributional collapse |
| Autoguidance | ✓ | Low | Training a degraded weak model | Requires additional training; weak model selection is difficult |
| SEG | × | Medium | Modifying attention regions | Task-specific, hyperparameter-sensitive, reduced aesthetic scores |
| CFG++ | × | High | Manifold constraint | Some metrics fall below vanilla CFG |
| CFG-Zero | × | High | Zero-initialization correction | Competitive but does not leverage weak model guidance direction |
| **S²-Guidance** | **×** | **High** | **Stochastic block-dropping self-guidance** | **Universal, training-free, best overall performance** |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The insight of using stochastic block-dropping as a weak model is both novel and natural.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage from toy examples to ImageNet, T2I, and T2V, with thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — The argument progresses logically from toy to real settings, with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — A plug-and-play universal enhancement for diffusion models with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](../../AAAI2026/image_generation/self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[ICML 2026\] Factored Classifier-Free Guidance](../../ICML2026/image_generation/factored_classifier-free_guidance.md)
- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](../../CVPR2026/image_generation/cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](../../AAAI2026/image_generation/melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)

</div>

<!-- RELATED:END -->
