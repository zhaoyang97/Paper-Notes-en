---
title: >-
  [Paper Note] SERUM: Simple, Efficient, Robust, and Unifying Marking for Diffusion-based Image Generation
description: >-
  [ICLR 2026][Image Generation][Diffusion model watermarking] SERUM is a watermarking method that injects unique watermark noise into the initial noise of diffusion models and trains a lightweight detector to identify watermarks directly from generated images — without costly DDIM inversion — achieving state-of-the-art detection rates under diverse attacks with extremely fast injection and detection, while supporting multi-user scenarios.
tags:
  - ICLR 2026
  - Image Generation
  - Diffusion model watermarking
  - lightweight detector
  - noise injection
  - robustness
  - multi-user
date: 2026-05-08
content_hash: e1e17b4d49f3158e
---

# SERUM: Simple, Efficient, Robust, and Unifying Marking for Diffusion-based Image Generation

**Conference**: ICLR 2026
**arXiv**: [2603.13396](https://arxiv.org/abs/2603.13396)
**Code**: [GitHub](https://github.com/Hubizon/SERUM)
**Area**: Image Generation
**Keywords**: Diffusion model watermarking, lightweight detector, noise injection, robustness, multi-user

## TL;DR
SERUM is a watermarking method that injects unique watermark noise into the initial noise of diffusion models and trains a lightweight detector to identify watermarks directly from generated images — without costly DDIM inversion — achieving state-of-the-art detection rates under diverse attacks with extremely fast injection and detection, while supporting multi-user scenarios.

## Background & Motivation

### State of the Field

**Background**: Diffusion models generate highly realistic images, necessitating watermarking to distinguish generated from real content. Existing approaches fall into two categories: fine-tuning-based methods (e.g., Stable Signature, which fine-tunes the decoder) and training-free methods (e.g., Tree-Ring/GaussMarker, which embed watermarks in the initial noise).

**Limitations of Prior Work**: (1) Stable Signature requires extensive training and lacks robustness against advanced attacks; (2) training-free methods such as Tree-Ring/GaussMarker are robust but rely on expensive DDIM inversion ($O(T)$ steps) for detection; (3) the two desiderata appear mutually exclusive — methods are either fast but weak, or strong but slow.

**Key Challenge**: Watermark detection requires DDIM inversion to recover the initial noise, which is computationally prohibitive and unsuitable for large-scale deployment.

**Key Insight**: Bypass DDIM inversion by training a lightweight external detector that directly identifies the signature of watermark noise from generated images, thereby combining the robustness of noise injection with near-instant detection.

### Mechanism

**Goal**: ### Overall Architecture
Watermark injection: $\eta' = \sqrt{1-\alpha}\eta + \sqrt{\alpha}A'$ (weighted mixture of random noise and watermark noise) → standard diffusion generation.

## Method

### Overall Architecture
Watermark injection: $\eta' = \sqrt{1-\alpha}\eta + \sqrt{\alpha}A'$ (weighted mixture of random noise and watermark noise) → standard diffusion generation. Watermark detection: a lightweight CNN trained in the LDM latent space to classify watermarked vs. clean images.

### Key Designs

1. **Watermark Injection**:

    - Function: Injects normalized watermark noise into the initial diffusion noise.
    - Mechanism: $A' = (A - \text{mean}(A))/\text{std}(A)$ normalization guarantees low KL divergence → high image quality.
    - Design Motivation: After normalization, $\eta'$ remains close to a standard normal distribution — provably achieving lower KL divergence than GaussMarker.

2. **Lightweight Detector**:

    - Function: Trains a binary classification CNN in the LDM latent space.
    - Mechanism: Training set consists of watermarked and clean latents, sampled via augmented prioritized experience replay to focus on hard perturbations.
    - Loss: $\mathcal{L} = \mathcal{L}_w + \mathcal{L}_n$, each comprising clean, augmented, and pre-computed terms.
    - Design Motivation: Operating in the latent space reduces input dimensionality, accelerating both training and inference.

3. **Multi-User Support**:

    - Function: Assigns each user a unique subset of noise patterns.
    - Mechanism: User $i$ employs a combination of $k$ noise patterns; the user detection score is $D_i(x) = \prod_{p \in S_i} d_p(x)$.
    - Training scale: $O(n^{1/k})$ rather than $O(n)$.

## Key Experimental Results

### Detection Robustness (SDv1.4/2.0/2.1)


### Main Results

| Method | TPR@1%FPR (Standard Attacks) | TPR@1%FPR (Watermark Removal) | Injection Speed | Detection Speed |
|--------|-------------------------------|-------------------------------|-----------------|-----------------|
| Stable Signature | Moderate | Poor | Fast | Fast |
| Tree-Ring | Good | Good | Fast | **Very Slow** (DDIM) |
| GaussMarker | Good | Good | Fast | **Very Slow** (DDIM) |
| **SERUM** | **Best** | **Best** | **Very Fast** | **Very Fast** |

### Image Quality


### Ablation Study

| Method | FID↓ | CLIP Score↑ | Notes |
|--------|------|-------------|-------|
| No Watermark | Baseline | Baseline | Reference |
| **SERUM** | **Near Baseline** | **Near Baseline** | Negligible quality degradation |

### Key Findings
- SERUM achieves the highest TPR in nearly all cases across 8 perturbation types and 7 watermark removal attacks.
- Detection requires no DDIM inversion, making it orders of magnitude faster than Tree-Ring/GaussMarker.
- The watermark exhibits "radioactivity" — outputs remain detectable even when the model is fine-tuned on watermarked images.
- Cross-user watermark interference is negligible in multi-user settings.

## Highlights & Insights
- **Elegant Unification of Two Paradigms**: Noise injection (robustness of training-free methods) + external detector (speed of fine-tuning-based methods) = optimal combination. The idea is intuitively straightforward yet had not been previously explored.
- **KL Divergence Guarantee**: Normalized watermark noise achieves theoretically lower KL divergence than GaussMarker, providing a mathematical foundation for superior image quality.
- **Prioritized Experience Replay Training**: Borrowing PER from reinforcement learning to sample "hard" augmentations enables the detector to automatically focus on its weaknesses, eliminating the need for manual augmentation strategy selection.

## Limitations & Future Work
- Detection requires access to the LDM encoder; purely pixel-level detection remains unexplored.
- The choice of $\alpha$ requires balancing detection rate against image diversity.
- In multi-user settings, the combinatorial structure limits the maximum number of supported users.
- Evaluation is limited to the Stable Diffusion family; generalization to other diffusion models (e.g., DALL-E/Imagen) is unknown.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of noise injection and an external detector is simple yet effective and previously unattempted.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 perturbation types + 7 attacks + 3 SD versions + multi-user + radioactivity analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear methodology with solid theoretical guarantees.
- Value: ⭐⭐⭐⭐⭐ — Directly applicable to real-world deployment for AI-generated content detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](../../ICCV2025/image_generation/lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICLR 2026\] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)
- [\[ICLR 2026\] Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)
- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)
- [\[ICLR 2026\] RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation](refany3d_3d_asset-referenced_diffusion_models_for_image_generation.md)

</div>

<!-- RELATED:END -->
