---
title: >-
  [Paper Note] From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging
description: >-
  [NeurIPS 2025][Image Generation][Face aging] This paper proposes Cradle2Cane, a two-pass face aging framework: the first pass achieves precise age control via Adaptive Noise Injection (AdaNI), and the second pass reinforces identity consistency through dual identity embeddings (IDEmb) comprising SVR-ArcFace and Rotate-CLIP. The framework achieves an optimal balance between age accuracy and identity preservation across the full lifespan (0–80 years).
tags:
  - NeurIPS 2025
  - Image Generation
  - Face aging
  - diffusion models
  - identity preservation
  - adaptive noise injection
  - SDXL-Turbo
  - age-identity trade-off
date: 2026-05-08
content_hash: b13c8419a6b03bcd
---

# From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging

**Conference**: NeurIPS 2025
**arXiv**: [2506.20977](https://arxiv.org/abs/2506.20977)
**Code**: [https://github.com/byliutao/Cradle2Cane](https://github.com/byliutao/Cradle2Cane)
**Area**: Image Generation
**Keywords**: Face aging, diffusion models, identity preservation, adaptive noise injection, SDXL-Turbo, age-identity trade-off

## TL;DR

This paper proposes Cradle2Cane, a two-pass face aging framework: the first pass achieves precise age control via Adaptive Noise Injection (AdaNI), and the second pass reinforces identity consistency through dual identity embeddings (IDEmb) comprising SVR-ArcFace and Rotate-CLIP. The framework achieves an optimal balance between age accuracy and identity preservation across the full lifespan (0–80 years).

## Background & Motivation

Face aging is an important task in computer vision with broad applications in entertainment, healthcare, and security. Its core objective is to generate visually realistic age-progression results while preserving the subject's identity.

**The Core Bottleneck of Existing Methods — Age-ID Trade-off:**

- GAN-based methods (SAM, CUSP, Lifespan, HRFAE) and diffusion-based methods (FADING, IPFE) universally suffer from a fundamental tension: a seesaw effect between age transformation accuracy and identity preservation.
- For large age gaps (e.g., from 20 to 70 years), methods either fail to produce convincing aging effects or suffer severe identity loss.
- Existing methods apply a uniform transformation strategy regardless of age gap, ignoring a natural principle: **small age differences require only subtle appearance adjustments, while large age differences demand more pronounced structural and texture changes**.
- In in-the-wild scenarios with extreme head poses or occlusions, existing methods frequently fail entirely.

**Core Insight of This Paper:** In few-step diffusion models (e.g., SDXL-Turbo), the noise level injected during the forward diffusion process directly controls editing intensity — high noise enables stronger age transformation but harms identity, while low noise preserves identity but limits aging effects. This motivates decoupling age accuracy and identity preservation into two independent passes.

## Method

### Overall Architecture

Cradle2Cane builds a two-pass pipeline on top of SDXL-Turbo (4-step generation):

1. **Pass 1**: Adaptive Noise Injection (AdaNI) — focused on age accuracy.
2. **Pass 2**: Identity-Aware Embedding (IDEmb) — focused on identity preservation.

The two passes are jointly trained end-to-end. The output of the first pass is re-noised and used as input to the second pass.

### Key Design 1: Adaptive Noise Injection (AdaNI)

Core idea: the larger the age gap, the more noise must be injected to allow stronger editing.

Specifics:
- The age prompt (containing age and gender descriptions) is encoded into text embeddings via a CLIP text encoder, guiding generation through cross-attention.
- Transformations are categorized into three tiers based on the age gap $|\Delta \text{age}|$, with thresholds at 5 and 20:
    - $|\Delta \text{age}| \leq 5$: low noise $z_1$ injected, biased toward identity preservation.
    - $5 < |\Delta \text{age}| \leq 20$: medium noise $z_2$ injected, balancing both objectives.
    - $|\Delta \text{age}| > 20$: high noise $z_3$ injected, biased toward strong aging effects.
- The thresholds 5 and 20 are derived from quantitative analysis (age accuracy degrades significantly beyond these values).
- The decoded intermediate result $\hat{x}_b$ exhibits high age accuracy but weaker identity preservation.

### Key Design 2: SVR-ArcFace Embedding

**Goal**: Remove age-related components from ArcFace face recognition embeddings to extract clean identity features.

**Problem**: ArcFace features inherently entangle age and identity — the same person at different ages yields substantially different embeddings.

**Solution — Singular Value Reweighting (SVR)**:

1. Extract ArcFace embeddings from the source image $x_a$ and multiple aged images produced at different noise levels in Pass 1.
2. Concatenate them into a matrix $U \in \mathbb{R}^{D \times (n+1)}$.
3. Apply SVD decomposition: $U = \mathbf{U} \Sigma \mathbf{V}^T$.
4. Nonlinearly reweight the singular values: $\hat{\sigma}_i = \beta e^{\alpha \sigma_i} \cdot \sigma_i$.
5. Take the first column of the reconstructed matrix as the refined identity embedding $\hat{u}_a$.

**Intuition**: Across multiple age variants of the same person, the dominant singular values encode shared identity information, while minor singular values encode age-related variation. Exponential reweighting amplifies the dominant components and suppresses the minor ones, achieving identity-age disentanglement.

### Key Design 3: Rotate-CLIP Embedding

**Goal**: Smoothly shift the age semantics of the source image toward the target age in CLIP space while preserving other identity information.

**Method**:

1. Extract the CLIP image embedding $i_a$ of the source image and text embeddings $t_a, t_b$ for the source and target ages.
2. Compute the age offset vector using spherical linear interpolation (slerp) instead of simple vector subtraction: $\Delta' = \text{slerp}(t_b, t_a, \lambda)$.
3. Obtain the Rotate-CLIP embedding: $\hat{i}_a = i_a + \Delta'$.

**Why not simple subtraction**: CLIP's representation of age is relatively coarse-grained; direct subtraction may introduce semantic inconsistencies, whereas slerp provides a smoother semantic transition.

Both embeddings are projected through separate MLPs, concatenated, and injected into the cross-attention modules of SDXL-Turbo.

### Loss & Training

The total loss consists of three terms: $\mathcal{L}_{total} = \mathcal{L}_{id} + \mathcal{L}_{age} + \mathcal{L}_{per}$

| Loss | Components | Purpose |
|------|------------|---------|
| Identity loss $\mathcal{L}_{id}$ | MS-SSIM (structural similarity) + ArcFace cosine distance | Preserve identity consistency between source and generated images |
| Age loss $\mathcal{L}_{age}$ | MiVOLO feature cosine distance + predicted age L2 error | Ensure age transformation accuracy |
| Quality loss $\mathcal{L}_{per}$ | LPIPS perceptual distance + GAN adversarial loss | Improve perceptual quality and realism |

During training, the ArcFace and CLIP encoders are frozen; only the MLP and UNet-LoRA modules are optimized.

## Key Experimental Results

### Main Results (Face++ Evaluation + Qwen-VL Evaluation, CelebA-HQ)

| Method | Type | Age Diff.↓ | ID Sim.↑ | Img Quality↑ | HCS↑ | Inference (s) | Training Data |
|--------|------|-----------|----------|-------------|------|--------------|--------------|
| Lifespan | GAN | ±22.07 | 79.80 | 66.68 | 57.40 | 0.95 | 70K |
| HRFAE | GAN | ±15.12 | 94.32 | 62.28 | 74.95 | 0.17 | 300K |
| SAM | GAN | ±8.42 | 81.96 | 68.38 | 80.42 | 0.39 | 70K |
| CUSP | GAN | ±9.59 | 85.92 | 64.98 | 80.67 | 0.24 | 30K |
| FADING | Diffusion | ±14.47 | 86.70 | 64.65 | 73.52 | 61.26 | - |
| IPFE | Diffusion | ±11.95 | 75.14 | 63.55 | 72.54 | 8.84 | - |
| **Cradle2Cane** | **Diffusion** | **±7.47** | 81.34 | **72.69** | **81.33** | **0.56** | **10K** |

### Ablation Study (Qwen-VL Evaluation)

| AdaNI | SVR-ArcFace | Rotate-CLIP | Age Diff.↓ | ID Sim.↑ | HCS↑ |
|-------|-------------|-------------|-----------|----------|------|
| ✗ | ✗ | ✗ | ±8.87 | 68.92 | 73.10 |
| ✓ | ✗ | ✗ | ±3.94 | 59.70 | 71.83 |
| ✗ | ✓ | ✗ | ±9.48 | 70.17 | 73.11 |
| ✓ | ✓ | ✗ | ±6.75 | 63.38 | 71.92 |
| ✓ | ✓ | ✓ | **±4.62** | **70.29** | **78.33** |

### Key Findings

1. **Best age accuracy**: Cradle2Cane achieves the lowest Age Diff. under both Face++ and Qwen-VL evaluation protocols (±7.47 / ±4.62).
2. **Extremely fast inference**: 0.56s, more than 100× faster than FADING (61.26s), comparable to GAN-based methods.
3. **High training data efficiency**: Only 10K training samples are required, far fewer than HRFAE (300K) and Lifespan (70K).
4. **Best composite metric HCS**: HCS reaches 81.33 under Face++ and 78.33 under Qwen-VL evaluation.
5. **Complementary components**: AdaNI substantially reduces age error (8.87→3.94); SVR-ArcFace improves identity consistency (59.70→63.38); Rotate-CLIP further raises HCS from 71.92 to 78.33.
6. **Robustness in the wild**: HCS of 75.94 on in-the-wild data, outperforming CUSP (70.94) and FADING (75.06).
7. **User study**: Among 50 volunteers, Cradle2Cane results were clearly preferred over SAM and CUSP in one-on-one comparisons.

## Highlights & Insights

1. **Precise problem formulation**: The core tension in face aging is clearly defined as the Age-ID Trade-off, supported by rigorous empirical evidence (trade-off curves across 60 age offset values × 100 faces).
2. **Elegant decoupling strategy**: Assigning age accuracy and identity preservation to two independent passes — each responsible for a single objective — is more tractable to optimize than a unified end-to-end approach.
3. **Intuitive adaptive noise injection**: Larger age gaps require more editing freedom (more noise); smaller age gaps require only fine-grained adjustment (less noise) — this aligns naturally with human understanding of the aging process.
4. **Novel SVR disentanglement**: Using SVD on ArcFace embeddings from multiple age variants of the same person, with singular value reweighting to amplify identity commonality and suppress age-specific variation, is a novel contribution.
5. **Training efficiency**: Only 10K data with LoRA fine-tuning achieves state-of-the-art performance, with inference speed on par with GAN-based methods.

## Limitations & Future Work

1. **Identity similarity is not optimal**: Although the composite HCS is the best overall, the standalone ID Sim. metric (81.34) falls below HRFAE (94.32) and CUSP (85.92), indicating room for improvement in identity preservation.
2. **Three-tier noise partition is somewhat coarse**: Fixed thresholds of 5 and 20 may not be globally optimal — whether a continuous adaptive noise schedule would be superior remains an open question.
3. **Limited evaluation datasets**: Evaluation is conducted primarily on CelebA-HQ; comparisons on commonly used face aging benchmarks such as MORPH and CACD are absent.
4. **Dependence on age estimator**: Training relies on the accuracy of the MiVOLO age estimator; biases inherent to the estimator propagate into the generated results.
5. **Lack of cross-ethnicity analysis**: Aging patterns differ across racial and cultural groups; the paper does not discuss model fairness across diverse populations.

## Related Work & Insights

- **Vs. FADING**: FADING relies on NTI inversion with a standard LDM, requiring 61s per inference; this work uses SDXL-Turbo with only 0.56s inference time, and achieves a better Age-ID balance through two-pass decoupling.
- **Vs. SAM**: SAM performs age transformation in the StyleGAN2 latent space, dependent on GAN latent structure; this work exploits the noise-control flexibility of diffusion models to achieve adaptive editing intensity.
- **Broader implications**: The two-pass decoupling strategy can be generalized to other image editing tasks with multi-objective conflicts (e.g., style-content trade-off, editing strength-fidelity trade-off); the SVR disentanglement technique is applicable to any scenario requiring separation of shared versus attribute-specific components from embeddings.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combined design of two-pass decoupling, AdaNI, and SVR-ArcFace is genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual evaluation protocols, ablation studies, and user studies are included, though the number of evaluation datasets is limited.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated; the trade-off analysis figures are well designed.
- Value: ⭐⭐⭐⭐ — Addresses the core tension in face aging with strong practical utility (fast inference + minimal training data).
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](../../CVPR2026/image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](../../CVPR2026/image_generation/preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[NeurIPS 2025\] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting](one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)
- [\[NeurIPS 2025\] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)

<!-- RELATED:END -->
