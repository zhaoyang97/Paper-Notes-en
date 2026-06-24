---
title: >-
  [Paper Note] Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders
description: >-
  [CVPR 2025][LLM (Other)][3D shape reconstruction] Proposes Dora-VAE, which focuses on sharp geometric edge regions via Sharp Edge Sampling (SES) and handles uniform and salient sampled points separately using Dual Cross-Attention. It achieves superior 3D shape reconstruction quality with only 1,280 latent codes (8× fewer than XCube-VAE's 10,000+), while establishing a new evaluation benchmark, Dora-Bench.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "3D shape reconstruction"
  - "VAE"
  - "sharp edge sampling"
  - "cross-attention"
  - "benchmark"
date: 2026-05-08
content_hash: 92fb79fcc7e7e77b
---

# Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders

**Conference**: CVPR 2025  
**arXiv**: [2412.17808](https://arxiv.org/abs/2412.17808)  
**Code**: [https://aruichen.github.io/Dora](https://aruichen.github.io/Dora)  
**Area**: LLM Evaluation  
**Keywords**: 3D shape reconstruction, VAE, sharp edge sampling, cross-attention, benchmark

## TL;DR
Proposes Dora-VAE, which focuses on sharp geometric edge regions via Sharp Edge Sampling (SES) and handles uniform and salient sampled points separately using Dual Cross-Attention. It achieves superior 3D shape reconstruction quality with only 1,280 latent codes (8× fewer than XCube-VAE's 10,000+), while establishing a new evaluation benchmark, Dora-Bench.

## Background & Motivation

**Background**: Variational Autoencoders (VAEs) for 3D shapes are core components in 3D generation, used to encode 3D shapes into a compact latent space. Existing methods, such as XCube-VAE, require a large number of latent codes (10,000+) to preserve geometric details.

**Limitations of Prior Work**: (1) Uniform point sampling undersamples sharp edges and geometrically complex regions, leading to poor reconstruction quality in these critical areas. (2) A massive number of latent codes increases the computational burden on downstream generative models. (3) There is a lack of evaluation benchmarks stratified by geometric complexity.

**Key Challenge**: A conflict exists between compact latent representations (few codes) and fine-grained geometric reconstruction (sharp edge preservation). Uniform sampling fails to allocate computational resources effectively to geometrically complex regions.

**Goal**: How to significantly reduce the number of latent codes while maintaining or even improving the reconstruction quality of sharp edge regions?

**Key Insight**: Propose a salient edge-guided sampling strategy to densely sample geometrically complex regions, coupled with a specialized dual cross-attention mechanism to handle uniform and edge-sampled points separately.

**Core Idea**: Detect sharp edges using dihedral angles for dense sampling, and employ dual cross-attention to separately encode uniform and salient region information, achieving fine-grained 3D reconstruction with a highly compact latent representation.

## Method

### Overall Architecture
The input is a 3D mesh. Sharp Edge Sampling is used to obtain a uniform point set $P_u$ and a salient point set $P_a$. The encoder utilizes Dual Cross-Attention to separately perform cross-attention on the two types of point sets, encoding them into the latent space (1,280 codes). The decoder then reconstructs the 3D shape from the latent codes.

### Key Designs

1. **Sharp Edge Sampling (SES)**

    - **Function**: Densely samples sharp edge regions to capture geometric details.
    - **Mechanism**: Calculates the dihedral angle between the face patches on both sides of each edge in the mesh, using a threshold $\tau=30°$ to detect salient edges. Salient points $P_a$ are sampled along these edges and merged with uniform samples $P_u$: $P_d = P_u \cup P_a$.
    - Total target sample count $N_{desired}=16384$.
    - **Design Motivation**: Sharp edges (e.g., object corners, detailed contours) are the most easily lost regions in 3D reconstruction, and uniform sampling severely undersamples them.

2. **Dual Cross-Attention**

    - **Function**: Separately performs cross-attention on uniform and salient sampled points.
    - **Mechanism**: $C = \text{CrossAttn}(P_s, P_u) + \text{CrossAttn}(P_s, P_a)$, where $P_s$ represents the latent tokens.
    - **Design Motivation**: Uniform points reflect the overall shape, while salient points reflect local details. Their attention patterns are distinctly different. Processing them separately allows the network to learn global and detailed encoding processes independently.

3. **Dora-Bench 评测基准**

    - **Function**: Stratifies 3D shapes into 4 levels (L1-L4) based on geometric complexity.
    - **Classification Basis**: Number of salient edges—from L1 (fewest edges) to L4 (most edges).
    - **Proposes Sharp Normal Error (SNE)**: Calculates the MSE of normal errors specifically in salient regions.
    - **Design Motivation**: Existing evaluation metrics (CD, F-score) are sensitive to the overall shape but insensitive to sharp edges.

### Loss & Training
- 400,000 3D meshes from Objaverse
- 32 A100 GPUs, trained for 2 days
- Batch size 2048, learning rate 5e-5

## Key Experimental Results

### Main Results

| Metric | Dora (1,280 codes) | XCube† (10,000+ codes) | Gain |
|------|-------------------|----------------------|------|
| F-score(L1)×100 | 99.988 | 99.393 | +0.595 |
| F-score(L4)×100 | 99.170 | 99.079 | +0.091 |
| CD(L1)×10000 | 2.097 | 4.015 | **-47.8%** |
| CD(L4)×10000 | 5.265 | 7.627 | **-31.0%** |
| SNE(L4)×100 | 1.579 | 1.639 | -3.7% |

Achieves comprehensively superior reconstruction quality with an 8× smaller latent space.

### Ablation Study

| Scenario | Results |
|------|------|
| vs Craftsman-VAE | Geometric detail preservation is significantly better |
| vs Commercial Tripo v2.0 | Comparable quality (under limited computational resources) |
| Single Image to 3D | Clear advantage in keeping fine edges |

### Key Findings
- An overall improvement of 47.8% in CD metrics demonstrates that the combination of SES and Dual Cross-Attention is highly effective.
- The improvement is more significant for more complex geometries (L4), validating the value of SES in complex regions.
- 1,280 latent codes are sufficient to represent fine-grained 3D shapes, significantly reducing the computing burden on downstream generative models.

## Highlights & Insights
- **Importance of Sampling Strategy is Underestimated**: Simply changing the sampling method can dramatically improve reconstruction quality, showing that innovation at the data preprocessing level is equally critical.
- **8× Compression + Better Quality**: This is highly significant in 3D generation—a smaller latent space implies faster generation speeds.
- **Stratified Evaluation of Dora-Bench**: Stratifying by geometric complexity is highly rational, as different methods may perform entirely differently on varying complexity levels.
- The Dual Cross-Attention concept can be extended to other tasks requiring distinction between important and ordinary regions.

## Limitations & Future Work
- The dihedral angle threshold $\tau=30°$ is fixed; an adaptive threshold might be more effective.
- SES depends on the quality of the input mesh; noisy meshes may lead to incorrect salient edge detection.
- Only trained on Objaverse; the ability to generalize to other 3D datasets has not yet been validated.

## Related Work & Insights
- **vs XCube-VAE**: Uses more latent codes but yields worse reconstruction quality, showing that the quantity of latents is not the key, and sampling/encoding strategies are more important.
- **vs Craftsman-VAE**: Shows a distinct advantage in detail preservation.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of SES and Dual Cross-Attention is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Introduces a new benchmark and provides extensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clarity in problem definition.
- Value: ⭐⭐⭐⭐ Directly advances the field of 3D generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SphericalDreamer: Generating Navigable Immersive 3D Worlds with Panorama Fusion](../../ICML2026/llm_nlp/sphericaldreamer_generating_navigable_immersive_3d_worlds_with_panorama_fusion.md)
- [\[ACL 2025\] A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive](../../ACL2025/llm_nlp/theory_of_llm_sampling.md)
- [\[ICLR 2026\] Eliciting Numerical Predictive Distributions of LLMs Without Auto-Regression](../../ICLR2026/llm_nlp/eliciting_numerical_predictive_distributions_of_llms_without_auto-regression.md)
- [\[NeurIPS 2025\] Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models](../../NeurIPS2025/llm_nlp/breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)
- [\[ACL 2025\] FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring](../../ACL2025/llm_nlp/feat_a_preference_feedback_dataset_through_a_cost-effective_auto-generation_and_.md)

</div>

<!-- RELATED:END -->
