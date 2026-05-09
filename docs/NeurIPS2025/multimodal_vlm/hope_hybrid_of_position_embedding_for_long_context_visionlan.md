---
title: >-
  [Paper Note] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Rotary Position Embedding] This work presents the first theoretical analysis of frequency allocation strategies in multimodal RoPE for long-context VLMs. It proposes HoPE, which sets the lowest frequency to zero for temporal modeling to guarantee the semantic preference property, coupled with a dynamic temporal scaling mechanism, achieving gains of 8.35% on long video understanding and 22.23% on retrieval tasks.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Rotary Position Embedding
  - Long Context
  - Frequency Allocation
  - Vision-Language Models
  - Video Understanding
date: 2026-05-08
content_hash: 5473f6693aeb7a24
---

# HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.20444](https://arxiv.org/abs/2505.20444)
**Code**: [GitHub](https://github.com/hrlics/HoPE)
**Area**: Multimodal VLM / Position Encoding
**Keywords**: Rotary Position Embedding, Long Context, Frequency Allocation, Vision-Language Models, Video Understanding

## TL;DR

This work presents the first theoretical analysis of frequency allocation strategies in multimodal RoPE for long-context VLMs. It proposes HoPE, which sets the lowest frequency to zero for temporal modeling to guarantee the semantic preference property, coupled with a dynamic temporal scaling mechanism, achieving gains of 8.35% on long video understanding and 22.23% on retrieval tasks.

## Background & Motivation

### State of the Field

**State of the Field**: VLMs suffer significant performance degradation in long-context scenarios, particularly struggling with object counting and temporal localization in long video tasks.

### Root Cause

**Root Cause**: While RoPE has successfully enabled length generalization in text-only LLMs, directly applying 1D RoPE fails to capture the spatiotemporal structure of video.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing multimodal RoPE extensions exhibit the following limitations:

### Starting Point

**Starting Point**: M-RoPE (Qwen2-VL) assigns the highest frequencies to the temporal dimension—a heuristic design lacking theoretical justification.

### Supplementary Notes

**Supplementary Notes**: VideoRoPE assigns the lowest frequencies to the temporal dimension and shows strong empirical performance, yet remains unreliable over long distances.

### Supplementary Notes

**Supplementary Notes**: Fixed and unidirectional temporal scaling factors cannot adapt to videos of varying speeds and information densities.

### Supplementary Notes

**Supplementary Notes**: Core question: **How do different frequency allocation strategies affect long-range semantic modeling? Can theoretical guarantees be obtained?**

## Method

### Overall Architecture

HoPE comprises two components: (1) a Hybrid Frequency Allocation (HFA) strategy—high frequencies are interleaved to encode spatial information $(x, y)$, while the lowest frequency is set to zero for temporal modeling, guaranteeing an upper bound on the semantic preference property; and (2) a Dynamic Temporal Scaling (DTS) mechanism—scaling factors are randomly sampled during training (including both compression and stretching), enabling flexible adaptation to varying sequence lengths at inference.

### Key Designs

1. **Hybrid Frequency Allocation (HFA)**:
    - **Function**: Partitions the 128-dimensional rotary encoding into 96 dimensions for spatial encoding and 32 dimensions for temporal encoding, with the temporal frequency set to zero (degenerating to NoPE).
    - **Mechanism**: Exploits the identity $\cos(0 \cdot \Delta t) = 1$ to eliminate the adverse effect of temporal distance on attention scores.
    - **Design Motivation**: Theorem 3.1 proves that any nonzero frequency will eventually violate the semantic preference property given sufficiently long contexts; zero frequency provides the strongest possible guarantee.
    - **Key Theorem**: $\sum_{i \in i_t} 2\sigma^2 \cdot 1 \geq \sum_{i \in i_t} 2\sigma^2 \cos(\Delta t \cdot \theta_i)$, establishing that the zero-frequency scheme dominates all other frequency choices.

2. **Dynamic Temporal Scaling (DTS)**:
    - **Function**: During training, scaling factors are randomly sampled from the set $\Gamma = \{0.5, 0.75, 1, 1.25, 1.5\}$.
    - **Mechanism**: Stretching ($\gamma > 1$) preserves spatial detail, while compression ($\gamma < 1$) reinforces semantic preference; bidirectional scaling enables the model to learn multi-scale temporal relations.
    - **Design Motivation**: Real-world videos vary in speed, making fixed scaling insufficient; at inference, the scaling factor can be flexibly selected based on task requirements.

### Loss & Training

- Built upon Qwen2-1.5B/7B backbones; trained on a subset of LLaVA-Video-178k (approximately 30k short videos and 3k medium-to-long videos).
- Training context length: 8k; maximum video frames: 128; sampling rate: 2.
- Learning rate: 1e-5 (2B) / 2e-5 (7B); cosine scheduler; batch size: 128.
- Training required approximately 304 H800 GPU hours.

## Key Experimental Results

### Main Results

Qwen2-7B-Video model, 32k context length:

| Method | MLVU | LongVideoBench | Video-MME |
|--------|------|----------------|-----------|
| Vanilla RoPE | 61.03 | 51.29 | 57.99 |
| M-RoPE | 62.46 | 53.49 | 58.37 |
| VideoRoPE | 62.51 | 53.82 | 59.13 |
| **HoPE** | **63.85** | **55.34** | **59.44** |

Long video retrieval (V-NIAH): HoPE achieves a 22.23% improvement over the best baseline.

### Ablation Study

- **3D Structure**: Introducing 3D structure from 1D RoPE alone yields consistent gains, validating Proposition 3.1.
- **HFA Strategy**: Adding HFA on top of the 3D structure yields an average improvement of 1.69.
- **DTS Mechanism**: DTS provides additional gains over HFA, enhancing robustness to varying video speeds.
- **Inference Scaling Factor**: Retrieval tasks favor smaller factors ($\gamma=0.75$), while understanding tasks favor larger factors ($\gamma=1.5$).

### Key Findings

- At 64k extrapolation, all methods degrade substantially, but HoPE remains the most robust (Video-MME: 27.34 vs. Vanilla 26.13).
- Scaling from 2B to 7B amplifies HoPE's advantage (LongVideoBench 32k gain increases from 0.66 to 4.05).
- Retrieval and understanding tasks exhibit opposite preferences for the scaling factor: retrieval requires preserving semantic preference, while understanding requires retaining spatial detail.

## Highlights & Insights

- This is the first work to provide rigorous theoretical analysis of frequency allocation in multimodal RoPE, rather than relying on purely empirical comparisons.
- The insight that "zero frequency = NoPE for the temporal dimension" is both elegant and profound—the lowest frequency is not low enough; it must be exactly zero to provide guarantees.
- The semantic preference property (Definition 3.1) offers a general analytical framework applicable to broader position encoding designs.
- Bidirectional scaling is a natural approach for learning multi-scale temporal relations, and affords strong flexibility at inference time.

## Limitations & Future Work

- Experiments are limited to the 7B scale; larger models and more training data may further amplify the observed advantages.
- Performance at 64k extrapolation still degrades substantially, and extreme-length generalization remains an open problem.
- The zero-frequency strategy entirely forgoes explicit temporal encoding, which may be detrimental for tasks requiring precise temporal localization.
- The interaction between HoPE and implicit positional learning in causal attention within decoder-only architectures is not discussed.

## Related Work & Insights

- HoPE is complementary to LLM length extension methods such as LongRoPE and YaRN, focusing specifically on multimodal extension.
- The success of NoPE in decoder-only LLMs provides prior support for the zero-frequency strategy.
- The dynamic scaling idea is generalizable to position encoding design for other multimodal inputs, such as audio and point clouds.

## Rating

- ⭐⭐⭐⭐⭐ — Theoretically rigorous, elegantly designed, experimentally comprehensive, and highly insightful for the long-context VLM community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [\[NeurIPS 2025\] Context Informs Pragmatic Interpretation in Vision-Language Models](context_informs_pragmatic_interpretation_in_vision-language_models.md)
- [\[ICCV 2025\] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](../../ICCV2025/multimodal_vlm/matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)
- [\[NeurIPS 2025\] GoalLadder: Incremental Goal Discovery with Vision-Language Models](goalladder_incremental_goal_discovery_with_vision-language_models.md)

<!-- RELATED:END -->
