---
title: >-
  [Paper Note] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens
description: >-
  [ICCV 2025][Model Compression][video LLM] This paper proposes B-VLLM, a framework that dynamically balances spatio-temporal tokens within the VLLM context window budget through three modules: text-conditioned adaptive frame selection, temporal frame token merging, and spatial token sampling. It addresses the dilemma between uniform sampling (which neglects temporal dynamics) and per-frame token reduction (which loses spatial detail), achieving a 10% improvement on MVBench.
tags:
  - ICCV 2025
  - Model Compression
  - video LLM
  - spatio-temporal balance
  - frame selection
  - token merging
  - adaptive token sampling
date: 2026-05-08
content_hash: 54a0cea0b080fc0a
---

# B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens

**Conference**: ICCV 2025
**arXiv**: N/A
**Code**: [GitHub](https://github.com/zhuqiangLu/B-VLLM)
**Area**: Model Compression
**Keywords**: video LLM, spatio-temporal balance, frame selection, token merging, adaptive token sampling

## TL;DR

This paper proposes B-VLLM, a framework that dynamically balances spatio-temporal tokens within the VLLM context window budget through three modules: text-conditioned adaptive frame selection, temporal frame token merging, and spatial token sampling. It addresses the dilemma between uniform sampling (which neglects temporal dynamics) and per-frame token reduction (which loses spatial detail), achieving a 10% improvement on MVBench.

## Background & Motivation

VLLMs encode visual content as token sequences for joint processing with text, but video visual tokens—especially from long videos—grow rapidly, exceeding context window limits and incurring substantial computational cost. Existing methods either uniformly sample a fixed number of frames (ignoring temporal dynamics and potentially missing key frames) or reduce per-frame token counts (losing spatial detail). The core issue is **spatio-temporal token imbalance**: reducing per-frame tokens causes temporal cues to dominate, while uniform frame sampling obscures spatial cues.

## Method

### Overall Architecture

B-VLLM first encodes all frames into visual tokens via a ViT, yielding both [CLS] and patch tokens. The [CLS] token provides coarse-grained semantics for frame selection, while patch tokens provide fine-grained spatial information. The pipeline proceeds as: (1) text-conditioned frame selection—selecting the L* most relevant frames using [CLS] tokens and the text prompt; (2) temporal frame token merging—removing redundant frames; (3) spatial token sampling with optional spatial token merging—ensuring the final token count stays within budget θ.

### Key Designs

1. **Text-Conditioned Adaptive Frame Selection**: A Q-Former jointly encodes the [CLS] tokens of all frames alongside the text context T, generating L* queries Q, from which L* most relevant frames are selected from L candidates via Gumbel-Softmax. Using [CLS] tokens rather than full patch tokens enables efficient frame selection. The selection is text-conditioned, allowing different tasks to focus on different key frames.

2. **Temporal Frame Token Merging**: After frame selection, further temporal redundancy is removed. Based on inter-frame [CLS] token similarity, the visual tokens of highly similar adjacent frames are merged by averaging, reducing temporal redundancy.

3. **Spatial Token Sampling and Merging**: Intra-frame spatial token sampling selects the Z most relevant visual tokens per frame conditioned on the text. An optional iterative spatial token merging strategy—similar to ToMe but applied iteratively—merges the most similar token pairs at each step until the target count is reached, enabling fine-grained control over the final token budget.

### Loss & Training

Standard VLLM training (visual instruction tuning) is adopted. The frame selection module uses Gumbel-Softmax for differentiable discrete selection. The lightweight Q-Former design minimizes overhead.

## Key Experimental Results

### Main Results

| Method | MVBench | Long Video Understanding | Spatial Detail Retention |
|--------|---------|--------------------------|--------------------------|
| Uniform frame sampling | Baseline | Moderate | Good |
| Reduced per-frame tokens | Poor | Good | Poor |
| **B-VLLM** | **+10%** | **Good** | **Good** |

B-VLLM achieves a 10% performance gain on MVBench and performs strongly on both temporal and spatial tasks.

### Ablation Study

- Frame selection vs. uniform sampling: frame selection significantly outperforms on temporally sensitive tasks.
- Spatial token sampling: preserves spatial detail effectively.
- Iterative spatial token merging: provides fine-grained control over token count.
- Different VLLM backbones: the framework generalizes across architectures.

### Key Findings

- [CLS] tokens are sufficient for efficient frame selection without requiring full patch tokens.
- Text-conditioned selection enables different tasks to attend to different frames and spatial regions.
- Balancing spatio-temporal tokens is critical for video understanding.
- Iterative merging offers finer-grained control than one-shot merging.

## Highlights & Insights

- The "spatio-temporal token balance" problem is clearly formulated and directly addresses the limitations of prior methods.
- Using [CLS] tokens for frame selection is computationally efficient.
- Text-conditioned adaptive selection allows the model to "attend to what is needed."
- The framework is generalizable to different VLLM architectures.

## Limitations & Future Work

- The Q-Former introduces additional model parameters and training complexity.
- Gumbel-Softmax may be unstable for very long videos containing hundreds to thousands of frames.
- The interpretability of frame selection is limited—it is difficult to verify whether the selected frames are optimal.
- Evaluation is restricted to standard video QA benchmarks; latency assessment for real-time applications is insufficient.

## Related Work & Insights

- LLaMA-VID compresses each frame to as few as 2 tokens, sacrificing spatial information to support long videos.
- The token merging idea from ToMe is extended to an iterative variant.
- The Q-Former is adapted from BLIP-2 for lightweight frame selection.

## Rating

- Novelty: ⭐⭐⭐⭐ — The spatio-temporal token balance formulation and text-conditioned selection are novel.
- Technical Depth: ⭐⭐⭐ — Individual components are relatively standard; innovation lies in their combination and control.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark, multi-architecture evaluation with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ — The problem illustration (Figure 1) is intuitive and compelling.
- Value: ⭐⭐⭐⭐ — A 10% performance gain with a general-purpose framework.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [\[ICCV 2025\] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)

<!-- RELATED:END -->
