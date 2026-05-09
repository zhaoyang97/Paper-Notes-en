---
title: >-
  [Paper Note] Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs
description: >-
  [ACL 2026][Video Understanding][Diffusion Language Model] This paper proposes Saber, a training-free sampling algorithm for Diffusion Language Models (DLMs) that improves average Pass@1 by 1.9% on code generation while achieving 251.4% inference acceleration, via two strategies: adaptive acceleration (dynamically adjusting parallel decoding based on established context) and backtracking-enhanced remasking (reverting tokens invalidated by new context).
tags:
  - ACL 2026
  - Video Understanding
  - Diffusion Language Model
  - Adaptive Sampling
  - Backtracking Remasking
  - Code Generation Acceleration
  - Speed-Quality Tradeoff
date: 2026-05-08
content_hash: 5e9ebdcc21ad82c0
---
# Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs

**Conference**: ACL 2026
**arXiv**: [2510.18165](https://arxiv.org/abs/2510.18165)
**Code**: [GitHub](https://github.com/zhaoyMa/Saber)
**Area**: Video Understanding
**Keywords**: diffusion language models, adaptive sampling, backtracking remasking, code generation acceleration, speed-quality trade-off

## TL;DR

This paper proposes Saber, a training-free sampling algorithm for diffusion language models (DLMs) that achieves an average Pass@1 improvement of 1.9% on code generation while delivering 251.4% inference speedup. This is accomplished through two strategies: adaptive acceleration (dynamically adjusting the amount of parallel decoding based on established context) and backtracking-enhanced remasking (revoking tokens falsified by newly established context).

## Background & Motivation

**State of the Field**: DLMs (e.g., LLaDA, Dream) achieve parallel generation through iterative demasking and represent a strong alternative to autoregressive models. However, on tasks with strong structural constraints such as code generation, reducing sampling steps causes a catastrophic drop in Pass@1 (by more than 60% in some cases).

**Limitations of Prior Work**: (1) Static acceleration strategies (fixed token counts or confidence thresholds) are too conservative during easy phases and too aggressive during difficult ones; (2) DLM decoding is irreversible—once a token is unmasked it cannot be revoked, so early errors become permanently locked in and propagate.

**Root Cause**: The speed advantage of parallel generation conflicts with quality collapse caused by error propagation—both non-uniform difficulty and error accumulation must be addressed simultaneously.

**Paper Goals**: Design a DLM sampling method that adaptively adjusts parallelism and allows self-correction.

**Starting Point**: Two key insights—(1) generation difficulty decreases as context is established (confidence monotonically increases); (2) the confidence of already-generated tokens changes as new context emerges (potentially dropping from high to low).

**Core Idea**: Adaptive thresholding + backtracking remasking—cautious early on (few tokens unmasked) and aggressive later (large-scale parallelism), while allowing revocation of "regretted" tokens.

## Method

### Overall Architecture

Each step consists of two phases: (1) adaptive acceleration—a dynamic threshold $\tau_t$ (the mean confidence of already-unmasked tokens) determines which new tokens may be unmasked; (2) backtracking remasking—previously unmasked tokens are re-evaluated under the new context, and the $\mu_t$ tokens with the largest confidence drops are re-masked.

### Key Designs

1. **Adaptive Dynamic Threshold Acceleration**:

    - Function: Naturally adjusts parallelism according to generation progress.
    - Mechanism: $\tau_t = \frac{1}{|\mathcal{U}_{t-1}|} \sum_{j \in \mathcal{U}_{t-1}} c_j^{\text{unmask}}$, i.e., the mean unmasking-time confidence of already-unmasked tokens. All masked tokens whose confidence exceeds $\tau_t$ are selected into draft set $\mathcal{D}_t$.
    - Design Motivation: $\tau_t$ rises naturally with progress—when context is sparse early on the mean is low and only the most certain tokens are unmasked; when context is rich later the mean is high and large-scale parallelism is permitted.

2. **Backtracking-Enhanced Remasking**:

    - Function: Revokes early decisions that are falsified under the new context.
    - Mechanism: The remasking count $\mu_t = \max(1, \lfloor |\mathcal{D}_t| / \mu \rfloor)$ is proportional to the aggressiveness of the current step. For each unmasked token, the confidence drop $\Delta_j = c_j^{t-1} - c_j^t$ is computed, and the $\mu_t$ tokens with the largest drops are re-masked.
    - Design Motivation: Conventional DLM sampling is irreversible—tokens locked in prematurely corrupt the context for all subsequent steps. The backtracking mechanism allows the model to "reconsider," fundamentally addressing error propagation.

3. **Training-Free Design**:

    - Function: Directly applicable to any DLM without retraining.
    - Mechanism: Saber modifies only the token selection and revocation strategy during sampling, leaving model weights and architecture unchanged.
    - Design Motivation: Orthogonal to research that improves DLM training—Saber can be stacked on top of any DLM.

### Loss & Training

Training-free method. Experiments are conducted on LLaDA-8B-Instruct with temperature 0 and a generation length of 256 tokens.

## Key Experimental Results

### Main Results

**Code Generation Pass@1 and Inference Speed**

| Method | HumanEval Pass@1 | MBPP Pass@1 | Avg. Steps | Relative Speedup |
|--------|-----------------|-------------|------------|-----------------|
| Confidence (standard) | 43.29 | 42.86 | 256 | 1.0× |
| Fast-dLLM | 38.54 | 38.95 | ~80 | ~3.2× |
| Saber | **45.12** | **44.76** | ~72 | ~3.5× |

### Ablation Study

| Configuration | HumanEval Pass@1 | Note |
|---------------|-----------------|------|
| Saber (full) | 45.12 | Full model |
| w/o backtracking | 42.68 | Removing backtracking degrades quality |
| w/o adaptive | 43.89 | Removing adaptive threshold reduces speed |
| Fixed threshold | 40.12 | Static threshold performs worst |

### Key Findings

- Saber simultaneously improves quality (+1.9% Pass@1) and speed (251.4% speedup), breaking the speed-quality trade-off of DLMs.
- The backtracking mechanism is the primary source of quality gains—allowing the model to correct early errors prevents cascading failures.
- Adaptive acceleration is the primary source of speed gains—large-scale parallel demasking in later stages.
- Saber is effective across different DLMs (LLaDA, Dream), demonstrating model-agnostic applicability.

## Highlights & Insights

- The "cautious → aggressive" adaptive strategy is highly intuitive and effective—richer context yields greater model confidence, which should permit more parallelism.
- Backtracking remasking is an important innovation in the DLM field, breaking the limitation that "decisions once made cannot be undone."
- The two strategies act synergistically—adaptive acceleration enables aggressive parallelism, while the backtracking mechanism ensures that aggressiveness does not lead to catastrophe.

## Limitations & Future Work

- Backtracking introduces additional computational overhead per step (confidence of already-unmasked tokens must be re-evaluated).
- The hyperparameter $\mu$ (backtracking ratio) requires tuning.
- Validation is limited to code generation; effectiveness on natural language generation remains unknown.
- DLMs as a whole still lag behind autoregressive models; Saber only narrows the gap.

## Related Work & Insights

- **vs. Fast-dLLM**: Uses a fixed threshold for acceleration; Saber employs a dynamic threshold for greater precision.
- **vs. ReMDM**: Adopts staged remasking; Saber performs step-level backtracking at finer granularity.
- **vs. ARM Speculative Decoding**: Addresses a different problem—ARM accelerates single-token generation, whereas Saber optimizes parallel demasking in DLMs.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of adaptive acceleration and backtracking is a first in the DLM field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five code benchmarks, multiple DLMs, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly analyzed; algorithmic pseudocode is complete.
- Value: ⭐⭐⭐⭐ Significant advancement toward practical deployment of DLMs.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](../../CVPR2026/video_understanding/adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](../../NeurIPS2025/video_understanding/videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[ICCV 2025\] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception](../../ICCV2025/video_understanding/egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[ACL 2025\] Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs](../../ACL2025/video_understanding/sparse-to-dense_a_free_lunch_for_lossless_acceleration_of_video_understanding_in.md)

<!-- RELATED:END -->
