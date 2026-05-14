---
title: >-
  [Paper Note] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models
description: >-
  [ICCV 2025][LLM/NLP][continual learning] This paper proposes Analytic Subspace Routing (Any-SSR), which assigns an independent LoRA subspace to each new task to eliminate knowledge interference…
tags:
  - "ICCV 2025"
  - "LLM/NLP"
  - "continual learning"
  - "LLM"
  - "recursive least squares"
  - "LoRA"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 115c116f73c47199
---

# Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models

**Conference**: ICCV 2025
**arXiv**: N/A
**Code**: [GitHub](https://github.com/ZHUANGHP/Any-SSR)
**Area**: Continual Learning / Large Language Models
**Keywords**: continual learning, LLM, recursive least squares, LoRA, catastrophic forgetting

## TL;DR

This paper proposes Analytic Subspace Routing (Any-SSR), which assigns an independent LoRA subspace to each new task to eliminate knowledge interference, while employing an analytic router based on a recursive least squares (RLS) closed-form solution to dynamically select subspaces. The approach provides theoretical guarantees against forgetting prior task knowledge, enabling replay-free continual learning for LLMs.

## Background & Motivation

LLM fine-tuning is inherently a continual learning process subject to catastrophic forgetting—fine-tuning on new tasks degrades the general capabilities acquired during pretraining. Existing methods either rely on data replay (computationally expensive and privacy-sensitive) or absorb all task knowledge into a single parameter-efficient module (e.g., shared LoRA), leading to severe inter-task knowledge interference. The core challenge is how to continuously absorb new skills without replaying historical data while avoiding catastrophic forgetting.

## Method

### Overall Architecture

Any-SSR adopts a hybrid structure: the first $L_f$ layers of the LLM are frozen as a general-purpose feature extractor, while the remaining layers maintain independent LoRA adapters (subspaces) for each task. An analytic router is placed at the output of the frozen layers, using an RLS closed-form solution to determine which task's LoRA module to activate for a given input. At inference time, the router selects the best-matching LoRA to load into the upper layers.

### Key Designs

1. **Task-Specific LoRA Subspace Isolation**: Based on the assumption that lower layers encode cross-task shared semantic features while upper layers handle task-specific semantic composition, the LLM is partitioned into a frozen general-purpose component and a task-specific component. A dedicated LoRA adapter is trained for each new task and attached to the upper layers. The LoRA parameter spaces of different tasks are entirely disjoint, fundamentally eliminating knowledge interference.

2. **Recursive Least Squares (RLS) Analytic Router**: The router is trained via an RLS closed-form solution, whose key property is that **sequential per-task training is equivalent to joint training on all tasks**. This provides a theoretical guarantee against forgetting. When a new task $\mathcal{D}_{k+1}$ arrives, the router parameters are updated via a recursive formula without revisiting $\mathcal{D}_1$ through $\mathcal{D}_k$. The router takes frozen-layer feature representations as input and outputs normalized weights over all tasks.

3. **Hierarchical Feature Decoupling**: Frozen layers $h \leq L_f$ preserve the pretrained general language understanding capability, providing stable shared representations. Upper layers $h > L_f$ adapt task-specific knowledge via LoRA. The router learns task discrimination in the shared representation space, decoupled from task-specific learning.

### Loss & Training

The LoRA components are trained with standard gradient descent using the conventional next-token prediction loss. The router is computed once via the RLS closed-form solution, requiring no iterative optimization. Upon arrival of a new task, only the new LoRA adapter is trained and the router is recursively updated, leaving existing LoRA parameters untouched.

## Key Experimental Results

### Main Results

| Method | Trace Metric | Forgetting | New Task Performance |
|---|---|---|---|
| Sequential FT | Poor | Severe | Good |
| O-LoRA | Moderate | Moderate | Moderate |
| SEEKR (+replay) | Good | Low | Good |
| **Any-SSR** | **SOTA** | **Near-zero** | **Good** |

Any-SSR achieves state-of-the-art performance on the Trace metric with near-perfect retention of prior task knowledge.

### Ablation Study

- Subspace isolation vs. shared LoRA: isolation significantly reduces inter-task interference.
- RLS router vs. gradient-trained router: RLS guarantees forgetting-free updates.
- Choice of frozen layer depth $L_f$: too few layers yield insufficient shared features; too many limit task-specific adaptation capacity.
- Router accuracy as a function of the number of tasks.

### Key Findings

- The RLS closed-form solution is critical for achieving replay-free, forgetting-free learning—sequential training is provably equivalent to joint training.
- The subspace isolation strategy, though simple, is highly effective.
- General capabilities are primarily encoded in lower layers, while upper layers are more responsible for task-specific reasoning.

## Highlights & Insights

- Creatively applies the classical RLS method to continual learning in LLMs, with rigorous theoretical guarantees.
- Requires no data replay, poses no privacy risks, and is computationally efficient.
- The LoRA subspace isolation design is both simple and effective.
- The forgetting-free property of the router is supported by formal theoretical proof.

## Limitations & Future Work

- The number of LoRA modules grows linearly with the number of tasks, potentially imposing storage pressure over long deployment horizons.
- The router introduces additional inference-time computation, increasing latency.
- Task boundaries are assumed to be known (explicit task delineation), limiting applicability to scenarios with ambiguous task boundaries.
- Evaluation is conducted solely on language tasks; extensibility to multimodal continual learning remains to be verified.

## Related Work & Insights

- O-LoRA performs continual learning in orthogonal subspaces but is limited in capacity due to shared parameter spaces.
- SEEKR reduces forgetting via attention distillation but requires data storage.
- Analytic learning methods (RLS) have been successfully applied in conventional continual learning; this paper extends them to LLMs.
- The router + LoRA bank architecture is potentially extensible to model merging scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of RLS and LoRA subspace routing is original.
- Technical Depth: ⭐⭐⭐⭐⭐ — Theoretical guarantees are rigorous with complete derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — SOTA on the Trace metric with comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ — Theory and practice are well integrated.
- Value: ⭐⭐⭐⭐ — Replay-free and forgetting-free, well-suited for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](va_gpt_aligning_effective_tokens_video_anomaly.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](vim_versatile_interactive_motion_language_model.md)
- [\[ICCV 2025\] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion](balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)
- [\[ICCV 2025\] ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer](shadowhack_hacking_shadows_via_luminance-color_divide_and_conquer.md)
- [\[ICCV 2025\] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)

</div>

<!-- RELATED:END -->
