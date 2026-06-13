---
title: >-
  [Paper Note] Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs
description: >-
  [ACL 2026][LLM Efficiency][Diffusion Language Models] This paper proposes Saber, a training-free sampling algorithm for Diffusion Language Models (DLMs). By employing adaptive acceleration (dynamically adjusting parallel…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Diffusion Language Models"
  - "Adaptive Sampling"
  - "Backtracking Remasking"
  - "Code Generation Acceleration"
  - "Speed-Quality Trade-off"
date: 2026-05-08
content_hash: cca97abf5c68ebea
---

# Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs

**Conference**: ACL 2026  
**arXiv**: [2510.18165](https://arxiv.org/abs/2510.18165)  
**Code**: [GitHub](https://github.com/zhaoyMa/Saber)  
**Area**: LLM Efficiency  
**Keywords**: Diffusion Language Models, Adaptive Sampling, Backtracking Remasking, Code Generation Acceleration, Speed-Quality Trade-off

## TL;DR

This paper proposes Saber, a training-free sampling algorithm for Diffusion Language Models (DLMs). By employing adaptive acceleration (dynamically adjusting parallel decoding based on established context) and backtracking enhanced remasking (undoing tokens invalidated by new context), it achieves a 251.4% inference acceleration while improving Pass@1 by an average of 1.9% in code generation.

## Background & Motivation

**Background**: DLMs (e.g., LLaDA, Dream) achieve parallel generation through iterative demasking, serving as a powerful alternative to autoregressive models. However, in tasks with strong structural constraints like code generation, reducing sampling steps leads to a catastrophic drop in Pass@1 (sometimes exceeding 60%).

**Limitations of Prior Work**: (1) Static acceleration strategies (fixed token counts or confidence thresholds) are too conservative for simple stages and too aggressive for complex ones; (2) DLM decoding is irreversible—once a token is unmasked, it cannot be undone, causing early errors to be permanently locked and propagated.

**Key Challenge**: The speed advantage of parallel generation vs. the quality collapse caused by error propagation—requiring a simultaneous solution for non-uniform difficulty and error accumulation.

**Goal**: Design a DLM sampling method that adaptively adjusts parallelism and allows for self-correction.

**Key Insight**: Two key observations—(1) Generation difficulty decreases as context is established (confidence rises monotonically); (2) The confidence of generated tokens changes as new context emerges (potentially shifting from high to low).

**Core Idea**: Adaptive threshold + Backtracking remasking—cautious in early stages (few unmasked tokens) and aggressive later (high parallelism), while allowing the "regret" of unmasking specific tokens.

## Method

### Overall Architecture

Each step consists of two stages: (1) Adaptive Acceleration—uses a dynamic threshold $\tau_t$ (the average confidence of already unmasked tokens) to determine which new tokens can be unmasked; (2) Backtracking Remasking—re-evaluates the confidence of unmasked tokens under the new context and remasks the $\mu_t$ tokens with the largest confidence drops.

### Key Designs

1.  **Adaptive Dynamic Threshold Acceleration**:
    - **Function**: Naturally adjusts parallelism based on generation progress.
    - **Mechanism**: $\tau_t = \frac{1}{|\mathcal{U}_{t-1}|} \sum_{j \in \mathcal{U}_{t-1}} c_j^{\text{unmask}}$, representing the average confidence of tokens at the time they were unmasked. All masked tokens with confidence exceeding $\tau_t$ are selected for the draft set $\mathcal{D}_t$.
    - **Design Motivation**: $\tau_t$ naturally increases with progress—low mean values when early context is sparse unmask only the most certain tokens, while high mean values in later stages with rich context allow for massive parallelism.

2.  **Backtracking Enhanced Remasking**:
    - **Function**: Revokes early decisions invalidated by new context.
    - **Mechanism**: The number of remasked tokens $\mu_t = \max(1, \lfloor |\mathcal{D}_t| / \mu \rfloor)$ is proportional to the aggressiveness of the current step. For each unmasked token, the confidence drop $\Delta_j = c_j^{t-1} - c_j^t$ is calculated, and the $\mu_t$ tokens with the largest drops are remasked.
    - **Design Motivation**: Traditional DLM sampling is irreversible—early locked errors ruin the context for all subsequent steps. The backtracking mechanism allows the model to "repent," fundamentally addressing the error propagation problem.

3.  **Training-Free Design**:
    - **Function**: Directly applicable to any DLM without retraining.
    - **Mechanism**: Saber only modifies the token selection and revocation strategy during sampling without changing model weights or architecture.
    - **Design Motivation**: Orthogonal to research improving DLM training—Saber can be stacked on top of any DLM.

### Loss & Training

A training-free method. Experiments were conducted on LLaDA-8B-Instruct with temperature 0 and a generation length of 256 tokens.

## Key Experimental Results

### Main Results

**Code Generation Pass@1 and Inference Speed**

| Method | HumanEval Pass@1 | MBPP Pass@1 | Avg. Steps | Relative Speedup |
|--------|------------------|-------------|------------|------------------|
| Confidence (Standard) | 43.29 | 42.86 | 256 | 1.0x |
| Fast-dLLM | 38.54 | 38.95 | ~80 | ~3.2x |
| **Saber** | **45.12** | **44.76** | **~72** | **~3.5x** |

### Ablation Study

| Configuration | HumanEval Pass@1 | Description |
|---------------|------------------|-------------|
| **Saber (Full)** | **45.12** | Full model |
| w/o Backtracking | 42.68 | Quality decreases without backtracking |
| w/o Adaptive | 43.89 | Speed decreases without adaptive threshold |
| Fixed Threshold | 40.12 | Static threshold performs worst |

### Key Findings

- Saber simultaneously improves quality (+1.9% Pass@1) and speed (251.4% acceleration)—breaking the DLM speed-quality trade-off.
- The backtracking mechanism is the primary source of quality improvement—allowing the model to correct early errors avoids cascading failures.
- Adaptive acceleration is the primary driver of speed improvement—enabling massive parallel unmasking in later stages.
- Saber is effective across different DLMs (LLaDA, Dream)—demonstrating model-agnosticism.

## Highlights & Insights

- The "cautious → aggressive" adaptive strategy is highly intuitive and effective—as context becomes richer, the model becomes more confident, allowing higher parallelism.
- Backtracking remasking is a significant innovation in the DLM field—breaking the constraint that "a decision once made cannot be revoked."
- The two strategies work synergistically—adaptive acceleration enables aggressive parallelism, while the backtracking mechanism ensures that aggressiveness does not lead to disaster.

## Limitations & Future Work

- Backtracking increases the computational overhead per step (requires re-evaluating confidence for unmasked tokens).
- The hyperparameter $\mu$ (backtracking ratio) requires tuning.
- Validated only on code generation; the effect on natural language generation is unknown.
- DLMs overall still lag behind ARMs; Saber only narrows the gap.

## Related Work & Insights

- **vs Fast-dLLM**: Uses fixed threshold acceleration; Saber is more precise with dynamic thresholds.
- **vs ReMDM**: Employs staged remasking; Saber provides finer-grained step-by-step backtracking.
- **vs ARM Speculative Decoding**: Addresses different problems—ARM speeds up single-token generation, while Saber optimizes parallel demasking in DLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of adaptive threshold and backtracking is a first in the DLM field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 code benchmarks + multiple DLMs + detailed ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation analysis and complete algorithmic pseudocode.
- **Value**: ⭐⭐⭐⭐ Significantly advances the practical application of DLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration](specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati.md)
- [\[ICML 2026\] TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](../../ICML2026/llm_efficiency/team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)
- [\[ICML 2026\] KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](../../ICML2026/llm_efficiency/knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)
- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](../../ICML2026/llm_efficiency/skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)
- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](native_hybrid_attention_for_efficient_sequence_modeling.md)

</div>

<!-- RELATED:END -->
