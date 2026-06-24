---
title: >-
  [Paper Note] CLaSp: In-Context Layer Skip for Self-Speculative Decoding
description: >-
  [ACL 2025][LLM Efficiency][speculative decoding] CLaSp proposes a training-free self-speculative decoding method that dynamically adjusts the layer skipping strategy based on context after each verification step using a dynamic programming algorithm. By utilizing the full hidden states of the previous verification step as the target to select the optimal set of skipped layers, it achieves $1.3-1.7\times$ speedup on the LLaMA3 series without altering the generation distributio…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "speculative decoding"
  - "layer skipping"
  - "self-speculative"
  - "dynamic programming"
  - "lossless acceleration"
date: 2026-05-08
content_hash: 59ef90ea0d28cdda
---

# CLaSp: In-Context Layer Skip for Self-Speculative Decoding

**Conference**: ACL 2025  
**arXiv**: [2505.24196](https://arxiv.org/abs/2505.24196)  
**Code**: None  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: speculative decoding, layer skipping, self-speculative, dynamic programming, lossless acceleration

## TL;DR
CLaSp proposes a training-free self-speculative decoding method that dynamically adjusts the layer skipping strategy based on context after each verification step using a dynamic programming algorithm. By utilizing the full hidden states of the previous verification step as the target to select the optimal set of skipped layers, it achieves $1.3-1.7\times$ speedup on the LLaMA3 series without altering the generation distribution.

## Background & Motivation

### Background

**Background**: Speculative decoding (SD) is a primary method for lossless LLM acceleration, which pre-generates tokens using a fast draft model and verifies them in parallel with a full verify model. Self-speculative decoding (Self-SD) constructs the draft model by skipping some layers of the verify model, avoiding compatibility issues associated with additional models.

**Limitations of Prior Work**: (1) Traditional SD requires finding or training a draft model, which is infeasible for specialized LLMs; (2) Self-SD searches for a fixed set of skipped layers using Bayesian optimization on a training set, which is time-consuming and lacks generalization—different tasks or contexts require different skipping strategies; (3) Although SWIFT optimizes progressively, it relies on accumulating sufficient user requests (leading to a slow cold start).

**Key Challenge**: The layer skipping strategy should adapt to changes in context (dynamic), but existing methods either use a statically fixed set of skipped layers or rely on offline optimization.

**Goal**: Design a plug-and-play dynamic layer skipping strategy that is training-free, requires no pre-optimization, and adjusts in real-time according to the context.

**Key Insight**: Utilize the full hidden states of the previous verification step as the "ground truth" target and apply dynamic programming to select the optimal layer-skipping combination in real-time.

**Core Idea**: Use the full hidden states from the previous verification as a reference, and select the layer-skipping set that minimizes the approximation error using dynamic programming before each speculation step.

## Method

### Overall Architecture
Verification phase (full $L$ layers) $\rightarrow$ Save hidden states of each layer $X = \{x_0, ..., x_{L-1}\}$ $\rightarrow$ Before the speculation phase, run the DP algorithm to select the optimal set $\mathcal{S}$ of skipping $M$ layers $\rightarrow$ Generate draft tokens using the skipped sub-model $\rightarrow$ Accept/Reject in the verification phase $\rightarrow$ Update hidden states $\rightarrow$ Repeat.

### Key Designs

1. **Dynamic Programming for Selecting Layer-Skipping Set**:

    - **Function**: Given an $L$-layer model and a budget of skipping $M$ layers, find the layer-skipping combination that minimizes the discrepancy between the skipped and full hidden states.
    - **Mechanism**: Define the state $g[l, m]$ as the "hidden state at layer $l$ when $m$ layers have been skipped," with the objective to minimize $\|g[L, M] - x_{L-1}\|$. The DP recurrence is: at layer $l$, either execute ($g[l+1, m] = f_l(g[l, m])$) or skip ($g[l+1, m+1] = g[l, m]$).
    - **Time Complexity**: $O(L \times M)$ forward passes (where each pass only computes a single layer); leveraging GPU parallelism makes the additional latency almost negligible.
    - **Design Motivation**: Based on the observation of "slowly changing inter-layer embeddings," the error introduced by layer skipping is controllable.

2. **Context Adaptivity**:

    - **Function**: Re-run DP after each verification to update the layer skipping strategy based on the new full hidden states.
    - **Design Motivation**: Different generation stages (e.g., beginning vs. middle vs. end) may require different critical layers. Static strategies fail to adapt.
    - **Implementation**: The verification phase inherently requires forward propagation through all layers to obtain hidden states, incurring no additional storage overhead.

3. **GPU Parallel Optimization**:

    - **Function**: Multiple candidate paths in DP can be computed in parallel.
    - Layer forward passes can be executed in batches.
    - Ensures that the additional latency (DP selection overhead) is significantly smaller than the speedup gain brought by layer skipping.

### Comparison with Traditional Self-SD
- Self-SD: Offline Bayesian optimization $\rightarrow$ Fixed layer-skipping set $\rightarrow$ Static during inference.
- CLaSp: Online DP optimization $\rightarrow$ Dynamic layer-skipping set $\rightarrow$ Updated after each verification step.

## Key Experimental Results

### Main Results: Spec-Bench (LLaMA3 Series)

| Method | LLaMA3-8B Speedup | LLaMA3-70B Speedup | Requires Training |
|------|----------------|-----------------|---------|
| Autoregressive Baseline | $1.0\times$ | $1.0\times$ | - |
| Self-SD (Static) | $\sim 1.2\times$ | $\sim 1.3\times$ | Requires BO |
| SWIFT | $\sim 1.3\times$ | $\sim 1.4\times$ | Dynamic but requires accumulation |
| **CLaSp** | **$1.3-1.5\times$** | **$1.5-1.7\times$** | **No** |

### Key Findings
- **Lossless Acceleration**: The generation distribution of CLaSp is mathematically identical to the original model (guaranteed by speculative decoding).
- **Dynamic Strategy Outperforms Static**: The layer-skipping set adapts contextually, consistently outperforming fixed layer skipping across diverse tasks.
- **Negligible DP Overhead**: With GPU parallelism, DP selection only introduces $<5\%$ latency overhead.
- **Plug-and-Play**: Training-free, requires no calibration data, and does not modify model parameters.

## Highlights & Insights
- **Formulating draft model design in speculative decoding as a combinatorial optimization problem**: Choosing which layers to skip is framed as a combinatorial selection problem, for which DP is a naturally suited solver.
- **Exploiting "free" information from the verification phase**: The full hidden states from the verification phase are computed anyway; using them as the target for DP introduces no additional computational overhead.
- **Plug-and-play practicality**: Highly valuable for proprietary/private LLMs—eliminating the need to search for a draft model or train additional modules.

## Limitations & Future Work
- Limited speedup ratio ($1.3-1.7\times$), lagged behind trained speculative decoding methods (e.g., EAGLE achieves $2\times+$).
- DP assumes that layer-skipping errors can be tracked sequentially, but error accumulation may become inaccurate when a large number of layers are skipped.
- The combination with tree attention to further improve acceptance rates remains unexplored.
- Not validated on Mixture-of-Experts (MoE) models.

## Related Work & Insights
- **vs. Self-SD (Zhang et al., 2024)**: Self-SD uses Bayesian optimization for a fixed layer-skipping set, whereas CLaSp dynamically adjusts it using DP, eliminating offline optimization.
- **vs. EAGLE (Li et al., 2024)**: EAGLE trains specialized draft heads (achieving greater speedup at the cost of training). CLaSp requires no training whatsoever.
- **vs. LayerSkip (Elhoushi et al., 2024)**: LayerSkip requires introducing layer-skipping regularization during training. CLaSp operates entirely during inference.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of using dynamic programming for layer-skipping selection is novel, and using verification hidden states as the target is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple tasks via Spec-Bench, with comprehensive hyperparameter analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed methodological descriptions.
- Value: ⭐⭐⭐⭐ Plug-and-play lossless acceleration provides practical value for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SAM Decoding: Speculative Decoding via Suffix Automaton](sam_decoding_speculative_decoding_via_suffix_automaton.md)
- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)
- [\[ICML 2026\] KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](../../ICML2026/llm_efficiency/knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)
- [\[ACL 2025\] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)
- [\[ACL 2025\] A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models](a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la.md)

</div>

<!-- RELATED:END -->
