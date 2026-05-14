---
title: >-
  [Paper Note] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning
description: >-
  [ACL 2026][LLM Reasoning][To be supplemented] To be supplemented after thorough reading.
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "To be supplemented"
date: 2026-05-08
content_hash: 6f4bf065a1ab558f
---

# CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning

**Conference**: ACL 2026
**arXiv**: [2604.17297](https://arxiv.org/abs/2604.17297)
**Code**: [GitHub](https://github.com/)
**Area**: LLM Reasoning Efficiency
**Keywords**: chain-of-thought compression, attention saliency, reasoning redundancy, greedy search, efficient inference

## TL;DR

This paper proposes CRISP, a framework that identifies the attention pattern of the `</think>` token as a reliable indicator for distinguishing critical from redundant steps in reasoning chains. Building on this insight, CRISP designs a greedy-search compression pipeline with four atomic operators, reducing token usage by 50–60% while preserving accuracy.

## Background & Motivation

**State of the Field**: Reasoning-oriented LLMs (e.g., DeepSeek-R1, OpenAI o1) achieve strong performance by generating long chains of thought (CoT), but this introduces substantial computational overhead and latency. CoT compression has become a practical necessity for deployment.

**Limitations of Prior Work**: Existing CoT compression methods typically rely on external surrogate models (e.g., independent LLMs) to evaluate and prune reasoning steps. However, such external compressors are misaligned with the source model's intrinsic reasoning dynamics—they frequently misclassify critical intermediate steps (e.g., self-correction) as redundant, thereby disrupting the logical coherence of the reasoning chain.

**Root Cause**: A signal is needed to distinguish "critical logical steps" from "redundant steps" within a reasoning chain. Crucially, this signal should not originate from an external model (which introduces misalignment) but from the model's own intrinsic mechanism.

**Paper Goals**: To guide CoT compression using signals intrinsic to the model itself, rather than external surrogates.

**Starting Point**: The observation that the `</think>` token acts as an "information anchor" in deep attention layers—when generating the final answer, the model attends primarily to the `</think>` position rather than to intermediate reasoning steps, and the attention distribution over `</think>` reflects each step's contribution to the final answer.

**Core Idea**: The attention pattern of the `</think>` token is used as an intrinsic measure of step saliency. A greedy search over four atomic operators (Keep, Prune, Rewrite, Fuse) constructs a compressed reasoning path, which is then refined by an LLM to restore grammatical coherence.

## Method

### Overall Architecture

CRISP consists of three stages: (1) **Raw CoT Generation**—obtaining a complete reasoning trajectory from the source model; (2) **Critical Reasoning Path Search**—assessing step saliency via `</think>` attention and compressing the reasoning chain through dynamic operators; (3) **Refinement and Fine-tuning**—restoring semantic coherence of the compressed path using an LLM, followed by multi-task fine-tuning of the target model.

### Key Designs

1. **Discovery of `</think>` as an Information Anchor**

    - **Function**: Provides step saliency signals without requiring an external model.
    - **Mechanism**: Attention visualization reveals that, in deeper layers, the `</think>` token progressively aggregates information from the preceding reasoning chain; during final answer generation, the model attends predominantly to the `</think>` position. Step saliency $S_i$ is defined as the normalized sum of attention weights from `</think>` to tokens in step $r_i$ across all layers and heads. Steps with high attention encode critical information (perplexity spikes upon removal), while low-attention steps can be safely removed (perplexity changes minimally).
    - **Design Motivation**: External surrogates are misaligned with the source model's reasoning dynamics, whereas the `</think>` attention pattern directly reflects what the source model itself considers important.

2. **Greedy Search over Four Atomic Operators**

    - **Function**: Enables flexible compression of the reasoning chain guided by saliency.
    - **Mechanism**: Four operators are defined—Keep (retain high-saliency steps), Prune (remove low-saliency steps), Rewrite (condense a step via LLM), and Fuse (merge semantically redundant steps). A dynamic action space constrains allowable operations based on saliency scores and semantic similarity. The reward function $R(a) = \log P_\theta(y|x, \mathcal{C} \oplus a(r_i)) - \log P_\theta(y|x, \mathcal{C}) - \beta \cdot \text{Len}(a(r_i))$ balances the gain in answer likelihood against a length penalty.
    - **Design Motivation**: Simple threshold-based filtering risks severing logical dependencies or retaining redundancy; the four operators provide a continuous compression granularity ranging from full retention to complete removal.

3. **Compressed Path Refinement and Multi-task Fine-tuning**

    - **Function**: Restores semantic coherence of the compressed path and trains the model.
    - **Mechanism**: The skeleton produced by greedy search may contain grammatical discontinuities; a high-capacity LLM refiner restores fluency using the original CoT as reference. Fine-tuning employs a multi-task strategy with a control token $\kappa$: inputs with $\kappa$ generate compressed reasoning, while inputs without $\kappa$ generate full reasoning, thereby avoiding catastrophic forgetting.
    - **Design Motivation**: Discrete search operations (especially Prune and Fuse) may introduce logical gaps that necessitate a refinement step.

### Loss & Training

Standard autoregressive negative log-likelihood loss, with training on a mixture of original and compressed trajectories. Training runs for 3 epochs with a learning rate of $1 \times 10^{-5}$, using 2,500 samples from the MATH dataset. Saliency thresholds $\tau_{\text{high}}$ and $\tau_{\text{low}}$ are set at the top-30% and bottom-20% quantiles, respectively.

## Key Experimental Results

### Main Results

| Method | Model | GSM8K Acc | GSM8K Tok | MATH-500 Acc | MATH-500 TE |
|--------|-------|-----------|-----------|--------------|-------------|
| Original | 1.5B | 81.6 | 1669 | 78.2 | 2.22 |
| CRISP | 1.5B | **80.6** | **587** | **75.0** | **4.14** |
| Original | 7B | 90.8 | 1376 | 87.4 | 2.86 |
| CRISP | 7B | **90.1** | **374** | **84.2** | **7.35** |

### Ablation Study

| Method | 1.5B Avg. TE | 7B Avg. TE | Notes |
|--------|-------------|-----------|-------|
| Original | 2.10 | 2.81 | Baseline |
| CoD (prompting) | 2.61 | 4.31 | Insufficient control granularity |
| TALE (external compression) | 2.31 | 3.15 | External misalignment |
| A*-Thought | 2.99 | 4.04 | Search without intrinsic signal |
| CRISP | **4.31** | **6.80** | Best efficiency–accuracy trade-off |

### Key Findings

- CRISP substantially outperforms all baselines in Token Efficiency (6.80 vs. 4.31 for the next-best method on the 7B model).
- On the 7B model, GSM8K accuracy drops by only 0.7% while token count falls from 1,376 to 374.
- Validation experiments on `</think>` attention are clear-cut: removing high-attention steps causes perplexity to spike, while removing low-attention steps leaves perplexity nearly unchanged.
- Saliency scores exhibit a non-uniform distribution; only a small fraction of steps contribute substantially to the final answer.

## Highlights & Insights

- **The discovery of `</think>` as an information anchor is particularly insightful**: it reveals how the intrinsic attention mechanism of reasoning models "summarizes" the entire reasoning process, a finding of independent value for understanding how reasoning models operate.
- **The four-operator design provides flexible compression granularity**: Fuse and Rewrite allow information to be preserved even while compressing, going beyond simple retain/delete decisions.
- **The adoption of the Token Efficiency metric enables quantifiable comparison of the efficiency–accuracy trade-off.**

## Limitations & Future Work

- The computational overhead of greedy search (evaluating multiple operators per step) may become a bottleneck for very long CoT sequences.
- The refinement step depends on an external LLM, introducing additional cost.
- Validation is limited to mathematical reasoning datasets; generalization to code and logical reasoning tasks remains untested.
- The multi-task training strategy with a control token is relatively straightforward; more sophisticated training schemes may exist.

## Related Work & Insights

- **vs. CoD / TALE (prompting / external compression)**: CoD constrains length via prompting but lacks fine-grained control; TALE employs an external model for compression but introduces misalignment. CRISP leverages the model's own attention signals, avoiding misalignment at its source.
- **vs. RL-based methods (e.g., length penalty)**: RL-based approaches incur high computational cost and are sensitive to reward design. CRISP achieves compression via post-processing, avoiding the instability associated with RL training.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The discovery of the `</think>` information anchor is original; the greedy search over four atomic operators is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two model scales, three benchmarks, and multiple baselines are covered, though domain coverage is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, findings are compelling, and experimental organization is sound.
**Code**: To be confirmed
**Area**: llm_reasoning
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[ACL 2026\] Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment](taming_actor-observer_asymmetry_in_agents_via_dialectical_alignment.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)

</div>

<!-- RELATED:END -->
