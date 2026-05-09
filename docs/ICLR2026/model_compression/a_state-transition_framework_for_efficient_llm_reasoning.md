---
title: >-
  [Paper Note] A State-Transition Framework for Efficient LLM Reasoning
description: >-
  [ICLR 2026][Model Compression][efficient reasoning] This paper proposes an efficient reasoning framework that models the LLM reasoning process as a state-transition process. It uses Linear Attention to compress information from historical reasoning steps into a state matrix, reducing attention complexity from $O(C^2)$ to $O(C)$ and KV cache from $O(C)$ to $O(1)$, while preserving the full CoT sequence and maintaining reasoning capability. An additional momentum strategy mitigates the overthinking problem caused by noisy reasoning steps.
tags:
  - ICLR 2026
  - Model Compression
  - efficient reasoning
  - linear attention
  - state transition
  - KV cache
  - long CoT
date: 2026-05-08
content_hash: a8e25008bbdea1b0
---

# A State-Transition Framework for Efficient LLM Reasoning

**Conference**: ICLR 2026
**arXiv**: [2602.01198](https://arxiv.org/abs/2602.01198)
**Code**: Available
**Area**: Model Compression
**Keywords**: efficient reasoning, linear attention, state transition, KV cache, long CoT

## TL;DR
This paper proposes an efficient reasoning framework that models the LLM reasoning process as a state-transition process. It uses Linear Attention to compress information from historical reasoning steps into a state matrix, reducing attention complexity from $O(C^2)$ to $O(C)$ and KV cache from $O(C)$ to $O(1)$, while preserving the full CoT sequence and maintaining reasoning capability. An additional momentum strategy mitigates the overthinking problem caused by noisy reasoning steps.

## Background & Motivation
**Background**: Long CoT (e.g., o1, R1) significantly improves LLM reasoning ability, but the quadratic complexity of Transformer attention makes computation and memory costs prohibitively high for long CoT.

**Limitations of Prior Work**: Existing efficient reasoning methods primarily compress the CoT sequence (shortening, token dropping, rewriting), which conflicts with test-time scaling — compressing CoT degrades reasoning capability.

**Key Challenge**: Efficiency demands reduced computation, while reasoning capability requires preserving the complete reasoning chain. Compressing CoT content and compressing the attention computation over CoT are fundamentally different problems.

**Goal**: How to reduce the computational and memory overhead of reasoning without shortening the CoT?

**Key Insight**: Within each reasoning step, the useful reasoning information (conclusions) is far less than the linguistic information (syntax, phrasing). A linear attention state matrix is used to retain only reasoning information, discarding linguistic information.

**Core Idea**: Each token in a reasoning step accesses historical reasoning information efficiently through the linear attention state matrix, rather than attending explicitly to historical tokens.

## Method

### Overall Architecture
The softmax attention in an LLM is replaced with a Mixed Attention Module (MAM): the SA sub-module attends only to tokens within the current step and the query prompt, while the LA sub-module maintains a state matrix $S_t$ that records information from completed reasoning steps. Each token in the current step retrieves historical reasoning information from the state matrix via $\mathbf{o} = \mathbf{q} \cdot S_t$.

### Key Designs

1. **Mixed Attention Module (MAM)**:

    - **Function**: Replaces original softmax attention with a dual-path SA+LA attention mechanism.
    - The SA sub-module preserves the original softmax attention of the LLM, but each token attends only to the query prompt and tokens within the current reasoning step (KV entries from completed steps are cleared).
    - The LA sub-module maintains a state matrix $S_t = \sum_{i=1}^{t} k_i^T v_i$ via linear attention, from which each token retrieves historical information using a query vector $q$. A gating mechanism controls the amount of historical information utilized.
    - **Design Motivation**: SA ensures precise intra-step attention without loss; LA provides efficient access to history. Attention complexity is reduced from $O(C^2)$ to $O(C)$, and KV cache from $O(C)$ to $O(1)$.

2. **State-based Reasoning Strategy (Momentum Correction)**:

    - **Function**: Uses a globally accumulated momentum direction to correct deviations introduced by noisy reasoning steps.
    - **Mechanism**: The state change per step $\nabla_t = S_t - S_{t-1}$ is treated as a "gradient," and momentum accumulates a global direction $\bar{\nabla}_{t-1} = \frac{1}{t-1}\sum \nabla_i$. After completing step $t$, correction is applied: $\hat{\nabla}_t = (1-\alpha)\nabla_t + \alpha\bar{\nabla}_{t-1}$.
    - **Design Motivation**: Grounded in the TTT perspective of linear attention — state updates are equivalent to gradient descent, and momentum is a classical technique for mitigating gradient noise.

3. **Training Strategy**:

    - Only the LA sub-module parameters (LoRA) and special thinking-mode tokens are trained.
    - Dual loss: autoregressive loss $\mathcal{L}_{AR}$ + knowledge distillation loss $\mathcal{L}_{KD}$ from the base model.
    - High-entropy tokens segment the CoT into reasoning steps; each step is annotated with a special token indicating its thinking-mode type.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{AR} + \beta \mathcal{L}_{KD}$. Training uses 95K high-quality mathematical CoT samples. Built upon the Qwen2.5 series, ranging from 1.5B to 14B parameters.

## Key Experimental Results

### Main Results (vs. Efficient Reasoning Baselines)

| Method | Type | GSM8K Acc | MATH-500 Acc | Inference Latency↓ |
|------|------|-----------|-------------|---------|
| Base model | Full attention | 80.1 | 78.8 | Baseline |
| LightThinker | CoT compression | Lower | Lower | Lower |
| INFTYTHINK | Summary compression | Lower | Lower | Lower |
| H2O | KV cache pruning | Lower | Lower | Lower |
| **Ours** | **State transition** | **≥Base** | **≥Base** | **Significantly lower** |

### Key Findings
- Reasoning performance does not degrade and in some cases improves: the state-transition framework matches or surpasses the full-attention base model on multiple benchmarks.
- Inference latency is substantially reduced, with greater gains as CoT length increases (due to the constant size of the state matrix).
- The momentum correction strategy is effective: ablations show significant accuracy improvements on challenging tasks such as AIME.
- Consistent effectiveness is demonstrated across three model scales: 1.5B, 7B, and 14B.
- The knowledge distillation loss is critical for training the LA sub-module.

## Highlights & Insights
- **"Shorten the attention, not the CoT"**: This approach elegantly sidesteps the tension between efficiency and reasoning capability — the full reasoning chain is preserved, while attention is computed only within the current step.
- **Momentum correction under the TTT perspective**: By leveraging the theoretical equivalence between linear attention and online learning, momentum is naturally introduced to suppress noisy steps, yielding an approach that is both theoretically elegant and empirically effective.
- **Test-time scaling friendly**: The framework allows the CoT to grow arbitrarily while keeping computation and memory complexity linear.

## Limitations & Future Work
- The expressiveness of linear attention may be inferior to softmax attention — complex cross-step reasoning dependencies may be lost in the state matrix.
- High-entropy token-based segmentation of reasoning steps may be imprecise and is sensitive to training data quality.
- Validation is limited to mathematical reasoning; performance on other domains such as code and scientific reasoning remains unknown.
- Initialization from a DeepSeek-R1 distilled model is required, which increases the complexity of the training pipeline.

## Related Work & Insights
- **vs. LightThinker (CoT compression)**: LightThinker compresses reasoning information with special tokens but remains within the softmax attention framework; this paper replaces the attention mechanism entirely via linear attention, representing a more fundamental approach.
- **vs. TTT (Test-Time Training)**: This paper draws on the theoretical TTT perspective of linear attention but applies it to reasoning efficiency rather than capability enhancement.
- **vs. KV cache compression (H2O, SapLLM)**: These methods selectively retain entries based on attention scores and may discard critical information; this paper's dual-path design (SA+LA) guarantees precise attention within the current step and efficient access to history.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The state-transition modeling combined with mixed attention design is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-scale models, 7 benchmarks, and comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Framework is clearly presented, though notation and formulas are dense.
- Value: ⭐⭐⭐⭐⭐ Provides a fundamental solution for efficient long-CoT reasoning.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)
- [\[ICLR 2026\] Efficient Reasoning with Balanced Thinking](efficient_reasoning_with_balanced_thinking.md)
- [\[ICLR 2026\] Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](incentivizing_agentic_reasoning_in_llm_judges_via_tool-integrated_reinforcement_.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](../../ACL2026/model_compression/wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[ICLR 2026\] A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)

<!-- RELATED:END -->
