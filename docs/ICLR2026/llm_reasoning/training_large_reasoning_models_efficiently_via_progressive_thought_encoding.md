---
title: >-
  [Paper Note] Training Large Reasoning Models Efficiently via Progressive Thought Encoding
description: >-
  [ICLR 2026][LLM Reasoning][Large Reasoning Models] This paper proposes Progressive Thought Encoding, which encodes evicted thought tokens into LoRA weights under KV cache constraints…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Large Reasoning Models"
  - "RL Training Efficiency"
  - "KV Cache Compression"
  - "Parameter-Efficient Fine-Tuning"
  - "Progressive Thought Encoding"
date: 2026-05-08
content_hash: a709d7496de45bb2
---

# Training Large Reasoning Models Efficiently via Progressive Thought Encoding

**Conference**: ICLR 2026
**arXiv**: [2602.16839](https://arxiv.org/abs/2602.16839)  
**Code**: No public code  
**Area**: LLM Reasoning
**Keywords**: Large Reasoning Models, RL Training Efficiency, KV Cache Compression, Parameter-Efficient Fine-Tuning, Progressive Thought Encoding

## TL;DR

This paper proposes Progressive Thought Encoding, which encodes evicted thought tokens into LoRA weights under KV cache constraints, halving GPU memory usage during RL training of large reasoning models while surpassing full-cache LoRA in reasoning accuracy (up to +23.4% on AIME2024/2025).

## Background & Motivation

**Background**: Large reasoning models (LRMs) undergo RL-based post-training (e.g., GRPO), requiring long rollout sequences to obtain outcome-based rewards. Autoregressive decoding makes the rollout phase the primary bottleneck in both time and memory—harder tasks demand longer chains of thought, further exacerbating resource consumption.

**Limitations of Prior Work**: A natural remedy is to apply a sliding window to limit KV cache size and reduce memory. However, experiments show this severely degrades reasoning quality—discarding intermediate thought tokens disrupts long-range context understanding, lowers rollout sample quality, and consequently impairs training. For example, Qwen2.5-3B suffers an average accuracy drop from 28.2% to 25.6% under a sliding window.

**Key Challenge**: Can LRMs be trained under strict memory budgets without sacrificing reasoning accuracy? That is, can a model effectively "see" all historical tokens within a limited cache window?

## Method

### Overall Architecture

The core idea of Progressive Thought Encoding is: **learn from evicted tokens before discarding them**. Specifically, when the KV cache is full, information from evicted tokens is encoded into a fixed-size vector representation and dynamically injected into lightweight LoRA adapters, enabling the model to maintain long-context understanding under limited cache.

Workflow:
1. Given a question $x$, decode thought tokens continuously during rollout until the KV cache is full.
2. Select tokens to evict $\{y_{e_1}, \ldots, y_{e_m}\}$ according to eviction policy $D$.
3. Compute a context state $S_e$ from the evicted tokens using a global query vector $q_g$.
4. Update LoRA weights via $S_e$: $\Delta W = A \cdot S_e \cdot B$.
5. Continue decoding under the updated policy; repeat whenever the cache fills.

### Key Designs

**Context State Computation**: A learnable global query vector $q_g$ serves as a summary carrier for all evicted context. Information from evicted tokens is aggregated via an attention mechanism:

$$S_e = (W_Q^a \cdot q_g) \cdot (W_K^a \cdot K_e)^T \cdot (W_V^a \cdot V_e)$$

where $K_e$ and $V_e$ are the key-value vectors of evicted tokens, and $W_Q^a$, $W_K^a$, $W_V^a$ are weight matrices that project the global query and evicted tokens into a compressed latent space.

**Cumulative Update Mechanism**: Each time a new batch of tokens is evicted, a new state $S_e'$ is computed and accumulated as $S_e \leftarrow \text{Normalize}(S_e + S_e')$, after which $\Delta W$ is recomputed. This enables streaming adaptation—the model continuously "remembers" evicted tokens throughout generation.

**Global Token Initialization**: Prior to processing any evicted tokens, the context state is initialized with a learnable global token $h_g$, making $q_g$ an explicit carrier of evicted context information from the outset.

### Loss & Training

- **Eviction Policy**: Question tokens are permanently retained during training (analogous to sink tokens); only thought tokens are subject to sliding-window eviction, with 25% of tokens evicted when the cache is saturated.
- **RL Algorithm**: GRPO with the DAPO-Math-17K dataset; global batch size 512.
- **Hyperparameters**: LoRA rank 32, 32 global tokens, learning rate $1\text{e-}5$.
- **Cache Size**: Set to the maximum question length in the current micro-batch.

## Key Experimental Results

### Main Results

Comparison across 3 models × 6 mathematical reasoning benchmarks (maximum generation length 3072):

| Method | Peak GPU Mem | Math500 | Olympiad | AMC | AIME24 (p@16) | AIME25 (p@16) | Avg |
|--------|-------------|---------|----------|-----|---------------|---------------|-----|
| **Qwen2.5-3B** | | | | | | | |
| Baseline | - | 50.8 | 27.2 | 34.3 | 20.0 | 13.3 | 26.9 |
| LoRA | 82.8% | 53.2 | 27.8 | 35.9 | 20.0 | 16.7 | 28.2 |
| LoRA_c (sliding window) | 38.0% | 50.0 | 27.7 | 33.1 | 16.7 | 10.0 | 25.6 |
| **Ours** | **45.3%** | **54.0** | **29.0** | **45.0** | **20.0** | **16.7** | **30.1** |
| **DeepSeek-R1-Distill-8B** | | | | | | | |
| Baseline | - | 53.6 | 28.7 | 42.5 | 20.0 | 20.0 | 30.1 |
| LoRA | 88.7% | 57.4 | 35.3 | 55.0 | 23.3 | 20.0 | 34.9 |
| LoRA_c | 59.1% | 54.2 | 31.9 | 45.0 | 36.7 | 26.7 | 35.1 |
| **Ours** | **59.8%** | **57.6** | **39.7** | **60.0** | **56.7** | **43.3** | **45.6** |

### Ablation Study

Effect of global tokens and eviction policy (DeepSeek-R1-Distill-8B, MATH-500):

| Configuration | Cache 768 | Cache 1K | Cache 2K | Notes |
|---------------|-----------|----------|----------|-------|
| Baseline | 34.4 | 39.6 | 47.8 | No RL training |
| #Global-0 (no global token) | 36.2 | 41.0 | 48.6 | Eviction encoding only; limited gain |
| Global-Only (no eviction update) | 46.8 | 50.2 | 54.0 | Global token effective but insufficient |
| **Ours (#Global-32)** | **48.4** | **52.2** | **55.4** | Global + eviction encoding optimal |
| Ours + HeadKV | 50.7 | 53.4 | 55.8 | Better eviction strategy yields further gains |

**Scalability**: With a fixed 1K cache window, extending maximum generation length from 3K to 64K shows that the proposed method scales consistently across the entire length range, whereas LoRA and LoRA_c progressively saturate.

### Key Findings

1. **Memory halved, accuracy surpassed**: On DeepSeek-R1-8B, peak GPU memory drops from 88.7% to 59.8% (−28.9%), while average accuracy rises from 34.9% to 45.6% (+10.7%).
2. **Remarkable AIME performance**: On AIME2024, DeepSeek-R1-8B improves from 23.3 to 56.7 (+33.4); on AIME2025, from 20.0 to 43.3 (+23.3).
3. **Longer reasoning = better results**: Progressive encoding allows safe expansion of rollout length during training (4K→6K) with negligible memory increase, lifting MATH-500 from 57.6 to 60.2.
4. **Greater stability on long sequences**: The proposed method excels especially on long responses, whereas gains from LoRA_c are concentrated on short responses.

## Highlights & Insights

- **Turning waste into resource**: Reframing KV cache eviction from "information loss" into an "online learning opportunity" represents a remarkably elegant perspective shift.
- **Practical viability of reasoning RL**: Memory consumption is the primary obstacle to scaling RL training; this method directly addresses that engineering bottleneck.
- **A new form of test-time learning**: Progressive encoding is essentially online adaptation during inference—the model continuously learns from its own intermediate thoughts throughout generation.

## Limitations & Future Work

1. **Evaluated solely on mathematical reasoning**: All six benchmarks are mathematical tasks; domains such as code generation and scientific reasoning remain unexplored.
2. **Conservative eviction policy**: Training employs only a simple sliding window; the authors note that superior token selection strategies (e.g., HeadKV) can yield further gains but incur a 37% runtime overhead.
3. **Sensitivity to global token count**: #Global-64 underperforms #Global-32, requiring careful hyperparameter tuning.
4. **No comparison with token-level reward methods**: The current work considers only outcome-based rewards; process rewards may yield additional improvements.

## Related Work & Insights

- Complementary to **test-time training** (e.g., entropy minimization)—the proposed method constitutes "training during generation."
- Orthogonal to **KV cache compression** methods (PyramidKV, H2O, HeadKV)—superior eviction strategies can be directly integrated.
- Insight: **Efficiency optimization in RL training** is a critical direction in current LRM research; this work opens a novel pathway through the lens of cache management.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Transforming cache eviction into an online learning signal is a highly creative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 models × 6 benchmarks with extensive ablations, though limited to the mathematical domain.
- Writing Quality: ⭐⭐⭐⭐ — Problem formalization is clear and mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses the core pain point of RL training for LRMs; significant engineering impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [\[ICLR 2026\] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)

</div>

<!-- RELATED:END -->
