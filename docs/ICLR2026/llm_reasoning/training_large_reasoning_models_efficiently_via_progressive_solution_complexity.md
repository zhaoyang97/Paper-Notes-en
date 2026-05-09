---
title: >-
  [Paper Note] Training Large Reasoning Models Efficiently via Progressive Thought Encoding
description: >-
  [ICLR2026][LLM Reasoning][Large Reasoning Models] This paper proposes Progressive Thought Encoding, which encodes evicted token information into fixed-size LoRA weight updates whenever KV cache entries are evicted, enabling efficient RL training of large reasoning models under constrained cache budgets while preserving long-range reasoning capability.
tags:
  - ICLR2026
  - LLM Reasoning
  - Large Reasoning Models
  - RL Training Efficiency
  - KV Cache Compression
  - Parameter-Efficient Fine-Tuning
  - Progressive Thought Encoding
date: 2026-05-08
content_hash: b34581ee6d3f014d
---

# Training Large Reasoning Models Efficiently via Progressive Thought Encoding

**Conference**: ICLR2026
**arXiv**: [2602.16839](https://arxiv.org/abs/2602.16839)
**Code**: Not open-sourced
**Area**: LLM Reasoning
**Keywords**: Large Reasoning Models, RL Training Efficiency, KV Cache Compression, Parameter-Efficient Fine-Tuning, Progressive Thought Encoding

## TL;DR

This paper proposes Progressive Thought Encoding, which encodes evicted token information into fixed-size LoRA weight updates whenever KV cache entries are evicted, enabling efficient RL training of large reasoning models under constrained cache budgets while preserving long-range reasoning capability.

## Background & Motivation

- Large Reasoning Models (LRMs) require long rollouts during RL training to obtain outcome rewards, with autoregressive decoding dominating both time and memory overhead.
- Sliding window cache strategies can bound memory usage, but discarding intermediate reasoning tokens disrupts long-range context comprehension and degrades reasoning quality.
- Experiments confirm that applying sliding window caching during RL training of Qwen models yields noticeably lower performance than full-cache training.
- **Key Challenge**: How to allow reasoning models to effectively "attend" to all preceding tokens under a fixed cache capacity.

## Core Problem

How to train LRMs efficiently under a strict memory budget without sacrificing reasoning accuracy?

## Method

### Cache-aware GRPO Objective

The standard GRPO objective is reformulated into a cache-aware variant. At each step $t$, a cache policy $D$ selects a truncated context:

$$\pi_\theta^D(y|p) = \prod_{t=1}^T \pi_\theta(y_t | \mathcal{C}_t^D)$$

The objective becomes:
$$\mathcal{L}_{\text{GRPO}}^D(\theta_g; \theta_{\text{ref}}) = \mathbb{E}_{y \sim \pi_{\theta_g}^D(\cdot|p)}\left[r(y) - \beta \text{KL}(\pi_{\theta_g}^D(\cdot|p) \| \pi_{\theta_{\text{ref}}}(\cdot|p))\right]$$

### Progressive Thought Encoding

**Core Idea**: Rather than discarding evicted tokens, the method learns from them and updates lightweight parameters accordingly.

**1. Context State Computation**

When the cache is full, the keys and values of evicted tokens are $K_e, V_e$. A learnable global query $q_g$ is used to compute the context state:

$$S_e = \left((W_Q^a q_g)(W_K^a K_e)^T\right)(W_V^a V_e)$$

where $W_Q^a, W_K^a, W_V^a$ project the global query and evicted key-value pairs into a compressed latent space.

**2. LoRA Weight Update**

The context state is converted into a LoRA weight increment:
$$\Delta W = A \cdot S_e \cdot B$$

Mapping matrices $A$ and $B$ inject the compressed contextual information into the model weights.

**3. Progressive Update**

- Decoding continues under the updated policy $\pi_{\theta'}^D$, where $\theta' = \theta + \Delta W$.
- Each time the cache is full, a new $S_e'$ is computed and accumulated: $S_e \leftarrow \text{Normalize}(S_e + S_e')$.
- A learnable global token $h_g$ serves as the carrier for $q_g$ at initialization.

**4. Cache Eviction Strategy**

- Question tokens are always retained in the cache (analogous to a sink-token mechanism).
- Sliding window eviction is applied only to reasoning tokens.
- When the cache is saturated, 25% of tokens are evicted.

### Training and Inference

- **Training**: The adapter is updated online as evicted tokens are continuously learned during rollout.
- **Inference**: The learned adapter enables the model to maintain reasoning capability under a constrained cache.
- Backpropagation through the full-cache rollout is not required.

## Key Experimental Results

### Main Results: Method Comparison (Maximum Generation Length 3072)

| Model | Method | Peak GPU Mem | Math500 | Olympiad | AMC | AIME24 | AIME25 | Avg |
|------|------|-----------|---------|----------|-----|--------|--------|------|
| Qwen2.5-3B | Baseline | – | 50.8 | 27.2 | 34.3 | 20.0 | 13.3 | 26.9 |
| | LoRA | 82.8% | 53.2 | 27.8 | 35.9 | 20.0 | 16.7 | 28.2 |
| | LoRA_c | 38.0% | 50.0 | 27.7 | 33.1 | 16.7 | 10.0 | 25.6 |
| | **Ours** | **45.3%** | **54.0** | **29.0** | **45.0** | 20.0 | 16.7 | **30.1** |
| Qwen2.5-7B | Baseline | – | 56.8 | 34.7 | 48.4 | 23.3 | 16.6 | 33.1 |
| | LoRA | 85.8% | 59.4 | 38.7 | 50.6 | 30.0 | 26.7 | 38.1 |
| | LoRA_c | 63.1% | 61.2 | 35.9 | 52.5 | 20.0 | 26.7 | 36.7 |
| | **Ours** | **67.2%** | 61.2 | **38.7** | 52.5 | **30.0** | **30.0** | **39.6** |
| DS-R1-8B | Baseline | – | 53.6 | 28.7 | 42.5 | 20.0 | 20.0 | 30.1 |
| | LoRA | 88.7% | 57.4 | 35.3 | 55.0 | 23.3 | 20.0 | 34.9 |
| | LoRA_c | 59.1% | 54.2 | 31.9 | 45.0 | 36.7 | 26.7 | 35.1 |
| | **Ours** | **59.8%** | **57.6** | **39.7** | **60.0** | **56.7** | **43.3** | **45.6** |

### Key Numbers

- Average gain of +15.5% on DS-R1-8B (30.1→45.6); AIME2024 +36.7%, AIME2025 +23.3%.
- Peak GPU memory reduced from 88.7% to 59.8% (nearly 30 percentage points).
- Computation (TFLOPs) reduced from 7.4 to 4.6 (38% reduction).
- Average gain of +10.5% over naive cache truncation (LoRA_c).

## Highlights & Insights

1. **Turning cache eviction into a learning opportunity**: Rather than discarding tokens outright, the method extracts and encodes their information as model weight updates.
2. **Training–inference consistency**: Both training and inference operate under constrained cache, eliminating the train–inference mismatch problem.
3. **Substantial efficiency gains**: ~50% memory reduction and ~38% compute reduction, with reasoning accuracy improving rather than degrading.
4. **Remarkable gains on DS-R1-8B**: Average accuracy improves by 15.5 percentage points, far surpassing full-cache LoRA.
5. **Seamless integration with GRPO**: The method can be directly embedded into existing RL training frameworks.

## Limitations & Future Work

- Validation is limited to mathematical reasoning; effectiveness on other reasoning tasks (code, science) remains unexplored.
- The sliding window eviction strategy is relatively simple; importance-based eviction may be more effective but incurs higher computational overhead.
- Sensitivity analysis of the global query $q_g$ dimensionality and the number of global tokens (32) is insufficiently detailed.
- The LoRA rank is fixed at 32; the impact of rank selection on performance is not thoroughly analyzed.
- Scalability of the 8×A100 training setup to larger models is uncertain.

## Related Work & Insights

| Method | Mechanism | Memory Efficiency | Reasoning Accuracy | Applicable Scenario |
|------|---------|---------|-----------|---------|
| Full-cache LoRA | Standard RL + LoRA | Low | Baseline | Small models / short sequences |
| Sliding window LoRA | Truncated cache | High | Below baseline | Constrained environments |
| TTT (test-time training) | Gradient updates at inference | Medium | Medium | Online adaptation |
| **Ours** | **Evicted tokens encoded into weights** | **High** | **Exceeds baseline** | **Long-range reasoning RL** |

The intuition of "learning from what is discarded" bears conceptual resemblance to knowledge distillation — information from evicted tokens is not lost but compressed and retained. The approach has significant practical implications for inference-time compute optimization, particularly for deploying long-chain reasoning models on edge devices, and is compatible with other efficient inference methods such as speculative decoding and early stopping.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Converting cache eviction into online learning is an elegant idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three models and six benchmarks with thorough comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and method description is complete.
- **Value**: ⭐⭐⭐⭐ — Substantially reduces RL training costs for LRMs with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)
- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)

</div>

<!-- RELATED:END -->
