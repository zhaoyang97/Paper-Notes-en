---
title: >-
  [Paper Note] ConFu: Contemplate the Future for Better Speculative Sampling
description: >-
  [ICLR 2026][Model Compression][speculative decoding] ConFu introduces contemplate tokens into the draft model of speculative decoding, enabling it to anticipate the target model's future generation direction. Combined with a MoE dynamic mechanism and anchor-point sampling training, ConFu achieves 8–11% improvements in acceptance rate and generation speed over EAGLE-3.
tags:
  - ICLR 2026
  - Model Compression
  - speculative decoding
  - contemplate tokens
  - future prediction
  - MoE
  - draft model
  - EAGLE
date: 2026-05-08
content_hash: 57cd160166c7f2e4
---

# ConFu: Contemplate the Future for Better Speculative Sampling

**Conference**: ICLR 2026
**arXiv**: [2603.08899](https://arxiv.org/abs/2603.08899)
**Code**: To be confirmed
**Area**: Model Compression
**Keywords**: speculative decoding, contemplate tokens, future prediction, MoE, draft model, EAGLE

## TL;DR

ConFu introduces contemplate tokens into the draft model of speculative decoding, enabling it to anticipate the target model's future generation direction. Combined with a MoE dynamic mechanism and anchor-point sampling training, ConFu achieves 8–11% improvements in acceptance rate and generation speed over EAGLE-3.

## Background & Motivation

**Speculative decoding paradigm**: A lightweight draft model proposes candidate token sequences, which are verified in a single forward pass by the target model; batched acceptance accelerates inference. The core metrics are token acceptance rate and end-to-end speedup ratio.

**EAGLE series as current SOTA**: EAGLE-1/2/3 progressively refine the draft head architecture (single-layer Transformer + target model hidden states), establishing the strongest baselines for speculative decoding.

**Core problem — error accumulation**: Existing draft models generate solely conditioned on the current prefix; as the number of draft steps increases, errors propagate and accumulate from upstream draft tokens, causing the draft distribution to drift away from the target distribution and degrading acceptance rates.

**Key insight**: If the draft model could access the target model's current "line of thought"—high-level semantic intent rather than specific tokens—it could generate candidates that better align with the target trajectory and reduce verification rejections.

**Latent reasoning inspiration**: Works such as COCONUT demonstrate that LLMs can produce continuous "thinking tokens" as intermediate reasoning states, but at the cost of multiple forward passes. Pause tokens (Goyal et al.) can yield additional computation "for free" within parallel computation.

## Method

### Core Innovation 1: Contemplate Tokens + Soft Prompts

- Learnable soft prompt tokens (in the KV cache dimension) are prepended to the target model's input, with a contemplate token appended at the end.
- Attention masking constraint: only contemplate tokens may attend to soft prompts, leaving the original prefix representations unaffected.
- The hidden state of the contemplate token encodes the target model's "intermediate thought," which is provided to the draft model as a future token $\mathbf{f}$.
- Verification phase: one contemplate token is inserted at each node of the draft tree; verification and future prediction generation proceed in parallel. Upon acceptance, the corresponding future prediction is passed to the next iteration.
- Computational overhead: $2T$ tokens are processed during verification ($T$ original draft nodes + $T$ contemplate tokens), where $T$ is typically 30–60.

### Core Innovation 2: MoE Dynamic Contemplate Token

- A static contemplate embedding is insufficient for diverse contexts. Mathematical reasoning may require "the next equation is," while creative writing may require "this passage is about."
- The contemplate token embedding is parameterized via MoE: the hidden state of the most recently accepted token serves as input, and a linear router selects a weighted combination of top-K experts.
- The $[\text{con}]$ module (target side) and the $[\text{f}]$ module (draft side) each have independent MoE components.
- This constitutes the first introduction of dynamic behavior into the pause token setting.

### Core Innovation 3: Training Framework

- **Anchor Token Sampling**: $K_{train}$ anchor tokens are randomly sampled and injected with contemplate tokens, reducing the sequence length from $2N$ to $N + K_{train}$.
- **Future Prediction Replication**: The future prediction of each anchor is reused for the neighboring $l$ tokens, improving robustness and sample efficiency.
- **Loss function**: KL divergence aligns the output distributions of the target and draft models; no additional auxiliary losses are required.

## Key Experimental Results

### Main Results (SpecBench, Llama-3.2-3B, T=0.0, 30 nodes)

| Method | Mean Accepted Length τ | Speedup Ratio SR |
|--------|----------------------|-----------------|
| EAGLE-3 | 4.00 | 1.83× |
| **ConFu** | **4.41** | **2.11×** |
| Relative Gain | **+10.3%** | **+15.3%** |

### Across Temperatures and Budgets

| Setting | EAGLE-3 τ → ConFu τ | Gain |
|---------|---------------------|------|
| T=0.0, 30 nodes | 4.00 → 4.41 | +10.3% |
| T=0.7, 30 nodes | 3.44 → 3.75 | +9.0% |
| T=1.0, 60 nodes | 3.89 → 4.27 | +9.8% |
| 8B model average | — | +8–11% |

### Key Findings

- Consistent improvements across all task types (writing / QA / translation / code / math / summarization).
- Robust and effective across different temperatures (0.0 / 0.7 / 1.0) and budgets (30 / 60 nodes).
- Initializing from the EAGLE-3 checkpoint and continuing training for the same number of steps as EAGLE-3 yields no improvement, confirming that the gains originate from the ConFu architecture rather than extended training.
- Training on 8×H100; inference on a single H100.

## Highlights & Insights

- **First bridge between continuous reasoning tokens and speculative decoding**: conceptually pioneers a new direction of "future-aware draft generation."
- Contemplate tokens leverage the pause token mechanism to achieve "thinking" at virtually no extra cost—no additional forward passes are required.
- MoE dynamic tokens adaptively select "prompt instructions" for different contexts, constituting an elegant design.
- The method builds on EAGLE-3 and achieves orthogonal improvements, remaining compatible with the baseline architecture's evolution.

## Limitations & Future Work

- Evaluated only on Llama-3 3B/8B; whether comparable gains hold for larger models (70B+) remains unknown.
- The optimal configurations for the number of soft prompt tokens (default 16) and MoE experts have not been systematically studied.
- The additional overhead of $2T$ tokens during verification may be non-negligible when the draft tree is very large.
- Compatibility with target model architectures other than LLaMA has not been verified.

## Related Work & Insights

- **EAGLE-1/2/3**: The strongest baselines with progressively improved draft architectures and training; ConFu provides orthogonal improvements.
- **BiTA**: Uses soft prompts to directly decode future tokens; ConFu uses them to guide the draft model rather than for direct decoding.
- **COCONUT / Latent Reasoning**: Requires multiple forward passes to obtain continuous thinking; ConFu obtains this in parallel via pause tokens.
- **Medusa / HASS**: Earlier speculative decoding methods, already surpassed by the EAGLE series.

## Rating

- Novelty: ⭐⭐⭐⭐ First combination of future prediction and speculative decoding
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across tasks, temperatures, and budgets with well-controlled variables
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive figures, and smooth reasoning flow
- Value: ⭐⭐⭐⭐ Opens a new direction for improving speculative decoding

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation](lookaheadkv_fast_and_accurate_kv_cache_eviction_by_glimpsing_into_the_future_wit.md)
- [\[ICLR 2026\] Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[NeurIPS 2025\] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](../../NeurIPS2025/model_compression/adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)

</div>

<!-- RELATED:END -->
