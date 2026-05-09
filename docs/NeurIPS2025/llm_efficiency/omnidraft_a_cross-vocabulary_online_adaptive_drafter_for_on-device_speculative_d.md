---
title: >-
  [Paper Note] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding
description: >-
  [NeurIPS 2025][LLM Efficiency][Speculative decoding] This paper proposes the OmniDraft framework, which achieves cross-vocabulary speculative decoding via an online n-gram cache, aligns the draft model with the target model through a hybrid distillation loss, and dynamically adjusts the proposal length with an adaptive drafting head. A single lightweight Llama-68M model thereby provides speculative decoding acceleration (1.5–2×) for diverse target models such as Vicuna-7B, Qwen2-7B, and Llama3-8B.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - Speculative decoding
  - cross-vocabulary
  - online distillation
  - adaptive drafting
  - on-device inference
date: 2026-05-08
content_hash: e4c55d97bad6832a
---

# OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2507.02659](https://arxiv.org/abs/2507.02659)
**Code**: Not available
**Area**: LLM Efficiency
**Keywords**: Speculative decoding, cross-vocabulary, online distillation, adaptive drafting, on-device inference

## TL;DR

This paper proposes the OmniDraft framework, which achieves cross-vocabulary speculative decoding via an online n-gram cache, aligns the draft model with the target model through a hybrid distillation loss, and dynamically adjusts the proposal length with an adaptive drafting head. A single lightweight Llama-68M model thereby provides speculative decoding acceleration (1.5–2×) for diverse target models such as Vicuna-7B, Qwen2-7B, and Llama3-8B.

## Background & Motivation

Speculative Decoding accelerates LLM inference by having a small draft model predict multiple future tokens, which a large target model then verifies in a single forward pass. Two fundamental challenges remain:

**Tight coupling between draft and target models**: Existing methods require the draft and target models to belong to the same model family (e.g., both from the Llama series), sharing the same tokenizer and vocabulary. Switching to a different family (e.g., Qwen) renders the draft model unusable.

**Dynamic demands in on-device deployment**: Users may switch between different target models at runtime and expect inference latency to improve over time.

Prior work UAG proposes a vocabulary-intersection mapping but handles only directly mapped tokens and cannot resolve "false rejections"—cases where multiple sub-tokens proposed by the draft model correspond to a single merged token in the target vocabulary and are therefore incorrectly rejected. Offline distillation-based alignment further assumes a fixed target model and cannot adapt to dynamic switching scenarios.

**Core Idea**: Establish a "one drafter for all" paradigm—resolving cross-vocabulary mapping via an n-gram cache, achieving dynamic alignment via online distillation, and controlling efficiency via adaptive drafting.

## Method

### Overall Architecture

OmniDraft consists of three core components: (1) a cross-vocabulary n-gram cache for token translation between draft and target vocabularies; (2) a hybrid distillation loss for online alignment of the draft model; and (3) an adaptive drafting head for dynamically adjusting the proposal length. The overall pipeline is: the draft model generates tokens → the n-gram cache translates them into the target vocabulary space → the target model verifies them → accept/reject outcomes are fed back to update the cache and distill the draft model.

### Key Designs

1. **Cross-Vocabulary N-gram Cache**: The core mechanism maintains a cache $\mathcal{C} = \{(t_i, [d_j^i]_{j=1:n})\}$ recording mappings between target tokens $t_i$ and draft token sequences $[d_1^i, d_2^i, \cdots, d_n^i]$. During the proposal phase, the draft token sequence is scanned and the n-gram cache is queried to perform merge mappings, with probabilities computed as:

    $q'(t_i) = \begin{cases} q(d_i), & \text{direct mapping} \\ \prod_j q(d_j^i), & \text{n-gram mapping} \end{cases}$

   For the residual distribution in the correction phase, $q'$ must be defined over the entire target vocabulary, requiring probability adjustment for prefix sub-tokens: $q'(d_1^i) = q(d_1^i) - \prod_j q(d_j^i)$, ensuring correct allocation of probability mass. The cache is updated online during inference whenever a previously unseen mapping instance is encountered. **Design Motivation**: Unlike UAG, which handles only vocabulary-intersection tokens, the n-gram cache accommodates merged tokens, thereby avoiding false rejections and improving the acceptance rate.

2. **Cross-Vocabulary Hybrid Distillation Loss**: Online distillation is decomposed into two parts—reverse KL divergence is applied to directly mapped tokens to obtain rich supervision signals, while negative log-likelihood (NLL) is applied to n-gram tokens because only reliable point probability estimates are available. The total loss is:

    $\mathcal{L}_{\text{cross\_vocab\_distill}}(\theta) = \mathcal{L}_{\text{DM}}(\theta) + \lambda \mathcal{L}_{\text{N-gram}}(\theta)$

   where $\mathcal{L}_{\text{DM}}$ computes KL divergence over directly mapped tokens and $\mathcal{L}_{\text{N-gram}}$ computes NLL over n-gram tokens. $\lambda$ can be set as a fixed hyperparameter or as a dynamic weight (e.g., the target model's verification probability for the n-gram). This design enables the draft model to continuously align with a potentially changing target model during online inference.

3. **Online Adaptive Drafting**: A lightweight head network $f_\phi$ predicts the acceptance rate of the current proposed token. Early termination of the proposal is triggered by monitoring the cumulative rejection probability:

    $P(\exists 1 \leq i \leq k, \text{s.t. } y_i \text{ rejected}) > \gamma \Rightarrow \text{exit}$

   Two training variants are proposed: **joint training** (distillation and adaptive head updated simultaneously) and **interleaved training** (the adaptive head is updated multiple times per single distillation update, with a larger buffer to mitigate distribution shift).

### Loss & Training

- Distillation uses on-policy data generated by the draft model itself.
- $\lambda = 0.2$ is fixed across all tasks and experiments.
- LoRA fine-tuning is supported as a lightweight alternative enabling dynamic adapter switching.
- The adaptive head is trained with a weighted BCE loss, using acceptance rate $\min(1, p/q)$ as labels.

## Key Experimental Results

### Main Results: Cross-Vocabulary Online Distillation

| Target Model | Method | GSM8K Acc/Speed | MBPP+HE Acc/Speed | Alpaca Acc/Speed | XSum Acc/Speed |
|---|---|---|---|---|---|
| Llama3-8B | SpD_DM (baseline) | 0.10 / 0.94× | 0.09 / 1.03× | 0.09 / 0.96× | 0.11 / 0.91× |
| Llama3-8B | $\mathcal{L}_{\text{DM}}$ + $\lambda\mathcal{L}_{\text{N-gram}}$ | **0.42 / 1.70×** | **0.27 / 1.33×** | **0.20 / 1.30×** | **0.24 / 1.24×** |
| Qwen2-7B | SpD_DM (baseline) | 0.14 / 1.04× | 0.09 / 0.91× | 0.13 / 1.01× | 0.12 / 0.96× |
| Qwen2-7B | $\mathcal{L}_{\text{DM}}$ + $\lambda\mathcal{L}_{\text{N-gram}}$ | **0.37 / 1.61×** | **0.26 / 1.36×** | **0.20 / 1.30×** | **0.22 / 1.22×** |

### Ablation Study: Adaptive Drafting (Vicuna-7B target)

| Method | GSM8K Acc/Speed | MBPP+HE Acc/Speed | Alpaca Acc/Speed | XSum Acc/Speed |
|---|---|---|---|---|
| SpD (vanilla) | 0.21 / 1.44× | 0.14 / 1.22× | 0.20 / 1.44× | 0.20 / 1.42× |
| Distill Only | 0.42 / 2.20× | 0.35 / 1.92× | 0.25 / 1.57× | 0.23 / 1.53× |
| Joint Distill+Adapt | **0.61 / 2.08×** | **0.51 / 1.91×** | **0.44 / 1.61×** | **0.42 / 1.59×** |
| Interleaved Distill+Adapt | 0.52 / **2.15×** | 0.48 / **1.94×** | 0.41 / 1.60× | 0.38 / 1.58× |

### Key Findings

- The n-gram cache yields significant gains even without training (cache hit rate 0.87), and combining it with distillation produces the best results.
- The cache footprint is small (1–5 MB), suitable for on-device deployment.
- The framework scales to larger target models (Qwen2.5-32B), achieving up to 2.05× speedup.
- The interleaved training variant achieves marginally higher speedup than joint training, while joint training attains a higher acceptance rate.
- LoRA fine-tuning performance is close to full-parameter fine-tuning, supporting dynamic switching across multiple target models.

## Highlights & Insights

- The "one drafter for all" paradigm is of significant practical value—only a single 68M draft model needs to be deployed on-device to serve all target LLMs.
- The n-gram cache is an elegant engineering design that reformulates cross-vocabulary mapping as an online cache lookup problem.
- The hybrid distillation loss applies different loss functions to directly mapped and n-gram tokens, reflecting a deep understanding of the problem structure.

## Limitations & Future Work

- Online adaptation performs only a single pass over the data stream, which may be unstable for entirely unseen data distributions.
- Cross-vocabulary mapping for special tokens (e.g., multimodal tokens) has not yet been addressed.
- Online training of the adaptive drafting head is insufficiently stable and may underestimate the optimal proposal length.
- The cache lacks an eviction policy; memory-constrained devices would require further optimization.

## Related Work & Insights

- Compared to UAG, the n-gram cache extends from vocabulary-intersection mapping to many-to-one mappings, resolving false rejections.
- Compared to OSD (Online Speculative Decoding), the proposed framework adds cross-vocabulary capability.
- Insight: In on-device inference scenarios, lightweight, general-purpose, and adaptable designs are more valuable than heavy but specialized solutions.

## Rating

- Novelty: ⭐⭐⭐⭐ The n-gram cache for cross-vocabulary mapping is a novel contribution, though the adaptive drafting component draws on SpecDec++.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple tasks and target models with complete ablations, though comparisons with additional baselines are lacking.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly presented, with detailed mathematical derivations and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ A universal on-device draft model addresses an important practical need; the framework is complete and deployment-ready.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](../../ACL2026/llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[NeurIPS 2025\] MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)
- [\[NeurIPS 2025\] Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding](yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)

<!-- RELATED:END -->
