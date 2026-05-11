---
title: >-
  [Paper Note] The Impact of Quantization on Large Reasoning Model Reinforcement Learning
description: >-
  [NeurIPS 2025 (Workshop: Efficient Reasoning)][LLM Reasoning][Quantization] This paper presents a systematic empirical study showing that quantization-aware fine-tuning (QAFT/STE) during RL training of large reasoning mo…
tags:
  - "NeurIPS 2025 (Workshop: Efficient Reasoning)"
  - "LLM Reasoning"
  - "Quantization"
  - "Reasoning Models"
  - "Reinforcement Learning"
  - "GRPO"
  - "QLoRA"
date: 2026-05-08
content_hash: d915624dcb9badc8
---

# The Impact of Quantization on Large Reasoning Model Reinforcement Learning

**Conference**: NeurIPS 2025 (Workshop: Efficient Reasoning)
**arXiv**: [2511.15694](https://arxiv.org/abs/2511.15694)
**Code**: [https://github.com/d-matrix-ai/rlquant](https://github.com/d-matrix-ai/rlquant)
**Area**: LLM Reasoning
**Keywords**: Quantization, Reasoning Models, Reinforcement Learning, GRPO, QLoRA

## TL;DR
This paper presents a systematic empirical study showing that quantization-aware fine-tuning (QAFT/STE) during RL training of large reasoning models (LRMs) degrades reasoning capability, whereas post-training quantization (PTQ) and QLoRA preserve reasoning performance well even at 4-bit precision. The authors recommend a practical pipeline of full-precision RL training followed by PTQ quantization.

## Background & Motivation

**Background**: Large reasoning models (LRMs) such as DeepSeek-R1 acquire strong reasoning capabilities on verifiable tasks like mathematics through RL, particularly via the GRPO algorithm. Quantization is a standard deployment technique for LLMs, but PTQ and QAT have primarily been studied in the SFT setting.

**Limitations of Prior Work**: RL training differs fundamentally from SFT — RL relies on discrete reward signals derived from model-sampled rollouts to update the policy, and the effect of quantization-induced noise on this process is entirely unknown. In practice, large numbers of LRM agents are derived from shared base models, specialized to different tasks via RL, and ultimately require quantized deployment.

**Key Challenge**: When is quantization best applied? During RL training (QAFT/QLoRA) or after training is complete (PTQ)? How do the two approaches trade off reasoning performance against memory efficiency?

**Goal**: To systematically evaluate the impact of different quantization strategies (QAFT-STE, QLoRA, PTQ-AWQ, PTQ-BnB) at various precisions (4/8-bit) and model scales (0.6B–8B) on the effectiveness of RL training for LRMs.

**Key Insight**: The Qwen3 model family (0.6B/1.7B/4B/8B) is trained on the MATH dataset using GRPO/drGRPO, with various quantization configurations compared.

**Core Idea**: Sudden quantization shock during the RL process degrades policy learning; deferring quantization to after training (via PTQ or QLoRA with frozen base weights) is a superior strategy.

## Method

### Overall Architecture

Base model → RL training (GRPO/drGRPO) → optional quantization → mathematical reasoning evaluation. Quantization can be applied at three stages: (1) quantization-aware throughout training (QAFT-STE), (2) QLoRA during training (frozen quantized base model with trainable low-rank adapters), or (3) post-training quantization (PTQ).

### Key Designs

1. **QAFT with 8-bit STE**:

    - **Function**: During every forward pass of RL training, linear weights in attention layers are quantized to INT8 RTN, with the straight-through estimator (STE) used to approximate backward gradients.
    - **Mechanism**: The simplest form of quantization-aware training, where weights always participate in computation at quantized precision.
    - **Problem**: Quantization noise permeates the entire RL process, producing worse policies → lower-quality rollouts → degraded reward signal quality → a vicious cycle.

2. **QLoRA Training**:

    - **Function**: The base model is frozen under NF4 quantization, with only low-rank adapter matrices trained (rank=8, α=16); adapters are merged back into the base model after training.
    - **Mechanism**: Quantized weights do not participate in gradient updates; only full-precision adapter parameters are optimized.
    - **Key Hyperparameter**: A higher learning rate ($10^{-4}$ vs. $10^{-6}$ for GRPO) is required to compensate for quantization noise.

3. **PTQ via AWQ and BitsAndBytes**:

    - **Function**: RL training is conducted entirely in full precision; the model is subsequently quantized to 4/8-bit using AWQ (data-calibrated) or BnB (data-free).
    - **Mechanism**: Policy learning during the RL phase is completely shielded from quantization interference; quantization error is incurred only at inference-time deployment.

### Loss & Training
- Base models: Qwen3-0.6B/1.7B/4B/8B
- RL algorithms: GRPO and drGRPO
- Training data: MATH Level 3–5, 10,000 samples, 1 epoch, learning rate $10^{-6}$
- Reward: correct answer 1.0 + correct format 0.1
- Evaluation: AIME2024, AMC, MATH500, Minerva Math, OlympiadBench

## Key Experimental Results

### Main Results (Mean Evaluation Reward)

| Method | 0.6B | 1.7B | 4B | 8B |
|--------|------|------|-----|-----|
| Base (full precision) | 0.164 | 0.212 | 0.451 | 0.473 |
| GRPO (full precision) | 0.307 | 0.418 | 0.555 | 0.594 |
| **QAFT STE 8-bit** | 0.242 | 0.325 | 0.443 | 0.496 |
| PTQ BnB 8-bit | 0.222 | 0.366 | 0.528 | 0.579 |
| PTQ AWQ 8-bit | 0.220 | 0.364 | 0.526 | 0.583 |
| **QLoRA 4-bit** | 0.240 | 0.382 | **0.554** | 0.556 |
| PTQ BnB 4-bit | 0.223 | 0.369 | 0.527 | 0.581 |
| PTQ AWQ 4-bit | 0.225 | 0.366 | 0.533 | 0.574 |

### Key Comparisons

| Quantization Strategy | vs. Full-Precision GRPO | Notes |
|----------------------|------------------------|-------|
| QAFT STE 8-bit | Significant drop (8B: 0.496 vs. 0.594) | Underperforms even the base model without RL (4B) |
| PTQ 4-bit | Marginal drop (8B: 0.574–0.581 vs. 0.594) | Retains ~97% of performance even at 4-bit |
| QLoRA 4-bit | Near-lossless at 4B; some degradation at 8B | Best cost-effectiveness at medium scale |

### Key Findings
- **QAFT is the worst choice**: Across all models larger than 0.6B, STE 8-bit performance falls below that of full-precision GRPO followed by 4-bit PTQ, indicating that quantization noise during RL is far more destructive than quantization error at inference time.
- **PTQ 4-bit ≈ PTQ 8-bit**: For larger models (4B/8B), the performance gap between 4-bit and 8-bit PTQ is negligible (<1%), demonstrating strong robustness to PTQ in models trained with full-precision RL.
- **Generation length matters**: 4B/8B models fail to learn adequately under a 512-token limit but improve substantially with 1,024 tokens (4B: 0.487 → 0.555), underscoring that reasoning models require sufficient "thinking space."
- **Quantized large models outperform smaller full-precision models**: Quantized large models dominate on the Pareto frontier (e.g., 4-bit 8B > full-precision 4B).

## Highlights & Insights
- **Fills a Gap**: This is the first systematic study of quantization effects on RL training of LRMs; this intersection was previously unexplored.
- **Clear Practical Guidance**: The paper provides an unambiguous recommendation — perform RL in full precision, then apply PTQ or QLoRA quantization, and avoid QAFT.
- **"Quantization Shock" Hypothesis**: RL is more sensitive to quantization noise than SFT because RL depends on sampling quality; policy degradation induced by quantization propagates through rollout quality to the reward signal.

## Limitations & Future Work
- **Workshop paper with limited experimental scope**: Only one model family (Qwen3) and one task domain (mathematical reasoning) are evaluated; generalizability remains unverified.
- **Pre-training QAT followed by RL unexplored**: Introducing quantization awareness during pre-training before RL may yield different outcomes.
- **Anomalous QLoRA behavior at 8B**: QLoRA 4-bit at 8B scale (0.556) underperforms PTQ 4-bit (0.581); the underlying cause is not analyzed in depth.
- **Absence of intermediate diagnostics**: Only final reward is reported; the stage and token types most affected by quantization remain unclear.
- **More aggressive quantization not tested**: Strategies such as 2-bit or mixed-precision quantization are not evaluated.

## Related Work & Insights
- **vs. Traditional QAT/PTQ Research**: In the SFT setting, QAT typically outperforms PTQ; this paper finds the opposite pattern in the RL setting, suggesting that the RL sampling-reward loop introduces a unique vulnerability to quantization noise.
- **vs. QLoRA (Dettmers et al.)**: The original QLoRA was designed for SFT; this paper validates its effectiveness in the RL setting and notes that the "isolation" effect of low-rank adapters — whereby quantized base weights are frozen — incidentally protects the RL learning process.

## Rating
- **Novelty**: ⭐⭐⭐ The problem is novel (first study of quantization in RL-trained LRMs), but the methodology combines existing techniques without proposing new methods.
- **Experimental Thoroughness**: ⭐⭐⭐ The matrix experimental design across 4 model scales and multiple quantization schemes is sound, but evaluation is limited to mathematical reasoning.
- **Writing Quality**: ⭐⭐⭐⭐ The workshop paper is concise, clear, and presents conclusions explicitly.
- **Value**: ⭐⭐⭐⭐ The findings offer direct practical guidance for LRM deployment; the conclusion that QAFT is the worst-performing strategy is both surprising and significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [\[NeurIPS 2025\] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning](expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
