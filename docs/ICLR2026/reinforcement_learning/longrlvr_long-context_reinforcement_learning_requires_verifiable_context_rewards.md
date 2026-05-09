---
title: >-
  [Paper Note] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards
description: >-
  [ICLR 2026][Reinforcement Learning][RLVR] This paper proposes LongRLVR, which introduces verifiable context rewards into RLVR training to address the gradient vanishing problem of contextual grounding caused by relying solely on final-answer rewards in long-context settings, significantly improving LLM long-context reasoning capabilities.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - RLVR
  - long-context reasoning
  - contextual grounding
  - verifiable rewards
  - gradient vanishing
  - GRPO
date: 2026-05-08
content_hash: 800e964adbeba6e7
---

# LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards

**Conference**: ICLR 2026
**arXiv**: [2603.02146](https://arxiv.org/abs/2603.02146)
**Code**: [real-absolute-AI/LongRLVR](https://github.com/real-absolute-AI/LongRLVR)
**Area**: Reinforcement Learning
**Keywords**: RLVR, long-context reasoning, contextual grounding, verifiable rewards, gradient vanishing, GRPO

## TL;DR

This paper proposes LongRLVR, which introduces verifiable context rewards into RLVR training to address the gradient vanishing problem of contextual grounding caused by relying solely on final-answer rewards in long-context settings, significantly improving LLM long-context reasoning capabilities.

## Background & Motivation

**RLVR fails in long-context settings**: RLVR (e.g., DeepSeek-R1) excels at reasoning tasks that rely on parametric knowledge such as math and coding, but performs poorly in long-context scenarios that require retrieving and reasoning over external documents.

**Contextual grounding is the core bottleneck**: Long-context reasoning requires accurately locating relevant evidence (contextual grounding) before generating an answer; the signal from final-answer-only rewards is too sparse to effectively guide the grounding process.

**Theoretical proof of gradient vanishing**: The authors theoretically demonstrate that outcome-only rewards cause the gradient of the grounding head to be scaled by the probability of an "activation event" $\Pr(\varepsilon_j)$—i.e., a positive gradient signal for selecting a specific evidence chunk is received only when all other necessary evidence chunks have already been selected, which is nearly impossible at the start of training.

**Empirical validation**: Under naive RLVR training, contextual recall stagnates rapidly, directly capping the achievable answer accuracy (Figure 1).

## Method

### Overall Architecture

The long-context RLVR policy is explicitly decomposed into two stages:
- **Grounding Head** $\pi_\theta^{gnd}(Z|X,Q)$: selects a relevant evidence subset $Z$ from context $X$
- **Answer Head** $\pi_\theta^{ans}(y|X,Q,Z)$: generates the final answer $y$ based on the selected evidence

During training, the model first generates a list of chunk identifiers (grounding), then generates the final answer.

### Verifiable Context Reward

Total reward = answer reward + context reward:

$$r_{total}(y,Z) = r_{ans}(y) + r_{ctx}(y,Z,G)$$

The context reward adopts a modulated F-score design:

$$r_{ctx}(y,Z,G) = \eta \cdot F_\beta(Z,G) + (1-\eta) \cdot r_{ans}(y) \cdot F_\beta(Z,G)$$

- **Unconditional grounding reward** $\eta \cdot F_\beta$: provides a stable dense learning signal for grounding
- **Synergistic success reward** $(1-\eta) \cdot r_{ans} \cdot F_\beta$: unlocks the full grounding reward only when the answer is correct, preventing grounding from decoupling from the final objective
- Hyperparameters: $\eta=0.1$, $\beta=2$ (recall-biased)

### Theoretical Guarantee (Proposition 2)

The gradient provided by the context reward for each ground-truth chunk $c_j$ contains a term $\alpha_j \cdot \mathrm{Var}(z_j)$, which does not depend on the probability of rare activation events, thereby eliminating gradient vanishing.

### Synthetic Data Pipeline

- Long documents of 8K–64K tokens are collected from book, arXiv, and code domains
- Semantic clustering is applied; Qwen3-235B generates candidate QA pairs and annotates grounding chunks for each cluster
- Two-stage rejection sampling (intra-cluster best → document best) with quality scores > 9/10
- A final dataset of 46K high-quality long-context QA instances is produced

## Key Experimental Results

### Main Results (Table 1)

| Model | RULER-QA (AVG) | LongBench v2 | LongReason (AVG) |
|------|:-:|:-:|:-:|
| Qwen2.5-14B-1M (base) | 75.20 | 40.2 | 73.55 |
| +RLVR | 73.17 | 39.8 | 72.33 |
| **+LongRLVR** | **88.90** | **46.5** | **78.42** |
| Qwen2.5-7B-1M (base) | 65.00 | 33.0 | 66.45 |
| +RLVR | 66.90 | 32.4 | 69.27 |
| **+LongRLVR** | **78.67** | **38.6** | **79.22** |
| LLaMA-3.1-8B (base) | 62.77 | 30.4 | 49.31 |
| +RLVR | 67.80 | 32.4 | 49.62 |
| **+LongRLVR** | **80.33** | **36.2** | **53.23** |

- Qwen2.5-14B-LongRLVR surpasses Qwen3-14B (RULER-QA 88.90 vs. 87.60) and QwenLong-L1-32B
- Qwen2.5-7B-LongRLVR substantially outperforms LLaMA-3.1-70B on LongReason (79.22 vs. 57.59)

### Ablation Study

| Ablation Dimension | Key Findings |
|---------|---------|
| **Reward components** (Figure 3) | Answer-only reward causes recall stagnation and a performance ceiling; context-only reward yields high recall but inaccurate answers; combining both is optimal |
| **Data quality** (Figure 4) | Rejection sampling best > median > worst (38.6 vs. 36.6 vs. 34.8); filtering easy questions is effective, while filtering hard questions is harmful |
| **Mixing factor $\eta$** (Figure 5a) | $\eta=0.1$ is optimal; $\eta=0$ gives too sparse an initial signal; $\eta=1$ decouples grounding from the answer objective |
| **F-score $\beta$** (Figure 5b) | $\beta=2$ is optimal; recall-biased scoring is critical for multi-evidence reasoning |
| **Number of chunks** (Figure 5c) | Performance is robust across 16–128 chunks, indicating the model learns semantic-level grounding rather than relying on chunking strategies |

## Highlights & Insights

- The fundamental failure of outcome-only RLVR in long-context settings is revealed from both theoretical (gradient vanishing proof) and empirical perspectives, with rigorous analysis.
- The context reward design is elegant: the modulated F-score simultaneously provides dense signals and maintains goal alignment, avoiding reward hacking.
- Small models (7B/14B) trained with LongRLVR surpass 70B+ models and even dedicated reasoning models (Qwen3-14B), demonstrating exceptional parameter efficiency.
- Robustness to the number of chunks indicates the model has acquired genuine semantic grounding ability.

## Limitations & Future Work

- Ground-truth grounding chunk annotations are required, relying on a high-quality synthetic data pipeline; generalization to unannotated settings remains unvalidated.
- Validation is limited to QA tasks; effectiveness on other long-context tasks such as summarization and information extraction is unknown.
- Training data lengths are restricted to 8K–64K tokens; scalability to longer contexts (e.g., 256K+) is not explored.
- The F-score reward assumes chunk-level annotations are available, which may be costly to obtain in practice.
- The theoretical analysis assumes independent chunk selection, whereas actual autoregressive generation in LLMs introduces dependencies among chunk selections.

## Related Work & Insights

- **RLVR reasoning enhancement**: DeepSeek-R1, Kimi, DAPO, etc.—this paper identifies their limitations in long-context settings.
- **Long-context alignment**: RoPE extensions (YaRN, LongRoPE), long-context SFT/DPO—this paper advances the RLVR direction.
- **QwenLong-L1-32B**: long-context RLVR based on reasoning models—LongRLVR achieves comparable performance with significantly smaller models.
- **Long-context agents**: chunk-based multi-turn collaborative approaches—orthogonal to this work and potentially complementary.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Theory-driven reward design is well-motivated; the gradient vanishing analysis is the core contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple models, multiple benchmarks, and extensive ablations with comprehensive data coverage.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theory and experiments are tightly connected with clear logical argumentation.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to long-context RLVR training, though synthetic annotated data are required.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[ICLR 2026\] Scalable In-Context Q-Learning](scalable_in-context_q-learning.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../NeurIPS2025/reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](loongrl_rl_for_reasoning_long_contexts.md)

</div>

<!-- RELATED:END -->
