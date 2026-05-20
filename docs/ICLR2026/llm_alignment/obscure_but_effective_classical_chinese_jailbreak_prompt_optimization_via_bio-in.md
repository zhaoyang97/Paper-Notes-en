---
title: >-
  [Paper Note] Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search
description: >-
  [ICLR 2026][LLM Alignment][LLM Safety] This paper proposes CC-BOS, a framework that exploits the semantic compression and inherent ambiguity of Classical Chinese…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "LLM Safety"
  - "Jailbreak Attack"
  - "Classical Chinese"
  - "Bio-Inspired Optimization"
  - "Black-Box Attack"
date: 2026-05-08
content_hash: 979c88d34883efb0
---

# Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search

**Conference**: ICLR 2026
**arXiv**: [2602.22983](https://arxiv.org/abs/2602.22983)  
**Code**: None  
**Area**: LLM Alignment
**Keywords**: LLM Safety, Jailbreak Attack, Classical Chinese, Bio-Inspired Optimization, Black-Box Attack

## TL;DR

This paper proposes CC-BOS, a framework that exploits the semantic compression and inherent ambiguity of Classical Chinese, combined with a Fruit Fly Optimization Algorithm to search an eight-dimensional strategy space for optimal jailbreak prompts, achieving nearly 100% attack success rate across six mainstream LLMs.

## Background & Motivation

The safety alignment mechanisms of LLMs exhibit uneven performance across different linguistic contexts. Low-resource languages are more prone to eliciting unsafe outputs due to insufficient training data. This paper is the first to systematically investigate the role of **Classical Chinese** in jailbreak attacks:
- Classical Chinese features semantic compression, rich rhetorical devices, and inherent polysemy.
- Its expression diverges substantially from Modern Chinese, enabling partial bypass of keyword- and template-matching-based defenses.
- While models can adequately comprehend Classical Chinese input, safety guardrails optimized for modern languages fail to detect malicious intent embedded within it.

Core insight: The safety vulnerability of Classical Chinese does not stem from insufficient data coverage, but rather constitutes a **security blind spot**.

## Method

### Overall Architecture

CC-BOS comprises three core components: (1) an eight-dimensional strategy space defining the generative dimensions of jailbreak prompts; (2) a bio-inspired Fruit Fly Optimization Algorithm (FOA) that searches the strategy space for optimal configurations; and (3) a two-stage translation module that converts Classical Chinese responses into English to ensure evaluation accuracy.

### Key Designs

1. **Eight-Dimensional Strategy Space $\mathbb{S} = D_1 \times D_2 \times \cdots \times D_8$**:

    - $D_1$: Role identity (e.g., ancient scholar, strategist)
    - $D_2$: Behavioral guidance
    - $D_3$: Mechanism (e.g., cascaded reasoning)
    - $D_4$: Metaphorical mapping
    - $D_5$: Expressive style (e.g., parallel prose, free prose)
    - $D_6$: Knowledge association
    - $D_7$: Trigger pattern
    - $D_8$: Contextual setting
    - Given an original query $q_0$ and strategy $\mathbf{s}$, the prompt generator $G$ produces an adversarial query $q = G(q_0; \mathbf{s})$

2. **Fruit Fly Optimization Algorithm (FOA)**:

    - **Olfactory search**: Adaptive local perturbation with step size decaying over iterations as $\Delta_t = \max(1, \lfloor \alpha |D_i| \cdot \gamma^t \rfloor)$
    - **Visual search**: Attraction toward the global optimum with attraction probability $\beta_t = \beta_0 + (1-\beta_0) \cdot t/N$
    - **Cauchy mutation**: Leverages the heavy-tailed property of the Cauchy distribution to escape local optima
    - Hash-based deduplication and early stopping are introduced to improve search efficiency

3. **Fitness Evaluation $F(\mathbf{s}) = S_c(\mathbf{s}) + S_k(\mathbf{s})$**:

    - Consistency score $S_c$: Scored by an evaluation model (0–100), measuring the alignment between the response and the malicious instruction
    - Keyword score $S_k$: Detects the presence of rejection keywords (0 or 20 points)
    - Total fitness range: $[0, 120]$

### Loss & Training

- DeepSeek-Chat is used as both the attack model and the translation model.
- Initial population size is 5; maximum number of iterations is 5.
- A fitness threshold of 80 is used to determine jailbreak success.
- Two-stage translation module: Classical Chinese → Modern Chinese → English.

## Key Experimental Results

### Main Results (AdvBench Dataset)

| Target Model | CC-BOS ASR | CC-BOS Avg. Score | ICRT ASR | ICRT Avg. Score |
|---|---|---|---|---|
| Gemini-2.5-flash | **100%** | 4.82 | 92% | 4.52 |
| Claude-3.7 | **100%** | 3.14 | 40% | 1.60 |
| GPT-4o | **100%** | 4.74 | 74% | 3.06 |
| DeepSeek-Reasoner | **100%** | 4.84 | 88% | 4.00 |
| Qwen3-235B | **100%** | 4.88 | 84% | 4.00 |
| Grok-3 | **100%** | 4.76 | 98% | 4.30 |

### Efficiency Comparison (Average Query Count Avg.Q)

| Method | Gemini | Claude | GPT-4o | DeepSeek | Qwen3 | Grok-3 |
|---|---|---|---|---|---|---|
| CC-BOS | **1.46** | **2.38** | **1.28** | **1.12** | **1.54** | **1.18** |
| CL-GSO | 3.62 | 21.42 | 4.00 | 3.26 | 5.06 | 1.24 |
| PAIR | 60.00 | 51.12 | 57.36 | 40.32 | 57.00 | 51.36 |

### Key Findings

- CC-BOS achieves 100% ASR on all six target models, substantially outperforming all baselines.
- The average number of queries is only 1–2, far exceeding the efficiency of competing methods.
- Near-100% ASR is also maintained on the CLAS and StrongREJECT datasets.
- CC-BOS retains high success rates even against the Llama-Guard-3-8B defense.

## Highlights & Insights

- This is the first systematic investigation of Classical Chinese in LLM security evaluation, opening a new research direction.
- The formalized eight-dimensional strategy space enables comprehensive coverage of attack vectors.
- The three-phase FOA search strategy—olfactory, visual, and Cauchy mutation—efficiently balances exploration and exploitation.
- The extremely low query count suggests that the Classical Chinese context itself possesses strong bypass capability.

## Limitations & Future Work

- The Classical Chinese attack relies on the model's comprehension of Classical Chinese; effectiveness may diminish for models with very limited exposure to such data.
- The selection of dimensions in the eight-dimensional strategy space depends on human expertise.
- Defensive countermeasures, such as incorporating Classical Chinese safety data during training, are relatively straightforward to implement.
- The paper focuses on attack capability without providing an in-depth discussion of defensive strategies.

## Related Work & Insights

- Compared to CL-GSO: CC-BOS leverages Classical Chinese context rather than strategy decomposition in modern English.
- Compared to white-box methods such as GCG: CC-BOS is fully black-box and requires no gradient information.
- Implication: LLM safety alignment must extend coverage to historical languages and specialized linguistic contexts.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Classical Chinese jailbreak attacks represent an entirely novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six models, three datasets, and multi-dimensional comparisons.
- Writing Quality: ⭐⭐⭐⭐ Methods are clearly described with a high degree of mathematical formalization.
- Value: ⭐⭐⭐⭐ Carries significant implications for LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)

</div>

<!-- RELATED:END -->
