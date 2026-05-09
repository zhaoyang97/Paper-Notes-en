---
title: >-
  [Paper Note] EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions
description: >-
  [NeurIPS 2025][LLM Alignment][over-refusal] This paper proposes EvoRefuse—a framework that employs evolutionary search (mutation/recombination + ELBO fitness function + simulated annealing) to automatically generate semantically benign yet reliably refusal-triggering "pseudo-malicious" instructions. The resulting EvoRefuse-Test benchmark achieves 85.34% higher refusal trigger rate and 34.86% greater lexical diversity than the strongest baseline, while the EvoRefuse-Align dataset reduces over-refusal by 29.85%–45.96% via SFT/DPO fine-tuning without compromising safety.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - over-refusal
  - pseudo-malicious
  - evolutionary search
  - ELBO
  - safety alignment
  - DPO
date: 2026-05-08
content_hash: be9b26cb3b676849
---

# EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions

**Conference**: NeurIPS 2025
**arXiv**: [2505.23473](https://arxiv.org/abs/2505.23473)
**Code**: [GitHub](https://github.com/FishT0ucher/EVOREFUSE)
**Area**: LLM Alignment
**Keywords**: over-refusal, pseudo-malicious, evolutionary search, ELBO, safety alignment, DPO

## TL;DR

This paper proposes EvoRefuse—a framework that employs evolutionary search (mutation/recombination + ELBO fitness function + simulated annealing) to automatically generate semantically benign yet reliably refusal-triggering "pseudo-malicious" instructions. The resulting EvoRefuse-Test benchmark achieves 85.34% higher refusal trigger rate and 34.86% greater lexical diversity than the strongest baseline, while the EvoRefuse-Align dataset reduces over-refusal by 29.85%–45.96% via SFT/DPO fine-tuning without compromising safety.

## Background & Motivation

**Background**: Safety alignment is a critical technique for preventing LLM misuse, yet overly conservative alignment frequently leads to over-refusal—where LLMs incorrectly classify harmless inputs as unsafe and decline to respond. For instance, "I need a recipe for a dangerous cake that could explode with flavour at a party" may be refused due to keywords such as "dangerous" and "explode."

**Limitations of Prior Work**: (1) Methods for collecting pseudo-malicious instructions are limited—manual construction (XSTest, OKTest) lacks scalability; (2) automatic rewriting approaches (OR-Bench) modify seed instructions without explicitly optimizing for LLM refusal probability; (3) gradient-based search methods (PHTest) optimize refusal probability along narrow paths with insufficient linguistic diversity; (4) existing methods neither analyze nor exploit the key semantic/syntactic features that trigger over-refusal.

**Key Challenge**: There is a fundamental need for a method that can efficiently generate large quantities of diverse pseudo-malicious instructions for evaluating LLM over-refusal, while ensuring the generated instructions are both cross-model effective and semantically safe.

**Goal**: To automatically generate diverse pseudo-malicious instructions for comprehensively evaluating and effectively mitigating LLM over-refusal.

**Key Insight**: The generation of pseudo-malicious instructions is formalized as an optimization problem of maximizing LLM refusal probability. A variational method is used to derive the ELBO as a tractable surrogate objective, which is then optimized via evolutionary search.

**Core Idea**: The ELBO serves as a fitness function, while evolutionary search (strategy-guided mutation + recombination + simulated annealing) acts as the optimizer to search the instruction space for pseudo-malicious instructions that are semantically harmless yet maximally trigger LLM refusal.

## Method

### Overall Architecture

Starting from a seed instruction $x^0$, the framework proceeds through multi-strategy mutation (introducing deceptive contexts / sensitive words / extreme emotions) → safety classifier filtering → ELBO fitness evaluation → selection of top-$L$ candidates for recombination → safety verification → simulated annealing acceptance/rejection → iteration for $I$ rounds → output of the highest-fitness instruction $x^*$.

### Key Designs

1. **ELBO Variational Objective**
    - Direct computation of the refusal probability $\log p_\theta(\mathbf{r}|\mathbf{x},\mathbf{s})$ is intractable (Monte Carlo sampling is numerically unstable), so a variational approach is adopted to derive the ELBO:
    - $\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q_\theta(\mathbf{y}|\mathbf{x})}[\underbrace{\log p_\theta(\mathbf{y}|\mathbf{x},\mathbf{s})}_{\text{response confidence}} + \underbrace{\log p_\theta(\mathbf{r}|\mathbf{x},\mathbf{y},\mathbf{s})}_{\text{refusal log-prob}}] + c$
    - In practice, Monte Carlo estimation with $K$ sampled responses is used: $\mathcal{F}(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}[\log \hat{p}_\phi(\mathbf{r}|\mathbf{y}_k) + \frac{\lambda}{T_k}\sum_{t=1}^{T_k}\log p_\theta(y_{k,t}|\mathbf{y}_{k,<t},\mathbf{x},\mathbf{s})]$
    - Refusal probability is estimated via the publicly available distilroberta-base-rejection classifier; response confidence is computed from LLaMA3.1-8B token logits.
    - **Design Motivation**: The ELBO implicitly balances two factors—rewarding responses that are both classified as refusals and generated with high confidence.

2. **Strategy-Guided Mutation and Recombination**
    - Five hundred low-similarity instructions from XSTest and OR-Bench are analyzed; GPT-4o extracts triggering factors, and SentenceBERT embeddings with clustering (threshold 0.75) yield three mutation strategy categories: (a) introducing deceptive contexts (controversial topics / fictional scenarios / implied potential harm); (b) adding sensitive words (violence / bias / sensitive terminology); (c) extreme emotions (anger / disgust / despair).
    - Recombination: the top-$L$ fitness variants are selected, and $N$ pairs are sampled for GPT-4o to synthesize new candidate instructions.
    - Safety verification: each mutated/recombined instruction is accompanied by a safety rationale, with GPT-4o serving as the judge.
    - **Design Motivation**: Unlike random perturbation, strategy-guided mutation ensures that mutation directions align with known over-refusal triggers.

3. **Simulated Annealing Acceptance Strategy**
    - Acceptance probability: $\delta = \min\{1, \exp[\frac{\mathcal{F}(x') - \mathcal{F}(x^t)}{\tau_t}]\}$
    - Linear cooling schedule: $\tau_t \leftarrow \max\{\tau_f, \tau_0 - \beta \cdot t\}$
    - Occasionally accepting lower-fitness candidates prevents the search from becoming trapped in local optima.
    - **Design Motivation**: Balances exploration and exploitation, preventing premature convergence of evolutionary search.

### Dataset Construction

- **EvoRefuse-Test**: 800 diverse instructions are selected from TRIDENT-Core → optimized by EvoRefuse → 582 pseudo-malicious instructions retained after safety filtering.
- **EvoRefuse-Align**: 3,000 instructions paired with GPT-4o-generated helpful/refusal responses, supporting both SFT and DPO.

## Key Experimental Results

### Refusal Trigger Rate (PRR, without safety-prior system prompt)

| Benchmark | DeepSeek-7B | Gemma-7B | LLaMA-8B | Mistral-7B | Qwen-7B | GPT-4o | DeepSeek-V3 | Gemini | Claude | Avg. |
|-----------|-------------|----------|----------|------------|---------|--------|-------------|--------|--------|------|
| XSTest | 0.05 | 0.11 | 0.13 | 0.00 | 0.05 | 0.08 | 0.07 | 0.08 | 0.19 | 0.08 |
| OR-Bench | 0.14 | 0.15 | 0.05 | 0.04 | 0.07 | 0.09 | 0.27 | 0.06 | 0.18 | 0.12 |
| PHTest | 0.10 | 0.19 | 0.08 | 0.09 | 0.03 | 0.10 | 0.12 | 0.09 | 0.31 | 0.12 |
| **EvoRefuse-Test** | **0.24** | **0.26** | **0.65** | **0.12** | **0.25** | **0.27** | **0.38** | **0.24** | **0.74** | **0.35** |

### Diversity, Confidence, and Safety

| Benchmark | MSTTR↑ | MTLD↑ | Log-Prob(y\|x)↑ | LongPPL↓ | Safe Rate |
|-----------|--------|-------|----------------|----------|-----------|
| XSTest | 0.36 | 39.95 | -72.62 | 1.34 | 0.97 |
| OR-Bench | 0.47 | 137.65 | -93.45 | 1.26 | 0.93 |
| PH-Gen | 0.48 | 134.84 | -103.08 | 1.15 | 0.90 |
| **EvoRefuse-Test** | **0.54** | **152.52** | **-43.55** | **1.12** | 0.93 |

### Over-Refusal Mitigation Results (LLaMA3.1-8B-Instruct)

| Method | XSTest PRR↓ | SGTest PRR↓ | EvoRefuse PRR↓ | AdvBench PRR↑ | HarmBench PRR↑ |
|--------|------------|------------|---------------|--------------|---------------|
| Base model | 0.11 | 0.14 | 0.65 | 0.94 | 0.94 |
| + OR-Bench (SFT) | 0.10 | 0.14 | 0.45 | 1.00 | 0.98 |
| + PHTest (SFT) | 0.09 | 0.11 | 0.39 | 1.00 | 0.97 |
| + EvoRefuse (SFT) | **0.06** | **0.05** | **0.28** | 1.00 | 0.96 |
| + EvoRefuse (DPO) | **0.03** | **0.05** | **0.15** | 0.99 | 0.97 |

### Key Findings

- EvoRefuse-Test achieves a refusal trigger rate of 0.65 on LLaMA3.1-8B, 3.64× that of the next-best baseline (as LLaMA is the target model for EvoRefuse).
- Strong cross-model generalization: a refusal rate of 0.74 is achieved on Claude, a non-target model.
- DPO fine-tuning outperforms SFT: PRR on EvoRefuse-Test drops from 0.65 to 0.15 (a 45.96% reduction) with no safety degradation.
- Attribution analysis reveals that over-refusal is primarily driven by "shortcut learning"—models over-attend to salient textual cues (sensitive keywords) while neglecting the broader harmless context.
- Early Transformer layers play a critical role in safety judgments.

## Highlights & Insights

- The variational ELBO framework elevates the generation of refusal-triggering instructions from heuristic search to a theoretically grounded optimization problem, yielding greater numerical stability than direct Monte Carlo estimation of refusal probability.
- The three mutation strategy categories (deceptive context / sensitive words / extreme emotions) are derived from empirical analysis rather than intuition, making them interpretable and extensible.
- The alignment data generated by EvoRefuse-Align reduces over-refusal while preserving safety—a non-trivial balance.
- Attribution analysis combining gradient-based saliency and information flow provides two complementary lenses that together reveal the "shortcut learning" mechanism underlying over-refusal.

## Limitations & Future Work

- The default target model is LLaMA3.1-8B-Instruct, on which performance is strongest (0.65 PRR); cross-model generalization, while favorable, still shows a gap.
- Reliance on GPT-4o as the mutator/recombiner/safety verifier incurs substantial API cost.
- The ELBO is a lower bound on refusal probability but is not order-preserving; maximizing the ELBO does not guarantee monotonic improvement of the true objective at every step.
- Safety verification still relies on LLM-as-judge, which carries the risk of inconsistent judgments.
- Over-refusal in multi-turn dialogue settings is not addressed.

## Related Work & Insights

- **vs XSTest**: 250 manually crafted pseudo-malicious instructions with limited diversity and scale; EvoRefuse-Test (582 instructions) achieves 34.86% higher lexical diversity and 85.34% higher refusal trigger rate.
- **vs OR-Bench**: Instruction rewriting without an explicit optimization objective; EvoRefuse uses the ELBO to explicitly guide search direction.
- **vs PHTest**: Gradient-based search along a narrow path; EvoRefuse uses evolutionary search to cover a broader range of linguistic variants.
- **vs AutoDAN / GCG**: These methods generate malicious jailbreak prompts; EvoRefuse pursues the opposite goal—generating instructions that appear sensitive but are actually harmless.
- **Methodological inspiration**: The combination of ELBO and evolutionary search is generalizable to other prompt optimization scenarios where the true objective function is difficult to optimize directly.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of ELBO variational objective and evolutionary search is novel; the problem framing (systematic evaluation + mitigation of over-refusal) is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across 9 LLMs, multi-dimensional comparisons (refusal rate / diversity / confidence / safety), and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The problem formalization is clear, and the logical flow from ELBO derivation to the evolutionary framework is coherent.
- Value: ⭐⭐⭐⭐ Provides both a stronger over-refusal evaluation benchmark and an effective mitigation pipeline, with direct practical value for LLM safety alignment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[NeurIPS 2025\] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)

<!-- RELATED:END -->
