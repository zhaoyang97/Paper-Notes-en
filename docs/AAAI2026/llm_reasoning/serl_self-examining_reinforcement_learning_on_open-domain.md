---
title: >-
  [Paper Note] SERL: Self-Examining Reinforcement Learning on Open-Domain
description: >-
  [AAAI 2026][Reasoning][Self-improvement] This paper proposes SERL, a self-improvement framework in which an LLM simultaneously acts as an Actor (generator) and a Judge (evaluator). It derives reward signals from the model's own judgments via the Copeland pairwise comparison method, requiring neither external reward models nor human annotations. SERL improves Qwen3-8B from 52.37% to 59.90% (+7.53%) on AlpacaEval 2.0, approaching the performance of Qwen3-32B.
tags:
  - "AAAI 2026"
  - "Reasoning"
  - "Self-improvement"
  - "Reinforcement Learning"
  - "Pairwise Comparison"
  - "Copeland Method"
  - "External-Reward-Free"
date: 2026-05-08
content_hash: c6b69917c6e1ad3c
---

# SERL: Self-Examining Reinforcement Learning on Open-Domain

**Conference**: AAAI 2026
**arXiv**: [2511.07922](https://arxiv.org/abs/2511.07922)  
**Code**: [GitHub](https://github.com/AlwaysOu/SERL)  
**Area**: LLM Reasoning / Self-Improvement
**Keywords**: Self-improvement, Reinforcement Learning, Pairwise Comparison, Copeland Method, External-Reward-Free

## TL;DR
This paper proposes SERL, a self-improvement framework in which an LLM simultaneously acts as an Actor (generator) and a Judge (evaluator). It derives reward signals from the model's own judgments via the Copeland pairwise comparison method, requiring neither external reward models nor human annotations. SERL improves Qwen3-8B from 52.37% to 59.90% (+7.53%) on AlpacaEval 2.0, approaching the performance of Qwen3-32B.

## Background & Motivation
**Background**: LLM self-improvement is a promising direction for reducing dependence on external annotations, but the quality of self-evaluation reward signals remains a critical bottleneck.

**Limitations of Prior Work**: (a) Self-evaluation is prone to preference cycles (A>B>C>A); (b) positional bias and length bias degrade judgment quality; (c) reward derivation methods lack theoretical guarantees.

**Key Challenge**: Self-evaluation is inherently subjective and inconsistent, yet external evaluation requires additional resources.

**Goal**: Derive high-quality training rewards from the model's own pairwise comparison judgments.

**Key Insight**: The Copeland method (from voting theory) is applied to resolve preference cycles, combined with dual rewards (actor + judge) for joint optimization.

**Core Idea**: Copeland pairwise comparison for reward derivation + actor/judge joint optimization = external-dependency-free self-improvement.

## Method

Training pipeline: For each input, the Actor samples $N$ candidate responses → the Judge performs pairwise comparisons over all response pairs → Copeland aggregation yields dual rewards → GRPO performs online updates.

### Key Designs
1. **Copeland Reward Derivation (Actor Reward $\mathcal{R}_A$)**:

    - For each input, $N$ candidate responses are sampled, and all $\binom{N}{2}$ pairs are compared.
    - $K$ independent judgments are sampled per pair and aggregated into a win-rate ranking via the Copeland method.
    - $\mathcal{R}_A(G_n) = \sum_{i\neq j,k} \mathbf{1}(G_n = G^{Win}_{(i,j),k}) / (M \times K)$
    - The win rate directly reflects each response's relative quality ranking within the group, making it more robust than point-wise scoring.

2. **Judge Consistency Reward ($\mathcal{R}_J$)**:

    - Measures the consistency between an individual pairwise judgment and the global Copeland ranking.
    - $\mathcal{R}_J(J_{(i,j),k}) = \text{sign}(\mathcal{R}_A(G^{Win}) - \mathcal{R}_A(G^{Lose}))$
    - Consistent judgments receive +1 and contradictory ones receive −1, compelling the Judge to learn more coherent evaluation criteria.

3. **Position Bias Mitigation Mechanism (PBMM)**:

    - Among the $K$ comparisons, half are presented in the order $(q, G_i, G_j)$ and the other half in the order $(q, G_j, G_i)$.
    - This eliminates the common positional preference (i.e., favoring responses appearing first or last) in LLM-as-Judge settings.

4. **Length Control Module (LCM)**:

    - Introduces a length ratio weight $\beta = |G^{Lose}|/|G^{Win}|$, granting higher rewards when a shorter response wins.
    - A hyperparameter $\alpha=0.2$ restricts comparisons to pairs with similar lengths.
    - Prevents the model from learning a spurious "longer is better" strategy.

### Loss & Training
- Built on the GRPO framework with the KL penalty term removed (in open-domain training, large distributional shifts make KL constraints overly restrictive for exploration).
- Advantage values for both Actor and Judge are computed via within-group normalization: $\hat{A}^{Actor} = (\mathcal{R}_A - \text{mean}) / \text{std}$
- Joint optimization objective $\mathcal{J}_{SERL} = \mathcal{J}_{Actor} + \mathcal{J}_{Judge}$ simultaneously updates generation and evaluation capabilities at each step.

## Key Experimental Results

### General QA (AlpacaEval 2.0)

| Method | LC Win Rate | Win Rate | Avg. Length |
|------|------------|---------|----------|
| Online-DPO | 54.07% | 59.74% | 3429 |
| Self-Rewarding | 51.29% | 53.69% | 3074 |
| Meta-Rewarding | 54.73% | 55.93% | 3081 |
| RLSC | 52.11% | 51.81% | 2060 |
| **SERL (Ours)** | **59.90%** | **69.88%** | **3017** |

### Summarization & Writing Tasks (Win Rate vs. Baselines)

| Comparison | Summarization Win Rate | Writing Win Rate |
|---------|-------------|-------------|
| vs Online-DPO | 55.17% (+10.33%) | 50.50% (+1.00%) |
| vs Self-Rewarding | 59.50% (+19.00%) | 55.17% (+10.33%) |
| vs Meta-Rewarding | 59.17% (+18.33%) | 56.67% (+13.33%) |
| vs RLSC | 86.17% (+72.33%) | — |

### Key Findings
- An 8B model trained with SERL achieves 59.90% LC Win Rate, approaching Qwen3-32B (~60%)—self-improvement bridges a 4× scale gap.
- The Copeland method effectively resolves preference cycles, exhibiting substantially greater robustness than point-wise self-evaluation (Self-Rewarding: 51.29%).
- Joint Actor+Judge optimization yields simultaneous improvements in both capabilities, forming a virtuous positive feedback loop.
- The advantage is most pronounced on summarization: a 59.50% win rate over Self-Rewarding (+19%) and 86.17% over RLSC.
- SERL's output length (3,017 tokens) is shorter than Online-DPO's (3,429), demonstrating that quality gains are not attributable to verbosity.
- Substantial improvements are achieved within tens of training steps, indicating extremely high training efficiency and accessibility for resource-constrained teams.
- Training remains stable after removing the KL penalty, suggesting that KL constraints may be unnecessary for open-domain tasks.

## Highlights & Insights
- **Genuine self-improvement without external dependencies**: No reward model, no human annotations, and no stronger LLM for evaluation are required—a fully self-driven closed-loop training paradigm that addresses the core bottleneck of RLHF/RLAIF.
- **Introduction of the Copeland method** creatively resolves preference cycles in LLM self-evaluation. Condorcet-style methods from voting theory carry inherent manipulation-resistance guarantees; applying them to LLM alignment represents a clever interdisciplinary transfer.
- **Actor+Judge joint optimization** forms a positive feedback loop: a better Judge produces more accurate reward signals → a better Actor is trained → the better Actor generates higher-quality responses → more discriminative comparison samples are provided to the Judge → the Judge improves further.
- **PBMM** and **LCM** are critical engineering details: the former eliminates positional bias in LLM-as-Judge by swapping response order, while the latter prevents length preference by applying a length ratio weight $\beta = |G^{Lose}|/|G^{Win}|$.

## Limitations & Future Work
- Self-evaluation quality remains bounded by the model's intrinsic capability—if the model's evaluative capacity has a hard ceiling, self-improvement will saturate accordingly.
- Copeland comparison requires $\binom{N}{2} \times K$ pairwise evaluations, incurring significant computational cost for large $N$ and $K$.
- The KL penalty is removed from GRPO—whether long-term training leads to excessive distributional shift warrants monitoring.
- Validation is limited to Qwen3-8B; generalizability to other architectures and larger models remains unknown.
- Whether iterative multi-round self-improvement is sustainable, and where its upper bound lies, requires longer experimental cycles to determine.

## Related Work & Insights
- **vs. Self-Rewarding (Yuan et al.)**: Self-Rewarding uses the Actor as Judge with point-wise scoring; SERL employs pairwise comparison + Copeland aggregation for greater robustness. Furthermore, Self-Rewarding does not optimize the Judge, whereas SERL jointly optimizes it via consistency rewards.
- **vs. Meta-Rewarding (Wu et al.)**: Meta-Rewarding jointly optimizes Actor and Judge but via off-policy learning; SERL uses on-policy learning, which is theoretically more stable.
- **vs. RLVR (GRPO/DAPO)**: RLVR requires verifiable answers and is limited to closed tasks such as mathematics and code; SERL derives rewards from self-comparison, making it applicable to open-domain settings.
- Insight: Voting-theoretic tools (Copeland, Borda, etc.) hold broader application potential in LLM alignment.
- Insight: The dual-reward joint optimization paradigm is extensible to other self-evaluation scenarios (e.g., simultaneously optimizing a generator and a test oracle in code self-debugging).

## Rating
- **Novelty**: ⭐⭐⭐⭐ An innovative combination of Copeland aggregation, dual rewards, and Actor-Judge joint optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ AlpacaEval as the primary benchmark, with multi-task validation on summarization, writing, and QA.
- **Writing Quality**: ⭐⭐⭐⭐ Clear methodology with rigorous formulations.
- **Value**: ⭐⭐⭐⭐⭐ External-dependency-free self-improvement approaching the performance of a model 4× larger in scale—significant practical implications.
- **Overall**: An important directional contribution to open-domain LLM post-training; the Copeland reward derivation is the central innovation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](../../NeurIPS2025/llm_reasoning/swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [\[ICLR 2026\] Semantic Voting: A Self-Evaluation-Free Approach for Efficient LLM Self-Improvement on Unverifiable Open-ended Tasks](../../ICLR2026/llm_reasoning/semantic_voting_a_self-evaluation-free_approach_for_efficient_llm_self-improveme.md)
- [\[ICLR 2026\] Learning to Reason over Continuous Tokens with Reinforcement Learning (HyRea)](../../ICLR2026/llm_reasoning/learning_to_reason_over_continuous_tokens_with_reinforcement_learning.md)
- [\[ICLR 2026\] Incentivizing LLM Reasoning via Reinforcement Learning with Functional Monte Carlo Tree Search](../../ICLR2026/llm_reasoning/incentivizing_llm_reasoning_via_reinforcement_learning_with_functional_monte_car.md)

</div>

<!-- RELATED:END -->
