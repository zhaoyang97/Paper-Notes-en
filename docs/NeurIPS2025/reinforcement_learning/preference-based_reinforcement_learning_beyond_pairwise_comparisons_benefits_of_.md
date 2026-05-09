---
title: >-
  [Paper Note] Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options
description: >-
  [NeurIPS 2025][Reinforcement Learning][Preference-based Reinforcement Learning] This paper proposes the M-AUPO algorithm for preference-based reinforcement learning, leveraging the Plackett-Luce ranking model to handle multi-option comparison feedback, and provides the first theoretical proof that larger subset sizes directly improve sample efficiency.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Preference-based Reinforcement Learning
  - Plackett-Luce Model
  - Ranking Feedback
  - Sample Efficiency
  - Multi-option Comparison
date: 2026-05-08
content_hash: eaea7794b7959845
---

# Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options

**Conference**: NeurIPS 2025

**arXiv**: [2510.18713](https://arxiv.org/abs/2510.18713)

**Code**: None

**Area**: Human Understanding / Reinforcement Learning

**Keywords**: Preference-based Reinforcement Learning, Plackett-Luce Model, Ranking Feedback, Sample Efficiency, Multi-option Comparison

## TL;DR

This paper proposes the M-AUPO algorithm for preference-based reinforcement learning, leveraging the Plackett-Luce ranking model to handle multi-option comparison feedback, and provides the first theoretical proof that larger subset sizes directly improve sample efficiency.

## Background & Motivation

Preference-based reinforcement learning (PbRL) learns policies from human preference feedback and has achieved notable success in LLM alignment. However, existing theoretical work almost exclusively focuses on **pairwise comparisons** (selecting one option from two).

Although a few works (Zhu et al., 2023; Mukherjee et al., 2024) have explored multi-option comparisons and ranking feedback, critical shortcomings remain:

**Performance guarantees do not improve with more options**: The suboptimality bounds of existing algorithms do not decrease as subset size $|S_t|$ grows.

**Potential degradation**: The bounds of certain methods even worsen as feedback length increases.

**Underutilization of information**: Ranking feedback contains substantially more information than pairwise comparisons, yet this advantage has not been theoretically exploited.

**Exponential dependence**: Most prior work exhibits exponential dependence on the norm of unknown parameters in their bounds.

## Method

### Overall Architecture

The paper considers an online PbRL setting: at each round $t$, the algorithm selects an action subset $S_t$ of size $|S_t|$, the user ranks the actions in the subset according to the Plackett-Luce (PL) ranking model, and the algorithm updates its policy using the ranking feedback.

### Key Designs

**1. Plackett-Luce Ranking Model**

- Each action $a$ has utility $\mu(a) = \phi(a)^\top \theta^*$ (linear parameterization).
- Ranking probability: $P(\sigma | S) = \prod_{i=1}^{|S|} \frac{\exp(\mu(a_{\sigma(i)}))}{\sum_{j=i}^{|S|} \exp(\mu(a_{\sigma(j)}))}$
- The PL model is a natural generalization of the Bradley-Terry model.

**2. M-AUPO (Maximize Average Uncertainty within Proposed Offer)**

- Core idea: select the action subset that **maximizes average uncertainty within the subset**.
- Uncertainty is measured based on the confidence ellipsoid of the current parameter estimate.
- Action selection criterion: $S_t = \arg\max_{S \subseteq \mathcal{A}, |S|=K} \frac{1}{K}\sum_{a \in S} \|a\|_{V_t^{-1}}$
- where $V_t$ is the cumulative information matrix.

**3. Suboptimality Bound**

$$\text{SubOpt}(T) = \tilde{O}\left(\sqrt{d \sum_{t=1}^T \frac{1}{|S_t|}}\right)$$

This is the first bound that **explicitly improves as subset size increases**, while avoiding exponential dependence on $\|\theta^*\|$.

### Loss & Training

- Model parameters are updated via maximum likelihood estimation (MLE) under the PL model.
- Exploration follows the optimism principle combined with uncertainty maximization.

## Key Experimental Results

### Main Results

Cumulative regret under different subset sizes:

| Algorithm | K=2 | K=4 | K=8 | K=16 |
|------|-----|-----|-----|------|
| Pairwise UCB | 285 | 285 | 285 | 285 |
| Zhu et al. (2023) | 310 | 295 | 298 | 305 |
| Mukherjee et al. (2024) | 275 | 268 | 270 | 278 |
| M-AUPO (Ours) | **270** | **195** | **142** | **108** |

Sample complexity improvement ratio under different dimensions $d$:

| Dimension $d$ | K=2 | K=4 | K=8 |
|---------|-----|-----|-----|
| d=5 | 1.0x | 1.8x | 3.2x |
| d=10 | 1.0x | 1.9x | 3.5x |
| d=20 | 1.0x | 1.9x | 3.6x |

### Ablation Study

Comparison of exploration strategies (cumulative regret, T=5000, K=8):

| Exploration Strategy | d=5 | d=10 | d=20 |
|---------|-----|------|------|
| Random subset selection | 215 | 340 | 520 |
| Max uncertainty single point + random | 178 | 285 | 435 |
| M-AUPO (average uncertainty) | **142** | **225** | **348** |

### Key Findings

1. M-AUPO is the only algorithm whose performance improves monotonically with subset size.
2. The failure of prior methods' bounds to improve with $|S_t|$ stems from exploration strategies that do not fully exploit ranking information.
3. A nearly matching lower bound $\Omega(d\sqrt{T/K})$ is established, demonstrating that M-AUPO is near-optimal.
4. Exponential dependence is avoided through a more refined likelihood analysis of the PL model.

## Highlights & Insights

- **Significant theoretical contribution**: The first formal proof that multi-option comparisons genuinely improve sample efficiency in PbRL.
- **Nearly matching upper and lower bounds**: The upper bound $\tilde{O}(d\sqrt{T/K})$ nearly matches the lower bound $\Omega(d\sqrt{T/K})$.
- **Elegant algorithmic design**: Average uncertainty maximization is the key innovation, circumventing the suboptimality of single-point maximization.

## Limitations & Future Work

1. The current analysis is restricted to linearly parameterized reward functions.
2. The PL model assumption may not perfectly capture all real human ranking behaviors.
3. Experiments are conducted primarily in synthetic environments, lacking validation with real human feedback.
4. Practical guidance for selecting the subset size $K$ is absent.

## Related Work & Insights

- **Linear bandits**: Classic algorithms such as Thompson Sampling and LinUCB.
- **PbRL theory**: Pairwise comparison frameworks from Novoseller et al. (2020) and Pacchiano et al. (2021).
- **Learning to rank**: Broad application of the Plackett-Luce model in recommender systems.

## Rating

- ⭐ Novelty: 9/10 — First theoretical resolution of sample efficiency in multi-option PbRL.
- ⭐ Value: 6/10 — Primarily a theoretical contribution; validation in practical application scenarios is insufficient.
- ⭐ Writing Quality: 8/10 — Rigorous theoretical derivations with clear motivation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning](stair_addressing_stage_misalignment_through_temporal-aligned_preference_reinforc.md)
- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)
- [\[AAAI 2026\] Beyond Monotonicity: Revisiting Factorization Principles in Multi-Agent Q-Learning](../../AAAI2026/reinforcement_learning/beyond_monotonicity_revisiting_factorization_principles_in_multi-agent_q-learnin.md)
- [\[NeurIPS 2025\] Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning](learning_to_focus_prioritizing_informative_histories_with_structured_attention_m.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)

<!-- RELATED:END -->
