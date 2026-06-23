---
title: >-
  [Paper Note] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization
description: >-
  [ICLR 2026][LLM Reasoning][efficient reasoning] The study diagnoses a fundamental flaw in GRPO when integrated with length penalties—correct but lengthy responses may obtain negative advantage values and thus be erroneously penalized. DRPO is proposed to decouple reward signals for positive and negative samples, ensuring that length penalties are normalized only wit
tags:
  - ICLR 2026
  - LLM Reasoning
  - efficient reasoning
  - overthinking
  - GRPO
  - length penalty
  - reinforcement-learning
date: 2026-05-08
content_hash: f869a9776c3fed71
---
# DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization

**Conference**: ICLR 2026  
**arXiv**: [2510.04474](https://arxiv.org/abs/2510.04474)  
**Code**: [https://github.com/Optimization-AI/DRPO](https://github.com/Optimization-AI/DRPO)  
**Area**: LLM Reasoning  
**Keywords**: efficient reasoning, overthinking, GRPO, length penalty, reinforcement-learning

## TL;DR
The study diagnoses a fundamental flaw in GRPO when integrated with length penalties—correct but lengthy responses may obtain negative advantage values and thus be erroneously penalized. DRPO is proposed to decouple reward signals for positive and negative samples, ensuring that length penalties are normalized only within the group of correct responses. On a 1.5B model, DRPO achieves a 77% reduction in length with only 1.1% performance loss (compared to a 68% reduction and 4.3% loss for the baseline).

## Background & Motivation
**Background**: Large reasoning models (such as DeepSeek-R1) acquire strong reasoning capabilities through GRPO training. however, they suffer from severe overthinking—generating ~1000 tokens even for simple queries like "2+3=?".

**Limitations of Prior Work**: Existing RL methods encourage concise reasoning by adding length penalties to rewards (e.g., RLOO-LP, ALP, HAPO), but almost all lead to significant performance degradation.

**Key Challenge**: GRPO's group-relative advantage function, when mixed with length penalties, can push the advantage of correct but verbose answers to negative values. The model is thus misled into punishing valid reasoning as a negative sample. For instance, in a group of 6 responses where 3 are correct, the advantage of the third correct answer can drop from $+1$ to $-0.17$ after adding a length penalty.

**Goal**: How to shorten reasoning length while minimizing performance loss?

**Key Insight**: Shift the computation of learning signals from mixed positive-negative samples to separated groups. The length penalty for correct responses should be normalized only within the set of correct responses, ensuring it never generates a negative learning signal.

**Core Idea**: Decouple the reward normalization of positive and negative samples, allowing the length penalty to attenuate (but never flip) the learning signal of correct answers.

## Method

### Overall Architecture
DRPO aims to resolve the performance drop induced by length penalties. It identifies the root cause as the mixed normalization of positive and negative samples in GRPO and adopts an objective function that avoids this mixture. Rather than building on GRPO, it utilizes the discriminative optimization framework DisCO, splitting the target into two non-interfering terms: the positive term uses a closed-form weighting based on length rewards for weighted likelihood (shorter correct answers receive higher weights), while the negative term aggregates hard negative samples using log-sum-exp. Since the length penalty resides only within the positive term, "encouraging conciseness" no longer penalizes correct reasoning.

### Key Designs

**1. Diagnosing the Negative Advantage Trap: Why length penalties flip signals for correct reasoning**

GRPO utilizes a group-relative advantage $A(o_i|q) = \frac{r(o_i) - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$, standardizing correct and incorrect responses sampled for the same prompt within a single group. While effective for binary 0/1 correctness rewards, incorporating a length penalty into $r$ causes correct but longer responses to fall below the group mean, resulting in a negative advantage. Consequently, the model is instructed to suppress these correct reasonings. The paper clarifies that this is not an implementation bug but a structural flaw shared by all methods relying on relative advantage or mixed normalization (e.g., RLOO-LP, ALP, HAPO). As long as rewards are composite (correctness + length), joint normalization carries the risk of flipping positive signals to negative.

**2. Decoupled Normalization: Confining length penalties to the positive sample group**

To address the mixed normalization issue, DRPO isolates the learning signal of correct responses for independent normalization within the positive sample group. It first solves for an optimal positive sample distribution under KL-regularization:

$$P_q^* = \arg\max_P \mathbb{E}_{o\sim P}[r_l(o)] - \lambda D_{KL}(P, \pi_{old}^+)$$

where $r_l(o) = 1 - |o|/C$ is a length reward that yields higher scores for shorter outputs. This objective has a closed-form solution $P_q^*(o) \propto \pi_{old}^+(o|q)\exp(r_l(o)/\lambda)$, resulting in a weight for each positive sample normalized strictly within the positive set:

$$\omega(o|q) = \frac{\exp(r_l(o)/\lambda)}{\mathbb{E}_{o\sim\pi^+}\exp(r_l(o)/\lambda)}.$$

Crucially, $\omega$ remains strictly positive: shorter answers receive higher $r_l$ and larger weights, while longer answers receive smaller weights that approach zero but never become negative. Thus, "encouraging conciseness" shifts from "rejection of long correct answers" to "weight redistribution among correct answers." The hyperparameter $\lambda$ smoothly controls the trade-off (as $\lambda\to\infty$, the weights become uniform, disabling length control).

**3. Discriminative Objective: Complete separation of positive and negative terms**

With the positive-only weight $\omega$, DRPO employs the DisCO framework to split the objective into two independent components. The positive term $\mathbb{E}_{o\sim\pi^+}\,\omega(o|q)\,s_\theta(o,q)$ uses $\omega$ for weighted likelihood to increase the scores $s_\theta$ of short, correct answers. The negative term $-\tau\log\mathbb{E}_{o'\sim\pi^-}\exp(s_\theta(o',q)/\tau)$ uses log-sum-exp aggregation to automatically focus gradients on the hardest negative samples. An outer trust region constraint $D_{KL}(\pi_{old}\,\|\,\pi_\theta) \leq \delta$ stabilizes updates. Because positive and negative terms are decoupled, the length penalty never influences the discrimination of incorrect samples.

### Loss & Training
The framework is based on DisCO, with constraints handled via penalty functions. Training is conducted on the DeepScaleR-Preview-Dataset (40.3K math problems) for 1000 steps, with a generation budget of 8K tokens and 8 samples per problem.

## Key Experimental Results

### Main Results (AES: Accuracy Efficiency Score)

| Method | Model | Pass@1 | Length | AES |
|------|------|--------|--------|-----|
| RLOO-LP | 1.5B | 0.567 | 2531 | -0.129 |
| ALP | 1.5B | 0.606 | 3494 | -0.387 |
| HAPO | 1.5B | 0.534 | 1791 | -0.519 |
| **DRPO** | **1.5B** | **0.624** | **1527** | **+0.178** |
| RLOO-LP | 7B | 0.692 | 2649 | -0.033 |
| **DRPO** | **7B** | **0.714** | **1502** | **+0.249** |

### Key Findings
- DRPO is the only method to achieve a positive AES across all model scales (1.5B/7B/8B), whereas baseline AES values are mostly negative.
- On GSM8K (1.5B model): DRPO achieves 77% length reduction with only 1.1% performance loss, compared to the baseline's 68% reduction with 4.3% loss.
- 7B model: DRPO achieves 51% length reduction with 2.6% loss, while RLOO-LP yields 38% reduction with 7.1% loss.
- $\lambda$ provides a smooth trade-off between length and accuracy: $\lambda\to\infty$ means no length control, while $\lambda\to 0$ maximizes length penalty.
- Effective on non-mathematical reasoning tasks (e.g., K&K logic puzzles).

## Highlights & Insights
- **Decoupling is the core insight**: The issue with GRPO is not the length penalty itself, but the mixed normalization of positive and negative samples. Decoupling provides a clean and elegant fix.
- **Theoretical elegance of closed-form optimal distribution**: The weighting scheme is derived directly from the KL-regularized RLHF framework without requiring additional reward model training or data collection.
- **General diagnosis for GRPO variants**: The paper identifies that all relative-advantage methods (e.g., RLOO, REINFORCE) suffer from this issue under composite rewards. DRPO’s decoupling principle is universally applicable.

## Limitations & Future Work
- Dependency on the DisCO framework rather than GRPO may require specific engineering adaptations.
- The length reward $r_l(o) = 1 - |o|/C$ is a simple linear function; more complex length-quality relationships may necessitate non-linear designs.
- Validation is primarily on mathematical reasoning; effectiveness in other domains like code or scientific reasoning remains to be confirmed.
- The generation budget is capped at 8K tokens; performance on extremely long reasoning sequences is unknown.

## Related Work & Insights
- **vs GRPO + length penalty (RLOO-LP/ALP/HAPO)**: These methods are limited by the negative advantage problem caused by mixed normalization, which DRPO resolves.
- **vs DisCO**: DRPO extends DisCO by introducing a closed-form weighting scheme for length rewards, serving as a natural expansion toward efficient reasoning.
- **vs L1-max / ShorterBetter**: While these use different mechanisms to control length, they still face performance-efficiency trade-offs where DRPO consistently maintains superior AES.
- **vs VIP (Adaptive Rollout)**: VIP optimizes computation allocation before sampling, whereas DRPO optimizes learning signals in the training objective; the two are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Clear diagnosis, elegant solution, and complete theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Tested across 3 models, 6 baselines, 4 difficulty levels, with quantitative AES comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Figure 1 provides an intuitive diagnosis, and theoretical derivations are concise.
- Value: ⭐⭐⭐⭐⭐ Highly practical, addressing a core conflict in training efficient reasoning models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning](reference-guided_policy_optimization_for_molecular_optimization_via_llm_reasonin.md)
- [\[ICLR 2026\] PEAR: Phase Entropy Aware Reward for Efficient Reasoning](pear_phase_entropy_aware_reward_for_efficient_reasoning.md)
- [\[ICLR 2026\] HiPO: Self-Hint Policy Optimization for RLVR](hipo_self-hint_policy_optimization_for_rlvr.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)

</div>

<!-- RELATED:END -->
