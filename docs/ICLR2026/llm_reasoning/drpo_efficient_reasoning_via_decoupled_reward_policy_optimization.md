---
title: >-
  [Paper Note] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization
description: >-
  [ICLR 2026][LLM Reasoning][efficient reasoning] This paper diagnoses a fundamental flaw in GRPO with length penalties — correct but verbose responses may receive negative advantage values and thus be incorrectly penalize…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "efficient reasoning"
  - "overthinking"
  - "GRPO"
  - "length penalty"
  - "reinforcement-learning"
date: 2026-05-08
content_hash: dc57cc1dba5fe1cb
---

# DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization

**Conference**: ICLR 2026
**arXiv**: [2510.04474](https://arxiv.org/abs/2510.04474)  
**Code**: [https://github.com/Optimization-AI/DRPO](https://github.com/Optimization-AI/DRPO)  
**Area**: LLM Reasoning
**Keywords**: efficient reasoning, overthinking, GRPO, length penalty, reinforcement-learning

## TL;DR
This paper diagnoses a fundamental flaw in GRPO with length penalties — correct but verbose responses may receive negative advantage values and thus be incorrectly penalized — and proposes DRPO, which decouples the reward signals for positive and negative samples to ensure length penalties are normalized only within the correct-response group. On a 1.5B model, DRPO achieves a 77% length reduction with only a 1.1% performance drop, compared to a 68% reduction with a 4.3% drop for the baseline.

## Background & Motivation
**Background**: Large reasoning models (e.g., DeepSeek-R1) acquire strong reasoning capabilities through GRPO training, but suffer from severe overthinking — even answering "2+3=?" requires generating ~1,000 tokens.

**Limitations of Prior Work**: Existing RL methods encourage concise reasoning by incorporating length penalties into the reward (e.g., RLOO-LP, ALP, HAPO), but almost universally lead to significant performance degradation.

**Key Challenge**: GRPO's group-relative advantage function, when combined with length penalties, can push the advantage of correct but verbose responses into negative territory — misleading the model into treating valid reasoning as a negative sample to be penalized. For example, among 6 responses where 3 are correct, the advantage of the third correct response can drop from +1 to −0.17 after applying a length penalty.

**Goal**: How can reasoning length be reduced while minimizing performance loss?

**Key Insight**: Decouple the computation of learning signals from the mixture of positive and negative samples — the length penalty for correct responses is normalized only within the correct-response group, never producing a negative learning signal.

**Core Idea**: Decouple reward normalization for positive and negative samples so that length penalties attenuate (rather than reverse) the learning signal for correct responses.

## Method

### Overall Architecture
DRPO is built on the Discriminative Contrastive Optimization (DisCO) framework rather than GRPO. The positive-sample term in the objective uses importance-sampling weighted by a length-based reward (with weights derived from the closed-form optimal distribution), while the negative-sample term uses log-sum-exp aggregation. Positive and negative samples are fully decoupled.

### Key Designs

1. **Problem Diagnosis: The Fundamental Flaw of GRPO with Length Penalties**:

    - GRPO's advantage function $A(o_i|q) = \frac{r(o_i) - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$ normalizes correct and incorrect samples jointly.
    - After introducing a length penalty, a correct but verbose response $r$ may fall below the group mean → negative advantage → the model is penalized for correct reasoning.
    - This issue is present in all GRPO-based methods, including RLOO-LP, ALP, and HAPO.

2. **DRPO Decoupling Design**:

    - **Function**: Learning signals for correct responses are normalized only within the correct-response group.
    - **Mechanism**: The optimal positive-data distribution is derived by solving $P_q^* = \arg\max_P \mathbb{E}[r_l(o)] - \lambda D_{KL}(P, \pi_{old}^+)$, yielding the closed-form solution $P_q^*(o) \propto \pi_{old}^+(o|q) \exp(r_l(o)/\lambda)$. The weight of each positive sample $\omega(o|q) = \frac{\exp(r_l(o)/\lambda)}{\mathbb{E}_{o\sim\pi^+}\exp(r_l(o)/\lambda)}$ is normalized within the positive-sample group.
    - **Design Motivation**: The weight $\omega$ is always positive — shorter responses receive higher weights and longer responses lower weights, but never negative. The hyperparameter $\lambda$ controls the length–accuracy trade-off.

3. **Discriminative Objective Function**:

    - Positive term: $\mathbb{E}_{o\sim\pi^+} \omega(o|q) s_\theta(o,q)$ — promotes shorter correct responses via weighted likelihood.
    - Negative term: $-\tau \log \mathbb{E}_{o'\sim\pi^-} \exp(s_\theta(o',q)/\tau)$ — log-sum-exp automatically up-weights hard negatives.
    - Constraint: $D_{KL}(\pi_{old} || \pi_\theta) \leq \delta$ ensures training stability.
    - When $\lambda=+\infty$, DRPO reduces to DisCO (no length penalty).

### Loss & Training
DRPO is built on the DisCO framework, with constraints handled via a penalty function. Training is conducted on the DeepScaleR-Preview-Dataset (40.3K math problems) for 1,000 steps, with a generation budget of 8K tokens and 8 sampled responses per question.

## Key Experimental Results

### Main Results (AES: Accuracy Efficiency Score)

| Method | Model | Pass@1 | Length | AES |
|--------|-------|--------|--------|-----|
| RLOO-LP | 1.5B | 0.567 | 2531 | -0.129 |
| ALP | 1.5B | 0.606 | 3494 | -0.387 |
| HAPO | 1.5B | 0.534 | 1791 | -0.519 |
| **DRPO** | **1.5B** | **0.624** | **1527** | **+0.178** |
| RLOO-LP | 7B | 0.692 | 2649 | -0.033 |
| **DRPO** | **7B** | **0.714** | **1502** | **+0.249** |

### Key Findings
- DRPO is the only method to achieve a positive AES across all model scales (1.5B/7B/8B) — all baselines yield negative AES in most settings.
- On GSM8K with the 1.5B model: DRPO achieves 77% length reduction with only 1.1% performance loss, vs. 68% reduction with 4.3% loss for the baseline.
- On the 7B model: DRPO achieves 51% length reduction with only 2.6% performance loss, vs. RLOO-LP's 38% reduction with 7.1% loss.
- $\lambda$ smoothly controls the length–accuracy trade-off: $\lambda\to\infty$ disables length control; $\lambda\to 0$ maximizes the length penalty.
- DRPO is also effective on non-mathematical reasoning tasks (K&K logic puzzles).

## Highlights & Insights
- **"Positive–negative decoupling" is the central insight**: The problem with GRPO lies not in the length penalty itself, but in the joint normalization of positive and negative samples. Decoupling naturally resolves the issue — a clear and elegant diagnosis and fix.
- **Theoretical elegance of the closed-form optimal distribution**: No additional reward model training or data collection is required; the weighting scheme is derived directly from the KL-regularized RLHF framework.
- **Universal diagnosis for all GRPO variants**: The paper shows that all relative-advantage methods — including RLOO and REINFORCE — exhibit this issue under composite rewards, making the decoupling principle of DRPO broadly applicable.

## Limitations & Future Work
- DRPO is built on the DisCO framework rather than GRPO, which may require additional engineering effort for adaptation.
- The length reward $r_l(o) = 1 - |o|/C$ is simple and linear; more complex length–quality relationships may call for nonlinear designs.
- Validation is limited to mathematical reasoning; applicability to other domains such as code generation and scientific reasoning remains to be confirmed.
- The generation budget is capped at 8K tokens; effectiveness on extremely long reasoning chains is unknown.

## Related Work & Insights
- **vs. GRPO with length penalties (RLOO-LP/ALP/HAPO)**: These methods are all constrained by the negative-advantage problem arising from joint normalization; DRPO addresses this fundamentally through decoupling.
- **vs. DisCO**: DRPO extends DisCO by introducing a closed-form weighting scheme for length rewards, representing a natural extension of DisCO toward efficient reasoning.
- **vs. L1-max / ShorterBetter**: These methods control length via different mechanisms but still face the performance–efficiency trade-off. DRPO consistently achieves superior AES.
- **vs. VIP (Adaptive Rollout)**: VIP optimizes computation allocation prior to sampling, while DRPO optimizes the learning signal at the training objective level — the two approaches are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Clear diagnosis, elegant solution, and theoretically complete closed-form derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three model scales, six baselines, four difficulty levels, and quantitative AES comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ The diagnostic illustration in Figure 1 is intuitive, and the theoretical derivation is clean.
- Value: ⭐⭐⭐⭐⭐ Resolves a core contradiction in efficient reasoning training with strong practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](../../ACL2026/llm_reasoning/calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ACL 2026\] Think Outside the Policy: In-Context Steered Policy Optimization](../../ACL2026/llm_reasoning/think_outside_the_policy_in-context_steered_policy_optimization.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)

</div>

<!-- RELATED:END -->
