---
title: >-
  [Paper Note] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning
description: >-
  [ACL 2026][Reinforcement Learning][Policy Optimization] This paper proposes the CE-GPPO algorithm, which reintroduces gradient signals for low-probability tokens outside the PPO clipping range via stop-gradient operation…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Policy Optimization"
  - "Entropy Dynamic Control"
  - "Gradient Preservation"
  - "PPO Improvement"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: 0fa343ad0e641330
---

# CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning

**Conference**: ACL 2026
**arXiv**: [2509.20712](https://arxiv.org/abs/2509.20712)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Policy Optimization, Entropy Dynamic Control, Gradient Preservation, PPO Improvement, Mathematical Reasoning

## TL;DR

This paper proposes the CE-GPPO algorithm, which reintroduces gradient signals for low-probability tokens outside the PPO clipping range via stop-gradient operations, enabling fine-grained coordination of policy entropy and achieving a better balance between exploration and exploitation.

## Background & Motivation

**Background**: Reinforcement learning (RL) has become the central paradigm for optimizing LLM reasoning capabilities, with PPO and its variants (GRPO, DAPO) being widely adopted.

**Limitations of Prior Work**: PPO's clipping mechanism discards gradient signals from low-probability tokens, leading to two failure modes: (1) PA&LP (positive-advantage & low-probability) tokens are clipped → entropy collapse; (2) NA&LP (negative-advantage & low-probability) tokens are clipped → entropy explosion.

**Key Challenge**: DAPO's clip-higher strategy only alleviates upper-bound clipping (preventing entropy collapse) but remains ineffective against lower-bound clipping (preventing entropy explosion), causing excessive exploration in the early stages of training.

**Goal**: To achieve stable and controllable entropy evolution by uniformly managing token gradients on both sides of the clipping range.

**Key Insight**: Entropy dynamics control is reformulated as a problem of managing gradients for tokens outside the clipping range.

**Core Idea**: Stop-gradient is used to decouple forward and backward passes; tunable coefficients β₁/β₂ independently control the magnitude of gradients beyond the left and right clipping boundaries, enabling precise regulation of the pace of entropy increase and decrease.

## Method

### Overall Architecture

Building upon GRPO, CE-GPPO reintroduces gradients for out-of-range tokens in a controlled manner rather than discarding them entirely. For PA&LP tokens (right boundary violation, positive advantage), β₂ amplifies their gradients to promote exploration; for NA&LP tokens (left boundary violation, negative advantage), β₁ amplifies their gradients to promote exploitation. Setting β₁=β₂=0 recovers standard PPO.

### Key Designs

1. **Stop-Gradient Gradient-Preserving Mechanism**:
    - **Function**: Reintroduces gradients from out-of-range tokens into parameter updates without disrupting the forward computation.
    - **Mechanism**: Applies sg(δ) (stop-gradient) to the importance ratio δ of out-of-range tokens, so that the forward pass still uses the clipped value while the backward pass propagates gradients, scaled by a coefficient.
    - **Design Motivation**: Directly relaxing the clipping constraint causes large policy shifts; stop-gradient ensures that gradient magnitudes remain bounded within β·(1±ε).

2. **Asymmetric Bilateral Gradient Scaling (β₁/β₂)**:
    - **Function**: Independently controls the intensity of exploration and exploitation.
    - **Mechanism**: β₂ governs PA&LP tokens (promoting exploration); β₁ governs NA&LP tokens (promoting exploitation); the two are independently tunable.
    - **Design Motivation**: Experiments show that β₂>β₁ is more conducive to maintaining exploration capability; β₁=0.75, β₂=1 yields the best performance on the 7B model.

3. **Theoretical Analysis of Entropy Dynamics**:
    - **Function**: Provides a theoretical foundation for the gradient-preserving strategy.
    - **Mechanism**: Using the entropy change formula (covariance between policy probabilities and the advantage function), the paper proves that PA&LP tokens slow entropy decline (promoting exploration) and NA&LP tokens accelerate entropy decline (promoting exploitation).
    - **Design Motivation**: Elevates empirical observations to theoretical explanations and guides hyperparameter selection.

### Loss & Training

The objective function modifies GRPO's three-branch loss: out-of-range negative advantage → β₁·(1-ε)/sg(δ)·δ·Â; out-of-range positive advantage → β₂·(1+ε)/sg(δ)·δ·Â; all other cases remain unchanged. Training uses the KlearReasoner-MathSub-30K dataset with lr=1e-6, rollout=8, ε=0.2, and a maximum of 1000 steps (~10 epochs).

## Key Experimental Results

### Main Results

| Method | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | HumanEval | LCB v6 |
|--------|--------|--------|--------|---------|-------|-----------|--------|
| DS-R1-1.5B (base) | 29.2 | 24.1 | 13.1 | 86.0 | 73.7 | 70.4 | 25.1 |
| + GRPO | 33.4 | 28.1 | 16.6 | 88.3 | 79.3 | 67.5 | 27.1 |
| + DAPO | 40.0 | 28.4 | 19.2 | 90.0 | 84.4 | 73.2 | 30.5 |
| + CE-GPPO (β₁=0.5) | 42.0 | 33.9 | 21.6 | 91.0 | 85.9 | 76.5 | 31.7 |
| DS-R1-7B (base) | 54.5 | 39.1 | 26.2 | 93.6 | 90.6 | 89.6 | 49.0 |
| + GRPO | 55.3 | 40.3 | 24.5 | 93.7 | 88.8 | 88.6 | 49.2 |
| + DAPO | 59.7 | 48.7 | 25.6 | 95.1 | 93.4 | 92.5 | 52.2 |
| + CE-GPPO (β₁=0.75) | **66.0** | **51.4** | **30.5** | **95.6** | **93.8** | **93.0** | **53.6** |

### Ablation Study

| β₁/β₂ Configuration | Entropy Trend | 1.5B AIME24 | 7B AIME25 |
|---------------------|---------------|-------------|-----------|
| β₁=1, β₂=0.5 | Rapid collapse | Performance rises then drops sharply | — |
| β₁=0, β₂=1 | Continuous increase | Good | Unstable |
| β₁=0.5, β₂=1 | Stable at high level | 42.0 | 49.1 |
| β₁=0.75, β₂=1 | Slow decline | 43.6 | **51.4** |

### Key Findings

- GRPO suffers from severe entropy collapse by default; DAPO mitigates this but exhibits excessively high entropy (over-exploration) in early training.
- CE-GPPO maintains stable entropy throughout training, with no anomalous fluctuations in KL divergence or gradient norms.
- Larger β₂ (amplifying exploration gradients) combined with smaller β₁ (constraining exploitation gradients) benefits performance; β₁=0.75, β₂=1 is the recommended default.
- Compared to CISPO: CISPO undergoes model collapse in late training, whereas CE-GPPO remains stable. Compared to GSPO: CE-GPPO substantially outperforms GSPO on AIME25 and HMMT25.

## Highlights & Insights

- The paper systematically explains entropy dynamics in RL training from the perspective of token probability–advantage function interactions, offering a novel and insightful viewpoint.
- The stop-gradient design is elegant: forward computation retains the stability of clipping, while backward propagation recovers the informational content of gradients, achieving both objectives simultaneously.
- The method is simple and efficient: it only modifies element-wise reweighting of the loss without introducing additional forward passes or auxiliary modules.
- The approach exhibits good robustness to hyperparameters; both β₁=0.5, β₂=1 and β₁=0.75, β₂=1 are effective across different model scales.

## Limitations & Future Work

- Although hyperparameter robustness is demonstrated, the optimal β₁/β₂ for different models still requires fine-tuning.
- Validation is limited to mathematical reasoning tasks; generalization to code generation, instruction following, and other tasks remains to be investigated.
- The paper claims complementarity with GSPO but does not conduct joint evaluation within a unified framework.
- The optimal entropy trajectory may vary across tasks and models; an adaptive adjustment mechanism is an important direction for future work.

## Related Work & Insights

- Relationship with DAPO (clip-higher): DAPO addresses only upper-bound clipping, whereas CE-GPPO handles both sides, offering a more complete solution.
- Comparison with entropy regularization: Traditional entropy regularization is highly sensitive to its coefficient (α=0.001 is ineffective; α=0.003 leads to explosion), while CE-GPPO is substantially more stable.
- Training insights from Kimi K2 (early exploration + late exploitation) align with the findings of this paper and can be realized by scheduling β across training phases.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Entropy control is reinterpreted through the lens of gradients from out-of-range tokens, offering a distinctive perspective with solid theoretical support.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-scale models, multiple baselines, detailed hyperparameter analysis, and stability verification.
- **Writing Quality**: ⭐⭐⭐⭐ The logic from problem analysis to method design is rigorous, with rich and intuitive figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](../../ICLR2026/reinforcement_learning/entropy-preserving_reinforcement_learning.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ACL 2026\] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](../../ICLR2026/reinforcement_learning/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)

</div>

<!-- RELATED:END -->
