---
title: >-
  [Paper Note] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning
description: >-
  [ACL 2026][Reinforcement Learning][Policy optimization] Ours proposes the CE-GPPO algorithm, which reintroduces gradient signals for low-probability tokens outside the PPO clipping range through stop-gradient operations.…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Policy optimization"
  - "dynamic entropy control"
  - "gradient preservation"
  - "PPO improvement"
  - "mathematical reasoning"
date: 2026-05-08
content_hash: 60936c27f6fd3b52
---

# CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2509.20712](https://arxiv.org/abs/2509.20712)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Policy optimization, dynamic entropy control, gradient preservation, PPO improvement, mathematical reasoning

## TL;DR

Ours proposes the CE-GPPO algorithm, which reintroduces gradient signals for low-probability tokens outside the PPO clipping range through stop-gradient operations. This enables fine-grained coordinated control of policy entropy, achieving a better balance between exploration and exploitation.

## Background & Motivation

**Background**: Reinforcement Learning (RL) has become the core paradigm for optimizing the reasoning capabilities of LLMs, with PPO and its variants (GRPO, DAPO) being widely utilized.

**Limitations of Prior Work**: The clipping mechanism of PPO discards gradient signals for low-probability tokens, leading to two classes of problems: (1) PA&LP (Positive Advantage & Low Probability) tokens are clipped $\rightarrow$ entropy collapse; (2) NA&LP (Negative Advantage & Low Probability) tokens are clipped $\rightarrow$ entropy explosion.

**Key Challenge**: The clip-higher strategy of DAPO only alleviates upper-bound clipping (preventing entropy collapse) but is powerless against lower-bound clipping (preventing entropy explosion), resulting in over-exploration during early training stages.

**Goal**: To achieve stable and controllable evolution of policy entropy by uniformly managing the token gradients on both sides of the clipping interval.

**Key Insight**: Re-define the control of entropy dynamics as a management problem of gradients for tokens outside the clipping interval.

**Core Idea**: Decouple forward and backward propagation via stop-gradient, using adjustable coefficients $\beta_1/\beta_2$ to respectively control the magnitude of gradients outside the left and right clipping boundaries, thereby finely regulating the rhythm of entropy increase and decrease.

## Method

### Overall Architecture

Based on GRPO, CE-GPPO no longer completely discards gradients for tokens outside the clipping interval but reintroduces them in a controlled manner. For PA&LP tokens (right out-of-bounds, positive advantage), $\beta_2$ is used for amplification to promote exploration; for NA&LP tokens (left out-of-bounds, negative advantage), $\beta_1$ is used for amplification to promote exploitation. When $\beta_1=\beta_2=0$, the method degrades to standard PPO.

### Key Designs

1. **Stop-Gradient Preservation Mechanism**:
    - **Function**: Re-incorporates the gradients of out-of-bounds tokens into updates without breaking the forward computation.
    - **Mechanism**: Applies $sg(\delta)$ stop-gradient to the importance ratio $\delta$ of out-of-bounds tokens, ensuring the forward pass remains the clipped value while the backward pass permits gradient flow, multiplied by a scaling coefficient.
    - **Design Motivation**: Directly removing the clip would cause significant policy shifts; stop-gradient ensures the gradient magnitude is restricted within $\beta \cdot (1 \pm \epsilon)$.

2. **Bilateral Asymmetric Gradient Scaling ($\beta_1/\beta_2$)**:
    - **Function**: Independently controls the intensity of exploration and exploitation.
    - **Mechanism**: $\beta_2$ controls PA&LP tokens (promoting exploration) and $\beta_1$ controls NA&LP tokens (promoting exploitation), with both being independently adjustable.
    - **Design Motivation**: Experiments indicate that $\beta_2 > \beta_1$ is more conducive to maintaining exploration capabilities; for instance, $\beta_1=0.75, \beta_2=1$ yields the best results on 7B models.

3. **Theoretical Analysis of Entropy Dynamics**:
    - **Function**: Provides a theoretical foundation for the gradient-preserving strategy.
    - **Mechanism**: Based on the entropy change formula (covariance of policy probability and advantage function), it proves that PA&LP tokens slow down entropy decay (promoting exploration), while NA&LP tokens accelerate entropy decay (promoting exploitation).
    - **Design Motivation**: Elevates empirical observations into theoretical explanations to guide hyperparameter selection.

### Loss & Training

The objective function modifies the three-branch loss based on GRPO: out-of-bounds negative advantage $\rightarrow$ $\beta_1 \cdot (1-\epsilon)/sg(\delta) \cdot \delta \cdot \hat{A}$; out-of-bounds positive advantage $\rightarrow$ $\beta_2 \cdot (1+epsilon)/sg(\delta) \cdot \delta \cdot \hat{A}$; others remain unchanged. Training utilizes the KlearReasoner-MathSub-30K dataset, $lr=1e-6$, $rollout=8$, $\epsilon=0.2$, for a maximum of 1000 steps (~10 epochs).

## Key Experimental Results

### Main Results

| Method | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | HumanEval | LCB v6 |
|------|--------|--------|--------|---------|-------|-----------|--------|
| DS-R1-1.5B (base) | 29.2 | 24.1 | 13.1 | 86.0 | 73.7 | 70.4 | 25.1 |
| + GRPO | 33.4 | 28.1 | 16.6 | 88.3 | 79.3 | 67.5 | 27.1 |
| + DAPO | 40.0 | 28.4 | 19.2 | 90.0 | 84.4 | 73.2 | 30.5 |
| + CE-GPPO ($\beta_1=0.5$) | 42.0 | 33.9 | 21.6 | 91.0 | 85.9 | 76.5 | 31.7 |
| DS-R1-7B (base) | 54.5 | 39.1 | 26.2 | 93.6 | 90.6 | 89.6 | 49.0 |
| + GRPO | 55.3 | 40.3 | 24.5 | 93.7 | 88.8 | 88.6 | 49.2 |
| + DAPO | 59.7 | 48.7 | 25.6 | 95.1 | 93.4 | 92.5 | 52.2 |
| + CE-GPPO ($\beta_1=0.75$) | **66.0** | **51.4** | **30.5** | **95.6** | **93.8** | **93.0** | **53.6** |

### Ablation Study

| $\beta_1/\beta_2$ Config | Entropy Trend | 1.5B AIME24 | 7B AIME25 |
|------------|--------|-------------|-----------|
| $\beta_1=1, \beta_2=0.5$ | Fast Collapse | Performance rises then drops sharply | — |
| $\beta_1=0, \beta_2=1$ | Continuous Rise | Moderate | Unstable |
| $\beta_1=0.5, \beta_2=1$ | High-level Stable | 42.0 | 49.1 |
| $\beta_1=0.75, \beta_2=1$ | Slow Decrease | 43.6 | **51.4** |

### Key Findings

- Native GRPO suffers from severe entropy collapse; while DAPO mitigates this, entropy remains too high during early training (over-exploration).
- CE-GPPO maintains stable entropy throughout the training process, with no abnormal fluctuations in KL divergence or gradient norms.
- Larger $\beta_2$ (amplifying exploration gradients) + smaller $\beta_1$ (restricting exploitation gradients) benefits performance; $\beta_1=0.75, \beta_2=1$ is the recommended default.
- Comparison with CISPO: CISPO experiences model collapse in late-stage training, whereas CE-GPPO remains stable. Comparison with GSPO: CE-GPPO significantly outperforms GSPO on AIME25/HMMT25.

## Highlights & Insights

- Systematically explains entropy dynamics in RL training from the perspective of token probability-advantage function interaction, providing a novel and profound viewpoint.
- The stop-gradient design is ingenious: it preserves clipping stability in the forward pass while restoring gradient information in the backward pass, achieving the best of both worlds.
- The method is simple and efficient: it only changes element-wise reweighting of the loss without introducing additional forward passes or auxiliary modules.
- It exhibits good robustness to hyperparameters; $\beta_1=0.5, \beta_2=1$ and $\beta_1=0.75, \beta_2=1$ are effective across different model scales.

## Limitations & Future Work

- Although hyperparameter robustness is demonstrated, the optimal $\beta_1/\beta_2$ for different models still requires fine-tuning.
- Validated only on mathematical reasoning tasks; generalization to tasks like code generation and instruction following remains to be investigated.
- The paper claims complementarity with GSPO but has not conducted joint testing within the same framework.
- The optimal dynamic trajectory of entropy may vary by task/model; an adaptive adjustment mechanism is an important future direction.

## Related Work & Insights

- Relationship with DAPO (clip-higher): DAPO only addresses upper-bound clipping, while CE-GPPO handles both sides, making it more comprehensive.
- Comparison with entropy regularization: Traditional entropy regularization is extremely sensitive to coefficients ($\alpha=0.001$ is ineffective, $\alpha=0.003$ explodes); CE-GPPO is more stable.
- Kimi K2 training experience (early exploration + late exploitation) is consistent with the findings of this paper and can be achieved via phased $\beta$ adjustments.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Uniquely understands entropy control from the perspective of out-of-clipping token gradients, supported by theory.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-scale models, multiple baseline comparisons, detailed hyperparameter analysis, and stability verification.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous logic from problem analysis to method design, with rich and intuitive charts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](../../ICLR2026/reinforcement_learning/entropy-preserving_reinforcement_learning.md)
- [\[ACL 2026\] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)
- [\[ACL 2026\] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)
- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)

</div>

<!-- RELATED:END -->
