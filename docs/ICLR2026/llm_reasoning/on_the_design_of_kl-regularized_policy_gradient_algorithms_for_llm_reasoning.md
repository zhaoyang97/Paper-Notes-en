---
title: >-
  [Paper Note] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][KL Regularization] This paper proposes the Regularized Policy Gradient (RPG) framework, which systematically derives and analyzes policy gradient methods based on Forward/Reverse KL divergence…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "KL Regularization"
  - "Policy Gradient"
  - "GRPO"
  - "REINFORCE"
date: 2026-05-08
content_hash: 4cac7cd45df5fd57
---

# On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning

**Conference**: ICLR 2026
**arXiv**: [2505.17508](https://arxiv.org/abs/2505.17508)
**Code**: [https://github.com/complex-reasoning/RPG](https://github.com/complex-reasoning/RPG)
**Area**: LLM Reasoning / Reinforcement Learning
**Keywords**: KL Regularization, Policy Gradient, LLM Reasoning, GRPO, REINFORCE

## TL;DR

This paper proposes the Regularized Policy Gradient (RPG) framework, which systematically derives and analyzes policy gradient methods based on Forward/Reverse KL divergence (in both normalized and unnormalized forms). It identifies a theoretical inconsistency in GRPO's KL term and achieves superior performance over GRPO, REINFORCE++, and DAPO on mathematical reasoning benchmarks.

## Background & Motivation

Policy gradient methods such as PPO and GRPO have been widely adopted for RLHF and reasoning capability enhancement in LLMs. KL divergence regularization is a critical technique for stabilizing policy optimization, preventing the policy from deviating excessively from the reference policy and mitigating catastrophic forgetting and overconfident outputs.

However, existing methods differ substantially in how KL divergence is implemented:
- **KL Direction**: Forward KL (zero-forcing) vs. Reverse KL (mode-seeking), each with distinct optimization properties
- **Normalization**: Standard normalized KL vs. Unnormalized KL (UKL), where the latter relates to the $k_3$ estimator used in GRPO
- **Estimator Type**: Fully differentiable forms vs. REINFORCE-style (using stop-gradient operators)
- **Off-policy Estimation**: Handling of importance weights in off-policy settings affects gradient correctness

The authors identify that GRPO's KL penalty term omits importance weights in the off-policy estimation, causing the gradient to fail to correspond exactly to the gradient of the intended objective. The KL handling in REINFORCE++ is also non-standard — its KL term is computed between the old policy and the SFT policy, rather than the policy currently being optimized.

## Method

### Overall Architecture

RPG is an iterative training framework in which the reference model $\pi_{\text{old}}$ is updated to the previous iteration's policy $\pi_{\theta^{(t)}}$, providing a dynamically adaptive regularization target. The framework centers on constructing the KL-regularized objective $J(\theta) = \mathbb{E}_{\pi_\theta}[R] - \beta \cdot \text{KL}$ and deriving the corresponding surrogate loss for gradient descent.

### Key Designs

1. **Forward KL Regularization (FKL)**:

    - Objective: $J_{\text{FKL}}(\theta) = \mathbb{E}_{\pi_\theta}[R(x)] - \beta \text{KL}(\pi_{\text{old}} \| \pi_\theta)$
    - Gradient: $\nabla_\theta J = \mathbb{E}_{x \sim \pi_{\text{old}}}[(w(x)R(x) + \beta) \nabla_\theta \log \pi_\theta(x)]$
    - Surrogate loss: $\mathcal{L}_{\text{FKL}} = \mathbb{E}[-w(x)R(x) - \beta \log \pi_\theta(x)]$
    - Degenerates to MLE when $R=0$, consistent with the SFT training objective
    - **Design Motivation**: Forward KL encourages $\pi_\theta$ to cover the support of $\pi_{\text{old}}$ (zero-forcing), avoiding neglect of high-probability regions

2. **Reverse KL Regularization (RKL)**:

    - Objective: $J_{\text{RKL}}(\theta) = \mathbb{E}_{\pi_\theta}[R(x)] - \beta \text{KL}(\pi_\theta \| \pi_{\text{old}})$
    - Surrogate loss: $\mathcal{L}_{\text{RKL}} = \mathbb{E}[w(x)(-R(x) + \beta \log w(x))]$
    - **Design Motivation**: Reverse KL encourages $\pi_\theta$ to concentrate on high-probability regions of $\pi_{\text{old}}$ (mode-seeking), suitable for focusing on known good policies

3. **Unnormalized Forward KL (UFKL)**:

    - Introduces unnormalized KL divergence, incorporating a mass correction term
    - Surrogate loss: $\mathcal{L}_{\text{UFKL}} = Z_{\text{old}} \mathbb{E}[-w(x)R(x) + \beta(w(x) - \log w(x) - 1)]$
    - The regularization term $w(x) - \log w(x) - 1$ takes exactly the form of the $k_3$ estimator
    - **Design Motivation**: Handles scenarios where distributions may be unnormalized, and establishes a theoretical connection to the $k_3$ estimator used in GRPO

4. **Unnormalized Reverse KL (URKL)**:

    - Proves that the expectation of $k_3(\pi_{\text{old}}/\pi_\theta)$ is equivalent to $\text{UKL}(\pi_\theta \| \pi_{\text{old}})$
    - Surrogate loss: $\mathcal{L}_{\text{URKL}} = Z_{\text{old}} \mathbb{E}[-w(x)R(x) + \beta(w(x)\log w(x) - w(x))]$
    - **Design Motivation**: The effective reward scaling factor in the gradient is more concise ($R(x) - \beta \log w(x)$), and is equivalent to the $k_3$ estimator

5. **REINFORCE-Style Variants**:

    - Provides REINFORCE-style surrogate losses for all four KL formulations, using the $\text{SG}(\cdot)$ (stop-gradient) operator
    - General form: $\mathcal{L}^{\text{REINFORCE}} = -\mathbb{E}[\text{SG}(\text{Weight}(x, \theta)) \log \pi_\theta(x)]$
    - Offers implementation flexibility to accommodate different framework requirements

6. **Analysis of Theoretical Inconsistency in GRPO**:

    - GRPO employs the $k_3$ estimator as a KL penalty, but directly subtracts this term in the off-policy setting without including the importance weight $w_{i,t}$
    - This causes the gradient of GRPO's objective to fail to correspond precisely to the gradient of $J_{\text{Clip}} - \beta \text{UKL}(\pi_\theta \| \pi_{\text{ref}})$
    - The RPG framework corrects this issue by explicitly incorporating importance weights

### Loss & Training

- **Dual-Clip Objective**: Adopts a Dual-Clip variant of PPO to stabilize training, handling positive and negative advantage values separately
- **Baseline Subtraction**: Uses the batch mean reward as a baseline to reduce gradient variance
- **Dynamic Sampling + Group Filtering**: Draws on DAPO's strategy of oversampling difficult prompts and filtering groups with near-perfect or near-zero accuracy
- **Length Penalty**: Penalizes excessively verbose outputs in the reward signal
- **Memory Efficiency**: Log probabilities from $\pi_{\text{old}}$ can be precomputed and cached; during training, only one model $\pi_\theta$ needs to be maintained on GPU

## Key Experimental Results

### Main Results

Based on Qwen2.5-7B-Instruct, trained on the English subset of DAPO-Math-17k (13.9k samples) for 400 steps.

| Method | AMC23 (Best) | AIME24 (Best) | AIME25 (Best) |
|--------|-------------|---------------|---------------|
| GRPO | 0.7250 | 0.1406 | 0.0948 |
| REINFORCE++ | 0.7664 | 0.1177 | 0.0740 |
| REINFORCE++-Baseline | 0.8711 | 0.1510 | 0.0969 |
| DAPO | 0.8734 | 0.1240 | 0.1063 |
| **RPG-FKL** | **0.8836** | 0.1490 | 0.1083 |
| **RPG-RKL** | 0.8672 | 0.1469 | **0.1240** |
| **RPG-UFKL** | 0.8703 | 0.1427 | 0.1177 |
| RPG-REINFORCE-FKL | 0.8727 | **0.1667** | 0.0875 |
| RPG-REINFORCE-URKL | 0.8531 | 0.1500 | 0.0938 |

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| Clip (0.1, 0.1) vs. (0.2, 0.28) | RPG-REINFORCE collapses with large clip parameters on Qwen-Math-7B; smaller clip values yield more stable training |
| AdamW vs. Schedule-Free AdamW | Schedule-Free optimizer improves stability for high-variance algorithms (GRPO, REINFORCE++) |
| Comparison across KL types | FKL achieves best performance on AMC23; RKL achieves best on AIME25; each variant has its advantages |

### Key Findings

- RPG variants substantially outperform GRPO in training stability, with GRPO exhibiting significantly larger fluctuations
- The connection between Forward KL and MLE suggests its regularization effect during training is analogous to that of SFT
- The fully differentiable and REINFORCE-style variants exhibit complementary strengths: the former performs better on AMC23, the latter on AIME24
- Models that have been thoroughly pretrained (e.g., Qwen-Math-7B) require tighter clip parameters to encourage exploitation over exploration
- The Schedule-Free optimizer provides more stable training dynamics through internal parameter averaging

## Highlights & Insights

- **Systematic Framework**: Unifies KL-regularized policy gradient methods under a single framework spanning Forward/Reverse × Normalized/Unnormalized × Differentiable/REINFORCE-style = 8 variants, facilitating understanding and selection for researchers
- **GRPO Correction**: Identifies the theoretical deficiency of missing importance weights in GRPO's KL term and provides a corrected formulation, offering important value for understanding and improving existing RLHF methods
- **Memory Efficiency**: RPG requires only a single model on GPU during training, making it more efficient than GRPO/REINFORCE++, which require both the current policy and a reference policy simultaneously
- **Theoretical Foundation for the $k_3$ Estimator**: Demonstrates that the $k_3$ estimator is equivalent to unnormalized KL divergence, providing theoretical grounding for its widespread use

## Limitations & Future Work

- Validation is limited to mathematical reasoning tasks; code generation and general instruction-following scenarios are not explored
- Experiments are conducted only with 7B-scale models; performance at larger scales remains unverified
- The optimal choice among KL variants depends on the specific task, and clear selection criteria are lacking
- Dual-Clip modifies the original KL-regularized objective, introducing approximation error
- Sensitivity analysis of training hyperparameters (e.g., $\beta$, clip range) is limited; different models may require different configurations

## Related Work & Insights

- **GRPO** (Shao et al., 2024): Uses group-relative advantage estimation and $k_3$ KL penalty, but omits importance weights in the KL term
- **REINFORCE++** (Hu, 2025): Introduces token-level KL penalty and normalization, but the KL term is computed between $\pi_{\text{old}}^{\text{RL}}$ and the SFT policy rather than $\pi_\theta$
- **DAPO** (Yu et al., 2025): Proposes Clip-Higher, oversampling, and token-level loss for training stabilization
- **Dr. GRPO** (Liu et al., 2025): Identifies and corrects bias in GRPO's advantage estimator
- **VAPO** (Yuan et al., 2025): Length-adaptive advantage estimation
- **DPO/SimPO**: Representative works in direct preference optimization, complementary to policy gradient approaches

**Insights**: This work highlights the importance of carefully examining the mathematical correctness of KL regularization when designing RL algorithms, particularly in off-policy settings. A framework-oriented perspective facilitates the identification and correction of theoretical deficiencies in existing methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — The framework itself is a systematic synthesis rather than an entirely novel method, but the GRPO correction and theoretical unification are valuable contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 variants × 2 optimizers × 2 models, with comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear, though the large number of variants makes the presentation somewhat lengthy
- Value: ⭐⭐⭐⭐ — Provides an important theoretical reference and practical methodology for the LLM RL community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](nudging_the_boundaries_of_llm_reasoning.md)
- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)

</div>

<!-- RELATED:END -->
