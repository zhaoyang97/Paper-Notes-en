---
title: >-
  [Paper Note] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Multi-objective alignment] This work proposes GAPO, a gradient-adaptive multi-objective policy optimization method. By combining the Multiple Gradient Descent Algorithm (MGDA) with gradient normalization, GAPO balances the trade-offs among conflicting objectives such as helpfulness and harmlessness in LLMs. Furthermore, P-GAPO is introduced to support user-preference-driven Pareto frontier generation.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Multi-objective alignment"
  - "RLHF"
  - "Gradient descent"
  - "Pareto optimality"
  - "Preference optimization"
date: 2026-05-08
content_hash: 09b91249729d6371
---

# Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2507.01915](https://arxiv.org/abs/2507.01915)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Multi-objective alignment, RLHF, Gradient descent, Pareto optimality, Preference optimization

## TL;DR

This work proposes GAPO, a gradient-adaptive multi-objective policy optimization method. By combining the Multiple Gradient Descent Algorithm (MGDA) with gradient normalization, GAPO balances the trade-offs among conflicting objectives such as helpfulness and harmlessness in LLMs. Furthermore, P-GAPO is introduced to support user-preference-driven Pareto frontier generation.

## Background & Motivation
**Background**: RLHF is the mainstream approach to align LLMs with human preferences. However, a single reward function struggles to capture diverse human preferences (such as helpfulness, harmlessness, and honesty), leading to the recent introduction of multi-objective RLHF frameworks.

**Limitations of Prior Work**: (1) Safe RLHF utilizes PPO-Lagrangian for constrained optimization, which is complex and unstable; (2) MORLHF combines multiple rewards via linear scalarization, which inherently cannot fully explore the Pareto frontier; (3) Conflicting objectives often lead to over-safety behaviors (i.e., harmless but useless responses).

**Key Challenge**: The inherent conflict between objectives in multi-objective RLHF—optimizing one objective often sacrifices another, and linear scalarization fails to handle gradient scale discrepancies.

**Goal**: Design an alignment algorithm that can adaptively balance multiple conflicting objectives to prevent imbalanced optimization.

**Key Insight**: Drawing from multi-objective optimization theory, MGDA is used to determine the update direction, with gradient normalization applied to eliminate scale differences among different objectives.

**Core Idea**: Use gradient adaptive rescaling to rectify the imbalance in MGDA, allowing the update direction to automatically focus on under-optimized objectives.

## Method

### Overall Architecture
GAPO models LLM alignment as a multi-objective optimization problem, differing from MORLHF methods that utilize linear scalarization. It adopts PPO as the base RL algorithm but determines the update direction via MGDA, introducing a gradient rescaling mechanism to balance the optimization progress of different objectives.

### Key Designs
1. **MGDA Foundation**: Solving $\min_{\alpha} \|\sum \alpha_i \nabla J_i\|^2$ s.t. $\sum \alpha_i = 1, \alpha_i \geq 0$ yields a Pareto descent direction that improves all objectives simultaneously. However, MGDA forces uniform optimization progress (Theorem 3.1), which restricts the optimization space for other objectives when some are already near-optimal.

2. **Gradient Rescaling**: The original gradient is divided by its $L_2$ norm raised to the power of $p$: $\nabla^N J_i = \nabla J_i / \|\nabla J_i\|^{2p}$. $p=1$ represents standard normalization, whereas $p=2$ makes the rescaled gradient length inversely proportional to its original length. MGDA is solved again using these rescaled gradients (Problem 7). Theorem 3.3 proves that the optimization amount of each objective is proportional to the $p$-th power of its gradient norm, meaning objectives with larger gradients (i.e., greater optimization needs) receive more optimization.

3. **P-GAPO (Preference-based GAPO)**: A user preference vector $\lambda = (\lambda_r, \lambda_c)$ is introduced. Gradients are first normalized with $p=1$, and then linearly combined using preference weights to form the update direction: $\theta' = \theta + \eta \sum \lambda_i \nabla^N J_i$. Multiple models are trained with different $\lambda$ values to cover the Pareto frontier.

### Loss & Training
It optimizes KL-regularized multi-objective rewards based on PPO. The helpfulness objective uses a reward model $R_{\phi}$, and the harmlessness objective uses the negative value of a cost model $-C_{\psi}$ (beaver-7b-v1.0 series). Mistral-7B-SFT is used as the base model. To reduce spatial complexity (from $O(mn)$ to an approximation), gradients are computed only for the parameters of the last layer. P-GAPO uses $\lambda_r \in \{0.2, 0.4, 0.6, 0.8\}$.

## Key Experimental Results

### Main Results

Mistral-7B results on the PKU-SafeRLHF and HH-RLHF test sets:

| Method | PKU Helpful↑ | PKU Harmless↑ | PKU Avg↑ | HH Helpful↑ | HH Harmless↑ | HH Avg↑ |
|------|-------------|---------------|----------|-------------|---------------|---------|
| SFT | 2.33 | 1.78 | 2.06 | 5.75 | 5.25 | 5.50 |
| PPO-H | 9.52 | -11.77 | -1.12 | 9.44 | -5.02 | 2.21 |
| PPO-S | -7.21 | 13.05 | 2.92 | -2.53 | 13.07 | 5.27 |
| Safe RLHF | 5.02 | 1.83 | 3.42 | 7.62 | 5.70 | 6.66 |
| Fast RL | 6.93 | 6.08 | 6.50 | 12.37 | 8.54 | 10.45 |
| MGDA | 7.34 | 5.94 | 6.64 | 10.90 | 8.22 | 9.56 |
| **GAPO p=1** | **7.48** | **7.92** | **7.70** | **12.56** | **9.82** | **11.19** |
| GAPO p=2 | 7.67 | 6.81 | 7.24 | 12.87 | 9.58 | 11.23 |

### Ablation Study

Proportion of harmless responses for different methods (PKU-SafeRLHF):

| Method | Harmless.ratio |
|------|---------------|
| PPO-H | 38.88% |
| PPO-S | 99.48% |
| Safe RLHF | 68.35% |
| MGDA | 79.64% |
| GAPO p=1 | **83.82%** |
| GAPO p=2 | 82.19% |

### Key Findings
- GAPO with $p=1$ consistently performs best on the Avg metric, achieving high performance on both helpfulness and harmlessness without sacrificing either.
- Single-objective PPO exhibits extreme behaviors: PPO-H achieves the highest helpfulness but extremely poor harmlessness, and vice versa for PPO-S. This validates the necessity of multi-objective alignment.
- Safe RLHF's PPO-Lagrangian method fails to further optimize harmlessness when the base model already possesses some level of safety—the Lagrangian multiplier rapidly decreases.
- GPT-4o evaluation confirms that GAPO with $p=1$ significantly outperforms the SFT baseline in both helpfulness and harmlessness.
- P-GAPO outperforms MORLHF under balanced preferences ($\lambda_r \in \{0.4, 0.6\}$), while MORLHF performs slightly better under extreme preferences (as it directly optimizes the primary objective).
- Both the Pareto frontiers of P-GAPO and MORLHF completely dominate Rewarded Soups (RS), indicating that simply merging the weights of single-objective models yields limited effectiveness.

## Highlights & Insights
- The analysis of MGDA's imbalance issue (Theorem 3.1) is profound, and the proposed gradient rescaling solution is theoretically elegant and simple to implement.
- The choice of $p$ offers flexibility: standard normalization with $p=1$ is notably better in GPT-4o evaluation, while $p=2$ performs slightly better in pure reward model evaluation.
- P-GAPO generates a high-quality Pareto frontier using a simple combination of gradient normalization and preference weighting.
- The practical technique of using only the last-layer gradients to approximate full gradients effectively reduces the computational overhead of MGDA in LLMs.
- The analysis of Safe RLHF's failure when the model already possesses some safety is insightful, highlighting the limitations of constrained optimization.
- The extreme performances of PPO-H and PPO-S (one dimension being excellent while the other is extremely poor) intuitively demonstrate the severity of multi-objective conflicts.
- The conclusion of Theorem 3.3 is elegantly formulated: the optimization amount is proportional to the $p$-th power of the gradient's $L_2$ norm, meaning that objectives needing more optimization automatically receive more resources.
- The result where Rewarded Soups is fully dominated suggests that linear interpolation in weight space is too coarse.

## Limitations & Future Work
- The method is only validated on Mistral-7B; differences in initial capabilities across various LLMs may affect the optimization balance.
- There is a lack of methods to evaluate whether the LLM's responses accurately reflect the user's preference weights.
- Currently, only two objectives (helpfulness and harmlessness) are considered, leaving settings with three or more objectives (e.g., adding honesty, conciseness) unexplored.
- The approximation accuracy of using only last-layer gradients lacks theoretical guarantees and may lose information in deep networks.
- P-GAPO requires training a model for each preference vector, incurring costs that scale linearly with preference granularity.
- It has not been compared with multi-objective methods under the DPO family (such as MODPO).

## Related Work & Insights
- The philosophy is consistent with gradient conflict mitigation methods in multi-task learning (e.g., GradNorm, PCGrad, Nash-MTL, CAGrad), but this is the first to systematically apply them to LLM alignment.
- Constrained optimization in Safe RLHF and linear scalarization in MORLHF both have theoretical limitations; GAPO provides a third paradigm.
- The preference Pareto frontier of P-GAPO provides a foundation for personalized alignment.
- The weight interpolation approach of Rewarded Soups is shown to have limited efficacy, suggesting gradient-level intervention is more effective.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing MGDA + gradient rescaling into multi-objective LLM alignment, with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, multiple baselines, GPT-4o evaluation, and Pareto frontier analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Strong integration of theoretical derivations and experiments, with well-formulated theorems.
- **Value**: ⭐⭐⭐⭐ Practical constructive significance for LLM safety alignment, with potential scalability to more objectives.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)
- [\[ACL 2025\] GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)
- [\[ACL 2025\] DeAL: Decoding-time Alignment for Large Language Models](deal_decoding_time_alignment.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)
- [\[ACL 2025\] GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization](gapo_preferential_prompt.md)

</div>

<!-- RELATED:END -->
