---
title: >-
  [Paper Note] On Entropy Control in LLM-RL Algorithms
description: >-
  [ICLR 2026][Robotics][Entropy control] This paper provides a theoretical explanation for why conventional entropy regularization is nearly ineffective in LLM-RL (due to the extremely large action space and sparse optimal actions causing entropy bias to overwhelm optimization gains), and proposes AEnt — a method combining clamped entropy (computed over a reduced token space) with an adaptive coefficient — to effectively balance bias and benefit, consistently outperforming baselines on mathematical reasoning tasks.
tags:
  - ICLR 2026
  - Robotics
  - Entropy control
  - RLVR
  - LLM-RL
  - policy optimization
  - exploration-exploitation
date: 2026-05-08
content_hash: fb95e96ff7801051
---

# On Entropy Control in LLM-RL Algorithms

**Conference**: ICLR 2026
**arXiv**: [2509.03493](https://arxiv.org/abs/2509.03493)
**Code**: None
**Area**: Robotics
**Keywords**: Entropy control, RLVR, LLM-RL, policy optimization, exploration-exploitation

## TL;DR
This paper provides a theoretical explanation for why conventional entropy regularization is nearly ineffective in LLM-RL (due to the extremely large action space and sparse optimal actions causing entropy bias to overwhelm optimization gains), and proposes AEnt — a method combining clamped entropy (computed over a reduced token space) with an adaptive coefficient — to effectively balance bias and benefit, consistently outperforming baselines on mathematical reasoning tasks.

## Background & Motivation

**Background**: Policy gradient methods (PPO/GRPO/DAPO) dominate LLM-RL. In traditional RL, entropy regularization (SAC/A3C/PPO) prevents premature convergence by maintaining policy stochasticity, with notable success.

**Limitations of Prior Work**: Empirical findings show that entropy regularization yields almost no gain in LLM-RL. Cui et al. observed that varying entropy coefficients has negligible impact on validation accuracy — a striking contrast to its effectiveness in robotics and game-playing RL.

**Key Challenge**: Although entropy regularization offers optimization advantages (improved convergence) in theory, the bias it introduces, $O(H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*(s_0)|^{1/H}})$, grows dramatically with the action space size $|\mathcal{A}|$ and the sparsity of optimal actions. With LLM vocabularies of ~100K tokens and extremely sparse optimal tokens, this bias far exceeds any optimization gain.

**Key Insight**: Since entropy computed over the full vocabulary incurs prohibitively large bias, the paper proposes computing a clamped entropy over a smaller, plausible token subspace — encouraging exploration only among reasonable candidates rather than across the entire vocabulary.

## Method

### Theoretical Analysis

1. **Proposition 1 (Without Entropy Control)**:
   - Policy entropy serves as an upper bound on the policy gradient: $\|\nabla V^{\pi_\theta}\| \leq 2\mathcal{H}(\pi_\theta)$ → entropy collapse leads to learning stagnation.
   - Performance bound: $V^{\pi^*} - V^{\pi_\theta} \leq \frac{\epsilon}{C^{\pi_\theta}(s_0)}$

2. **Proposition 2 (Conventional Entropy Regularization)**:
   - Performance bound: $V^{\pi^*} - V^{\pi_\theta} \leq \frac{\epsilon^2}{2\lambda C_\lambda} + \lambda H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*|^{1/H}}$
   - The optimization term improves ($\epsilon^2/2\lambda$), but the bias term $\lambda H\log|\mathcal{A}|/|\mathcal{A}_H^*|^{1/H}$ dominates in the LLM setting.

### AEnt Method

1. **Clamped Entropy**:
   - Function: Entropy is computed not over the full vocabulary but over the top-$k$ tokens after renormalization.
   - Mechanism: Define subspace $\mathcal{A}_k(s) = \text{top-k tokens}$, renormalize the policy as $\tilde{\pi}(a|s) = \pi(a|s)/\sum_{a' \in \mathcal{A}_k} \pi(a'|s)$, and compute entropy using $\tilde{\pi}$.
   - Design Motivation: Encouraging exploration only among plausible candidates reduces the bias from $\log|\mathcal{A}|$ to $\log k$ (where $k \ll |\mathcal{A}|$).

2. **Adaptive Coefficient**:
   - Function: The coefficient $\lambda$ is automatically adjusted based on the current clamped entropy value.
   - Mechanism: High clamped entropy → small $\lambda$ (policy is already sufficiently stochastic); low clamped entropy → large $\lambda$ (more exploration is needed).
   - Design Motivation: A fixed $\lambda$ cannot adapt to the dynamic changes in entropy throughout training.

### Loss & Training
- $\mathcal{L} = \mathcal{L}_{\text{PO}}(\theta) + \lambda \cdot \min(\mathcal{H}_k(\pi_\theta), H_{\text{target}})$
- The coefficient adapts once the clamped entropy reaches the target level.

## Key Experimental Results

### Mathematical Reasoning

| Method | AIME | AMC | MATH500 | Minerva |
|--------|------|-----|---------|---------|
| GRPO (no entropy) | baseline | baseline | baseline | baseline |
| GRPO + conventional entropy | ~baseline | ~baseline | ~baseline | ~baseline |
| **GRPO + AEnt** | **↑** | **↑** | **↑** | **↑** |

### Multi-Model Validation

| Base Model | AEnt Gain | Notes |
|------------|-----------|-------|
| Qwen2.5-Math-1.5B | Significant | Smaller models benefit more |
| Qwen2.5-7B | Significant | Effective on larger models as well |

### Key Findings
- Conventional entropy regularization indeed yields nearly no gain, corroborating prior observations.
- AEnt consistently improves performance across all benchmarks and models, confirming that clamped entropy effectively resolves the bias issue.
- Synthetic MDP experiments verify that when the number of optimal actions is fewer than 5 and $|\mathcal{A}|=10^5$, conventional entropy fails while AEnt remains effective.
- The adaptive coefficient yields more stable training than a fixed coefficient.

## Highlights & Insights
- **Theoretical resolution of a long-standing LLM-RL puzzle**: Why does conventional entropy regularization not work in LLM-RL? Because the $O(H\log|\mathcal{A}|)$ bias overwhelms all optimization gains when $|\mathcal{A}|=10^5$. This explanation is concise and compelling.
- **Intuition behind clamped entropy**: The model should not be encouraged to explore all 100K tokens indiscriminately; diversity should be maintained only among plausible candidates. Sampling from the top-1,000 tokens is far more reasonable than sampling from the entire vocabulary.
- **Quantifying the bias-gain tradeoff**: Propositions 1 and 2 provide actionable theoretical guidance — when $\log|\mathcal{A}|$ is large and optimal actions are sparse, special treatment is necessary.

## Limitations & Future Work
- The value of $k$ in top-$k$ requires manual specification; an adaptive $k$ selection strategy may be preferable.
- The theoretical analysis assumes a softmax policy, whereas practical LLMs have more complex structures.
- Validation is limited to mathematical reasoning; effectiveness on code generation and general reasoning remains unknown.
- Clamped entropy may overly restrict exploration in scenarios that genuinely require broad search.

## Related Work & Insights
- **vs. DAPO**: DAPO indirectly controls entropy via clipping and constraints, whereas AEnt directly applies regularization in the clamped token subspace.
- **vs. Cui et al.**: Their work observes that entropy bonuses are ineffective but provides no theoretical explanation; this paper offers both an explanation and a solution.
- **vs. SAC**: SAC's entropy regularization succeeds in robotics tasks because $|\mathcal{A}|$ is small (tens to hundreds), whereas LLM action spaces are several orders of magnitude larger.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the theoretical explanation and the clamped entropy approach offer genuine insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models, benchmarks, and synthetic MDP settings.
- Writing Quality: ⭐⭐⭐⭐ Theory and practice are integrated naturally.
- Value: ⭐⭐⭐⭐⭐ Addresses an important practical problem in LLM-RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning](thor_tool-integrated_hierarchical_optimization_via_rl_for_mathematical_reasoning.md)
- [\[ICLR 2026\] PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)
- [\[ICLR 2026\] Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)
- [\[ICLR 2026\] ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](odesteer_a_unified_ode-based_steering_framework_for_llm_alignment.md)
- [\[ICLR 2026\] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)

</div>

<!-- RELATED:END -->
