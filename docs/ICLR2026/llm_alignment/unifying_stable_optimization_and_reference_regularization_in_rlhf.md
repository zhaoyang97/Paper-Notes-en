---
title: >-
  [Paper Note] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)
description: >-
  [ICLR 2026][LLM Alignment][RLHF] This paper proposes DAR (Dual-regularized Advantage Regression), identifying that reference-model regularization (for preventing reward hacking) and policy stability constraints (for prev…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "RLHF"
  - "dual KL regularization"
  - "advantage regression"
  - "reference policy interpolation"
  - "reward hacking"
date: 2026-05-08
content_hash: 0c715b5c3f7ae3ef
---

# Unifying Stable Optimization and Reference Regularization in RLHF (DAR)

**Conference**: ICLR 2026
**arXiv**: [2602.11523](https://arxiv.org/abs/2602.11523)  
**Code**: [https://github.com/tmllab/2026_ICLR_DAR](https://github.com/tmllab/2026_ICLR_DAR)  
**Area**: Alignment / RLHF
**Keywords**: RLHF, dual KL regularization, advantage regression, reference policy interpolation, reward hacking

## TL;DR
This paper proposes DAR (Dual-regularized Advantage Regression), identifying that reference-model regularization (for preventing reward hacking) and policy stability constraints (for preventing collapse) in standard RLHF progressively conflict, excessively restricting the optimization space. DAR addresses this via a dual-KL objective that interpolates reference policies in log-space and applies a regression transformation to eliminate policy-ratio instability, achieving an average win rate of 92.42% in direct AI alignment and standard RLHF settings, surpassing GRPO by 7.27%.

## Background & Motivation

**Background**: Online RLHF methods (PPO/RLOO/GRPO) optimize LLM policies via reinforcement learning. Two core challenges exist: reward hacking (policies over-optimizing proxy rewards) and training instability (catastrophic policy shifts leading to collapse).

**Limitations of Prior Work**:
- Reward hacking is mitigated via $\text{KL}(\pi_\theta \| \pi_0)$ to constrain the policy toward the initial model.
- Training instability is mitigated via clipping or $\text{KL}(\pi_t \| \pi_\theta)$ to constrain the policy toward the current iterate.
- **Key finding**: These two constraints progressively conflict — the policy must simultaneously remain close to both $\pi_0$ and $\pi_t$, but as training proceeds and $\pi_t$ drifts from $\pi_0$, their intersection shrinks and high-reward policies are excluded.

**Key Challenge**: The conflict between stability constraints and reference regularization leads to an overly restricted optimization space.

**Core Idea**: Unify both constraints via a dynamically interpolated reference policy $\pi_0^\alpha \cdot \pi_t^{1-\alpha}$ in log-space, combined with a regression transformation to eliminate policy-ratio instability.

## Method

### Overall Architecture
The dual-KL alignment objective is: $\mathcal{J} = \max_{\pi_\theta} \mathbb{E}[A(x,y)] - \beta(\alpha \text{KL}[\pi_\theta\|\pi_0] + (1-\alpha)\text{KL}[\pi_\theta\|\pi_t])$, which is equivalent to a single KL constraint against the dynamically interpolated reference $\pi_{\text{ref}} \propto \pi_0^\alpha \pi_t^{1-\alpha}$. This is then converted into a weighted SFT (regression) loss to eliminate RL instability.

### Key Designs

1. **Dual-KL Alignment Objective**:

    - Function: Unifies reward-hacking prevention and training stability constraints.
    - Mechanism (Proposition 4.1): $\alpha \text{KL}[\pi_\theta\|\pi_0] + (1-\alpha)\text{KL}[\pi_\theta\|\pi_t]$ is equivalent to $\text{KL}[\pi_\theta \| \frac{1}{C}\pi_0^\alpha \pi_t^{1-\alpha}]$.
    - Effect: As $\pi_t$ evolves, the interpolated reference automatically tracks high-reward regions, providing improved support coverage.
    - $\alpha$ controls the trade-off: $\alpha \to 1$ is more conservative (closer to the initial model); $\alpha \to 0$ encourages more exploration (closer to the current policy).

2. **Advantage Regression**:

    - Function: Transforms the RL objective into a weighted SFT loss.
    - Closed-form optimal policy (Theorem 4.2): $\pi^* \propto \pi_0^\alpha \pi_t^{1-\alpha} \exp(\frac{1}{\beta}A)$.
    - Practical loss: $\mathbb{E}[(w_{\text{reg}} \cdot w_{\text{adv}}) \cdot \log\pi_\theta(y|x)]$
        - $w_{\text{reg}} = (\pi_0/\pi_t)^\alpha$: regularization weight penalizing deviation from the reference.
        - $w_{\text{adv}} = \exp(\frac{1}{\beta}A)$: advantage weight rewarding high-quality responses.
    - Design Motivation: Avoids the instability of policy ratios in PPO; the regression loss is smoother and more stable.
    - Weight clipping: $\min(w_{\text{reg}} \cdot w_{\text{adv}}, w_{\text{clip}})$ to prevent gradient explosion.

### Loss & Training
- Monte Carlo sampling is used to estimate advantages (avoiding a separate value model).
- Intra-batch advantage normalization is applied.
- Hyperparameters: $w_{\text{clip}} = 20$, $\alpha = 0.1$, $\beta = 0.05$.

## Key Experimental Results

### Main Results: Direct AI Alignment (Qwen2-7B, evaluated by GPT-4-Turbo)

| Method | TL;DR | Helpful | Harmless | Avg. Win Rate |
|--------|-------|---------|----------|---------------|
| DPO (offline) | 67.17% | 81.34% | 77.91% | 75.47 |
| Online DPO | 78.47% | 88.86% | 83.55% | 83.63 |
| GRPO | 83.03% | 86.93% | 85.50% | 85.15 |
| **DAR** | **98.27%** | **93.16%** | **85.84%** | **92.42** |

### Standard RLHF: Qwen2-7B-Instruct

| Method | MT-Bench (GPT-4) | LC% vs $\pi_0$ | Length |
|--------|-----------------|----------------|--------|
| GRPO | 8.425 | 50.50 | 1559 |
| RLOO | 8.409 | 52.25 | 1580 |
| **DAR** | **8.538** | **54.17** | **1358** |

### Ablation Study: Effect of $\alpha$

| $\alpha$ | Effect | Notes |
|----------|--------|-------|
| $\alpha=1.0$ | Conservative, low reward | Fully anchored to initial model |
| $\alpha=0.1$ | **Best balance** | Allows exploration with constraint |
| $\alpha=0.0$ | High reward but reward hacking | 8% missing-EOS rate |

### Key Findings
- **DAR achieves a 98.27% win rate on TL;DR**, representing near-perfect preference alignment.
- **The regression transformation is critical**: the dual-KL variant without regression (DAO) suffers training instability, dual PPO exhibits high variance, and only DAR is stably superior.
- **Sample efficiency**: DAR achieves performance comparable to DAP methods using **half the annotation budget**.
- **Length control**: DAR's output length (1358) closely matches the original model (1340), with no evidence of length hacking.

## Highlights & Insights
- **Insightful discovery of conflicting constraints**: The paper demonstrates that the two regularization families in RLHF (anti-hacking vs. anti-collapse) are progressively adversarial during optimization, explaining why many RLHF methods underperform.
- **Elegant log-space interpolation**: Unifying two KL terms into a single KL against an interpolated reference is theoretically equivalent and empirically releases the optimization space.
- **Regression transformation eliminates RL instability**: Recasting the RL problem as weighted SFT avoids the variance issues of policy-ratio estimation; weight clipping provides additional stability.

## Limitations & Future Work
- **Requires online sampling**: Advantages must be estimated by sampling from the current policy at each step, incurring greater cost than offline DPO.
- **Joint tuning of $\alpha$ and $\beta$**: The Pareto frontier depends on the choice of $(\alpha, \beta)$.
- **Potential improvement**: Integrating the null-space projection from NSPO into DAR's weighted SFT could ensure that safety gradients do not degrade general capabilities.

## Related Work & Insights
- **vs. PPO**: PPO handles the two constraint types independently, leading to conflict; DAR unifies them, improving the Pareto frontier.
- **vs. GRPO**: GRPO eliminates the value model and uses group-relative advantages but still applies policy ratios for RL; DAR removes ratio instability via regression.
- **vs. DPO (offline)**: DPO trains on fixed preference data; DAR uses online sampling with a dynamic reference, yielding stronger generalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the discovery of conflicting constraints and the log-space interpolation solution are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple settings (direct alignment + standard RLHF) × multiple models × multiple evaluators, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and clearly motivated problem formulation.
- Value: ⭐⭐⭐⭐⭐ Provides a new theoretical perspective and a practical solution for RLHF training stability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)
- [\[ICLR 2026\] General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[ICLR 2026\] Swap-guided Preference Learning for Personalized RLHF (SPL)](swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)
- [\[ICML 2026\] UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models](../../ICML2026/llm_alignment/udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)

</div>

<!-- RELATED:END -->
