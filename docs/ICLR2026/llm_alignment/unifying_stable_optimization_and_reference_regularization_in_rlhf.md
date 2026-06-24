---
title: >-
  [Paper Note] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)
description: >-
  [ICLR 2026][LLM Alignment][RLHF] The authors propose DAR (Dual-regularized Advantage Regression). They observe that in standard RLHF, reference model regularization (to prevent reward hacking) and policy stability constraints (to prevent collapse) progressively conflict, leading to an overly restricted optimization space. By defining a dual KL objective that interpolates the reference policy in log-space and applying a regression transformation to eliminate policy-ratio insta…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "RLHF"
  - "Dual KL Regularization"
  - "Advantage Regression"
  - "Reference Policy Interpolation"
  - "Reward Hacking"
date: 2026-05-08
content_hash: 6bd1d19a5a0aea2c
---

# Unifying Stable Optimization and Reference Regularization in RLHF (DAR)

**Conference**: ICLR 2026  
**arXiv**: [2602.11523](https://arxiv.org/abs/2602.11523)  
**Code**: [https://github.com/tmllab/2026_ICLR_DAR](https://github.com/tmllab/2026_ICLR_DAR)  
**Area**: RLHF Alignment  
**Keywords**: RLHF, Dual KL Regularization, Advantage Regression, Reference Policy Interpolation, Reward Hacking

## TL;DR
The authors propose DAR (Dual-regularized Advantage Regression). They observe that in standard RLHF, reference model regularization (to prevent reward hacking) and policy stability constraints (to prevent collapse) progressively conflict, leading to an overly restricted optimization space. By defining a dual KL objective that interpolates the reference policy in log-space and applying a regression transformation to eliminate policy-ratio instability, DAR achieves a 92.42% average win rate in direct AI alignment and standard RLHF settings, outperforming GRPO by 7.27%.

## Background & Motivation

**Background**: Online RLHF (PPO/RLOO/GRPO) optimizes LLM policies via Reinforcement Learning. Two core challenges exist: reward hacking (policy over-optimizing the proxy reward) and training instability (drastic policy shifts causing collapse).

**Limitations of Prior Work**:
   - To prevent reward hacking, $\text{KL}(\pi_\theta||\pi_0)$ is used to constrain the model to the initial policy.
   - To prevent training instability, clipping or $\text{KL}(\pi_t||\pi_\theta)$ is used to constrain it to the current policy.
   - **Key Finding**: These two constraints gradually conflict. The policy must simultaneously stay close to $\pi_0$ and $\pi_t$. However, as training progresses and $\pi_t$ moves away from $\pi_0$, their intersection shrinks, excluding high-reward policies.

**Key Challenge**: The conflict between stability constraints and reference regularization leads to an excessively restricted optimization space.

**Core Idea**: Unify the two constraints using a dynamic reference policy $\pi_0^\alpha \cdot \pi_t^{1-\alpha}$ interpolated in log-space, combined with a regression transformation to eliminate policy-ratio instability.

## Method

### Overall Architecture
DAR addresses the conflict between two types of regularization in RLHF: the constraint to the initial model $\pi_0$ to prevent reward hacking and the constraint to the current policy $\pi_t$ to prevent training collapse. As training proceeds and $\pi_t$ drifts from $\pi_0$, the feasible intersection of these constraints narrows, pushing out high-reward policies. DAR's approach involves two steps: first, formulating the two KL constraints into a dual KL alignment objective with an adjustable weight $\alpha$, and proving this is equivalent to a single KL constraint against a **dynamic interpolated reference** $\pi_0^\alpha\pi_t^{1-\alpha}$. This merges the antagonistic constraints into a reference that evolves with training. Second, the authors derive the closed-form optimal policy for this KL-constrained RL objective and analytically transform it into a **weighted SFT (Advantage Regression) loss**. This bypasses the high variance and instability inherent in PPO-style policy-ratio estimation. Essentially, the derivation proceeds from "unifying dual constraints" to an "interpolated reference," then to a "closed-form optimal policy," and finally to a "weighted regression loss," converting an unstable RL optimization problem into a stable supervised fitting process.

### Key Designs

**1. Dual KL Alignment Objective: Merging "Anti-hacking" and "Anti-collapse" Constraints**

Standard practice adds $\text{KL}[\pi_\theta\|\pi_0]$ (to prevent reward over-optimization) and $\text{KL}[\pi_\theta\|\pi_t]$ (to prevent drastic shifts) separately. DAR linearly combines them using a weight $\alpha$ into a single objective:

$$\mathcal{J} = \max_{\pi_\theta} \mathbb{E}[A(x,y)] - \beta\big(\alpha\,\text{KL}[\pi_\theta\|\pi_0] + (1-\alpha)\,\text{KL}[\pi_\theta\|\pi_t]\big)$$

A critical step is Proposition 4.1: this weighted dual KL is equivalent in log-space to a KL constraint against a single interpolated reference, $\alpha\,\text{KL}[\pi_\theta\|\pi_0] + (1-\alpha)\,\text{KL}[\pi_\theta\|\pi_t] = \text{KL}\big[\pi_\theta \,\big\|\, \tfrac{1}{C}\pi_0^\alpha \pi_t^{1-\alpha}\big]$, where $C$ is a normalization constant. Thus, the true reference policy is $\pi_{\text{ref}} \propto \pi_0^\alpha \pi_t^{1-\alpha}$—a dynamic target that evolves with $\pi_t$, automatically tracking high-reward regions and providing better support coverage rather than being locked to the initial model. The weight $\alpha$ acts as the trade-off knob: $\alpha\to1$ is conservative (closer to initial), while $\alpha\to0$ is exploratory (closer to current).

**2. Advantage Regression: Converting the RL Objective to Weighted SFT to Eliminate Policy-Ratio Variance**

With the single KL objective, the closed-form optimal policy (Theorem 4.2) is $\pi^* \propto \pi_0^\alpha \pi_t^{1-\alpha} \exp(\tfrac{1}{\beta}A)$, which represents exponential advantage weighting over the interpolated reference. Instead of estimating and clipping policy ratios like PPO, DAR fits this optimal solution directly into a weighted log-likelihood (SFT) loss:

$$\mathcal{L} = -\,\mathbb{E}\big[(w_{\text{reg}} \cdot w_{\text{adv}}) \cdot \log\pi_\theta(y|x)\big]$$

Two weights manage the optimization: $w_{\text{reg}} = (\pi_0/\pi_t)^\alpha$ is the regularization weight that penalizes deviations from the reference, and $w_{\text{adv}} = \exp(\tfrac{1}{\beta}A)$ is the advantage weight that rewards good responses. Because the loss is essentially a weighted SFT rather than RL, it bypasses the high variance sources found in PPO's policy ratio estimation, resulting in smoother and more stable gradients. To prevent exponential weights from causing gradient explosions, the product is clipped as $\min(w_{\text{reg}} \cdot w_{\text{adv}},\, w_{\text{clip}})$.

### Loss & Training
- Monte Carlo sampling is used to estimate advantages, avoiding the need for a separate Value model.
- Advantages are normalized within the batch.
- Hyperparameters: $w_{\text{clip}} = 20$, $\alpha = 0.1$, $\beta = 0.05$.

## Key Experimental Results

### Main Results: Direct AI Alignment (Qwen2-7B, evaluated by GPT-4-Turbo)

| Method | TL;DR | Helpful | Harmless | Average Win Rate |
|------|-------|---------|----------|---------|
| DPO (offline) | 67.17% | 81.34% | 77.91% | 75.47 |
| Online DPO | 78.47% | 88.86% | 83.55% | 83.63 |
| GRPO | 83.03% | 86.93% | 85.50% | 85.15 |
| **DAR** | **98.27%** | **93.16%** | **85.84%** | **92.42** |

### Main Results: Standard RLHF (Qwen2-7B-Instruct)

| Method | MT-Bench (GPT-4) | LC% vs $\pi_0$ | Length |
|------|-----------------|-----------|------|
| GRPO | 8.425 | 50.50 | 1559 |
| RLOO | 8.409 | 52.25 | 1580 |
| **DAR** | **8.538** | **54.17** | **1358** |

### Ablation Study: Impact of $\alpha$

| $\alpha$ | Effect | Description |
|---|------|------|
| $\alpha=1.0$ | Conservative, low reward | Completely bound to the initial model |
| $\alpha=0.1$ | **Optimal Balance** | Allows exploration with constraints |
| $\alpha=0.0$ | High reward but reward hacking | 8% missing-EOS rate |

### Key Findings
- **DAR achieves a 98.27% win rate on TL;DR**: Near-perfect preference alignment.
- **Regression transformation is critical**: Direct RL with dual KL (DAO) is unstable, and dual PPO suffers from high variance; only DAR is stable and superior.
- **Sample Efficiency**: DAR achieves performance comparable to DAP methods using **half the annotation volume**.
- **Length Control**: DAR's generation length (1358) is close to the original model (1340), avoiding length hacking.

## Highlights & Insights
- **Profound discovery of constraint conflict**: Points out that the two types of regularization in RLHF (anti-hacking vs. anti-collapse) actually work against each other during optimization. This observation explains why many RLHF methods underperform expectations.
- **Elegant log-space interpolation**: Unifying two KL terms into a single KL against an interpolated reference is theoretically sound and practically liberates optimization space.
- **Regression transformation eliminates RL instability**: Converting the RL problem into weighted SFT avoids variance issues in policy-ratio estimation; weight clipping provides further stability.

## Limitations & Future Work
- **Requires online sampling**: Sampling from the current policy at every step to calculate advantages incurs higher overhead than offline DPO.
- **Co-tuning of $\alpha$ and $\beta$**: The Pareto frontier depends heavily on the selection of $(\alpha, \beta)$.
- **Future directions**: Potential integration with Null-Space Projection (NSPO)—ensuring safety gradients do not impair general capabilities within DAR's weighted SFT framework.

## Related Work & Insights
- **vs. PPO**: PPO treats the two types of constraints independently, leading to conflict; DAR unifies them, lifting the Pareto frontier.
- **vs. GRPO**: GRPO eliminates the value model using group relative advantages but still applies RL via policy ratios; DAR uses regression transformation to eliminate ratio instability.
- **vs. DPO (offline)**: DPO trains on fixed preference data; DAR uses online sampling and dynamic references, leading to stronger generalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of constraint conflict and the log-interpolation solution are both profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive settings (Direct Alignment + Standard RLHF) across multiple models and evaluators, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation and clear problem motivation.
- Value: ⭐⭐⭐⭐⭐ Provides a new theoretical perspective and practical solution for RLHF training stability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)
- [\[ICLR 2026\] Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)
- [\[ICLR 2026\] Learning to Summarize User Information for Personalized RLHF (PLUS)](learning_to_summarize_user_information_for_personalized_reinforcement_learning_f.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](../../ACL2025/llm_alignment/t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)

</div>

<!-- RELATED:END -->
