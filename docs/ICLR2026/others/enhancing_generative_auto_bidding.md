---
title: >-
  [Paper Note] Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search
description: >-
  [ICLR 2026 Oral][auto-bidding] This paper proposes AIGB-Pearl, which introduces an offline trajectory evaluator and a KL-Lipschitz constrained score maximization scheme for generative auto-bidding. The framework enables…
tags:
  - "ICLR 2026 Oral"
  - "auto-bidding"
  - "generative planning"
  - "offline RL"
  - "trajectory evaluator"
  - "KL-Lipschitz constraint"
date: 2026-05-08
content_hash: 3d353ae48768deda
---

# Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search

**Conference**: ICLR 2026 Oral
**arXiv**: [2509.15927](https://arxiv.org/abs/2509.15927)  
**Code**: None  
**Area**: Other
**Keywords**: auto-bidding, generative planning, offline RL, trajectory evaluator, KL-Lipschitz constraint

## TL;DR
This paper proposes AIGB-Pearl, which introduces an offline trajectory evaluator and a KL-Lipschitz constrained score maximization scheme for generative auto-bidding. The framework enables generative models to safely surpass the performance ceiling imposed by static offline data under theoretical guarantees, achieving a significant GMV improvement of +3% on Taobao's real-world advertising system.

## Background & Motivation
**Background**: Auto-bidding is a core technology in online advertising. AI-Generated Bidding (AIGB) models bidding as a conditional trajectory generation task using generative models such as diffusion models, learning the conditional trajectory distribution $p_\theta(\tau|y)$ from offline data. At inference time, high-quality conditions $y^*$ are specified to generate high-return bidding trajectories. AIGB avoids the bootstrapping instability of TD learning and outperforms standard offline RL methods.

**Limitations of Prior Work**: AIGB is essentially conditional behavior cloning—it learns to imitate from offline data without any mechanism to leverage feedback signals to improve generation quality. When inference-time conditions extrapolate beyond the training data distribution, generation quality becomes uncontrollable and may yield risky bidding trajectories. By analogy with LLMs, AIGB corresponds to SFT without the RLHF step.

**Key Challenge**: Applying policy optimization to AIGB (i.e., maximizing evaluator scores) is desirable, yet the evaluator is unreliable outside the offline data support—if the generative model deviates too far from the offline data, evaluator scores become inaccurate (OOD problem), causing optimization to diverge.

**Goal**: How to enable AIGB to improve generation quality through policy optimization while ensuring safety (i.e., not deviating too far from the data)?

**Key Insight**: A theoretical analysis of the upper bound on evaluator bias reveals that the bias can be controlled by two factors: (1) the Lipschitz continuity of the generative model with respect to condition $y$ (controlling extrapolation sensitivity), and (2) the KL divergence between the generative model and the offline data (controlling imitation error).

**Core Idea**: Construct a trajectory evaluator to provide feedback, and impose dual KL-Lipschitz constraints to ensure safe extrapolation, organically unifying generative planning with policy optimization.

## Method

### Overall Architecture
AIGB-Pearl consists of three components: (1) **Generative Planner**: a diffusion-model-based conditional trajectory generator $p_\theta(\tau|y)$ that takes a target quality condition $y^*$ and generates bidding trajectories; (2) **Trajectory Evaluator**: learns $\hat{y}_\phi(\tau)$ via supervised learning on offline data to assess trajectory quality; (3) **Inverse Dynamics Controller**: extracts executable bidding actions from the generated trajectories. During training, the evaluator is first trained on offline data, after which the planner is iteratively improved by maximizing the evaluator scores.

### Key Designs

1. **Trajectory Evaluator**:

    - **Function**: Learns a function $\hat{y}_\phi(\tau)$ to predict the cumulative return (GMV/Budget) of a trajectory.
    - **Mechanism**: Performs supervised regression on the offline dataset $\mathcal{D}$: $\min_\phi \mathbb{E}_{\tau \sim \mathcal{D}}[(\hat{y}_\phi(\tau) - y(\tau))^2]$, with $\sqrt{T}R_m$-Lipschitz regularization to inherit the Lipschitz property of the true trajectory quality function.
    - **Design Motivation**: AIGB lacks feedback signals; the evaluator fills this gap. Lipschitz regularization ensures that evaluator predictions do not exhibit drastic variation outside the data support.

2. **KL-Lipschitz Constrained Score Maximization**:

    - **Function**: Maximizes evaluator scores while constraining planner behavior to prevent OOD collapse.
    - **Mechanism**: The optimization objective is $\max_\theta \mathbb{E}_{\tau \sim p_\theta(\tau|y^*)}[\hat{y}_\phi(\tau)]$, subject to two constraints: (a) KL constraint $\mathbb{E}_{y}[D_{KL}(p_D(\tau|y) \| p_\theta(\tau|y))] \leq \delta_K$, ensuring the planner does not deviate too far from the offline data; (b) Lipschitz constraint $\text{Lip}_{W_1}(p_\theta(\tau|y)) \leq L_p$, controlling the planner's sensitivity to condition $y$.
    - **Design Motivation**: Theorem 2 proves that the upper bound on evaluator bias decomposes into a training error $\delta_D$, a KL divergence term (imitation error), and a Wasserstein distance term (generation sensitivity); the two constraints control the latter two terms respectively.

3. **Synchronous Coupling**:

    - **Function**: Enforces the Lipschitz constraint on the planner during practical training.
    - **Mechanism**: When generating trajectories for two different conditions $y_1, y_2$, the same Gaussian noise sequence $\{\eta_1, ..., \eta_T\}$ is used, and the Lipschitz constant is constrained via the penalty $\hat{W}_1(y_1, y_2; \theta) / |y_1 - y_2| \leq L_p$.
    - **Design Motivation**: Directly computing the Wasserstein distance between distributions is intractable; synchronous coupling converts it into a computable sample-level distance.

### Loss & Training
- Two-stage training: the evaluator is trained first (supervised learning), followed by the planner (constrained optimization).
- The constrained optimization of the planner is converted into an unconstrained problem via Lagrange multipliers.
- The diffusion model variance $\sigma_\theta$ is fixed as a constant to simplify computation of the Lipschitz penalty.

## Key Experimental Results

### Main Results (Simulation Environment, GMV)

| Budget | USCB | BCQ | CQL | DT | DiffBid | **AIGB-Pearl** | Δ |
|--------|------|-----|-----|----|---------|----------------|------|
| 1.5k | 454.25 | 454.72 | 461.82 | 477.39 | 480.76 | **502.98** | +4.62% |
| 2.0k | 482.67 | 483.50 | 475.78 | 507.30 | 511.17 | **521.84** | +2.09% |
| 2.5k | 497.66 | 498.77 | 481.37 | 527.88 | 531.29 | **545.03** | +2.59% |
| 3.0k | 500.60 | 501.86 | 491.36 | 550.66 | 556.32 | **574.17** | +3.21% |

### Real-System A/B Test (Taobao, 6k Advertisers, 19 Days)

| Comparison | GMV Gain | BuyCnt Gain | ROI Gain | Cost Δ |
|------------|---------|-------------|---------|--------|
| vs DiffBid | +3.00% | +2.20% | +1.89% | +1.10% |
| vs DT | +3.30% | +0.64% | +0.16% | +0.66% |
| vs USCB | +3.43% | +0.74% | +4.24% | -0.78% |
| vs MOPO | +3.13% | +2.14% | +4.87% | -1.77% |

### Ablation Study (Real A/B Test, 6k Advertisers, 8 Days)

| Configuration | GMV Change | Note |
|---------------|-----------|------|
| Full AIGB-Pearl | baseline | Complete model |
| w/o KL constraint | -1.09% | GMV drops without KL constraint |
| w/o Lipschitz constraint | -1.81% | Larger drop without Lipschitz constraint |

### Key Findings
- AIGB-Pearl consistently outperforms all baselines across all budget levels, achieving approximately +3% GMV improvement (corresponding to millions of RMB in daily incremental revenue at Taobao scale).
- The contribution of the Lipschitz constraint (+1.8%) exceeds that of the KL constraint (+1.1%), indicating that controlling generation sensitivity to the condition is more critical than constraining deviation from offline data.
- The evaluator achieves AUC 89.9% on training data and 85.5% on OOD data (5-fold CV), demonstrating good generalization.
- For 4k unseen advertisers, AIGB-Pearl maintains a +3% GMV improvement, exhibiting stronger generalization than the original AIGB.
- Removing the dual constraints leads to clearly pathological trajectory behaviors: excessive budget consumption, reversed budget allocation, and insufficient budget utilization.

## Highlights & Insights
- **Unification of Theory and Practice**: The necessity of the KL + Lipschitz dual constraints is rigorously derived from the upper bound on evaluator bias (Theorem 2), and synchronous coupling renders the theoretical constraints computationally tractable. The methodology is transferable to other offline decision-making scenarios using generative models.
- **"RLHF for AIGB"**: The transition from AIGB to AIGB-Pearl perfectly parallels the SFT → RLHF paradigm in LLMs. The evaluator corresponds to the reward model, and the KL constraint corresponds to the policy constraint in PPO.
- **Real-Deployment Validation**: A large-scale 19-day A/B test is conducted on Taobao's real-world advertising system with 6k advertisers, providing industrial-grade validation.

## Limitations & Future Work
- The evaluator is trained on offline data, so its predictive ceiling bounds the potential gains from policy optimization.
- The hyperparameter $L_p$ must be estimated from data; estimation accuracy affects constraint tightness.
- Validation is limited to the advertising bidding domain; transfer to other decision-making settings such as robotic control requires further experimentation.
- Synchronous coupling increases training overhead, as two trajectories must be generated per step for comparison.
- The evaluator and planner are trained in separate stages; whether joint training yields further improvements remains unexplored.

## Related Work & Insights
- **vs DiffBid/AIGB**: The original AIGB performs only conditional behavior cloning without feedback-driven optimization; AIGB-Pearl adds evaluator feedback and constrained optimization.
- **vs Offline RL (CQL/IQL)**: Offline RL relies on TD bootstrapping for value estimation, leading to training instability; AIGB-Pearl uses a supervised-learning evaluator for more stable training.
- **vs MORL**: Model-based offline RL performs conservative search via an environment model; AIGB-Pearl optimizes directly in trajectory space.

## Rating
- Novelty: ⭐⭐⭐⭐ — The framework integrating RL policy optimization into generative planning is novel; the KL-Lipschitz constraint carries theoretical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Combines simulation and large-scale real-system A/B testing on Taobao, with complete ablation and theoretical validation.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and clear.
- Value: ⭐⭐⭐⭐⭐ — A practically deployed system validated at Taobao scale with direct commercial impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States](../../AAAI2026/others/enhancing_control_policy_smoothness_by_aligning_actions_with_predictions_from_pr.md)
- [\[AAAI 2026\] Expressive Temporal Specifications for Reward Monitoring](../../AAAI2026/others/expressive_temporal_specifications_for_reward_monitoring.md)
- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](../../AAAI2026/others/reward_redistribution_via_gaussian_process_likelihood_estimation.md)
- [\[ICLR 2026\] Chart Deep Research in LVLMs via Parallel Relative Policy Optimization](chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization.md)
- [\[ICLR 2026\] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)

</div>

<!-- RELATED:END -->
