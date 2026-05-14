---
title: >-
  [Paper Note] Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)
description: >-
  [ICLR 2026][Optimization][GRPO] This paper proposes VIP (Variance-Informed Predictive allocation), which uses a Gaussian process to predict the success probability of each prompt and then solves a convex optimization pro…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "GRPO"
  - "rollout allocation"
  - "gradient variance"
  - "Gaussian process"
  - "sampling efficiency"
date: 2026-05-08
content_hash: 0c5a15cacb300197
---

# Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)

**Conference**: ICLR 2026  
**arXiv**: [2602.01601](https://arxiv.org/abs/2602.01601)  
**Code**: [https://github.com/HieuNT91/VIP](https://github.com/HieuNT91/VIP)  
**Area**: Optimization  
**Keywords**: GRPO, rollout allocation, gradient variance, Gaussian process, sampling efficiency  

## TL;DR
This paper proposes VIP (Variance-Informed Predictive allocation), which uses a Gaussian process to predict the success probability of each prompt and then solves a convex optimization problem to allocate rollout counts under a compute budget constraint, minimizing gradient variance. VIP consistently improves the sampling efficiency of GRPO/RLOO on mathematical reasoning tasks, achieving up to 12.3-point gains in Pass@32 on AIME24/25.

## Background & Motivation
**Background**: Group-based RL methods such as GRPO/RLOO train LLMs by generating multiple rollouts per prompt and estimating relative advantages. A fixed, uniform number of rollouts (e.g., 16) is typically allocated to each prompt.

**Limitations of Prior Work**: Uniform allocation implicitly assumes all prompts are equally informative. However, prompts with success rates near 0 or 1 yield nearly zero gradient signal (zero variance), wasting the compute budget. Existing filtering approaches require sampling before filtering, potentially negating efficiency gains.

**Key Challenge**: It is necessary to predict, *before* sampling, which prompts are most informative (those with success rates near 0.5 exhibit the highest gradient variance), yet success rates shift as model weights are updated during training.

**Goal**: How should rollouts be optimally allocated across prompts in a mini-batch under a fixed compute budget?

**Key Insight**: (1) Theoretically analyze the relationship between gradient variance and success probability $p$ for Dr.GRPO and RLOO — both are proportional to $p(1-p)$; (2) predict each prompt's $p$ via a Gaussian process; (3) solve for the optimal allocation via convex optimization.

**Core Idea**: Use a GP to predict success probabilities → predict gradient variance → minimize total gradient variance via convex optimization → adaptively allocate rollouts.

## Method

### Overall Architecture
At each training iteration: (1) a GP predicts the success probability of each prompt in the mini-batch based on historical rollout outcomes; (2) a closed-form convex optimization allocates rollout counts under the budget constraint; (3) rollouts are sampled according to the allocation; (4) rollout outcomes are used to update the GP posterior and model parameters.

### Key Designs

1. **Gradient Variance Analysis (Theoretical Contribution)**:

    - Dr.GRPO: $\text{Var}(\tilde{G}) = \frac{n-1}{n^2} 4\sigma_Z^2 p(1-p)$
    - RLOO: $\text{Var}(\tilde{G}) = \frac{1}{n-1} 4\sigma_Z^2 p(1-p)$
    - Key insight: variance is proportional to $p(1-p)$ — prompts with success rate 0.5 carry the highest gradient variance (most informative), while those with success rate 0 or 1 yield no gradient signal.

2. **Gaussian Process Success Probability Prediction**:

    - Function: A GP over prompt embeddings predicts the current success probability of each prompt.
    - Mechanism: Prompts are encoded into 384-dimensional vectors via MiniLM; an RBF kernel models inter-prompt similarity; a sigmoid link function maps latent values to probabilities; recursive Bayesian updates leverage historical rollout outcomes and embedding similarity across prompts.
    - Design Motivation: As a non-parametric model, the GP requires no tracking of model weight changes and adapts naturally through Bayesian updates.

3. **Convex Optimization Allocation**:

    - Function: Minimize total gradient variance subject to a total budget $C$ and per-prompt bounds $[L, U]$.
    - Mechanism: The continuous relaxation admits a closed-form solution (Theorem 5.1/5.2); the Lagrange multiplier $\lambda^*$ is found via bisection, and a greedy heuristic rounds the solution to integer counts.
    - Efficiency: Hashed embeddings and a cached distance matrix make the runtime overhead negligible.

### Loss & Training
VIP integrates as a plug-and-play module with Dr.GRPO/RLOO. Models are trained on DAPO-MATH-17K and evaluated on AIME24/25 under two budget settings (8×Q and 16×Q).

## Key Experimental Results

### Main Results (AIME24/25 Pass@32)

| Model | Method | AIME24 Pass@32 | AIME25 Pass@32 |
|-------|--------|----------------|----------------|
| Qwen2.5-Math-1.5B | RLOO | baseline | baseline |
| | **RLOO+VIP** | **+12.3** | - |
| | Dr.GRPO | baseline | baseline |
| | **Dr.GRPO+VIP** | **improved** | **improved** |
| Qwen2.5-Math-7B | GRPO+VIP | improved (smaller margin) | improved |

### Key Findings
- VIP consistently improves Pass@32 and Mean@32 across all model × baseline × budget configurations.
- Smaller models (1.5B, 3B) benefit more — weaker models are more prone to wasting rollouts on prompts that are too hard or too easy.
- Gains are smaller for the 7B model, as stronger models have success rate distributions more concentrated near the middle.
- GP-predicted success probabilities correlate strongly with actual success rates, validating prediction quality.
- Runtime overhead is negligible — embedding and distance matrix precomputation, GP updates, and convex optimization all run on CPU.

## Highlights & Insights
- **Solid theoretical foundation**: The analysis derives the key $p(1-p)$ relationship from gradient variance, providing a mathematical basis for adaptive allocation.
- **Closed-form solution to the convex allocation problem**: Although the allocation problem is an integer program, the continuous relaxation admits an efficient closed-form solution (bisection + greedy rounding), incurring zero additional overhead in practice.
- **GP is a principled choice**: Prompt embedding similarity enables information sharing — unseen prompts can be predicted from the historical outcomes of similar prompts.

## Limitations & Future Work
- The analysis assumes $\sigma_Z^2$ (projected gradient variance) is identical across all prompts, which may not hold in practice.
- The GP kernel bandwidth is set via the median heuristic, which may be suboptimal.
- Validation is limited to mathematical reasoning (RLVR setting) — the analysis may need modification for RLHF scenarios with noisy reward models.
- GP covariance matrix $\Sigma$ computation and storage may become a bottleneck when the prompt pool is very large.

## Related Work & Insights
- **vs. uniform-allocation GRPO**: VIP is a strict improvement over uniform GRPO, with a theoretical guarantee of lower gradient variance.
- **vs. filtering methods (Yu et al. 2025)**: Filtering discards uninformative prompts after sampling; VIP predicts and allocates before sampling, avoiding waste entirely.
- **vs. heuristic difficulty-based allocation (Zhang et al. 2025)**: VIP offers theoretical optimality guarantees via convex optimization rather than relying on heuristics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — A complete theoretical framework spanning gradient variance analysis, GP prediction, and convex optimization allocation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-model, multi-budget, multi-baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and theorems have closed-form solutions.
- Value: ⭐⭐⭐⭐⭐ — Provides a plug-and-play efficiency improvement tool for GRPO/RLOO training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)
- [\[CVPR 2026\] Dynamic Momentum Recalibration in Online Gradient Learning](../../CVPR2026/optimization/dynamic_momentum_recalibration_in_online_gradient_learning.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)
- [\[NeurIPS 2025\] Efficient Adaptive Experimentation with Noncompliance](../../NeurIPS2025/optimization/efficient_adaptive_experimentation_with_noncompliance.md)

</div>

<!-- RELATED:END -->
