---
title: >-
  [Paper Note] Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Bayesian Optimization] This paper proposes REBMBO, a framework that unifies Gaussian Processes (local modeling), Energy-Based Models (EBM, global exploration), and PPO-based reinforcement learning (multi-step look-ahead) into a closed-loop Bayesian optimization system, achieving significant improvements over conventional BO methods on high-dimensional and multi-modal black-box optimization tasks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Bayesian Optimization
  - Energy-Based Model
  - PPO
  - Black-Box Optimization
  - Multi-Step Look-Ahead
date: 2026-05-08
content_hash: 9b347ba94a7a7dd2
---

# Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.19530](https://arxiv.org/abs/2510.19530)
**Code**: Unavailable (no explicit link provided)
**Area**: Reinforcement Learning
**Keywords**: Bayesian Optimization, Energy-Based Model, PPO, Black-Box Optimization, Multi-Step Look-Ahead

## TL;DR

This paper proposes REBMBO, a framework that unifies Gaussian Processes (local modeling), Energy-Based Models (EBM, global exploration), and PPO-based reinforcement learning (multi-step look-ahead) into a closed-loop Bayesian optimization system, achieving significant improvements over conventional BO methods on high-dimensional and multi-modal black-box optimization tasks.

## Background & Motivation

**Background**: Bayesian Optimization (BO) is the dominant approach for optimizing expensive black-box functions, centered on a GP surrogate model paired with an acquisition function (e.g., UCB, EI). Notable extensions include TuRBO (local trust regions), BALLET-ICI (alternating global/local GPs), and EARL-BO (RL-assisted multi-step BO).

**Limitations of Prior Work**: Standard BO suffers from severe one-step myopia—optimizing only the expected gain of the current step while neglecting long-term exploration strategies. In high-dimensional or multi-modal settings, this leads to rapid convergence to local optima.

**Key Challenge**: GPs excel at local uncertainty modeling but lack global structural information; multi-step look-ahead methods (e.g., 2-step EI, Knowledge Gradient) are computationally expensive yet still limited in horizon; RL-integrated approaches (e.g., EARL-BO) rely on local posteriors and lack global signals.

**Goal**: Simultaneously address insufficient global exploration and one-step myopia by integrating global structural information and multi-step planning capability into the BO framework.

**Key Insight**: Introduce an EBM to learn a global energy landscape supplementing GP's local modeling, and formulate each BO step as an MDP solved via PPO for adaptive multi-step look-ahead.

**Core Idea**: The EBM provides information on "which regions are globally promising," the GP provides "how certain local estimates are," and PPO performs multi-step planning to leverage both.

## Method

### Overall Architecture

REBMBO consists of three tightly coupled modules (Figure 1):
- **Module A**: GP surrogate model (providing local mean $\mu_{f,t}(\mathbf{x})$ and variance $\sigma_{f,t}(\mathbf{x})$)
- **Module B**: EBM global energy landscape $E_\theta(\mathbf{x})$ (trained via short-run MCMC)
- **Module C**: PPO multi-step planning policy $\pi_{\phi_{ppo}}(\mathbf{a}_t | \mathbf{s}_t)$

All three modules are updated synchronously after each evaluation, forming an adaptive closed loop.

### Key Designs

**Module 1: GP Variants (Module A)**
- REBMBO-C: Classic GP, $\mathcal{O}(n^3)$ exact inference
- REBMBO-S: Sparse GP, $m \ll n$ inducing points, $\mathcal{O}(nm^2)$
- REBMBO-D: Deep kernel GP, $h(\mathbf{x}) = W_2 \psi(W_1 \mathbf{x})$ mapped to latent space
- Composite kernel: $k_f = \sigma_f^2[w_{\text{RBF}} k_{\text{RBF}} + w_{\text{Matérn}} k_{\text{Matérn}}]$, with weights learned automatically via marginal likelihood

**Module 2: EBM Global Exploration (Module B)**
- Function: Learns a global energy landscape where low-energy regions correspond to high-probability/high-promise areas
- Training: MLE via short-run MCMC
  - Positive phase: Lower energy $E_\theta(\mathbf{x}_i)$ at observed data points
  - Negative phase: Generate negative samples via Langevin dynamics and raise their energy
- EBM-UCB acquisition function: $\alpha_{\text{EBM-UCB}}(\mathbf{x}) = \mu_{f,t}(\mathbf{x}) + \beta \sigma_{f,t}(\mathbf{x}) - \gamma E_\theta(\mathbf{x})$
- Design Motivation: The $-\gamma E_\theta(\mathbf{x})$ term biases search toward globally promising regions identified by the EBM, preventing wasted evaluations in uncertain but unpromising areas

**Module 3: PPO Multi-Step Planning (Module C)**
- MDP state: $\mathbf{s}_t = (\mu_{f,t}(\mathbf{x}), \sigma_{f,t}(\mathbf{x}), E_\theta(\mathbf{x}))$
- MDP action: Next query point $\mathbf{a}_t \in \mathcal{X}$
- Reward: $r_t(\mathbf{s}_t, \mathbf{a}_t) = nf(\mathbf{a}_t) - \lambda E_\theta(\mathbf{a}_t)$
- PPO objective: $\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t)]$
- Design Motivation: Generalizes BO from a static single-step selection rule to multi-step planning over an MDP

**Evaluation Metric: Landscape-Aware Regret (LAR)**
$$R_t^{LAR} = [f(\mathbf{x}^*) - f(\mathbf{x}_t)] + \alpha[E_\theta(\mathbf{x}^*) - E_\theta(\mathbf{x}_t)]$$

Setting $\alpha = 0$ recovers standard regret.

### Loss & Training

- GP: Composite RBF+Matérn kernel; hyperparameters optimized via Type-II marginal likelihood
- EBM: Short-run MCMC (10–20 SGLD steps per iteration)
- PPO: 2-layer policy network, 64–256 hidden units, clip $\varepsilon$ to prevent policy collapse
- $\lambda \in [0.2, 0.5]$ serves as a safety band

## Key Experimental Results

### Main Results (Table 1, Synthetic Benchmarks)

| Model | Branin 2D (T=50) | Ackley 5D (T=50) | Rosenbrock 8D (T=50) | HDBO 200D (T=100) | Mean |
|------|:---:|:---:|:---:|:---:|:---:|
| BALLET-ICI | 90.44 | 87.78 | 90.76 | 85.85 | 83.80 |
| EARL-BO | 88.76 | 87.22 | 88.47 | 83.74 | 81.57 |
| TuRBO | 88.63 | 83.79 | 85.74 | 80.69 | 78.56 |
| KG | 91.53 | 90.23 | 90.29 | 85.17 | 87.52 |
| **REBMBO-C** | **97.37** | **94.46** | 96.77 | 90.95 | **89.40** |
| **REBMBO-D** | 95.21 | 91.53 | **96.98** | **94.42** | 89.17 |

### Ablation Study (Appendix)

| Ablated Component | Effect |
|--------|------|
| Remove EBM | Performance degrades; reduces to standard GP-BO |
| Remove PPO multi-step | Degrades to single-step acquisition; significant performance drop in high dimensions |
| Remove short-run MCMC | EBM quality degrades |
| Composite vs. single kernel | RBF+Matérn composite is optimal across all tasks |
| $\lambda \in [0.2, 0.5]$ | Performance is stable; too large or too small $\lambda$ causes degradation |

### Key Findings

1. REBMBO outperforms all baselines on all 6 benchmarks, with the largest advantage on HDBO 200D (REBMBO-D: 94.42 vs. KG: 85.17)
2. REBMBO-D performs best on high-dimensional tasks (deep kernel captures complex latent structure)
3. REBMBO-C performs best on low-dimensional tasks (benefits of exact GP inference)
4. On the Nanophotonic 3D real-world task, convergence is approximately 30% faster
5. Computational overhead is only a small constant factor compared to TuRBO

## Highlights & Insights

1. **Three-Module Synergy**: GP, EBM, and PPO are not simply stacked but tightly coupled—all three are updated synchronously after each evaluation, with the RL policy co-evolving with the latest GP posterior and EBM energy landscape
2. **EBM-UCB**: Directly embedding EBM energy into the UCB acquisition function is elegant and effective; the $-\gamma E_\theta(\mathbf{x})$ term provides a principled global exploration bias
3. **LAR Metric**: Incorporating global exploration quality into regret evaluation yields a more comprehensive measure than standard regret ($\alpha=0$ recovers the standard version, ensuring backward compatibility)
4. **Three GP Variants**: Provide flexible choices for problems of varying scale and complexity

## Limitations & Future Work

1. EBM training may introduce unavoidable errors; the learned energy landscape may be inaccurate when the number of evaluations is very small
2. PPO multi-step planning is sensitive to RL hyperparameters; theoretical convergence rate analysis is left for future work
3. Scale mismatch between the EBM and $f$ may degrade performance (though experiments suggest normalization and adaptive $\lambda$ can mitigate this)
4. Each iteration requires EBM training and PPO updates, incurring higher computational cost than conventional BO (negligible when function evaluations dominate)
5. The sub-linear LAR guarantee in the theoretical section (Appendix E) relies on "mild alignment and regularity assumptions" whose specific conditions are not clearly articulated

## Related Work & Insights

- **vs. EARL-BO**: The latter integrates RL into BO but lacks a global energy signal; REBMBO adds EBM-based global guidance
- **vs. GLASSES**: The latter approximates multi-step loss via forward simulation; REBMBO directly learns a multi-step policy via PPO
- **vs. TuRBO**: The latter excels at local search but lacks long-range jumps; REBMBO achieves global jumps through the EBM
- **Insights**: The idea of using EBMs as global structural priors in BO is transferable to other sequential decision-making scenarios

## Rating

⭐⭐⭐⭐ (4/5)

The method is strongly innovative in design—the three-module synergy of GP, EBM, and PPO represents a new paradigm in BO. The experiments are comprehensive and convincing (6 benchmarks + ablations + real-world task). Weaknesses include theoretical guarantees that rely on strong assumptions, questionable EBM training reliability in low-data regimes, and an overall framework of considerable complexity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimization](metabox-v2_a_unified_benchmark_platform_for_meta-black-box_optimization.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] When Can Model-Free Reinforcement Learning be Enough for Thinking?](when_can_model-free_reinforcement_learning_be_enough_for_thinking.md)

</div>

<!-- RELATED:END -->
