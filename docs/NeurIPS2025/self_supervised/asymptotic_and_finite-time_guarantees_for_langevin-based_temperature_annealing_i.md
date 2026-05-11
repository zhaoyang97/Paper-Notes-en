---
title: >-
  [Paper Note] Asymptotic and Finite-Time Guarantees for Langevin-Based Temperature Annealing in InfoNCE
description: >-
  [NeurIPS 2025 (Optimization for Machine Learning Workshop)][Self-Supervised Learning][Contrastive Learning] By modeling embedding evolution as Langevin dynamics on a compact Riemannian manifold…
tags:
  - "NeurIPS 2025 (Optimization for Machine Learning Workshop)"
  - "Self-Supervised Learning"
  - "Contrastive Learning"
  - "InfoNCE"
  - "Temperature Annealing"
  - "Langevin Dynamics"
  - "Simulated Annealing"
date: 2026-05-08
content_hash: c4c55508c8f714ae
---

# Asymptotic and Finite-Time Guarantees for Langevin-Based Temperature Annealing in InfoNCE

**Conference**: NeurIPS 2025 (Optimization for Machine Learning Workshop)  
**arXiv**: [2603.12552](https://arxiv.org/abs/2603.12552)  
**Code**: None  
**Area**: Self-Supervised Learning / Optimization Theory  
**Keywords**: Contrastive Learning, InfoNCE, Temperature Annealing, Langevin Dynamics, Simulated Annealing

## TL;DR

By modeling embedding evolution as Langevin dynamics on a compact Riemannian manifold, this paper proves that the convergence guarantees of classical simulated annealing extend to the temperature scheduling setting in contrastive learning: a sufficiently slow logarithmic inverse-temperature schedule guarantees probabilistic convergence to the globally optimal representation set, whereas faster schedules risk trapping the system in suboptimal minima.

## Background & Motivation

The InfoNCE loss is the core objective function in contrastive learning (e.g., SimCLR, MoCo), defined as:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(z_i \cdot z_j^+ / \tau)}{\sum_{k=1}^{K} \exp(z_i \cdot z_k / \tau)}$$

The temperature parameter $\tau$ plays a critical role in contrastive learning:
- **Low temperature** ($\tau \to 0$): the loss becomes more sensitive to hard negatives, but gradients may become unstable.
- **High temperature** ($\tau \to \infty$): the loss approaches uniformity and the learning signal weakens.
- **Temperature annealing**: gradually decreasing $\tau$ from high to low values theoretically helps avoid early entrapment in local optima.

Nevertheless, a rigorous theoretical comparison between fixed-temperature and annealing schedules has been lacking. This paper establishes, for the first time, a formal theoretical connection between temperature scheduling in contrastive learning and classical simulated annealing.

## Method

### Overall Architecture

The analytical framework consists of three levels:
1. **Dynamics modeling**: the embedding update process in contrastive learning is modeled as Langevin diffusion on a compact Riemannian manifold $\mathcal{M}$.
2. **Temperature schedule analysis**: the temperature parameter of InfoNCE is treated as an analog of the inverse temperature in simulated annealing.
3. **Convergence proofs**: classical simulated annealing results (Hajek, 1988; Holley et al., 1989) are leveraged to derive convergence guarantees.

**Langevin Dynamics**: the evolution of embedding $z_t$ satisfies the stochastic differential equation:

$$dz_t = -\nabla V(z_t) dt + \sqrt{2/\beta(t)} \, dW_t$$

where $V(z)$ is the potential function induced by the InfoNCE loss, $\beta(t)$ is a time-dependent inverse temperature, and $W_t$ is a Brownian motion.

### Key Designs

**Core Assumptions**:
1. **Compactness**: the embedding space is a compact Riemannian manifold (e.g., the unit sphere $\mathbb{S}^{d-1}$), naturally satisfied after $\ell_2$ normalization in contrastive learning.
2. **Smoothness**: the potential function $V$ satisfies Lipschitz continuity and certain regularity conditions.
3. **Energy barrier assumption**: the depth of the deepest energy barrier $H^*$ determines the required annealing rate for convergence.

**Theorem 1 (Asymptotic Convergence)**: if the inverse-temperature schedule satisfies $\beta(t) \geq \frac{H^* + \epsilon}{\log(t+2)}$ (logarithmic growth), then the embedding converges in probability to the globally optimal representation set $\mathcal{M}^*$:

$$\lim_{t \to \infty} P(z_t \in \mathcal{M}^* \setminus B_\delta) = 0, \quad \forall \delta > 0$$

**Theorem 2 (Finite-Time Guarantee)**: within a finite horizon $T$, the logarithmic schedule yields an $\epsilon$-optimal solution with probability at least $1 - C \cdot T^{-(H^*/\epsilon - 1)}$.

**Theorem 3 (Failure of Fast Schedules)**: if $\beta(t)$ grows faster than the logarithmic rate (e.g., polynomial $\beta(t) = t^\alpha$), there exists a positive probability of becoming trapped in a suboptimal local minimum.

### Loss & Training

This paper is a purely theoretical work and does not involve specific training strategies. The core contribution is establishing the following correspondence:

| Simulated Annealing | Contrastive Learning |
|---|---|
| Energy function $E(x)$ | InfoNCE potential $V(z)$ |
| State space | Embedding manifold (unit sphere) |
| Temperature $T$ | InfoNCE temperature $\tau$ |
| Globally optimal configuration | Optimal contrastive representation |
| Logarithmic cooling schedule | Logarithmic temperature annealing |

## Key Experimental Results

### Main Results

As a workshop paper with a theoretical focus, the experimental section is relatively concise. The authors validate the theoretical conclusions on synthetic data.

**Convergence Comparison of Different Temperature Scheduling Strategies**:

| Annealing Strategy | Schedule Form | Converges to Global Optimum | Convergence Speed |
|---|---|---|---|
| Fixed low temperature | $\tau = 0.05$ | No (trapped in local minimum) | — |
| Fixed high temperature | $\tau = 0.5$ | No (converges to suboptimal solution) | — |
| Logarithmic annealing | $\tau(t) = \tau_0 / \log(t+2)$ | Yes | Slow but stable |
| Linear annealing | $\tau(t) = \tau_0 (1 - t/T)$ | Sometimes (initialization-dependent) | Faster but not guaranteed |
| Exponential annealing | $\tau(t) = \tau_0 \cdot \gamma^t$ | No (overcooling) | Fast but high failure rate |

**Effect of Energy Barrier Height on Convergence**:

| Energy Barrier $H^*$ | Steps Required (Log Annealing) | Success Rate (Linear Annealing) | Optimal Fixed $\tau^*$ |
|---|---|---|---|
| Low ($H^* = 1$) | $\sim 10^3$ steps | 85% | 0.1 |
| Medium ($H^* = 5$) | $\sim 10^5$ steps | 52% | 0.07 |
| High ($H^* = 10$) | $\sim 10^8$ steps | 18% | 0.05 |

### Ablation Study

- **Effect of embedding dimension**: in high-dimensional embedding spaces, energy barriers tend to be lower (a blessing of dimensionality), accelerating convergence of logarithmic annealing.
- **Number of negative samples**: increasing $K$ in InfoNCE smooths the potential function and reduces the effective energy barrier.
- **Manifold curvature**: convergence is faster on compact manifolds with higher positive curvature, consistent with the strong empirical performance of contrastive learning on the unit sphere.

### Key Findings

1. **Logarithmic scheduling is necessary and sufficient**: analogous to the classical result in simulated annealing, temperature annealing in contrastive learning also requires logarithmic slowness.
2. **Limitations of fixed temperature**: regardless of the chosen value, a fixed temperature may fail to reach certain global optima.
3. **Limited practical applicability but profound theoretical significance**: logarithmic annealing is too slow for practical use, but the theoretical connection provides a principled foundation for designing better annealing strategies.

## Highlights & Insights

- **An elegant theoretical connection**: the paper is the first to rigorously establish a mathematical equivalence between temperature scheduling in contrastive learning and simulated annealing.
- **Classical results in a modern context**: the classical theorems of Hajek (1988) et al. on simulated annealing are transplanted into the deep learning setting.
- **Riemannian manifold perspective**: treating the embedding space as a manifold rather than a Euclidean space is more faithful to the $\ell_2$ normalization commonly used in contrastive learning.
- **A theoretical answer to the practical question of what the optimal temperature parameter should be**.

## Limitations & Future Work

1. **Limited scope as a workshop paper**: the completeness of the analysis is constrained, and several technical details are deferred to future work.
2. **Practical infeasibility of logarithmic annealing**: convergence time grows exponentially with the energy barrier, rendering such a slow schedule infeasible in practice.
3. **Applicability of the Langevin dynamics approximation**: SGD updates only approximately satisfy Langevin dynamics, and the gap requires quantification.
4. **Finite negative sample correction not addressed**: practical InfoNCE relies on a finite number of negatives, which deviates from the continuous distribution assumed in theory.
5. **Absence of large-scale experimental validation**: the effect of temperature annealing is not evaluated on standard benchmarks such as ImageNet.

## Related Work & Insights

- **SimCLR** (Chen et al., 2020): empirically finds $\tau = 0.1$ to be optimal but provides no theoretical explanation.
- **Simulated annealing theory** (Hajek, 1988): logarithmic cooling is a necessary and sufficient condition for global convergence.
- **Learnable temperature** (Zhang et al., 2021): treats temperature as a learnable parameter, complementing the annealing approach studied here.
- This paper suggests that future work could explore "nearly logarithmic" yet practically feasible annealing strategies, such as piecewise logarithmic or adaptive annealing schedules.

## Rating

- Theoretical Depth: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Novelty: ⭐⭐⭐⭐⭐
- Practicality: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InfoNCE Induces Gaussian Distribution](../../ICLR2026/self_supervised/infonce_induces_gaussian_distribution.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](../../CVPR2026/self_supervised/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](../../CVPR2026/self_supervised/a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[CVPR 2026\] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](../../CVPR2026/self_supervised/redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)

</div>

<!-- RELATED:END -->
