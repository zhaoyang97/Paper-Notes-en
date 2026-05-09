---
title: >-
  [Paper Note] Personalized Collaborative Learning with Affinity-Based Variance Reduction
description: >-
  [ICLR 2026][Optimization][Personalized Federated Learning] This paper proposes AffPCL, a personalized collaborative learning framework that enables heterogeneous agents to collaboratively learn personalized solutions without prior knowledge, via bias correction and importance correction mechanisms. It achieves an adaptive convergence rate of $O(t^{-1} \cdot \max\{n^{-1}, \delta\})$—yielding linear speedup when agents are similar, and no worse than independent learning when they are dissimilar.
tags:
  - ICLR 2026
  - Optimization
  - Personalized Federated Learning
  - Collaborative Learning
  - Variance Reduction
  - Heterogeneity
  - Affinity-Based Acceleration
date: 2026-05-08
content_hash: e9864c0f5f203f9a
---

# Personalized Collaborative Learning with Affinity-Based Variance Reduction

**Conference**: ICLR 2026
**arXiv**: [2510.16232](https://arxiv.org/abs/2510.16232)
**Code**: None
**Area**: Optimization
**Keywords**: Personalized Federated Learning, Collaborative Learning, Variance Reduction, Heterogeneity, Affinity-Based Acceleration

## TL;DR

This paper proposes AffPCL, a personalized collaborative learning framework that enables heterogeneous agents to collaboratively learn personalized solutions without prior knowledge, via bias correction and importance correction mechanisms. It achieves an adaptive convergence rate of $O(t^{-1} \cdot \max\{n^{-1}, \delta\})$—yielding linear speedup when agents are similar, and no worse than independent learning when they are dissimilar.

## Background & Motivation

### State of the Field

**Background**: Multi-agent learning faces a fundamental tension: **collaboration efficiency vs. personalization needs**. In the presence of heterogeneity, the unified solution pursued by Federated Learning (FL) may be suboptimal or even irrelevant for individual agents. This issue is widespread in practice:

### Limitations of Prior Work

**Limitations of Prior Work**: Personalized recommendation requires adaptation to different users.

### Root Cause

**Key Challenge**: Autonomous driving requires adaptation to local traffic conditions.

### Solution Direction

**Solution Direction**: Medical diagnosis requires adaptation to different patient populations.

### Additional Notes

**Additional Notes**: LLM agents require adaptation to specific user styles.

An ideal personalized collaborative learning algorithm should simultaneously satisfy three objectives:
1. Find a **fully personalized** solution for each agent.
2. Achieve **performance gains** through collaboration.

**Adaptively** handle unknown heterogeneity—accelerate when agents are similar, and not degrade when they are dissimilar.

Shortcomings of existing approaches:

### Additional Notes

**Additional Notes**: **Classic FL**: Learns only a unified solution, with no personalization guarantee.

### Additional Notes

**Additional Notes**: **Regularization/mixture methods**: Provide only partial personalization; the trade-off may be heuristic.

### Additional Notes

**Additional Notes**: **Clustering methods**: No personalization within clusters; sensitive to hyperparameters.

### Additional Notes

**Additional Notes**: **FL + fine-tuning**: Suboptimal rates; small initialization error from FL vanishes quickly.

### Additional Notes

**Additional Notes**: **Global + local decomposition**: Requires specific structural assumptions; the independent learning component dominates overall complexity.

### Additional Notes

**Additional Notes**: **Selective collaboration** (Chayti et al., Even et al.): Collaborates only with similar agents, requiring prior knowledge of heterogeneity or a bias estimation oracle.

## Method

### Overall Architecture

Consider a general multi-agent linear system with $n$ agents:

$$\bar{A}^i x^i_* = \bar{b}^i, \quad i = 1, ..., n$$

where $\text{sym}(\bar{A}^i) \succ 0$. Each agent can only access stochastic observations $A(s^i_t)$ and $b^i(s^i_t)$. Agents may have different objectives (objective heterogeneity) and environment distributions (environment heterogeneity).

### Key Designs

1. **Personalized Bias Correction**:

    - In FL, bias correction is shared across all agents (aligned to a unified direction), enabling federated variance reduction.
    - In personalized learning, bias correction must be tailored to each agent's unique direction, which prevents federated variance reduction.
    - Core update rule of AffPCL: $x^i_{t+1} = x^i_t - \alpha_t \tilde{g}^i_t$
    - Where $\tilde{g}^i_t = g^i_t(x^i_t) + g^0_t(x^0_t) - g^{0 \to i}_t(x^0_t)$
    - Three components: local update + federated aggregation − personalized bias correction.
    - **Design Motivation**: Even when learning fully personalized solutions, variance reduction can still be obtained via federated aggregation.

2. **Importance Correction**:

    - Handles environment heterogeneity (different agents have different environment distributions $\mu_i$).
    - Adjusts for bias introduced by differing environment distributions via importance weights.

3. **Affinity Adaptation**:

    - Heterogeneity measure $\delta \in [0, 1]$, where $\delta = 0$ denotes homogeneity and larger values indicate stronger heterogeneity.
    - Convergence rate: $O(t^{-1} \cdot \max\{n^{-1}, \delta\})$
    - When $\delta \ll n^{-1}$: achieves FL's linear speedup $O(t^{-1} n^{-1})$.
    - When $\delta \gg n^{-1}$: degrades to the independent learning rate $O(t^{-1})$.
    - Intermediate cases are interpolated automatically, without prior knowledge.

### Loss & Training

- Step size strategy: $\alpha \equiv \frac{\ln t}{\lambda t}$ (a variant of constant step size).
- Federated variance reduction is activated by automatically detecting inter-agent affinity.
- Progressive paper organization: simplified FL → personalization → adaptivity → environment heterogeneity → full setting.

## Key Experimental Results

### Main Results

The paper adopts a progressive theoretical analysis. Core theoretical results:

| Setting | Convergence Rate | Note |
|---------|-----------------|------|
| FL (homogeneous) | $\tilde{O}(\kappa^2 t^{-1} n^{-1})$ | Baseline: linear speedup |
| Independent learning | $O(t^{-1})$ | Baseline: no collaboration |
| AffPCL | $O(t^{-1} \cdot \max\{n^{-1}, \delta\})$ | Adaptive interpolation |
| Asynchronous AffPCL | Same + asynchronous importance estimation | Relaxed synchrony requirement |

### Ablation Study

| Configuration | Key Metric | Note |
|--------------|-----------|------|
| Pure FL vs. AffPCL | FL bias does not diminish with $t$ | FL converges to incorrect solution under heterogeneity |
| Independent learning vs. AffPCL | AffPCL variance reduction factor $\max\{n^{-1}, \delta\}$ | No worse than independent learning |
| Selective collaboration vs. AffPCL | AffPCL can achieve speedup even when collaborating with dissimilar agents | The former requires prior knowledge or an oracle |

### Key Findings

1. **Affinity-based variance reduction**: AffPCL enables variance reduction from federated aggregation via personalized bias correction, even when different agents optimize different objectives.
2. **Linear speedup for individual agents**: Even if a given agent is dissimilar to all others, it may still benefit from linear speedup, as similarity among other agents yields better aggregated estimates.
3. **No prior knowledge required**: No need to know the heterogeneity level $\delta$, a bias estimation oracle, or hyperparameter tuning.
4. **Tight theoretical rates**: Matches known lower bounds in $\kappa$, $t$, and $n$.

## Highlights & Insights

- **Elegant theoretical framework**: Formalizes the tension between personalization and collaboration as a unified learning rate problem, with clear and precise analysis.
- **Adaptive interpolation**: Smoothly transitions from FL's linear speedup to the independent learning baseline without parameter tuning.
- **Counterintuitive finding**: Collaborating with dissimilar agents can still be beneficial—challenging the intuition of "only collaborate with similar agents."
- **Strong generality**: The framework covers supervised learning, reinforcement learning (TD learning), and statistical decision-making.

## Limitations & Future Work

1. **Linear system assumption**: The core theory is grounded in the linear system $\bar{A}^i x^i_* = \bar{b}^i$; extension to nonlinear deep learning settings requires further investigation.
2. **Communication efficiency**: The current framework assumes per-step communication; more practical settings should consider intermittent communication and compression.
3. **One sample per agent**: The assumption of one sample per agent per step is idealized.
4. **Privacy considerations**: Performance under constraints such as differential privacy is not discussed.
5. **Experimental scale**: The work is primarily theoretical; large-scale deep learning experiments are limited.

## Related Work & Insights

- **SCAFFOLD** (Karimireddy et al., 2021): Variance reduction in federated learning, but without personalization support.
- **Ditto** (Li et al., 2021): Achieves partial personalization via regularization.
- **MAML/Per-FedAvg** (Fallah et al., 2020): FL + fine-tuning strategies.
- **Chayti et al., Even et al.**: Fully personalized collaborative learning, but requires selective collaboration or prior knowledge.
- **Insight**: The key to variance reduction in personalized learning lies in identifying and leveraging inter-agent affinity, rather than enforcing consensus.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Affinity-driven personalized collaborative learning is an entirely novel perspective)
- Experimental Thoroughness: ⭐⭐⭐ (Primarily theoretical; empirical validation is relatively limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Progressive exposition is exceptionally clear; theory is refined)
- Value: ⭐⭐⭐⭐ (Provides a new theoretical foundation and practical framework for personalized federated learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](../../NeurIPS2025/optimization/personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)
- [\[AAAI 2026\] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](../../AAAI2026/optimization/personalized_federated_learning_with_bidirectional_communication_compression_via.md)
- [\[NeurIPS 2025\] Sharper Convergence Rates for Nonconvex Optimisation via Reduction Mappings](../../NeurIPS2025/optimization/sharper_convergence_rates_for_nonconvex_optimisation_via_reduction_mappings.md)
- [\[ICLR 2026\] SCRAPL: Scattering Transform with Random Paths for Machine Learning](scrapl_scattering_transform_with_random_paths_for_machine_learning.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)

</div>

<!-- RELATED:END -->
