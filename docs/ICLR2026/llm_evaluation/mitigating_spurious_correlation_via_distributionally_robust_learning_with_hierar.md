---
title: >-
  [Paper Note] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets
description: >-
  [ICLR 2026][LLM Evaluation][Spurious correlation] A hierarchical DRO framework is proposed to simultaneously capture inter-group (group proportion shifts) and intra-group (intra-group distributional shifts) uncertainty.…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Spurious correlation"
  - "distributionally robust optimization"
  - "hierarchical ambiguity sets"
  - "Wasserstein distance"
  - "minority group shift"
date: 2026-05-08
content_hash: 693f8dc051e9b51c
---

# Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets

**Conference**: ICLR 2026
**arXiv**: [2510.02818](https://arxiv.org/abs/2510.02818)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: Spurious correlation, distributionally robust optimization, hierarchical ambiguity sets, Wasserstein distance, minority group shift

## TL;DR
A hierarchical DRO framework is proposed to simultaneously capture inter-group (group proportion shifts) and intra-group (intra-group distributional shifts) uncertainty. By defining intra-group ambiguity sets in the semantic space via the $W_\infty$ distance, the method achieves state-of-the-art performance on standard benchmarks and maintains strong robustness under a newly designed minority group distributional shift setting where all competing methods fail.

## Background & Motivation
ERM trained on data with spurious correlations suffers severe performance degradation on minority groups (e.g., waterbirds on land backgrounds). Group DRO alleviates this by minimizing the worst-group loss, but implicitly assumes that the training distribution within each group reliably represents the true distribution.

In practice, minority groups contain very few samples, leading to a large gap between the training and true distributions. This "intra-group distributional shift" has been entirely overlooked in existing spurious correlation research, and even state-of-the-art methods collapse under this setting.

The paper addresses this through hierarchical ambiguity sets: the first level allows arbitrary changes in group proportions (as in Group DRO), while the second level permits distributional shifts within each group bounded by $W_\infty$. A cost function defined in the semantic space renders intra-group shifts both meaningful and computationally tractable.

## Method

### Overall Architecture
A hierarchical extension of Group DRO: $\mathcal{Q} = \{\sum \beta_g Q_g : \beta \in \Delta,\ W_\infty(Q_g, P_g) \leq \varepsilon_g\}$.

### Key Designs

1. **Hierarchical Ambiguity Sets (Eq. 5/8)**: $\rho = \infty$ (arbitrary group proportions) + $\varepsilon_g = \varepsilon / \sqrt{n_g}$ (larger intra-group ambiguity radius for minority groups — smaller sample size permits larger shift).

2. **$W_\infty$ Distance + Semantic Cost**: A cost function defined using penultimate-layer features $z(x)$ is employed, with finite distance only between samples sharing the same label. $W_\infty$ enables support shifts, unlike $f$-divergences which can only reweight.

3. **Surrogate Objective (Thm. 4.1)**: The hierarchical DRO is reformulated as an adversarial perturbation problem in the semantic space: $\max_\beta \sum \beta_g \mathbb{E}_{P_g}[\max_{z': \|z'-z(x)\| \leq \varepsilon_g} L(f_L(z'), y)]$.

4. **Three-Step Coordinate Optimization**: One projected gradient step on $z'$ → exponentiated gradient update on $\beta$ → SGD update on $\theta$. Convergence rate $O(1/\sqrt{T})$.

### Loss & Training
Algorithm 1: PGD on $z'$ + mirror descent on $\beta$ + SGD on $\theta$. Selection of $\varepsilon$ follows a heuristic procedure (Appendix G).

## Key Experimental Results

### Main Results

| Method | Waterbirds WGA | CelebA WGA | CMNIST WGA |
|---|---|---|---|
| ERM | 72.6% | 47.2% | 27.1% |
| Group DRO | 91.4% | 88.9% | 89.3% |
| **Hierarchical DRO** | **92.8%** | **89.5%** | **91.2%** |

### Minority Group Shift Setting (Core Novel Contribution)

| Method | Waterbirds (shifted) | CelebA (shifted) | Note |
|---|---|---|---|
| Group DRO | Collapse | Collapse | Assumes stable intra-group distribution |
| JTT | Collapse | Collapse | Same assumption |
| **Hierarchical DRO** | **Stable** | **Stable** | Intra-group ambiguity sets provide robustness |

### Ablation Study

| Configuration | WGA | Note |
|---|---|---|
| $\varepsilon=0$ (pure Group DRO) | 91.4% | No intra-group robustness |
| $\varepsilon=0.5$ | 92.3% | Moderate intra-group perturbation |
| $\varepsilon=1.0$ | **92.8%** | Optimal |
| $\varepsilon=2.0$ | 92.1% | Overly conservative |
| Raw-space cost (non-semantic) | 90.1% | Semantic space is more effective |

### Key Findings
- Hierarchical DRO marginally outperforms Group DRO under the standard setting.
- The critical distinction arises in the minority group shift setting — the fragility of Group DRO is exposed merely by altering the train/test split without any artificial noise.
- The design $\varepsilon_g = \varepsilon / \sqrt{n_g}$ is intuitively sound: minority groups require a larger protection radius.
- $W_\infty$ vs. $f$-divergence: the former admits support shifts, which is more natural for intra-group distributional shifts.

## Highlights & Insights
- The paper identifies an important failure mode overlooked in the spurious correlation literature — intra-group distributional shift within minority groups.
- The hierarchical ambiguity set provides an elegant unification of Group DRO and standard DRO.
- The newly proposed evaluation setting is itself a significant contribution to the community.

## Limitations & Future Work
- The surrogate objective is an upper bound whose tightness depends on the surjectivity of the feature mapping $z$.
- Selection of $\varepsilon$ still relies on heuristics — adaptive $\varepsilon$ is a potential direction for improvement.
- Computational overhead is slightly higher than Group DRO due to the additional $z'$ optimization step.

## Related Work & Insights
- A natural combination of Group DRO and Wasserstein DRO, with a novel organization and motivation.

## Rating
- Novelty: ⭐⭐⭐⭐ Hierarchical ambiguity set design + new evaluation setting
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated under both standard and novel settings
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous
- Value: ⭐⭐⭐⭐ Meaningful advancement in robust learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)
- [\[CVPR 2026\] Free-Grained Hierarchical Visual Recognition](../../CVPR2026/llm_evaluation/free-grained_hierarchical_visual_recognition.md)
- [\[ICLR 2026\] In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](../../NeurIPS2025/llm_evaluation/leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](../../NeurIPS2025/llm_evaluation/aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)

</div>

<!-- RELATED:END -->
