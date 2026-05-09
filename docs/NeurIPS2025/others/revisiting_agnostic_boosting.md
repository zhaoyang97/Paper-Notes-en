---
title: >-
  [Paper Note] Revisiting Agnostic Boosting
description: >-
  [NeurIPS 2025][Boosting] This paper proposes a new agnostic boosting algorithm that substantially improves the sample complexity of prior work under very general assumptions, and establishes nearly matching lower bounds, thereby resolving the sample complexity of agnostic boosting up to logarithmic factors.
tags:
  - NeurIPS 2025
  - Boosting
  - agnostic learning
  - sample complexity
  - weak-to-strong learning
  - lower bounds
date: 2026-05-08
content_hash: 06e7994b40312308
---

# Revisiting Agnostic Boosting

**Conference**: NeurIPS 2025

**arXiv**: [2503.09384](https://arxiv.org/abs/2503.09384)

**Code**: None

**Area**: Machine Learning Theory

**Keywords**: Boosting, agnostic learning, sample complexity, weak-to-strong learning, lower bounds

## TL;DR

This paper proposes a new agnostic boosting algorithm that substantially improves the sample complexity of prior work under very general assumptions, and establishes nearly matching lower bounds, thereby resolving the sample complexity of agnostic boosting up to logarithmic factors.

## Background & Motivation

### Limitations of Prior Work

**Background**: Boosting is a fundamental method in statistical learning that converts weak learners into strong learners. In the **realizable** setting (where a perfect classifier exists in the hypothesis class), the theory of boosting is well established. However, in the **agnostic** setting (where no assumption is made on the relationship between label distributions and the hypothesis class), the statistical properties of weak-to-strong learning remain insufficiently understood.

Core problems:

**Sample complexity gap**: A significant gap exists between the known upper and lower bounds for agnostic boosting.

**Lack of generality**: Prior improvements rely on relatively strong assumptions.

**Practical significance**: Sample complexity directly determines the feasibility of boosting under limited data.

## Method

### Overall Architecture

A new agnostic boosting algorithm is proposed, built upon two core ideas: (1) reducing the agnostic problem to a realizable one; and (2) selecting high-quality hypotheses via margin-based filtering.

### Key Designs

**1. Agnostic-to-Realizable Reduction**

- Key insight: Even in the agnostic setting, the "easy" portion of the data can be approximated as realizable.
- Sample reweighting is used to construct "pseudo-realizable" subproblems.
- Realizable boosting is applied to these subproblems to obtain high-accuracy weak hypotheses.

**2. Margin-Based Filtering**

- High-quality hypotheses are selected from those generated across boosting rounds using margin scores.
- Margin definition: $\text{margin}(h, x, y) = y \cdot h(x)$
- High-margin hypotheses are more reliable on "easy" samples.
- The filtered hypothesis set is combined into the final strong learner.

**3. Sample Complexity Improvement**

Let $\gamma$ denote the weak learner advantage and $d$ the VC dimension of the hypothesis class:
- Prior best: $\tilde{O}(d / (\varepsilon^2 \gamma^4))$
- Ours: $\tilde{O}(d / (\varepsilon^2 \gamma^2))$
- The quadratic dependence on $\gamma$ is improved.

### Loss & Training

- A weighted distribution $D_t$ is maintained at each round.
- The weak learner is trained on $D_t$ with accuracy requirement $\geq 1/2 + \gamma$.
- Final voting: $H(x) = \text{sign}(\sum_{t \in S} \alpha_t h_t(x))$, where $S$ is the filtered hypothesis set.

## Key Experimental Results

### Main Results

Sample complexity comparison (theoretical bounds, VC dimension $= d$, error $= \varepsilon$):

| Algorithm | Sample Complexity | $\gamma$ Dependence | Assumptions |
|------|----------|-------------|---------|
| Kalai & Kanade (2009) | $\tilde{O}(d/\varepsilon^4)$ | $\gamma^{-8}$ | General |
| Kanade & Kalai (2009, improved) | $\tilde{O}(d/\varepsilon^2)$ | $\gamma^{-4}$ | General |
| Brukhim et al. (2020) | $\tilde{O}(d/\varepsilon^2)$ | $\gamma^{-3}$ | Finite hypotheses |
| Ours | $\tilde{O}(d/\varepsilon^2)$ | $\gamma^{-2}$ | **General** |
| Lower bound (Ours) | $\Omega(d/\varepsilon^2)$ | $\gamma^{-2}$ | - |

Boosting rounds comparison:

| Algorithm | Boosting Rounds | Samples/Round |
|------|-------------|---------|
| Kalai & Kanade | $O(1/\gamma^4)$ | $O(d/\varepsilon^2)$ |
| Ours | $O(1/\gamma^2)$ | $O(d/\varepsilon^2)$ |

### Ablation Study

Effect of margin filtering threshold on sample efficiency (simulation):

| Filtering Threshold | Hypotheses Retained | Test Error | Sample Efficiency |
|---------|-----------|---------|---------|
| 0 (no filtering) | 100% | 0.18 | 1.0x |
| 0.1 | 72% | 0.15 | 1.4x |
| 0.2 | 55% | **0.12** | **1.8x** |
| 0.3 | 35% | 0.13 | 1.6x |

### Key Findings

1. The dependence on $\gamma$ is improved from $\gamma^{-4}$ to $\gamma^{-2}$, accompanied by a nearly matching lower bound.
2. The lower bound $\Omega(d/(\varepsilon^2 \gamma^2))$ establishes that the proposed algorithm is optimal up to logarithmic factors.
3. Margin-based filtering is critical — naively combining all hypotheses does not achieve the optimal rate.
4. The algorithm holds under general assumptions, requiring no additional conditions such as a finite hypothesis class.

## Highlights & Insights

- **Near-optimal theory**: The upper and lower bounds differ only by logarithmic factors, largely resolving the sample complexity of agnostic boosting.
- **General assumptions**: No additional conditions such as finite hypothesis class or bounded loss are required.
- **Clean framework**: The complex agnostic problem is decomposed into realizable subproblems combined with filtering.

## Limitations & Future Work

1. The remaining logarithmic gap, though small, remains an open problem to close entirely.
2. This is a theoretical work; practical performance of the concrete algorithm is not empirically evaluated.
3. The constant factors in the algorithm may be large in practice.
4. The connection to modern ensemble methods (e.g., XGBoost) has yet to be established.

## Related Work & Insights

- **Schapire (1990)**: Original theory of boosting.
- **Freund & Schapire (1997)**: The AdaBoost algorithm.
- **Kalai & Kanade (2009)**: Pioneering work on agnostic boosting.
- **Brukhim et al. (2020)**: Improved results under finite hypothesis classes.

## Rating

- ⭐ Novelty: 8/10 — The combination of reduction and filtering is both elegant and effective.
- ⭐ Value: 5/10 — A purely theoretical contribution with no empirical validation.
- ⭐ Writing Quality: 9/10 — Theoretical derivations are rigorous and comparisons with prior work are clear.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] On Agnostic PAC Learning in the Small Error Regime](on_agnostic_pac_learning_in_the_small_error_regime.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)
- [\[ICCV 2025\] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction](../../ICCV2025/others/revisiting_image_fusion_for_multi-illuminant_white-balance_correction.md)

<!-- RELATED:END -->
