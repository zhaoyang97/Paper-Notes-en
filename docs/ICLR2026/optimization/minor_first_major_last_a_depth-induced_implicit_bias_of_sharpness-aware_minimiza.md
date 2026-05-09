---
title: >-
  [Paper Note] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization
description: >-
  [ICLR 2026][Optimization][Sharpness-Aware Minimization] This paper provides a rigorous theoretical analysis of the implicit bias of SAM when training linear diagonal networks, revealing a qualitative phase transition induced by increasing depth from $L=1$ to $L=2$: the limiting direction of $\ell_\infty$-SAM is highly sensitive to initialization, while $\ell_2$-SAM exhibits a **sequential feature amplification** phenomenon — "minor first, major last" — demonstrating that analyses focused solely on the $t\to\infty$ limit are insufficient to characterize the full dynamics of SAM.
tags:
  - ICLR 2026
  - Optimization
  - Sharpness-Aware Minimization
  - implicit bias
  - linear diagonal networks
  - feature amplification
  - depth-induced
date: 2026-05-08
content_hash: f08bf293bcb1ed4e
---

# Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization

**Conference**: ICLR 2026
**arXiv**: [2603.08290](https://arxiv.org/abs/2603.08290)
**Code**: None
**Area**: Optimization Theory
**Keywords**: Sharpness-Aware Minimization, implicit bias, linear diagonal networks, feature amplification, depth-induced

## TL;DR

This paper provides a rigorous theoretical analysis of the implicit bias of SAM when training linear diagonal networks, revealing a qualitative phase transition induced by increasing depth from $L=1$ to $L=2$: the limiting direction of $\ell_\infty$-SAM is highly sensitive to initialization, while $\ell_2$-SAM exhibits a **sequential feature amplification** phenomenon — "minor first, major last" — demonstrating that analyses focused solely on the $t\to\infty$ limit are insufficient to characterize the full dynamics of SAM.

## Background & Motivation

### State of the Field
Sharpness-Aware Minimization (SAM) seeks flat minima by minimizing the worst-case loss within a neighborhood, achieving significant empirical generalization gains. Prior theoretical work has primarily analyzed the implicit bias of SAM in settings with finite minimizers (e.g., squared loss), while the regime in which the infimum of the loss is attained at infinity (e.g., logistic loss) remains poorly understood.

### Limitations of Prior Work & Root Cause
The authors examine the implicit bias of SAM applied to linearly separable binary classification data (under logistic loss) using $L$-layer linear diagonal networks. The key observations are:

- **Depth $L=1$ (linear model)**: Both $\ell_\infty$-SAM and $\ell_2$-SAM converge to the $\ell_2$ max-margin classifier, consistent with gradient descent (GD).
- **Depth $L=2$**: A qualitative change emerges — even on the single-sample dataset $\{(\boldsymbol\mu, +1)\}$ with $\boldsymbol\mu=(1,2)$, the trajectory of SAM can deviate from the $\ell_1$ max-margin direction found by GD.

This observation reveals that adding a single layer can fundamentally alter the implicit bias of SAM.

## Method

### Overall Architecture

The paper adopts a **theoretical analysis + experimental validation** framework:
- **Model**: $L$-layer linear diagonal network $f(\mathbf{x}) = \langle \boldsymbol\beta(\boldsymbol\theta), \mathbf{x}\rangle$, where $\boldsymbol\beta(\boldsymbol\theta) = \bigodot_{\ell=1}^L \mathbf{w}^{(\ell)}$
- **Data**: Linearly separable datasets and the single-sample dataset $\mathcal{D}_{\boldsymbol\mu} = \{(\boldsymbol\mu, +1)\}$
- **Loss**: Logistic loss $\ell(u) = \log(1+\exp(-u))$
- **Analysis tools**: Continuous-time SAM flow (ODE) and rescaled SAM flow

### Key Designs & Theoretical Results

1. **Results for depth $L=1$**

   **Theorem 3.1**: For almost all linearly separable datasets, any perturbation radius $\rho$, and any initialization, the direction of the $\ell_\infty$-SAM flow converges to the $\ell_2$ max-margin direction.

   *Significance*: At depth 1, SAM does not alter the implicit bias of GD.

2. **$\ell_\infty$-SAM for depth $L\geq 2$**

   **Theorem 3.2**: On the single-sample dataset, the behavior of each coordinate $\beta_j(t)$ is entirely determined by the relationship between the initialization $\alpha_j$ and the perturbation radius $\rho$:
   - $\alpha_j < \rho$: $\beta_j(t) \to 0$ (even $L$) or $\to \rho^L$ (odd $L$)
   - $\alpha_j = \rho$: $\beta_j(t) = \rho^L$, remains constant
   - $\alpha_j > \rho$: $\beta_j(t)$ diverges; grows exponentially when $L=2$

   **Corollary 3.5**: The limiting direction is determined by $j^* = \arg\max_{j: \alpha_j > \rho} \mu_j(\alpha_j - \rho)^{L-2}$. This implies that initialization can cause SAM to converge to **any standard basis direction** — including minor feature directions — in sharp contrast to GD, which consistently selects the dominant feature.

3. **$\ell_2$-SAM at depth 2: Sequential Feature Amplification**

   **Theorem 4.2** (Limiting direction): The limiting direction of the $\ell_2$-SAM flow is the $\ell_1$ max-margin solution, identical to GD.

   However, the **finite-time** dynamics differ substantially. The paper identifies the phenomenon of *Sequential Feature Amplification*:

   - **Time dimension** (Theorem 4.4): The predictor $\boldsymbol\beta(t)$ initially **relies on minor coordinates** and gradually **shifts toward major coordinates** as training progresses.
   - **Initialization dimension** (Theorem 4.5): As the initialization scale increases, a similar transition from minor to major features is observed.

   **Key Challenge**: The gradient normalization factor in $\ell_2$-SAM amplifies small gradient coordinates during the perturbation step, causing the corresponding $\beta_j$ to grow faster in the early phase. As training continues, major coordinates eventually dominate due to their larger feature weights $\mu_j$, but this transition is gradual.

### Rescaled Flow Technique

For the single-sample dataset, a rescaled flow is obtained by removing the loss derivative factor $-\ell'(\langle\boldsymbol\beta(\hat{\boldsymbol\theta}(t)),\boldsymbol\mu\rangle) > 0$. This corresponds to a time reparameterization that preserves the spatial trajectory of the original SAM flow while substantially simplifying the analysis.

## Key Experimental Results

### Synthetic Experiments
- The 2D single-sample dataset $\boldsymbol\mu=(1,2)$ clearly illustrates the trajectory differences among GD, $\ell_\infty$-SAM, and $\ell_2$-SAM.
- Multi-sample datasets validate the applicability of theoretical predictions in more realistic settings.
- Experiments at depth $L=3$ confirm that $\ell_\infty$-SAM becomes even more sensitive to initialization in deeper networks.

### Real Network Experiments (MNIST + CNN)

| Method | Grad-CAM Observation | Notes |
|--------|----------------------|-------|
| GD | Focuses on primary digit pixels | Conventional behavior |
| $\ell_2$-SAM | Emphasizes background/weak pixel regions | Consistent with "minor first, major last" theory |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| $L=1$, any $\rho$ | Consistent with GD | SAM does not alter the bias |
| $L=2$, $\ell_\infty$, $\alpha_j < \rho$ | $\beta_j \to 0$ | Coordinate suppressed |
| $L=2$, $\ell_\infty$, $\alpha_j > \rho$ | Exponential growth | Can select minor features |
| $L=2$, $\ell_2$ | Minor before major | Sequential feature amplification |

### Key Findings
- $\ell_\infty$-SAM is highly sensitive to initialization when $L\geq 2$; the perturbation radius $\rho$ acts as a coordinate-level gating threshold.
- The finite-time behavior of $\ell_2$-SAM differs fundamentally from its infinite-time limit — limiting analysis alone is insufficient.
- The behavior of discrete SAM updates closely matches the continuous-time SAM flow, validating the practical relevance of the theory.

## Highlights & Insights

1. **"Minor first, major last" phenomenon**: Reveals a counterintuitive finite-time behavior of $\ell_2$-SAM — the optimizer attends to weak features before strong ones, driven by gradient normalization.
2. **Finite-time vs. infinite-time**: Provides a concrete example demonstrating that implicit bias analyses focused solely on $t\to\infty$ may miss critical dynamical information, calling for greater attention to finite-time analysis.
3. **Qualitative phase transition induced by depth**: Adding a single layer ($L=1\to L=2$) fundamentally alters the behavior of SAM, revealing deep interactions between depth and optimization algorithms.
4. **Exact trajectory characterization**: The coordinate-wise independence of $\ell_\infty$-SAM dynamics enables precise trajectory characterization, yielding an elegant theoretical simplification.

## Limitations & Future Work

- The theoretical analysis is restricted to **linear diagonal networks**, which are simplified proxies for practical deep nonlinear networks.
- Analysis of multi-sample datasets faces additional technical challenges; results in this setting are currently limited to experimental validation.
- The limiting direction theorem for $\ell_2$-SAM (Theorem 4.2) relies on an assumption of directional convergence.
- The implications of sequential feature amplification for generalization in practical deep learning remain unclear.
- SAM variants (e.g., ASAM, GSAM) are not analyzed.

## Related Work & Insights

- **Soudry et al. (2018)**: The classical result that GD on linear models converges to the $\ell_2$ max-margin direction.
- **Gunasekar et al. (2018)**: GD on linear diagonal networks is biased toward $\ell_1$ sparse solutions.
- **Pesme & Flammarion (2023)**: Saddle-to-saddle dynamics of GD — a similar staged learning process, but with a different mechanism.
- **Foret et al. (2020)**: The original SAM paper.
- *Insight*: Finite-time analysis is essential for understanding optimizer behavior; future work should focus more on training dynamics rather than convergence outcomes alone.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — "Sequential feature amplification" is a genuinely novel and counterintuitive finding.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic experiments and MNIST/CNN validation are solid, but large-scale experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations, clear figures, and a thought-provoking contrast between finite-time and infinite-time behavior.
- Value: ⭐⭐⭐⭐ — A significant theoretical advance in understanding SAM, though practical guidance remains limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](../../NeurIPS2025/optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [\[NeurIPS 2025\] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](../../NeurIPS2025/optimization/the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [\[ICLR 2026\] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)
- [\[CVPR 2026\] UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation](../../CVPR2026/optimization/unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source.md)

</div>

<!-- RELATED:END -->
