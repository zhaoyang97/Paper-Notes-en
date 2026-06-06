---
title: >-
  [Paper Note] Coupled Training with Privileged Information and Unlabeled Data
description: >-
  [ICML2026][privileged information] Addressing the "available during training, unavailable during deployment" privileged feature $W$…
tags:
  - "ICML2026"
  - "privileged information"
  - "semi-supervised learning"
  - "negative transfer"
  - "coupled training"
  - "greedy selection"
date: 2026-05-08
content_hash: 3503da9349faa660
---

# Coupled Training with Privileged Information and Unlabeled Data

**Conference**: ICML2026  
**arXiv**: [2605.23268](https://arxiv.org/abs/2605.23268)  
**Code**: Not yet released  
**Area**: Semi-supervised Learning / Privileged Information / Statistical Learning Theory  
**Keywords**: privileged information, semi-supervised learning, negative transfer, coupled training, greedy selection

## TL;DR
Addressing the "available during training, unavailable during deployment" privileged feature $W$, the authors propose a framework for **coupled training of a deployment model $f$ and a rich-view model $g$**. By explicitly constraining the fitting error of $g$ on labeled data to adaptively control the influence of privileged information, the method avoids negative transfer when the $W$ signal is weak or noisy, unlike traditional two-stage pseudo-labeling methods.

## Background & Motivation

**Background**: In scenarios such as medical imaging, longitudinal studies, and transfer learning, "privileged" features $W$ (e.g., expensive biomarkers, expert evaluations, or intermediate variables available only in the future) are often accessible during training, while the deployment model must rely solely on regular features $X$. A popular approach is the LUPI framework proposed by Vapnik, recently extended to non-parametric settings via **two-stage pseudo-labeling** by Xia & Wainwright (2024): the first stage fits a rich-view model $\hat{g}$ using $Z=(X,W)$ on labeled data $\{(Z_i, Y_i)\}_{i=1}^n$; the second stage treats $\hat{g}(Z_j)$ as pseudo-responses for a large unlabeled set $\{Z_j\}_{j=n+1}^N$ to train the deployment model $\hat{f}$ using only $X$.

**Limitations of Prior Work**: While this pipeline reduces sample complexity when privileged signals are strong, it suffers when $W$ is weak, noisy, or contains high-dimensional redundancies. In such cases, the pseudo-responses from the first stage deviate significantly from the true regression function $\mu$. The second stage treats these errors as "ground truth," leading to prediction accuracy that may be worse than training on labeled data alone. This **negative transfer** problem is particularly acute in clinical tasks where expensive privileged variables may not be more predictive than routine checks.

**Key Challenge**: Two-stage methods treat $\hat{g}$ pseudo-responses as "hard targets" with no mechanism for $\hat{f}$ to downweight them if $\hat{g}$ is unreliable. Conversely, ignoring $W$ entirely wastes valuable signals provided by unlabeled samples.

**Goal**: Construct an **adaptive hybrid** mechanism that behaves like the two-stage method (fully utilizing $W$) when privileged signals are strong, and degrades to OLS (using only labeled data) when signals are weak. This transition should be data-driven rather than manually tuned.

**Key Insight**: The authors transform the "pseudo-response" from a hard target into a **bidirectional coupling variable** between $f$ and $g$. While $g$ provides pseudo-responses to $f$, $f$ "recalibrates" $g$ on unlabeled data, while $g$ is constrained to not deviate too far from labeled responses. This draws from co-regularization (Sindhwani et al., 2005) but is tailored for asymmetric privileged information settings.

**Core Idea**: Utilize a **constrained joint convex optimization** to simultaneously learn $f$ and $g$. The constraint level $\nu$ (or $\lambda$ in the dual form) acts as a single knob interpolating between the two extremes of "two-stage" and "OLS."

## Method

### Overall Architecture
Given a labeled set $\mathscr{D}_L=\{(Z_i,Y_i)\}_{i=1}^n$ and an unlabeled set $\mathscr{D}_U=\{Z_j\}_{j=n+1}^N$ (where $Z=(X,W)$ and $m=N-n\gg n$), the goal is to learn a predictor $f$ depending only on $X$. Any $f:\mathcal{X}\to\mathbb{R}$ is lifted to $\mathcal{Z}$ as $\tilde{f}(x,w)=f(x)$. The **constrained joint optimization problem** is:

$$\min_{(f,g)\in\mathcal{F}\times\mathcal{G}} \frac{1}{N}\Big(\sum_{i=1}^n (Y_i-f(X_i))^2 + \sum_{j=n+1}^N (g(Z_j)-f(X_j))^2\Big) \text{ s.t. } \frac{1}{n}\sum_{i=1}^n (Y_i-g(Z_i))^2 \le \nu$$

The first term is the supervised loss of $f$ on labeled data. The second is the "proxy fitting" loss between $f$ and $g$ on unlabeled data (replacing hard injection of pseudo-labels). The constraint forces $g$ to remain a reasonable regressor on labeled data. A smaller $\nu$ approaches the two-stage method, while a larger $\nu$ approaches pure labeled OLS.

### Key Designs

1.  **Alternating Coupled Training Algorithm**:
    *   **Function**: Solves the constrained joint optimization via block coordinate descent, alternating updates between $f$ and $g$.
    *   **Mechanism**: Initialize $g_0$. In step $k$, fix $g_{k-1}$ to solve for $f_k$, then fix $f_k$ to solve for the constrained $g_k$. For convex function classes $\mathcal{F}, \mathcal{G}$, both sub-problems are convex, ensuring monotonic descent and global optimality at cluster points.
    *   **Design Motivation**: Avoids the complexity of optimizing the high-dimensional joint space of $(f,g)$ directly, reducing it to standard sub-problems solvable by existing solvers.

2.  **Lagrangian Dual + Bidirectional Interpolation Perspective**:
    *   **Function**: Relaxes the constraint into a single-parameter penalty form to clarify the interpolation between OLS and two-stage methods.
    *   **Mechanism**: The Lagrangian is $\hat{\mathcal{L}}(f,g;\lambda)=\frac{1}{N}(\sum_i (Y_i-f(X_i))^2 + \sum_j (g(Z_j)-f(X_j))^2 + \lambda\sum_i (Y_i-g(Z_i))^2)$. As $\lambda\to 0$, $g$ has no pressure to fit labeled data, making the unlabeled term ineffective (OLS). As $\lambda\to\infty$, $g$ must fit labeled responses strictly (two-stage). Theorem 2.1 shows $g^\star$ performs a weighted interpolation between the deployment target $\mu$ and the rich-view target $\eta$.
    *   **Design Motivation**: Provides an interpretable "interpolation strength" parameter $\lambda$ and facilitates the derivation of risk bounds.

3.  **Alternating Greedy Forward Selection in High-Dimensional Spaces**:
    *   **Function**: Adapts the algorithm for high-dimensional dictionary-spanned spaces (e.g., sparse linear or additive models) where $p \gg n$.
    *   **Mechanism**: Replaces the alternating minimization sub-problems with greedy forward selection, picking the atom that most reduces residual loss at each step.
    *   **Design Motivation**: Theorem 3.1 proves **global sublinear convergence** ($O(1/T)$) for this approach, extending classic greedy approximation theory to privileged information scenarios.

### Loss & Training
The study focuses on squared loss $\ell(y,y')=(y-y')^2$ for theoretical tractability, though the algorithm is adaptable for classification using soft labels. $\lambda$ is tuned via a validation set.

## Key Experimental Results

### Main Results
Comparing Two-Stage, OLS, and the proposed Coupled method on synthetic Gaussian linear models and real benchmarks:

| Scenario | $\|\theta\|_2$ (Privileged Signal) | Two-Stage | Labeled OLS | Ours (Coupled) |
| :--- | :--- | :--- | :--- | :--- |
| Strong Privileged | Large | Optimal | Significantly worse | Near optimal |
| Weak Privileged | Small | Worse than OLS (Neg. Transfer) | Better | Equal or better than OLS |
| Medium Privileged | Medium | Slightly better than OLS | Baseline | Superior to both |

A key observation is that Coupled never performs worse than the better of the two baselines across the entire signal spectrum.

### Ablation Study
| Configuration | Behavior | Description |
| :--- | :--- | :--- |
| Full Coupled (Moderate $\lambda$) | Lowest Error | Bidirectional coupling with calibrated pseudo-responses. |
| $\lambda\to\infty$ | Two-stage behavior | Potential negative transfer in weak signal scenarios. |
| $\lambda\to 0$ | OLS behavior | Wastes $W$ in strong signal scenarios. |
| High-D Greedy | Near closed-form precision | Validates the convergence of the greedy implementation. |

### Key Findings
*   The optimal $\lambda$ is negatively correlated with signal strength: stronger signals favor larger $\lambda$ (bringing $g$ closer to $\eta$).
*   The risk bound (Corollary 2.3) utilizes a correlation coefficient $\rho_\star \in [0,1]$ to measure residual alignment. Benefit is maximized when $W$ provides information not captured by $X$.
*   Greedy implementations allow the algorithm to scale to thousands of dictionary dimensions while maintaining accuracy.

## Highlights & Insights
*   **Pseudo-labels as Coupling Variables**: Instead of treating pseudo-labels as hard targets, they are treated as quantities that can be recalibrated by $f$. This "soft target + feedback loop" is applicable to knowledge distillation and co-training.
*   **Interpolation Perspective**: Using a single $\lambda$ to bridge OLS and two-stage methods provides a fully interpretable spectrum of algorithmic behavior.
*   **Multiplicative vs. Additive Risk Bounds**: The derivation of a multiplicative relative error bound suggests a more robust theoretical foundation compared to additive absolute error bounds when $g$ is poorly specified.

## Limitations & Future Work
*   Theoretical guarantees are primarily for squared loss; non-asymptotic bounds for classification losses (e.g., logistic) are not yet as rigorous.
*   Parameter tuning for $\lambda$ still relies on validation sets; an automated selection based solely on unlabeled data is missing.
*   Realizability assumptions ($\mu \in \mathcal{F}$) may not strictly hold for deep models, requiring further characterization of risk degradation under model misspecification.

## Related Work & Insights
*   **vs Xia & Wainwright (2024)**: Replaces hard pseudo-labels with coupled variables and explicit consistency constraints to mitigate negative transfer.
*   **vs LUPI (Vapnik & Vashist, 2009)**: Extends privileged information use cases by explicitly incorporating large-scale unlabeled data.
*   **vs Sindhwani et al. (2005) Co-Regularization**: While both use agreement terms, this work focuses on the asymmetric privileged setting where only $f$ is deployed.

## Rating
*   Novelty: ⭐⭐⭐⭐
*   Experimental Thoroughness: ⭐⭐⭐⭐
*   Writing Quality: ⭐⭐⭐⭐
*   Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)
- [\[ICML 2026\] Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases](less_data_faster_training_repeating_smaller_datasets_speeds_up_learning_via_samp.md)
- [\[ICML 2026\] ParalESN: Enabling Parallel Information Processing in Reservoir Computing](paralesn_enabling_parallel_information_processing_in_reservoir_computing.md)
- [\[AAAI 2026\] Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server](../../AAAI2026/others/bipartite_mode_matching_for_vision_training_set_search_from_a_hierarchical_data_.md)
- [\[ICML 2026\] Consistency Training Can Entrench Misalignment](consistency_training_can_entrench_misalignment.md)

</div>

<!-- RELATED:END -->
