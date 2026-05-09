---
title: >-
  [Paper Note] Beyond Benign Overfitting in Nadaraya-Watson Interpolators
description: >-
  [NeurIPS 2025][LLM Pretraining][benign overfitting] By tuning a single bandwidth parameter $\beta$ in the Nadaraya-Watson interpolator, this paper precisely characterizes the complete phase transition spectrum from catastrophic overfitting ($\beta < d$) → benign overfitting ($\beta = d$) → tempered overfitting ($\beta > d$), demonstrating that overestimating the intrinsic dimensionality of data is safer than underestimating it.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - benign overfitting
  - Nadaraya-Watson
  - kernel regression
  - interpolation
  - generalization theory
  - overfitting phase transition
date: 2026-05-08
content_hash: c4b90492025e0b3d
---

# Beyond Benign Overfitting in Nadaraya-Watson Interpolators

**Conference**: NeurIPS 2025
**arXiv**: [2502.07480](https://arxiv.org/abs/2502.07480)
**Code**: None
**Area**: Statistical Learning Theory
**Keywords**: benign overfitting, Nadaraya-Watson, kernel regression, interpolation, generalization theory, overfitting phase transition

## TL;DR
By tuning a single bandwidth parameter $\beta$ in the Nadaraya-Watson interpolator, this paper precisely characterizes the complete phase transition spectrum from catastrophic overfitting ($\beta < d$) → benign overfitting ($\beta = d$) → tempered overfitting ($\beta > d$), demonstrating that overestimating the intrinsic dimensionality of data is safer than underestimating it.

## Background & Motivation
**Background**: The phenomenon of benign overfitting—where overparameterized models interpolate noisy training data yet still generalize—has attracted extensive research. Classical analyses focus on consistency, i.e., whether the test error converges asymptotically to zero.

**Limitations of Prior Work**: Mallinar et al. (2022) proposed a finer-grained taxonomy: benign, tempered, and catastrophic overfitting. However, most analyses only distinguish "consistent vs. inconsistent" interpolators, lacking precise characterization of how inconsistent interpolators actually behave.

**Key Challenge**: For the classical Nadaraya-Watson (NW) interpolator, Devroye et al. (1998) established benign overfitting at $\beta = d$, but the behavior for $\beta \neq d$ remains entirely unknown—whether it is tempered or catastrophic.

**Goal**: To fully characterize the type of overfitting exhibited by the NW interpolator across all values of $\beta$, providing exact phase transition conditions.

**Key Insight**: The NW interpolator has a single hyperparameter $\beta$ (controlling the locality/globality of the kernel), making it an ideal object for studying overfitting phase transitions—simple enough to admit precise theory, yet rich enough to exhibit all three overfitting behaviors.

**Core Idea**: By analyzing the locality/globality of the distance weights $\|\mathbf{x} - \mathbf{x}_i\|^{-\beta}$ in the NW interpolator, the paper proves that the relationship between $\beta$ and the data dimensionality $d$ fully determines the type of overfitting.

## Method

### Overall Architecture
The paper considers the NW interpolating predictor $\hat{h}_\beta(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^{m} \frac{y_i}{\|\mathbf{x} - \mathbf{x}_i\|^\beta}\right)$, where training labels are flipped with probability $p$. The goal is to analyze the asymptotic behavior of the "clean" classification error $\mathcal{L}(\hat{h}_\beta) = \lim_{m \to \infty} \Pr[\hat{h}_\beta(\mathbf{x}) \neq f^*(\mathbf{x})]$ under different values of $\beta$.

### Key Designs

1. **Tempered Overfitting ($\beta > d$)**:

    - **Function**: Proves that when $\beta > d$, $C_1 \cdot p^{c(\beta/d)} \leq \mathcal{L}(\hat{h}_\beta) \leq \tilde{O}(p)$.
    - **Mechanism**: When $\beta > d$, the sum of weights $i^{-\beta/d}$ converges, so predictions depend only on a finite number $k$ of nearest neighbors. Contributions from distant samples are negligible. By approximating the distance distribution with an exponential distribution, the NW predictor is shown to be equivalent to a $k$-NN classifier.
    - **Design Motivation**: $\beta > d$ makes the kernel sufficiently local, so only nearby flipped labels affect the prediction, and the error rate naturally scales with the noise level $p$.
    - **Novelty**: Does not rely on spectral analysis; instead, locality of distances is used to argue directly.

2. **Catastrophic Overfitting ($\beta < d$)**:

    - **Function**: Constructs an explicit distribution such that $\mathcal{L}(\hat{h}_\beta) \geq C_2 \cdot c(\beta, d) > 0$, independent of the noise level $p$.
    - **Mechanism**: A distribution is constructed consisting of an inner ball (label $-1$, probability mass $c$) and an outer annulus (label $+1$). When $\beta < d$, the weight sum diverges and the cumulative contribution of the many $+1$ points in the outer region dominates the few $-1$ points nearby, causing the inner ball to be misclassified. The contribution is decomposed into three terms: $T_1$ (inner ball), $T_2$ (expected contribution from outer annulus), and $T_3$ (perturbation), and it is shown that $T_2$ dominates.
    - **Design Motivation**: $\beta < d$ makes the kernel excessively global, so all training samples have non-negligible influence on the prediction, and minority-class regions are overwhelmed by the majority class.

3. **Generalization to Intrinsic Dimensionality**:

    - **Function**: Conjectures that for distributions supported on low-dimensional manifolds, $d$ can be replaced by the intrinsic dimension $d_\text{int}$.
    - **Mechanism**: In the proof, $d$ appears solely to characterize the local probability mass $\int_{B(\mathbf{x},r)} \mu \asymp r^d$.
    - **Practical Implication**: Overestimating $d_\text{int}$ (i.e., $\beta > d_\text{int}$) leads only to tempered overfitting, while underestimating it may be catastrophic.

### Loss & Training
The NW interpolator involves no training process; it directly performs weighted voting over training data at test time.

## Key Experimental Results

### Main Results
Classification error under different $\beta$ and noise levels $p$ on one-dimensional synthetic data ($d=1$, $m=2000$):

| $\beta$ range | $p=0.04$ | $p=0.1$ | $p=0.2$ | Overfitting type |
|--------------|----------|---------|---------|-----------------|
| $\beta < 1$ | ~0.10 | ~0.10 | ~0.10 | Catastrophic (independent of $p$) |
| $\beta = 1$ | ~0.0 | ~0.0 | ~0.0 | Benign |
| $\beta > 1$ | slight increase | moderate increase | notable increase | Tempered (proportional to $p$) |

MNIST 0/1 classification (extrinsic dimension $d_\text{extrinsic}=784$):

| $\beta$ | $p=0.05$ | $p=0.1$ | $p=0.2$ | Notes |
|---------|----------|---------|---------|-------|
| $\beta \approx 8$ | lowest | lowest | lowest | Optimal value corresponds to intrinsic dimension $\approx 8$ |
| $\beta < 8$ | error spikes | error spikes | error spikes | Catastrophic |
| $\beta > 8$ | gradual increase | gradual increase | gradual increase | Tempered |

### Ablation Study

| Configuration | Key Finding | Notes |
|--------------|-------------|-------|
| Spherical data $\mathbb{S}^2 \subset \mathbb{R}^3$ | Optimal $\beta = 2$ (not 3) | Intrinsic dimension governs the phase transition |
| Sphere + Gaussian noise | Optimal $\beta$ gradually shifts from 2 toward 3 | Noise disrupts the low-dimensional structure |
| MNIST | $\beta \approx 8 \ll 784$ | Consistent with intrinsic dimension estimates in the literature |
| Varying sample size | More samples → more pronounced effect | Asymptotic theory holds even in finite-sample regimes |

### Key Findings
- Overfitting behavior is highly asymmetric around $\beta = d$: error grows mildly for $\beta > d$ but becomes abruptly catastrophic for $\beta < d$.
- The optimal $\beta \approx 8$ on MNIST is far below the extrinsic dimension 784, consistent with the intrinsic dimension estimates of Pope et al. (2021).
- Experiments on spheres with added Gaussian noise show that as data deviates from the low-dimensional manifold, the optimal $\beta$ shifts toward the extrinsic dimension.

## Highlights & Insights
- **Complete phase transition spectrum**: To the best of the authors' knowledge, this is the first work to prove that a single kernel method exhibits all three overfitting behaviors (benign / tempered / catastrophic) by varying the bandwidth parameter alone.
- **Practical guidance**: When the intrinsic dimensionality of the data is uncertain, it is preferable to choose a larger $\beta$ (tempered overfitting incurs at most $O(p)$ loss) rather than a smaller one (catastrophic overfitting incurs $\Omega(1)$ loss).
- **Novel proof technique**: Spectral analysis is avoided; instead, the argument is based on "locality"—approximating distance distributions with exponential distributions to equate the NW predictor with a $k$-NN classifier. This technique is transferable to analyzing the generalization behavior of other nonparametric methods.

## Limitations & Future Work
- The theory handles only a fixed noise level $p$; heterogeneous noise settings where $p$ depends on $\mathbf{x}$ are not fully covered.
- The generalization to low-dimensional manifold data remains a conjecture (Remark 5.2); technically, handling nonlinear manifolds requires additional work.
- A gap remains between the upper and lower bounds for tempered overfitting: upper bound $\tilde{O}(p)$ vs. lower bound $\Omega(p^{c(\beta/d)})$, where $c \to \infty$ as $\beta \to d^+$.
- Only classification tasks are analyzed; whether analogous phase transition structures exist for regression tasks is left unexplored.

## Related Work & Insights
- **vs. classical $k$-NN**: $k$-NN with $k>1$ does not interpolate, whereas NW always interpolates but behaves similarly to $k$-NN when $\beta > d$ (tempered overfitting, error $\leq kp$).
- **vs. Kornowski et al. (2024)**: They analyze overfitting types in shallow ReLU networks; the present paper obtains a complete three-phase transition diagram using the simpler NW model.
- **vs. Medvedev et al. (2024)**: They prove that Gaussian kernels are catastrophic under high noise, but exhibit only catastrophic behavior; this paper is the first to demonstrate the full spectrum within a single kernel method.

## Rating
- Novelty: ⭐⭐⭐⭐ First to demonstrate the complete overfitting phase transition spectrum within a single method, though the core tool (exponential approximation of distance distributions) is classical.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments are highly illustrative, but only one real-world dataset (MNIST) is used.
- Writing Quality: ⭐⭐⭐⭐⭐ Proof intuition is clear, figures are excellent, and the exposition progresses systematically from simple to complex.
- Value: ⭐⭐⭐⭐ Makes an important contribution to generalization theory; the practical recommendation to overestimate dimensionality is valuable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](../../ICLR2026/llm_pretraining/a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)
- [\[NeurIPS 2025\] Learning in Compact Spaces with Approximately Normalized Transformer](learning_in_compact_spaces_with_approximately_normalized_transformer.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)

<!-- RELATED:END -->
