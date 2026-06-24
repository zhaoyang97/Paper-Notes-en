---
title: >-
  [Paper Note] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras
description: >-
  [ICML2025][Equivariant Neural Networks] This paper proposes Generalized Lipschitz Group Equivariant Neural Networks (GLGENN), which leverage weight sharing across four fundamental subspaces defined by grade involution and reversion in geometric algebra. While maintaining equivariance to pseudo-orthogonal groups, GLGENN significantly reduces trainable parameters (to approximately 1/2 to 1/3 of CGENN) and matches or key performance metrics of CGENN on multiple benchmark tasks.
tags:
  - "ICML2025"
  - "Equivariant Neural Networks"
  - "Clifford Algebras"
  - "Geometric Algebra"
  - "Lipschitz Groups"
  - "Pseudo-Orthogonal Groups"
  - "Weight Sharing"
  - "Parameter-Efficient"
date: 2026-05-08
content_hash: 398cb26621a2eff7
---

# GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras

**Conference**: ICML2025  
**arXiv**: [2506.09625](https://arxiv.org/abs/2506.09625)  
**Code**: [GitHub](https://github.com/katyafilimoshina/glgenn)  
**Area**: Equivariant Neural Networks / Geometric Deep Learning  
**Keywords**: Equivariant Neural Networks, Clifford Algebras, Geometric Algebra, Lipschitz Groups, Pseudo-Orthogonal Groups, Weight Sharing, Parameter-Efficient

## TL;DR

This paper proposes Generalized Lipschitz Group Equivariant Neural Networks (GLGENN), which leverage weight sharing across four fundamental subspaces defined by grade involution and reversion in geometric algebra. While maintaining equivariance to pseudo-orthogonal groups, GLGENN significantly reduces trainable parameters (to approximately 1/2 to 1/3 of CGENN) and matches or key performance metrics of CGENN on multiple benchmark tasks.

## Background & Motivation

Equivariant Neural Networks (ENNs) have been widely applied in molecular property prediction, particle physics, protein structure analysis, and robot planning by explicitly embedding symmetries (e.g., rotation, reflection) into network architectures. Equivariant networks based on Clifford Geometric Algebra (GA), such as CGENN, represent a major direction in recent years, achieving equivariance to pseudo-orthogonal transformations via Lipschitz groups and twisted adjoint representations.

However, existing GA-equivariant networks face **overparameterization** issues:
- CGENN decomposes multivectors into $n+1$ subspaces according to a fixed grade $k=0,1,...,n$, with each subspace independently parameterized.
- The number of parameters grows rapidly as the dimension $n$ increases.
- Overfitting easily occurs in small-data regimes, which are common in natural sciences.
- Training times prolong as the number of parameters escalates.

The core insight of this work is that grade involution ($\hat{\cdot}$) and reversion ($\tilde{\cdot}$) are the two most fundamental involution operations in GA. They naturally partition GA into **4 fundamental subspaces** $C\ell^{\bar{k}}$ ($k=0,1,2,3$) instead of $n+1$ grade subspaces. Based on this coarser-grained decomposition, one can design networks with significantly fewer parameters that still preserve equivariance.

## Method

### Core Mathematical Framework: Generalized Lipschitz Groups

**Geometric Algebra Basics**: Given a vector space $V=\mathbb{R}^{p,q,r}$, an element (multivector) in its Clifford geometric algebra $C\ell_{p,q,r}$ can be decomposed by grade: $U = \langle U \rangle_0 + \langle U \rangle_1 + \cdots + \langle U \rangle_n$.

**Four Fundamental Subspaces**: Defined by the sign patterns of grade involution and reversion:

$$C\ell^{\bar{k}}_{p,q,r} := C\ell^k \oplus C\ell^{k+4} \oplus C\ell^{k+8} \oplus \cdots, \quad k=0,1,2,3$$

| Subspace | Grade involution $\hat{\cdot}$ | Reversion $\tilde{\cdot}$ |
|--------|:---:|:---:|
| $C\ell^{\bar{0}}$ | $+$ | $+$ |
| $C\ell^{\bar{1}}$ | $-$ | $+$ |
| $C\ell^{\bar{2}}$ | $+$ | $-$ |
| $C\ell^{\bar{3}}$ | $-$ | $-$ |

**Generalized Lipschitz Group** $\tilde{\Gamma}^{\bar{1}}_{p,q,r}$: The set of invertible elements that preserve the four subspaces $C\ell^{\bar{k}}$ under the twisted adjoint representation. Key theorems:

- **Theorem 3.1**: The standard Lipschitz group $\tilde{\Gamma}^1 \subseteq \tilde{\Gamma}^{\bar{1}}$, meaning the generalized group contains the standard group.
- **Theorem 3.4**: A $\tilde{\Gamma}^{\bar{1}}$-equivariant map is automatically equivariant to the pseudo-orthogonal group $O(V,\mathfrak{q})$.

### GLGENN Layer Design

**1. $C\ell^{\bar{k}}$-Linear Layer** (replacing the $C\ell^k$-linear layer of CGENN):

$$\langle y_{c_{out}} \rangle_{\bar{k}} := \sum_{c_{in}=0}^{l} \phi_{c_{out} c_{in} \bar{k}} \langle x_{c_{in}} \rangle_{\bar{k}}$$

Parameter count: $4lm$ ($l$ input channels, $m$ output channels), whereas CGENN requires $(n+1)lm$ parameters.

**2. $C\ell^{\bar{k}}$-Geometric Product Layer** (second-order interaction terms):

$$P(x_1, x_2)^{\bar{k}} := \sum_{i=0}^{3} \sum_{j=0}^{3} \phi_{ijk} \langle \langle x_1 \rangle_{\bar{i}} \langle x_2 \rangle_{\bar{j}} \rangle_{\bar{k}}$$

Parameter count: $4l^2 + 4^3 l$, whereas CGENN requires $(n+1)l^2 + (n+1)^3 l$.

**3. $C\ell^{\bar{k}}$-Normalization Layer**:

$$\langle x \rangle_{\bar{k}} \mapsto \frac{\langle x \rangle_{\bar{k}}}{\sigma(\phi_{\bar{k}})(\langle \widetilde{\langle x \rangle_{\bar{k}}} \langle x \rangle_{\bar{k}} \rangle_0 - 1) + 1}$$

Parameter count: $4l$ (CGENN requires $(n+1)l$).

**Key to Parameter Savings**: CGENN projects multivectors onto $n+1$ grade subspaces and processes them independently, while GLGENN projects them onto only 4 fundamental subspaces. When $n \geq 4$, the parameter count is significantly reduced (the partitioning step size effectively changes from 1 to 4).

## Key Experimental Results

### O(5,0)-Regression Task

Estimating the function $\sin(\|x_1\|) - \|x_2\|^3/2 + \frac{x_1^T x_2}{\|x_1\|\|x_2\|}$ for $x_1,x_2 \in \mathbb{R}^{5,0}$:

| Model | 30 samples | 300 samples | 3000 samples | 30000 samples |
|------|:------:|:------:|:-------:|:--------:|
| **GLGENN** | **0.1055** | **0.0020** | 0.0031 | 0.0011 |
| CGENN | 0.0791 | 0.0089 | **0.0012** | **0.0003** |
| EMLP-O(5) | 0.152 | 0.0344 | 0.0310 | 0.0273 |
| MLP | 28.10 | 0.248 | 0.0623 | 0.0622 |

- GLGENN GA parameters ≈0.6K vs CGENN ≈1.8K (**reduced by ~67%**)
- GLGENN performs better on small datasets (30-300 samples), indicating stronger anti-overfitting capabilities.

### O(5,0)-Convex Hull Volume Estimation (16 points)

| Training Samples | GLGENN | CGENN |
|:---------:|:------:|:-----:|
| $2^8$ | **16.94** | 18.71 |
| $2^{12}$ | 6.2 | 6.1 |
| $2^{16}$ | 3.04 | 2.52 |

- GLGENN parameters 24.1K vs CGENN 58.8K (**reduced by 59%**)
- GLGENN is superior with small samples; CGENN is slightly better with large samples, but the performance gap remains narrow.

### O(5,0)-Convex Hull Volume Estimation (256/512 points, High Difficulty)

| K | Training Samples | GLGENN | CGENN |
|---|:-------:|:------:|:-----:|
| 256 | $2^{10}$ | **2908** | 5177 |
| 256 | $2^{14}$ | **2918** | 3385 |
| 512 | $2^{10}$ | **8539** | 14728 |
| 512 | $2^{14}$ | **4872** | 7212 |

- GLGENN consistently outperforms CGENN in high-dimensional, large-scale scenarios.
- Parameter count: GLGENN 791K vs CGENN 1.72M (for K=256), a reduction of 54%.

### N-Body Experiment

Motion prediction of 5 charged particles in $\mathbb{R}^{5,0}$. GLGENN achieves comparable performance to CGENN with approximately half the parameter budget.

## Highlights & Insights

1. **Elegant Mathematical Motivation**: Utilizing the fundamental algebraic structures of grade involution and reversion in GA, the model naturally compresses parameters from an $n+1$ dimensional decomposition to a 4-dimensional decomposition, strictly guaranteeing equivariance in theory.
2. **Significant Parameter Efficiency**: In all experiments, parameters are reduced by 50%-67%, with training times decreasing accordingly.
3. **Advantage on Small Data**: Fewer parameters lead to inductive regularization, resulting in better performance on small training sets, which is crucial for applications in the natural sciences.
4. **Generality**: Applicable to pseudo-orthogonal groups of any signature $(p,q,r)$, including degenerate cases.
5. **Plug-and-Play**: Can directly replace corresponding layers in CGENN, and can also be integrated with standard architectures like MLPs.

## Limitations & Future Work

1. **Performance on Large Datasets**: Parameter reduction provides a regularization benefit, but may limit model capacity when large amounts of data are available, where CGENN sometimes slightly outperforms.
2. **Limited Experimental Scope**: Currently only evaluated on non-degenerate GAs ($C\ell_{p,q}$); experiments on the degenerate case $C\ell_{p,q,r}$ remain to be conducted.
3. **Lack of Real-World Applications**: Experiments are primarily based on synthetic benchmarks, and the model has not yet been validated on real-world tasks like molecular modeling or protein folding.
4. **Restricted Nonlinearity**: The geometric product layer provides non-linear interactions, but standard activation functions can only act on scalar subspaces, which may limit expressiveness.
5. **Low-Dimensional Equivalence**: When $n \leq 3$, $C\ell^{\bar{k}} = C\ell^k$, causing GLGENN to degenerate into CGENN without any parameter advantages.

## Related Work & Insights

- **CGENN** (Ruhe et al., 2023): The direct baseline for GLGENN, parameterized by grade subspaces.
- **GATr** (Brehmer et al., 2023): Integrates GA into Transformers, which entails higher computational costs.
- **EMLP** (Finzi et al., 2021): Equivariant MLPs based on irreducible representations.
- **Insight**: Weight-sharing strategies can be generalized to the design of equivariant networks with other algebraic structures.

## Rating
- Novelty: ⭐⭐⭐⭐ (Generalized Lipschitz groups represent a new mathematical contribution, and the weight-sharing strategy is novel)
- Experimental Thoroughness: ⭐⭐⭐ (Good coverage of benchmark experiments but lacks real-world applications)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous theoretical derivation, clear structure)
- Value: ⭐⭐⭐⭐ (Provides a solid theoretical foundation and a practical solution for parameter efficiency in equivariant networks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)
- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](../../ICML2026/others/identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[ICML 2025\] The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)
- [\[ICML 2025\] Improved Learning via k-DTW: A Novel Dissimilarity Measure for Curves](improved_learning_via_k-dtw_a_novel_dissimilarity_measure_for_curves.md)

</div>

<!-- RELATED:END -->
