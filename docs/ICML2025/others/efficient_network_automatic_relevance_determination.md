---
title: >-
  [Paper Note] Efficient Network Automatic Relevance Determination
description: >-
  [ICML2025][ARD] Extends Automatic Relevance Determination (ARD) from single-output to multi-output regression scenarios, proposes the NARD framework to jointly estimate sparse regression coefficients and the output precision matrix, and designs three acceleration algorithms (Sequential/Surrogate/Hybrid) to reduce complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(p^2)$.
tags:
  - "ICML2025"
  - "ARD"
  - "Sparse Bayesian"
  - "Precision Matrix Estimation"
  - "Graphical Lasso"
  - "Multi-output Regression"
  - "Feature Selection"
date: 2026-05-08
content_hash: 1bc9dc6df1fe65f2
---

# Efficient Network Automatic Relevance Determination

**Conference**: ICML2025  
**arXiv**: [2506.12352](https://arxiv.org/abs/2506.12352)  
**Code**: Not publicly available  
**Area**: Sparse Bayesian Learning / Feature Selection / Multi-output Regression  
**Keywords**: ARD, Sparse Bayesian, Precision Matrix Estimation, Graphical Lasso, Multi-output Regression, Feature Selection

## TL;DR
Extends Automatic Relevance Determination (ARD) from single-output to multi-output regression scenarios, proposes the NARD framework to jointly estimate sparse regression coefficients and the output precision matrix, and designs three acceleration algorithms (Sequential/Surrogate/Hybrid) to reduce complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(p^2)$.

## Background & Motivation
- **Multi-output regression** is widely applied in fields such as genomics and proteomics, requiring simultaneous modeling of input-output relationships and dependency structures among outputs.
- Biological data typically feature **ultra-high dimensional features** (thousands of genes/proteins), but only a small number of features truly affect the phenotypes, necessitating **sparse feature selection**.
- Limitations of prior work:
    - Frequentist methods such as MRCE use $\ell_1$ penalties but are highly sensitive to hyperparameters.
    - Bayesian MCMC methods like HS-GHS and JRNS achieve high accuracy but are computationally expensive (>3000 seconds).
    - Classical ARD/SBL only handle single-output scenarios and fail to capture correlations among outputs.
- **Core Motivation**: To design a Bayesian framework that can jointly estimate sparse regression coefficients and the output precision matrix while efficiently handling high-dimensional data.

## Method

### NARD Framework
Imposes a **matrix normal prior** on the regression coefficient matrix $W$:

$$W \sim \mathcal{MN}(0, V, K^{-1})$$

where $K = \text{diag}(\alpha_1, \cdots, \alpha_d)$ is the column precision matrix and $V$ is the row covariance matrix. Through the ARD mechanism, when $\alpha_i \to \infty$, the corresponding feature is automatically pruned.

An $\ell_1$ penalty is imposed on the precision matrix $\Omega = V^{-1}$ to encourage sparsity, which is updated using **Graphical Lasso**:

$$\hat{V}, \hat{V}^{-1} \leftarrow \textbf{glasso}(V, \lambda)$$

The overall method employs **Block Coordinate Descent (BCD)** to worker out-of-order updates alternatingly among $W$, $\alpha$, and $V$, with a complexity of $\mathcal{O}(m^3 + d^3)$ per iteration.

### Sequential NARD
Inspired by Tipping's greedy method, it **evaluates the contribution of each feature sequentially**:
- Decomposes the marginal likelihood into two parts: one dependent on and one independent of a single $\alpha_i$.
- Defines $\eta_i = \text{Tr}(q_i q_i^\top V^{-1}) - m s_i$, where the optimal $\alpha_i$ is given by:

$$\alpha_i = \begin{cases} \frac{m s_i^2}{\eta_i}, & \eta_i > 0 \\ \infty, & \eta_i \leq 0 \end{cases}$$

- Starts from an almost empty model, sequentially adding or deleting features, retaining the update only when the marginal likelihood increases.
- Reduces complexity to $\mathcal{O}(m^3 + p^3)$, where $p \ll d$ is the number of finally retained features.

### Surrogate NARD
Introduces a **surrogate function** to approximate the lower bound of the marginal likelihood:
- Uses Lipschitz continuous gradients to construct a lower bound $R(W, W')$, satisfying $\text{Tr}(g(W)) \leq \text{Tr}(R(W, W'))$.
- Key advantage: $S_{xx} = K + \rho I$ remains a diagonal matrix, which requires only $\mathcal{O}(d)$ for inversion.
- Updates $W, W', V, K$ alternately via BCD, reducing complexity to $\mathcal{O}(m^3 + d^2)$.

### Hybrid NARD
Combines the feature selection of Sequential NARD with the computational efficiency of Surrogate NARD:
- Evaluates whether features should be included in the model using the Sequential approach.
- Once included, updates matrices using the Surrogate approach.
- Further reduces complexity to $\mathcal{O}(m^3 + p^2)$.

## Key Experimental Results

### Synthetic Data ($d=5000, m=1500, N=1500$)

| Method | TPR | FPR | Total Time (s) |
|------|-----|-----|-----------|
| MRCE | 0.9083 | 0.0072 | 53 |
| CAPME | 0.8972 | 0.0124 | 52 |
| HS-GHS | 0.9463 | 0.0033 | >3000 |
| JRNS | 0.9485 | 0.0037 | >3000 |
| NARD | 0.9483 | 0.0062 | 49 |
| Sequential NARD | 0.9459 | 0.0067 | 35 |
| Surrogate NARD | 0.9462 | 0.0072 | 31 |
| Hybrid NARD | 0.9471 | 0.0068 | 23 |

### Scalability ($N=20000, m=2000$, per-step time in seconds)

| $d$ | MRCE | NARD | Surrogate NARD |
|-----|------|------|----------------|
| 5000 | 12.2 | 10.7 | 3.7 |
| 10000 | 33.4 | 30.8 | 8.9 |
| 20000 | 201.9 | 168.6 | 33.7 |
| 30000 | 421.3 | 376.7 | 64.7 |

### Aging Phenotype Data
- 1,022 healthy individuals, 5,641 phenotypes (1,522 macro + 4,119 molecular phenotypes).
- The Jaccard index between various NARD variants and baseline methods is >98.5%, indicating highly consistent feature selection.
- NARD takes about 24 minutes, while Sequential NARD takes about 14 minutes.

### TCGA Cancer Data
- 7 tumor types, 10 key signaling pathways.
- NARD successfully identifies the GSK3-AKT regulatory relationship in the PI3K/AKT pathway within COAD, which aligns with known biological knowledge.

## Highlights & Insights
- **Unified Framework**: Seamlessly integrates ARD feature selection with Graphical Lasso precision matrix estimation, solving both problems within a single framework.
- **Theoretical Elegance**: All three acceleration schemes are supported by rigorous mathematical derivations (proofs of surrogate lower bounds, complexity analysis).
- **High Practicality**: Hybrid NARD achieves about 2× speedup at $d=5000$ and about 6× speedup at $d=30000$, with almost no loss in accuracy.
- **Biological Interpretability**: Protein interaction networks discovered on TCGA data align with known biological pathways.
- Can be naturally extended to non-linear scenarios through kernel methods.

## Limitations & Future Work
- The core assumption of the model is a **linear relationship**; although kernel expansion is mentioned, it is not verified with in-depth experiments.
- The selection of $\lambda$ in Graphical Lasso relies on 5-fold cross-validation, and this computational cost is not included in the total time comparison.
- Fails to compare against recent deep feature selection methods (e.g., gating networks, attention mechanisms).
- Theoretical analysis lacks rigorous bounds for the convergence rate.
- The code is **not publicly available**, posing obstacles to reproducibility.

## Related Work & Insights
- **MRCE** (Rothman et al., 2010): Jointly penalizes $W$ and $\Omega$ with $\ell_1$ norm, representing a classic frequentist approach.
- **HS-GHS** (Li et al., 2021): Horseshoe + Graphical Horseshoe prior, which offers high accuracy but is extremely slow.
- **Tipping's Fast ARD** (Tipping & Faul, 2003): The inspiration behind the Sequential update.
- Valuable reference for research in high-dimensional multi-output sparse learning, Bayesian feature selection, and biological network inference.

## Rating
- Novelty: ⭐⭐⭐⭐ — The extension of ARD to multi-output scenarios is natural and meaningful, and the three acceleration schemes are elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple scenarios including synthetic, aging phenotype, and TCGA datasets, though lacks comparison with deep learning methods.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear and rigorous, and the algorithm pseudocode is complete.
- Value: ⭐⭐⭐⭐ — Possesses solid practical application value in the field of high-dimensional biostatistics; the lack of open-source code is a minor drawback.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](../../NeurIPS2025/others/face_faithful_automatic_concept_extraction.md)
- [\[AAAI 2026\] Autonomous Concept Drift Threshold Determination](../../AAAI2026/others/autonomous_concept_drift_threshold_determination.md)
- [\[ACL 2025\] AutoMixer: Checkpoint Artifacts as Automatic Data Mixers](../../ACL2025/others/automixer_checkpoint_artifacts_as_automatic_data_mixers.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](../../ACL2025/others/improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)

</div>

<!-- RELATED:END -->
