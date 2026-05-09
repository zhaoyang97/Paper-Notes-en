---
title: >-
  [Paper Note] Tensor learning with orthogonal, Lorentz, and symplectic symmetries
description: >-
  [ICLR 2026][Time Series][Equivariant learning] This paper provides a complete parameterization of equivariant polynomial functions under the diagonal action of the orthogonal group $O(d)$, the indefinite orthogonal group (including the Lorentz group), and the symplectic group $Sp(d)$ on tensors. The framework is applied to design learnable sparse vector recovery algorithms that outperform existing sum-of-squares spectral methods across multiple data-generating assumptions.
tags:
  - ICLR 2026
  - Time Series
  - Equivariant learning
  - tensor functions
  - orthogonal group
  - Lorentz group
  - symplectic group
  - sparse vector recovery
date: 2026-05-08
content_hash: 61f3f0954c569473
---

# Tensor learning with orthogonal, Lorentz, and symplectic symmetries

**Conference**: ICLR 2026
**arXiv**: [2406.01552](https://arxiv.org/abs/2406.01552)
**Code**: [https://github.com/WilsonGregory/TensorPolynomials](https://github.com/WilsonGregory/TensorPolynomials)
**Area**: Time Series
**Keywords**: Equivariant learning, tensor functions, orthogonal group, Lorentz group, symplectic group, sparse vector recovery

## TL;DR

This paper provides a complete parameterization of equivariant polynomial functions under the diagonal action of the orthogonal group $O(d)$, the indefinite orthogonal group (including the Lorentz group), and the symplectic group $Sp(d)$ on tensors. The framework is applied to design learnable sparse vector recovery algorithms that outperform existing sum-of-squares spectral methods across multiple data-generating assumptions.

## Background & Motivation

Incorporating symmetry constraints into machine learning has become a mainstream trend — graph neural networks, geometric deep learning, and AI for Science all exploit equivariant/invariant structures to improve generalization and sample efficiency. Prior work (e.g., Villar et al., NeurIPS 2021) has studied the construction of $O(d)$-equivariant functions from vector inputs to tensor outputs, but no unified theoretical framework has addressed higher-order tensor inputs, mixed parities, and broader Lie groups such as the Lorentz and symplectic groups.

The planted sparse vector problem is a well-studied problem in theoretical computer science. Hopkins et al. (2016) and Mao & Wein (2022) proposed sum-of-squares-based spectral methods for sparse vector recovery, but these methods come with theoretical guarantees only under specific assumptions (e.g., identity covariance). Real-world data distributions are far more diverse, necessitating more flexible algorithms.

The central motivation of this paper is: can one construct equivariant machine learning models from first principles that respect underlying symmetries while learning algorithms from data that surpass hand-crafted designs?

## Method

### Overall Architecture

The paper is organized into two parts: theoretical characterization (Sections 3–4) and empirical validation (Section 5). The theoretical part provides a complete parametric representation of equivariant tensor polynomial functions; the application part leverages this parameterization to build learnable equivariant neural networks for the sparse vector recovery problem.

### Key Designs

1. **Parameterization of $O(d)$-equivariant polynomial functions (Theorem 1)**: For $O(d)$-equivariant polynomial functions $f$ mapping multiple tensor inputs to a tensor output, the paper proves that $f$ can be written as a linear combination of tensor products and $k$-contractions, where the coefficient tensors $c$ must be $O(d)$-isotropic. A key lemma (Lemma 1) further shows that all $O(d)$-isotropic tensors can be constructed from the Kronecker delta $\delta$ and the Levi-Civita symbol $\epsilon$, providing a complete set of "building blocks" for constructing equivariant models in practice.

2. **Specialization to vector inputs (Corollary 1)**: When inputs are restricted to vectors (1-tensors), $O(d)$-equivariant functions admit a concise representation as linear combinations of outer-product permutations over all input vector pairs and Kronecker deltas, with coefficients depending only on inner products between input vectors. This form is well-suited for practical implementation — the coefficient functions $q_{t,\sigma,J}$ can be approximated by MLPs.

3. **Extension to other groups (Theorem 2 & Corollary 2)**: Through complexification and averaging over Haar measure on Zariski-dense compact subgroups, the results are extended to the indefinite orthogonal group $O(s, d-s)$ (which includes the Lorentz group) and the symplectic group $Sp(d)$. The key modification replaces the Kronecker delta with the metric tensor $\theta_G$ preserved by the respective group, and replaces the Euclidean inner product with the group inner product $\langle \cdot, \cdot \rangle_G$.

4. **SparseVectorHunter (SVH) model**: Using the parameterization from Corollary 1, an equivariant model is designed to learn a $d \times d$ symmetric matrix $h$, whose leading eigenvector serves as the estimate of the sparse vector. Specifically, $h$ is formed as a combination of symmetric outer products of all input row vectors, where the coefficient for each combination is determined by an MLP taking all pairwise inner products as input.

5. **SVH-Diag simplified model**: Uses only diagonal terms ($a_i \otimes a_i$), with coefficients depending solely on the squared norms $\|a_\ell\|^2$. This variant has fewer parameters and performs better under diagonal covariance settings.

### Loss & Training

- Loss function: $1 - \langle \hat{v}, v_0 \rangle^2$, i.e., one minus the squared inner product between the predicted and true sparse vectors
- Training set: 5,000 samples; validation and test sets: 500 samples each
- Optimizer: Adam with exponential learning rate decay (0.999/epoch), batch size 100
- SVH: ~99K parameters; SVH-Diag: ~59K parameters; non-equivariant baseline (BL): ~1.33M parameters
- Trained on a single RTX 6000 Ada GPU over 18 hours
- Early stopping: training halts if validation error does not improve for 20 consecutive epochs

## Key Experimental Results

### Main Results

Experimental setup: $n=100$, $d=5$, $\epsilon=0.25$; four sparse vector sampling schemes (AR, BG, CBG, BR) × three covariance matrices (Identity, Diagonal, Random). Evaluation metric: $\langle v_0, \hat{v} \rangle^2$ (higher is better).

| Sampling | Covariance | SOS-I | SOS-II | BL | SVH-Diag | SVH |
|----------|-----------|-------|--------|----|----------|-----|
| BR | Random | 0.526 | 0.526 | 0.923 | 0.437 | **0.957** |
| BR | Diagonal | 0.334 | 0.334 | 0.864 | **0.588** | 0.903 |
| BR | Identity | 0.524 | 0.524 | 0.845 | 0.317 | **0.889** |
| CBG | Random | 0.412 | 0.412 | 0.239 | 0.372 | **0.935** |
| A/R | Random | 0.610 | 0.610 | 0.241 | 0.493 | **0.938** |
| BG | Identity | **0.962** | **0.962** | 0.196 | 0.908 | 0.342 |

### Ablation Study

| Configuration | Key Performance | Notes |
|---------------|----------------|-------|
| SVH (full inner products) | Best under Random covariance | Exploits all pairwise information |
| SVH-Diag (norms only) | Best under Diagonal covariance | Fewer parameters; excels when matched to data structure |
| BL (non-equivariant) | Severe overfitting on training set | 1.33M parameters but poor generalization |
| SOS-I / SOS-II | Best under Identity covariance | Hand-crafted methods with theoretical guarantees |

### Key Findings

- **Equivariance substantially improves generalization**: The non-equivariant baseline BL frequently achieves the best training fit but generalizes far worse on the test set, vividly confirming the theoretical prediction that symmetry improves generalization.
- **Learned methods outperform SOS where theory provides no guarantees**: Under non-identity covariances (diagonal, random), SVH/SVH-Diag significantly outperform SOS methods — settings for which no theoretical analysis currently exists.
- **Clear performance stratification**: SVH is optimal under Random covariance, SVH-Diag under Diagonal, and SOS under Identity; the sole exception is the Bernoulli-Rademacher sampling scheme, where SVH dominates across all settings.

## Highlights & Insights

- An elegant example of combining theory and practice: mathematical characterization is established first, from which a computable architecture is derived.
- Corollary 1 renders the highly abstract parameterization of equivariant functions practically implementable — coefficients depend only on scalar invariants (inner products) and are learned via MLPs.
- Consistent with the philosophy of neural algorithmic reasoning: model structure is aligned with known algorithmic strategies.
- The extension pathway to the Lorentz and symplectic groups is transparent: one need only substitute the metric tensor and bilinear form.
- Although the experimental scale is small ($n=100$, $d=5$), it suffices to clearly demonstrate the value of equivariance.

## Limitations & Future Work

- Experiments are validated only at small scale ($n=100$, $d=5$); larger-scale settings remain untested.
- Implementation efficiency degrades when inputs are higher-order tensors or involve mixed parities.
- Additional permutation invariance among input tensors (e.g., $S_n$-invariance), which relates to graph isomorphism problems, is not addressed.
- In the SVH model, each vector pair $(i,j)$ has an independent coefficient function, limiting scalability.
- No comparison is made against other equivariant architectures (e.g., EGNN, SE(3)-Transformers).

## Related Work & Insights

- Villar et al. (2021), "Scalars are universal," is the direct predecessor; this paper generalizes their framework from vectors to tensors.
- The sum-of-squares methods of Hopkins et al. (2016) provide baseline algorithms.
- This work is complementary to Kunisky, Moore & Wein's tensor cumulants approach, which focuses on symmetric tensors.
- The Stone–Weierstrass theorem guarantees that polynomial equivariant functions can approximate any continuous equivariant function.
- Insight: this "theory-driven architecture design" paradigm can be applied to a broader class of problems involving physical or mathematical symmetries.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)

</div>

<!-- RELATED:END -->
