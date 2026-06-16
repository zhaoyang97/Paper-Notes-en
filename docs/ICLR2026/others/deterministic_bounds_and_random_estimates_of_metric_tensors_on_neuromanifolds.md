---
title: >-
  [Paper Note] Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds
description: >-
  [ICLR 2026][Fisher Information Matrix] By analyzing the spectral properties of the Fisher Information Matrix (FIM) in the low-dimensional kernel space of probability distributions…
tags:
  - "ICLR 2026"
  - "Fisher Information Matrix"
  - "Neuromanifold"
  - "Hutchinson Estimator"
  - "Metric Tensor"
  - "Spectral Analysis"
date: 2026-05-08
content_hash: 195e7d99e563c8dd
---

# Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds

**Conference**: ICLR 2026
**arXiv**: [2505.13614](https://arxiv.org/abs/2505.13614)  
**Code**: None  
**Area**: Information Geometry / Deep Learning Theory
**Keywords**: Fisher Information Matrix, Neuromanifold, Hutchinson Estimator, Metric Tensor, Spectral Analysis

## TL;DR
By analyzing the spectral properties of the Fisher Information Matrix (FIM) in the low-dimensional kernel space of probability distributions, this paper establishes deterministic upper and lower bounds for the metric tensor on the neural network parameter space (neuromanifold), and introduces a family of unbiased stochastic estimators with bounded variance based on the Hutchinson trace estimator, computable efficiently with a single backward pass.

## Background & Motivation
The high-dimensional parameter space of deep neural networks—the neuromanifold—is endowed with a Riemannian metric tensor uniquely defined by the Fisher Information Matrix. This metric tensor is of central importance to natural gradient optimization, model compression, generalization analysis, and related theoretical and practical pursuits. However, since the dimensionality of the FIM equals the number of parameters (ranging from millions to billions), direct computation is infeasible.

Limitations of prior work include:
- **Empirical FIM (eFIM)**: Replaces expectations with training labels, offering computational convenience but introducing bias that can be amplified under adversarial labeling.
- **Monte Carlo estimation**: Variance depends on the fourth-order moments of the parameter–output Jacobian; the coefficient of variation (CV) is unbounded, providing no quality guarantees.
- **Kronecker approximations**: Impose block-structure assumptions, leading to accumulated approximation errors.

The root cause is that exact FIM computation is prohibitively expensive, while existing approximations are either biased or have uncontrolled variance. This paper's starting point is to return to the low-dimensional probability distribution space (kernel space), analyze its spectral structure via matrix perturbation theory, and then lift the results to the high-dimensional neuromanifold through the Jacobian pullback map, ultimately yielding estimates of controllable quality.

## Method

### Overall Architecture
For a classification network $p(y|x,\theta)$, the FIM admits a pullback decomposition: $\mathcal{F}(\theta) = \sum_x (\partial z / \partial \theta)^\top \cdot \mathcal{I}(z(x,\theta)) \cdot (\partial z / \partial \theta)$, where $z$ denotes the pre-softmax logits and $\mathcal{I}$ is the FIM of the low-dimensional kernel space. Accordingly, analyzing the geometric structure of the kernel space is the key step.

### Key Designs

1. **Spectral Analysis of the Kernel-Space FIM (Theorem 1)**:
   For a $C$-class classifier with softmax outputs, the kernel space is the probability simplex $\Delta^{C-1}$, whose FIM takes the form $\mathcal{I}^\Delta(z) = \text{diag}(p) - pp^\top$. Since this is a rank-1 perturbation of a diagonal matrix, the Cauchy interlacing theorem precisely characterizes its spectrum: the smallest eigenvalue is $\lambda_1=0$ (corresponding to the all-ones vector), the sum of eigenvalues equals $1 - \|p\|^2$, and tight upper and lower bounds are established for the largest eigenvalue $\lambda_C$. These spectral properties underpin all subsequent results.

2. **Deterministic Upper and Lower Bounds (Proposition 6)**:
   Exploiting the Löwner partial order $\lambda_C v_C v_C^\top \preceq \mathcal{I}^\Delta(z) \preceq \text{diag}(p)$ in the kernel space and pulling back through the Jacobian to the neuromanifold yields deterministic upper and lower bounds for $\mathcal{F}(\theta)$. A key finding is that the lower bound (a rank-1 approximation based on the largest eigenvalue) incurs vanishing error as the model output approaches a one-hot vector, and is of higher quality than the upper bound. The Frobenius-norm error is controlled by the "trimmed norm" of the probability vector and the singular values of the Jacobian.

3. **Hutchinson FIM Estimator (Proposition 12)**:
   A scalar function $\mathfrak{h}(\mathcal{D}_x, \theta) = \sum_{x,y} \tilde{p}(y|x,\theta) \ell_{xy}(\theta) \xi_{xy}$ is introduced, where $\xi$ is a Rademacher random vector and $\tilde{p}$ is a detached copy of $p$ (with zero gradient). Computing $\partial \mathfrak{h}/\partial \theta$ via automatic differentiation yields $\mathbb{F}(\theta) = (\partial \mathfrak{h}/\partial \theta)(\partial \mathfrak{h}/\partial \theta)^\top$, which is an unbiased estimator of the FIM with bounded CV ($\leq \sqrt{2}$), requiring only a single backward pass.

4. **Diagonal-Kernel and Low-Rank-Kernel Hutchinson Variants**:
    - **Diagonal-kernel estimator** $\mathbb{F}^{DG}$: Suited for multi-label classification or estimation of the FIM upper bound.
    - **Low-rank-kernel estimator** $\mathbb{F}^{LR}$: Suited for estimating the FIM lower bound; requires only $|\mathcal{D}_x|$ Rademacher samples (rather than $C|\mathcal{D}_x|$), offering higher computational efficiency. A preliminary power iteration is needed to obtain the dominant eigenvalue/eigenvector of the kernel space ($O(MC|\mathcal{D}_x|)$ complexity).

### Loss & Training
This paper does not propose a new training procedure but provides analytical and estimation tools for the FIM. The Hutchinson estimator can be directly applied to:
- Replacing the eFIM in natural gradient optimization;
- Serving as a regularization term (estimating the FIM trace via $\mathbb{E}[\|\partial\mathfrak{h}/\partial\theta\|^2] = \text{tr}(\mathcal{F}(\theta))$);
- Assessing parameter importance in model compression.

## Key Experimental Results

### Main Results
Numerical simulations are conducted on DistilBERT, validated on AG News (4 classes) and SST-2 (2 classes).

| Setting | Model | Dataset | Core Finding |
|---------|-------|---------|--------------|
| Pre-fine-tuning | DistilBERT | AG News (C=4) | $\mathbb{F}^{DG} > \mathbb{F} > \mathbb{F}^{LR}$, consistent with the theoretical bound ordering |
| Post-fine-tuning | DistilBERT | SST-2 (C=2) | $\mathbb{F}^{LR} \approx \mathbb{F}$ (the kernel matrix is already rank-1 when C=2); upper bound is relatively loose |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|-----------|---------|
| eFIM vs. Hutchinson | CV (coefficient of variation) | CV of eFIM is unbounded (Lemma 5); CV of Hutchinson $\leq \sqrt{2}$ (Proposition 12) |
| MC estimation vs. Hutchinson | Computational cost | MC requires independent gradient computations per $x$; Hutchinson requires only one backward pass |
| Upper-bound error vs. lower-bound error | Frobenius norm | Lower-bound error is controlled by the trimmed probability and can vanish; upper-bound error is at least $1/C$ |

### Key Findings
- The FIM exhibits a pathological spectral structure: across all layers, more than 20% of parameters have FIM diagonal entries below $10^{-5}$.
- Fisher information values decrease toward the input layers, with the classification head exhibiting the largest values.
- Rademacher-distributed Hutchinson estimators have lower variance than Gaussian-distributed ones.
- When model outputs approach one-hot vectors (well-trained models), the low-rank lower bound is an excellent approximation of the FIM.
- On SST-2 (C=2), the low-rank estimator $\mathbb{F}^{LR}$ is nearly identical to the unbiased estimator $\mathbb{F}$, since the kernel matrix is inherently rank-1 in binary classification.
- The FIM density distribution on a log scale exhibits a sharp spike near zero and sparse mass at large values, reflecting a highly non-uniform pathological structure.
- Embedding layers exhibit the lowest Fisher information, consistent with the empirical observation that embedding layers typically do not require large learning rates during fine-tuning.

## Highlights & Insights
- **Solid theoretical contributions**: Starting from the low-dimensional kernel space and systematically establishing bounds for the neuromanifold FIM via pullback maps represents a significant advance in the computation of Fisher information.
- **Strong practical applicability**: The Hutchinson estimator requires only a single backward pass plus a detach operation and can be integrated directly into PyTorch training pipelines.
- **Unified framework**: Analysis, deterministic approximation, and stochastic estimation of the FIM are subsumed under a single theoretical framework, enabling principled comparison among the FIM, eFIM, and MC estimates.
- **Novel kernel-space perspective**: Conducting a complete analysis in the low-dimensional probability simplex before lifting to the high-dimensional space avoids direct manipulation of enormous matrices.

## Limitations & Future Work
- Numerical experiments are limited to DistilBERT; validation on large-scale models (e.g., GPT-scale) is absent.
- The paper does not demonstrate performance gains from the Hutchinson estimator within practical optimization algorithms.
- Advanced variance-reduction techniques (e.g., Hutch++) are not explored.
- The analysis is restricted to classification networks and is not extended to generative models or regression tasks.
- The low-rank-kernel estimator relies on power iteration to obtain the dominant eigenvalue/eigenvector, introducing additional computational steps.

## Related Work & Insights
- **Natural Gradient (Amari, 1998)**: The foundational work establishing the FIM as a metric on parameter space; this paper provides a new avenue for efficient FIM computation.
- **KFAC (Martens & Grosse, 2015)**: Kronecker-factored approximation of the FIM; the bounds derived in this paper can serve as a reference for evaluating KFAC accuracy.
- **AdaHessian (Yao et al., 2021)**: Approximates the diagonal Hessian using Hutchinson probing; this paper applies analogous ideas directly to the FIM.
- **Monte Carlo Information Geometry (Nielsen & Hadjeres, 2019)**: The Hutchinson estimator proposed here offers stronger variance guarantees than MC estimation.
- **eFIM in Adam (Kingma & Ba, 2015)**: Adam essentially employs an empirical diagonal FIM; this paper analyzes the associated bias.
- **Information Geometry and Deep Learning**: This paper is a representative contribution to the systematic application of differential-geometric tools to the analysis of deep learning parameter spaces.
- **General Insight**: For high-dimensional matrix estimation problems, the strategy of "conducting fine-grained analysis in a low-dimensional space and then lifting to high dimensions via a mapping" is a broadly transferable methodology.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tight Bounds On the Distortion of Randomized and Deterministic Distributed Voting](../../NeurIPS2025/others/tight_bounds_on_the_distortion_of_randomized_and_deterministic_distributed_votin.md)
- [\[ICLR 2026\] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)
- [\[AAAI 2026\] DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design](../../AAAI2026/others/deeprwcap_neural-guided_random-walk_capacitance_solver_for_ic_design.md)
- [\[AAAI 2026\] Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets](../../AAAI2026/others/intermediate_n-gramming_deterministic_and_fast_n-grams_for_large_n_and_large_dat.md)
- [\[ICML 2026\] New Bounds for Kernel Sums via Fast Spherical Embeddings](../../ICML2026/others/new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)

</div>

<!-- RELATED:END -->
