---
title: >-
  [Paper Note] Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds
description: >-
  [ICLR 2026][Others][Paper Note] This paper analyzes the spectral properties of the Fisher Information Matrix (FIM) in the kernel space of low-dimensional probability distributions to establish deterministic upper and lower bounds for the metric tensor on the parameter space (neuromanifold). Based on the Hutchinson trace estimator, it introduces a fam
tags:
  - ICLR 2026
  - Others
date: 2026-05-08
content_hash: bd373dd392cefd4a
---
# Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds

**Conference**: ICLR 2026  
**arXiv**: [2505.13614](https://arxiv.org/abs/2505.13614)  
**Code**: None  
**Area**: Information Geometry / Deep Learning Theory  
**Keywords**: Fisher Information Matrix, Neuromanifold, Hutchinson estimator, Metric tensor, Spectral analysis

## TL;DR
This paper analyzes the spectral properties of the Fisher Information Matrix (FIM) in the kernel space of low-dimensional probability distributions to establish deterministic upper and lower bounds for the metric tensor on the parameter space (neuromanifold). Based on the Hutchinson trace estimator, it introduces a family of unbiased stochastic estimators with bounded variance that require only a single backpropagation for efficient computation.

## Background & Motivation
The high-dimensional parameter space of deep neural networks—the Neuromanifold—is uniquely defined by a Riemannian metric tensor via the Fisher Information Matrix. This metric tensor is crucial for natural gradient optimization, model compression, and generalization analysis. However, since the dimension of the FIM equals the number of parameters (millions to billions), direct computation is infeasible.

**Limitations of Prior Work**:
- **Empirical FIM (eFIM)**: Replaces the expectation with training labels; simple to compute but biased, and errors can be amplified under adversarial labels.
- **Monte Carlo Estimation**: Variance depends on the fourth moment of the parameter-output Jacobian, leading to unbounded coefficients of variation (CV) and unreliable quality.
- **Kronecker Approximation (KFAC)**: Assumes block structures, leading to error accumulation.

**Key Challenge**: The exact computation of the FIM is too costly, while existing approximations are either biased or have uncontrollable variance.  
**Key Insight**: The paper returns to the low-dimensional probability distribution space (kernel space) to analyze its spectral structure using matrix perturbation theory, then extends the results to the high-dimensional neuromanifold via the pullback mapping of the Jacobian to obtain controlled quality estimates.

## Method

### Overall Architecture
The core of the method is a pullback decomposition: the FIM of a classification network $p(y|x,\theta)$ can be written as $\mathcal{F}(\theta) = \sum_x (\partial z / \partial \theta)^\top \, \mathcal{I}(z(x,\theta)) \, (\partial z / \partial \theta)$, where $z$ is the linear output of the last layer and $\mathcal{I}$ is the "kernel space" FIM related only to the $C$-dimensional probability distribution. Thus, geometric problems on the million-dimensional neuromanifold are compressed into two steps: analyzing the spectral structure in the low-dimensional kernel space and mapping the results back to the high-dimensional parameter space using the Jacobian $\partial z/\partial\theta$.

### Key Designs

**1. Spectral Characterization of Kernel Space FIM: Reducing High-Dim Problems to the Probability Simplex**
Instead of tackling the massive FIM directly, the paper starts with the probability simplex $\Delta^{C-1}$ spanned by the softmax output. The FIM there has a clean closed form: $\mathcal{I}^\Delta(z) = \text{diag}(p) - pp^\top$. This structure is a diagonal matrix perturbed by a rank-1 term $pp^\top$. By applying the Cauchy interlacing theorem, the entire spectrum is bounded: the minimum eigenvalue $\lambda_1=0$ corresponds to the all-ones direction (degeneracy from probability normalization), the sum of all eigenvalues equals $1 - \|p\|^2$, and the maximum eigenvalue $\lambda_C$ is bracketed by tight bounds. These spectral properties are the foundation for all subsequent deterministic bounds and stochastic estimates.

**2. Deterministic Bounds: Approaching One-Hot Outputs with Rank-1 Lower Bounds**
Using the kernel space spectrum, the paper establishes the Löwner ordering $\lambda_C v_C v_C^\top \preceq \mathcal{I}^\Delta(z) \preceq \text{diag}(p)$. Through the Jacobian pullback, deterministic upper and lower bounds for the entire $\mathcal{F}(\theta)$ are derived. A critical observation is the asymmetry: the lower bound comes from a rank-1 approximation of the maximum eigenvalue direction. When the model is well-trained and outputs approach one-hot vectors, the kernel matrix itself degrades to rank-1, causing the lower bound error to vanish. Conversely, the upper bound $\text{diag}(p)$ remains loose with an error of at least $O(1/C)$.

**3. Hutchinson FIM Estimator: Unbiased Trace Estimation via One Backpropagation**
The paper constructs unbiased stochastic estimators using the Hutchinson trace estimation approach. A scalar function is defined as $\mathfrak{h}(\mathcal{D}_x, \theta) = \sum_{x,y} \tilde{p}(y|x,\theta)\, \ell_{xy}(\theta)\, \xi_{xy}$, where $\xi$ is a Rademacher random vector and $\tilde{p}$ is a detached version of $p$. A single automatic differentiation yields $\partial\mathfrak{h}/\partial\theta$, and the outer product $\mathbb{F}(\theta) = (\partial\mathfrak{h}/\partial\theta)(\partial\mathfrak{h}/\partial\theta)^\top$ serves as an unbiased estimate of the FIM. Its trace satisfies $\mathbb{E}[\|\partial\mathfrak{h}/\partial\theta\|^2] = \text{tr}(\mathcal{F}(\theta))$. Unlike MC estimates with unbounded CV or biased eFIM, this estimator's CV is strictly bounded by $\sqrt{2}$.

**4. Diagonal and Low-Rank Kernel Variants**
To leverage the asymmetric bound structure, two specialized versions are provided. The diagonal kernel estimator $\mathbb{F}^{DG}$ corresponds to the upper bound $\text{diag}(p)$, suitable for multi-label classification. The low-rank kernel estimator $\mathbb{F}^{LR}$ corresponds to the rank-1 lower bound, requiring only $|\mathcal{D}_x|$ Rademacher samples instead of $C|\mathcal{D}_x|$, saving a factor of $C$ in sampling cost. This requires power iteration for the kernel space principal eigenvalue (complexity $O(MC|\mathcal{D}_x|)$). For binary classification ($C=2$), $\mathbb{F}^{LR}$ almost perfectly matches the unbiased estimate $\mathbb{F}$.

### Loss & Training
The paper does not propose a new loss function but positions these tools as plug-and-play FIM computation primitives. The Hutchinson estimator can replace the biased eFIM in natural gradient optimization, act as a regularizer via $\text{tr}(\mathcal{F}(\theta))$, or measure parameter importance in model compression, all sharing the same backpropagation cost.

## Key Experimental Results

### Main Results
Numerical simulations were performed on DistilBERT using AG News (4 classes) and SST-2 (2 classes).

| Setup | Model | Dataset | Key Finding |
|------|------|--------|---------|
| Non-finetuned | DistilBERT | AG News (C=4) | $\mathbb{F}^{DG} > \mathbb{F} > \mathbb{F}^{LR}$, consistent with theoretical bounds |
| Finetuned | DistilBERT | SST-2 (C=2) | $\mathbb{F}^{LR} \approx \mathbb{F}$ (kernel is naturally rank-1 for $C=2$); upper bound is loose |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| eFIM vs Hutchinson | CV (Coeff. Var.) | eFIM CV is unbounded (Lemma 5); Hutchinson CV $\leq \sqrt{2}$ (Prop 12) |
| MC vs Hutchinson | Comp. Cost | MC requires gradients per $x$; Hutchinson needs only one backprop |
| Upper vs Lower Error | Frobenius Norm | Lower bound error controlled by pruned probability mass; upper bound error $\geq 1/C$ |

### Key Findings
- FIM exhibits pathological spectral structures: diagonally, over 20% of parameters in all layers have values $< 10^{-5}$.
- Fisher information values are smallest in layers near the input and largest at the classification head.
- Hutchinson estimators with Rademacher distributions have lower variance than those with Gaussian distributions.
- The low-rank lower bound is an excellent approximation of the FIM when model outputs are near one-hot.
- The FIM density distribution shows a sharp peak near zero and sparsity at large values on a log scale.
- Embedding layers have the lowest Fisher information, aligning with the empirical observation that they require smaller learning rates.

## Highlights & Insights
- **Solid Theoretical Contribution**: Establishing FIM bounds on neuromanifolds via pullback mappings from low-dimensional kernel space is a significant advancement.
- **High Practicality**: The Hutchinson estimator requires only one backprop and one detach operation, making it easy to integrate into PyTorch workflows.
- **Unified Framework**: FIM, eFIM, and MC estimates can all be compared within the same theoretical framework.
- **Novel Kernel Space Perspective**: Analyzing the low-dimensional probability simplex before generalizing to high dimensions avoids the difficulty of handling massive matrices directly.

## Limitations & Future Work
- Numerical experiments are limited to DistilBERT; verification on large-scale models (e.g., GPT-scale) is lacking.
- Performance gains of the Hutchinson estimator in actual optimization algorithms are not demonstrated.
- Advanced variance reduction techniques (e.g., Hutch++) are not explored.
- The work focuses on classification; extension to generative models or regression is needed.
- The low-rank estimator depends on power iteration, adding a small computational overhead.

## Related Work & Insights
- **Natural Gradient (Amari, 1998)**: Provides the foundational metric for parameter spaces; this work offers efficient computation.
- **KFAC (Martens & Grosse, 2015)**: Uses Kronecker factors; this work's bounds can serve as a reference for KFAC accuracy.
- **AdaHessian (Yao et al., 2021)**: Uses Hutchinson probes for diagonal Hessian; this work applies the logic to FIM.
- **Monte Carlo Information Geometry (Nielsen & Hadjeres, 2019)**: The proposed Hutchinson estimator provides better variance guarantees than MC estimates.
- **General Insight**: For high-dimensional matrix estimation, the strategy of "performing detailed analysis in low-dimensional space and mapping to high-dimension" is a powerful general approach.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Fields to Random Trees](from_fields_to_random_trees.md)
- [\[NeurIPS 2025\] Tight Bounds On the Distortion of Randomized and Deterministic Distributed Voting](../../NeurIPS2025/others/tight_bounds_on_the_distortion_of_randomized_and_deterministic_distributed_votin.md)
- [\[ICLR 2026\] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)
- [\[ICLR 2026\] How NOT to benchmark your SITE metric: Beyond Static Leaderboards and Towards Realistic Evaluation](how_not_to_benchmark_your_site_metric_beyond_static_leaderboards_and_towards_rea.md)
- [\[ICLR 2026\] Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification](random_anchors_with_low-rank_decorrelated_learning_a_minimalist_pipeline_for_cla.md)

</div>

<!-- RELATED:END -->
