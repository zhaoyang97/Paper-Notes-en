---
title: >-
  [Paper Note] Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces
description: >-
  [ICML2025][Transfer Learning] Proposes a taxonomy of transfer learning from the geometric perspective of Hilbert spaces (convex hull interpolation / linear span extrapolation / full-space extrapolation), and designs the Function Encoder method utilizing learnable neural network basis functions to achieve all three types of transfer, outperforming methods such as MAML and Transformers on multiple benchmarks.
tags:
  - "ICML2025"
  - "Transfer Learning"
  - "Hilbert Space"
  - "Basis Functions"
  - "Function Encoder"
  - "Least Squares"
date: 2026-05-08
content_hash: 093bbc72e74758f4
---

# Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces

**Conference**: ICML2025  
**arXiv**: [2501.18373](https://arxiv.org/abs/2501.18373)  
**Code**: [tyler-ingebrand/FEtransfer](https://tyler-ingebrand.github.io/FEtransfer)  
**Area**: Transfer Learning  
**Keywords**: Transfer Learning, Hilbert Space, Basis Functions, Function Encoder, Least Squares

## TL;DR

Proposes a taxonomy of transfer learning from the geometric perspective of Hilbert spaces (convex hull interpolation / linear span extrapolation / full-space extrapolation), and designs the Function Encoder method utilizing learnable neural network basis functions to achieve all three types of transfer, outperforming methods such as MAML and Transformers on multiple benchmarks.

## Background & Motivation

- **Core Problem**: When can transfer learning algorithms effectively transfer to new tasks? Existing methods lack characterization of the "conditions for successful transfer."
- **Limitations of Prior Work**: Meta-learning methods like MAML require fine-tuning on new tasks and can easily fail when source and target tasks are only weakly related; large-scale pre-training relies on massive data rather than structural insights; kernel methods suffer from Gram matrix expansion as the data volume grows.
- **Design Motivation**: Frame the transfer learning problem as a geometric one in Hilbert spaces—the geometric position of the target task relative to the set of source tasks determines the transfer difficulty.
- Based on the existing Function Encoder theory (Ingebrand et al., 2024b), the authors further extend it to transfer learning scenarios across the entire Hilbert space.

## Method

### 1. Geometric Taxonomy of Transfer Learning

The inductive transfer problem is modeled in a Hilbert space $\mathcal{H}$ and categorized into three types based on the geometric relationship between the target function $f_T$ and the set of source functions $\{f_{S_1}, \ldots, f_{S_n}\}$:

| Type | Name | Definition | Difficulty |
|------|------|------|------|
| Type 1 | Convex Hull Interpolation | $f_T \in \text{Conv}(f_{S_1}, \ldots, f_{S_n})$, i.e., $f_T = \sum \alpha_i f_{S_i}$, $\alpha_i \ge 0$, $\sum \alpha_i = 1$ | Easiest |
| Type 2 | Linear Span Extrapolation | $f_T \in \text{span}\{f_{S_1}, \ldots, f_{S_n}\}$, coefficients are unconstrained | Moderate |
| Type 3 | Full-Space Extrapolation | $f_T \in \mathcal{H}$ but $f_T \notin \text{span}\{f_{S_1}, \ldots, f_{S_n}\}$ | Hardest |

### 2. Function Encoder Architecture

Learn a set of basis functions $\{g_1, \ldots, g_k\}$ parameterized by neural networks, representing any function $f \in \mathcal{H}$ as:

$$f(x) = \sum_{j=1}^{k} c_j g_j(x \mid \theta_j)$$

**Coefficient Computation via Least Squares (LS, newly proposed in this work)**:

$$c = G^{-1} b, \quad G_{ij} = \langle g_i, g_j \rangle_{\mathcal{H}}, \quad b_j = \langle f, g_j \rangle_{\mathcal{H}}$$

where the inner product is approximated via Monte Carlo integration: $\langle f, g_j \rangle \approx \frac{1}{m} \sum_{i=1}^{m} y_i \cdot g_j(x_i)$.

Key advantages of the LS method compared to the original Inner Product (IP) method:
- **Does not require orthogonal basis functions**, only linear independence (a weaker condition).
- Provides the theoretically optimal projection (in the least-squares sense).
- Yields faster training convergence and higher accuracy.

### 3. Loss & Training

$$L = \frac{1}{n} \sum_{\ell=1}^{n} \|f_{S_\ell} - \sum_{j=1}^{k} c_j^\ell g_j\|_{\mathcal{H}}^2 + \sum_{i=1}^{k} (\|g_i\|_{\mathcal{H}}^2 - 1)^2$$

The second term is a regularization term to prevent the divergence of the basis function magnitudes.

### 4. Universal Function Space Approximation Theorem

**Theorem 1**: For any separable Hilbert space $\mathcal{H}$, there exists a set of neural network basis functions such that any function in $\mathcal{H}$ can be approximated with arbitrary precision.

Proof sketch: A separable Hilbert space possesses a countable orthonormal basis $\rightarrow$ the universal approximation theorem of neural networks guarantees that each orthonormal basis can be approximated by an NN $\rightarrow$ error decays geometrically $\rightarrow$ overall approximation with finite precision.

### 5. Online Inference

Given a small amount of data $D_{f_T}$ from the target task, the coefficients can be directly computed using the LS formula without retraining. The size of the Gram matrix is $k \times k$ (a hyperparameter), independent of the data size, which allows extremely fast inference.

## Key Experimental Results

Compare FE (LS), FE (IP), AutoEncoder, Transformer, TFE, MAML, BF, BFB, and other methods on 4 benchmark tasks:

| Benchmark Task | Type 1 (Interpolation) | Type 2 (Span Extrapolation) | Type 3 (Full-Space Extrapolation) |
|----------|---------------|-------------------|---------------------|
| Polynomial Regression | FE(LS) is optimal, other methods perform acceptably | FE(LS) leads by several orders of magnitude | FE(LS) leads by several orders of magnitude |
| CIFAR-100 Classification | FE(LS) is slightly better than Siamese Network | — | FE(LS) is optimal, close to Siamese |
| 7-Scenes Pose Estimation | FE(LS) is optimal | — | FE(LS) is optimal |
| MuJoCo Ant Dynamics | FE(LS) is optimal | FE(LS) significantly leads | FE(LS) is optimal and stable |

**Key Findings**:

- In polynomial regression, FE(LS) achieves $L^2$ errors that are several orders of magnitude lower than other methods on Type 2/3.
- On CIFAR-100, although FE is a general-purpose method, its performance is comparable to or slightly better than specialized Siamese/Prototypical Networks.
- In the MuJoCo dynamics task, AutoEncoder performs well on Type 3 during early training but degrades sharply as training progresses, whereas FE(LS) remains stable.
- Increasing the number of basis functions (e.g., from 3 to 100) significantly improves Type 3 transfer, as the redundant dimensions are optimally utilized by LS.

## Highlights & Insights

1. **Novel Geometric Taxonomy**: Systematically classifies three transfer types from the geometric perspective of Hilbert spaces for the first time, providing intuitive understanding.
2. **LS Coefficient Computation is a Core Innovation**: Does not rely on orthogonality assumptions, making basis function training more flexible and speeding up convergence.
3. **Universal Approximation Theorem**: Provides a theoretical guarantee for the expressing capability of the Function Encoder.
4. **No Fine-Tuning Required**: Unlike MAML, inference only requires solving least squares, without gradient computation.
5. **Utilization of Redundant Dimensions**: When the number of basis functions $k$ is larger than the number of source tasks, LS can automatically utilize the redundant dimensions to adapt to Type 3 tasks, which is unique compared to other methods.

## Limitations & Future Work

1. **Inner Product Selection**: Inner products (such as $L^2$, probability distribution inner product) need to be manually designed for different problems, limiting generality.
2. **Number of Basis Functions $k$ Needs Tuning**: If $k$ is too small, it limits expression capability; if too large, it increases computational and regularization difficulty.
3. **Monte Carlo Approximation Error**: When data is scarce, inner product estimation is inaccurate, affecting coefficient computation quality.
4. **Scalability**: Scaling efficiency on extremely large task spaces (e.g., thousands of source tasks) has not been verified.
5. **Limited Theoretical Guarantees for Type 3**: The universal approximation theorem is an existence proof and does not provide a quantitative relationship between $k$ and the approximation error.
6. **Inductive Transfer Only**: Does not cover cross-domain scenarios such as domain adaptation.

## Related Work & Insights

- **MAML (Finn et al., 2017)**: Learns a good initialization for fast fine-tuning, but requires gradient steps; FE does not require fine-tuning.
- **Kernel Methods**: Also use the concept of basis functions, but the number of basis functions grows with data and requires pre-selected kernels.
- **Dictionary Learning**: Performs atomic decomposition on discrete points, whereas FE's basis functions can be evaluated on a continuous domain.
- **Transformer/TFE**: Performs poorly even in simple polynomial regression, lacking structural inductive biases.

## Rating

- Novelty: ⭐⭐⭐⭐ — Geometric taxonomy + LS training scheme + universal approximation theorem, solid theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 benchmarks from different domains and comprehensive ablation analyses, but lacks larger-scale experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear framework, intuitive figures, and tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐ — Provides a new geometric perspective for transfer learning; the LS-based FE offers strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving Generalization with Flat Hilbert Bayesian Inference](improving_generalization_with_flat_hilbert_bayesian_inference.md)
- [\[ICML 2026\] Decision Tree Learning on Product Spaces](../../ICML2026/others/decision_tree_learning_on_product_spaces.md)
- [\[CVPR 2025\] Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach](../../CVPR2025/others/wear_classification_of_abrasive_flap_wheels_using_a_hierarchical_deep_learning_a.md)
- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](../../ACL2025/others/principled_generalization_arithmetic.md)
- [\[ICLR 2026\] Hilbert-Guided Sparse Local Attention](../../ICLR2026/others/hilbert-guided_sparse_local_attention.md)

</div>

<!-- RELATED:END -->
