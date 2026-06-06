---
title: >-
  [Paper Note] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations
description: >-
  [ICLR 2026][LLM Evaluation][function encoders] This paper proves that Function Encoders, which learn neural network basis functions, implicitly define a valid kernel…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "function encoders"
  - "kernel methods"
  - "Hilbert space"
  - "generalization bounds"
  - "PCA-guided basis selection"
date: 2026-05-08
content_hash: 07dfba9132b28a42
---

# Function Spaces Without Kernels: Learning Compact Hilbert Space Representations

**Conference**: ICLR 2026
**arXiv**: [2509.20605](https://arxiv.org/abs/2509.20605)  
**Code**: [Available](https://github.com/suann124/function_encoder_kernels)  
**Area**: LLM Evaluation
**Keywords**: function encoders, kernel methods, Hilbert space, generalization bounds, PCA-guided basis selection

## TL;DR

This paper proves that Function Encoders, which learn neural network basis functions, implicitly define a valid kernel, thereby bridging neural feature learning and RKHS theory. It further proposes PCA-guided compact basis selection algorithms and establishes finite-sample generalization bounds.

## Background & Motivation

Machine learning methods have long faced a **trade-off between computational efficiency and theoretical guarantees**:

- **Neural networks**: flexible and scalable, but with limited theoretical guarantees (existing bounds are often vacuous or rely on restrictive assumptions).
- **Kernel methods**: theoretically well-founded (RKHS theory, precise statistical guarantees), but poorly scalable — inference cost scales with training set size $m$ (requiring inversion of an $m \times m$ Gram matrix in the dual representation).

Function Encoders are a recently proposed transfer/representation learning technique that learns a set of neural network basis functions $\{\psi_j\}_{j=1}^n$ as an explicit feature map $\phi(x) = [\psi_1(x), \ldots, \psi_n(x)]^\top$, with inference cost depending only on the number of bases $n$ rather than dataset size $m$. However, their theoretical foundations remain incomplete:

1. No formal connection to Hilbert space theory.
2. No principled method for selecting the number of bases $n$.
3. No finite-sample generalization guarantees.

This paper aims to close all three theoretical gaps.

## Method

### Overall Architecture

The central insight is that Function Encoders can be understood from two complementary perspectives — primal and dual:

- **Primal space**: provides an explicit feature map $\phi(x)$, enabling efficient $\mathcal{O}(n)$ linear prediction.
- **Dual space**: the inner product $\langle\phi(x), \phi(x')\rangle$ corresponds to kernel evaluation, amenable to the full suite of kernel-theoretic analysis tools.

Function Encoders operate in two stages:
1. **Offline phase**: jointly train basis functions on a collection of functions $\{D_1, \ldots, D_N\}$ by minimizing a multi-task reconstruction loss.
2. **Online inference**: fix the basis functions and solve for coefficients $c$ via regularized least squares.

### Key Designs

**1. Function Encoders as Learnable Kernels**

**Core theorem (Proposition 1)**: the inner product defined by the basis functions automatically constitutes a valid kernel:

$$k(x, x') = \langle\phi(x), \phi(x')\rangle = \sum_{j=1}^n \psi_j(x)\psi_j(x')$$

This kernel is symmetric and positive semi-definite (since $\|\sum_i \alpha_i \phi(x_i)\|^2 \geq 0$), and hence a legitimate reproducing kernel. This connection implies that Function Encoders not only enable efficient prediction in primal space but also instantiate a valid kernel in dual space — making the entire theoretical toolkit of kernel methods applicable.

**2. Progressive Training**

Analogous to PCA's incremental addition of principal components, basis functions are trained one at a time:

- Step $b=1$: train $\psi_1$, then freeze it.
- Step $b=2$: add $\psi_2$, train only $\psi_2$ to fit the residual.
- At each step $b$: compute the coefficient covariance matrix $\Sigma_b$ with eigenvalues $\lambda_1 \geq \ldots \geq \lambda_b$.
- Stopping criterion: cumulative explained variance $\text{CEV}_r = \sum_{i=1}^r \lambda_i / \sum_{j=1}^b \lambda_j \geq \tau$ (e.g., 99%).

**Advantage**: ordered and interpretable bases with a clear stopping rule.
**Limitation**: training is inherently sequential and cannot be parallelized on GPU.

**3. Train-Then-Prune**

- Jointly train $B$ bases ($B \gg r$).
- Compute the eigendecomposition of the coefficient covariance matrix.
- Effective rank: $r = \min\{n : \sum_{i=1}^n \lambda_i / \sum_{j=1}^B \lambda_j \geq \tau\}$.
- Score each basis by importance: $s_p = \sum_{i=1}^r \lambda_i U_{pi}^2$.
- Retain the top-$r$ bases and prune the rest.
- Fine-tune briefly to recover performance.

**Advantage**: training is fully parallelizable and computationally efficient.

**4. Generalization Bounds (Theoretical Core)**

**Rademacher complexity bound**:

$$L(f_{\hat{c}_\lambda}) \lesssim \hat{L}_m(f_{\hat{c}_\lambda}) + \tilde{\mathcal{O}}\left(Y^2 R^2 \frac{n}{\lambda\sqrt{m}}\right)$$

Key scaling: complexity grows with the number of bases $n$ and decays with data size $m$ and regularization $\lambda$. More bases improve expressiveness but increase overfitting risk.

**PAC-Bayes bound**:

$$L(f_{\hat{c}_\lambda}) \lesssim \hat{L}_m(f_{\hat{c}_\lambda}) + \tilde{\mathcal{O}}\left(Y^2 R^2 \frac{n^{3/2}}{\lambda\sqrt{m}}\right)$$

Technical contribution: a truncated Gaussian distribution is used to handle unbounded loss functions, extending PAC-Bayes analysis to multivariate outputs and learned feature maps.

### Loss & Training

- **Offline training loss**: multi-task regularized least squares
  $$\frac{1}{Nm}\sum_{j=1}^N \sum_{i=1}^m \|f_j(x_i) - \hat{f}_j(x_i)\|^2 + \lambda\|\hat{f}_j\|^2$$
- **Online inference**: solve for coefficients via regularized least squares with fixed bases.

## Key Experimental Results

### Main Results

**Polynomial space benchmark** (known intrinsic dimension $d+1$):

| Polynomial degree | Intrinsic dim. | Progressive recovery | Train-Prune recovery |
|---|---|---|---|
| $d=3$ | 4 | 4 ✓ | 4 ✓ |
| $d=4$ | 5 | 5 ✓ | 5 ✓ |
| $d=5$ | 6 | 6 ✓ | 6 ✓ |

Both algorithms exactly recover the known intrinsic dimension; pruned models match the accuracy of overparameterized originals.

**Van der Pol oscillator**:

| Method | # Bases | Prediction accuracy |
|---|---|---|
| Prior work (Ingebrand et al.) | 100 | Baseline |
| Progressive / Prune | **2** | **Same accuracy** |

Basis count reduced from 100 to 2 — a **50× reduction** — with no loss in accuracy.

**Two-body orbital problem**:

| Method | # Bases | Explained variance >99% |
|---|---|---|
| Overparameterized | Many | — |
| Progressive / Prune | **5–6** | ✓ |

Five to six bases capture >99% of variance, consistent with the 5-dimensional space of orbital parameters.

### Ablation Study

**Comparison with deep kernel learning** (degree-3 polynomial):
- Training time: Function Encoders are approximately **10× faster** than deep kernels (RBF) at $m=20$.
- Reason: deep kernels require $\mathcal{O}(m^3)$ Gram matrix inversion.
- The geometric structure of the Gram matrices is nearly identical, validating the interpretation of Function Encoders as adaptive kernels in primal space.

### Key Findings

1. Both PCA-guided algorithms accurately identify the intrinsic dimensionality of the function space.
2. Compact basis representations can reduce the number of bases by up to 50× without accuracy loss.
3. Function Encoders are significantly more training-efficient than deep kernel methods.
4. The derived bounds provide non-vacuous guarantees at inference time.
5. Learned basis functions exhibit interpretable physical structure in dynamical systems.

## Highlights & Insights

1. **Unifying two worlds**: the approach bridges the scalability of neural networks with the theoretical guarantees of kernel methods — Function Encoders sit precisely at this intersection.
2. **Elegant analogy between PCA and basis selection**: the problem of selecting the dimensionality of a function space is recast as PCA truncation in coefficient space.
3. **Solid theoretical contributions**: the truncated Gaussian PAC-Bayes technique has independent value and is applicable to other unbounded-loss settings.
4. **Layered experimental design**: validation progresses from polynomial spaces with known dimensionality to real dynamical systems, demonstrating the method's effectiveness in increasing complexity.

## Limitations & Future Work

1. The stopping threshold $\tau$ remains **heuristic**; a principled, non-heuristic basis selection criterion is lacking.
2. Progressive training is inherently sequential, limiting efficiency on large-scale problems.
3. Experiments are confined to low-dimensional dynamical systems; performance on high-dimensional complex settings (PDEs, vision tasks) is unknown.
4. Constants in the generalization bounds may be large; while non-vacuous, their practical tightness remains to be verified.
5. End-to-end integration with downstream applications (e.g., robotic control, trajectory prediction) has not been explored.

## Related Work & Insights

- **Kernel methods (GP/KRR/SVM)**: theoretically rigorous but poorly scalable; Function Encoders address this limitation while preserving theoretical properties.
- **Kernel approximations (Nyström / random Fourier features)**: alleviate computational cost but use fixed kernels that cannot adapt to data.
- **Deep kernel learning**: parameterized kernels enhance expressiveness but retain $\mathcal{O}(m)$ inference cost.
- **Dictionary learning / Koopman / SINDy**: also learn basis functions but pursue different objectives (sparse recovery / dynamics linearization).
- **Insight**: the kernel-theoretic perspective on Function Encoders may provide theoretical analysis tools for other transfer learning approaches.

## Rating

- **Novelty**: ★★★★☆ — The unifying perspective has theoretical originality, though Function Encoders themselves are not new.
- **Technical depth**: ★★★★★ — Dual theoretical analysis via Rademacher complexity and PAC-Bayes; the truncated Gaussian technique has independent value.
- **Experimental persuasiveness**: ★★★★☆ — Polynomial verification is thorough and dynamical system results are compelling, but large-scale experiments are absent.
- **Practical value**: ★★★☆☆ — Theoretical contributions outweigh immediate applicability; compact bases show potential for embedded systems.
- **Clarity**: ★★★★★ — Theoretical derivations are well-structured and the experimental design builds progressively.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [\[ICLR 2026\] Same Content, Different Representations: A Controlled Study for Table QA](same_content_different_representations_a_controlled_study_for_t.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ICML 2026\] Stop Automating Peer Review Without Rigorous Evaluation](../../ICML2026/llm_evaluation/stop_automating_peer_review_without_rigorous_evaluation.md)
- [\[ICLR 2026\] In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)

</div>

<!-- RELATED:END -->
