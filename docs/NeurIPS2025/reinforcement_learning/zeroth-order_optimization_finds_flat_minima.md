---
title: >-
  [Paper Note] Zeroth-Order Optimization Finds Flat Minima
description: >-
  [NeurIPS 2025][Reinforcement Learning][Zeroth-order optimization] This paper provides the first theoretical proof that standard zeroth-order optimization (two-point gradient estimation) exhibits an implicit regularization effect—converging to flat minima that minimize the Hessian trace—with a convergence complexity of $T = \mathcal{O}(d^4/\epsilon^2)$ established under convexity and sufficient smoothness conditions.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Zeroth-order optimization
  - implicit regularization
  - Hessian trace
  - flat minima
  - language model fine-tuning
  - sharpness-aware minimization
date: 2026-05-08
content_hash: 6d9ca4546998871e
---

# Zeroth-Order Optimization Finds Flat Minima

**Conference**: NeurIPS 2025
**arXiv**: [2506.05454](https://arxiv.org/abs/2506.05454)
**Authors**: Liang Zhang (ETH Zurich / MPI-IS), Bingcong Li (ETH Zurich), Kiran Koshy Thekumparampil (Amazon), Sewoong Oh (UW), Michael Muehlebach (MPI-IS), Niao He (ETH Zurich)
**Code**: [Liang137/FlatZero](https://github.com/Liang137/FlatZero)
**Area**: Reinforcement Learning
**Keywords**: Zeroth-order optimization, implicit regularization, Hessian trace, flat minima, language model fine-tuning, sharpness-aware minimization

## TL;DR

This paper provides the first theoretical proof that standard zeroth-order optimization (two-point gradient estimation) exhibits an implicit regularization effect—converging to flat minima that minimize the Hessian trace—with a convergence complexity of $T = \mathcal{O}(d^4/\epsilon^2)$ established under convexity and sufficient smoothness conditions.

## Background & Motivation

### State of the Field
Zeroth-order (ZO) optimization is widely applied in settings where gradients are unavailable or prohibitively expensive to compute, including black-box adversarial attacks, reinforcement learning policy search, and large language model (LLM) fine-tuning. Malladi et al. (2023) demonstrated that ZO methods can fine-tune a 30-billion-parameter model on a single A100 GPU, whereas gradient-based methods require eight. The standard two-point gradient estimator approximates the gradient via finite differences:

$$g_\lambda(x,u) = \frac{f(x+\lambda u) - f(x - \lambda u)}{2\lambda} u, \quad u \sim \mathcal{N}(0, I_d)$$

### Limitations of Prior Work
- Existing ZO optimization theory focuses on convergence to **arbitrary stationary points**, without characterizing **which type of solution** is found.
- Research on flat minima almost exclusively concerns first-order methods (SGD, SAM, etc.); the implicit regularization effects of ZO methods have been overlooked.
- Ahn et al. (2024) define a local notion of flat minima (stationary points of the Hessian trace under gradient flow), which is not globally optimal.
- There is no theoretical explanation for why ZO methods achieve reasonable performance in high-dimensional LLM fine-tuning (i.e., why dimensional dependence can be replaced by the Hessian trace).

### Root Cause
Zeroth-order optimization effectively minimizes the smoothed function $f_\lambda(x) = \mathbb{E}_{u}[f(x+\lambda u)]$. Via Taylor expansion:

$$f_\lambda(x) = f(x) + \frac{\lambda^2}{2} \mathrm{Tr}(\nabla^2 f(x)) + o(\lambda^2)$$

This suggests that ZO optimization implicitly treats the Hessian trace as a regularization term. This paper formalizes this intuition and provides a rigorous convergence complexity analysis.

## Method

### Core Definitions

**Flat Minima (Definition 3.1)**: Let $\mathcal{X}^* = \arg\min_x f(x)$ denote the set of optimal solutions. A point $x^*$ is a flat minimum if and only if $x^* \in \arg\min_{x \in \mathcal{X}^*} \mathrm{Tr}(\nabla^2 f(x))$, i.e., it achieves the minimum Hessian trace among all optimal solutions.

**Approximate Flat Minima (Definition 3.2)**: A point $\hat{x}$ is an $(\epsilon_1, \epsilon_2)$-approximate flat minimum if:
- $f(\hat{x}) - \min_x f(x) \leq \epsilon_1$ (function value near optimal)
- $\mathrm{Tr}(\nabla^2 f(\hat{x})) - \min_{x \in \mathcal{X}^*} \mathrm{Tr}(\nabla^2 f(x)) \leq \epsilon_2$ (Hessian trace near minimum)

### Theoretical Analysis Framework

The core idea is to analyze convergence of the regularized loss $F(x) = f(x) + \frac{\lambda^2}{2} \mathrm{Tr}(\nabla^2 f(x))$, rather than treating the Hessian trace term as a bias to be controlled.

**Assumptions**:
- **Assumption 3.3 (Smoothness)**: $f(x)$ is three-times continuously differentiable and satisfies first-order ($L_1$), second-order ($L_2$), and third-order ($L_3$) smoothness conditions.
- **Assumption 3.4 (Convexity)**: $f(x)$ is convex on $\mathbb{R}^d$.

**Key Lemma (Lemma 3.5)**: Establishes the approximation relationship between $F(x)$ and $f_\lambda(x)$, as well as an upper bound on the second moment of the two-point estimator:

$$|f_\lambda(x) - F(x)| \leq \frac{L_3}{24}\lambda^4(d+4)^2$$

$$\mathbb{E}\|g_\lambda(x,u)\|^2 \leq 2(d+6)\|\nabla F(x)\|^2 + \mathcal{O}(\lambda^4 d^4)$$

This result requires computing the 8th-order Gaussian moments (via Isserlis' theorem), involving 105 combinatorial terms—far exceeding the 4th-order moments (3 terms) required by standard analyses.

### Main Theorem (Theorem 1)

Under Assumptions 3.3 and 3.4, Algorithm 1 with step size $\eta = 1/(8(d+6)L_1)$ and smoothing parameter $\lambda^2 \leq \sqrt{2}L_1/(d^{3/2}L_3)$ satisfies:

$$\mathbb{E}[F(x_\tau) - \min_x F(x)] \leq \frac{8(d+6)L_1\|x_0 - x_F^*\|^2}{T} + \mathcal{O}(\lambda^4 d^3)$$

### Corollary (Corollary 2): Convergence Complexity to Flat Minima

Setting $\lambda^2 = \mathcal{O}(\epsilon/d^3)$ and $T = \mathcal{O}(d^4/\epsilon^2)$, Algorithm 1 outputs an $(\mathcal{O}(\epsilon/d^2), \epsilon)$-approximate flat minimum:
- Function value error: $\mathbb{E}[f(x_\tau) - f^*] \leq \mathcal{O}(\epsilon/d^2)$
- Hessian trace error: $\mathbb{E}[\mathrm{Tr}(\nabla^2 f(x_\tau)) - \min_{x\in\mathcal{X}^*}\mathrm{Tr}(\nabla^2 f(x))] \leq \epsilon$

### Comparison with Standard Analysis

| Analysis Type | $\lambda^2$ Choice | Iterations $T$ | Function Value Guarantee | Hessian Trace Guarantee |
|---|---|---|---|---|
| Standard [Nesterov & Spokoiny] | $\mathcal{O}(\epsilon/d)$ | $\mathcal{O}(d/\epsilon)$ | $\leq \epsilon$ | None |
| **Ours** | $\mathcal{O}(\epsilon/d^3)$ | $\mathcal{O}(d^4/\epsilon^2)$ | $\leq \mathcal{O}(\epsilon/d^2)$ | $\leq \epsilon$ |

The proposed analysis trades more iterations for a convergence guarantee on the Hessian trace, while also providing a finer function value error bound.

## Key Experimental Results

### Experiment 1: Convex Binary Classification

Experiments are conducted on LIBSVM datasets using overparameterized SVM (squared hinge loss) and logistic regression, with feature dimensionality expanded from $d$ to $D=10000$.

| Model | Dataset | Method | Train Loss | Test Accuracy | Hessian Trace Trend |
|---|---|---|---|---|---|
| SVM | a5a ($N$=6414, $d$=123) | GD | Converges near 0 | ~81% | Unchanged |
| SVM | a5a | ZO ($\lambda$=0.005) | Converges near 0 | ~81% | **Continuously decreasing** |
| SVM | w5a ($N$=9888, $d$=300) | GD | Converges near 0 | ~97% | Unchanged |
| SVM | w5a | ZO ($\lambda$=0.005) | Converges near 0 | ~97% | **Continuously decreasing** |
| Logistic | a5a | GD | Converges | ~82% | Unchanged |
| Logistic | a5a | ZO ($\lambda$=0.01) | Converges | ~82% | **Continuously decreasing** |

Key observation: GD and ZO perform comparably in training loss and test accuracy, but ZO consistently reduces the Hessian trace and converges to flatter solutions.

### Experiment 2: RoBERTa-Large Fine-Tuning

Few-shot text classification fine-tuning is performed on RoBERTa-Large (355M parameters) using full-batch training to eliminate the effect of mini-batch noise.

| Dataset | K (samples/class) | Method | Train Loss | Test Accuracy | Hessian Trace Trend |
|---|---|---|---|---|---|
| SST-2 | 32 | GD | Lower | ~91% | Decreasing |
| SST-2 | 32 | ZO | Slightly higher | ~89% | **Decreasing** |
| SST-2 | 256 | GD | Lower | ~93% | Decreasing |
| SST-2 | 256 | ZO | Slightly higher | ~92% | **Decreasing** |
| SST-5 | 32 | GD | Lower | ~48% | Decreasing |
| SST-5 | 32 | ZO | Slightly higher | ~45% | **Decreasing** |
| TREC | 32 | GD | Lower | ~74% | Decreasing |
| TREC | 32 | ZO | Slightly higher | ~70% | **Decreasing** |

In LLM fine-tuning, both GD and ZO reduce the Hessian trace. The implicit regularization of GD is attributed to large learning rate effects, while ZO behavior is consistent with the theoretical predictions of this paper. Experiments also verify that the Hessian trace is substantially smaller than the dimension $d$, supporting prior work's assumption explaining ZO effectiveness in high dimensions.

## Highlights & Insights

- **Pioneering theoretical contribution**: This is the first paper to prove that standard zeroth-order optimization converges to globally defined flat minima—rather than merely local stationary points—establishing a new analytical framework.
- **Elegant implicit regularization mechanism**: The two-point estimator is shown to naturally encode $\mathrm{Tr}(\nabla^2 f(x))$ as a regularization term, achieving sharpness minimization effects analogous to SAM without any algorithmic modification.
- **Analytical depth**: The proof requires computing 8th-order moments of Gaussian random vectors (105 combinatorial terms), significantly advancing the theoretical toolkit for ZO optimization analysis.
- **Broad applicability**: The conclusions extend to other unbiased estimators satisfying $\mathbb{E}[uu^\top]=I_d$ (one-point estimators, spherical uniform distributions, etc.), and the analytical framework is generalizable to first-order methods (SAM, SGD).

## Limitations & Future Work

- **Convexity assumption**: The main theorem applies only to convex functions; non-convex optimization in deep learning requires additional assumptions (e.g., local convexity), and the definition of flat minima itself needs modification in the globally non-convex setting.
- **Higher-order smoothness assumption**: The requirement of three-times continuous differentiability with $L_1$, $L_2$, $L_3$ smoothness limits applicability to non-smooth problems (e.g., ReLU networks).
- **Dimension dependence**: The $d^4$ factor in the convergence complexity $\mathcal{O}(d^4/\epsilon^2)$ is heavy, albeit necessary for simultaneously guaranteeing Hessian trace convergence.
- **Restricted to standard two-point estimators**: Whether algorithmic modifications (e.g., incorporating first-order information, variance reduction) can improve complexity remains an open question.
- **GD also reduces the Hessian trace in LLM fine-tuning**: This prevents the experiments from demonstrating a unique advantage of ZO over GD in terms of flatness in non-convex settings.

## Related Work & Insights

- **Nesterov & Spokoiny (2017)**: Established the classical theory of ZO optimization converging to arbitrary stationary points; this paper builds upon that foundation to further characterize convergence to flat minima.
- **Ahn et al. (2024)**: Define local flat minima as stationary points of the Hessian trace under gradient flow, and propose two gradient-based algorithms. This paper adopts a global definition and achieves the result using ZO methods without requiring gradient or Hessian computations.
- **Wen et al. (2023, SAM)**: Prove that SAM with batch size 1 minimizes the Hessian trace. This paper shows that standard ZO optimization achieves a similar effect without any special design.
- **Malladi et al. (2023, MeZO)**: Demonstrate the practical feasibility of ZO methods for LLM fine-tuning. This paper provides a theoretical explanation of their effectiveness through implicit regularization.
- **Zhang et al. (2024)**: Inspired by $f_\lambda$, propose a gradient-based method that explicitly minimizes the smoothed loss. This paper proves that ZO methods already accomplish this goal implicitly.
- **Zheng et al. (2025), Tang et al. (2024)**: Combine ZO methods with SAM to explicitly seek flat minima. This paper focuses on the implicit effects already present in standard ZO methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Establishes the first theoretical connection between zeroth-order optimization and flat minima, opening a new research direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three levels: toy examples, convex classification tasks, and LLM fine-tuning; however, the fact that GD exhibits similar behavior in non-convex settings reduces the discriminative power of the results.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Proceeds systematically from intuition to formalization; the warm-up design in Example 2.1 and Proposition 2.2 is particularly elegant.
- **Value**: ⭐⭐⭐⭐ — Theoretically rigorous and offers new perspectives on ZO optimization and LLM fine-tuning, though the convexity restriction limits practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm](trust_region_reward_optimization_and_proximal_inverse_reward_optimization_algori.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[NeurIPS 2025\] Complexity Scaling Laws for Neural Models using Combinatorial Optimization](complexity_scaling_laws_for_neural_models_using_combinatorial_optimization.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)

</div>

<!-- RELATED:END -->
