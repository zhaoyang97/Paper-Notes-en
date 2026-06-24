---
title: >-
  [Paper Note] Constant Stepsize Local GD for Logistic Regression: Acceleration by Instability
description: >-
  [ICML2025][Optimization][Local GD] It is proven that Local GD can converge with **any positive stepsize** $\eta > 0$ for distributed logistic regression. By allowing non-monotonic objective decrease during an initial unstable phase, it achieves a faster convergence rate of $\widetilde{\mathcal{O}}(M/(\gamma^5 R^2))$ compared to the existing worst-case lower bounds of convex optimization.
tags:
  - "ICML2025"
  - "Optimization"
  - "Local GD"
  - "Logistic Regression"
  - "Distributed Optimization"
  - "Large Stepsize"
  - "Edge of Stability"
  - "Federated Learning"
date: 2026-05-08
content_hash: b1257c2b5a352aca
---

# Constant Stepsize Local GD for Logistic Regression: Acceleration by Instability

**Conference**: ICML2025  
**arXiv**: [2506.13974](https://arxiv.org/abs/2506.13974)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Local GD, Logistic Regression, Distributed Optimization, Large Stepsize, Edge of Stability, Federated Learning

## TL;DR
It is proven that Local GD can converge with **any positive stepsize** $\eta > 0$ for distributed logistic regression. By allowing non-monotonic objective decrease during an initial unstable phase, it achieves a faster convergence rate of $\widetilde{\mathcal{O}}(M/(\gamma^5 R^2))$ compared to the existing worst-case lower bounds of convex optimization.

## Background & Motivation
- **Key Challenge**: Local SGD/GD is widely used and performs exceptionally well in practice (federated learning, large-scale distributed training). However, existing theoretical analyses require the stepsize $\eta \leq \mathcal{O}(1/K)$ (where $K$ is the communication interval), which guarantees monotonic descent of the objective function but is severely disconnected from the non-monotonic descent commonly observed in practice.
- **Theoretical Gap**: Single-machine GD has been proven to converge under any stepsize for logistic regression (Wu et al., 2024), where large stepsizes lead to accelerated convergence. However, whether distributed Local GD exhibits similar properties remained theoretically unanswered. Prior work (Crawshaw et al., 2025) only analyzed a **two-phase** variant (small stepsize followed by large stepsize), leaving the original constant-stepsize Local GD an open question.
- **Limitations of Worst-Case Lower Bounds**: The existing worst-case lower bound of $\Omega(R^{-2/3})$ for Local GD targets general convex and smooth objectives. Logistic regression (with linearly separable data) does not belong to this class of problems (as no bounded optimal solution $\mathbf{w}_*$ exists), suggesting that problem-specific analysis could break through these lower bounds.

## Method

### Problem Setup
- $M$ clients, each with $n$ data points, and data dimension $d$.
- The global dataset is linearly separable with a maximum margin $\gamma$, and the maximum margin classifier is $\mathbf{w}_*$.
- Goal: Minimize the global logistic loss $F(\mathbf{w}) = \frac{1}{M}\sum_{m=1}^{M} F_m(\mathbf{w})$, where $F_m(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n} \ell(\langle \mathbf{w}, \mathbf{x}_i^m \rangle)$.
- Algorithm: Standard Local GD (Algorithm 1), performing $K$ local gradient descent steps on each client before averaging in each round.

### Core Theoretical Results

**Theorem 4.1 (Unstable Phase, Average Loss Upper Bound)**: For any $r \geq 0$,
$$\frac{1}{r}\sum_{s=0}^{r-1} F(\mathbf{w}_s) \leq 26 \cdot \frac{\|\mathbf{w}_0\|^2 + 1 + \log^2(K + \eta K \gamma^2 r) + \eta^2 K^2}{\eta \gamma^4 r}$$
holds for **any** $\eta > 0$ and $K \geq 1$. The right-hand side grows at most linearly with respect to $\eta$ and quadratically with respect to $K$.

**Theorem 4.2 (Stable Phase, Last-Iterate Upper Bound)**: Define the transition time $\tau$. For $r \geq \tau$:
$$F(\mathbf{w}_r) \leq \frac{16}{\eta \gamma^2 K (r - \tau)}$$
The transition time is $\tau = \widetilde{\mathcal{O}}(\eta K M / \gamma^3)$.

**Corollary 4.3 (Optimal Parameter Selection)**: Choosing $\eta K = \widetilde{\Theta}(\gamma^3 R / M)$, when $R \geq \widetilde{\Omega}(\max(Mn/\gamma^2, KM/\gamma^3))$:
$$F(\mathbf{w}_R) \leq \widetilde{\mathcal{O}}\left(\frac{M}{\gamma^5 R^2}\right)$$
This represents an $R^{-2}$ rate, **breaking through** the $\Omega(R^{-2/3})$ worst-case lower bound for general distributed convex optimization.

### Key Technical Techniques

1. **Trajectory Decomposition and $\beta$ Coefficients**: The one-round update of Local GD is decomposed into a linear combination of contributions from each data point, introducing coefficients $\beta_{r,i}^m$ to connect the trajectories of Local GD and GD. The analysis is conducted using the upper and lower bounds $1/K \leq \beta_{r,i}^m \leq 1 + \exp(\|\mathbf{w}_r\|)$.
2. **Split Comparator Technique** (adapted from Wu et al., 2024a): A split comparator $\mathbf{u} = \mathbf{u}_1 + \mathbf{u}_2$ is used to analyze the unstable phase, establishing a recursive bound for $\|\mathbf{w}_r\|$ (Lemma 4.4) to prove that $\|\mathbf{w}_r\|$ grows at most logarithmically.
3. **Adaptive Smoothness**: Utilizing the special properties of the logistic loss, $\|\nabla^2 F(\mathbf{w})\| \leq F(\mathbf{w})$ and $\|\nabla F(\mathbf{w})\| \leq F(\mathbf{w})$, it is shown that when the objective value is sufficiently small, the local smoothness constant is also small, thereby ensuring monotonic descent even with large stepsizes (Lemma 4.5 modifies the descent inequality).
4. **Drift Control (Lemma 4.8)**: It is proven that when $F(\mathbf{w}_r) \leq \gamma/(70\eta K M)$, the update drift satisfies $\|\mathbf{b}_r\| \leq \frac{1}{5}\|\nabla F(\mathbf{w}_r)\|$, guaranteeing the descent recurrence during the stable phase.

## Key Experimental Results

### Experimental Settings

| Dataset | Number of Clients $M$ | Data per Client $n$ | Comm. Rounds $R$ |
|--------|:---:|:---:|:---:|
| Synthetic | 2 | 1 | 2048 |
| MNIST (Binary) | 5 | 200 | 2048 |

Parameter search range: $\eta \in \{2^{-2}, 2^0, 2^2, 2^4, 2^6, 2^8, 2^{10}\}$, $K \in \{2^0, 2^2, 2^4, 2^6\}$

### Key Findings

| Question | Conclusion | Details |
|------|------|------|
| Q1: Can large $\eta, K$ accelerate? | ✅ Yes | Increasing $\eta$ or $K$ yields smaller final errors, despite an initial unstable phase; diverges when $\eta=2^{10}$ |
| Q2: Is tuning $\eta$ with fixed $K$ beneficial? | ✅ Yes | Large $K$ enables the use of larger $\eta$ (e.g., on MNIST, $\eta=256$ diverges when $K=1$, but converges when $K=16$) |
| Q3: Is varying $K$ with fixed $\eta K$ beneficial? | ❌ Mostly No | The final error is almost identical, though larger $K$ reduces the number of rounds to transition to the stable phase |

### Key Observations
- Increasing $K$ **does not increase** (and may even decrease) the number of communication rounds to reach the stable phase, which is stronger than the theoretical prediction ($\tau \propto \eta K$).
- Large $K$ has a "stabilizing" effect on $\eta$: it allows using larger stepsizes that would otherwise cause divergence.
- All experiments with $\eta = 2^{10}$ diverged, indicating that there is still an effective upper bound on the stepsize.

## Highlights & Insights
- **First convergence guarantee for arbitrary stepsize**: Completely removes the constraint $\eta \leq \mathcal{O}(1/K)$, representing a significant theoretical breakthrough for Local GD.
- **"Acceleration by instability" paradigm**: Echoes the Edge of Stability phenomenon—non-monotonic descent is not a defect, but a key mechanism for acceleration.
- **Value of problem-specific analysis**: Reveals phenomena overlooked by general analyses through a specialized and precise analysis (logistic regression), providing a model for aligning theory with practice.
- **Elegant decomposition technique**: The $\beta$ coefficients link the Local GD trajectory to the GD trajectory, and the upper/lower bound analysis is highly clever (the lower bound $1/K$ is simple yet tight up to logarithmic factors).

## Limitations & Future Work
1. **Local GD performs worse than GD**: The current guarantee $M/(\gamma^5 R^2)$ is worse than single-machine GD's $1/(\gamma^4 R^2)$, failing to prove the benefit of $K > 1$.
2. **$\eta$ and $K$ only appear as the product $\eta K$**: The theory cannot distinguish between different $(\eta, K)$ combinations (as long as $\eta K$ is the same), contradicting the independent benefits of $K$ observed in experiments.
3. **Suboptimal dependency on $\gamma$**: Originating from the need for an extra $\gamma$ factor when controlling the drift term $\|\mathbf{b}_r\|$ (Lemma B.2); improvement requires analyzing the implicit bias of Local GD.
4. **Narrow problem setting**: Limited to logistic regression with linearly separable data, without extending to Local SGD or neural network training.
5. **No implicit bias conclusion**: Does not prove that Local GD converges to the maximum-margin solution.

## Related Work & Insights
- **Wu et al. (2024a/b)**: Arbitrary stepsize convergence and acceleration analysis of single-machine GD for logistic regression; this paper extends its core techniques (split comparator, gradient potential) to the distributed setting.
- **Crawshaw et al. (2025)**: Two-phase Local GD analysis (small stepsize first, then large stepsize); this paper unifies it into a single-phase constant stepsize.
- **Cohen et al. (2021)**: Empirical discovery of the Edge of Stability phenomenon; this paper provides theoretical support for it in distributed optimization.
- **Patel et al. (2024)**: Lower bounds of Local GD for general convex optimization; this paper's acceleration rate breaks through this lower bound.
- **Insight**: Problem-specific analysis may be the key path to bridging the theory-practice gap in distributed optimization, which can be extended to two-layer networks with approximately homogeneous activations.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to prove arbitrary stepsize convergence for Local GD, offering a novel perspective on "acceleration by instability".
- Experimental Thoroughness: ⭐⭐⭐ — Synthesis + MNIST analysis is thorough but of limited scale; CIFAR-10 is only in the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, concise proof outline, exemplary layout of theorems, corollaries, and experiments.
- Value: ⭐⭐⭐⭐ — Solid theoretical contribution, though the limitation of Local GD not outperforming GD restricts its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression](benefits_of_early_stopping_in_gradient_descent_for_overparameterized_logistic_re.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](../../NeurIPS2025/optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[ICLR 2026\] Birch SGD: A Tree Graph Framework for Local and Asynchronous SGD Methods](../../ICLR2026/optimization/birch_sgd_a_tree_graph_framework_for_local_and_asynchronous_sgd_methods.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)
- [\[ICLR 2026\] SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration](../../ICLR2026/optimization/sgd_with_adaptive_preconditioning_unified_analysis_and_momentum_acceleration.md)

</div>

<!-- RELATED:END -->
