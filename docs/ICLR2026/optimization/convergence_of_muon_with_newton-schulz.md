---
title: >-
  [Paper Note] Convergence of Muon with Newton-Schulz
description: >-
  [ICLR 2026][Optimization][Muon optimizer] This work provides the first convergence guarantees for the Muon optimizer as it is actually used in practice—with Newton-Schulz (NS) approximation rather than exact SVD-based po…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Muon optimizer"
  - "Newton-Schulz"
  - "polar decomposition"
  - "matrix optimization"
  - "convergence analysis"
date: 2026-05-08
content_hash: dc56b6998365d590
---

# Convergence of Muon with Newton-Schulz

**Conference**: ICLR 2026
**arXiv**: [2601.19156](https://arxiv.org/abs/2601.19156)  
**Code**: To be confirmed  
**Area**: Optimization / Theory
**Keywords**: Muon optimizer, Newton-Schulz, polar decomposition, matrix optimization, convergence analysis

## TL;DR
This work provides the first convergence guarantees for the Muon optimizer as it is actually used in practice—with Newton-Schulz (NS) approximation rather than exact SVD-based polar decomposition. It proves that the convergence rate matches the idealized SVD variant up to a constant factor $C_q$ that decays doubly exponentially in the number of NS iterations $q$, and that Muon enjoys a $\sqrt{r}$ advantage over its vector-space counterpart SGD-M due to reduced rank loss.

## Background & Motivation

**Background**: The Muon optimizer updates matrix parameters by orthogonalizing the momentum matrix—rather than vectorizing it as Adam does—and has demonstrated strong empirical performance in LLM training. In practice, it employs Newton-Schulz (NS) iterations to approximate the polar decomposition, avoiding the costly SVD.

**Limitations of Prior Work**: Existing theoretical analyses of Muon (Shen et al.; Li & Hong) replace NS with exact SVD—yet SVD is never used in practice. It remains unclear how the NS approximation error affects convergence, and how many NS steps suffice.

**Key Challenge**: Muon with only a few NS steps achieves SVD-level performance empirically (with superior wall-clock time), yet no theoretical framework covers this regime—practice has far outpaced theory.

**Key Insight**: Directly analyze the polar decomposition approximation error $\varepsilon_q$ induced by NS iterations, and prove that it decays doubly exponentially in the number of steps.

**Core Idea**: The NS approximation error $\varepsilon_q$ decays doubly exponentially → a few NS steps suffice to bring Muon's convergence rate to the SVD level → each step costs far less than SVD → superior wall-clock time.

## Method

### Overall Architecture
At each step, Muon performs: (1) stochastic gradient computation $G_t$; (2) momentum update $M_t = \beta M_{t-1} + G_t$; (3) pre-scaling $X_{t,0} = M_t/\alpha_t$; (4) $q$ NS iterations $X_{t,j} = p_\kappa(X_{t,j-1}X_{t,j-1}^\top)X_{t,j-1}$ to approximate orthogonalization; (5) parameter update $W_t = W_{t-1} - \eta O_t$. The analysis targets $\frac{1}{T}\sum_t \mathbb{E}[\|\nabla f(W_{t-1})\|_*] \leq \epsilon$.

### Key Designs (Theoretical Contributions)

1. **Theorem 1: Non-convex convergence of NS-Muon**:
    - Number of iterations for Muon with $q$ NS steps to reach an $\epsilon$-stationary point: $T = O\left(\frac{C_q \cdot L D}{\epsilon^2}\right)$
    - $C_q$ is the sole constant factor depending on the NS approximation quality.

2. **Theorem 2: Doubly exponential decay of polar decomposition approximation error**:
    - $\varepsilon_q \leq \varepsilon_0^{(2\kappa+1)^q}$ — doubly exponential decay in $q$ and decay in polynomial degree $\kappa$.
    - Implication: $q = 3\text{–}5$ steps with $\kappa = 2\text{–}3$ suffices to achieve $C_q \approx 1$, matching the SVD variant.
    - Wall-clock advantage: NS requires only matrix multiplications (GPU-efficient), whereas SVD costs $O(mn\min(m,n))$.

3. **Theorem 3: Rank advantage over SGD-M**:
    - Muon converges $\sqrt{r}$ times faster than SGD-M, where $r = \min(m,n)$ is the matrix rank.
    - Reason: Muon operates under the nuclear norm → exploits low-rank matrix structure → more efficient search directions.

## Key Experimental Results

### Main Results (Convergence Comparison)

| Method | Metric | Convergence Rate | Rank Dependence |
|--------|--------|-----------------|-----------------|
| SGD-M | Frobenius gradient | $O(1/\sqrt{T})$ | $\sqrt{r}$ loss |
| Muon (SVD) | Nuclear gradient | $O(1/\sqrt{T})$ | No $\sqrt{r}$ loss |
| **Muon (NS, $q$ steps)** | Nuclear gradient | $O(C_q/\sqrt{T})$ | No $\sqrt{r}$ loss |

### Ablation Study ($C_q$ vs. Number of NS Steps $q$)

| NS Steps $q$ | $C_q$ ($\kappa=2$) | $C_q$ ($\kappa=3$) |
|--------------|--------------------|--------------------|
| 1 | Large | Moderate |
| 3 | $\approx 1.01$ | $\approx 1.001$ |
| 5 | $\approx 1.0$ | $\approx 1.0$ |

### Key Findings
- **3–5 NS steps match SVD**: $C_q$ converges doubly exponentially to 1, providing rigorous theoretical justification for the choices made in practice.
- **$\sqrt{r}$ advantage of Muon over SGD-M**: The benefit is pronounced for high-rank matrix parameters such as large attention layers.
- **Wall-clock advantage explained**: NS iterations cost far less than SVD per step while achieving nearly identical iteration counts, yielding lower total runtime.

## Highlights & Insights
- **First convergence guarantees for practical Muon**: Closes the practice–theory gap; all prior theoretical work implicitly assumed exact SVD.
- **Doubly exponential decay as the key insight**: $\varepsilon_q \leq \varepsilon_0^{5^q}$ (for $\kappa=2$) — after 3 steps, the error is below $10^{-100}$.
- **Nuclear norm as the natural metric**: Working with the nuclear norm in matrix space naturally aligns with polar decomposition and reveals the rank advantage.
- **Implications for future matrix optimizers**: The general analysis framework for NS approximation can be extended to other matrix-based optimizers.

## Limitations & Future Work
- This is a purely theoretical contribution with no new experiments (though the paper's explicit goal is to provide theoretical grounding for existing practice).
- The analysis assumes standard smoothness and bounded variance; adaptive methods in the style of Adam are not covered.
- Comparison with second-order methods such as Shampoo/SOAP is not analyzed.

## Related Work & Insights
- **vs. Shen et al. / Li & Hong**: Prior work analyzes SVD-Muon. This paper is the first to analyze NS-Muon—the only theory that matches actual practice.
- **vs. Shampoo/SOAP**: Second-order preconditioners that maintain curvature information. Muon is not a second-order method—it orthogonalizes momentum, a distinct mechanism that is potentially complementary.
- **vs. Orthogonal-SGDM**: Orthogonalization is applied before momentum accumulation. Muon applies momentum first, then orthogonalizes, and replaces SVD with NS.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First convergence theory for NS-Muon as used in practice; the doubly exponential decay result is elegant.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical with no new experiments (justified, as the goal is to provide theoretical explanation for existing empirical practice).
- Writing Quality: ⭐⭐⭐⭐⭐ The research question is clearly stated, theorems build progressively, and the exposition is rigorous and fluent.
- Value: ⭐⭐⭐⭐⭐ Provides much-needed theoretical foundations for one of the most prominent matrix optimizers in current practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICLR 2026\] When to Restart? Exploring Escalating Restarts on Convergence](when_to_restart_exploring_escalating_restarts_on_convergence.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](../../NeurIPS2025/optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [\[AAAI 2026\] A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](../../AAAI2026/optimization/a_unified_convergence_analysis_for_semi-decentralized_learni.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)

</div>

<!-- RELATED:END -->
