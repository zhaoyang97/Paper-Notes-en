---
title: >-
  [Paper Note] Improved Convergence Analysis of Topology Dependence in Decentralized SGD
description: >-
  [ICML 2026][Optimization][Decentralized SGD] This paper provides a tighter convergence analysis for Decentralized SGD by replacing the topological factor—previously determined solely by the "spectral gap (second largest eigenvalue)"—with the "entire spectrum of the mixing matrix." This theoretically explains for the first time why sparse topologies like rings perform significantly better in training than pessimistic predictions suggest when data is near-homogeneous.
tags:
  - "ICML 2026"
  - "Optimization"
  - "Decentralized SGD"
  - "gossip averaging"
  - "network topology"
  - "spectral gap"
  - "mixing matrix spectrum"
date: 2026-05-08
content_hash: b47ef03f3fec78c7
---

# Improved Convergence Analysis of Topology Dependence in Decentralized SGD

**Conference**: ICML 2026  
**arXiv**: [2606.09154](https://arxiv.org/abs/2606.09154)  
**Code**: To be confirmed  
**Area**: Optimization Theory / Decentralized Learning  
**Keywords**: Decentralized SGD, gossip averaging, network topology, spectral gap, mixing matrix spectrum

## TL;DR
This paper provides a tighter convergence analysis for Decentralized SGD by replacing the topological factor—previously determined solely by the "spectral gap (second largest eigenvalue)"—with the "entire spectrum of the mixing matrix." This theoretically explains for the first time why sparse topologies like rings perform significantly better in training than pessimistic predictions suggest when data is near-homogeneous.

## Background & Motivation

**Background**: In decentralized learning, $n$ nodes each hold data and communicate only with neighbors, utilizing **gossip averaging (weighted averaging via a mixing matrix $\bm{W}$)** to approximate the global average of node parameters. Decentralized SGD (D-SGD) is the fundamental algorithm, where updates are expressed as $\bm{x}_i^{(r+1)}=\sum_j W_{ij}(\bm{x}_j^{(r)}-\eta\nabla F_j(\bm{x}_j^{(r)};\xi_j^{(r)}))$. It reduces communication overhead compared to Federated Learning, but sparse communication can slow down convergence.

**Limitations of Prior Work**: Classical analyses (e.g., Koloskova et al. 2020b) characterize topology using only the **spectral gap** $p\coloneqq 1-\max_{i\ge2}\lambda_i^2$ (defined by the second-largest absolute eigenvalue of the mixing matrix). Such analyses predict that as topology becomes sparser ($p$ approaches 0), convergence **slows down significantly in both homogeneous and heterogeneous cases**. However, extensive experiments have repeatedly observed a contradiction: while topology carries a significant impact under **heterogeneity** (diverse data across nodes, $\zeta\gg0$), training is nearly unaffected under **homogeneity** ($\zeta\approx0$) even with extremely sparse topologies like rings.

**Key Challenge**: Prior analyses are "overly pessimistic" in the homogeneous regime. By bounding gossip averaging errors using only the worst-case scenario (spectral gap), they fail to capture the information carried by the **entire spectrum** of the mixing matrix, and thus cannot explain the "topology insensitivity" under homogeneity. While previous researchers such as Neglia et al. introduced the full spectrum, they either relied on stronger bounded gradient assumptions or failed to achieve a truly improved convergence rate.

**Goal**: To provide a tighter convergence bound than Proposition 1 (the current state-of-the-art) without introducing additional assumptions, such that the bound quantitatively explains the "topology independence" under homogeneity.

**Key Insight**: The authors observe a structural property overlooked in previous proofs: **stochastic gradient noise across nodes is independent and zero-mean**. Worst-case bounds (Lemma 1) are tight only for adversarial inputs where parameters deviate in the same direction; for independent zero-mean noise, the variance is suppressed by the entire spectrum after weighted mixing, rather than following the worst-case eigenvalue.

**Core Idea**: Replace the spectral gap ratio $\frac{1-p}{p}$ with a new graph-theoretic quantity $\frac{1}{n}\sum_{i=2}^n\frac{\lambda_i^2}{1-\lambda_i^2}$ that characterizes the full spectrum as the metric for topological impact.

## Method

### Overall Architecture
This is a purely theoretical paper; the "method" consists of a refined convergence proof. The logical chain is as follows: the slow convergence of D-SGD stems from the **consensus error** $\Xi^{(r)}\coloneqq\frac1n\mathbb{E}\|\bm{X}^{(r)}-\bar{\bm{X}}^{(r)}\|_F^2$, which is driven by three factors: stochastic gradient noise $\sigma$, data heterogeneity $\zeta$, and the inaccuracy of gossip averaging. Previous analyses bounded **all** three factors using the spectral gap $p$. The key contribution of Ours is refining the bound for the **noise component** from the worst-case $\max_{i\ge2}\lambda_i^2$ (i.e., $1-p$) to the average $\frac1n\sum_{i=2}^n\lambda_i^2$, and then integrating this via a multi-step gossip recursion into the final rate as $\frac1n\sum_{i=2}^n\frac{\lambda_i^2}{1-\lambda_i^2}$. The coefficient for the heterogeneity term $\zeta$ is proved to be inherently tied to $p$, representing a fundamental limitation of D-SGD itself rather than a loose analysis.

The proof construction begins with a standard SGD-style descent inequality (Eq. 5), reducing the convergence rate to the summation of consensus errors $\Xi$; it then utilizes Lemma 2 (tight bound for the noise term) and Lemma 3/4 (dericving the $\Xi$ upper bound via multi-step recursion) to bound $\Xi$; finally, the learning rate $\eta$ is tuned to obtain Theorem 1.

### Key Designs

**1. Replacing Spectral Gap with the Full Spectrum: More Precise Topology Characterization**

In prior analyses, topology enters the convergence rate only through the spectral gap ratio $\frac{1-p}{p}$ (the second term of Proposition 1). Theorem 1 replaces this with:
$$\frac{1}{n}\sum_{i=2}^{n}\frac{\lambda_i^2}{1-\lambda_i^2},$$
meaning all non-trivial eigenvalues $\lambda_2,\dots,\lambda_n$ jointly determine convergence. The relationship is as follows:
$$\frac{1-p}{p}=\max_{i\ge2}\frac{\lambda_i^2}{1-\lambda_i^2}\;\ge\;\frac{1}{n}\sum_{i=2}^{n}\frac{\lambda_i^2}{1-\lambda_i^2},$$
The new quantity is **never larger** than the old one, with equality holding only when $\lambda_2=\dots=\lambda_n=0$ (complete graph). For common sparse topologies like rings or tori, the new quantity is significantly smaller—while $\frac{1-p}{p}$ grows quadratically for rings and linearly for tori, the new quantity grows only linearly for rings and logarithmically for tori. This explains why "sparse topologies are not slow" under homogeneity.

**2. Key Lemma: Leveraging Noise Independence to Replace Worst-Case Bounds with Average Bounds**

This is the engine of the improvement (Lemma 2). Prior proofs applied a worst-case inequality $\frac1n\|\bm{X}\bm{W}-\bar{\bm{X}}\|_F^2\le(1-p)\frac1n\|\bm{X}-\bar{\bm{X}}\|_F^2$ (Lemma 1) for gossip averaging errors, which is tight only when $\bm{X}$ is aligned along the second eigenvector. However, the variables being mixed are stochastic gradient noises $\nabla F_i(\bm{x}_i;\xi_i)-\nabla f_i(\bm{x}_i)$, which are **independent and zero-mean**. Leveraging this, it is proved that:
$$\frac1n\mathbb{E}\Big\|(\nabla F(\bm{X};\xi)-\nabla F(\bm{X}))(\bm{W}-\tfrac1n\bm{1}\bm{1}^\top)\Big\|_F^2\le\frac{\sigma^2}{n}\sum_{i=2}^n\lambda_i^2,$$
This compresses the bound from $\max_{i\ge2}\lambda_i^2$ (a single maximum) to $\frac1n\sum_{i=2}^n\lambda_i^2$ (full-spectrum average). Intuitively, independent noises are attenuated by their respective $\lambda_i$ across different characteristic directions; the average is far more moderate than "concentrating all power in the worst-case direction." Note this improvement **only holds for the noise term $\sigma$**; heterogeneity $\zeta$ represents deterministic bias and cannot leverage independence, thus its coefficient remains tied to $p$.

**3. Multi-step Gossip Recursion: Integrating Tight Bounds into Final Convergence Rates**

The right side of the single-step consensus error recursion (Eq. 6) contains $\mathbb{E}\|\bm{X}\bm{W}-\bar{\bm{X}}\|_F^2$, which does not yield a clean recursive form for the $\Xi$ upper bound. The authors introduce a quantity with $k$-step mixing $\Xi^{(r,k)}\coloneqq\frac1n\mathbb{E}\|\bm{X}^{(r)}\bm{W}^k-\bar{\bm{X}}^{(r)}\|_F^2$, obtaining a recursion (Lemma 3) connecting $\Xi^{(r,k)}$, $\Xi^{(r,k+1)}$, and $\Xi^{(r,0)}$ where the noise term is $\frac{\sigma^2\eta^2}{n}\sum_{i=2}^n\lambda_i^{2(k+1)}$. By repeatedly expanding this recursion (Lemma 4), $\sum_k\lambda_i^{2(k+1)}$ sums to the geometric series $\frac{\lambda_i^2}{1-\lambda_i^2}$, bounding consensus error as:
$$\Xi^{(r)}\le\frac{24\zeta^2(1-p)}{p^2}\eta^2+3\sigma^2\Big(\frac1n\sum_{i=2}^n\frac{\lambda_i^2}{1-\lambda_i^2}\Big)\eta^2,$$
This matches the two components appearing in the second term of Theorem 1. Substituting this back into the descent inequality (Eq. 5) yields the final rate, which holds for (strongly) convex and non-convex cases.

**4. Homogeneous Special Case: Convergence on Disconnected Topologies**

In strictly homogeneous ($f_i=f_j$) convex cases, Proposition 2 shows that the convergence rate is the $\min$ of two upper bounds, one of which is $\sqrt{\frac{r_0\sigma_\star^2}{R}}+\frac{Lr_0}{R}$, **which is entirely independent of $p$**. This implies that even if the topology is disconnected ($p=0, \lambda_2=1$, where old analyses predict non-convergence), D-SGD under convex homogeneity still converges at the same rate as single-node SGD (though without $n$-fold linear speedup). This aligned with the analysis of Local SGD. The authors emphasize this is unique to convex functions; in non-convex cases, nodes may converge to different stationary points, and disconnectedness does not guarantee convergence.

## Key Experimental Results

The experiments verify that the full-spectrum quantity $\frac1n\sum_{i=2}^n\frac{\lambda_i^2}{1-\lambda_i^2}$ predicts convergence more accurately than the spectral gap.

### Comparison of Transient Iterations (Non-convex + Homogeneous)
Transient iteration = number of rounds $R$ required for the linear speedup term $\mathcal{O}(\sqrt{L\sigma^2F_0/(nR)})$ to dominate (smaller is better).

| Topology | Proposition 1 (Old) | Theorem 1 (Ours) | Gain |
|------|--------------------|------------------|------|
| Ring | $\mathcal{O}(n^7)$ | $\mathcal{O}(n^5)$ | Reduction by $n^2$ |
| Torus | $\mathcal{O}(n^5)$ | $\mathcal{O}(n^3\log^2 n)$ | From 5th to ~3rd power |
| Hypercube | $\mathcal{O}(n^3\log^2 n)$ | $\mathcal{O}(n^3)$ | Removal of $\log^2 n$ |

Sparse topologies (Ring) show the most significant improvement, consistent with practical observations that "rings are very effective."

### Numerical and Neural Network Verification

| Experiment | Setting | Key Findings |
|------|------|---------|
| Graph Metrics (Fig.1) | ring/line/torus/hypercube + Metropolis | $\frac1n\sum\frac{\lambda_i^2}{1-\lambda_i^2}$ is significantly smaller than $\frac{1-p}{p}$; Ring is linear vs quadratic, Torus is logarithmic vs linear |
| Logistic / Ridge (Fig.2) | $n=200$, MNIST, Homogeneous, **fixed $p$, varying other eigenvalues** | With $p$ constant, loss increases with $\frac1n\sum\frac{\lambda_i^2}{1-\lambda_i^2}$, proving that all eigenvalues matter |
| LeNet / Fashion-MNIST (Fig.3) | $n=25$, Dirichlet($\alpha$), Homogeneous+Heterogeneous | Test accuracy decreases as the full-spectrum quantity increases in both cases; the new quantity remains predictive under heterogeneity |

### Key Findings
- **Controlled variable analysis is a highlight**: Fig.2 maintains the spectral gap $p$ as a constant while varying other eigenvalues. The loss changes monotonically with the full-spectrum quantity, directly falsifying the claim that "spectral gap is the sole determinant."
- **Improvement only in noise terms**: Theoretically, the $\zeta$ coefficient still depends on $p$. The authors argue this is an inherent limitation of D-SGD, consistent with experiments showing topology is critical under heterogeneity—a self-consistent boundary explanation.
- **Value of the Homogeneous Regime**: The authors cite the fact that real-world data heterogeneity is often moderate or that over-parameterization leads to $\zeta_\star=0$, demonstrating that near-homogeneous analysis has practical significance.

## Highlights & Insights
- **Perspective shift from "worst-case" to "average-case"**: Old analyses treated stochastic noise as adversarial input. This paper recognizes that noise is independent across nodes and zero-mean, necessitating the use of the full-spectrum average rather than the worst-case value—addressing a long-ignored structural slack point without new assumptions.
- **New graph-theoretic quantity for topology design**: Traditionally, communication-efficient topology design focused on "maximizing the spectral gap." This paper suggests that $\frac1n\sum\frac{\lambda_i^2}{1-\lambda_i^2}$ is a more refined and potentially superior optimization target for the trade-off between communication and convergence.
- **Fundamental improvement**: Since D-SGD is a foundational algorithm, improving its analysis likely leads to improvements in various decentralized methods built upon it.

## Limitations & Future Work
- **Limited improvement in heterogeneous regimes**: When $\zeta\gg0$, the gap between the new bound and the old bound is small; major gains are concentrated in near-homogeneous regimes.
- **Assumptions**: Requires the mixing matrix to be symmetric and doubly stochastic (Assumption 1); directed or time-varying topologies are not covered.
- **Disconnected convergence only for convex functions**: In non-convex settings, disconnectedness does not guarantee convergence, limiting practical value.
- **Future Directions**: Whether the $\zeta$ coefficient can be decoupled from $p$ under alternative heterogeneity assumptions remains open; and the actual application of the new quantity as a target for topology design.

## Related Work & Insights
- **vs Koloskova et al. (2020b) (Proposition 1)**: They established the state-of-the-art rate using the spectral gap $p$. Ours improves the noise component using the full-spectrum quantity, providing a strictly tighter rate without extra assumptions.
- **vs Neglia et al. (2020)**: They also introduced full eigenvalues (for convex cases) but relied on the "bounded gradient" assumption and did not yield a real improvement over Proposition 1; Ours uses $L$-smoothness and achieves actual improvement.
- **vs Vogels et al. (2022)**: They proposed "effective number of neighbors" to replace the spectral gap, but focused on learning rate conditions, and the quantity depends on hyperparameters. Ours is parameter-free and directly comparable.

## Rating
- Novelty: ⭐⭐⭐⭐ Replacing spectral gap with the full spectrum without extra assumptions is the first substantive improvement on this problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Controlled variables (fixed $p$) are clean and powerful; neural network experiments cover both homogeneous and heterogeneous cases.
- Writing Quality: ⭐⭐⭐⭐ Proof logic is clear; the explanation for why only the noise term is improved is honest and self-consistent.
- Value: ⭐⭐⭐⭐ Theoretically bridges the gap between theory and practice and provides a new target for topology design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](../../AAAI2026/optimization/a_unified_convergence_analysis_for_semi-decentralized_learni.md)
- [\[ICLR 2026\] SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration](../../ICLR2026/optimization/sgd_with_adaptive_preconditioning_unified_analysis_and_momentum_acceleration.md)
- [\[NeurIPS 2025\] Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](../../NeurIPS2025/optimization/unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](../../ICLR2026/optimization/a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)

</div>

<!-- RELATED:END -->
