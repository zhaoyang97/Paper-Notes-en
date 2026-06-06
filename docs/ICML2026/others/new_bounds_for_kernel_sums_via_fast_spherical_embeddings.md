---
title: >-
  [Paper Note] New Bounds for Kernel Sums via Fast Spherical Embeddings
description: >-
  [ICML 2026][KDE] By accelerating the Bartal-Recht-Schulman 2011 "randomized Nash device" spherical embedding theorem using iterative Fastfood transforms (time $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$)…
tags:
  - "ICML 2026"
  - "KDE"
  - "Gaussian kernel"
  - "Fastfood"
  - "Randomized Hadamard Transform"
  - "Wiener chaos"
date: 2026-05-08
content_hash: 83c9062aeeba5a3e
---

# New Bounds for Kernel Sums via Fast Spherical Embeddings

**Conference**: ICML 2026  
**arXiv**: [2605.01263](https://arxiv.org/abs/2605.01263)  
**Code**: None  
**Area**: Algorithm Theory / Kernel Density Estimation / Random Projection  
**Keywords**: KDE, Gaussian kernel, Fastfood, Randomized Hadamard Transform, Wiener chaos

## TL;DR
By accelerating the Bartal-Recht-Schulman 2011 "randomized Nash device" spherical embedding theorem using iterative Fastfood transforms (time $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$), and using it as a preprocessing step for Gaussian KDE to compress the diameter to $\widetilde{O}(1/\sqrt{\varepsilon})$, this work obtains a new Gaussian KDE query time bound $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$, which outperforms RFF / FJLT+RFF / Fastfood in the regime of small $\varepsilon$ and moderate diameter.

## Background & Motivation

**Background**: Kernel density estimation (KDE) is a fundamental tool in ML, aiming to estimate $\frac{1}{|X|} \sum_{x \in X} \mathbf{k}(x, y)$ for a query $y$ up to accuracy $\pm \varepsilon$ (with high probability). Over the past decade, high-dimensional Gaussian KDE query time has been dominated by three main approaches: (i) RFF $O(d/\varepsilon^2)$, (ii) FJLT + RFF $\widetilde{O}(d + 1/\varepsilon^4)$, (iii) Fastfood $\widetilde{O}(d + \Delta_\sigma^2/\varepsilon^2)$. These are incomparable, with performance depending on the dimension $d$, error $\varepsilon$, and effective diameter $\Delta_\sigma = \Delta/\sigma$.

**Limitations of Prior Work**: Each method has parameter regimes where it is not optimal. Fastfood is best for small diameters, but the $\Delta_\sigma^2 / \varepsilon^2$ term means larger diameters are penalized; RFF/FJLT do not depend on diameter but have heavy dependence on $\varepsilon$. Is it possible to design a bound where the diameter appears in a more favorable way (e.g., $\varepsilon \Delta_\sigma^2$, so smaller $\varepsilon$ helps)?

**Key Challenge**: Fastfood's bottleneck is the output dimension $d' = O(\Delta_\sigma^2 / \varepsilon^2)$, dictated by the data region's diameter. If one could preprocess to "compress the diameter" before Fastfood, reducing the effective diameter seen by Fastfood, overall complexity could improve. However, this preprocessing must be fast and must not distort distances in a way that harms kernel estimation accuracy.

**Goal**: Construct a "fast spherical embedding" with complexity $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$—mapping points to the unit sphere, preserving "small" distances $\leq \sqrt{\varepsilon}$ up to $(1 \pm \varepsilon)$, preventing "large" distances from collapsing below $\Omega(\sqrt{\varepsilon})$, and compressing the diameter to $1/\sqrt{\varepsilon}$. Then, stack a Fastfood layer for KDE.

**Key Insight**: For Gaussian KDE, when the distance $\geq \sqrt{\log(1/\varepsilon)}$, $e^{-\|x-y\|^2} \leq \varepsilon$, so these "large" distances do not need to be preserved precisely—just avoid collapsing them below $\sqrt{\log(1/\varepsilon)}$. This "precise for small distances, non-collapse for large distances" requirement matches the spherical embedding theorem of BRS 2011, but their implementation uses a full Gaussian matrix, $O(d/\varepsilon^2)$, which is slow.

**Core Idea**: Use iterative Fastfood $\psi(H D_2 H D_1 x)$ (two layers of randomized Hadamard transforms) as a "fast version of BRS spherical embedding," prove it satisfies the three properties of spherical embedding, and use it as a preprocessing step for Gaussian KDE, stacking with Fastfood: $\psi(H D_4 H D_3 \cdot s^{-1} \psi(H D_2 H D_1 (s x)))$.

## Method

### Overall Architecture
A two-stage embedding. The first stage is **fast spherical embedding** $\Phi: \mathbb{R}^d \to \mathbb{S}^m$, $m = \widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$, where data and queries are scaled by $s = \Theta(\sqrt{\varepsilon / \log(1/\varepsilon)})$ and passed through an inner Fastfood, outputting on $\mathbb{S}^{2m-1}$, with scaled diameter $\Lambda = s\Delta = \widetilde{O}(\sqrt{\varepsilon} \Delta)$. The second stage is **unscaling + second Fastfood layer** for KDE: after unscaling, points lie on a sphere of radius $s^{-1}$, with new diameter $\widehat{\Delta} = 2 s^{-1} = \widetilde{O}(1/\sqrt{\varepsilon})$. Standard Fastfood (Le-Sarlós-Smola 2013) is then used for KDE approximation, with complexity $\widetilde{O}(m + \widehat{\Delta}^2/\varepsilon^2) = \widetilde{O}(m + 1/\varepsilon^3)$. The total is $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$.

### Key Designs

1. **Fastfood as Spherical Embedding (Theorem 1.3 + 1.2)**:

    - **Function**: Maps any $x \in \mathbb{R}^m$ to the unit sphere $\mathbb{S}^{2m-1}$ (deterministically ensuring $\|\Phi(x)\|_2 = 1$), with three probabilistic properties: (1) distances do not expand by more than $1+\varepsilon$, (2) small distances $\|x-y\|^2 \leq \varepsilon$ do not contract by more than $1-\varepsilon$, (3) distances in $(\varepsilon, \Lambda^2]$ do not collapse below $\Omega(\varepsilon)$.
    - **Mechanism**: The Fastfood matrix $V = \sqrt{m} \cdot H G H B$, where $H$ is a normalized Hadamard matrix, $G = \text{diag}(g)$ is a Gaussian diagonal, $B$ is a Rademacher diagonal sign matrix. The mapping is $\Phi(x)_{2j-2} = \frac{1}{\sqrt{m}} \cos((Vx)_j)$, $\Phi(x)_{2j-1} = \frac{1}{\sqrt{m}} \sin((Vx)_j)$. Fast computation of $Vx$ uses the Walsh-Hadamard transform $O(m \log m)$; output lies on the unit sphere by the trigonometric identity $\sin^2 + \cos^2 = 1$.
    - **Design Motivation**: BRS 2011 used a full Gaussian $W$ matrix for spherical embedding, with time $O(d/\varepsilon^2)$; this work replaces $W$ with the structured matrix $H G H B$ via RHT, reducing time to $O(m \log m)$ while retaining the statistical properties of RHT as a "Gaussian surrogate."

2. **Fourth-Order Wiener Chaos Analysis for Distance Contraction Control**:

    - **Function**: Proves that $\Phi$ does not contract small distances by more than $(1-\varepsilon)$—i.e., Theorem 1.3 item 2.
    - **Mechanism**: Using the Taylor expansion $1 - \cos(\theta) \geq \frac{1}{2}\theta^2 - \frac{1}{24}\theta^4$, one obtains $\|\Phi(x) - \Phi(y)\|^2 \geq Q(z) - \frac{1}{12} W(z)$, where $Q(z) = \frac{1}{m}\|Vz\|^2$ (quadratic term), $W(z) = \frac{1}{m}\sum (Vz)_j^4$ (quartic term). $Q(z)$ is controlled via Bernstein's inequality; $W(z)$ is a Gaussian chaos function, decomposed as $t^4 - 3 = 6 h_2(t) + h_4(t)$ into second and fourth-order Wiener chaos, controlled by Bernstein and Wiener chaos hypercontractivity (Theorem 3.6).
    - **Design Motivation**: The Fastfood analysis in Le-Sarlós-Smola 2013 used Lipschitz Gaussian concentration, which only gives second-moment bounds; to prove contraction lower bounds, the quartic term must be controlled, requiring Wiener chaos decomposition—a key technical innovation of this work.

3. **Scaling Trick to Align "Small Distance Threshold" with Gaussian KDE's Effective Range**:

    - **Function**: Adjusts the BRS embedding's "small distance threshold $\sqrt{\varepsilon}$" to the $\sqrt{\log(1/\varepsilon)}$ required by KDE.
    - **Mechanism**: Inputs are scaled by $s = \Theta(\sqrt{\varepsilon / \log(1/\varepsilon)})$ before embedding, and outputs are unscaled by $s^{-1}$. Thus, original pairs with distance $\leq \sqrt{\log(1/\varepsilon)}$ are preserved accurately in the embedding space, while those $\geq \sqrt{\log(1/\varepsilon)}$ remain at least $\Omega(\sqrt{\log(1/\varepsilon)})$ (so the Gaussian term $e^{-\|x-y\|^2}$ is always below $\varepsilon$ and can be ignored).
    - **Design Motivation**: The BRS embedding directly solves the "$\sqrt{\varepsilon}$ threshold," which is insufficient for KDE; scaling shifts the threshold to the correct scale, preserving KDE accuracy.

### Loss & Training
This is a purely theoretical work, with no training or optimization objectives. All "parameters" (embedding dimension $m$, scaling factor $s$, Hadamard order) are explicitly determined by theoretical analysis.

## Key Experimental Results
This is a theoretical paper with no experimental tables. Complexity comparisons are visualized in Table 1 and Figure 1.

### Main Results

| Method | Query Time | Optimal Regime |
|--------|------------|---------------|
| RFF | $O(d / \varepsilon^2)$ | $d \lesssim \varepsilon^{-2}$ and $\Delta_\sigma \gtrsim \sqrt{d} \varepsilon^{-1.5}$ |
| FJLT + RFF | $\widetilde{O}(d + 1/\varepsilon^4)$ | $d \gtrsim \varepsilon^{-2}$ and $\Delta_\sigma \gtrsim \varepsilon^{-2.5}$ |
| Fastfood | $\widetilde{O}(d + \Delta_\sigma^2/\varepsilon^2)$ | $\Delta_\sigma \lesssim \min\{\sqrt{d}, \varepsilon^{-0.5}\}$ |
| **Ours (Theorem 1.2)** | $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$ | $\varepsilon^{-0.5} \lesssim \Delta_\sigma \lesssim \min\{\sqrt{d} \varepsilon^{-1.5}, \varepsilon^{-2.5}\}$ |

The four methods are incomparable; each is optimal in its own parameter regime. The new method introduced here covers the "moderate diameter + small $\varepsilon$" regime, previously uncovered.

### Ablation Study

| Extension | Kernel | Query Time |
|-----------|--------|------------|
| Theorem 1.4 | Inverse Multi-Quadratic $\mathbf{k}_\beta^{\text{IMQ}}(x,y) = (1 + \|x-y\|^2/\sigma^2)^{-\beta}$ | $\widetilde{O}(d + \varepsilon (\beta \Delta_\sigma)^2 + 1/\varepsilon^3)$ |
| Theorem 1.5 | Gaussian + Differential Privacy (function release) | Same as Theorem 1.2, provided $|X| \geq \widetilde{O}(1/(\varepsilon^2 \varepsilon_{\text{DP}}))$ |

These two extensions validate that the core technique (fast spherical embedding) applies beyond Gaussian KDE—IMQ is handled via Cherapanamjeri-Silwal-Woodruff 2024's function approximation; DP is achieved by controlling probabilistic dependencies among RHT output coordinates (an extra step needed for Fastfood and this work, but not for RFF).

### Key Findings
- In the new bound, $\varepsilon$ appears as a **multiplier** in the diameter term $\varepsilon \Delta_\sigma^2$, not a **divisor**—so smaller $\varepsilon$ actually reduces the diameter term, a fundamental polarity reversal compared to Fastfood's $\Delta_\sigma^2/\varepsilon^2$.
- The key is fourth-order chaos control: Bernstein's second-order analysis only gives upper bounds (distance expansion), but to prove contraction, one must control the variance of $(Vz)_j^4$, which is a Wiener 4th-order chaos quantity, requiring hypercontractivity.
- The two-layer Fastfood composite structure algorithmically echoes the heuristic multi-layer RHT approaches in SORF 2017 and Andoni et al. LSH 2015, but this work provides the first theoretical guarantee for a double-layer version.
- Embedding Theorem 1.3 may have applications far beyond KDE—any application needing "precise for small distances, non-collapse for large distances" fast embeddings can use it directly; the authors list it as an independent contribution.

## Highlights & Insights
- The algorithmic recomposition of "compressing diameter with fast embedding then cascading Fastfood" is particularly efficient—no new algorithm is invented, just a new arrangement of existing blocks, but careful scale alignment yields a new bound.
- Wiener chaos decomposition + hypercontractivity for controlling RHT fourth-order terms is a technical skill worth learning for the ML community: hypercontractivity for Gaussian polynomials is a classical tool but rarely seen in random projection literature; this work provides a clean demonstration, likely to reappear in future analyses of SORF / multi-layer RHT / structured sketches.
- Translating the conceptual result of BRS 2011 ("randomized Nash device") into a fast version brings a tool previously confined to metric embedding into the mainstream of KDE/kernel approximation—a nice bridging contribution.

## Limitations & Future Work
- The upper bound is only optimal in a narrow $\Delta_\sigma$ interval ($\varepsilon^{-0.5} \lesssim \Delta_\sigma \lesssim \varepsilon^{-2.5}$); outside this, previous bounds dominate.
- No experimental validation of constant factors; the $\widetilde{O}$ hides logarithmic factors (especially from the Wiener chaos part) that may be large in practice.
- Only addresses additive error KDE; applying this embedding to relative error KDE (Backurs et al. series) requires further work.
- The spherical embedding theorem itself may have independent applications (Bartal et al. used BRS embedding for Lipschitz extension and small distortion embedding), but this work only applies it to KDE.

## Related Work & Insights
- **vs RFF / FJLT + RFF**: Both do not depend on diameter; this work explicitly depends on $\Delta_\sigma$, but in the more favorable form $\varepsilon \Delta_\sigma^2$.
- **vs Fastfood**: This work is essentially Fastfood stacked on Fastfood—the first for spherical embedding compression, the second for actual KDE estimation.
- **vs Charikar-Siminelakis 2017 series (importance sampling-based KDE)**: That line gives relative error but with higher polynomial complexity; this work focuses on the additive error regime.
- **vs SORF (Yu et al. 2016) / multi-layer RHT LSH**: Both use multiple RHTs in sequence, but SORF is heuristically optimal without theory; this work provides the first theoretical guarantee for double-layer RHT via Wiener chaos.
- **Inspiration**: Spherical embedding + Wiener chaos tools may be applicable to ML systems problems such as attention sketching / transformer KV-cache compression, where "precise for small distances, error-tolerant for large distances" kernel-like operations are also needed.

## Rating
- Novelty: ⭐⭐⭐⭐ Algorithmically a block recomposition; analytically, the introduction of Wiener chaos is a new tool
- Experimental Thoroughness: ⭐⭐ Pure theory, no empirical constant factors or Fastfood/RFF comparisons
- Writing Quality: ⭐⭐⭐⭐⭐ The combination of conceptual diagrams, tables, and complexity regime analysis clearly explains the improvement over previous bounds
- Value: ⭐⭐⭐⭐ Fills a gap in kernel approximation theory for a specific parameter regime; Spherical Embedding Theorem 1.3 has independent application potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](../../ICLR2026/others/probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/others/kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[ICML 2026\] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)
- [\[ICLR 2026\] HEEGNet: Hyperbolic Embeddings for EEG](../../ICLR2026/others/heegnet_hyperbolic_embeddings_for_eeg.md)
- [\[AAAI 2026\] A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](../../AAAI2026/others/a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)

</div>

<!-- RELATED:END -->
