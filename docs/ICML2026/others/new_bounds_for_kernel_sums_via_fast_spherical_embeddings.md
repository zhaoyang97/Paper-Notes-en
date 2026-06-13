---
title: >-
  [Paper Note] New Bounds for Kernel Sums via Fast Spherical Embeddings
description: >-
  [ICML 2026][KDE] By accelerating the "Randomized Nash Device" spherical embedding theorem from Bartal-Recht-Schulman 2011 using iterative Fastfood transforms (time $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$) and u…
tags:
  - "ICML 2026"
  - "KDE"
  - "Gaussian kernel"
  - "Fastfood"
  - "Randomized Hadamard Transform"
  - "Wiener chaos"
date: 2026-05-08
content_hash: 1c3a5d14daad14a5
---

# New Bounds for Kernel Sums via Fast Spherical Embeddings

**Conference**: ICML 2026  
**arXiv**: [2605.01263](https://arxiv.org/abs/2605.01263)  
**Code**: None  
**Area**: Algorithm Theory / Kernel Density Estimation / Random Projections  
**Keywords**: KDE, Gaussian kernel, Fastfood, Randomized Hadamard Transform, Wiener chaos

## TL;DR
By accelerating the "Randomized Nash Device" spherical embedding theorem from Bartal-Recht-Schulman 2011 using iterative Fastfood transforms (time $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$) and utilizing it as a preprocessing step for Gaussian KDE to compress the diameter to $\widetilde{O}(1/\sqrt{\varepsilon})$, a new Gaussian KDE query time bound of $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$ is achieved. This outperforms RFF / FJLT+RFF / Fastfood in the regime of small $\varepsilon$ and moderate diameter.

## Background & Motivation

**Background**: Kernel Density Estimation (KDE) is a fundamental tool in ML, aiming to estimate $\frac{1}{|X|} \sum_{x \in X} \mathbf{k}(x, y)$ for a query $y$ with precision $\pm \varepsilon$ (with high probability). High-dimensional Gaussian KDE query times have been improved by three major methods over the past decade: (i) RFF $O(d/\varepsilon^2)$, (ii) FJLT + RFF $\widetilde{O}(d + 1/\varepsilon^4)$, and (iii) Fastfood $\widetilde{O}(d + \Delta_\sigma^2/\varepsilon^2)$. These are mutually incomparable, depending on specific values of dimension $d$, error $\varepsilon$, and effective diameter $\Delta_\sigma = \Delta/\sigma$.

**Limitations of Prior Work**: Each method has "uncovered" parameter intervals. Fastfood is optimal for small diameters, but in $\Delta_\sigma^2 / \varepsilon^2$, the diameter appears in the numerator (worsening as diameter increases); RFF/FJLT do not depend on diameter but have a heavy dependence on $\varepsilon$. Is it possible to construct a bound where the diameter appears in a "friendlier" way (e.g., $\varepsilon \Delta_\sigma^2$, where smaller $\varepsilon$ is better)?

**Key Challenge**: The bottleneck of Fastfood is the output dimension $d' = O(\Delta_\sigma^2 / \varepsilon^2)$, which must be determined by the diameter of the data region. If a "diameter compression" preprocessing step could be performed before Fastfood to reduce the effective diameter, the overall complexity could be improved. However, this preprocessing must be fast and must not distort distances in a way that affects kernel estimation accuracy.

**Goal**: Construct a "fast spherical embedding" with time complexity $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$ that maps points to a unit sphere, preserves "small" distances $\leq \sqrt{\varepsilon}$ within $(1 \pm \varepsilon)$, and prevents "large" distances from collapsing below $\Omega(\sqrt{\varepsilon})$, effectively compressing the diameter to $1/\sqrt{\varepsilon}$. Then, stack a Fastfood layer for KDE.

**Key Insight**: A key observation for Gaussian KDE is that when distances $\geq \sqrt{\log(1/\varepsilon)}$, the value $e^{-\|x-y\|^2} \leq \varepsilon$. Thus, these "large" distances do not need precise preservation as long as they do not collapse to values smaller than $\sqrt{\log(1/\varepsilon)}$. This requirement of "precise small distances, non-collapsing large distances" exactly corresponds to the spherical embedding theorem introduced by BRS 2011. However, their implementation used full Gaussian matrices $O(d/\varepsilon^2)$, which is not fast enough.

**Core Idea**: Use an iterative Fastfood transform $\psi(H D_2 H D_1 x)$ (two layers of randomized Hadamard transforms) as a "fast version of BRS spherical embedding." This is proven to satisfy the three properties of spherical embedding and is then utilized as a Gaussian KDE preprocessing step layered above Fastfood, resulting in a dual-layer Fastfood: $\psi(H D_4 H D_3 \cdot s^{-1} \psi(H D_2 H D_1 (s x)))$.

## Method

### Overall Architecture
A two-stage embedding. Stage 1: **Fast Spherical Embedding** $\Phi: \mathbb{R}^d \to \mathbb{S}^m$, $m = \widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$. Data and queries are scaled by $s = \Theta(\sqrt{\varepsilon / \log(1/\varepsilon)})$ and passed through the inner Fastfood. The output lies on $\mathbb{S}^{2m-1}$ with a scaled diameter $\Lambda = s\Delta = \widetilde{O}(\sqrt{\varepsilon} \Delta)$. Stage 2: **Unscaling + Second Layer Fastfood** for KDE. After unscaling, points lie on a sphere of radius $s^{-1}$ with a new diameter $\widehat{\Delta} = 2 s^{-1} = \widetilde{O}(1/\sqrt{\varepsilon})$. Standard Fastfood (Le-Sarlós-Smola 2013) is then applied for KDE approximation with complexity $\widetilde{O}(m + \widehat{\Delta}^2/\varepsilon^2) = \widetilde{O}(m + 1/\varepsilon^3)$. Summing both stages yields exactly $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$.

### Key Designs

1.  **Fastfood as Spherical Embedding (Theorem 1.3 + 1.2)**:
    - **Function**: Maps any $x \in \mathbb{R}^m$ to a unit sphere $\mathbb{S}^{2m-1}$ (deterministic guarantee $\|\Phi(x)\|_2 = 1$) with three probabilistic conditions: (1) distance expansion does not exceed $1+\varepsilon$, (2) small distances $\|x-y\|^2 \leq \varepsilon$ do not contract more than $1-\varepsilon$, (3) distances $\in (\varepsilon, \Lambda^2]$ do not collapse below $\Omega(\varepsilon)$.
    - **Mechanism**: The Fastfood matrix is $V = \sqrt{m} \cdot H G H B$, where $H$ is a normalized Hadamard matrix, $G = \text{diag}(g)$ is a Gaussian diagonal, and $B$ is a Rademacher diagonal sign matrix. The mapping is defined as $\Phi(x)_{2j-2} = \frac{1}{\sqrt{m}} \cos((Vx)_j)$ and $\Phi(x)_{2j-1} = \frac{1}{\sqrt{m}} \sin((Vx)_j)$. Fast computation of $Vx$ relies on the Walsh-Hadamard Transform $O(m \log m)$; the unit sphere constraint is automatically guaranteed by the identity $\sin^2 + \cos^2 = 1$.
    - **Design Motivation**: BRS 2011 implemented the same spherical embedding using a full Gaussian matrix $W$ with time $O(d/\varepsilon^2)$. This work replaces $W$ with the structured matrix $H G H B$ via RHT, reducing time to $O(m \log m)$ while retaining the statistical properties of Gaussian "approximations" inherent in RHT.

2.  **Fourth-order Wiener Chaos Analysis for Distance Contraction Control**:
    - **Function**: Proves that $\Phi$ does not contract small distances by more than $(1-\varepsilon)$ (Item 2 of Theorem 1.3).
    - **Mechanism**: Using the Taylor expansion $1 - \cos(\theta) \geq \frac{1}{2}\theta^2 - \frac{1}{24}\theta^4$, the bound $\|\Phi(x) - \Phi(y)\|^2 \geq Q(z) - \frac{1}{12} W(z)$ is derived, where $Q(z) = \frac{1}{m}\|Vz\|^2$ (quadratic term) and $W(z) = \frac{1}{m}\sum (Vz)_j^4$ (fourth-order term). $Q(z)$ is controlled via Bernstein inequalities; $W(z)$ is a Gaussian chaos function, decomposed using the identity $t^4 - 3 = 6 h_2(t) + h_4(t)$ into a sum of 2nd and 4th order Wiener chaos, controlled respectively by Bernstein and Wiener chaos hypercontractivity (Theorem 3.6).
    - **Design Motivation**: The Fastfood analysis by Le-Sarlós-Smola 2013 used Lipschitz Gaussian concentration, providing only second-order moments. To prove the collapse lower bound, the four-order term must be controlled, which requires Wiener chaos decomposition—the core technical innovation of this paper.

3.  **Scaling Trick to Align "Small Distance Threshold" with Gaussian KDE Effective Distance**:
    - **Function**: Converts the BRS embedding "small distance threshold $\sqrt{\varepsilon}$" to the $\sqrt{\log(1/\varepsilon)}$ required for KDE.
    - **Mechanism**: Inputs are scaled by $s = \Theta(\sqrt{\varepsilon / \log(1/\varepsilon)})$ before embedding and inverse-scaled by $s^{-1}$ after. This ensures pairs with original distances $\leq \sqrt{\log(1/\varepsilon)}$ are precisely preserved in the embedding space, while pairs with distances $\geq \sqrt{\log(1/\varepsilon)}$ maintain at least $\Omega(\sqrt{\log(1/\varepsilon)})$ (thus the Gaussian term $e^{-\|x-y\|^2}$ remains less than $\varepsilon$ and can be ignored).
    - **Design Motivation**: Directly applying the BRS embedding solves for a "$\sqrt{\varepsilon}$ threshold," which is insufficient for KDE; scaling shifts the threshold to the correct scale to maintain KDE precision.

### Loss & Training
This is a purely theoretical paper; there is no training or optimization objective. All "parameters" (embedding dimension $m$, scaling factor $s$, Hadamard order) are explicitly determined by theoretical analysis.

## Key Experimental Results
This is a purely theoretical paper with no experimental tables. Complexity comparison visualizations are provided in Table 1 and Figure 1.

### Main Results

| Method | Query Time | Optimal Regime |
|------|----------|----------|
| RFF | $O(d / \varepsilon^2)$ | $d \lesssim \varepsilon^{-2}$ and $\Delta_\sigma \gtrsim \sqrt{d} \varepsilon^{-1.5}$ |
| FJLT + RFF | $\widetilde{O}(d + 1/\varepsilon^4)$ | $d \gtrsim \varepsilon^{-2}$ and $\Delta_\sigma \gtrsim \varepsilon^{-2.5}$ |
| Fastfood | $\widetilde{O}(d + \Delta_\sigma^2/\varepsilon^2)$ | $\Delta_\sigma \lesssim \min\{\sqrt{d}, \varepsilon^{-0.5}\}$ |
| **Ours (Theorem 1.2)** | $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$ | $\varepsilon^{-0.5} \lesssim \Delta_\sigma \lesssim \min\{\sqrt{d} \varepsilon^{-1.5}, \varepsilon^{-2.5}\}$ |

The four methods are mutually incomparable; each is optimal within its own parameter interval. The method introduced in this paper occupies the "moderate diameter + small $\varepsilon$" interval, a regime not previously covered.

### Ablation Study

| Extension | Kernel | Query Time |
|------|------|----------|
| Theorem 1.4 | Inverse Multi-Quadratic $\mathbf{k}_\beta^{\text{IMQ}}(x,y) = (1 + \|x-y\|^2/\sigma^2)^{-\beta}$ | $\widetilde{O}(d + \varepsilon (\beta \Delta_\sigma)^2 + 1/\varepsilon^3)$ |
| Theorem 1.5 | Gaussian + Differential Privacy (function release) | Same as Theorem 1.2, given $|X| \geq \widetilde{O}(1/(\varepsilon^2 \varepsilon_{\text{DP}}))$ |

Two extensions verify that the core technology of the main theorem (fast spherical embedding) is applicable beyond Gaussian KDE—IMQ is achieved via functional approximation from Cherapanamjeri-Silwal-Woodruff 2024; DP is achieved by controlling probabilistic dependence between RHT output coordinates (an extra step required for Fastfood and this work that RFF does not need).

### Key Findings
- In the new bound, $\varepsilon$ in the diameter term $\varepsilon \Delta_\sigma^2$ is a **multiplier** rather than a **divisor**—this means a smaller $\varepsilon$ actually makes the diameter term smaller, a fundamental polarity shift compared to Fastfood's $\Delta_\sigma^2/\varepsilon^2$.
- The fourth-order chaos control is critical: while Bernstein second-order analysis can only provide an upper bound (distance expansion), proving the distance contraction lower bound requires observing the variance of $(Vz)_j^4$, which is a 4th order Wiener chaos quantity requiring hypercontractivity.
- The composite structure of two-layer Fastfood algorithmically echoes heuristic practices using 3 layers of RHT in SORF 2017 and Andoni et al. LSH 2015, but this paper provides the first theoretical guarantee for a double-layer version.
- The applications of the Embedding Theorem 1.3 likely extend far beyond KDE—any application requiring fast embeddings with "precise small distances and non-collapsing large distances" can use it directly; the authors list it as an independent contribution.

## Highlights & Insights
- The algorithmic re-composition idea of "using fast embedding to compress diameter then cascading Fastfood" is highly efficient—it does not invent entirely new algorithms but reconnects existing blocks, achieving a new bound through precise scale alignment.
- Using Wiener chaos decomposition and hypercontractivity for RHT fourth-order term control is a robust technique for the ML community: hypercontractivity of Gaussian polynomials is an old tool rarely seen in random projection literature. This paper provides a clean demonstration; this toolkit will likely reappear in future analyses of SORF, multi-layer RHT, or structured sketches.
- Translating the conceptual result (the "randomized Nash device") of BRS 2011 into a fast version bridges a tool previously isolated in metric embedding theory into the mainstream of KDE and kernel approximation.

## Limitations & Future Work
- The upper bound is optimal only in a narrow $\Delta_\sigma$ interval ($\varepsilon^{-0.5} \lesssim \Delta_\sigma \lesssim \varepsilon^{-2.5}$); outside this range, previous bounds still dominate.
- There is no experimental verification of the constant factors; the logarithmic factors hidden in $\widetilde{O}$ (especially for Wiener chaos) might be large in practice.
- The work focuses only on additive error KDE; applying this embedding to relative error KDE (like the Backurs et al. series) would require additional work.
- While the spherical embedding theorem itself may have independent applications (Bartal et al. used BRS embedding for Lipschitz extension and small distortion embedding), this paper only utilizes it for KDE.

## Related Work & Insights
- **vs RFF / FJLT + RFF**: These do not depend on diameter, whereas this work explicitly depends on $\Delta_\sigma$, albeit in the friendlier $\varepsilon \Delta_\sigma^2$ form.
- **vs Fastfood**: This work essentially nests Fastfood within Fastfood—the former for spherical embedding compression, the latter for actual KDE estimation.
- **vs Charikar-Siminelakis 2017 series (importance sampling-based KDE)**: That line provides relative error but with higher polynomial complexity; this paper focuses on the additive error track.
- **vs SORF (Yu et al. 2016) / Multi-layer RHT LSH**: These use multiple RHTs in sequence, but SORF is a heuristic for empirical optimality lacking theory; this paper provides the first theoretical guarantee for two-layer RHT using Wiener chaos.
- **Insight**: Spherical embedding and Wiener chaos tools might be applicable to ML systems problems like attention sketching or transformer KV-cache compression, where kernel-like operations needing "precise small distances and allowed error for large distances" are required.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Algorithmic block re-composition, with Wiener chaos as a new tool in the analysis.
- **Experimental Thoroughness**: ⭐⭐ Purely theoretical, with no measurement of constant factors or comparisons against Fastfood/RFF.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The combination of conceptual diagrams, tables, and complexity interval analysis clearly explains why the new bound is superior.
- **Value**: ⭐⭐⭐⭐ Fills a specific parameter interval gap in kernel approximation theory; the spherical embedding Theorem 1.3 has potential for independent application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](../../ICLR2026/others/probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ICML 2026\] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)
- [\[ICLR 2026\] HEEGNet: Hyperbolic Embeddings for EEG](../../ICLR2026/others/heegnet_hyperbolic_embeddings_for_eeg.md)
- [\[AAAI 2026\] A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](../../AAAI2026/others/a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)
- [\[ICLR 2026\] Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](../../ICLR2026/others/characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)

</div>

<!-- RELATED:END -->
