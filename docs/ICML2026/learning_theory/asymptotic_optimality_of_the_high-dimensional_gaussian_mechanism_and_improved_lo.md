---
title: >-
  [Paper Note] Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy
description: >-
  [ICML 2026][learning_theory][Paper Note] This theoretical paper answers two long-standing open questions: whether the Gaussian mechanism is the optimal choice for additive noise differential privacy in high dimensions (Ans: as the dimension $T\to\infty$, no additive noise can asymptotically outperform the Gaussian at a fixed mean squared error), and whether t
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: 4bf6a80593a427ae
---
# Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy

**Conference**: ICML2026  
**arXiv**: [2606.08681](https://arxiv.org/abs/2606.08681)  
**Code**: None (Theory + numerical algorithms; no code link provided in the paper)  
**Area**: Learning Theory / Differential Privacy  
**Keywords**: Differential Privacy, Gaussian Mechanism, Additive Noise, Spherical Symmetry, Tight Composition

## TL;DR
This theoretical paper answers two long-standing open questions: whether the Gaussian mechanism is the optimal choice for additive noise differential privacy in high dimensions (Ans: as the dimension $T\to\infty$, no additive noise can asymptotically outperform the Gaussian at a fixed mean squared error), and whether there exist mechanisms superior to both Gaussian and $\ell_2$ mechanisms in low dimensions (Ans: yes—the authors propose a three-parameter family of Spherical Generalized Gamma noise, which reduces MSE by up to 15% in certain low-dimensional settings, and they provide tight composition guarantees for this family, resolving an open question by Joseph et al. regarding the $\ell_2$ mechanism).

## Background & Motivation
**Background**: The standard approach to protecting a $T$-dimensional real-valued query $Q(D)\in\mathbb{R}^T$ under Differential Privacy (DP) is the additive noise mechanism $\widehat{Q}=Q+X$, where noise $X$ is sampled from a specific distribution. The Gaussian mechanism $X\sim\mathcal{N}(0,\sigma^2 I_T)$ is most commonly used due to its reliance on $\ell_2$ sensitivity (crucial for maintaining utility in high-dimensional queries), its spherical symmetry (privacy is independent of query direction), tight closed-form privacy analysis, and tight composition guarantees.

**Limitations of Prior Work**: Despite its widespread use, it remained unclear whether the Gaussian mechanism is truly optimal. Recently, Joseph et al. (ICML 2025) demonstrated that the so-called $\ell_2$ mechanism (a high-dimensional version of the Laplace mechanism) can outperform the Gaussian in certain low-dimensional settings, while performing similarly in high dimensions. This revealed two gaps: (Q1) Is the Gaussian truly optimal in high dimensions? (Q3) Exist there mechanisms that simultaneously outperform both Gaussian and $\ell_2$ in low dimensions?

**Key Challenge**: DP is a worst-case model—the addition or removal of a single record can perturb the query in any direction. Consequently, the effect of the "shape" of the noise distribution on the privacy-utility trade-off varies across dimensions. Existing optimality conclusions either restrict the class of noise too strictly (e.g., the restricted additive noise class in Dong et al. 2021, which the $\ell_2$ mechanism has been shown to outperform in low dimensions), focus only on integer/scalar queries, or hold only in the asymptotic sense as the number of compositions $k\to\infty$. A high-dimensional optimality result valid for all additive noise under practical $\delta$ values (e.g., $10^{-3}$) has been missing.

**Goal**: (1) Prove the asymptotic optimality of the Gaussian mechanism in high dimensions within the $(\varepsilon,\delta)$-DP framework for all additive noise; (2) Construct a sufficiently broad noise family and identify members within it that outperform both Gaussian and $\ell_2$ in low dimensions; (3) Provide tight composition guarantees for this family.

**Key Insight**: All spherically symmetric noise can be written as $X=RU$ (where the radial component $R\ge0$ is independent of the uniform direction $U$ on the unit sphere). This allows reducing the complex high-dimensional privacy analysis into a one-dimensional problem concerning $R$.

**Core Idea**: Privacy loss is reduced to a 1D integral using "radial-directional decomposition," and Gaussian asymptotic optimality is proved via a custom Jensen-type supporting line inequality. Simultaneously, Gaussian and $\ell_2$ mechanisms are embedded into a three-parameter Spherical Generalized Gamma family, where numerical search is used to find superior low-dimensional solutions.

## Method

### Overall Architecture
This work is a combination of pure theory and numerical algorithms. There is no pipeline network; the main results consist of two components.

The first component (Q1, High-Dimensional Optimality) argues that focusing on spherically symmetric noise is sufficient: any additive noise remains at the same MSE and its worst-case $\delta$ does not increase after Haar symmetrization. Thus, the spherically symmetric class is at least as good as any additive noise. For any spherically symmetric mechanism, the optimal $\delta$ lower bound is expressed as the expectation $\mathbf{Ex}[g(R^2/T)]$ of the Gaussian privacy function $g(\cdot)$ over the normalized radial energy $R^2/T$. Finally, in the high-privacy regime (small $\delta$), it is proved that $g$ satisfies a supporting line property, such that $\mathbf{Ex}[g(R^2/T)]\ge g(\mathbf{Ex}[R^2/T])=g(e/T)=\delta_G$. This implies no additive noise can achieve a smaller limiting worst-case $\delta$ than the Gaussian at the same MSE.

The second component (Q3, Low-Dimensional Improvements) defines the Spherical Generalized Gamma (SGG) noise family $X\sim\mathsf{SGG}(\alpha,\beta,p)$ with radial component $R\sim\mathsf{GGamma}(\alpha,\beta,p)$. This family includes Gaussian ($p=2,\alpha=T-1,\beta=1/(2\sigma^2)$) and $\ell_2$-Laplace ($p=1,\alpha=T-1,\beta=1/\theta$) as special cases. The authors provide a 1D integral representation of the privacy for this family and a provable $\eta$-tight upper bound oracle. Optimization algorithms are then used to search for $(\alpha,\beta,p)$ that minimize MSE under fixed privacy constraints. Finally, tight composition accounting based on PRV convolution is provided.

### Key Designs

**1. Spherical Reduction + Radial Decomposition: Reducing High-Dimensional Privacy Analysis to 1D**

This handles the challenge where the computational complexity of comparing arbitrary noise distributions on $\mathbb{R}^T$ grows exponentially. Two tools are used: first, **Haar Symmetrization** (Lemma 3.3) — multiplying any noise $X$ by a Haar-uniform orthogonal matrix $M$ yields $X'=MX$, which is spherically symmetric, has the same MSE ($\mathbf{Ex}[\|X'\|_2^2]=\mathbf{Ex}[\|X\|_2^2]$), and has a worst-case $\delta'\le\delta$. Second, **Radial-Directional Decomposition** $X=RU$ combined with the radial-to-spherical density transform $f_X(x)=\frac{\Gamma(T/2)}{2\pi^{T/2}}\|x\|_2^{1-T}f_R(\|x\|_2)$ ensures the noise density is entirely determined by the 1D radial density $f_R$. The optimal $\delta$ lower bound for any spherically symmetric mechanism (Lemma 3.1) is then:

$$\delta \ge \mathbf{Ex}\!\left[g\!\left(\frac{R_T^2}{T}\right)\right] + o(1),$$

where $g(\cdot)$ is the Gaussian privacy function. This converts the comparison of two distributions in $\mathbb{R}^T$ into the comparison of the randomization of $R^2/T$ in 1D.

**2. Jensen-type Supporting Line Property: Proving Asymptotic Optimality**

In 1D, the problem becomes (Proposition 3.1): under the mean-preserving constraint $\mathbf{Ex}[R_T^2/T]=u_0$ (the MSE budget), can any randomization of effective variance $U=R^2/T$ make $\mathbf{Ex}[g(U)]$ smaller than $g(u_0)$? For Gaussian, $R^2/T$ converges to the constant $u_0$ (zero variance) as $T\to\infty$. The authors prove that if $g$ is convex on $[u_0,\infty)$ and lies above its tangent line $\ell_\delta(u)=g(u_0)+g'(u_0)(u-u_0)$ on $[0,u_0]$, then $\mathbf{Ex}[g(U)]\ge g(u_0)$ for any $U$ with $\mathbf{Ex}[U]=u_0$. Lemma 3.2 proves this property holds in the high-privacy regime ($\delta\le\delta_\star$), establishing that the Gaussian is asymptotically optimal (Theorem 3.1).

**3. SGG Mechanism Family + 1D Numerical Accounting: Transcending Gaussian and $\ell_2$ in Low Dimensions**

The three-parameter Generalized Gamma radial density is:

$$f_{\alpha,\beta,p}(r)=\frac{p\,\beta^{(\alpha+1)/p}}{\Gamma(\frac{\alpha+1}{p})}r^{\alpha}\exp(-\beta r^p),$$

where $\alpha$ controls near-zero behavior, $\beta$ sets the scale, and $p$ adjusts tail decay. This family strictly generalizes both Gaussian ($p=2$) and $\ell_2$-Laplace ($p=1$). Privacy is calculated via a 1D integral of $r\in(0,\infty)$ (Lemma 4.1). Using Bayesian optimization to search $(\alpha,p)$ and bisection for the MSE, the authors identify parameters that achieve lower MSE than both baselines in low dimensions.

**4. Tight Composition Accounting: Resolving the $\ell_2$ Mechanism Open Question**

The Privacy Loss Random Variables (PRVs) sum under independent composition (Proposition 4.2). Once the single-step PRV distribution is calculable, the composition profile is obtained via convolution. The authors use FFT to compute $k$-fold convolutions of discretized PRVs. Since the $\ell_2$ mechanism is a special case of SGG, this framework provides the tight composition accounting for $\ell_2$ mechanisms requested by Joseph et al. (ICML 2025).

## Key Experimental Results

### Main Results (Numerical Verification)
Numerical lower bounds for the threshold $\delta_\star$ (fixed sensitivity $s=1$), proving Theorem 3.1 covers practical privacy values:

| $\varepsilon$ | lower bound of $\delta_\star$ | $\varepsilon$ | lower bound of $\delta_\star$ |
|---------------|---------------------|---------------|---------------------|
| 0.25 | 0.7367 | 4.00 | 0.4170 |
| 0.50 | 0.7070 | 8.00 | 0.2922 |
| 1.00 | 0.6492 | 16.00 | 0.1976 |
| 2.00 | 0.5491 | — | — |

Since these $\delta_\star$ are much larger than the commonly used $\delta\le10^{-3}$, Gaussian asymptotic optimality holds within practical ranges.

### Comparison of SGG Family and Special Cases
The SGG family recovers classic mechanisms and identifies superior members:

| Mechanism | $(\alpha,\beta,p)$ | Role |
|------|---------------------|------|
| Gaussian $\mathcal{N}(0,\sigma^2 I_T)$ | $(T-1,\ 1/(2\sigma^2),\ 2)$ | SGG Special Case |
| $\ell_2$-Laplace (scale $\theta$) | $(T-1,\ 1/\theta,\ 1)$ | SGG Special Case |
| Low-Dim Optimal SGG | Searched $p^*$ away from 1, 2 | MSE ≤15% lower than both |

### Key Findings
- **Gaussian asymptotic optimality** stems from the fact that its $R^2/T$ becomes a constant in high dimensions; Jensen’s inequality implies any randomization increases $\mathbf{Ex}[g(U)]$.
- **Low-dimensional improvements are localized**: MSE reductions (up to 15%) exist only in selected low-dimensional regions and vanish quickly as $T$ increases, consistent with Theorem 3.1.
- **Correction of prior literature**: A bug was found in the proof of the R1SMG mechanism (Ji & Li 2024), which claimed errors an order of magnitude lower than Gaussian; numerical analysis shows its privacy guarantee is significantly weaker than claimed.
- **The study focuses on practical $\delta$ (e.g., $10^{-3}$)** in single-shot settings, unlike Alghamdi et al. who focused on $k\to\infty$ and extremely small $\delta=10^{-8}$.

## Highlights & Insights
- **Radial-directional decomposition is a powerful reduction tool**: Converting comparisons in $\mathbb{R}^T$ into 1D integrals of $R$ enables both the optimality proof and high-precision numerical accounting.
- **Custom Jensen condition**: Replacing global convexity with a supporting line property (upper convexity + lower tangent bound) allows proving optimality even where the privacy function $g$ is not convex.
- **Unified SGG Family**: Treating Gaussian vs. $\ell_2$ as a continuous optimization problem within the SGG family provides a more comprehensive perspective and solves the $\ell_2$ composition problem.

## Limitations & Future Work
- **Asymptotic nature**: Theorem 3.1 is for $T\to\infty$ and does not provide a finite $T_0$ threshold or a convergence rate.
- **Narrow low-dimensional advantage**: Improvements in MSE are $\le15\%$ and diminish rapidly with dimensionality.
- **Scope of noise**: The study is restricted to additive noise and does not cover data-dependent or non-additive mechanisms.

## Related Work & Insights
- **vs. Dong et al. (2021)**: They proved Gaussian optimality within a narrower class using Gaussian DP (GDP); this work proves it for all additive noise under $(\varepsilon,\delta)$-DP.
- **vs. Joseph et al. (2025)**: This work answers their open question by providing tight composition for the $\ell_2$ mechanism.
- **vs. Alghamdi et al. (2023)**: Their proof required $k\to\infty$ and very small $\delta$; this work addresses the single-step setting with realistic $\delta$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](../../NeurIPS2025/learning_theory/transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](../../NeurIPS2025/learning_theory/a_highdimensional_statistical_method_for_optimizing_transfer.md)
- [\[ICLR 2026\] A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data](../../ICLR2026/learning_theory/a_generalized_geometric_theoretical_framework_of_centroid_discriminant_analysis_.md)
- [\[ICML 2026\] Active Learning with Low-Rank Structure for Data Selection](active_learning_with_low-rank_structure_for_data_selection.md)
- [\[ICML 2026\] Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation](catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta.md)

</div>

<!-- RELATED:END -->
