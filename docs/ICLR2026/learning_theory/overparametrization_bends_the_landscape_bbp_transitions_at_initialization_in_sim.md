---
title: >-
  [Paper Note] Overparametrization bends the landscape: BBP transitions at initialization in simple Neural Networks
description: >-
  [ICLR 2026][Learning Theory][BBP Transition] By generalizing classical phase retrieval to a "two-layer quadratic activation teacher-student network with arbitrary width," this work uses field-theoretic methods to analytically calculate the **Hessian spectrum of the loss at initialization**. It discovers that the BBP transition threshold $\alpha_{\text{BBP}}$, where outlier eigenvalues (carrying teacher signal information) emerge from the bulk…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Loss Landscape"
  - "Statistical Physics"
  - "BBP Transition"
  - "Overparametrization"
  - "Phase Retrieval"
  - "Random Matrix Theory"
date: 2026-05-08
content_hash: 4a9b1663a24c2328
---

# Overparametrization bends the landscape: BBP transitions at initialization in simple Neural Networks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=xDLE5n3x9Y](https://openreview.net/forum?id=xDLE5n3x9Y)  
**Code**: To be confirmed  
**Area**: Learning Theory / Loss Landscape / Statistical Physics  
**Keywords**: Loss Landscape, BBP Transition, Overparametrization, Phase Retrieval, Random Matrix Theory

## TL;DR
By generalizing classical phase retrieval to a "two-layer quadratic activation teacher-student network with arbitrary width," this work uses field-theoretic methods to analytically calculate the **Hessian spectrum of the loss at initialization**. It discovers that the BBP transition threshold $\alpha_{\text{BBP}}$, where outlier eigenvalues (carrying teacher signal information) emerge from the bulk, decreases with overparametrization. As the student becomes wider, less data is required for the signal to emerge within the curvature of the random initial point, even reaching the information-theoretic weak recovery lower bound $p^*/2$ in the limit.

## Background & Motivation
**Background**: The high-dimensional non-convex loss landscape is a central puzzle in understanding neural network optimization. A recurring observation is that when the amount of data is sufficiently large relative to the dimension $N$ (high signal-to-noise ratio, SNR), the landscape "trivializes" and becomes approximately convex. Surprisingly, gradient-based methods often succeed even in regimes with low-to-medium SNR where numerous uninformative pseudo-minima exist. The statistical physics community attributes this "blessing of dimensionality" to a phenomenon where high-dimensional basins of attraction near random initial points, though initially uninformative, develop a direction of instability toward the signal as SNR increases. This instability corresponds to a **Baik–Ben Arous–Péché (BBP) transition** in the local Hessian spectrum, where an eigenvalue escapes the continuous bulk.

**Limitations of Prior Work**: Spectral methods exploit this structure by constructing a data-dependent matrix and using its principal eigenvector as a signal estimate or a warm start for iterative algorithms. Previous works (Biroli 2020, Bonnaire 2025) noted that for problems like phase retrieval, the spectral matrix can be viewed as the "Hessian averaged over student weights." However, these analyses are mostly restricted to the single-node case ($p=p^*=1$). The question of **how overparametrization (student wider than teacher) changes the signal information in the initial Hessian and shifts the BBP threshold remains largely unexplored**.

**Key Challenge**: Intuitively, overparametrization "smooths" the landscape to aid optimization, but there is no quantitative characterization of how this smoothing acts on the curvature at initial random points, whether it truly causes the signal to appear earlier, or if counterexamples exist.

**Goal**: Within a solvable model where the degree of overparametrization can be freely tuned, analytically answer three questions: (1) How the BBP threshold $\alpha_{\text{BBP}}$ of the initial Hessian spectrum changes with student width $p$, teacher width $p^*$, and the loss normalization constant $a$; (2) Whether the transition is continuous or discontinuous; (3) The gap between real behavior at finite dimension $N$ and $N\to\infty$ predictions.

**Key Insight**: Generalize single-node phase retrieval to a **teacher-student model of two-layer soft committee machines with quadratic activation**, where student width $p$ and teacher width $p^*$ are both arbitrary and finite, and input dimension $N\to\infty$. When $p=p^*=1$, it reduces to standard phase retrieval; $p>p^*$ represents overparametrization. The authors employ field theory techniques, rarely used in the ML community, to directly compute the spectrum of the "true Hessian" (rather than the averaged Hessian).

**Core Idea**: Overparametrization is equivalent to "implicitly averaging over many student nodes" on the landscape, thereby **bending the landscape to push the BBP transition toward a lower SNR** and altering the qualitative nature of the transition (from continuous to discontinuous).

## Method

### Overall Architecture
This work does not study the dynamics of gradient descent but focuses on the "geometry of the loss landscape at initialization." Specifically, it examines whether the local curvature (Hessian) of the empirical loss can leak information about the teacher signal when student weights are randomly sampled from the sphere $S^{N-1}(\sqrt N)$. The analytical pipeline is: **Construct a solvable model → Derive the initial Hessian → Analytically compute its spectrum (bulk + outliers) → Use the spectral criterion to locate the BBP threshold → Scan $p, p^*, a$ for overparametrization effects → Validate with finite-$N$ simulations**.

At the model level, both teacher and student are two-layer networks with quadratic activations:

$$y(x^\mu) = \frac{1}{p^*}\sum_{l=1}^{p^*}(w_l^*\cdot x^\mu)^2,\qquad \hat y(x^\mu) = \frac{1}{p}\sum_{k=1}^{p}(w_k\cdot x^\mu)^2$$

Training uses $M=\alpha N$ Gaussian samples, where the ratio $\alpha=M/N$ acts as the SNR. The loss is a family of normalized squared losses adjusted by a constant $a>0$:

$$L_w = \frac12\sum_{\mu=1}^{\alpha N}\frac{[y(x^\mu)-\hat y(x^\mu)]^2}{a + y(x^\mu)}$$

The parameter $a$ in the denominator is critical: it suppresses pathologies caused by occasionally near-zero or extremely large teacher outputs, ensuring the Hessian spectrum has a finite left edge—a prerequisite for analyzing an eigenvalue escaping the bulk. This $a$ later becomes the key knob determining whether the transition is continuous or discontinuous.

### Key Designs

**1. Generalized Phase Retrieval Teacher-Student Model: Overparametrization as a Tunable Knob**

Standard phase retrieval (recovering a hidden signal from squared projections $|w^*\cdot x|^2$) is famously non-convex but involves only a single hidden node, precluding a discussion on "overparametrization." This work generalizes it to a two-layer quadratic network of arbitrary width ($p\ge p^*\ge 1$). Thus, the degree of overparametrization is continuously characterized by $p/p^*$, and $p=p^*=1$ recovers classical phase retrieval. Since student output is invariant under $W\mapsto OW$ (orthogonal $O$), the overlap matrix Frobenius norm $m_{kl}$ is used to measure signal recovery.

**2. Analytical Hessian Spectrum via Field Theory: Diagrammatic Expansion of Self-energy $\Sigma(z)$**

The initial Hessian $H\in\mathbb R^{pN\times pN}$ is a random matrix with $p^2$ blocks of size $N\times N$. The authors calculate the Stieltjes transform of the spectral distribution:

$$g(z)=\lim_{N\to\infty}\mathbb E_x \frac1N \mathrm{Tr}\Big(\frac{1}{zI-H}\Big)$$

The technique involves expressing this as a Gaussian integral of an $N$-dimensional scalar field $\psi$, expanding $e^{-\frac12\psi^\top H\psi}$, and performing Gaussian averaging over field $\psi$ and data $x^\mu$. Using Wick’s theorem and Feynman diagrams, the sum is reduced to **one-particle irreducible (1PI) diagrams**, denoted as the self-energy $\Sigma(z)$. The transform then follows a simple closed form:

$$g(z)=\frac{1}{z-\Sigma(z)}$$

This field-theoretic approach to random matrices (Zee 1996) allows for deriving both the continuous bulk and outlier eigenvalues $\lambda^*$.

**3. BBP Criterion + Continuous/Discontinuous Dichotomy**

The critical SNR is determined by the meeting of the bulk left edge $\lambda_-$ and the outlier $\lambda^*$:

$$\lambda^*(\alpha_{\text{BBP}})=\lambda_-(\alpha_{\text{BBP}})$$

Below $\alpha_{\text{BBP}}$, the spectrum is uninformative; above it, the outlier eigenvector develops non-zero overlap with the signal. The transition is categorized by the shape of the spectral density at the left edge:
- **Continuous BBP**: The edge is "steep," density vanishes as a square root $\rho(\lambda)\propto(\lambda-\lambda_-^{\text{sh}})^{1/2}$. Overlap $m$ grows continuously from 0.
- **Discontinuous BBP**: The edge is "smooth," density decays exponentially $\rho(\lambda)\propto\exp\!\big(-\frac{A}{\lambda-\lambda_-^{\text{sm}}}\big)$. Overlap $m$ jumps from 0 to a finite value at the threshold.

**4. Bending the Landscape and Finite-$N$ Corrections**

Analysis reveals that **increasing $p$ for a fixed $a$ generally lowers $\alpha_{\text{BBP}}$**. In the infinite overparametrization limit $p\to\infty$, the transition is **always discontinuous**, with the threshold:

$$\alpha^{p=\infty}_{\text{BBP}}=\frac{p^*(a+1)}{2}$$

As $a\to 0$, this reaches $p^*/2$, the information-theoretic weak recovery threshold. Furthermore, smooth edges in discontinuous cases lead to $O(1/\log N)$ deviations in finite dimensions (much larger than the $O(N^{-2/3})$ in continuous cases), causing "premature recovery" where signal information is retained even for $\alpha < \alpha_{\text{BBP}}$.

## Key Experimental Results

The "experiments" compare analytical predictions with finite-dimensional numerical simulations.

### Main Results (Analytical Predictions)

| Setting | Conclusion | Formula / Phenomenon |
|------|------|------------|
| General $p, p^*, a$ | BBP transition occurs in initial Hessian | $\lambda^*(\alpha_{\text{BBP}})=\lambda_-(\alpha_{\text{BBP}})$ |
| Fixed $a$, Increasing $p$ | $\alpha_{\text{BBP}}$ generally decreases (earlier recovery) | Wider students require less data |
| $p\to\infty$ | Transition is always discontinuous | $\alpha^{p=\infty}_{\text{BBP}}=p^*(a+1)/2$ |
| $p\to\infty, a\to0$ | Reaches information-theoretic weak recovery | $\alpha_{\text{BBP}}\to p^*/2$ |

### Continuous vs Discontinuous / Finite-N Analysis

| Type | Spectral Density at Left Edge | Behavior of overlap $m$ across threshold | Scaling of deviation from edge at finite $N$ |
|------|--------------|----------------------|------------------------|
| Continuous BBP | $\propto(\lambda-\lambda_-)^{1/2}$ (Steep) | Smooth growth from 0 | $N^{-2/3}$ |
| Discontinuous BBP | $\propto\exp(-A/(\lambda-\lambda_-))$ (Smooth) | Discrete jump from 0 | $1/\log N$ (Strong correction) |

### Key Findings
- **Overparametrization pushes the signal recovery threshold forward**: Data requirements decrease as $p$ increases.
- **Numerical transitions at finite $N$ are significantly lower than theoretical $\alpha_{\text{BBP}}$** for discontinuous cases due to the exponential tail of the smooth edge.
- **A lower bound $\alpha_0$ scales monotonically with $p$**: Even when $N\to\infty$ predictions show slight non-monotonicity, finite-$N$ corrections confirm overparametrization is beneficial in practical dimensions.

## Highlights & Insights
- **Overparametrization as implicit averaging**: A wide student effectively "sees" an averaged landscape, making signal information emerge earlier in the curvature.
- **Discontinuous BBP in ML**: This work provides one of the first practical applications of the "discontinuous BBP transition" concept in a neural network context.
- **Counter-intuitive finite-$N$ corrections**: The finding that $1/\log N$ scaling can dominate $N^{-2/3}$ suggests that $N\to\infty$ thresholds may be overly pessimistic for spectral methods in discontinuous regimes.

## Limitations & Future Work
- **Simple Model**: Analysis is limited to two-layer quadratic networks with Gaussian inputs. While qualitative similarities are expected for other activations, quantitative transferability to deep networks is unproven.
- **Initialization focus**: The study does not cover the full training dynamics. The fate of gradient descent depends on the interaction between "signal emergence in the Hessian" and "algorithmic transitions," the latter of which remains an open question under overparametrization.
- **Heuristic $\alpha_0$**: The $\alpha_0$ threshold is based on a square-root vanishing conjecture for residual overlap, which is reasonable but not strictly proven.

## Related Work & Insights
- **vs Classical Phase Retrieval (Mondelli & Montanari 2018)**: Classical works use fixed spectral matrices; this work studies the true Hessian, which depends on both labels and current student configurations.
- **vs Biroli et al. 2020**: They identified the spectral matrix as an averaged Hessian; this work extends that to any $p, p^*$ and shows that $p\to\infty$ converges to the optimal spectral method.
- **vs Bayesian Recovery (Maillard et al. 2024)**: They proved the weak recovery threshold $p^*/2$. This work shows that even non-Bayesian-optimal overparametrized students can reach this bound via the curvature of the random initial point.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Connects overparametrization and BBP transitions through a rigorous field-theoretic framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid analytical derivations and simulations, though limited to toy models.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of complex physics concepts, though some derivations are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Provides a quantitative mechanism for why overparametrization aids optimization at the initialization level.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A New Initialization to Control Gradients in Sinusoidal Neural Networks](a_new_initialization_to_control_gradients_in_sinusoidal_neural_networks.md)
- [\[ICLR 2026\] Random Spiking Neural Networks are Stable and Spectrally Simple](random_spiking_neural_networks_are_stable_and_spectrally_simple.md)
- [\[ICLR 2026\] Feature Compression is the Root Cause of Adversarial Fragility in Neural Networks](feature_compression_is_the_root_cause_of_adversarial_fragility_in_neural_network.md)
- [\[ICLR 2026\] Unveiling the Basin-like Loss Landscape in Large Language Models](unveiling_the_basin-like_loss_landscape_in_large_language_models.md)
- [\[ICLR 2026\] The Logical Expressiveness of Topological Neural Networks](the_logical_expressiveness_of_topological_neural_networks.md)

</div>

<!-- RELATED:END -->
