---
title: >-
  [Paper Note] Learning Single-Index Models via Harmonic Decomposition
description: >-
  [NeurIPS 2025][Optimization][single-index models] This paper proposes spherical harmonics as a natural basis for single-index models (SIMs) in place of Hermite polynomials, leveraging rotational symmetry to characterize sample and computational complexity for learning SIMs under arbitrary spherically symmetric input distributions. Two families of optimal estimators are constructed (tensor unfolding + online SGD), and a sample–runtime tradeoff is revealed that does not arise in the Gaussian setting.
tags:
  - NeurIPS 2025
  - Optimization
  - single-index models
  - spherical harmonics
  - harmonic decomposition
  - sample complexity
  - computational complexity
  - tensor unfolding
  - online SGD
date: 2026-05-08
content_hash: a4b7c0b89e489ace
---

# Learning Single-Index Models via Harmonic Decomposition

**Conference**: NeurIPS 2025
**arXiv**: [2506.09887](https://arxiv.org/abs/2506.09887)
**Authors**: Nirmit Joshi (TTIC), Hugo Koubbi (Yale / ENS Paris-Saclay), Theodor Misiakiewicz (Yale), Nathan Srebro (TTIC)
**Code**: To be confirmed
**Area**: Optimization
**Keywords**: single-index models, spherical harmonics, harmonic decomposition, sample complexity, computational complexity, tensor unfolding, online SGD

## TL;DR

This paper proposes spherical harmonics as a natural basis for single-index models (SIMs) in place of Hermite polynomials, leveraging rotational symmetry to characterize sample and computational complexity for learning SIMs under arbitrary spherically symmetric input distributions. Two families of optimal estimators are constructed (tensor unfolding + online SGD), and a sample–runtime tradeoff is revealed that does not arise in the Gaussian setting.

## Background & Motivation

**Single-index models (SIMs)** are a fundamental class of models in statistics and machine learning: the label $y$ depends on the input $\bm{x}$ only through a one-dimensional projection $\langle \bm{w}_*, \bm{x} \rangle$. In recent years, SIMs have served as a canonical prototype for studying statistical-computational gaps, non-convex optimization, and feature learning in high dimensions.

**Prior results for Gaussian SIMs**: Damian et al. introduced the generative exponent $\mathsf{k}_\star$ via Hermite expansion, establishing optimal sample complexity $\mathsf{m} = \Theta_d(d^{\mathsf{k}_\star/2})$ and runtime $\mathsf{T} = \tilde{\Theta}_d(d^{\mathsf{k}_\star/2+1})$. Several algorithms have progressively closed the gap: vanilla SGD achieves the suboptimal $d^{\mathsf{k}_\star}$, while smoothing SGD and partial trace attain optimality.

**Remaining conceptual questions**:
- Why does vanilla SGD achieve runtime $d^{\mathsf{k}_\star}$ rather than the optimal $d^{\mathsf{k}_\star/2+1}$?
- Why do landscape smoothing and partial trace, borrowed from tensor PCA, achieve optimality?
- What role does the Gaussian assumption actually play in these results?

**Core insight**: The complexity of SIM learning is governed by the **rotational symmetry** of the problem, not by Gaussianity. Spherical harmonics—as irreducible representations of the orthogonal group $\mathcal{O}_d$—constitute the natural basis for this problem. This viewpoint not only clarifies the questions above but also extends the theory to arbitrary spherically symmetric input distributions.

## Method

### Spherical Single-Index Models

The input $\bm{x}$ is decomposed in polar coordinates as $\bm{x} = r\bm{z}$, where $r = \|\bm{x}\|_2$ is the radius and $\bm{z} = \bm{x}/\|\bm{x}\|_2$ is the direction. The model is:

$$(y, \bm{x}) \sim \mathbb{P}_{\bm{w}_*}: \quad \bm{x} = (r, \bm{z}) \sim \mu_r \otimes \tau_d, \quad y|(r, \bm{z}) \sim \nu_d(\cdot | r, \langle \bm{w}_*, \bm{z} \rangle)$$

The label may depend on both the radius $r$ and the projection $\langle \bm{w}_*, \bm{z} \rangle$, making this a strictly more general setting than Gaussian SIMs.

### Harmonic Decomposition and Complexity Characterization

$L^2(\mathbb{S}^{d-1})$ is decomposed into harmonic subspaces $\bigoplus_{\ell=0}^{\infty} V_{d,\ell}$, with $\dim(V_{d,\ell}) = \Theta_d(d^\ell)$. The $\ell$-th order Gegenbauer coefficient is defined as:

$$\xi_{d,\ell}(Y,R) := \mathbb{E}_{\nu_d}[Q_\ell(Z) | Y, R]$$

**Lower bounds** (Theorem 1), established via the LDP and SQ frameworks:

$$\mathsf{m} \gtrsim \inf_{\ell \geq 1} \frac{d^{\ell/2}}{\|\xi_{d,\ell}\|_{L^2}^2}, \qquad \mathsf{T} \gtrsim \inf_{\ell \geq 1} \frac{d^{\ell}}{\|\xi_{d,\ell}\|_{L^2}^2}$$

A key structural property is that the lower bounds **decouple** across irreducible subspaces: each subspace $V_{d,\ell}$ independently contributes a term reflecting a competition between $\dim(V_{d,\ell})$ and the signal strength $\|\xi_{d,\ell}\|_{L^2}^2$.

### Three Families of Optimal Estimators

**1. Spectral algorithms ($\ell = 1, 2$)**:
- $\ell = 1$: Construct the empirical mean vector $\hat{\bm{v}} = \frac{1}{\mathsf{m}} \sum_i \mathcal{T}_1(y_i, r_i) \sqrt{d}\, \bm{z}_i$ and normalize to obtain the estimate.
- $\ell = 2$: Construct the empirical matrix $\hat{\bm{M}} = \frac{1}{\mathsf{m}} \sum_i \mathcal{T}_2(y_i, r_i) [d \cdot \bm{z}_i \bm{z}_i^\top - \bm{I}_d]$ and extract the leading eigenvector.
- Both achieve optimal sample and runtime complexity simultaneously.

**2. Online SGD ($\ell \geq 3$, runtime-optimal)**:
- Projected online SGD is applied to the population loss $\min_{\bm{w}} \mathbb{E}[(\mathcal{T}_\ell(y,r) - Q_\ell(\langle \bm{w}, \bm{z} \rangle))^2]$.
- Sample complexity $\mathsf{m} \asymp d^{\ell-1} / \|\xi_{d,\ell}\|_{L^2}^2$; runtime $\mathsf{T} \asymp d^\ell / \|\xi_{d,\ell}\|_{L^2}^2$ (runtime-optimal).

**3. Harmonic tensor unfolding ($\ell \geq 3$, sample-optimal)**:
- A harmonic tensor $\hat{\bm{T}} = \frac{1}{\mathsf{m}} \sum_i \mathcal{T}_\ell(y_i, r_i) \mathcal{H}_\ell(\bm{z}_i)$ is constructed; by a reproducing property, its expectation is proportional to $\mathcal{H}_\ell(\bm{w}_*)$.
- The tensor is matricized (unfolded) and the leading component is extracted via power iteration.
- For even $\ell$: balanced unfolding suffices; for odd $\ell$: a diagonal correction removes a $\sqrt{d}$ factor loss arising from symmetric noise.
- Sample complexity $\mathsf{m} \asymp d^{\ell/2} / \|\xi_{d,\ell}\|_{L^2}^2$ (sample-optimal); runtime $\mathsf{T} \asymp d^{\ell+1} / \|\xi_{d,\ell}\|_{L^2}^2 \cdot \log(d)$.

### Specialization to Gaussian SIMs

Via Hermite–Gegenbauer decomposition, it is shown that $\|\xi_{d,\ell}\|_{L^2}^2 \asymp d^{-(\mathsf{k}_\star - \ell)/2}$ (when $\ell \equiv \mathsf{k}_\star \pmod{2}$). Substituting into the lower bounds yields:

- The optimal degree $\mathsf{l}_\star$ for both sample and runtime complexity is always 1 or 2 (depending on the parity of $\mathsf{k}_\star$).
- Spectral algorithms simultaneously achieve both optima: $\mathsf{m} = \Theta_d(d^{\mathsf{k}_\star/2})$, $\mathsf{T} = \Theta_d(d^{\mathsf{k}_\star/2+1} \log d)$.

### Reinterpretation of Prior Algorithms

- **Vanilla SGD** is dominated by the high-frequency harmonic component $V_{d,\mathsf{k}_\star}$ and does not exploit the radial information $\|\bm{x}\|_2$—it is, however, optimal within the class of algorithms that use only directional information.
- **Landscape smoothing** is essentially a low-pass filter that projects the dynamics onto the low-order harmonic components $V_{d,1}$ or $V_{d,2}$.
- **Partial trace** is essentially a projection of the high-order Hermite tensor onto the low-order spherical harmonic subspace.

## Key Experimental Results

This is a purely theoretical paper; all core results are presented as complexity bounds.

### Table 1: Algorithmic Complexity per Subspace for Spherical SIMs

| Subspace $V_{d,\ell}$ | Sample-optimal estimator | Runtime-optimal estimator |
|---|---|---|
| $\ell = 1$ | Spectral: $\mathsf{m} \asymp d \vee \frac{d^{1/2}}{\|\xi_{d,1}\|^2}$ | $\mathsf{T} \asymp d^2 \vee \frac{d^{3/2}}{\|\xi_{d,1}\|^2}$ |
| $\ell = 2$ | Spectral: $\mathsf{m} \asymp \frac{d}{\|\xi_{d,2}\|^2}$ | $\mathsf{T} \asymp \frac{d^2}{\|\xi_{d,2}\|^2} \log d$ |
| $\ell \geq 3$ (tensor unfolding) | $\mathsf{m} \asymp \frac{d^{\ell/2}}{\|\xi_{d,\ell}\|^2}$ | $\mathsf{T} \asymp \frac{d^{\ell+1}}{\|\xi_{d,\ell}\|^2} \log d$ |
| $\ell \geq 3$ (online SGD) | $\mathsf{m} \asymp \frac{d^{\ell-1}}{\|\xi_{d,\ell}\|^2}$ | $\mathsf{T} \asymp \frac{d^{\ell}}{\|\xi_{d,\ell}\|^2}$ |

### Table 2: Algorithm Comparison for Gaussian SIMs ($\mathsf{k}_\star > 1$)

| Algorithm | Uses $\|\bm{x}\|_2$ | Sample $\mathsf{m}$ | Runtime $\mathsf{T}$ | Optimality |
|---|---|---|---|---|
| Spectral ($\ell=1$ or $2$) | Yes | $d^{\mathsf{k}_\star/2}$ | $d^{\mathsf{k}_\star/2+1} \log d$ | Sample + runtime optimal |
| Harmonic tensor unfolding ($\ell=\mathsf{k}_\star$) | No | $d^{\mathsf{k}_\star/2}$ | $d^{\mathsf{k}_\star+1} \log d$ | Sample-optimal |
| Online SGD ($\ell=\mathsf{k}_\star$) | No | $d^{\mathsf{k}_\star-1}$ | $d^{\mathsf{k}_\star}$ | Runtime-optimal (direction only) |
| Vanilla SGD [AGJ21] | No | $\tilde{O}(d^{\mathsf{k}_\star-1})$ | $\tilde{O}(d^{\mathsf{k}_\star})$ | Suboptimal |
| Smoothed SGD [DNGL24] | Yes | $\tilde{O}(d^{\mathsf{k}_\star/2})$ | $\tilde{O}(d^{\mathsf{k}_\star/2+1})$ | Near-optimal |
| Partial Trace [DPVLB24] | Yes | $d^{\mathsf{k}_\star/2}$ | $d^{\mathsf{k}_\star/2+1} \log d$ | Optimal |

**Key Findings**:
- Algorithms that do not use radial information have a runtime lower bound of $\Omega_d(d^{\mathsf{k}_\star})$, even when the radius itself carries no information about $\bm{w}_*$.
- Normalizing the input ($\bm{x} \to \bm{x}/\|\bm{x}\|_2$) degrades runtime from $d^{\mathsf{k}_\star/2+1}$ to $d^{\mathsf{k}_\star}$.
- For general spherical SIMs, one can construct distributions where $\mathsf{l}_{\mathsf{m},\star} \gg \mathsf{l}_{\mathsf{T},\star}$, in which case no single algorithm can simultaneously achieve both optima.

## Highlights & Insights

- **Unified perspective**: Spherical harmonics provide a more natural basis than Hermite polynomials, elevating rotational symmetry to a core principle rather than a technical convenience, and unifying the explanations for vanilla SGD's suboptimality, as well as the success of smoothing and partial trace.
- **Beyond the Gaussian setting**: This is the first work to characterize optimal complexity for SIMs under arbitrary spherically symmetric distributions, revealing a sample–runtime tradeoff that does not arise in the Gaussian case (where both optima can be achieved simultaneously).
- **Role of radial information**: The impact of using or ignoring the input norm $\|\bm{x}\|_2$ is precisely quantified—even when the radius carries no information about $\bm{w}_*$, ignoring it leads to a quadratic degradation in runtime.
- **Harmonic tensor unfolding**: A new technique for diagonal correction of odd-order tensors is introduced, avoiding the $\sqrt{d}$ factor loss caused by symmetric noise.

## Limitations & Future Work

- **Purely theoretical**: No empirical validation is provided; all results are asymptotic complexity bounds with constants depending on the link function $\rho$ or $\nu_d$.
- **Known link function assumption**: The model $\nu_d$ is assumed known (Remark 2.1 briefly discusses replacing it with a random nonlinearity in the unknown case, but this is not developed).
- **Gap for odd orders**: Whether a single algorithm can simultaneously achieve both sample and runtime optimality when $\ell \geq 3$ is odd remains an open problem.
- **Multi-index models not covered**: Extending the framework to multi-index models via higher-order spherical harmonics is left for future work.
- **Detection vs. recovery gap**: When $\ell = 1$ and the signal is sufficiently strong, the lower bound for the detection problem is weaker than the information-theoretic lower bound $\Omega(d)$ for recovery.

## Related Work & Insights

| Dimension | Ours | Damian et al. (DPVLB24) | Ben Arous et al. (AGJ21) |
|---|---|---|---|
| Basis | Spherical harmonics (Gegenbauer) | Hermite polynomials | Hermite polynomials |
| Input distribution | Arbitrary spherically symmetric | Gaussian only | Gaussian only |
| Core principle | Rotational symmetry + irreducible representations | Tensor PCA analogy | Saddle dynamics of non-convex loss |
| Sample-optimal | Yes (tensor unfolding) | Yes (partial trace) | No ($d^{\mathsf{k}_\star-1}$) |
| Runtime-optimal | Yes (online SGD / spectral) | Yes (partial trace) | No ($d^{\mathsf{k}_\star}$) |
| Sample–runtime tradeoff | First revealed (non-Gaussian) | Absent (does not arise in Gaussian case) | Absent |
| Radial information analysis | Precisely quantified | Used implicitly | Not used |

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Reconstructs the theory of SIM learning through rotational symmetry and spherical harmonics, providing a new unified perspective while revealing phenomena absent in the Gaussian setting.
- Experimental Thoroughness: ⭐⭐⭐ — Purely theoretical; no empirical validation, though the completeness of the theoretical results compensates for this.
- Writing Quality: ⭐⭐⭐⭐⭐ — Exceptionally well-structured, building progressively from intuition to formalization; the reinterpretation of prior methods is particularly insightful.
- Value: ⭐⭐⭐⭐⭐ — Provides a unified framework grounded in group representation theory for statistical-computational gaps in high-dimensional learning, with far-reaching theoretical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[NeurIPS 2025\] Revisiting Orbital Minimization Method for Neural Operator Decomposition](revisiting_orbital_minimization_method_for_neural_operator_decomposition.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)

</div>

<!-- RELATED:END -->
