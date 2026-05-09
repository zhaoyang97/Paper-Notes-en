---
title: >-
  [Paper Note] Modeling Neural Activity with Conditionally Linear Dynamical Systems
description: >-
  [NeurIPS 2025][Linear Dynamical Systems] This paper proposes Conditionally Linear Dynamical Systems (CLDS), where Gaussian process priors allow the parameters of a linear dynamical system to vary nonlinearly as a function of observed experimental covariates, preserving the interpretability and efficient inference of linear models while capturing the nonlinear dynamics prevalent in neural circuits.
tags:
  - NeurIPS 2025
  - Linear Dynamical Systems
  - Gaussian Processes
  - Neural Activity Modeling
  - Bayesian Inference
  - Ring Attractor
date: 2026-05-08
content_hash: 2d4143b5c94c9082
---

# Modeling Neural Activity with Conditionally Linear Dynamical Systems

**Conference**: NeurIPS 2025
**arXiv**: [2502.18347](https://arxiv.org/abs/2502.18347)
**Code**: [GitHub](https://github.com/neurostatslab/clds)
**Area**: Computational Neuroscience / State Space Models
**Keywords**: Linear Dynamical Systems, Gaussian Processes, Neural Activity Modeling, Bayesian Inference, Ring Attractor

## TL;DR

This paper proposes Conditionally Linear Dynamical Systems (CLDS), where Gaussian process priors allow the parameters of a linear dynamical system to vary nonlinearly as a function of observed experimental covariates, preserving the interpretability and efficient inference of linear models while capturing the nonlinear dynamics prevalent in neural circuits.

## Background & Motivation

Neural population activity exhibits complex nonlinear dynamics across time, trials, and experimental conditions. Classical linear dynamical systems (LDS) are efficient and interpretable via Kalman filtering and EM algorithms, but their time-invariant linear assumptions prevent them from capturing nonlinear structures widely observed in neural circuits, such as ring attractors.

Recent machine learning approaches (RNNs, Transformers, diffusion models, etc.) have improved predictive accuracy but suffer from several limitations: (1) model fitting is difficult, with heterogeneous inference methods lacking a unified framework; (2) models resist scientific interpretation—analyzing a trained RNN is far more difficult than analyzing a linear system; and (3) performance degrades in data-scarce regimes (e.g., a single trial per condition).

The core insight of CLDS is that in many neuroscience experiments, external covariates (e.g., head direction, movement direction) are known, and the nonlinearity of neural circuits often arises from modulation of dynamical parameters by these covariates. CLDS explicitly models this dependence: **conditioned on the covariates, the dynamics are linear**, yet the parameters can vary nonlinearly and smoothly over the covariate space. This strikes a principled balance between classical and modern nonlinear approaches.

## Method

### Overall Architecture

CLDS defines a family of LDS models parameterized by experimental condition $\bm{u}_t$. Given recordings from $N$ neurons across $K$ trials of length $T$:

$$\mathbf{x}_{t+1} = \mathbf{A}(\bm{u}_t)\mathbf{x}_t + \mathbf{b}(\bm{u}_t) + \epsilon_t$$

$$\mathbf{y}_t = \mathbf{C}(\bm{u}_t)\mathbf{x}_t + \mathbf{d}(\bm{u}_t) + \omega_t$$

where $\mathbf{x}_t \in \mathbb{R}^D$ is the latent state and $\mathbf{y}_t \in \mathbb{R}^N$ is the observed neural activity. The key property is that the system matrices $\{\mathbf{A}, \mathbf{b}, \mathbf{C}, \mathbf{d}, \mathbf{m}\}$ are all functions of the conditioning variable $\bm{u}_t$, with this mapping being nonlinear and learnable.

### Key Designs

1. **Gaussian Process Prior (GP Prior)**: An approximate GP prior is placed on each element of each parameter matrix via a finite basis function expansion $\mathbf{M}_{ij}(\bm{u}) = \sum_{\ell=1}^{L} w_\ell^{(ij)} \phi_\ell(\bm{u})$, with weights $w_\ell^{(ij)} \sim \mathcal{N}(0,1)$. Basis functions are chosen as canonical Fourier features to approximate the squared exponential kernel. **Design Motivation**: The GP prior encodes a smoothness assumption over the covariate space, enabling statistical power sharing across neighboring conditions. The length scale $\kappa$ controls model expressivity—$\kappa \to \infty$ recovers a time-invariant LDS, while $\kappa \to 0$ yields fully independent per-condition models.

2. **Conditionally Linear Regression with Closed-Form MAP Inference**: Using the Kronecker product, the conditioned regression problem is reformulated as Bayesian linear regression in an augmented feature space, $\mathbf{y}_n = \mathbf{W}^\top \mathbf{z}_n + \epsilon_n$, where $\mathbf{z}_n = \phi(\bm{u}_n) \otimes \mathbf{x}_n$. The MAP estimate reduces to solving the Sylvester equation $\mathbf{Z}^\top\mathbf{Z}\mathbf{W} + \mathbf{W}\Sigma = \mathbf{Z}^\top\mathbf{Y}$, which admits an analytic solution. **Design Motivation**: This preserves the closed-form solution of linear regression, avoiding iterative optimization such as stochastic gradient descent.

3. **EM Inference Framework**: The E-step obtains posterior moments of the latent states via Kalman smoothing (computed exactly), and the M-step updates parameters using the closed-form conditional linear regression described above. Both steps are analytic, guaranteeing monotonic increase of the marginal log-likelihood. **Design Motivation**: The conditional linear structure is fully exploited, retaining all inference advantages of the classical LDS.

4. **Composite Dynamics Visualization**: The conditionally dependent linear dynamics are marginalized by taking the expectation over $\bm{u}_t$ conditioned on $\mathbf{x}_t$: $\mathbf{x}_{t+1} = \mathbb{E}_{p(\bm{u}|\mathbf{x}_t)}[\mathbf{A}(\bm{u})\mathbf{x}_t + \mathbf{b}(\bm{u})]$. This "stitches" the condition-dependent linear dynamics into a global nonlinear flow field, facilitating visualization and interpretation.

### Loss & Training

The objective function is the posterior log-probability under MAP estimation. The model is optimized via the EM algorithm, initialized by sampling from the GP prior. Hyperparameters $\{L, \kappa, \sigma\}$ and the latent dimensionality $D$ are selected using an 80/20 trial split for validation. Extensions to non-Gaussian likelihoods (e.g., Poisson) are supported—in such cases the posterior remains log-concave and can be optimized with standard routines.

## Key Experimental Results

### Main Results

| Experimental Setting | Metric | CLDS | LDS | gpSLDS | LFADS |
|---|---|---|---|---|---|
| Synthetic HD ring attractor | Co-smoothing $R^2$ | **0.86** | — | — | — |
| Real mouse HD (ADn) | Tuning curve recovery | Near-perfect | — | — | — |
| Macaque center-out reaching (1 trial/condition) | Co-smoothing $R^2$ | **Highest** | Lowest | Close to CLDS | Lowest |
| Macaque center-out reaching (multi-trial/condition) | Co-smoothing $R^2$ | **Highest** | — | Close to CLDS | Improves but remains lower |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Fixed $\mathbf{C}$ vs. learned $\mathbf{C}$ | Parameter recovery quality | Fixing $\mathbf{C}$ avoids non-identifiability; learning $\mathbf{C}$ still yields good recovery |
| Varying SNR | Parameter recovery $R^2$ | Recovery remains reliable under reduced SNR |
| Varying number of training trials | Co-smoothing $R^2$ | CLDS advantage is largest at 1 trial/condition (excellent data efficiency) |
| GP length scale $\kappa$ | Model expressivity | $\kappa$ controls a continuous spectrum from time-invariant LDS to fully nonparametric |

### Key Findings

- CLDS substantially outperforms baseline models in extremely low-data regimes (1 trial/condition), demonstrating the advantages of the Bayesian framework and cross-condition statistical power sharing.
- In synthetic ring attractor experiments, CLDS recovers the true dynamics matrices $\mathbf{A}(\theta)$ and biases $\mathbf{b}(\theta)$ with near-perfect accuracy.
- In real mouse HD data, CLDS recovers the ring attractor structure, with empirical tuning curves closely matching model predictions.
- Even when the underlying data are not generated by a CLDS (model mismatch), CLDS still captures the core nonlinear structure.

## Highlights & Insights

- **Elegant Design Philosophy**: Rather than simply stacking nonlinearities, CLDS introduces a principled division of labor between observed covariates and latent dynamics—nonlinearity is assigned to the covariate space, while linearity is preserved in the latent space. This decomposition is both physically meaningful and computationally advantageous.
- **Deep Connection to Wishart Process Models**: CLDS can be viewed as a dynamical extension of the Wishart process, providing a new framework for estimating noise correlations across time.
- **Data Efficiency**: Trial-data scarcity is the norm in neuroscience; CLDS addresses this practical challenge elegantly through Bayesian priors and inter-condition interpolation.

## Limitations & Future Work

- The current implementation supports only Gaussian observation models and has not been extended to more realistic spike-count likelihoods such as Poisson.
- The conditionally linear dynamics assumption may introduce increasing approximation error when the relationship between latent states and covariates is weak (e.g., during prolonged internal inference in cognitive tasks).
- The model depends on observed covariate time series $\bm{u}_{1:T}$, leading to performance degradation when covariates are partially unobserved or when predictions are required.
- The GP prior over parameters is not fully exploited to propagate a complete posterior distribution over parameters.

## Related Work & Insights

- **Comparison with gpSLDS**: CLDS drives dynamical switching via observed covariates rather than discrete latent processes, making inference more tractable at the cost of not being fully unsupervised.
- **Comparison with LFADS**: LFADS is a fully nonlinear alternative that may be more powerful with large datasets, but is outperformed by CLDS in low-data regimes.
- **Broader Implications**: The conditionally linear idea is generalizable to other sequence modeling problems—any time-series data with rich external covariates may benefit from this "locally linear, globally nonlinear" modeling paradigm.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — While the conditionally linear idea has precedents, the complete framework combining GP priors with closed-form EM represents a significant advance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Combines synthetic and real-data experiments, including model mismatch tests and low-data analyses, though validation with non-Gaussian likelihoods is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear exposition, rigorous mathematical derivations, and excellent figure design.
- **Value**: ⭐⭐⭐⭐ — Provides a practical and principled modeling tool for computational neuroscience, especially valuable in data-scarce settings.

## Supplementary Notes

- The core distinction between CLDS and SLDS: in SLDS, "switching" is driven by a discrete latent process (making inference difficult); in CLDS, "switching" is driven by observed covariates (making inference straightforward).
- The macaque reaching experiment uses two-dimensional conditions (angle + delay/movement cue), demonstrating CLDS's ability to handle mixed continuous-discrete conditioning variables.
- Code is publicly available, implemented in JAX, and amenable to GPU acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)
- [\[NeurIPS 2025\] Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems](efficient_parametric_svd_of_koopman_operator_for_stochastic_dynamical_systems.md)
- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](infrequent_exploration_in_linear_bandits.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](generalized_linear_mode_connectivity_for_transformers.md)

</div>

<!-- RELATED:END -->
