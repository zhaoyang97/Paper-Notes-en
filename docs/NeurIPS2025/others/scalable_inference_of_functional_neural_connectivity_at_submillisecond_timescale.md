---
title: >-
  [Paper Note] Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales
description: >-
  [NeurIPS 2025][Functional connectivity] This paper generalizes the conventional discrete-time Poisson GLM to a continuous-time Poisson point process framework. Two approaches—Monte Carlo sampling and second-order polynomial approximation—are proposed to bypass the intractable integral in the likelihood. Combined with orthogonal generalized Laguerre basis functions, the method achieves minute-scale training on recordings spanning hundreds of neurons and thousands of seconds, enabling synaptic connectivity inference at submillisecond resolution.
tags:
  - NeurIPS 2025
  - Functional connectivity
  - Poisson point process
  - GLM
  - Monte Carlo estimation
  - synaptic coupling filters
date: 2026-05-08
content_hash: 4ad817bb5295e1be
---

# Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales

**Conference**: NeurIPS 2025
**arXiv**: [2510.20966](https://arxiv.org/abs/2510.20966)
**Code**: [Available](https://github.com/macari216/poisson-process-glm.git)
**Area**: Computational Neuroscience
**Keywords**: Functional connectivity, Poisson point process, GLM, Monte Carlo estimation, synaptic coupling filters

## TL;DR
This paper generalizes the conventional discrete-time Poisson GLM to a continuous-time Poisson point process framework. Two approaches—Monte Carlo sampling and second-order polynomial approximation—are proposed to bypass the intractable integral in the likelihood. Combined with orthogonal generalized Laguerre basis functions, the method achieves minute-scale training on recordings spanning hundreds of neurons and thousands of seconds, enabling synaptic connectivity inference at submillisecond resolution.

## Background & Motivation

**Background**: The Poisson generalized linear model (GLM) is a foundational tool for analyzing neural spike train data and inferring functional connectivity. Coupling filters estimated between neurons serve as proxies for synaptic connections.

**Limitations of Prior Work**: Conventional GLMs require discretizing continuous spike times into bin-count data, resulting in massive design matrices $\mathbf{X}$. Synaptic dynamics operate on submillisecond timescales (rapid rise and decay within 1–5 ms), yet commonly used bin sizes of 1–10 ms are far too coarse. Reducing the bin size to 0.1 ms inflates the design matrix to $10^{10}$–$10^{12}$ bits—far beyond memory capacity. Even with batched strategies, the sparsity of neural spikes induces prohibitively large gradient variance across batches, causing unstable optimization and poor model fit.

**Key Challenge**: There is a fundamental tension between temporal resolution and computational feasibility. Resolving individual synaptic connections requires submillisecond precision, yet discrete-time methods become computationally intractable at such resolution.

**Goal**: (1) How can a Poisson GLM be fit without discretization? (2) How can the intractable integral in the continuous-time likelihood be efficiently approximated? (3) How can scalable inference be achieved on large-scale neural recordings?

**Key Insight**: Taking the bin size to zero in the limit transforms the model into a continuous-time Poisson point process. The input shifts from a massive design matrix to a compact spike time sequence, fundamentally resolving the memory bottleneck.

**Core Idea**: Replace the discrete-time GLM with a continuous-time Poisson point process, and handle the intractable integral in the likelihood via MC sampling or polynomial approximation, enabling large-scale neural connectivity inference at submillisecond precision.

## Method

### Overall Architecture
The input consists of multi-neuron spike time sequences $\mathbf{X} = \{(n_s, t_s)\}$ (neuron index and spike time), and the output is a set of inter-neuron coupling filters $\mathbf{w}$. The model is based on a continuous-time Poisson point process. The log-likelihood comprises two terms: the first is the sum of log firing rates evaluated at observed spike times (tractable); the second is the integral of the firing rate over the entire recording duration—the cumulative intensity function (CIF)—which is intractable and must be approximated.

### Key Designs

1. **Monte Carlo (MC) Estimation of the CIF**

   - *Function*: Approximate the CIF integral $\int_0^T \lambda(t)\, dt$ via stratified sampling.
   - *Mechanism*: The interval $[0, T]$ is divided into $M$ equal subintervals, from each of which one sample point $\tau_m$ is drawn uniformly. The integral is approximated as $\frac{T}{M}\sum_{m=1}^M \lambda(\tau_m)$. Samples are redrawn at each gradient update to maintain an unbiased estimate.
   - *Design Motivation*: Unlike discrete batched methods, the MC approach evaluates the spike-time term exactly over all observed spikes rather than a subset. Stratified sampling ensures uniform coverage of the full recording, yielding substantially lower gradient variance—achieving high-quality fits even with $M \ll T/\Delta t$ samples.

2. **Polynomial Approximation (PA) for the Continuous GLM**

   - *Function*: Approximate the nonlinear function in the CIF with a second-order Chebyshev polynomial, reducing the likelihood to a quadratic form in the model parameters.
   - *Mechanism*: Approximating $\exp(x)\Delta$ as $a_2 x^2 + a_1 x + a_0$ yields a CIF of the form $a_2 \mathbf{w}^\top \mathbf{M} \mathbf{w} + a_1 \mathbf{m}^\top \mathbf{w} + Ta_0$, where $\mathbf{M}$ and $\mathbf{m}$ are precomputable sufficient statistics. The resulting log-likelihood is quadratic and admits a closed-form solution.
   - *Design Motivation*: The closed-form solution eliminates iterative optimization, enabling extremely fast computation. The polynomial approximation does introduce error, particularly when firing rates vary substantially. PA can serve as a high-quality warm-start initialization for MC, and a hybrid PA-MC mode combines the advantages of both.

3. **Generalized Laguerre (GL) Polynomial Basis Functions**

   - *Function*: Replace conventional raised cosine (RC) basis functions for parameterizing coupling filters.
   - *Mechanism*: Laguerre polynomials of the form $L_n^{(\alpha)}(ct) \cdot e^{-ct/2} \cdot (ct)^{\alpha/2}$, parameterized by $\alpha$ and scale $c$, are orthogonal under the weight $t^\alpha e^{-t}$ and exhibit a gamma-function-like envelope that naturally matches the fast-rise, slow-decay profile of synaptic conductances.
   - *Design Motivation*: (1) Orthogonality enables efficient filter representation with fewer basis functions (3 GL bases outperform 4 RC bases); (2) integrals of individual and pairwise basis functions have analytic closed forms via the lower incomplete gamma function, eliminating numerical integration; (3) GL bases are defined over the full history window, avoiding the need to specify support boundaries for each cosine bump as required by RC bases.

### Loss & Training
The MC model is optimized via gradient descent with adaptive step sizes (backtracking line search), and convergence is assessed by the gradient step norm $u_t = \|\eta_t \cdot \nabla \mathcal{L}(\theta_t)\|$. The hybrid PA-MC model initializes parameters using the PA closed-form solution and then refines them with MC. All models employ ridge regularization ($\beta = 1000$) to encourage connection sparsity.

## Key Experimental Results

### Main Results

| Method | Training Time (1000 s recording) | Filter MSE (8 neurons) | Training Time (350 neurons) | Filter MSE (350 neurons) |
|---|---|---|---|---|
| Discrete Batched (DB) | ~5 hours | High | Slowest | Highest |
| Discrete PA (PA-d) | Moderate | Moderate | Degrades with scale | Moderate |
| Continuous PA (PA-c) | Fastest | Moderately high | Fastest | Moderate |
| Continuous MC | Fast | Lowest | Fast | Lowest |
| Hybrid PA-MC | Relatively fast | Lowest | Moderate | Lowest |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| GL (3 bases) vs. RC (4 bases) | Lower MSE for GL | Orthogonal bases are more efficient |
| PA-c vs. PA-d | PA-c slightly better | Continuous version avoids discretization error |
| MC vs. Gauss–Lobatto quadrature | MC MSE = 2.54 vs. quadrature = 5.37 | MC is 400× faster and more accurate |
| Cross-validation: GL vs. RC | GL achieves higher log-likelihood at all $J$ | GL superior on real data as well |

### Key Findings
- **Discrete batched methods fail entirely at scale**: Even with the SVRG optimizer, inter-batch gradient variance remains too large; runtime exceeds continuous methods by orders of magnitude without converging to a good solution.
- **The hybrid mode offers the best trade-off**: PA initialization followed by MC refinement combines the speed of PA with the accuracy of MC.
- **Hippocampal validation**: Applied to 106 neurons over a 2.7-hour recording, the hybrid model recovers coupling filters that are highly consistent with cross-correlograms (CCGs). The density of identified excitatory connections agrees with known hippocampal anatomy: CA3→CA3 connectivity (~4%, the highest density, consistent with dense recurrent E–E connectivity); cross-region connection latencies exceed within-region latencies (consistent with axonal conduction delays).

## Highlights & Insights
- **The conceptual shift from discrete to continuous time** is particularly elegant: rather than attempting to accelerate the discrete approach (a path already exhausted), the paper reframes the problem within the point process framework, fundamentally bypassing design matrix inflation. This paradigm of "redefining the problem" is transferable to many domains.
- **The hybrid warm-start strategy** is a highly reusable technique: a fast but approximate method (PA closed-form solution) initializes the parameters, which are then refined by a slower but more accurate method (MC). The combination outperforms either approach alone.
- **The choice of Laguerre basis functions** reflects the value of domain knowledge: their gamma-like envelope naturally matches synaptic temporal dynamics, in contrast to the uncritical adoption of general-purpose basis functions.

## Limitations & Future Work
- The Poisson distribution assumption may be inadequate—the variance structure of neural spike trains may be better captured by alternative distributions such as the negative binomial.
- The method cannot explicitly distinguish monosynaptic connections from correlations arising from shared input, a limitation common to all GLM-based approaches.
- The approximation range for PA must be set manually (a window of 3–7 Hz around the mean firing rate); too wide a range increases approximation error, while too narrow a range constrains filter amplitude.
- Latent variable modeling of slow population dynamics (e.g., GPFA) has not been incorporated, which could help disentangle fast coupling dynamics from slow coordinated activity.

## Related Work & Insights
- **vs. Discrete GLM (Pillow et al., Zoltowski & Pillow)**: This work extends their PA framework to continuous time, eliminating binning error and addressing large-scale scalability.
- **vs. Hawkes processes**: Hawkes processes are restricted to excitatory interactions, whereas the Poisson GLM framework here captures both excitatory and inhibitory connections.
- **vs. Gauss–Lobatto quadrature (Cai et al.)**: Quadrature-based methods require inserting nodes between every pair of spikes, with computational cost scaling linearly with spike count, rendering them infeasible for large-scale data.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Continuous-time GLMs are not entirely new, but the combination of MC/PA estimators with GL bases constitutes a clear technical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on synthetic data (multiple configurations) and real hippocampal recordings, with full comparisons across time, accuracy, and scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The transition from discrete to continuous GLMs is logically well-motivated; mathematical derivations are rigorous yet accessible.
- **Value**: ⭐⭐⭐⭐ Directly applicable to computational neuroscience; open-source code facilitates rapid adoption.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](generalized_linear_mode_connectivity_for_transformers.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)

<!-- RELATED:END -->
