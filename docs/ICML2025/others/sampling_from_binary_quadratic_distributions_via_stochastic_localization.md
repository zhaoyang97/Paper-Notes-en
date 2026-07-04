---
title: >-
  [Paper Note] Sampling from Binary Quadratic Distributions via Stochastic Localization
description: >-
  [ICML2025][Stochastic Localization] This work represents the first application of the Stochastic Localization (SL) framework to sampling from general Binary Quadratic Distributions (BQDs). It proves that after a sufficient number of SL iterations, the posterior distribution almost surely satisfies the Poincaré inequality, thereby guaranteeing polynomial-time mixing for discrete MCMC samplers. Consistent improvements in sampling efficiency are verified on QUBO combinatorial op…
tags:
  - "ICML2025"
  - "Stochastic Localization"
  - "Binary Quadratic Distribution"
  - "Discrete MCMC"
  - "Poincaré Inequality"
  - "Combinatorial Optimization"
  - "Gibbs Sampling"
date: 2026-05-08
content_hash: b16dc29610415da7
---

# Sampling from Binary Quadratic Distributions via Stochastic Localization

**Conference**: ICML2025  
**arXiv**: [2505.19438](https://arxiv.org/abs/2505.19438)  
**Code**: To be confirmed  
**Area**: Others/Sampling  
**Keywords**: Stochastic Localization, Binary Quadratic Distribution, Discrete MCMC, Poincaré Inequality, Combinatorial Optimization, Gibbs Sampling

## TL;DR

This work represents the first application of the Stochastic Localization (SL) framework to sampling from general Binary Quadratic Distributions (BQDs). It proves that after a sufficient number of SL iterations, the posterior distribution almost surely satisfies the Poincaré inequality, thereby guaranteeing polynomial-time mixing for discrete MCMC samplers. Consistent improvements in sampling efficiency are verified on QUBO combinatorial optimization problems.

## Background & Motivation

**Problem Definition**: Sampling from a Binary Quadratic Distribution (BQD):

$$\nu(x) \propto e^{-\frac{\beta}{2}\langle x, Wx \rangle + \langle x, b \rangle}, \quad x \in \{-1, I\}^N$$

where $\beta$ is the inverse temperature parameter. Such distributions arise widely in statistical physics (Ising models, spin systems) and combinatorial optimization (QUBO problems).

**Key Challenge**:
- The complex dependencies in discrete state spaces make sampling extremely difficult.
- Theoretical guarantees of SL in the continuous domain (such as log-concavity duality) do not directly transfer to discrete settings.
- Existing work on discrete SL is limited to specific models (e.g., the SK model) and lacks a general framework.

**Goal**: Can SL facilitate sampling from general BQDs, as it does in continuous settings, by constructing posterior distributions that are easier to sample?

## Method

### Framework: SL + Discrete MCMC

The mechanism is to decompose the difficult direct sampling task into a sequence of simpler posterior distribution samplings, based on the observation process:

$$Y_t^\alpha = \alpha(t) X + \sigma B_t, \quad t \in [0, T_{\text{gen}})$$

where $X \sim \nu$ and $B_t$ is a standard Brownian motion. The corresponding SDE is:

$$dY_t^\alpha = \dot{\alpha}(t) u_t^\alpha(Y_t^\alpha) dt + \sigma dB_t$$

**Algorithm Flow (Algorithm 1)**:
1. Initialize $\tilde{Y}_0 \sim \mathcal{N}(0, t_0 \sigma^2 I)$
2. In each SL iteration $i$:
    - Sample $n$ samples from the posterior $\mathbb{P}(X|\tilde{Y}_i)$ using a discrete MCMC sampler.
    - Estimate the posterior expectation $\tilde{U}_i^\alpha = \frac{1}{n}\sum_{j=1}^n x_j^i$.
    - Update $\tilde{Y}_{i+1} \sim \mathcal{N}(\tilde{Y}_i + w_i \tilde{U}_i^\alpha, \sigma^2 \delta_i \mathbf{I})$.
3. Output $\tilde{Y}_T / \alpha(t_T)$.

### Key Designs: Posterior Distribution Structure

For BQDs, the posterior distribution has an explicit form:

$$q_t^\alpha(x|y) \propto e^{-\frac{\beta}{2}\langle x, Wx \rangle + \langle x, b + \frac{\alpha(t) Y_t}{\sigma^2 t} \rangle}$$

**Key Insight**: Since $x \in \{-1,1\}^N$, the quadratic term $\langle x,x \rangle = 1$ in the Gaussian likelihood disappears. The posterior differs from the original distribution only in the linear term (external field $h_t$):

$$h_t = b + \frac{\alpha(t) Y_t}{\sigma^2 t}$$

### Core Theory: Poincaré Inequality

**Theorem 3.1**: As $\frac{\alpha(t)}{\sigma\sqrt{t}} \to +\infty$, the external field $|h_t| \to \infty$, meaning the posterior distribution approaches a product of independent Bernoulli distributions—which is extremely easy to sample.

**Condition 4.1 (Strong External Field Condition)**: Requires $|h| \geq 2\beta \sup_{i \in [N]} \sum_{k \neq i} |W_{ik}|$.

Under this condition, Poincaré inequalities are established for various MCMC samplers:

| Sampler Type | Spectral Gap $\gamma_{\text{gap}}$ |
|---|---|
| Glauber Dynamics | $1 - \frac{|h|}{e^{3|h|/4} + e^{-3|h|/4}}$ |
| Classical Metropolis Chain | $1 - 2|h|e^{-|h|}$ |
| Single-site Gradient MH | $1 - \frac{|h|}{(e^{|h|/4} + e^{-|h|/4})^2}$ |
| DULA (All-site Update) | $1 - \frac{4|h|N}{(e^{|h|/4} + e^{-|h|/4})^2}$ |

In all cases, $\gamma_{\text{gap}}$ approaches 1 as $|h|$ increases, guaranteeing polynomial-time mixing.

**Corollary 4.4 (Chernoff Error Bound)**: The error of the Monte Carlo estimator decays exponentially with the sample size $n$:

$$P_q\left[\left|\frac{1}{n}\sum_{i=1}^n X_i - \mathbb{E}_{\nu_{\beta,h}}[X]\right| \geq \varepsilon\right] \leq C \cdot e^{-\frac{n\varepsilon^2 \gamma_{\text{gap}}}{c}}$$

## Key Experimental Results

On three types of QUBO combinatorial optimization problems (MIS, MaxCut, MaxClique), the original and SL-enhanced versions were tested using three MCMC samplers (GWG, PAS, DMALA) across a total of 14 datasets.

| Task | Sampler | Original | Ours | Gain |
|---|---|---|---|---|
| MIS ER-0.05 | PAS | 104.375 | **104.531** | +0.15% |
| MIS ER-0.25 | GWG | 27.813 | **28.000** | +0.67% |
| MaxClique RB | PAS | 87.544% | **87.649%** | +0.12% |
| MaxCut BA-512 | PAS | 100.883% | **100.928%** | +0.045% |
| MaxCut ER-1024 | GWG | 100.098% | **100.101%** | +0.003% |

**Key Findings**:
- The SL variants achieve consistent improvements or competitive performance across all 14 datasets and all 3 samplers.
- SL-PAS overall performs the best.
- Ablation studies show that the "exponentially decaying MCMC step allocation + uniform time discretization" strategy is optimal, consistent with theoretical predictions.

## Highlights & Insights

1. **Theoretical Breakthrough**: This work is the first to establish rigorous Poincaré inequality guarantees for the SL framework on general BQDs, without relying on specific distribution structures.
2. **Elegant Physical Intuition**: The external field $|h_t| \to \infty$ renders the interaction terms negligible, causing the posterior to degenerate into independent Bernoulli distributions—analagous to how a strong external field suppresses spin correlations in physics.
3. **Broad Applicability**: The theory covers two major classes of discrete MCMC methods: Glauber dynamics and MH, including variants enhanced by gradient information.
4. **Plug-and-Play**: SL serves as a general wrapper that can enhance any existing discrete MCMC sampler without modifying its internal architecture.
5. **Theory-Experiment Alignment**: The superiority of the exponentially decaying step allocation strategy perfectly validates the theoretical prediction that "sampling is easier in later stages."

## Limitations & Future Work

1. **Limited to Binary Variables**: The current framework is restricted to $x \in \{-1,1\}^N$ and has not been generalized to multi-valued discrete variables.
2. **Limited to Quadratic Distributions**: It cannot handle distributions of unknown forms (e.g., deep energy models) without efficient second-order Taylor approximations.
3. **Lack of Overall Convergence Theory**: It only guarantees the mixing of the posterior sampling at each step, leaving the overall convergence rate theory of SL unestablished.
4. **Limited Empirical Gain**: Since baseline samplers are already close to optimal, the improvements from SL are typically <1%, requiring a trade-off with additional computational costs in practice.
5. **Practical Reachability of the Strong External Field Condition**: When Condition 4.1 is precisely satisfied in practice remains dependent on the problem structure.

## Related Work & Insights

- **SLIPS** (Grenioux et al., 2024): An SL sampling framework in the continuous domain, which this work extends theoretically to the discrete domain.
- **GWG/PAS/DMALA**: Three representative discrete MCMC samplers enhanced by gradient information (Grathwohl 2021, Sun 2021, Zhang 2022).
- **SK model SL** (El Alaoui et al., 2022): Application of SL to specific models; this work generalizes it to general BQDs.
- **DISCS** (Goshvadi et al., 2024): A benchmarking framework for discrete sampling.

**Related Work & Insights**: The core idea of SL, "decomposing difficult sampling into simple subproblems," aligns with the central concept of diffusion models. Can similar theoretical tools be used to analyze the convergence of discrete diffusion models?

## Rating

- Novelty: ⭐⭐⭐⭐ — First to establish complete theoretical guarantees for SL on general BQDs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 14 datasets, 3 samplers, comprehensive ablations, though the improvement margins are small.
- Writing Quality: ⭐⭐⭐⭐ — The logical progression from physical intuition to rigorous theory to experimental validation is clear and complete.
- Value: ⭐⭐⭐⭐ — Significant theoretical contribution, though practical gains are limited and framework scalability is yet to be fully validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DiLQR: Differentiable Iterative Linear Quadratic Regulator via Implicit Differentiation](dilqr_differentiable_iterative_linear_quadratic_regulator_via_implicit_different.md)
- [\[ICML 2026\] Networked Information Aggregation for Binary Classification](../../ICML2026/others/networked_information_aggregation_for_binary_classification.md)
- [\[CVPR 2025\] Scene-Agnostic Pose Regression for Visual Localization](../../CVPR2025/others/scene-agnostic_pose_regression_for_visual_localization.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/others/estimation_of_stochastic_optimal_transport_maps.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](../../NeurIPS2025/others/coresets_for_clustering_under_stochastic_noise.md)

</div>

<!-- RELATED:END -->
