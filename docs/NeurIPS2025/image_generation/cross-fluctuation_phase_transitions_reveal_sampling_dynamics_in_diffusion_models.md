---
title: >-
  [Paper Note] Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Diffusion models] Drawing on fluctuation theory from statistical physics, this work proposes a framework for detecting discrete phase transitions in the sampling process of diffusion mode…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion models"
  - "phase transitions"
  - "cross-fluctuation"
  - "sampling dynamics"
  - "conditional generation"
date: 2026-05-08
content_hash: 4102443bc13e6002
---

# Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.00124](https://arxiv.org/abs/2511.00124)  
**Code**: None  
**Area**: Image Generation
**Keywords**: Diffusion models, phase transitions, cross-fluctuation, sampling dynamics, conditional generation
**arXiv**: [2511.00124](https://arxiv.org/abs/2511.00124)  
**Code**: None  
**Area**: Image Generation

## TL;DR

Drawing on fluctuation theory from statistical physics, this work proposes a framework for detecting discrete phase transitions in the sampling process of diffusion models via **cross-fluctuations**, enabling accelerated sampling, improved conditional generation, zero-shot classification, and style transfer—all without retraining.

## Background & Motivation

**Background**: Diffusion models have become foundational to generative systems, demonstrating strong performance in image synthesis, 3D scene generation, audio, and molecular structure generation. However, their sampling process remains a black box—each step mixes thousands of values in ways that are difficult to predict.

**Limitations of Prior Work**: Existing methods lack a principled understanding of when "successful" and "failed" sampling trajectories diverge, and hyperparameter tuning (e.g., the time window for conditional guidance) typically relies on expensive grid search.

**Key Challenge**: The internal dynamics of diffusion model sampling lack interpretability tools—it is unclear at which timestep the generation paths of different classes or events become statistically distinguishable.

**Goal**: To provide a systematic framework for detecting statistical distinguishability transition points (phase transitions) between different "events" (e.g., different classes) during the diffusion process, and to leverage these transition points to directly optimize sampling.

**Key Insight**: Fluctuation theory from statistical physics is introduced into the analysis of diffusion models, treating sampling dynamics as a phase transition from Gaussian noise to the target distribution.

**Core Idea**: Different events undergo discrete merging/splitting phase transitions along diffusion trajectories at discontinuities of $n$-th order cross-fluctuations; detecting these phase transitions directly informs sampling strategies.

## Method

### Overall Architecture

The user's objective is defined as a "desirable event." The forward diffusion process is used to track how the statistical properties of different events converge to a Gaussian distribution. **Algorithm 1** systematically detects discrete phase transitions in cross-fluctuations, identifying the critical timestep $i^\star$ at which events "merge."

### Key Design 1: Cross-fluctuation Statistics

- **Function**: Quantifies the statistical similarity between two events $\Omega_1, \Omega_2$ at different timesteps of the diffusion process.
- **Mechanism**: For a state variable $\rho$, the $n$-th order fluctuation tensor is defined as $\mathcal{F}_\rho^{(n)}(\omega) = \bigotimes_{k=1}^n (\rho(\omega) - \mathbb{E}[\rho])$. The normalized cosine similarity between the conditional expectation tensors of the two events is then computed:
$$\mathcal{M}_\rho^{(n)}(\Omega_1, \Omega_2) = \frac{|\langle \mathbb{E}_1[\mathcal{F}_\rho^{(n)}], \mathbb{E}_2[\mathcal{F}_\rho^{(n)}] \rangle|}{\|\mathbb{E}_1[\mathcal{F}_\rho^{(n)}]\| \cdot \|\mathbb{E}_2[\mathcal{F}_\rho^{(n)}]\|}$$
- **Design Motivation**: $\mathcal{M} \approx 1$ indicates that events have "merged" (are indistinguishable), while $\mathcal{M} \ll 1$ indicates they are distinguishable. For $n=2$, this is equivalent to **Centered Kernel Alignment (CKA)** between the conditional covariance matrices of the two events.

### Key Design 2: Discrete Phase Transition Detection and Thresholding

- **Function**: Converts the continuous cross-fluctuation curve into a discrete "merged / not merged" determination.
- **Mechanism**: A thresholded correction operator is introduced:
$$\widetilde{\mathcal{M}}_\rho^{(n)}(i) = \begin{cases} \mathcal{M}_\rho^{(n)}(\Omega_{1,i}, \Omega_{2,i}), & d(\widehat{F}_\rho^{(2n)}(\Omega_{1,i}), \widehat{F}_\rho^{(2n)}(\Omega_{2,i})) > \varepsilon \\ 1, & \text{otherwise} \end{cases}$$
  where $\varepsilon \approx \max_k \lambda_k^{\max}(0) / 400$, using the maximum absolute eigenvalue difference as the metric.
- **Critical Timestep**: $i^\star = \min\{i : \widetilde{\mathcal{M}}_\rho^{(n)}(i) = 1\}$, which generalizes the notion of coupling time in Markov chains.

### Key Design 3: Five Application Scenarios

1. **Accelerated Sampling**: Reverse sampling is initiated from $t = i^\star$ rather than $t = n$ (using the D'Agostino–Pearson normality test to determine the convergence point).
2. **Class-conditional Generation**: The class merging time $t_{\text{end}}$ and convergence time $t_{\text{start}} = i^\star$ are used to automatically determine the guidance window for Interval Guidance.
3. **Rare Class Generation**: Combines a merge-aware guidance window with an ILVR strategy using noisy reference samples.
4. **Zero-shot Classification**: The score integration interval is truncated at the merging time, with inverse-SNR weighting applied.
5. **Zero-shot Style Transfer**: It is shown that the fluctuation trajectories of the source distribution and target style distribution agree to within $O(\delta)$ accuracy under Fourier regularity conditions, allowing direct reuse of the source distribution's merging time.

### Loss & Training

This paper requires no training of new models and constitutes a purely analytical framework. All cross-fluctuation terms can be estimated without bias via forward Monte Carlo sweeps.

## Key Experimental Results

### Main Results: Accelerated Sampling

| Model / Dataset | FID (↓) | Steps (↓) | GFLOPs (↓) |
|---|---|---|---|
| DiT-XL/2 (ImageNet, full) | 3.42±0.21 | 250 | 4100 |
| DiT-XL/2 (ImageNet, **Ours**) | **3.37±0.31** | 175 | 2870 |
| DDPM (MNIST, full) | 2.27±0.19 | 1000 | 2000 |
| DDPM (MNIST, **Ours**) | 2.29±0.17 | 600 | 1200 |
| DDPM (CIFAR-10, full) | 3.62±0.35 | 500 | 6000 |
| DDPM (CIFAR-10, **Ours**) | **3.47±0.34** | 300 | 3600 |

**Key Findings**: The proposed method reduces sampling steps by 30–40% while maintaining or slightly improving FID.

### Main Results: Class-conditional Generation (IG)

| Model | FID (↓) | Precision (↑) | Recall (↑) | Density (↑) | Coverage (↑) |
|---|---|---|---|---|---|
| DiT-XL/2 (ImageNet, IG Baseline) | 3.22±0.16 | 0.78 | 0.23 | 0.83 | 0.35 |
| DiT-XL/2 (ImageNet, **IG Ours**) | **2.86±0.15** | **0.83** | **0.26** | **0.85** | **0.39** |
| DDPM (CIFAR10, IG Baseline) | 3.32±0.25 | 0.77 | 0.19 | 0.81 | 0.32 |
| DDPM (CIFAR10, **IG Ours**) | **3.01±0.14** | **0.79** | **0.22** | **0.84** | **0.35** |

### Main Results: Zero-shot Classification

| Method | ImageNet (↑) | CIFAR-10 (↑) | Oxford Pets (↑) |
|---|---|---|---|
| SD, uniform (Li et al.) | 54.96 | 84.67 | 82.87 |
| SD, trunc. inverse-SNR (**Ours**) | **65.28** | **88.38** | **89.15** |
| CLIP RN-50 | 58.41 | 75.42 | 85.61 |

### Ablation Study

- **Merger cascade visualization**: Different classes merge in a tree-structured hierarchy at different timesteps, forming a "merger cascade."
- Truncated inverse-SNR weighting outperforms uniform weighting and pure inverse-SNR, validating that timesteps prior to the merging time are most discriminative.
- The fluctuation adaptation lemma for style transfer shows that, under Fourier-domain distance constraints, the fourth-moment discrepancy satisfies $\leq C_n \delta$.

### Key Findings

- Fluctuation-driven merging times generalize from majority classes to long-tail classes without additional hyperparameter tuning.
- A single forward Monte Carlo sweep suffices to obtain all necessary diagnostic information.
- This perspective unifies classical coupling/mixing results for finite Markov chains with continuous SDE dynamics.

## Highlights & Insights

1. **Theoretical Elegance**: Fluctuation theory from statistical physics is seamlessly bridged with diffusion model sampling dynamics, with CKA providing an intuitive practical connection.
2. **One Framework, Multiple Applications**: The same phase transition detection algorithm serves five distinct tasks: accelerated sampling, conditional generation, rare class generation, classification, and style transfer.
3. **Zero-cost Improvement**: All improvements require no model retraining—only a single forward-pass analysis.
4. **Strong Interpretability**: The merger cascade provides an intuitive visualization of how structure forms during the diffusion process.

## Limitations & Future Work

1. **VP Schedule Assumption**: The current analysis is restricted to variance-preserving SDEs and has not been extended to non-VP schedules such as EDM.
2. **Isotropy Restriction**: The forward SDE is assumed to have isotropic noise; anisotropic diffusion has not been addressed.
3. **Cost of Higher-order Fluctuations**: Computing high-order fluctuations for vector-valued states is computationally expensive; in practice, only $n=2$ (CKA) is used.
4. **Threshold Selection**: Although a heuristic rule for choosing $\varepsilon$ is provided, an adaptive mechanism is lacking.
5. **Cross-modal Extension**: Effectiveness on other generative modalities such as audio and 3D geometry has not been validated.

## Related Work & Insights

- **Relation to Interval Guidance (Kynkäänniemi et al., 2024)**: Interval Guidance requires expensive grid search to determine the guidance interval; the proposed method directly provides $t_{\text{start}}$ and $t_{\text{end}}$ via fluctuation analysis.
- **Connection to Brownian Motion Equilibration Time**: Empirical findings suggest that the equilibration time theorem provides a good prediction of $i^\star$, though a theoretical explanation is left for future work.
- **Implications for Sampling Acceleration**: No new scheduling strategy needs to be learned; detecting statistical phase transitions in the forward process alone is sufficient to determine when to stop.

## Rating

⭐⭐⭐⭐ (4/5)

An excellent work that combines theoretical depth with practical utility. The statistical physics perspective is novel, multi-task validation is thorough, and no additional training cost is incurred. The primary limitations are the restriction to VP schedules and the limited practical applicability of higher-order fluctuations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PID-controlled Langevin Dynamics for Faster Sampling of Generative Models](pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)
- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](split_gibbs_discrete_diffusion_posterior_sampling.md)

</div>

<!-- RELATED:END -->
