---
title: >-
  [Paper Note] On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering
description: >-
  [ICML 2026][Computational Biology][Marginal path collapse] This paper identifies **Marginal Path Collapse (MPC)** in inference-time steering when combining multiple heterogeneous diffusion/flow models via "ratio-of-densi…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Marginal path collapse"
  - "path existence criterion"
  - "adaptive exponent correction"
  - "Feynman–Kac guidance"
  - "heterogeneous noise schedules"
date: 2026-05-08
content_hash: deab84a46e9a22f6
---

# On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering

**Conference**: ICML 2026  
**arXiv**: [2512.10339](https://arxiv.org/abs/2512.10339)  
**Code**: https://ziseoklee.github.io/projects/ACE/ (Available, Project Page)  
**Area**: Computational Biology  
**Keywords**: Marginal path collapse, path existence criterion, adaptive exponent correction, Feynman–Kac guidance, heterogeneous noise schedules

## TL;DR
This paper identifies **Marginal Path Collapse (MPC)** in inference-time steering when combining multiple heterogeneous diffusion/flow models via "ratio-of-densities"—where the intermediate composite density becomes non-integrable. It proposes a necessary and sufficient **Path Existence Criterion (PEC)** $C(t)>0$ to diagnose this collapse and designs **ACE** (Adaptive Exponent Bump) to dynamically correct paths by adding a bump function to the exponents $\gamma_i(t)$. By extending the Feynman–Kac corrector to time-varying exponents, ACE significantly outperforms constant-exponent baselines like NR and FKC on synthetic Checkerboard, flexible pose scaffold decoration (combining DN/CONF/SBDD experts), and COCO-MIG multi-attribute generation.

## Background & Motivation

**Background**: Inference-time steering for diffusion and flow matching models has become the de facto standard for modifying tasks without retraining. Methods such as classifier-free guidance, product-of-experts, and Bayesian compositions can essentially be formulated as time-dependent ratio-of-densities $p^*_t \propto \prod_i (q^{(i)}_t)^{\gamma_i(t)}$. In homogeneous scenarios where the same model and schedule are used (e.g., CFG), a constant exponent $\gamma$ almost always works effectively.

**Limitations of Prior Work**: Scientific applications—such as drug molecule scaffold decoration—naturally require ensemble-like combinations of heterogeneous experts: a de-novo unconditional prior, a conformer topology expert, and a pocket-conditioned SBDD expert. These are trained on entirely different noise schedules and dimensions, each with its own optimal schedule shape (refinement requires fast decay, while exploration requires slow decay). The authors discover that forcibly combining their ratios with constant exponents can result in a path that **does not exist** at intermediate timestamps—the partition function diverges, and the score field becomes undefined. Despite this, numerical solvers may still produce "normal-looking" samples, leading to silently biased paths.

**Key Challenge**: The superposition of heterogeneous schedules, negative exponents (denominator experts), and high guidance strength $\omega$ causes the variance in the numerator to contract more slowly than in the denominator. This leads $h_t(x)$ to explode at $\infty$ rather than decaying, even when endpoints $h_0, h_1$ are valid densities. This is a failure of **global integrability**, not a local numerical instability, making it undetectable via standard numerical monitoring (e.g., NaNs or energy spikes).

**Goal**: (i) Provide a necessary and sufficient criterion, computable prior to sampling, to diagnose whether a composite intermediate density exists; (ii) Propose a correction scheme to transform invalid ratio paths into valid ones targeting the same endpoints, accompanied by a weighted SDE for accurate sampling.

**Key Insight**: The focus is shifted from "numerical score stability" to the "integrability of $h_t$." Under the setting of a Gaussian prior to a compactly supported target, all experts at time $t$ are Gaussian convolutions. By analyzing the quadratic coefficients of individual coordinates in the ratio, a closed-form criterion is derived that depends solely on $\{\alpha^{(i)}_t, \gamma_i(t)\}$.

**Core Idea**: Define $C_k(t) = \sum_{i: k \in I_i} \gamma_i(t)/(\alpha^{(i)}_t)^2$. The path exists if $C(t) = \min_k C_k(t) > 0$ and collapses if $< 0$. Since $\gamma_i(t)$ is controllable, the authors **adjust only the intermediate values of the exponents while preserving the endpoints** $\gamma_i(0)$ and $\gamma_i(1)$. A bump function raises $C(t)$ above zero, and a Feynman–Kac weighted sampler supporting time-varying exponents is used to execute the corrected path.

## Method

### Overall Architecture
ACE is a three-step "diagnose, correct, and sample" framework applied to existing heterogeneous expert combinations $h_t(x) = \prod_i (\tilde q^{(i)}_t(x))^{\gamma_i(t)}$:

1.  **Diagnosis (PEC)**: Before sampling, $C_k(t)$ is calculated for each coordinate based on noise schedules $\alpha^{(i)}_t$ and exponents $\gamma_i$. Existence is verified by checking if $C_k(t) > 0$ at discrete sampling steps $\{0, t_1, \dots, t_{\text{end}}\}$.
2.  **Correction (Bump on $\gamma$)**: If $C_k(t) \le 0$ for certain coordinates $k$ during a time interval, a bump function $b(t)$ (zero at endpoints, strictly positive in between) is added to a subset of positive-exponent experts covering those coordinates. This transforms $\gamma_i(t)$ into $\tilde\gamma_i(t) = \gamma_i(t) + b(t)$, ensuring the path is valid while the target distribution remains unchanged.
3.  **Sampling (ACE SDE)**: A weighted particle SDE is used to simulate the corrected path $p^*_t \propto \prod (\tilde q^{(i)}_t)^{\tilde\gamma_i(t)}$. The weight dynamics incorporate an additional term $\sum \dot{\tilde\gamma}_i(t) \log \tilde q^{(i)}_t(X_t)$, necessitating auxiliary SDEs to track $\log \tilde q^{(i)}_t(X_t)$; resampling occurs when the effective sample size (ESS) drops below a threshold.

The input is a set of pre-trained heterogeneous experts and a target ratio combination; the output is samples from the corrected path that preserve the target endpoint distribution.

### Key Designs

1.  **Path Existence Criterion (PEC, Theorem 2.1)**:
    -   **Function**: Determines whether a heterogeneous ratio combination is integrable at each sampling step without running the sampler or writing out the analytic form of $h_t$.
    -   **Mechanism**: For stochastic interpolants from Gaussian priors to compact targets, the quadratic coefficient of the log-density of each $\tilde q^{(i)}_t$ as $\|x\|\to\infty$ is determined by $1/(\alpha^{(i)}_t)^2$. The quadratic term of the log-density of the ratio is the summation $C_k(t) = \sum_{i: k \in I_i} \gamma_i(t)/(\alpha^{(i)}_t)^2$. $C_k(t) > 0$ implies the density decays sub-Gaussianly and is integrable; $C_k(t) < 0$ implies explosion and path collapse. As a universal condition, this check consumes nearly zero engineering overhead.
    -   **Design Motivation**: The goal is to move "path existence" from a post-hoc qualitative judgment to an a priori, provable geometric quantity. $C(t)$ is further linked to the "concentration radius" $R_t(\varepsilon) \propto 1/\sqrt{C(t)}$ (Proposition 2.1), explaining why samples diverge even when $C(t)$ is slightly above zero.

2.  **Adaptive Exponent Bump (Theorem 2.2)**:
    -   **Function**: Constructs new time-varying exponents $\tilde\gamma_i(t)$ such that $\tilde\gamma_i(0) = \gamma_i(0)$ and $\tilde\gamma_i(1) = \gamma_i(1)$ (preserving targets) while ensuring $C(t) > 0$ for all $t$.
    -   **Mechanism**: The authors prove that if $C(0) > 0$, one can always add a bump function $b(t) = B_1 \cdot t(1-t) + B_2 \cdot \min(t, \tau(1-t))$ to positive-exponent experts to lift the criterion. Lower bounds for $B_1, B_2$ can be solved based on the most negative point of the original $C(t)$.
    -   **Design Motivation**: Common practice involves reducing the global guidance strength $\omega$ to avoid collapse, which weakens guidance at all times. Bump correction only adds mass during intermediate periods, allowing high guidance without sacrificing endpoint accuracy or target distribution.

3.  **ACE Sampler (Weighted SDE for Time-Varying Exponents, Theorem 2.3)**:
    -   **Function**: Correctly samples from $p^*_t$ under time-varying $\tilde\gamma_i(t)$, generalizing the Feynman–Kac corrector (Skreta et al., 2025) which assumes $\dot{\tilde\gamma}_i = 0$.
    -   **Mechanism**: The particle $X_t$ follows the SDE $dX_t = (v^*_t + \frac{\sigma_t^2}{2} s^*_t) dt + \sigma_t dW_t$. The log-weight $\log w_t$ evolution includes an additional term $\sum \dot{\tilde\gamma}_i(t) \log \tilde q^{(i)}_t(X_t)$ induced by the time-varying bump. Since $\log \tilde q^{(i)}_t$ lacks a closed form, auxiliary SDEs derived via Itô's formula track these values online. Resampling replicates high-weight samples and eliminates outliers.
    -   **Design Motivation**: A valid path requires a mathematically rigorous sampler. Using standard FKC with $\dot\gamma \neq 0$ leads to biased weights and terminal distribution shifts. ACE explicitly accounts for the time-varying exponent term to ensure mathematical correctness.

### Loss & Training
ACE is a **purely inference-time** framework and requires no retraining. Hyperparameters include bump coefficients $B_1, B_2, \tau$ and the ESS threshold. In scaffold decoration experiments, $B_1 = 30$ and $B_2$ is scaled with $\omega$; synthetic experiments utilize a scan of $B_1 \in [0, 50]$.

## Key Experimental Results

### Main Results

**2D Synthetic Checkerboard**: A combination of three experts (1D X constraint, 2D global constraint, and a denominator expert) using disparate schedules like $\cos(\pi t/2)$ and DDPM.

| Method | $W_1$ ↓ (mean±std) | $W_2$ ↓ | MMD (RBF) ↓ |
| :--- | :--- | :--- | :--- |
| NR (No resampling, constant $\gamma$) | 0.89 ± 0.02 | 1.18 ± 0.02 | 0.092 ± 0.004 |
| FKC (Skreta'25, constant $\gamma$) | 1.37 ± 1.09 | 1.59 ± 1.16 | 0.419 ± 0.579 |
| **ACE ($B_1=10, B_2=1.5$)** | **0.20 ± 0.04** | **0.29 ± 0.05** | **0.012 ± 0.003** |

ACE reduces error by ~4× compared to NR and by an order of magnitude compared to FKC, which exhibits high variance under collapse.

**Flexible Pose Scaffold Decoration (CrossDocked2020, DN+CONF+SBDD Ensemble)**:

| Method | $\omega$ | PEC Satisfied? | OSR (Top 25%) ↑ | Vina Avg ↓ | QED ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ACE** | 1.4 | ✓ | **0.75** | **−7.10** | **0.53** |
| FKC | 1.4 | ✗ | 0.40 | −6.24 | 0.44 |
| NR | 1.4 | ✗ | — | −6.32 (collapse at ω=1.1) | 0.42 |
| Ref. Molecules | — | — | — | −6.77 | 0.48 |

ACE maintains PEC satisfaction throughout higher guidance regimes, leading to monotonic docking improvements. NR and FKC collapse at $\omega \ge 1.1$, resulting in performance degradation. ACE is also the first modular ensemble to exceed the optimization success rates of specialized monolithic baselines (Delete, AutoFragDiff).

### Ablation Study

| Configuration | Synthetic $W_1$ ↓ | Description |
| :--- | :--- | :--- |
| **Full ACE ($B_1=10$)** | 0.20 | Full correction + resampling |
| ACE ($B_1=0$) | 0.78 | No bump; only time-varying weights + resampling |
| FKC (Constant $\gamma$ + Resampling) | 1.37 | Divergent weights due to neglect of correction terms |
| NR (Constant $\gamma$, No Resampling) | 0.89 | No correction; off-path samples remain in batch |

### Key Findings
-   **Collapse is common**: Scanning 125 schedule combinations (DDPM, cosine, sigmoid, etc.) reveals that collapse rates for heterogeneous pairs rise from 41% at $\omega=1.0$ to 80% at $\omega=15$.
-   **Transient collapse is fatal**: Even if $C(t) < 0$ for only 10% of the trajectory, the weights in standard FKC diverge, making path correction essential.
-   **Numerical finiteness is deceptive**: A mixed score field may remain numerically finite during collapse, allowing SDE solvers to run while transporting to an unspecified distribution $p'_t \neq p^*_t$.
-   **$C(t)$ as a quality knob**: Proposition 2.1 shows $R_t \propto 1/\sqrt{C(t)}$; as $C(t) \to 0^+$, the distribution radius expands, weight estimation degrades, and sample quality drops. ACE provides a "focusing force" by keeping $C(t)$ positively stable.

## Highlights & Insights
-   **Separating path existence from numerical stability**: This paper clarifies that there is a class of failures where paths do not exist even if scores are numerically finite—a methodology shift for diffusion steering.
-   **PEC as a zero-cost tool**: Since the criterion only depends on $\alpha^{(i)}_t$ and $\gamma_i$, it can be computed as a simple vector sum during the schedule design phase with almost zero overhead.
-   **Bump-on-exponent vs. Reschedule-on-noise**: Instead of trying to align noisy schedules (which often requires retraining), the authors elegantly vary $\gamma_i(t)$ to ensure path validity while preserving optimal individual expert schedules.
-   **Generalizing Feynman–Kac**: The inclusion of the $\sum \dot\gamma_i \log q^{(i)}$ term and its auxiliary SDE tracking provides a rigorous template for any future work utilizing time-varying guidance schedules.

## Limitations & Future Work
-   **Theoretical assumptions**: The proofs currently rely on Gaussian priors and compactly supported targets. Extending this to unbounded targets or non-Gaussian priors requires further derivation.
-   **Heuristic bump selection**: Parameters $B_1, B_2, \tau$ currently require manual tuning or scanning. An automated scheme to select the "most efficient" bump among infinite valid solutions is not yet provided.
-   **Particle resampling cost**: Full ACE requires resampling, which may be computationally expensive for high-resolution image or video generation. ACE-lite (without resampling) provides a practical but less theoretically rigorous alternative.
-   **Automated subset selection**: When multiple experts must be bumped to cover all coordinates, a clear heuristic for selecting the minimal subset of experts is missing.

## Related Work & Insights
-   **vs. Feynman–Kac Correctors (Skreta et al., 2025a)**: ACE generalizes FKC to time-varying exponents and identifies the fundamental requirement of path existence which FKC implicitly assumes.
-   **vs. Classifier-Free Guidance**: CFG is a homogeneous case where $C(t) > 0$ holds naturally. ACE provides a "safety net" for the heterogeneous Extension of CFG.
-   **vs. Monolithic Baselines**: By ensuring path correctness, a modular "plug-and-play" combination of generalists can outperform task-specific models like Delete or AutoFragDiff.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐
- **Experimental Thoroughness**: ⭐⭐⭐⭐
- **Writing Quality**: ⭐⭐⭐⭐⭐
- **Value**: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2026\] Towards A Generative Protein Evolution Machine with DPLM-Evo](towards_a_generative_protein_evolution_machine_with_dplm-evo.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)
- [\[ICML 2026\] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models](temporal_score_rescaling_for_temperature_sampling_in_diffusion_and_flow_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2026\] Towards A Generative Protein Evolution Machine with DPLM-Evo](towards_a_generative_protein_evolution_machine_with_dplm-evo.md)
- [\[ICML 2025\] Steering Protein Language Models](../../ICML2025/computational_biology/steering_protein_language_models.md)
- [\[ICML 2026\] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models](temporal_score_rescaling_for_temperature_sampling_in_diffusion_and_flow_models.md)

</div>

<!-- RELATED:END -->
