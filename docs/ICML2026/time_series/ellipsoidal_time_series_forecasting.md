---
title: >-
  [Paper Note] Ellipsoidal Time Series Forecasting
description: >-
  [ICML 2026][Time Series][Long-term Forecasting] Fern reformulates long-term time series forecasting as "Optimal Transport from a fixed Gaussian source to a data-dependent ellipsoid." By leveraging Brenier's Theorem…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Long-term Forecasting"
  - "SPD Jacobian"
  - "Brenier's Theorem"
  - "Ellipsoidal Transport"
  - "Non-stationary Robustness"
date: 2026-05-08
content_hash: b20d3e4777d55528
---

# Ellipsoidal Time Series Forecasting

**Conference**: ICML 2026  
**arXiv**: [2505.17370](https://arxiv.org/abs/2505.17370)  
**Code**: None  
**Area**: Time Series Forecasting / Optimal Transport / Dynamical Systems  
**Keywords**: Long-term Forecasting, SPD Jacobian, Brenier's Theorem, Ellipsoidal Transport, Non-stationary Robustness

## TL;DR
Fern reformulates long-term time series forecasting as "Optimal Transport from a fixed Gaussian source to a data-dependent ellipsoid." By leveraging Brenier's Theorem, the search space is restricted to the SPD (Symmetric Positive Definite) Jacobian class. Utilizing a low-rank spectral decomposition via Householder reflections, the computational cost is reduced from $O(n^3)$ to $O(Rn)$. This approach achieves up to a 790× stability gain relative to baselines like DLinear and Koopa in non-stationary shock scenarios.

## Background & Motivation

**Background**: The Long-Term Time Series Forecasting (LTSF) community has established strong baselines such as PatchTST, DLinear, Koopa, and iTransformer. Mainstream approaches typically use channel-independent (CI) linear heads to fit conditional means or channel-dependent (CD) Transformers to mix multiple channels. Performance metrics on benchmarks have become highly competitive, with ongoing debates between the CI and CD paradigms.

**Limitations of Prior Work**: The authors point out that existing evaluations mask model fragility in non-stationary scenarios. Most benchmarks consist of mildly drifting power, traffic, or weather data. In the presence of regime shifts, chaotic shocks, or true random noise, strong baselines collapse rapidly, and point metrics like MSE fail to identify local failures. Furthermore, traditional "direct Jacobian modeling" requires $n^2$ components for an $n$-dimensional horizon, and eigen-decomposition takes $O(n^3)$, which is computationally prohibitive.

**Key Challenge**: Effective long-term forecasting requires preserving **local geometric structure** (spectral information indicating the directions of maximum system stretching) while operating within **computational budgets**. Searching for an arbitrary $n\times n$ matrix is both expensive and lacks necessary structure.

**Goal**: (1) Identify a predictor that is both data-dependent and geometrically aware; (2) Incorporate spectral structure as an "intrinsic parameter" rather than a "post-hoc byproduct"; (3) Design evaluation protocols that expose model fragility in non-stationary scenarios.

**Key Insight**: The authors shift the forecasting perspective from "$x \to y$ temporal evolution" to "transport from a fixed Gaussian source $\mathcal{N}(0, I)$ to a target distribution." According to Brenier's Theorem, if the target is restricted to a Gaussian, the optimal transport map is uniquely affine (SPD scaling + translation), naturally constraining the Jacobian to the SPD cone.

**Core Idea**: Instead of learning an implicit non-linear mapping and then extracting its Jacobian, the model directly parameterizes an SPD matrix $A = U^\top \Lambda U$ as the optimal transport map using Householder reflections and diagonal spectra. This allows spectral information (eigenvalues and eigenvectors) to serve as built-in interpretable diagnostic quantities.

## Method

### Overall Architecture
Fern's pipeline is a lightweight model featuring "bidirectional coupled encoding + SPD projection." The input is a univariate time series window $x$ of length $L$, and the output is a future patch prediction of length $n$. The entire long horizon is decoded in parallel through patch segmentation. Intermediate states include a low-dimensional Gaussian latent variable $z \sim \mathcal{N}(\mu(x), \Sigma(x))$ and a fixed noise source $y_0 \sim \mathcal{N}(0, I)$. The noise is transformed into the target ellipsoid via the affine mapping $y^* = U^\top \Lambda U (y_0 + t_y)$. The architecture follows the CI principle: each channel is processed independently, leveraging Takens' Embedding Theorem to guarantee that single-channel time-delay embeddings can topologically reconstruct the entire attractor.

### Key Designs

1.  **Spectral Parameterization of the SPD Jacobian**:
    - **Function**: Directly formulates the transport map Jacobian as $A = U^\top \Lambda U$, where $\Lambda$ is a diagonal vector of non-negative eigenvalues and $U$ is an orthogonal matrix constructed from $R$ Householder reflections $I - 2vv^\top$.
    - **Mechanism**: Brenier's Theorem implies that W2 optimal transport between Gaussians must be an affine SPD map. Instead of searching the $n^2$ matrix space and performing eigen-decomposition, spectral factors are used as parameters. The complexity is $O(Rn)$ compared to $O(n^3)$.
    - **Design Motivation**: Eliminate $O(n^3)$ eigen-decomposition and allow eigenvalues to serve as "stretch signals" comparable across patches for local stability diagnosis.

2.  **Bidirectional Coupled Encoder (Inspired by ANF)**:
    - **Function**: Links context $x$ and latent variable $z$ through 5 layers of mutually affine coupling blocks to output a low-dimensional Gaussian $z$ that summarizes the geometry of $x$.
    - **Mechanism**: Each layer generates four vectors $(s^i_x, t^i_x, s^i_z, t^i_z)$ to iteratively update $z^{i+1} = s^i_z \odot z^i + t^i_z$ and $x^{i+1} = s^i_x \odot x^i + t^i_x$. $z$ starts as isotropic $\mathcal{N}(0, I)$ and is molded into an anisotropic ellipsoid.
    - **Design Motivation**: Direct $s(x) \odot x$ formulations often lead to gradient explosions. Introducing $z$ stabilizes training and compresses Takens embedding information into a low-dimensional Gaussian via diffeomorphism.

3.  **Patch-wise Parallel Decoding**:
    - **Function**: Splits the long horizon $n$ into $n_p$ patches, each of which independently performs an SPD transport prediction.
    - **Mechanism**: Each patch prediction costs $O(R \cdot p)$, resulting in a total cost of $O(R \cdot n)$. Patches are processed in parallel without sequential dependencies.
    - **Design Motivation**: Converts the curse of dimensionality into a dividend—searching 14 independent 24D SPD spaces is computationally cheaper than searching a single 336D SPD space, while sharing the backbone allows for "patch-wise ellipsoidal chains."

### Loss & Training
The model is supervised using the standard Huber loss for point prediction, with **no explicit supervision on eigenvalues**. Despite this, in Lorenz-63 experiments, the learned maximum eigenvalues spontaneously increase in high-velocity regions (outer loops) and decrease at bottlenecks. Spectral structure emerges as a diagnostic signal from MSE training rather than through handcrafted priors.

## Key Experimental Results

### Main Results

| Dataset Type | Metric | Fern | Strongest Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Non-stationary Synthetic Shock | EPT (Effective Prediction Time) | Significantly Lead | DLinear / Koopa | **Up to 790×** |
| Lorenz-63 (Single Channel) | Attractor Reconstruction | Geometrically Consistent | Mainstream LTSF | Qualitatively Superior |
| Real Stationary Benchmarks | MSE | Comparable to SOTA | PatchTST, etc. | Competitive |

### Ablation Study

| Configuration | Key Finding | Description |
| :--- | :--- | :--- |
| Full Fern | Ellipsoidal Prediction + Spectral Diagnosis | Complete model |
| w/o SPD Spectral Param. | Cost $O(n^3)$, not scalable | Validates necessity of SPD constraint |
| w/o Bidirectional Coupling | Gradient Explosion | Validates stabilization by $z$ latent |
| Single Patch vs. Patch-wise | Patch-wise reduces cost significantly | 14x24D is cheaper than 1x336D |

### Key Findings
-   **Emergence of Spectral Structure**: Under MSE supervision, the model's maximum eigenvalues align with the velocity field of the Lorenz-63 system, proving that "structure as diagnosis" is more direct than probabilistic scoring like CRPS.
-   **CI remains superior to CD**: The authors reinterpret this via Takens' Theorem and Mori-Zwanzig formalism, suggesting that single-channel TDEs already topologically cover the state space, and mixing channels may dilute the manifold with noise.
-   **Benchmark Blind Spots**: Traditional LTSF benchmarks are dominated by mild drifts. The new EPT metric specifically measures how long a model can maintain geometric accuracy under regime shifts.

## Highlights & Insights
-   **Geometric Constraint of Search Space**: Brenier's Theorem provides the existence of affine SPD transport between Gaussians. This allows the search space to be reduced from $n^2$ matrices to the SPD cone and then to $O(Rn)$ Householder representations, yielding exponential computational dividends.
-   **Spectral Factors as Interpretable Byproducts**: Direct parameterization of $\Lambda$ and $U$ makes eigenvalues comparable across patches (due to the shared Gaussian source), an advantage not shared by methods that calculate the Jacobian post-hoc.
-   **Theoretic Narrative for CI vs. CD**: By introducing dynamical systems theory (Takens / Mori-Zwanzig) into the LTSF discussion, the paper argues that CI is a theoretical consequence rather than an engineering coincidence.

## Limitations & Future Work
-   The use of Brenier's Theorem is currently limited to Gaussian targets; non-Gaussian tails (heavy-tailed or bimodal) would require more general OT tools not covered by Fern.
-   The focus is primarily on univariate point prediction. Extension to probabilistic evaluation (NLL / CRPS) and true multivariate scenarios is left for future work.
-   The EPT metric and synthetic shock benchmarks are newly proposed and require community validation.
-   The number of Householder reflections $R$ is a critical hyperparameter, and currently lacks an automatic selection mechanism.

## Related Work & Insights
-   **vs. DLinear / PatchTST**: While these use linear heads or Transformers to fit conditional means, Fern explicitly models the Jacobian spectral structure, matching their performance on stationary data and significantly exceeding it on non-stationary data.
-   **vs. Koopa (Koopman Operator)**: Koopa seeks to linearize dynamical systems using global operators, whereas Fern uses local, data-dependent SPD maps, providing better robustness to regime shifts.
-   **vs. Neural ODE / Flow Matching**: These methods often solve ODEs/SDEs to obtain the Jacobian, whereas Fern provides closed-form spectral parameters.
-   **vs. Probabilistic Metrics**: The authors argue that "structure = diagnosis," shifting uncertainty quantification from probability scores to geometric spectra.

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ First systematic application of Brenier's Theorem in LTSF; unique spectral parameterization.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Validated across synthetic non-stationary shocks, Lorenz-63, and real datasets, though lacks some probabilistic metrics.
-   Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations with a strong integration of dynamical systems and engineering.
-   Value: ⭐⭐⭐⭐⭐ Provides a new baseline and a robust theoretical framework for the CI vs. CD debate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Fern: Chaining Spectral Pearls — Ellipsoidal Forecasting Beyond Trajectories for Time Series](../../NeurIPS2025/time_series/friren_beyond_trajectories_--_a_spectral_lens_on_time.md)
- [\[ICML 2026\] Time-series Forecasting Through the Lens of Dynamics](time-series_forecasting_through_the_lens_of_dynamics.md)
- [\[ICML 2026\] From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)
- [\[ICML 2026\] Nested Spatio-Temporal Time Series Forecasting](nested_spatio-temporal_time_series_forecasting.md)
- [\[ICML 2026\] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks.md)

</div>

<!-- RELATED:END -->
