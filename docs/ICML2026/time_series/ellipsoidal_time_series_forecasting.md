---
title: >-
  [Paper Note] Ellipsoidal Time Series Forecasting
description: >-
  [ICML 2026][Time Series][Long-term Forecasting] Fern reformulates long-term time series forecasting as "optimal transport from a fixed Gaussian source to a data-dependent ellipsoid…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Long-term Forecasting"
  - "SPD Jacobian"
  - "Brenier Theorem"
  - "Ellipsoidal Transport"
  - "Non-stationary Robustness"
date: 2026-05-08
content_hash: 61acb0a4585b7c7d
---

# Ellipsoidal Time Series Forecasting

**Conference**: ICML 2026  
**arXiv**: [2505.17370](https://arxiv.org/abs/2505.17370)  
**Code**: None  
**Area**: Time Series Forecasting / Optimal Transport / Dynamical Systems  
**Keywords**: Long-term Forecasting, SPD Jacobian, Brenier Theorem, Ellipsoidal Transport, Non-stationary Robustness

## TL;DR
Fern reformulates long-term time series forecasting as "optimal transport from a fixed Gaussian source to a data-dependent ellipsoid," leveraging the Brenier theorem to restrict the search space to SPD (symmetric positive definite) class Jacobians. Using low-rank spectral decomposition via Householder reflections, the computational cost is reduced from $O(n^3)$ to $O(Rn)$. In non-stationary shock scenarios, Fern achieves up to 790× stability improvement over baselines like DLinear/Koopa.

## Background & Motivation

**Background**: The long-term time series forecasting (LTSF) community has established strong baselines such as PatchTST, DLinear, Koopa, and iTransformer. Mainstream approaches either use channel-independent (CI) linear heads to fit conditional means directly or channel-dependent (CD) Transformers to mix multiple channels. Benchmark results are highly competitive, with ongoing debates between CI and CD approaches.

**Limitations of Prior Work**: The authors point out that current evaluations obscure model fragility in non-stationary scenarios—most benchmarks involve mild drifts in electricity/traffic/weather data. When faced with regime shifts, chaotic shocks, or genuine random noise, strong baselines collapse rapidly, and pointwise metrics like MSE fail to reveal where and how models break down. Additionally, the traditional "direct Jacobian modeling" approach requires $n^2$ components for an $n$-dimensional horizon, and eigen-decomposition costs $O(n^3)$, making it computationally infeasible.

**Key Challenge**: Effective long-term forecasting must preserve **local geometric structure** (spectral information indicating "which direction the system stretches most at each step") while remaining within computational budgets. Searching over arbitrary $n\times n$ matrices is both expensive and structurally uninformative.

**Goal**: (1) Find a predictor that is both data-dependent and geometry-aware; (2) treat spectral structure as an "intrinsic parameter" rather than a "post hoc byproduct"; (3) design evaluation protocols that truly expose model fragility under non-stationarity.

**Key Insight**: The authors shift the forecasting perspective from "temporal evolution $x \to y$" to "transport from a fixed Gaussian source $\mathcal{N}(0, I)$ to the target distribution." When the target is Gaussian, the Brenier theorem guarantees the optimal transport is uniquely affine (SPD scaling + translation), naturally constraining the Jacobian to the SPD cone.

**Core Idea**: Rather than learning an implicit nonlinear mapping and then extracting its Jacobian, directly parameterize an SPD matrix $A = U^\top \Lambda U$ (using Householder reflections + diagonal spectrum) as the optimal transport map, making spectral information (eigenvalues, eigenvectors) an interpretable diagnostic built into the model.

## Method

### Overall Architecture
Fern's pipeline is a "bidirectional coupled encoder + SPD projection" compact model: the input is a univariate time series window $x$ of length $L$ (e.g., 70), and the output is a future patch prediction of length $n$ (e.g., 24). The long horizon is split into patches and decoded in parallel. The intermediate state includes a low-dimensional Gaussian latent variable $z \sim \mathcal{N}(\mu(x), \Sigma(x))$ and a fixed noise source $y_0 \sim \mathcal{N}(0, I)$; the final affine mapping $y^* = U^\top \Lambda U (y_0 + t_y)$ "molds" the noise into the target ellipsoid.

The architecture follows the channel-independent principle: each channel is processed separately, relying on Takens' embedding theorem to ensure that single-channel time-delay embedding can topologically reconstruct the entire attractor, thus eliminating the need for explicit cross-channel mixing.

### Key Designs

1. **Spectral Parameterization of SPD Jacobian**:

    - **Function**: Directly express the transport map's Jacobian as $A = U^\top \Lambda U$, where $\Lambda$ is a diagonal non-negative eigenvalue vector and $U$ is an orthogonal matrix composed of $R$ Householder reflections $I - 2vv^\top$.
    - **Mechanism**: The Brenier theorem states that the W2 optimal transport between Gaussians is always an affine SPD map. Instead of searching over arbitrary $n^2$ matrices and then performing eigen-decomposition, parameterize the spectral factors directly. The cost for $\Lambda$ is $O(n)$, and for $R$ reflection vectors is $O(Rn)$, totaling $O(Rn)$. Setting $R=n$ gives full capacity; smaller $R$ compresses capacity.
    - **Design Motivation**: Eliminates the $O(n^3)$ eigen-decomposition cost and makes eigenvalues natural "cross-patch comparable stretching signals," directly usable for local stability diagnostics.

2. **Bidirectional Coupled Encoder (Inspired by ANF)**:

    - **Function**: Couples context $x$ and latent $z$ through five layers of mutually affine-updating blocks, ultimately outputting a low-dimensional Gaussian $z$ summarizing the geometric information of $x$.
    - **Mechanism**: Each layer's head simultaneously generates four vectors $(s^i_x, t^i_x, s^i_z, t^i_z)$, iteratively updating $z^{i+1} = s^i_z \odot z^i + t^i_z$ and $x^{i+1} = s^i_x \odot x^i + t^i_x$ (Alg. 1). Updates alternate; $z$ starts as isotropic $\mathcal{N}(0, I)$ and is "squeezed" into an anisotropic ellipsoid.
    - **Design Motivation**: Directly using $s(x) \odot x$ leads to gradient explosion; introducing latent $z$ stabilizes training and allows Takens embedding information to be compressed into a low-dimensional Gaussian via "diffeomorphic" mapping, facilitating SPD projection.

3. **Patch-wise Parallel Decoding**:

    - **Function**: Splits the long horizon $n$ into $n_p$ patches (e.g., 14 patches of 24 dimensions each), with each patch independently predicted via SPD transport.
    - **Mechanism**: The Brenier theorem applies equally to 24-to-24 dimensional mappings, yielding different optimal transports per patch. Each patch costs $O(R \cdot p)$, total cost $O(R \cdot n)$; patches do not depend on previous outputs, enabling full parallelism.
    - **Design Motivation**: Turns the curse of dimensionality into a benefit—14 independent 24-dimensional SPD searches are much cheaper than a single 336-dimensional one. The backbone is shared across patches, and predictions can be interpreted as a "chain of ellipsoids per patch."

### Loss & Training
A simple Huber loss supervises point predictions, **with no supervision on eigenvalues**. Nevertheless, in Lorenz-63 experiments, the model's learned maximum eigenvalue spontaneously increases in high-speed regions (outer ring) and decreases at bottlenecks—demonstrating that spectral structure emerges as a diagnostic signal from MSE training, not as a manually injected prior.

## Key Experimental Results

### Main Results

| Dataset Type         | Metric         | Fern         | Strongest Baseline | Stability Gain   |
|----------------------|---------------|--------------|--------------------|------------------|
| Non-stationary synthetic shock | EPT (Effective Prediction Time) | Significantly ahead | DLinear / Koopa | **Up to 790×**   |
| Lorenz-63 univariate | Attractor reconstruction | Geometrically consistent | Mainstream LTSF | Qualitatively superior |
| Real stationary benchmarks | MSE           | Comparable to SOTA | PatchTST, etc.      | On par           |

### Ablation Study

| Configuration        | Key Findings           | Notes                        |
|----------------------|-----------------------|------------------------------|
| Full Fern            | Ellipsoid prediction + spectral diagnostics | Complete model              |
| w/o SPD spectral parameterization (arbitrary matrix Jacobian) | Cost $O(n^3)$, not scalable | Validates necessity of SPD constraint |
| w/o bidirectional coupling encoder ($s \odot x$ directly) | Gradient explosion           | Validates stabilizing role of $z$ latent |
| Single patch (no parallelism) vs patch-wise | Patch-wise greatly reduces cost | 14×24D patches much cheaper than 336D monolithic |

### Key Findings
- **Emergent Spectral Structure**: With only MSE supervision, the model's maximum eigenvalue spontaneously aligns with the Lorenz-63 velocity field, confirming that "structure itself is a diagnostic"—a more direct signal than CRPS-type probabilistic scores.
- **CI Still Superior to CD**: The authors reinterpret this using Takens' theorem and the Mori-Zwanzig formalism—single-channel TDE already topologically covers the full state, so indiscriminate channel mixing dilutes the analytic manifold with noise.
- **Benchmark Blind Spots**: Traditional LTSF benchmarks (electricity/traffic/weather) mostly feature mild drifts, lacking true regime shifts, thus long concealing the fragility of strong linear baselines like DLinear. The newly proposed EPT metric specifically measures "how long a model maintains geometric accuracy over the horizon."

## Highlights & Insights
- **Geometric Constraint of Search Space**: The Brenier theorem provides an existence result—"target Gaussian ⇒ affine SPD transport"—allowing the search space to shrink from $n^2$ matrices to the SPD cone, then to $O(Rn)$ Householder representations, with each reduction yielding exponential computational gains.
- **Spectral Factors as Interpretable Byproducts**: Directly parameterizing $\Lambda, U$ makes eigenvalues "cross-patch comparable" (since they share the same Gaussian source), which is unattainable for "learn nonlinear then compute Jacobian" approaches.
- **Rewriting the CI vs CD Theoretical Narrative**: By introducing dynamical systems theory (Takens/Mori-Zwanzig) into LTSF discussions, the authors show that CI is a theorem-driven consequence, not an engineering coincidence—an insight with broad implications for the time series community.

## Limitations & Future Work
- The authors explicitly restrict Brenier to Gaussian targets—non-Gaussian tails (heavy-tailed, bimodal) require more general OT tools, which Fern does not currently address.
- Focus is solely on univariate point prediction; probabilistic evaluation (NLL/CRPS) and true multivariate cases are deferred to future work. While the SPD spectrum is present in the model's internal covariance, using it for uncertainty quantification requires further calibration.
- The EPT metric and synthetic shock benchmark are newly proposed and only evaluated in the paper; community adoption remains to be seen, and baseline comparisons are based on the authors' own non-stationary protocol.
- The number of Householder reflections $R$ is a key hyperparameter; the paper discusses trade-offs from $R=2$ to $R=n$ but lacks an automatic selection mechanism.

## Related Work & Insights
- **vs DLinear / PatchTST**: These use linear heads/Transformers to fit conditional means directly; Fern explicitly models the Jacobian's spectral structure, matching SOTA on stationary benchmarks and outperforming on non-stationary ones.
- **vs Koopa (Koopman Operator)**: Koopa also aims to linearize dynamical systems but uses global operators; Fern is locally SPD, data-dependent, and more robust to regime shifts.
- **vs Neural ODE / Flow Matching**: Also inspired by OT, but NeuralODEs require solving ODEs/SDEs to obtain Jacobians, while Fern provides closed-form spectral parameters.
- **vs CRPS/NLL Probabilistic Forecasting**: The authors advocate "structure = diagnosis," shifting uncertainty quantification from probabilistic scores to geometric spectra, potentially inspiring new diagnostic protocols.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic use of the Brenier theorem in LTSF; spectral parameterization is unique in the time series community.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-tier validation with synthetic non-stationary benchmarks, Lorenz-63, and real datasets, but lacks probabilistic metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations, integrating dynamical systems frameworks into engineering papers; highly readable.
- Value: ⭐⭐⭐⭐⭐ Provides a new community baseline and reshapes the CI vs CD theoretical narrative, with significant long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Fern: Chaining Spectral Pearls — Ellipsoidal Forecasting Beyond Trajectories for Time Series](../../NeurIPS2025/time_series/friren_beyond_trajectories_--_a_spectral_lens_on_time.md)
- [\[ICML 2026\] Time-series Forecasting Through the Lens of Dynamics](time-series_forecasting_through_the_lens_of_dynamics.md)
- [\[ICML 2026\] From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)
- [\[ICML 2026\] CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)
- [\[ICML 2026\] DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)

</div>

<!-- RELATED:END -->
