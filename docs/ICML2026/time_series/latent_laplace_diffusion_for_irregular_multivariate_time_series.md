---
title: >-
  [Paper Note] Latent Laplace Diffusion for Irregular Multivariate Time Series
description: >-
  [ICML 2026][Time Series][Irregular Time Series] LLapDiff is a generative framework for **diffusion in latent space**. By parameterizing **stable modal evolution** in the Laplace domain using learnable complex conjugate p…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Irregular Time Series"
  - "Diffusion Models"
  - "Latent Space Generation"
  - "Laplace Domain"
  - "Port-Hamiltonian Systems"
date: 2026-05-08
content_hash: 72cc9482d5b112f0
---

# Latent Laplace Diffusion for Irregular Multivariate Time Series

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.19805](https://arxiv.org/abs/2605.19805)  
**Code**: TBD  
**Area**: Time Series / Generative Models  
**Keywords**: Irregular Time Series, Diffusion Models, Latent Space Generation, Laplace Domain, Port-Hamiltonian Systems

## TL;DR
LLapDiff is a generative framework for **diffusion in latent space**. By parameterizing **stable modal evolution** in the Laplace domain using learnable complex conjugate poles, it achieves long-term forecasting and missing value imputation for irregular time series **without requiring step-by-step physical time integration**; it achieves an average rank of 2.1±1.7 across 7 datasets.

## Background & Motivation

**Background**: Modeling irregular multivariate time series (IMTS) generally falls into three categories: (1) discrete pipelines that interpolate/re-grid data before applying strong sequence models; (2) continuous-time models like Neural ODEs/Continuous RNNs that handle timestamps naturally but require step-by-step numerical integration; (3) diffusion generative models that provide uncertainty quantification but often denoise directly in the observation space, lacking dynamical structure and stability control.

**Limitations of Prior Work**: Discrete methods tend to distort temporal structures under severe irregularity. Step-wise integration in continuous-time models accumulates errors and numerical drift during long-term forecasting. Existing diffusion methods lack explicit stability constraints, making long-term generation unstable under irregular sampling.

**Key Challenge**: How to design a long-term forecasting method that preserves timestamp fidelity, avoids the costs and error accumulation of numerical integration, and guarantees long-term dynamical stability without aggressive grid reorganization?

**Goal**: Design a conditional generative model that incorporates continuous-time inductive biases but eliminates the need for ODE/SDE solvers.

**Key Insight**: Represent the target time series as low-dimensional latent space trajectories and perform diffusion in this latent space; inspired by the energy conservation of **Stochastic Port-Hamiltonian systems**, use stable modal parameterization (complex conjugate poles) in the Laplace domain to guide the reverse process.

**Core Idea**: Use stable modal parameterization $\mathcal{G}(s) = \sum_{k=1}^K \frac{\omega_k \mathbf{c}_k \mathbf{b}_k^\top}{s^2 + 2 \rho_k s + (\rho_k^2 + \omega_k^2)}$ to evaluate generation directly at any query timestamp of the latent trajectory, bypassing step-by-step time integration.

## Method

### Overall Architecture
(1) A pre-trained VAE encoder maps ground truth target sequences to a low-dimensional latent space $\mathbf{z} = \text{VAE}_{\text{enc}}(\mathcal{Y}_{t_i})$. (2) A gap-aware history summarizer $\mathcal{S}_\phi$ compresses observed history $\mathcal{H}_{t_i}$ into a condition vector $\mathbf{E}_{t_i}$. (3) A standard DDPM forward process is executed in the latent space. (4) During reverse denoising, a modal predictor $\mathcal{L}_\theta$ predicts continuous-time modal parameters (decay rate $\rho_k$, oscillation frequency $\omega_k$, and residual vectors $\mathbf{c}_k, \mathbf{b}_k$) based on the current noisy latent state and history summary. A modal synthesizer $\mathcal{L}_\theta^+$ uses these poles to compute the denoised latent trajectory $\hat{\mathbf{z}}_0(t_r) = \sum_k e^{-\hat{\rho}_k \tilde{t}_r}(\hat{\mathbf{c}_k} \cos(\hat{\omega}_k \tilde{t}_r) + \hat{\mathbf{b}}_k \sin(\hat{\omega}_k \tilde{t}_r))$ directly at any query time. (5) A VAE decoder restores the observation space.

### Key Designs

1.  **Port-Hamiltonian Inspired Stable Modal Parameterization**:
    - **Function**: Designs stability biases for latent dynamics using energy conservation principles, preventing numerical drift and infinite energy growth in long-term generation.
    - **Mechanism**: Starting from a Stochastic Port-Hamiltonian SDE, the energy balance equation is derived where the dissipation term $\mathbf{R} \succ 0$ ensures energy decay. The transfer function of the locally linearized system consists of $K$ pairs of complex conjugate poles $(-\rho_k \pm i \omega_k)$. The learner directly predicts $(\hat{\rho}_k, \hat{\omega}_k, \hat{\mathbf{c}}_k, \hat{\mathbf{b}}_k)$, and the constraint $\rho_k > 0$ guarantees stability from the source.
    - **Design Motivation**: Physics-inspired energy constraints guide stable trajectory generation more directly than pure black-box learning; the Hurwitz property of the poles (all real parts are negative) automatically ensures asymptotic stability for long-term forecasting.

2.  **Gap-Aware Conditioning via Renewal-Average Perspective**:
    - **Function**: Associates continuous-time poles with the statistical properties of irregular sampling intervals and designs a structured history summary to encode irregularity patterns.
    - **Mechanism**: Under a renewal process (random sampling intervals $\Delta_j$ i.i.d.), continuous-time poles $s_k = -\rho_k + i \omega_k$ map to equivalent poles $\lambda_k = \mathbb{E}[e^{s_k \Delta}]$ in the event domain. Their logarithm $\bar{s}_k = \log \lambda_k$, via Taylor expansion $\bar{s}_k \approx s_k \mathbb{E}[\Delta] + \frac{1}{2} s_k^2 \text{Var}(\Delta)$, demonstrates how gap statistics modulate decay and oscillation. The history summarizer encodes three types of information: port signals (observations), dynamics signals (finite difference features), and temporal signals (timestamps, $\Delta t$ encoding, and masks).
    - **Design Motivation**: Theoretical connections suggest the model must learn to "disentangle" intrinsic dynamical poles from effective pole variations introduced by sampling.

3.  **Dual-layer Framework of Latent Space Generation + VAE Encoding**:
    - **Function**: Performs diffusion on low-dimensional latent trajectories to avoid difficulties caused by sparse masks and high dimensionality when denoising directly in the observation space.
    - **Mechanism**: Discrete diffusion is performed on latent vectors $\mathbf{z} \in \mathbb{R}^{h \times d_z}$ ($d_z \ll d_y$, latent dimension typically 4-16) rather than the $h \times d_y$ observation trajectory. Pre-training the VAE (frozen) is done independently on the training set, after which the diffuser learns conditional generation $p_\theta(\mathbf{z} \mid \mathbf{E}_{t_i})$.
    - **Design Motivation**: Diffusion in latent space is more stable and efficient; the VAE prior provides good initialization and regularization for diffusion learning.

## Key Experimental Results

### Main Results

| Dataset | Metric | DLinear | PatchTST | TimeGrad | mTAN | NeuralCDE | ContiFormer | **Ours** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BMS Air (h=168) | CRPS | 1.448 | 0.929 | 0.537 | 0.547 | 1.019 | 0.984 | **0.516** |
| UCI Air (h=168) | CRPS | 2.751 | 1.149 | 1.122 | 0.836 | 1.991 | 2.143 | **1.003** |
| PhysioNet (h=12) | CRPS | 0.476 | 0.486 | 0.446 | 0.452 | 0.431 | 0.420 | **0.318** |
| NOAA US (h=168) | CRPS | 0.355 | 0.333 | 0.639 | 0.869 | 0.511 | 0.468 | **0.440** |
| NOAA UK (h=168) | CRPS | 1.546 | 0.750 | 0.639 | 0.869 | 1.114 | 1.354 | **0.557** |
| US Equity (h=100) | CRPS | 0.572 | 0.565 | 0.423 | 0.417 | 0.561 | 0.563 | **0.406** |

Average rank 2.1 ± 1.7 (significantly better than 3.0-6.6 of other diffusion methods).

### Ablation Study

| Configuration | BMS Air | NOAA US | US Equity | Description |
| :--- | :--- | :--- | :--- | :--- |
| Full model | 0.516 | 0.440 | 0.406 | Full model |
| w/o conditioning | 0.816 (+0.30) | 1.450 (+1.01) | 0.466 (+0.06) | Remove history summary |
| w/o learned poles | 0.696 (+0.18) | 1.310 (+0.87) | 0.476 (+0.07) | Remove pole parameterization |
| w/o latent space | 0.666 (+0.15) | 1.030 (+0.59) | 0.446 (+0.04) | Diffuse directly in observation space |
| joint-trained summarizer | 0.806 (+0.29) | 1.360 (+0.92) | 0.476 (+0.07) | Jointly trained summarizer |

### Key Findings
- **Significant Long-term Stability Advantage**: On the longest prediction horizon (h=168) and highly irregular datasets, LLapDiff improves by 15-30% compared to mr-Diff, while the gain narrows to 5-10% at h=24.
- **Effectiveness of Gap-Awareness**: Qualitative results show that LLapDiff maintains coherent trajectories and well-calibrated uncertainty across intervals with multiple missing values.
- **Dual Efficacy in Imputation**: By adding historical missing timestamps to the query set, LLapDiff performs causal filtering-style imputation (CRPS 0.321 vs. CSDI 0.469).
- **Stress Test**: Performance remains stable under manually induced missingness (CRPS change < 0.1 even after a 20% drop in coverage).

## Highlights & Insights
- **Physics-inspired Stability Design**: Port-Hamiltonian energy balance effectively injects second-order dynamical constraints (pole Hurwitz property) into the diffusion denoiser, forcing stability from the root.
- **Ingenuity in Avoiding Step-wise Integration**: By using closed-form modal summation in the Laplace domain instead of matrix exponentials, the model achieves "one-step computation for all timestamps" parallelization, reducing costs from $O(h \cdot T \cdot d_z^3)$ to $O(h \cdot K)$.
- **Creative Application of Renewal-Average Theory**: Drawing inspiration from classic tools in probability theory (renewal theory, characteristic functions), it derives how gap statistics modulate continuous-time dynamics.
- **Unified Prediction and Imputation**: The same model can perform both long-term forecasting and missing value imputation simply by changing the query timestamps (future vs. history).

## Limitations & Future Work
- Latent Space Dimension Trade-off: The impact of latent dimension $d_z$ on long-term stability and computational efficiency is not fully explored.
- Gap between Theory and Practice: Renewal-average analysis assumes i.i.d. intervals, but real-world gaps are often non-stationary and state-dependent.
- Choice of Pole Count $K$: The paper uses a fixed $K$, but different datasets may require varying modal richness.
- Scalability to Ultra-Long-Term Forecasting (h > 500): The experiments were limited to a maximum h=168.

## Related Work & Insights
- **vs. TimeGrad / mr-Diff** (Diffusion Baselines): These mostly denoise in observation space and rely on masks + time embeddings for irregularity, lacking explicit dynamical constraints; LLapDiff introduces rigid energy conservation and pole stability in latent space.
- **vs. NeuralCDE / ContiFormer** (Continuous-Time Baselines): These handle timestamps naturally using Neural ODEs or Continuous Transformers but require step-by-step integration; LLapDiff bypasses integration entirely through Laplace domain parameterization.
- **vs. Structured SSMs (S4, etc.)**: SSMs are efficient for long sequences but mostly designed for synchronous sampling; LLapDiff's gap-aware conditioning and modal parameterization specifically for irregular sampling are novel contributions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Port-Hamiltonian inspired stability design and Laplace pole parameterization is entirely novel, naturally merging physics-inspired energy constraints with modern diffusion frameworks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven datasets, comprehensive ablations, stress tests, and visualizations are solid; lacks a deep verification of ultra-long-term stability and detailed computation time comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clear, motivation is well-established, and experimental results are persuasive.
- Value: ⭐⭐⭐⭐⭐ Addresses the practically important problem of long-term forecasting for irregular time series; the method's physics-inspired nature and transferability (the pole parameterization idea can be extended to other generative tasks) are high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] QuITE: Query-based Irregular Time Series Embedding](quite_query-based_irregular_time_series_embedding.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[ICML 2026\] From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)
- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](../../NeurIPS2025/time_series/omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)

</div>

<!-- RELATED:END -->
