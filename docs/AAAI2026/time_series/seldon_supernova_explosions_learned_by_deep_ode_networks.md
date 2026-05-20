---
title: >-
  [Paper Note] SELDON: Supernova Explosions Learned by Deep ODE Networks
description: >-
  [AAAI 2026][Time Series][Continuous-time modeling] This paper proposes SELDON, a continuous-time VAE combining a masked GRU-ODE encoder, an implicit Neural ODE propagator…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Continuous-time modeling"
  - "Neural ODE"
  - "Variational Autoencoder"
  - "Supernova light curves"
  - "Irregular time series forecasting"
date: 2026-05-08
content_hash: 93c020476ca82dad
---

# SELDON: Supernova Explosions Learned by Deep ODE Networks

**Conference**: AAAI 2026
**arXiv**: [2603.04392](https://arxiv.org/abs/2603.04392)  
**Code**: [GitHub](https://github.com/skai-institute/seldon)  
**Area**: Time Series / Astrophysics
**Keywords**: Continuous-time modeling, Neural ODE, Variational Autoencoder, Supernova light curves, Irregular time series forecasting

## TL;DR
This paper proposes SELDON, a continuous-time VAE combining a masked GRU-ODE encoder, an implicit Neural ODE propagator, and an interpretable Gaussian basis function decoder, designed for sparse and irregularly sampled astronomical light curve prediction. SELDON outperforms baseline methods in accurate multi-band flux prediction using only 20% of observed data.

## Background & Motivation

**Background**: The Vera C. Rubin Observatory's LSST survey is about to go online, expected to generate approximately 10 million public alerts per night. Traditional physics-based inference methods (MCMC) require hours to process a single supernova light curve and are entirely incapable of keeping pace with this data deluge. AI models capable of millisecond-level inference are needed to predict light curves in real time and prioritize spectroscopic follow-up observations.

**Limitations of Prior Work**:
   - **Classical time series methods are inapplicable**: ARMA/ARIMA assume equally spaced intervals and stationarity, whereas astronomical light curves are highly irregularly sampled and non-stationary. Continuous-time extensions such as CARMA have computational complexity of $O(N^3)$.
   - **Existing deep learning methods offer limited functionality**: SuperNNova, RAPID, and similar models primarily perform classification or coarse parameter regression rather than full multi-band flux prediction.
   - **Fixed-time-grid VAEs are inapplicable**: STORN, VRNN, and similar models require equally spaced inputs and cannot handle sparse, irregularly sampled astronomical data.
   - **Continuous-time models lack interpretability**: Existing Neural ODE methods (ODE-RNN, Latent ODE) use MLP decoders whose outputs lack physical interpretability.

**Key Challenge**: Astronomical light curves are simultaneously sparse (averaging only 18 observations per curve), irregular, heteroscedastic, multi-band coupled, and non-stationary, demanding a model that can handle these data characteristics while delivering interpretable predictions at millisecond timescales.

**Goal**: Starting from limited early observations (particularly pre-peak), the paper aims to predict complete multi-band light curves of Type Ia supernovae in real time and extract physically interpretable parameters (rise time, decay rate, peak flux, etc.) to guide follow-up observation strategies.

**Key Insight**: Organically combining four components — GRU-ODE (for irregular time series), Neural ODE (for continuous-time propagation), Deep Sets (for permutation-invariant aggregation), and a Gaussian basis function decoder (for physical interpretability).

**Core Idea**: Encode sparse irregular observations with GRU-ODE, propagate the latent state in continuous time via Neural ODE, and decode physically interpretable light curve parameters using a Gaussian basis function decoder.

## Method

### Overall Architecture
SELDON is a customized VAE: multi-band (u, g, r, i, z, y) sparse irregular light curves are input → encoded to an initial hidden state via a masked GRU-ODE → forward-evolved onto a regular time grid via Neural ODE → aggregated into a latent distribution via Deep Sets → decoded into a continuous function per band via Gaussian basis functions → enabling flux prediction queries at arbitrary time points.

### Key Designs

1. **Masked GRU-ODE Encoder**:

    - **Function**: Encodes sparse, irregularly sampled multi-band light curves into a fixed-length hidden state.
    - **Mechanism**: Observations are processed in reverse chronological order; at each observation time a GRU update is performed, while gaps between observations are filled by continuously propagating the hidden state via Neural ODE $\frac{dh}{dt} = f_\theta(h)$. The input is a 5-dimensional vector $[\tilde{g}_i, \mathbf{e}_i^\top]$ (log-scaled flux + learnable band embedding), with temporal information implicitly handled by the ODE propagation.
    - **Design Motivation**: The GRU handles discrete observation arrival events while the ODE smoothly fills inter-observation gaps. Compared to a pure GRU (which ignores temporal irregularity) or a pure ODE (which cannot handle discrete observation updates), GRU-ODE combines the advantages of both.

2. **Implicit Neural ODE Propagator + Deep Sets Aggregation**:

    - **Function**: Forward-evolves the encoder's final hidden state onto a regular time grid, then aggregates it into a latent distribution.
    - **Mechanism**: Using the encoder output as the initial condition, a Tsit5 adaptive solver integrates the Neural ODE at 50 equally spaced time points (covering ~72 days of evolution), yielding a hidden_dim×50 trajectory. This trajectory is mapped to an approximate posterior $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}\boldsymbol{\sigma}^2)$ with latent dimension 64 via Deep Sets (element-wise network $\phi$ → sum pooling → MLP $\rho$).
    - **Design Motivation**: Neural ODE produces a temporally continuous dense trajectory, while Deep Sets provides permutation-invariant aggregation. This combination transforms sparse inputs into a structured latent representation.

3. **Interpretable Gaussian Basis Function Decoder**:

    - **Function**: Decodes the latent vector into a continuous light curve for each band.
    - **Mechanism**: The flux in each band $b$ is modeled as a weighted sum of $K=8$ Gaussian basis functions: $\hat{f}_b(t) = \sum_{k=1}^K w_{bk} \exp[-(( t - \mu_{bk})\sigma_{bk})^2]$. A 4-layer ResNet decodes the latent vector to predict amplitudes $w$, center times $\mu$, and width parameters $\sigma$, which directly correspond to physical quantities (rise time, decay rate, peak flux).
    - **Design Motivation**: (1) Physical interpretability — parameters carry explicit astronomical meaning and can directly drive downstream observation scheduling; (2) Global amplitude and center time are disentangled — decoded from independent subsets of the latent vector, providing scale and temporal translation invariance; (3) Predictions can be queried at arbitrary continuous time points.

### Loss & Training
- **Reconstruction loss**: Huber loss ($\delta=1$) on standardized residuals $r = (f - \hat{f})/\sigma_f$, naturally accounting for heteroscedasticity.
- **Regularization**: KL divergence constrains the latent space, with $\beta = 10^{-4}$.
- **Data augmentation**: At each training step, the first $K$ observations of the light curve are randomly sampled ($K$ drawn uniformly from 10 to the full length), simulating real survey scenarios where only partial curves are available.
- **Band frequency balancing**: Embedding gradients are weighted by the inverse of per-band occurrence frequency to mitigate band imbalance.
- **Training setup**: 180 epochs, Adam optimizer, batch accumulation 4×512 on an Nvidia H100, with 7.5 seconds per step.

## Key Experimental Results

### Main Results

Out-of-sample prediction performance of three encoder variants at different observation fractions (ELAsTiCC dataset, Type Ia supernovae):

| Obs. Fraction | Metric | Deep Sets | Masked-GRU | **SELDON** |
|---------|------|-----------|------------|-----------|
| 20% | Mean\|Z\|↓ | 9.862 | 10.237 | **8.929** |
| 20% | Max\|Z\|↓ | 309.186 | 164.536 | **151.551** |
| 20% | NRMSE↓ | 0.065 | 0.067 | **0.045** |
| 50% | Mean\|Z\|↓ | 5.193 | 9.526 | **4.295** |
| 50% | NRMSE↓ | 0.044 | 0.073 | **0.034** |
| 90% | Mean\|Z\|↓ | 2.329 | 9.209 | **1.906** |
| 90% | NRMSE↓ | 0.034 | 0.085 | **0.028** |

### Ablation Study

| Encoder | Mean\|Z\|(50%) | Max\|Z\|(50%) | NRMSE(50%) | Notes |
|--------|-------------|-------------|---------|------|
| Masked-GRU | 9.526 | 147.793 | 0.073 | Does not handle temporal irregularity |
| Deep Sets | 5.193 | 185.677 | 0.044 | Permutation-invariant but prone to catastrophic outliers |
| **SELDON (GRU-ODE+DS)** | **4.295** | **104.237** | **0.034** | Best overall performance with tightest tail behavior |

### Key Findings
- **SELDON consistently leads across all observation fractions (≥20%)**: NRMSE is 20–35% lower than GRU and 30–50% lower than Deep Sets.
- **Best control of extreme residuals**: Max|Z| is consistently the lowest; Deep Sets exhibits catastrophic outliers (reaching 848σ at 10%), GRU reaches ~168σ, while SELDON is contained within 159σ.
- **Advantage grows with more observations**: At 10%, GRU is marginally better (Mean|Z| 10.3 vs. 10.5), but SELDON pulls ahead from 20% onward and the gap continues to widen, indicating that Neural ODE makes better use of incremental information.
- **Masked-GRU's anomalous non-improvement**: GRU's Mean|Z| barely decreases as observations increase (10.3→9.2), suggesting that fixed-step updates cannot effectively exploit long-sequence information.
- **Inference speed**: Single-step inference takes 1.1 seconds, easily keeping pace with 10 million nightly alerts.

## Highlights & Insights
- **Physically interpretable decoder design**: The parameters of the Gaussian basis functions (amplitude = peak flux, center = peak time, width = rise/decay rate) map directly to astronomical physical quantities without post-processing. This design principle generalizes to any scientific time series modeling with a prior functional form.
- **Global–local disentangled latent space**: Of the 64-dimensional latent vector, 48 dimensions encode local basis function parameters (relative shape), 8 dimensions encode global center (temporal offset), and 8 dimensions encode global amplitude (brightness scale), achieving scale and temporal translation invariance.
- **Data augmentation strategy**: Randomly truncating light curves at each training step to simulate real early-observation scenarios enables the model to learn extrapolation from limited data during training, which is critical for early-epoch prediction.
- **Architectural generality**: Although designed for astronomy, the architecture itself is a general-purpose solution for "sparse irregular multivariate time series → continuous-time interpretable prediction" and is transferable to domains such as medical monitoring and sensor networks.

## Limitations & Future Work
- **Validation limited to Type Ia supernovae**: Only one type of astronomical transient event is tested; light curve morphologies differ substantially for other types (Type II supernovae, kilonovae, etc.), and generalizability remains to be verified.
- **Single dataset (ELAsTiCC simulation)**: Although ELAsTiCC is a realistic survey simulation, a gap remains with real observational data.
- **Expressive capacity of Gaussian basis functions**: $K=8$ Gaussian basis functions may be insufficiently flexible for atypical light curves, such as those exhibiting multiple peaks or plateau phases.
- **No comparison with physical models**: Quantitative comparison with traditional supernova physical template fitting methods such as SALT3 is absent.
- **Max|Z| remains large**: Even for SELDON, Max|Z| reaches 159σ at 10% observations, indicating that predictions can still deviate severely in extreme cases.

## Related Work & Insights
- **vs. ODE-RNN/GRU-ODE-Bayes**: Encoders are similar but decoders differ. ODE-RNN uses an MLP decoder, which is not interpretable; SELDON uses Gaussian basis functions whose parameters directly correspond to physical quantities.
- **vs. Latent ODE**: Latent ODE generates in continuous time but lacks band-specific interpretable outputs. SELDON adds band embeddings and physical basis functions.
- **vs. SuperNNova/RAPID/ORACLE**: These models perform classification rather than flux prediction; SELDON produces full multi-band light curve predictions and physical parameter inference.
- **vs. GP-VAE**: GP-VAE employs a Gaussian process prior in the latent space but still assumes equally spaced inputs; SELDON natively handles irregular sampling.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combined architecture of GRU-ODE + Neural ODE + Deep Sets + Gaussian basis functions is innovative, and the physically interpretable decoder design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐ The three-baseline comparison is adequate, but comparisons with physical models and validation on real observational data are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The architecture is described in detail, data preprocessing is transparent, and figures are clear.
- **Value**: ⭐⭐⭐⭐ Directly useful to the astronomical AI community; the continuous-time and interpretable architecture design is also instructive for scientific time series modeling in other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Transparent Networks for Multivariate Time Series](transparent_networks_for_multivariate_time_series.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[AAAI 2026\] Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)
- [\[ICLR 2026\] WARP: Weight-Space Linear Recurrent Neural Networks](../../ICLR2026/time_series/weight-space_linear_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](../../NeurIPS2025/time_series/selective_learning_for_deep_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
