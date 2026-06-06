---
title: >-
  [Paper Note] HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction
description: >-
  [AAAI 2026][Time Series][Domain Generalization] This paper proposes HydroDCM, the first framework to introduce Domain Generalization (DG) into hydrological forecasting. It constructs pseudo-domain labels from spatial met…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Domain Generalization"
  - "Cross-Reservoir Inflow Prediction"
  - "Adversarial Training"
  - "FiLM Modulation"
  - "Spatial Meta-Attributes"
date: 2026-05-08
content_hash: a8a56491239a5577
---

# HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction

**Conference**: AAAI 2026
**arXiv**: [2512.03300](https://arxiv.org/abs/2512.03300)  
**Code**: [github.com/humphreyhuu/HydroDCM](https://github.com/humphreyhuu/HydroDCM)  
**Area**: Time Series
**Keywords**: Domain Generalization, Cross-Reservoir Inflow Prediction, Adversarial Training, FiLM Modulation, Spatial Meta-Attributes

## TL;DR

This paper proposes HydroDCM, the first framework to introduce Domain Generalization (DG) into hydrological forecasting. It constructs pseudo-domain labels from spatial meta-attributes to guide adversarial learning for invariant feature extraction, then employs a FiLM adapter to modulate features conditioned on the target reservoir's geographical information, enabling cross-domain inflow prediction for unseen reservoirs.

## Background & Motivation

Reservoir Inflow Prediction is critical for flood control scheduling, water resource allocation, and hydropower generation. Deep learning methods have achieved strong performance on individual reservoirs, yet face two core challenges:

**Domain Shift Problem**: Each reservoir's inflow pattern is shaped by unique climatic conditions, geographical location, and catchment operations. Models trained on one reservoir suffer severe performance degradation when directly applied to another.

**Data Scarcity Problem**: Newly constructed or ungauged reservoirs lack sufficient historical data to independently train reliable predictive models.

Conventional DG methods face two particular difficulties in the hydrological domain:

- **Multi-Domain Classification Difficulty**: Reservoir networks typically comprise dozens to hundreds of stations, each of which should be treated as an independent domain to fully capture domain-specific characteristics. However, the computational overhead of DG methods such as meta-learning scales linearly with the number of domains, making them impractical for large-scale hydrological systems. Methods like DANN assume a small number of latent domains and struggle with fine-grained domain partitioning.
- **Meta-Attribute Dependency**: Reservoirs possess rich spatial and environmental metadata (latitude, longitude, elevation, etc.) that exert an indirect yet significant influence on distinguishing reservoirs. Existing DG methods primarily focus on invariant feature learning and largely overlook the utilization of auxiliary metadata.

**Core Motivation**: Can a DG framework be designed that efficiently handles a large number of domains while leveraging spatial meta-attributes? This is the central challenge in extending DG to hydrological applications for the first time.

## Method

### Overall Architecture

HydroDCM comprises four modules operating in two stages:
- **Stage 1** (training): Temporal feature extractor $f_\phi$ + domain discriminator $d_\theta$ (adversarial training to remove domain information)
- **Stage 2** (training + inference): FiLM adapter $m_\beta$ (injecting domain-specific information) + prediction head $p_\omega$
- **Inference**: The domain discriminator is discarded; only $f_\phi$, $m_\beta$, and $p_\omega$ are retained

### Key Designs

1. **Pseudo-Domain Labels Driven by Spatial Meta-Attributes**: Unlike conventional DG approaches that assign one-hot labels per domain, HydroDCM uses a spatial meta-attribute vector $\mathbf{s}_i \in \mathbb{R}^3$ (longitude, latitude, elevation) as a pseudo-domain identifier. Specifically, meta-attributes and observational data are concatenated and projected linearly to obtain pseudo-domain embeddings $\mathbf{v}_i = \mathbf{W}[\mathbf{s}_i : \mathbf{X}_i] + \mathbf{b}$. A contrastive learning loss then ensures that geographically proximate reservoirs are closer in the embedding space:

    $L_{\text{con}} = -\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_i^+)/\tau)}{\exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_i^+)/\tau) + \sum_{j \in \mathcal{N}} \exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_j^-)/\tau)}$

   **Design Motivation**: Continuous spatial meta-attributes are more suitable for multi-domain scenarios than discrete domain labels—they naturally encode inter-domain similarity relationships while avoiding the computational explosion of one-hot encoding when the number of domains is large.

2. **Adversarial Training for Generalization**: The domain discriminator $d_\theta$ attempts to predict the pseudo-domain label $\mathbf{v}_i$ from the temporal feature $\mathbf{h}_i = f_\phi(\mathbf{X}_i)$, while the feature extractor is forced via adversarial training to remove domain-related information:

    $L_{\text{adv}} = -\mathbb{E}_{(\mathbf{x}_i, \mathbf{v})} \|d_\theta(\mathbf{h}_i) - \mathbf{v}_i\|_2^2$

   Gradients propagate only through the discriminator parameters $\theta$, with the direction reversed to encourage discriminator failure. The resulting features $\mathbf{z}$ retain hydrodynamic information relevant to the label (inflow) while filtering out domain-specific noise.

   **Design Motivation**: Unlike standard DANN, using continuous pseudo-domain labels rather than discrete class labels maintains computational efficiency under a large number of domains while better capturing gradual inter-domain relationships.

3. **FiLM Domain-Conditioned Modulation**: Invariant features obtained after adversarial training may lose domain-specific information that is hydrologically meaningful. The FiLM adapter generates scaling and translation coefficients from spatial meta-attributes and applies an affine transformation to the invariant features:

    $\tilde{\mathbf{z}}_i = \gamma(\mathbf{s}_i) \odot \mathbf{z}_i + \delta(\mathbf{s}_i)$

   where $\gamma$ and $\delta$ are generated from meta-attributes $\mathbf{s}_i$ via a lightweight MLP. The modulated features are passed to the prediction head $p_\omega$ to output 7-day inflow forecasts $\hat{y}_i = \text{MLP}(\tilde{\mathbf{z}}_i)$.

   **Design Motivation**: This is the essence of the "remove-then-reintroduce" strategy—adversarial training removes domain bias, while FiLM re-injects domain-specific information in a controlled manner. Compared to predicting directly from invariant features, FiLM modulation can incorporate spatial priors such as terrain gradients and climatic variability.

### Loss & Training

The total loss is a weighted sum of three terms:

$$L_{\text{total}} = \lambda_{\text{con}} L_{\text{con}} + \lambda_{\text{adv}} L_{\text{adv}} + \lambda_{\text{sup}} L_{\text{sup}}$$

where $L_{\text{sup}}$ is the MSE regression loss. Training proceeds in two phases:
- **First 10 epochs**: Only $L_{\text{con}} + L_{\text{sup}}$, for learning invariant representations
- **Subsequent epochs**: $L_{\text{adv}}$ and FiLM modulation are incorporated

Hyperparameter settings: $\lambda_{\text{sup}}=1.0$, $\lambda_{\text{adv}}=0.1$, Adam optimizer, initial learning rate $10^{-3}$, ReduceLROnPlateau scheduler, gradient clipping with max norm=1.0. The feature extractor is a 2-layer Encoder-Decoder LSTM with hidden dimension 64.

## Key Experimental Results

### Main Results

Evaluation is conducted on 30 reservoirs in the Upper Colorado River Basin, with 27 source-domain reservoirs for training and 3 target-domain reservoirs (MCR, JVR, MCP) for testing. The metric is Nash-Sutcliffe Efficiency (NSE, %):

| Method | Overall NSE | Day 1 | Day 4 | Day 7 | Category |
|--------|-------------|-------|-------|-------|----------|
| Base (lower bound) | 78.29 | 87.96 | 78.95 | 67.95 | Source-domain training only |
| Few-shot | 80.08 | 89.08 | 80.19 | 71.03 | Few target-domain samples |
| DANN | 78.89 | 88.63 | 79.50 | 68.94 | DG baseline |
| MLDG | 80.67 | 89.82 | 80.53 | 71.42 | DG baseline |
| CondAdv | 80.77 | 90.06 | 80.73 | 71.61 | DG baseline |
| IRM | 78.50 | 88.15 | 79.13 | 68.20 | DG baseline |
| **HydroDCM** | **82.90** | **92.92** | **82.26** | **73.96** | Ours |
| Oracle (upper bound) | 83.93 | 93.79 | 83.25 | 75.29 | Full target-domain data |

HydroDCM consistently outperforms all DG baselines across all forecast horizons, with an overall NSE only 1.03% below the Oracle.

### Ablation Study

| Configuration | Overall NSE | Note |
|---------------|-------------|------|
| HydroDCM (full) | **82.90** | All modules |
| w/o Adversarial Loss | 79.09 | **Largest drop**; adversarial training is the core |
| w/o Contrastive Loss | 80.30 | Unclear inter-domain separation degrades performance |
| w/o FiLM Adaptor | 81.39 | Absence of domain-adaptive fine-tuning causes consistent degradation |
| w/ Spatial Shuffle | 80.63 | Corrupted spatial meta-attributes cause degradation |

### Key Findings

1. **Adversarial loss is the most critical component**: Removing adversarial training causes the largest performance drop (−3.81%), confirming that eliminating domain covariate shift is central to cross-reservoir generalization.
2. **HydroDCM outperforms Few-shot** (+2.82% without any target-domain supervision): This suggests that the DG paradigm may be more competitive than limited labeled data in hydrological scenarios.
3. **Advantage is more pronounced at longer horizons**: The NSE advantage at Day 7 reaches 2.5–5%, indicating that FiLM modulation effectively suppresses the accumulation of prediction uncertainty.
4. **Spatial meta-attributes provide effective signals**: The performance drop under Spatial Shuffle confirms the importance of geographical information.

## Highlights & Insights

- **First application of DG to hydrological forecasting**: Fills a gap in domain generalization research within hydrology, establishing a well-defined problem formulation and evaluation protocol.
- **Elegant pseudo-domain label design**: Replacing discrete domain labels with continuous spatial meta-attributes naturally accommodates multi-domain scenarios.
- **"Remove-then-reintroduce" strategy**: Adversarial training removes domain bias, and FiLM re-injects domain-specific information, offering greater flexibility than purely invariant feature learning.
- **Computationally efficient**: The FiLM adapter has very few parameters and requires only a single forward pass at inference, making it suitable for real-time forecasting.

## Limitations & Future Work

1. **Only 3 target reservoirs**: The evaluation scale is small; validation on larger basins (e.g., the 671 catchments of the CAMELS dataset) is needed.
2. **Only 3-dimensional spatial meta-attributes**: Only latitude, longitude, and elevation are used; richer catchment characteristics (e.g., drainage area, mean annual precipitation, soil type) are not included.
3. **Fixed 7-day forecast horizon**: Generalization capability at longer prediction ranges is not evaluated.
4. **Limited choice of temporal feature extractor**: Only LSTM is employed; stronger temporal encoders such as Transformers have not been explored.

## Related Work & Insights

- **FiLM Modulation (Perez et al., 2018)**: Originally proposed for visual QA, this work innovatively applies it to hydrological domain adaptation by conditioning features on domain information.
- **DANN Family**: Variants of the standard gradient reversal layer. This paper replaces discrete domain labels with continuous pseudo-labels, a strategy worth generalizing to other multi-domain settings.
- **Hydrological Foundation Models**: Kratzert et al. advocate against training LSTMs on individual catchments; this paper provides a concrete mechanism for generalizing from multiple catchments to unseen ones.

## Rating

- Novelty: ⭐⭐⭐⭐ — The DG framework itself is not novel, but its application in hydrology and the pseudo-domain label design are innovative.
- Experimental Thoroughness: ⭐⭐⭐ — Dataset scale is limited (30 reservoirs / 3 targets); ablation design is reasonable but lacks multi-basin validation.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, motivation is well-argued, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Significant practical value for the hydrological domain; generalizability to other domains remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ResCP: Reservoir Conformal Prediction for Time Series Forecasting](../../ICLR2026/time_series/rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)
- [\[AAAI 2026\] Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)
- [\[ICML 2026\] HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series](../../ICML2026/time_series/hepa_a_self-supervised_horizon-conditioned_event_predictive_architecture_for_tim.md)
- [\[ICML 2026\] HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](../../ICML2026/time_series/helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)
- [\[AAAI 2026\] Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)

</div>

<!-- RELATED:END -->
