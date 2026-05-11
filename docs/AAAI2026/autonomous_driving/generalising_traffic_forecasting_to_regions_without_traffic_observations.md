---
title: >-
  [Paper Note] GenCast: Generalizing Traffic Forecasting to Regions without Observations
description: >-
  [AAAI 2026][Autonomous Driving][traffic forecasting] This paper proposes GenCast, which achieves generalization of traffic forecasting from sensor-covered regions to unobserved continuous regions via three key innovation…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "traffic forecasting"
  - "spatial-temporal"
  - "physics-informed"
  - "graph neural networks"
  - "generalization"
date: 2026-05-08
content_hash: c197ba4db5080772
---

# Generalising Traffic Forecasting to Regions without Traffic Observations

**Conference**: AAAI 2026
**arXiv**: [2508.08947](https://arxiv.org/abs/2508.08947)
**Code**: [https://github.com/suzy0223/GenCast](https://github.com/suzy0223/GenCast)
**Area**: Autonomous Driving
**Keywords**: traffic forecasting, spatial-temporal, physics-informed, graph neural networks, generalization

# GenCast: Generalizing Traffic Forecasting to Regions without Observations

## TL;DR

This paper proposes GenCast, which achieves generalization of traffic forecasting from sensor-covered regions to unobserved continuous regions via three key innovations: a physics-informed neural network (incorporating the LWR traffic equation as a soft constraint), dynamic external weather signal fusion, and a spatial grouping module. GenCast consistently outperforms existing state-of-the-art methods across five real-world datasets.

## Background & Motivation

**Insufficient sensor coverage**: The high cost of deploying and maintaining traffic sensors results in sparse and limited spatial coverage, leaving many regions without historical traffic observations and creating a large gap between limited observations and the demand for fine-grained, wide-area forecasting.

**Limitations of Prior Work on scattered scenarios**: Kriging-based methods (e.g., IGNNK, INCREASE) and extrapolation methods (e.g., STGNP) perform reasonably well on scattered unobserved points but fail severely when faced with large continuous unobserved regions—they rely on information propagation paths within observed areas and cannot extrapolate effectively.

**SOTA methods rely on static features**: The current best-performing method, STSM, defines similarity based on static auxiliary features such as POI categories and geographic coordinates for selective masking training, and thus cannot capture dynamic traffic pattern changes, limiting generalization capability.

**Structural assumption issues in KITS**: KITS creates virtual nodes within observed regions, implicitly assuming a known-density scattered distribution, which constrains its ability to generalize to large-scale continuous unobserved regions.

**Lack of external knowledge guidance**: Existing purely data-driven methods rely solely on traffic data, neglecting universally applicable knowledge such as physical laws (e.g., traffic flow conservation) and external signals (e.g., weather), missing important sources for enhancing generalization.

**Local noise interfering with generalization**: Unusual traffic patterns at individual locations (e.g., anomalies caused by traffic accidents) introduce non-generalizable local noise, and existing models lack effective mechanisms to filter such interference.

## Method

### Overall Architecture

GenCast is built on a contrastive learning backbone (inherited from STSM). Each epoch generates two views via a randomly masked subgraph (original graph $G_o$ and masked graph $G_o^m$), training the model to maintain prediction consistency with and without masked nodes. Four innovative modules are added upon this backbone:

### 1. Spatio-Temporal Encoder — Differentiable Spatial Representation

To enable physical constraints to backpropagate on discrete graphs, two continuously differentiable spatial embeddings are designed:

- **SE-L (LLM-based embedding)**: Uses LLaMA3 8B Instruct to process text descriptions of each location (including geographic coordinates, POIs, and road segment attributes), generating 4096-dimensional frozen embeddings that encode rich semantic spatial information.
- **SE-H (GeoHash-based embedding)**: Encodes geographic coordinates into strings via GeoHash, then produces trainable embeddings through CharBERT + Transformer encoder, suitable for scenarios with limited location features.

Temporal embeddings use sine/cosine functions to encode intra-day periodicity: $\mathbf{TE}_{enc} = [\sin(2\pi \cdot \mathbf{TE}/T_d), \cos(2\pi \cdot \mathbf{TE}/T_d)]$

### 2. External Signal Encoder — Weather Data Fusion

ECMWF ERA5 global weather data (9km×9km resolution, with four variables: temperature, solar radiation, precipitation, and runoff) is used, associating weather station data to traffic nodes via nearest-neighbor matching on geographic coordinates. A cross-attention mechanism computes traffic-weather temporal correlations:

$$\mathbf{H}_{fuse}^0 = \text{ReLU}(\text{FC}_h(z \odot \mathbf{H}^0 + (1-z) \odot \mathbf{H}_{wx}))$$

where the gating vector $z = \sigma(\text{FC}_s(\mathbf{H}^0) + \text{FC}_t(\mathbf{H}_{wx}))$ adaptively controls the fusion ratio. A 12-hour weather window ($T_w=12$) is used empirically to capture the persistent effects of weather.

### 3. Spatial Grouping Module — Filtering Local Noise

Feature channels from each ST model layer output are divided into $cg$ channel groups, and nodes are soft-assigned to $sg$ spatial groups via learnable weights. An entropy minimization loss is applied to the assignment weights to encourage approximately one-hot group assignments:

$$\mathcal{L}_{spg} = \frac{1}{N \cdot cg} \sum_{i=1}^{N \cdot cg} \mathcal{H}(\mathbf{W}_s[i,:])$$

This causes each node to primarily belong to one representative group, thereby suppressing location-specific signals and learning generalizable group-level patterns.

### 4. Physics-Informed Module — LWR Traffic Equation Constraint

The LWR traffic flow conservation equation is reformulated in a velocity-only form (avoiding reliance on density data, which is typically unavailable):

$$R = \frac{\partial x}{\partial t} + (2x - x_{fspd}) \frac{\partial x}{\partial l}$$

where $x_{fspd}$ is the free-flow speed. Continuously differentiable spatial embeddings enable automatic differentiation on discrete graphs. The physics loss adopts an adaptive Huber loss: $\mathcal{L}_{phy} = \text{Huber}(R, \delta)$.

### Total Loss Function

$$L = L_{pred} + \lambda L_{cl} + \mu L_{spg} + \theta L_{phy}$$

## Key Experimental Results

### Main Results: Overall Forecasting Performance (Five Real-World Datasets)

| Dataset | Metric | INCREASE | STSM | KITS | **GenCast-L** | Gain |
|--------|:----:|:--------:|:----:|:----:|:----:|:----:|
| PEMS07 | RMSE↓ | 8.399 | 8.390 | 9.574 | **8.253** | 1.64% |
| PEMS08 | MAE↓ | 5.097 | 4.899 | 4.863 | **4.728** | 2.78% |
| PEMS-Bay | MAPE↓ | 0.134 | 0.134 | 0.138 | **0.131** | 2.10% |
| METR-LA | R²↑ | 0.025 | 0.048 | -0.086 | **0.086** | 79.58% |
| Melbourne | R²↑ | -0.042 | 0.027 | -0.165 | **0.061** | 125.56% |

GenCast achieves the best results across all 5 datasets and all 20 metric combinations, with paired t-test $p \ll 10^{-8}$. On the Melbourne dataset, R² improves by as much as 125.56%.

### Ablation Study

| Ablation Variant | Description | Impact |
|----------|------|------|
| w/o-phy | Remove physics constraint | Significant degradation of GenCast-L (frozen embeddings lack physical guidance) |
| w/o-spg | Remove spatial grouping | Consistent degradation across datasets; entropy distribution becomes flatter |
| w/o-wx | Remove weather encoder | Increased prediction error |
| w/o-SE | Remove spatial embedding + physics | Severe degradation |
| w/o-TE | Remove temporal embedding + physics | Large increase in error |

As the unobserved ratio increases from 0.2 to 0.8, GenCast consistently maintains optimal performance, demonstrating robustness to varying degrees of observation missing.

### Domain Generalization Validation

GenCast variants also outperform all baselines on the solar energy NREL dataset, demonstrating cross-domain generalization capability.

## Highlights & Insights

- **First application of PINN to discrete traffic graphs**: The core challenge of non-differentiability of physical equations on graph structures is resolved through continuously differentiable spatial embeddings.
- **Velocity-only reformulation of the LWR equation**: Cleverly avoids the requirement for traffic density data, which is typically unavailable.
- **Novel use of external signals**: Rather than using weather data to detect anomalous events, it is employed as a guidance signal for cross-region generalization.
- **Spatial grouping for local noise filtering**: Entropy minimization-driven soft clustering effectively distinguishes generalizable patterns from location-specific signals.
- **Two complementary spatial embedding strategies**: SE-L suits scenarios with recent data and rich features; SE-H suits scenarios with sparse features or older data.

## Limitations & Future Work

- **Overall low R² values**: Even the best results show R² around 0.2 on most datasets, indicating substantial room for improvement in cross-region prediction accuracy.
- **SE-L is temporally sensitive**: LLM embeddings use the latest OpenStreetMap data, but PEMS-Bay/METR-LA data are relatively old, causing embedding mismatches due to environmental changes.
- **Coarse temporal resolution of weather data**: ERA5 provides hourly data at 9km resolution, which may be insufficient for capturing fine-grained urban short-distance traffic patterns.
- **Physical constraints depend on model assumptions**: The LWR equation is based on the Greenshields density-speed relationship and open-system assumptions, which are frequently violated in real-world traffic scenarios.
- **Substantial computational overhead**: SE-L requires LLaMA3 8B forward inference for embedding generation, and the physics loss requires additional automatic differentiation steps.

## Related Work & Insights

| Dimension | GenCast | STSM |
|------|---------|------|
| Generalization signal sources | Physical laws + dynamic weather + spatial grouping | POI and geographic coordinates only (static) |
| Masking strategy | Random subgraph masking | Similarity-based selective masking |
| Robustness to unobserved ratio | Consistently optimal at 0.2–0.8 | Performance degrades at high ratios |

| Dimension | GenCast | KITS |
|------|---------|------|
| Applicable scenario | Large continuous unobserved regions | Scattered unobserved locations |
| Spatial assumption | No density prior | Implicitly assumes known density distribution |
| External knowledge | Weather + physical constraints | None |

## Rating

- ⭐⭐⭐⭐ **Novelty**: The idea of introducing PINN to discrete traffic graphs is elegant; differentiable spatial embeddings constitute a key technical contribution.
- ⭐⭐⭐⭐ **Technical Depth**: Module designs including LWR velocity reformulation, gated weather fusion, and entropy-regularized spatial grouping are well-motivated and theoretically grounded.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Five datasets, multiple split strategies (horizontal/vertical/ring), varying unobserved ratios, cross-domain transfer, and comprehensive ablations.
- ⭐⭐⭐⭐ **Practical Value**: Open-source code available; hyperparameters require minimal tuning across datasets; extensible to other spatial forecasting domains such as solar energy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](meta_dynamic_graph_for_traffic_flow_prediction.md)
- [\[AAAI 2026\] AI-based Traffic Modeling for Network Security and Privacy: Challenges Ahead](ai-based_traffic_modeling_for_network_security_and_privacy_challenges_ahead.md)
- [\[AAAI 2026\] RAST: A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction](rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[CVPR 2026\] C2T: LLM-Aligned Common-Sense Reward Learning for Traffic-Vehicle Coordination](../../CVPR2026/autonomous_driving/c2t_llm_traffic_coordination.md)

</div>

<!-- RELATED:END -->
