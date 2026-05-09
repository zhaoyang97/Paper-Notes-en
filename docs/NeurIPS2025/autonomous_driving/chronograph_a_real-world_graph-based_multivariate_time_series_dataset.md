---
title: >-
  [Paper Note] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset
description: >-
  [NEURIPS2025][Autonomous Driving][microservice telemetry] This paper presents ChronoGraph — the first real-world microservice dataset that simultaneously provides multivariate time series, explicit service dependency graphs, and event-level anomaly labels (6 months / ~700 services / 5-dimensional metrics / 8005 timesteps). Benchmark results reveal substantial room for improvement in long-horizon forecasting and topology-aware modeling among existing methods.
tags:
  - NEURIPS2025
  - Autonomous Driving
  - microservice telemetry
  - graph time series
  - anomaly detection
  - real-world dataset
  - service dependency graph
date: 2026-05-08
content_hash: aeeabe363ab128a2
---

# ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset

**Conference**: NEURIPS2025  
**arXiv**: [2509.04449](https://arxiv.org/abs/2509.04449)  
**Code**: [https://github.com/bit-ml/ChronoGraph](https://github.com/bit-ml/ChronoGraph)  
**Area**: Autonomous Driving  
**Keywords**: microservice telemetry, graph time series, anomaly detection, real-world dataset, service dependency graph  

## TL;DR
This paper presents ChronoGraph — the first real-world microservice dataset that simultaneously provides multivariate time series, explicit service dependency graphs, and event-level anomaly labels (6 months / ~700 services / 5-dimensional metrics / 8005 timesteps). Benchmark results reveal substantial room for improvement in long-horizon forecasting and topology-aware modeling among existing methods.

## Background & Motivation

**Background**: In large-scale microservice systems, forecasting short-to-medium-term evolution of service metrics is critical for alerting, auto-scaling, and capacity planning. Existing graph-based time series benchmark datasets are predominantly drawn from traffic (e.g., METR-LA) and air quality domains, and have been widely adopted in time series forecasting research.

**Limitations of Prior Work**:
   - Traffic and air quality datasets are univariate and lack anomaly annotations;
   - Industrial control datasets such as SWaT and WADI provide anomaly labels and are multivariate, but only supply process diagrams rather than true adjacency matrices;
   - No existing dataset simultaneously provides **multivariate time series + explicit dependency graphs + anomaly labels**.

**Key Challenge**: The absence of real graph structures forces existing forecasting and anomaly detection methods to either process each series independently (topology-agnostic) or learn dense implicit graphs (e.g., fully connected attention, top-$k$ similarity graphs). Such data-driven graphs may be inconsistent with the true service topology.

**Goal**:
   - Provide a multivariate time series dataset with a real service dependency graph, enabling the community to evaluate topology-aware methods;
   - Annotate real operational incidents to support anomaly detection evaluation on genuine failures;
   - Identify, through benchmarking, the dimensions along which existing methods fall short.

**Key Insight**: Six months of production telemetry from a large enterprise microservice platform, comprising ~700 service nodes, inter-service call edges, and 17 manually annotated anomaly segments.

**Core Idea**: Construct the first benchmark that jointly provides multivariate time series, a real service dependency graph, and anomaly labels within a single dataset, filling a critical data gap in graph-aware temporal modeling.

## Method

### Overall Architecture

ChronoGraph is a **dataset and benchmark** contribution rather than a new model. The overall pipeline consists of:
- **Data Collection**: System-level metrics are collected at 30-minute intervals from all services on a production microservice platform.
- **Graph Construction**: A directed dependency graph is built from actual inter-service call relationships.
- **Anomaly Annotation**: Affected services and time windows are extracted from internal incident reports.
- **Benchmark Evaluation**: Six categories of methods are evaluated on forecasting and anomaly detection tasks.

### Key Designs

1. **Dataset Composition**:

    - Function: Provide a real-world graph-structured multivariate time series dataset.
    - Core Details: 708 service nodes, each with a 5-dimensional time series (CPU utilization, memory usage, memory working set, network ingress, network egress), totaling 8005 timesteps at 30-minute granularity (~6 months). Edges represent inter-service call dependencies, each carrying 8-dimensional features (request count, return codes, latency, etc.).
    - Design Motivation: Existing datasets either lack a true graph structure or lack anomaly annotations; this dataset is the first to provide all three simultaneously.

2. **Anomaly Annotation Pipeline**:

    - Function: Provide anomaly labels aligned with real operational incidents.
    - Mechanism: Internal incident reports written by engineers are parsed to extract affected service names and timestamps, which are then mapped to fixed-length windows centered on the report time, yielding 17 labeled anomaly segments.
    - Design Motivation: Conventional anomaly detection evaluation relies on synthetic anomalies or rule-based injection; this work provides labels derived from genuine failure events.

3. **Evaluation Protocol**:

    - Function: Evaluate multiple method families on forecasting and anomaly detection tasks.
    - Mechanism: A 60/40 train-test split is adopted. Forecasting performance is measured with MAE, MSE, and MASE; anomaly detection is evaluated with $F1_K$-AUC and $ROC_K$-AUC, which overcome the over-optimism of traditional point-adjustment (PA).
    - Design Motivation: The conventional F1 + PA paradigm substantially overestimates anomaly detection performance; $F1_K$-AUC integrates over varying $K$ ratios to provide a more balanced segment-level evaluation.

### Baseline Coverage

Three major categories of methods are included:
- **Statistical Models**: Prophet (trend + seasonality decomposition)
- **Time Series Foundation Models**: Chronos-Bolt Base (zero-shot/few-shot), TabPFN-TS (transformer-based prior-data-fitted network)
- **Anomaly Detectors**: Autoencoder (reconstruction error), Isolation Forest, OC-SVM (one-class SVM)
- **Ensemble Methods**: Combination of Prophet, Isolation Forest, and Autoencoder

## Key Experimental Results

### Main Results — Forecasting Performance

| Model | MAE (full 3202 steps) | MSE (full) | MASE (full) | MAE (first 500 steps) | MSE (first 500 steps) | MASE (first 500 steps) |
|------|-------------------|------------|-------------|---------------|---------------|----------------|
| Prophet | **0.125**±0.067 | **0.044**±0.054 | 7.182±11.21 | 0.069±0.044 | 0.013±0.022 | 3.143±3.663 |
| Chronos | 0.150±0.173 | 0.343±2.426 | 7.902±12.71 | **0.044**±0.030 | **0.007**±0.015 | **1.938**±1.731 |
| TabPFN-TS | **0.125**±0.125 | 0.089±1.172 | **6.205**±9.315 | 0.109±0.061 | 0.026±0.031 | 5.082±11.01 |

### Main Results — Anomaly Detection Performance

| Method | $F1_K$ ↑ | $ROC_K$ ↑ | FP Rate ↓ | FN Rate ↓ | F1 ↑ |
|------|----------|-----------|---------|---------|------|
| Prophet | **20.57** | **62.97** | 2.02 | 97.98 | 2.39 |
| Isolation Forest | 17.49 | 56.39 | 46.9 | 50.48 | 7.08 |
| OC-SVM | 14.46 | 54.31 | 22.13 | 77.08 | 5.50 |
| Autoencoder | 13.86 | 59.79 | 0.38 | 99.58 | 0.72 |
| TabPFN-TS | 12.37 | 54.08 | 0.55 | 99.79 | 0.31 |
| Chronos | 12.41 | 49.78 | 2.49 | 97.84 | 2.49 |
| Ensemble* | 16.92 | 60.95 | **0.20** | 99.58 | 0.73 |

### Key Findings

- **Large gap between short-term and long-term forecasting**: Chronos achieves the best performance over the first 500 steps (MAE 0.044) but degrades substantially over the full 3202-step horizon (MAE 0.150), indicating that current methods cannot sustain long-horizon accuracy. TabPFN-TS shows the most consistent performance across both horizons.
- **All anomaly detection methods perform poorly**: The best $F1_K$ is only 20.57 (Prophet), and all methods exhibit extremely high FN rates (97.98% for Prophet), demonstrating that topology-agnostic anomaly detection is far from practical in microservice settings.
- **Spatial clustering of anomalies**: Predicted anomalies from the ensemble tend to cluster around densely connected service regions in the graph, suggesting that failures propagate along the dependency graph — a direct motivation for topology-aware methods.
- **Limitations of foundation models**: Chronos and TabPFN-TS, as per-series models, are unable to capture cross-node propagation effects, resulting in the worst anomaly detection performance.

## Highlights & Insights

- **First "three-in-one" real-world dataset**: Simultaneously provides multivariate time series, an explicit dependency graph, and anomaly labels. This combination fills a critical gap in graph-aware temporal modeling research, where practitioners previously had to validate forecasting and anomaly detection on separate datasets.
- **$F1_K$-AUC evaluation protocol**: The use of metrics integrated over varying $K$ values avoids the performance overestimation caused by traditional point-adjustment. This protocol is broadly applicable to other time series anomaly detection benchmarks.
- **Empirical evidence for anomaly propagation**: Figure 1 visually demonstrates the spatial clustering of predicted anomalies in the graph, providing compelling empirical support for topology-aware anomaly detection.
- **Value of dark data**: The paper notes that the dataset may contain transient anomalous behaviors that were never escalated to formal incidents and are currently counted as false positives — yet carry operational significance. This observation offers methodological insights for anomaly detection evaluation.

## Limitations & Future Work

- **Sparse annotations**: Only 17 anomaly segments are provided, covering only escalated service failures; numerous transient or self-recovering anomalies remain unlabeled, limiting the statistical significance of anomaly detection evaluation.
- **All baselines are topology-agnostic**: No method that genuinely leverages graph structure (e.g., GNN-based forecasting or anomaly detection) is evaluated, making it impossible to quantify the benefit of topology awareness.
- **Single data source**: The data originates from one enterprise's microservice platform; generalizability is unknown, as microservice architectures vary considerably across organizations.
- **Coarse temporal resolution**: The 30-minute interval may miss fine-grained propagation details of rapid failures.
- **Future Directions**:
    - Evaluate graph neural networks (e.g., DCRNN, MTGNN, StemGNN) on this dataset for joint spatial-temporal forecasting.
    - Model anomaly propagation along the dependency graph for root cause analysis.
    - Leverage edge features (8-dimensional communication data) for edge-conditioned graph convolution.

## Related Work & Insights

- **vs. SWaT/WADI**: Industrial control datasets provide anomaly labels and are multivariate, but only supply process diagrams without true adjacency matrices. ChronoGraph provides a machine-readable service dependency graph directly usable by graph models.
- **vs. METR-LA / PEMS-BAY**: Traffic datasets include spatial graphs but are univariate (speed only) and lack anomaly annotations. ChronoGraph provides 5 dimensions per node and 8 dimensions per edge, along with event labels.
- **vs. Chronos / TabPFN-TS and other foundation models**: These models are designed for per-series zero-shot forecasting and cannot exploit graph structure. This dataset directly exposes their deficiencies in structure-aware settings.

## Rating

- Novelty: ⭐⭐⭐⭐ First real-world multivariate time series dataset to simultaneously include an explicit service dependency graph and anomaly labels, filling a critical data gap.
- Experimental Thoroughness: ⭐⭐⭐ Baselines are comprehensive (statistical models + foundation models + classical AD methods), but no topology-aware method is evaluated.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed data descriptions and candid discussion of limitations.
- Value: ⭐⭐⭐⭐ As a benchmark dataset, it offers lasting value for graph-aware time series research, though future work is needed to build genuine graph-based methods on top of it.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DBLoss: Decomposition-based Loss Function for Time Series Forecasting](dbloss_decomposition-based_loss_function_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [\[ICCV 2025\] LookOut: Real-World Humanoid Egocentric Navigation](../../ICCV2025/autonomous_driving/lookout_real-world_humanoid_egocentric_navigation.md)
- [\[ICCV 2025\] RTMap: Real-Time Recursive Mapping with Change Detection and Localization](../../ICCV2025/autonomous_driving/rtmap_real-time_recursive_mapping_with_change_detection_and_localization.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)

<!-- RELATED:END -->
