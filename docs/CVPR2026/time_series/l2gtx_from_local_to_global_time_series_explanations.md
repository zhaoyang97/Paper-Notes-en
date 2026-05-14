---
title: >-
  [Paper Note] L2GTX: From Local to Global Time Series Explanations
description: >-
  [CVPR 2026][Time Series][Time series explainability] L2GTX proposes a fully model-agnostic local-to-global explanation framework for time series classification. It extracts parameterized temporal event primitives (PEPs)—…
tags:
  - "CVPR 2026"
  - "Time Series"
  - "Time series explainability"
  - "global explanation"
  - "parameterized event primitives"
  - "model-agnostic"
  - "local-to-global aggregation"
date: 2026-05-08
content_hash: 9fbd706a2f1032e0
---

# L2GTX: From Local to Global Time Series Explanations

**Conference**: CVPR 2026
**arXiv**: [2603.13065](https://arxiv.org/abs/2603.13065)
**Code**: None
**Area**: Time Series
**Keywords**: Time series explainability, global explanation, parameterized event primitives, model-agnostic, local-to-global aggregation

## TL;DR

L2GTX proposes a fully model-agnostic local-to-global explanation framework for time series classification. It extracts parameterized temporal event primitives (PEPs)—trends and extrema—from LOMATCE local explanations, merges redundant clusters across instances via hierarchical clustering, selects representative instances through submodular optimization, and aggregates these into concise class-level global explanations. The method maintains stable global faithfulness across six time series classification datasets.

## Background & Motivation

**Background**: Deep learning has achieved high accuracy in time series classification, with broad applications in finance, sensor monitoring, and healthcare. However, these models are inherently black boxes that map input sequences directly to predictions, offering no interpretability of the underlying decision rationale.

**Limitations of Prior Work**: Existing XAI methods face three critical limitations: (i) model-agnostic methods designed for images and tabular data (e.g., LIME/SHAP) do not transfer well to time series due to strong temporal dependencies and non-i.i.d. observations; (ii) global explanation synthesis for time series is severely underexplored—most methods provide only local explanations that highlight important timesteps or subsequences for individual predictions; (iii) the few existing global methods are typically tied to specific model architectures (e.g., CAM or LRP), precluding architecture-neutral interpretability.

**Key Challenge**: Local explanations can only account for individual instance predictions and fail to reveal the model's systematic decision behavior at the class level. Extracting global features directly from model internals, on the other hand, is architecture-dependent. A general method is needed that neither relies on model internals nor sacrifices the ability to synthesize class-level understanding from local temporal patterns.

**Goal**: (a) How to obtain high-quality local time series explanations without accessing model internals? (b) How to merge similar temporal events across instances to reduce redundancy? (c) How to select the most representative instances under a limited budget? (d) How to aggregate local events into concise class-level global explanations?

**Key Insight**: The authors observe that LOMATCE local explanations already provide semantically rich local descriptions in the form of PEPs—capturing temporal behaviors such as "increasing trend," "decreasing trend," "local maximum," and "local minimum." These primitives are more human-interpretable than raw timestep importance scores and support structured cross-instance comparison and merging.

**Core Idea**: Merge parameterized event primitives across instances via hierarchical clustering, select representative instances that maximize coverage through submodular optimization, and aggregate local events into class-level global time series explanations.

## Method

### Overall Architecture

L2GTX takes as input a trained black-box time series classifier $f$ and a dataset $\mathcal{X}$, and outputs a global explanation per class in the form of statistical summaries over parameterized event primitives. The pipeline consists of five sequential steps:

1. **Local Attribution**: Apply LOMATCE to sampled instances to generate local explanations (PEP clusters + importance scores).
2. **Cluster Merging**: Use hierarchical clustering to merge similar PEP clusters across instances.
3. **Global Importance**: Construct an instance–cluster matrix and compute the importance of each global cluster.
4. **Instance Selection**: Greedily select representative instances covering the most important clusters under a budget constraint.
5. **Event Aggregation**: Summarize event attributes of selected instances to produce class-level global explanations.

To ensure class balance, L2GTX samples $n_{\text{inst}}=15$ instances per class for small/medium datasets and $n_{\text{inst}}=30$ for large datasets.

### Key Designs

1. **LOMATCE Local Attribution (Step 1)**:

    - **Function**: Generate PEP-based local explanations for each instance $X_i$.
    - **Mechanism**: Construct a local neighborhood of $S$ perturbed samples for each instance by randomly masking temporal segments. Extract four PEP types from all neighborhood samples—increasing trend (parameters: start_time, duration, avg_gradient), decreasing trend (same parameters), local maximum (parameters: time, value), and local minimum (same parameters). Independently apply K-means clustering to each PEP type, with $K$ determined automatically via silhouette score. Construct an event matrix $\mathbf{Z}_i \in \mathbb{R}^{S \times K}$, train a weighted ridge regression surrogate to obtain cluster importances $\hat{\beta}_i \in \mathbb{R}^K$, and retain the top-$n$ clusters.
    - **Design Motivation**: Replacing raw timestep importance with parameterized event primitives provides semantic "why" explanations—describing trends and extrema as human-interpretable temporal behaviors rather than merely indicating "where" importance lies.

2. **Hierarchical Cluster Merging and Instance–Cluster Matrix (Steps 2–3)**:

    - **Function**: Merge similar PEP clusters across instances to construct a global perspective.
    - **Mechanism**: Apply agglomerative hierarchical clustering to all cluster centroids of the same PEP type, cutting the dendrogram at a distance threshold determined by the user-specified merging percentile $p$ to obtain global clusters $\mathcal{G}_e$. Construct an instance–cluster matrix $\mathbf{M} \in \mathbb{R}^{N \times |\mathcal{G}|}$ where $M_{i,j} = \sum_{C_{i,k} \in G_j} I(C_{i,k})$. Global importance follows the SP-LIME strategy: $I_j = \sqrt{\sum_{i=1}^N |M_{i,j}|}$.
    - **Design Motivation**: Local PEP clusters differ across instances and must be "aligned" before cross-instance comparison is possible. Hierarchical clustering offers flexible granularity control—larger $p$ yields fewer, more compact global clusters.

3. **Submodular Optimization for Instance Selection (Step 4)**:

    - **Function**: Select the most representative set of instances under a budget constraint $B$.
    - **Mechanism**: Greedily select instances that maximize the weighted coverage of uncovered clusters. The coverage vector is updated after each selection to ensure the chosen set maximizes coverage of the most important global clusters.
    - **Design Motivation**: Aggregating all instances introduces redundancy and noise. Inspired by SP-LIME, submodular optimization enables a small number of instances to cover the most important global clusters, ensuring conciseness and representativeness.

4. **Event Aggregation and Global Explanation Generation (Step 5)**:

    - **Function**: Summarize PEP events from selected instances into class-level statistical descriptions.
    - **Mechanism**: Remove the local cluster hierarchy and directly assign all events to their corresponding global clusters. Compute mean and standard deviation for each event attribute. Trend-type events are described by statistics over (start_time, duration) to characterize temporal extent; extremum-type events are described by statistics over (time, value) to characterize location and magnitude.

### Loss & Training

L2GTX is a post-hoc explanation method and does not involve end-to-end training. The core evaluation metric is **Global Faithfulness** (GF), defined as the mean local surrogate fidelity over selected instances:

$$\text{GF}(\mathcal{S}) = \frac{1}{|\mathcal{S}|} \sum_{x_i \in \mathcal{S}} F(x_i)$$

where $F(x_i)$ is the $R^2$ score of the local ridge regression surrogate for instance $x_i$. All experiments are repeated with three random seeds; macro-averaged GF and 95% confidence intervals are reported.

## Key Experimental Results

### Main Results

Evaluated on six UCR time series datasets using two architectures, FCN and LSTM-FCN:

| Dataset | Model | GF (p=25) | GF (p=50) | GF (p=75) | GF (p=95) |
|--------|------|-----------|-----------|-----------|-----------|
| ECG200 | FCN | 0.784 | 0.788 | 0.780 | 0.792 |
| GunPoint | FCN | 0.593 | 0.599 | 0.601 | 0.597 |
| Coffee | FCN | 0.683 | 0.678 | 0.678 | 0.678 |
| FordA | FCN | 0.674 | 0.672 | 0.673 | 0.672 |
| FordB | FCN | 0.675 | 0.679 | 0.673 | 0.673 |
| CBF | FCN | 0.625 | 0.626 | 0.633 | 0.625 |
| ECG200 | LSTM-FCN | 0.828 | 0.832 | 0.829 | 0.831 |
| FordB | LSTM-FCN | 0.661 | 0.656 | 0.651 | 0.655 |
| CBF | LSTM-FCN | 0.519 | 0.508 | 0.519 | 0.502 |

### Ablation Study

| Configuration | Key Metric | Observation |
|------|---------|------|
| Merging percentile p=25 to 95 | GF stable, overlapping CIs | Aggressive compression does not sacrifice faithfulness |
| Increasing p | Number of global clusters decreases monotonically | More compact explanation space |
| FCN vs. LSTM-FCN | Both yield high importance in overlapping regions | Method captures architecture-agnostic decision cues |
| ECG200 case study | Normal vs. Infarction consistent with medical knowledge | Infarction signal dominated by a small number of salient deflections |
| Coffee case study | Robusta: high-amplitude maxima vs. Arabica: low-amplitude | Consistent with coffee spectroscopy literature |

### Key Findings

- **Cluster merging does not degrade faithfulness**: GF remains stable as $p$ increases from 25 to 95, with overlapping confidence intervals.
- **Cross-architecture consistency**: FCN and LSTM-FCN produce structurally consistent explanations that share common decision-relevant temporal cues.
- **Case studies align with domain knowledge**: The infarction class in ECG200 is characterized by prominent deflections; Robusta coffee is dominated by high-intensity local maxima.
- **Lower GF on CBF with LSTM-FCN** (approximately 0.5) likely reflects the approximation limitations of the local linear surrogate.

## Highlights & Insights

- **First fully model-agnostic local-to-global explanation method for time series.** The approach does not rely on any model internals and is applicable to arbitrary black-box time series classifiers, extending model-agnosticism to the global level.
- **Parameterized event primitives enable semantic explanations.** Describing time series patterns via trends and extrema is more meaningful than reporting "timestep $t$ is important," and naturally supports cross-instance alignment and domain semantic mapping.
- **Greedy submodular optimization balances coverage and budget.** A small number of selected instances suffices to cover the most important global clusters.
- **Merging percentile provides tunable granularity.** A single parameter $p$ allows users to control explanation compactness without sacrificing faithfulness.

## Limitations & Future Work

- **Computational cost**: LOMATCE event clustering is the computational bottleneck, particularly for long time series.
- **Univariate only**: Extension to multivariate settings requires handling cross-channel interactions.
- **No human-centered evaluation**: Domain expert subjective assessment is absent.
- **Low GF on some datasets**: CBF achieves approximately 0.5 and GunPoint approximately 0.6, constrained by the local linear surrogate.
- **No quantitative comparison with other global explanation methods**.

## Related Work & Insights

- **vs. SP-LIME**: Selects representative instances but does not aggregate them. L2GTX adds cross-instance merging and global statistical aggregation.
- **vs. GLocalX**: Performs local-to-global aggregation for tabular data. L2GTX adapts this paradigm to the parameterized event structure of time series.
- **vs. LOMATCE**: Serves as the local explanation foundation of L2GTX. The contribution lies in providing a systematic local-to-global pipeline.
- **vs. CAM/LRP family**: These methods depend on model internals and are architecture-specific. L2GTX is more general but can only infer decision rationale indirectly.

## Rating

- **Novelty**: ⭐⭐⭐ Local-to-global aggregation is a novel attempt in time series XAI, though individual components lack methodological breakthroughs.
- **Experimental Thoroughness**: ⭐⭐⭐ Six datasets, two models, and multiple percentile settings are evaluated, but quantitative comparison with other global methods is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear, formulations are complete, and case studies are convincing.
- **Value**: ⭐⭐⭐ Addresses a gap in global time series interpretability, though the discussion of application scenarios lacks depth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[NeurIPS 2025\] Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions](../../NeurIPS2025/time_series/scalable_signature_kernel_computations_for_long_time_series_via_local_neumann_se.md)
- [\[NeurIPS 2025\] DemandCast: Global hourly electricity demand forecasting](../../NeurIPS2025/time_series/demandcast_global_hourly_electricity_demand_forecasting.md)
- [\[AAAI 2026\] Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints](../../AAAI2026/time_series/mitigating_error_accumulation_in_co-speech_motion_generation_via_global_rotation.md)

</div>

<!-- RELATED:END -->
