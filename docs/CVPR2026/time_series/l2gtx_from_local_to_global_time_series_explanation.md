---
title: >-
  [Paper Note] L2GTX: From Local to Global Time Series Explanations
description: >-
  [CVPR 2026][Time Series][time series explanation] L2GTX is proposed as a fully model-agnostic local-to-global explanation method for time series, employing parameterized event primitives (increasing/decreasing trends, local extrema) as explanation units. Through hierarchical clustering merging, greedy budget selection, and attribute statistics aggregation, it produces compact and faithful class-level global explanations across 6 UCR datasets (GF = 0.792 on ECG200 with FCN).
tags:
  - CVPR 2026
  - Time Series
  - time series explanation
  - local-to-global aggregation
  - model-agnostic XAI
  - parameterized event primitives
  - representative instance selection
date: 2026-05-08
content_hash: d06b82f5cb59f74a
---

# L2GTX: From Local to Global Time Series Explanations

**Conference**: CVPR 2026
**arXiv**: [2603.13065](https://arxiv.org/abs/2603.13065)
**Code**: N/A
**Area**: Explainable AI / Time Series Classification
**Keywords**: time series explanation, local-to-global aggregation, model-agnostic XAI, parameterized event primitives, representative instance selection

## TL;DR

L2GTX is proposed as a fully model-agnostic local-to-global explanation method for time series, employing parameterized event primitives (increasing/decreasing trends, local extrema) as explanation units. Through hierarchical clustering merging, greedy budget selection, and attribute statistics aggregation, it produces compact and faithful class-level global explanations across 6 UCR datasets (GF = 0.792 on ECG200 with FCN).

## Background & Motivation

**Background**: Deep learning achieves high accuracy in time series classification (finance, sensors, medical ECG), yet operates as a black box, undermining trust and regulatory compliance.

**Limitations of Prior Work**: (1) Image/tabular XAI methods such as LIME/SHAP treat each time step as an independent feature, ignoring temporal dependencies; (2) global explanation synthesis for time series remains largely unexplored; (3) the few existing global methods (CAM/LRP-based) are architecture-specific and lack generality.

**Key Challenge**: The temporal position, duration, and amplitude of time series events vary substantially across instances, so directly aggregating local explanations introduces heavy redundancy and loses temporal structural information.

**Goal**: Generate class-level global explanations for arbitrary black-box time series classifiers while preserving faithfulness and compactness.

**Key Insight**: Parameterized event primitives (PEPs) serve as semantic units, enabling structured local-to-global aggregation via hierarchical clustering merging, greedy selection, and attribute statistics.

**Core Idea**: Replace time-step attributions with event primitives such as "increasing trend / decreasing trend / local extrema," endowing time series explanations with behavioral semantics.

## Method

### Overall Architecture

A five-step pipeline: input $n_{inst}$ instances per class → **Step 1** LOMATCE generates local explanations (event primitives + importance) → **Step 2** hierarchical clustering merges similar event clusters across instances, constructing instance–cluster matrix $\mathbf{M}$ → **Step 3** compute global cluster importance $I_j = \sqrt{\sum_i |M_{i,j}|}$ → **Step 4** greedy selection of $B$ representative instances to maximize coverage → **Step 5** aggregate event attribute statistics (mean ± std) to output class-level global explanations.

### Key Designs

1. **LOMATCE Parameterized Event Primitives (Step 1)**: For each instance, $S$ perturbed neighborhood samples are constructed and four types of PEPs are extracted—increasing segment $(start\_time, duration, avg\_gradient)$, decreasing segment, local maximum $(time, value)$, and local minimum. K-means clustering (with silhouette-based $K$ selection) constructs an event matrix $\mathbf{Z} \in \mathbb{R}^{S \times K}$; a weighted Ridge regression surrogate is trained to obtain cluster importance scores $\hat{\beta}$, from which the top-$n$ clusters are retained. Core motivation: using "event behaviors" rather than "time steps" as explanation units preserves temporal structural semantics—conveying not only *where* is important but also *what behavior* is important.

2. **Adaptive Hierarchical Clustering Merging (Step 2)**: Agglomerative hierarchical clustering (Euclidean distance) is applied to the cluster centroids of all instances, grouped by PEP type. A user-specified merging percentile $p$ determines the cut distance $\tau = \text{percentile}_p(\{d_r\})$. Larger $p$ yields fewer, more compact clusters; after merging, $M_{i,j} = \sum_{C_{i,k} \in G_j} I(C_{i,k})$. Design motivation: similar events across instances exhibit natural redundancy and require a unified representation to support global reasoning.

3. **Greedy Budget Selection (Step 4)**: Given budget $B$, the greedy strategy maximizes marginal gain over uncovered high-importance clusters: $i^* = \arg\max_{i \notin S} \sum_j I_j \cdot \mathbf{1}\{M_{i,j} > 0 \wedge c_j = 0\}$. This adapts the submodular optimization idea of SP-LIME to time series event clusters, ensuring diversity and representativeness among selected instances.

### Loss & Training

L2GTX is a post-hoc explanation method that does not modify the classifier. The primary evaluation metric is Global Faithfulness (GF)—the mean local surrogate $R^2$ across the $B$ selected representative instances. Black-box classifiers (FCN / LSTM-FCN) are trained independently over 100 random splits; L2GTX results are reported with 3 seeds, macro-averaged with 95% confidence intervals.

## Key Experimental Results

### Main Results (FCN Model, Global Faithfulness GF)

| Dataset | p=25 | p=50 | p=75 | p=95 |
|---|---|---|---|---|
| ECG200 | 0.784±0.015 | 0.788±0.013 | 0.780±0.026 | 0.792±0.014 |
| GunPoint | 0.593±0.007 | 0.599±0.019 | 0.601±0.007 | 0.597±0.011 |
| Coffee | 0.683±0.010 | 0.678±0.006 | 0.678±0.005 | 0.678±0.015 |
| FordA | 0.674±0.021 | 0.672±0.029 | 0.673±0.021 | 0.672±0.028 |
| FordB | 0.675±0.008 | 0.679±0.034 | 0.673±0.006 | 0.673±0.029 |
| CBF | 0.625±0.018 | 0.626±0.011 | 0.633±0.016 | 0.625±0.008 |

### Ablation Study (LSTM-FCN Model, GF)

| Dataset | p=25 | p=50 | p=75 | p=95 |
|---|---|---|---|---|
| ECG200 | 0.828±0.010 | 0.832±0.013 | 0.829±0.021 | 0.831±0.007 |
| GunPoint | 0.617±0.074 | 0.619±0.067 | 0.588±0.086 | 0.638±0.011 |
| Coffee | 0.617±0.008 | 0.609±0.004 | 0.616±0.036 | 0.608±0.003 |
| FordA | 0.618±0.028 | 0.621±0.015 | 0.614±0.039 | 0.627±0.035 |
| FordB | 0.661±0.021 | 0.656±0.039 | 0.651±0.050 | 0.655±0.027 |
| CBF | 0.519±0.020 | 0.508±0.025 | 0.519±0.033 | 0.502±0.015 |

### Key Findings

- **GF is highly stable with respect to merging granularity**: GF varies minimally as $p$ ranges from 25 to 95 (confidence intervals strongly overlap), indicating that the explanation space can be substantially compressed without sacrificing faithfulness.
- **Global cluster count decreases monotonically with $p$ without degrading GF**: redundant clusters can be safely merged.
- **Cross-architecture consistency**: FCN and LSTM-FCN yield highly consistent explanation structures on the same datasets (e.g., similar discriminative regions for Normal vs. Infarction in ECG200).
- **Alignment with domain knowledge**: The Infarction class in ECG200 is characterized by local maxima—consistent with the clinical knowledge of prominent deflections in myocardial infarction; the Robusta class in Coffee is characterized by high-intensity spectral peaks.

## Highlights & Insights

- Using parameterized event primitives as explanation units enables a qualitative leap in semantic interpretability—conveying not merely "step 30 is important" but "steps 25–40 exhibit an increasing trend."
- The local-to-global aggregation pipeline is complete and principled: clustering merging → importance estimation → budget selection → attribute statistics.
- The method is fully model-agnostic and applicable to arbitrary black-box time series classifiers.
- The adjustable merging percentile $p$ provides users with fine-grained control over explanation granularity, from detailed to compact.

## Limitations & Future Work

- **Validation limited to univariate time series**: extension to multivariate settings (multi-channel sensors / EEG) has not been explored, limiting practical applicability.
- **GF upper bound is modest**: approximately 0.6 on GunPoint, reflecting the inherent approximation limitations of the Ridge surrogate model.
- **Computational overhead**: LOMATCE event clustering is a bottleneck; cost is high for long sequences or large neighborhoods.
- **Absence of user studies**: no expert evaluation of the subjective utility of the generated explanations has been conducted.

## Related Work & Insights

- **vs. SP-LIME**: borrows the budget selection idea, but SP-LIME targets tabular data, does not perform aggregation, and does not produce class-level summaries.
- **vs. GLocalX**: aggregates local rules for tabular data but does not handle temporal structure.
- **vs. LOMATCE**: serves as the single-instance explanation foundation; L2GTX extends it to the global level.
- **Inspiration**: the local-to-global aggregation paradigm is transferable to explainability in video classification—aggregating frame-level attributions into class-level global video explanations.

## Rating

⭐⭐⭐ (3/5)

**Rationale**: The research problem (global explanation for time series) has clear value; the methodological pipeline is complete and principled; and the event primitive design carries meaningful semantics. However, (1) individual components offer limited novelty in isolation; (2) evaluation is restricted to small UCR datasets; (3) absolute GF values are not high (0.5–0.6 in some cases); and (4) no human evaluation is provided. Recommended primarily for readers specializing in XAI subfields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)
- [\[CVPR 2026\] PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning](pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi.md)
- [\[CVPR 2026\] Competition-Aware CPC Forecasting with Near-Market Coverage](competition-aware_cpc_forecasting_with_near-market_coverage.md)
- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [\[CVPR 2026\] Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)

</div>

<!-- RELATED:END -->
