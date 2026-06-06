---
title: >-
  [Paper Note] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series
description: >-
  [ICML 2026][Time Series][Time series forecasting] By **detecting distribution drift via MMD** and dynamically expanding a heterogeneous expert pool…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time series forecasting"
  - "dynamic experts"
  - "distribution drift detection"
  - "mixture of experts"
  - "non-stationary data"
date: 2026-05-08
content_hash: 26dd4af7d5c4b6a9
---

# Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series

**Conference**: ICML 2026  
**arXiv**: [2605.20678](https://arxiv.org/abs/2605.20678)  
**Code**: TBD  
**Area**: Time Series / Mixture of Experts  
**Keywords**: Time series forecasting, dynamic experts, distribution drift detection, mixture of experts, non-stationary data

## TL;DR
By **detecting distribution drift via MMD** and dynamically expanding a heterogeneous expert pool, combined with a **temporal memory router** to ensure selection consistency, Dynamic-TMoE achieves a new SOTA on nine time series benchmarks—reducing MSE by 10.4% and MAE by 7.8% on average compared to all baselines.

## Background & Motivation

**Background**: Time series forecasting is a cornerstone for critical decision-making systems ranging from energy management to healthcare monitoring. However, real-world time series are inherently non-stationary, characterized by continuous distribution drift and evolving temporal dependencies.

**Limitations of Prior Work**: Existing methods primarily fall into two categories: input-level normalization (RevIN, SAN, IN-Flow), which maps non-stationary inputs to stable distributions through a remove-predict-restore paradigm but often discards non-stationary signals crucial for prediction; and in-model adaptation (Non-stationary Transformer, Koopa, TimeStacker), which redesigns attention or utilizes physical/spectral dynamics to capture evolving features, though monolithic architectures lack modularity. Recent MoE methods (TFPS, Time-MoE) employ static expert pools and memoryless routing, failing to adapt to abrupt distribution drifts.

**Key Challenge**: MoE frameworks face two inherent problems—(1) **Temporal Rigidity**: Fixed expert pools cannot accommodate new patterns from severe distribution drift, and memoryless gateways ignore temporal connectivity, leading to unstable selection; (2) **Insufficient Specialization**: Homogeneous experts lack functional diversity to decouple different drift components (trend vs. seasonality).

**Goal**: Design an MoE framework that adapts to evolving distributions while maintaining selection consistency.

**Key Insight**: (1) Distribution drift can be quantified via MMD to trigger expert pool evolution; (2) Temporal continuity can be maintained through a GRU router with historical memory; (3) Heterogeneous expert designs can specialize in different temporal patterns.

**Core Idea**: Dynamically expand and prune a heterogeneous expert pool during the learning phase (based on drift detection), while ensuring context-aware stable selection using a temporal memory router supported by an anomalous state bank.

## Method

### Overall Architecture
"Perception-Decision-Adaptation" closed loop—Patch Embedding $\rightarrow$ Distribution Drift Detector (Perception) $\rightarrow$ Temporal Memory Router (Decision) $\rightarrow$ Evolvable Expert Manager (Adaptation) $\rightarrow$ Heterogeneous Expert Pool (Base experts: Identity, Trend, Seasonality, Volatility + Drift experts instantiated on demand).

### Key Designs

1. **MMD Drift Detection + Adaptive Threshold**:

    - **Function**: Continuously monitor distribution drift to trigger expert expansion.
    - **Mechanism**: Employs a kernel method (RBF kernel) to define the MMD distance between the reference window and the current window $\mathcal{D}_{\text{mmd}}^2 = \frac{1}{N_r^2} \sum_{i, j} k(x_i^{\text{ref}}, x_j^{\text{ref}}) - \frac{2}{N_r N_c} \sum_{i, j} k(x_i^{\text{ref}}, x_j) + \frac{1}{N_c^2} \sum_{i, j} k(x_i, x_j)$. To handle time-varying noise levels, a dynamic threshold $\epsilon = \mu_\mathcal{H} + \lambda \sigma_\mathcal{H}$ (k-sigma rule) is used; the expert manager is triggered when $\mathcal{D}_{\text{mmd}}^2 > \epsilon$.
    - **Design Motivation**: Fixed thresholds cannot adapt to fluctuating noise levels; dynamic thresholds provide robust drift detection. Generalization bounds prove that increasing MMD directly raises the upper bound of target prediction risk.

2. **Temporal Memory Router + Anomalous State Bank**:

    - **Function**: Achieves context-aware stable expert selection over the evolving pool and accelerates adaptation to recurring drifts.
    - **Mechanism**: A GRU maintains the hidden state $\mathbf{h}_t = \text{GRU}(\phi(\mathbf{x}_{p, t}), \mathbf{h}_{t-1})$. Projecting the hidden state yields logits for each expert, activating the top-k experts via a sparse gateway. The anomalous state bank $\mathcal{A}$ stores hidden states during drift detection; during routing, relevant historical states are retrieved via cosine similarity and fused with the current state $\tilde{\mathbf{h}}_t = \alpha \mathbf{h}_t + (1 - \alpha) \mathbf{h}_{\text{ref}}$ using a learnable gateway $\alpha$.
    - **Design Motivation**: Memoryless routing causes unstable selection between adjacent patches; GRU maintains sequential consistency; the anomaly bank accelerates adaptation to periodic drifts to avoid cold starts.

3. **Heterogeneous Expert Design + Drift Pattern Analysis**:

    - **Function**: Specializes in different temporal components (trend, seasonality, volatility) using diverse expert architectures and intelligently selects new expert types via a drift pattern analyzer.
    - **Mechanism**: Identity expert $E_{\text{id}} = \text{Linear}(\mathbf{X}_p)$; Trend expert $E_{\text{trend}} = \text{MLP}(\text{AvgPool}(\mathbf{X}_p))$; Seasonality expert processes in the frequency domain via FFT-MLP-iFFT $\mathbf{Z} = \text{iFFT}(\text{MLP}(\text{FFT}(\mathbf{X}_p)))$; Volatility expert uses causal convolution + gated linear units. After drift detection, the residual is analyzed to calculate trend/seasonality/volatility scores, automatically instantiating the best-fitting expert type. New experts are fine-tuned while freezing other parameters to avoid catastrophic forgetting.
    - **Design Motivation**: Homogeneous experts cannot decouple complex patterns; heterogeneous design provides different inductive biases; drift analysis avoids blind expansion; cold-start alignment ensures stable integration.

## Key Experimental Results

### Main Results

| Dataset | Metric | **Ours** | TFPS | ST-MTM | RAFT | Gain vs TFPS |
|--------|------|-------------|------|--------|------|------------|
| Weather | MSE | **0.240** | 0.241 | 0.262 | 0.271 | ↓0.4% |
| Exchange | MSE | **0.351** | 0.395 | 0.408 | 0.432 | ↓11.1% |
| ETTh1 | MSE | **0.429** | 0.448 | 0.432 | 0.432 | ↓4.2% |
| Electricity | MSE | **0.170** | 0.183 | 0.208 | 0.184 | ↓7.1% |
| ILI | MSE | **1.981** | 2.642 | 2.820 | 5.916 | ↓25.0% |

Ranked top 2 in 16 out of 18 evaluation metrics (1st place 11 times, 2nd place 5 times).

### Ablation Study

| Configuration | Drift-Aware | Temporal Memory | Anomaly Bank | ETTh1 MSE | Weather MSE |
|------|--------|--------|-------|-----------|-------------|
| ① Full | ✓ | GRU | ✓ | **0.429** | **0.240** |
| ② w/o Drift | ✗ | GRU | ✓ | 0.436 | 0.246 |
| ③ Linear Router | ✓ | Linear | ✓ | 0.436 | 0.247 |
| ④ MLP Router | ✓ | MLP | ✓ | 0.438 | 0.245 |
| ⑤ w/o Anomaly Bank | ✓ | GRU | ✗ | 0.438 | 0.246 |
| Hetero. → Homo. | ✓ | GRU | ✓ | 0.440 | 0.246 |

### Key Findings
- Both drift awareness and temporal continuity are necessary for handling non-stationarity; removing either leads to significant degradation.
- GRU routing improves performance by 1.6%-2.1% compared to static routing—sequential memory is crucial for expert allocation in non-stationary environments.
- Heterogeneous experts improve results by 2.5% over homogeneous designs—expert diversity effectively decouples complex distributions.
- Relational layers (recurrent correlation modeling) improve performance by 2.8%.
- The anomalous state bank contribution is relatively small (< 1%), but stability improves significantly on data with periodic drifts.

## Highlights & Insights
- **Trinity of Dynamic-Heterogeneous-Memory**: The framework unifies three dimensions—architectural evolution (dynamic experts), functional diversity (heterogeneous design), and temporal consistency (memory routing). Compared to existing works focusing on only one dimension, this combination addresses the core contradictions of non-stationary forecasting.
- **MMD as a Drift Signal**: Compared to manual thresholds or simple residual monitoring, kernel methods better capture complex distribution changes. The dynamic threshold (k-sigma) avoids over-sensitivity.
- **Engineering Wisdom in Cold-Start Alignment**: Freezing existing parameters and fine-tuning new experts only on drift data cleverly balances the integration of new and old knowledge while avoiding catastrophic forgetting.
- **Empirical Design for Seasonality**: The FFT-MLP-iFFT approach with periodic activation serves as a reusable pattern for time series decomposition.

## Limitations & Future Work
- Computational overhead is not fully discussed: the impact of MMD calculation, GRU inference, and adding experts during drift on large-scale data or long sequences lacks specific metrics.
- Incomplete hyperparameter sensitivity: Analysis for the drift threshold $\lambda$ and fusion coefficient $\alpha$ is only provided in the supplementary material.
- Generalization boundary conditions: Theoretical characterization of which drift patterns or magnitudes the framework remains effective for is not yet clear.
- Improvements: Online learning paradigms; adaptive hyperparameters; analyzing memory usage and indexing efficiency for ultra-long sequences.

## Related Work & Insights
- **vs RevIN / IN-Flow**: These handle distribution invariance but discard non-stationary signals; Dynamic-TMoE retains and utilizes these signals internally via heterogeneous experts.
- **vs Non-stationary Transformer**: Monolithic architectures struggle to specialize in multiple coexisting patterns; this work decomposes complexity via MoE.
- **vs TFPS / Time-MoE**: These use static homogeneous experts and stateless routing; Dynamic-TMoE introduces dynamic heterogeneous pools and temporal memory.
- **vs Koopa / DERITS**: These use global frequency processing; ours processes seasonality in the frequency domain then refines in the time domain, offering a more flexible hybrid strategy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Extends MoE from static to dynamic, homogeneous to heterogeneous, and stateless to temporal memory routing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 datasets + multiple baselines + full ablation + multiple prediction lengths; evidence is sufficient.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and detailed methodology; rich charts (though some details are in the appendix).
- Value: ⭐⭐⭐⭐⭐ Addresses core issues of non-stationary time series (drift and continuity), highly valuable for industrial applications (finance, energy, healthcare).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](../../AAAI2026/time_series/towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[AAAI 2026\] Task-Aware Retrieval Augmentation for Dynamic Recommendation](../../AAAI2026/time_series/task-aware_retrieval_augmentation_for_dynamic_recommendation.md)
- [\[AAAI 2026\] LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs](../../AAAI2026/time_series/loretta_a_low_resource_framework_to_poison_continuous_time_dynamic_graphs.md)
- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)

</div>

<!-- RELATED:END -->
