---
title: >-
  [Paper Note] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks
description: >-
  [ICML 2026][Time Series][Time Series Forecasting] TIME is a next-generation benchmark designed for **Time Series Foundation Models (TSFMs)**. It addresses four major pain points of existing benchmarks—data reuse…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Forecasting"
  - "Foundation Models"
  - "Zero-shot Evaluation"
  - "Benchmark Design"
  - "Pattern-level Evaluation"
date: 2026-05-08
content_hash: 1ad4b02c6c61a2d5
---

# It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks

**Conference**: ICML 2026  
**arXiv**: [2602.12147](https://arxiv.org/abs/2602.12147)  
**Code**: TBD  
**Area**: Time Series / Benchmark Design  
**Keywords**: Time Series Forecasting, Foundation Models, Zero-shot Evaluation, Benchmark Design, Pattern-level Evaluation

## TL;DR
TIME is a next-generation benchmark designed for **Time Series Foundation Models (TSFMs)**. It addresses four major pain points of existing benchmarks—data reuse, quality issues, improper task configurations, and low evaluation granularity—through **human annotation + LLM-driven data cleaning**, **context-aligned task design**, and a **pattern-level evaluation perspective**. It includes 50 brand-new datasets, 98 tasks, and evaluations of 12 TSFMs.

## Background & Motivation

**Background**: The emergence of TSFMs has shifted the forecasting evaluation paradigm from dataset-centric to task-centric. Existing benchmarks such as Monash and LSF have been widely adopted but are increasingly revealing core limitations.

**Limitations of Prior Work**:
- **Data Reuse and Leakage Risks**: Existing benchmarks rely heavily on repeated data, leading to pollution risks; new-generation large-scale pre-trained models may have already ingested this legacy data.
- **Data Quality Flaws**: Insufficient automated processing and a lack of rigorous quality assurance result in issues like outlier explosions, excessive missing values, and constant sequences.
- **Task Configurations Detached from Reality**: Traditional benchmarks often use a "one-size-fits-all" strategy (e.g., a fixed 720-step forecasting horizon), completely ignoring differences in application scenarios, frequencies, and predictability.
- **Coarse-grained Evaluation Perspectives**: Performance is typically aggregated by static meta-labels (e.g., dataset or frequency), which hides insights into how models perform across similar patterns in different datasets.

**Key Challenge**: TSFMs require universal evaluation across heterogeneous data, yet current benchmarks lack sufficiently fresh data, appropriate task designs, and fine-grained analytical capabilities.

**Goal**: To construct TIME (Task-centric Benchmark for Universal Forecasting Models), upgrading benchmarks across three dimensions: data, tasks, and evaluation.

**Key Insight**: Utilize LLMs combined with human professional judgment to achieve high-fidelity data annotation and task design, and apply cluster analysis based on interpretable time series features rather than static labels.

**Core Idea**: Transform benchmark design from "dataset collection + mechanical evaluation" into a three-layer progression: "fresh data + human annotation + pattern-driven analysis."

## Method

### Overall Architecture
The framework consists of four stages: (1) Data collection and cleaning: 50 brand-new datasets acquired from government portals, industrial collaborations, open-source libraries, and competitions; (2) Context-aligned task design: Customized horizons based on real-world application contexts rather than fixed lengths; (3) Interpretable feature design: 7-dimensional structural features extracted via STL decomposition; (4) Rolling window evaluation and pattern stratification.

### Key Designs

1. **Human-in-the-loop Data Cleaning Pipeline**:
    - **Function**: High-quality transformation of raw data into benchmark-ready data.
    - **Mechanism**: A five-step automated process (timestamp repair → rule validation → statistical testing → outlier elimination → correlation check) followed by final human decision-making. Humans review quality summary reports, combining domain knowledge and LLM insights to distinguish between real data corruption and valid domain-specific features.
    - **Design Motivation**: While automation is efficient for scaling, it struggles with contextual truth; the human "final mile" ensures semantic correctness.

2. **Context-Aligned Task Configuration**:
    - **Function**: Shifts forecasting from "fixed horizons" to "application-driven" tasks.
    - **Mechanism**: Customizes the horizon $H$ and test length $L_{\text{test}}$ based on the application background and operational constraints of each dataset. High-frequency data are assigned three horizons (short/medium/long), while low-frequency or sample-limited data use a single feasible operational horizon. Test windows cover complete seasonal cycles, and tasks are validated using LLMs and domain expertise.
    - **Design Motivation**: A 720-step setting may be meaningless for low-frequency data but too short for high-frequency data; context alignment ensures evaluation results map to real decision scenarios.

3. **Pattern-Level Interpretable Evaluation Perspective**:
    - **Function**: Transitions from dataset-level to pattern-level aggregation.
    - **Mechanism**: Each variable undergoes STL decomposition $\mathbf{x} = T + S + R$ to extract 7 features (trend strength, linearity, seasonality strength, seasonal correlation, residual ACF, complexity, and stationarity). Each feature $F_k$ is binary-encoded based on the benchmark-wide median $\tilde{F}_k$, resulting in a 7-bit binary code $\mathbf{B} \in \{0, 1\}^7$. Pattern-specific performance is then calculated using scale-invariant MASE and CRPS metrics across matching variables.
    - **Design Motivation**: Dataset-level aggregation often obscures underlying patterns. Pattern-level analysis reveals a model's true generalization strengths, such as which models handle strong seasonality or complex noise more effectively.

### Loss & Training
The benchmark adopts a rolling window evaluation protocol. Metrics include MASE (point forecasting) and CRPS (probabilistic forecasting). A relativized evaluation framework is used by normalizing metrics against a Seasonal Naive (S-Naive) baseline: $\text{Metric}_{\text{model}}^{\text{norm}}(u) = \frac{\text{Metric}_{\text{model}}(u)}{\text{Metric}_{\text{S-Naive}}(u)}$. Normalized metrics are aggregated across units using the geometric mean rather than the arithmetic mean.

## Key Experimental Results

### Benchmark Scale and Model Coverage

| Metric | Value | Description |
|-----|------|------|
| Number of Datasets | 50 | Entirely new collections |
| Number of Tasks | 98 | Combinations of frequencies and horizons |
| Evaluated Models | 12 | Covers Decoders, Encoder-Decoders, etc. |
| Application Domains | 8 | Finance, System Metrics, Energy, Transport, etc. |

### Main Results

| Model | Release Date | Architecture | Parameters | MASE | CRPS | Rating |
|-----|---------|------|------|------|------|------|
| Chronos-2 | 10-25 | Enc. | 120M | 0.645 | 0.421 | ⭐⭐⭐ |
| TimesFM-2.5 | 10-25 | Dec. | 200M | 0.648 | 0.425 | ⭐⭐⭐ |
| TiRex | 05-25 | xLSTM | 35M | 0.672 | 0.438 | ⭐⭐⭐ |
| Moirai-2 | 08-25 | Dec. | 11M | 0.698 | 0.455 | ⭐⭐⭐⭐ |
| TimesFM-2.0 | 12-24 | Dec. | 500M | 0.741 | 0.489 | ⭐⭐⭐⭐ |

The latest model iterations consistently outperform predecessors, validating the benchmark's discriminative power.

### Pattern-level Key Findings

| Time Series Feature | Inter-model Divergence | Key Insight |
|-----------|-------------|---------|
| Trend Strength | High | Chronos-2 / TimesFM-2.5 show more significant gains on strong trends. |
| Seasonality Strength | Medium | Models perform similarly on weak seasonality but diverge on strong seasonality. |
| Seasonal Correlation | Medium-High | Early models showed a large gap between stable and unstable seasonality; newer TSFMs narrow this gap. |
| Stationarity | High | Most models gain more on non-stationary data, but rankings are sensitive to stationarity. |
| Complexity | High | Performance converges on high-complexity data, while top models diverge on low-complexity data. |

### Key Findings
- Task-level rankings diverge from feature-level rankings, suggesting that aggregation granularity significantly influences conclusions.
- Time series can exhibit different patterns at different granularities (e.g., a global spike may appear periodic locally).
- Models generally predict obvious seasonality and trends accurately but tend toward conservative predictions on high-volatility sequences, a failure often hidden by pure metric rankings.

## Highlights & Insights
- **Data Freshness**: 50 new datasets from four sources ensure data has not been ingested during pre-training, preventing contamination.
- **Three-layer Quality Assurance**: The progression from automation to summary reports to human decision-making efficiently handles large scales while closing automation blind spots.
- **Context-driven Design**: Moving away from "one-size-fits-all" allows horizons and test lengths to reflect real operational needs, shifting the evaluation philosophy toward practical application.
- **Pattern-level Paradigm Shift**: STL-based dynamic feature clustering enables unified analysis of similar patterns across disparate domains while maintaining interpretability.
- **Symmetric Aggregation**: The use of relativized metrics and geometric means prevents extreme outliers in specific tasks from skewing overall rankings.

## Limitations & Future Work
- While significant, 50 datasets offer limited coverage compared to benchmarks containing tens of thousands of series.
- Pattern-level analysis currently relies on single features; complex multi-feature interactions require interactive exploration on the leaderboard.
- Visualization reveals conservative prediction issues, but quantitative "reliability" metrics are still needed.
- MASE/CRPS are standard but may not perfectly align with specific application-layer loss functions.
- Future work: Joint multi-feature pattern clustering, application-customized metrics, continuous data expansion, and temporal evolution analysis.

## Related Work & Insights
- **vs M4 / LSF**: Earlier benchmarks rely heavily on legacy data and fixed settings; TIME uses fresh sources and context-aligned tasks for fairer assessment.
- **vs Recent Large-scale Benchmarks**: While recent work has increased scale, many still reuse historical data; TIME’s innovations directly target these data lineage issues.
- **vs Feature Analysis**: Previous works used features for classification or visualization; TIME elevates this by using features for **evaluation aggregation**, moving from descriptive to normative analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Addressing data, annotation, and evaluation simultaneously solves major legacy benchmark issues.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 TSFMs across 98 tasks and multiple grains of analysis provide a robust overview.
- Writing Quality: ⭐⭐⭐⭐ High logic and informative visuals; however, some LLM prompt details are missing from the main text.
- Value: ⭐⭐⭐⭐⭐ Provides a contamination-proof, reality-aligned benchmark for the TSFM community; the pattern-level analysis serves as a diagnostic tool for model improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TimeOmni-VL: Unified Models for Time Series Understanding and Generation](timeomni-vl_unified_models_for_time_series_understanding_and_generation.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICML 2026\] From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_llm.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
