---
title: >-
  [Paper Note] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment
description: >-
  [Time Series] This paper proposes TSRating, a framework that leverages LLMs to perform pairwise quality comparisons of time series (TS) data segments across four dimensions—trend, frequency, amplitude…
tags:
  - "Time Series"
date: 2026-05-08
content_hash: f24213c3d48fad56
---

# Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment

## TL;DR

This paper proposes TSRating, a framework that leverages LLMs to perform pairwise quality comparisons of time series (TS) data segments across four dimensions—trend, frequency, amplitude, and pattern. Pairwise judgments are converted to scalar quality scores via the Bradley-Terry model. A TSRater model (MOMENT encoder + MLP) is then trained using MAML meta-learning across 9 domains and 22 subsets, enabling efficient and unified cross-domain TS data quality assessment.

## Background & Motivation

**Importance of TS data quality**: Whether fine-tuning LLMs for TS tasks or training TS foundation models from scratch, data quality is a critical performance bottleneck. Real-world data frequently suffers from missing values, sensor failures, and irregular sampling.

**Domain limitations of existing methods**: TimeInf adapts influence functions to TS data, and TimeShap extends Shapley values to TS—yet both operate within a single domain, neglecting the fact that real-world TS data spans vastly different domains such as healthcare, finance, meteorology, and industry.

**Root cause of computational inefficiency**: Influence functions require expensive Hessian and gradient computations, while Shapley values incur exponential combinatorial costs. Both families of methods struggle to balance efficiency and accuracy, and per-domain recomputation further renders them prohibitive.

**Inspiration from LLM-based text quality assessment**: Works such as Qurating and Ask-LLM have demonstrated that LLMs can accurately assess text quality via prompting. LLMs have acquired rich cross-domain TS knowledge through large-scale pretraining—motivating the question of whether this capability can be transferred to TS quality assessment.

**Core scientific questions**: Do LLMs genuinely understand the key characteristics of TS quality? How can LLMs be effectively guided to distinguish high- from low-quality TS data? How can LLM judgment be efficiently distilled into a lightweight model for large-scale deployment?

**Goal**: To construct a unified cross-domain TS data quality assessment framework that simultaneously addresses accuracy, efficiency, and generalization.

## Method

### Overall Architecture

TSRating consists of three stages: (1) **LLM quality judgment**: sliding-window segmentation → pairwise comparison → Bradley-Terry scalar conversion; (2) **TSRater training**: MOMENT encoding → MLP mapping → BCE loss; (3) **Meta-learning generalization**: MAML + signSGD trained across 9 domains. At inference time, only a single TSRater forward pass is required for efficient scoring.

### Key Designs

**Design 1: LLM Pairwise Quality Judgment and Bradley-Terry Scalar Conversion**

- **Mechanism**: TS samples are divided into overlapping segments via a sliding window. For each pair $\mathbf{B}_i, \mathbf{B}_j$, an LLM performs pairwise comparisons across 4 dimensions (trend, frequency, amplitude, pattern). Confidence is estimated over $M$ comparisons:

$$p_{i \succ j} = \frac{1}{M} \sum_{k=1}^{M} m_{i \succ j}^{(k)}$$

- **Bradley-Terry conversion**: Pairwise preferences are converted to scalar scores $s(\mathbf{B}_i)$:

$$p_{i \succ j} = \sigma(s(\mathbf{B}_i) - s(\mathbf{B}_j))$$

solved via maximum likelihood estimation:

$$\mathcal{P} = \sum_{(\mathbf{B}_i, \mathbf{B}_j, p_{i \succ j}) \in \mathcal{J}} \left[ p_{i \succ j} \log \sigma(s(\mathbf{B}_i) - s(\mathbf{B}_j)) + (1-p_{i \succ j}) \log \sigma(s(\mathbf{B}_j) - s(\mathbf{B}_i)) \right]$$

- **Multivariate handling**: Per-channel scoring followed by averaging: $s(\mathbf{B}_i) = \frac{1}{D} \sum_{d=1}^{D} s(\mathbf{B}_i^d)$
- **Debiasing**: $\mathbf{B}_i$ and $\mathbf{B}_j$ are swapped in the prompt and results are averaged to eliminate positional bias.
- **Validation**: Recognition accuracy on synthetic data reaches 94.5% (trend), 92.25% (frequency), 98.75% (amplitude), and 95.75% (pattern).

**Design 2: TSRater Model Architecture**

- **Representation learning**: MOMENT (~109M parameters), a frozen TS foundation model encoder, is used to extract temporal features.
- **Quality mapping**: A 3-layer MLP (hidden dimension 256, LayerNorm + ReLU + residual connections) outputs a scalar quality score.
- **Training loss**: Binary cross-entropy loss aligns the model with LLM pairwise judgments:

$$\mathcal{L}_\theta = \mathbb{E}_{(\mathbf{B}_i, \mathbf{B}_j, p_{i \succ j}) \in \mathcal{J}} \left[ -p_{i \succ j} \log \sigma(s_\theta(\mathbf{B}_i) - s_\theta(\mathbf{B}_j)) - (1-p_{i \succ j}) \log \sigma(s_\theta(\mathbf{B}_j) - s_\theta(\mathbf{B}_i)) \right]$$

- **Design Motivation**: LLM pairwise comparison incurs $O(n^2)$ API call costs, whereas TSRater inference requires only a single forward pass.

**Design 3: MAML Meta-learning for Cross-domain Training**

- **Task construction**: 22 subsets from 9 domains (energy, retail, finance, healthcare, traffic, meteorology, industry, synthetic, and others) are drawn from the Time-300B corpus.
- **Training objective**:

$$\min_\theta \sum_{\mathcal{T}_i \sim \mathcal{T}} \mathcal{L}_{\mathcal{T}_i}^{\text{query}} \left( \theta - \alpha \cdot \text{sign}(\nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{support}}(\theta)) \right)$$

- **signSGD acceleration**: The inner loop replaces standard gradient descent with signSGD, updating only gradient signs and naturally avoiding second-order hypergradient computations.
- **Score aggregation**: Scores from 4 dimensions are individually normalized and then aggregated into a final quality score.

### Hierarchical Score Aggregation

Segment-level → point-level: $s(x_i) = \frac{1}{|B(x)|} \sum_{\mathbf{B}_k \in B(x)} s(\mathbf{B}_k)$

Point-level → sample-level: $s(\mathbf{S}) = \frac{1}{T} \sum_{i=1}^{T} s(x_i)$

## Experimental Setup

- **Tasks**: Long-term forecasting (4 datasets), short-term forecasting (3 M4 subsets), classification (4 datasets) — 11 benchmarks in total.
- **Models**: Linear, CNN, PatchTST, as well as TimeMixer, DLinear, and iTransformer in extended evaluations.
- **Baselines**: DataShapley, KNNShapley, DataOob, TimeInf.
- **Strategy**: TSRating selects the top-50% highest-quality samples for training; performance is evaluated on the test set.

## Key Experimental Results

### Table 1: Main Results (3 Tasks × 3 Models × 11 Datasets)

| Model | Method | LT-Forecast RMSE (avg. 4) | ST-Forecast MAPE (avg. 3) | Classification Acc. (avg. 4) |
|-------|--------|--------------------------|--------------------------|------------------------------|
| Linear | Random | 0.900 | 1.528 | 0.291 |
| Linear | TimeInf | 0.722 | 1.536 | 0.323 |
| Linear | **TSRating** | **0.740** | **1.366** | **0.344** |
| CNN | Random | 1.085 | 1.550 | 0.449 |
| CNN | TimeInf | 1.077 | 1.503 | 0.455 |
| CNN | **TSRating** | **1.026** | **1.322** | **0.494** |
| PatchTST | Random | 0.366 | 2.725 | 0.408 |
| PatchTST | TimeInf | 0.374 | 2.690 | 0.406 |
| PatchTST | **TSRating** | **0.357** | **2.574** | **0.444** |

TSRating achieves the best or second-best results across a significantly larger proportion of the 36 evaluation cases.

### Table 2: Runtime Comparison

| Method | Time (seconds) | Notes |
|--------|---------------|-------|
| DataShapley | 210,000 | Extremely slow |
| KNNShapley | 152 | Fast but low accuracy |
| DataOob | 4,785 | — |
| TimeInf | 4,938 | — |
| **TSRater (total)** | **4,687** | Includes LLM judgment + meta-training + fine-tuning + inference |
| — New dataset inference | **~200** | Meta-trained model is reusable |

Key advantage: The meta-trained TSRater model is reusable; new datasets require only few-shot fine-tuning and inference (~200 seconds).

### Table 3: Meta-learning Generalization Ablation (Electricity Dataset)

| Method | Linear | CNN | PatchTST | iTransformer | TimeMixer |
|--------|--------|-----|----------|-------------|-----------|
| Meta-rater | 1.390 | 1.511 | 0.397 | 0.300 | 0.345 |
| In-domain training | 1.471 | 1.497 | 0.398 | 0.306 | 0.332 |
| Out-of-domain training | 1.556 | 1.602 | 0.418 | 0.310 | 0.382 |

Meta-rater matches or surpasses in-domain training while substantially outperforming direct out-of-domain transfer.

## Key Findings

1. **LLMs genuinely understand TS quality**: Judgment accuracy across 4 dimensions reaches 92–99% on synthetic data, and visualizations on real data are consistent with human intuition. This constitutes the first systematic validation of this capability.

2. **Quality assessment is generalizable**: Meta-rater matches or exceeds domain-specific raters with only few-shot adaptation on unseen datasets, while direct out-of-domain transfer degrades significantly—validating the necessity of meta-learning.

3. **High-quality data outperforms full data**: Fine-tuning TS foundation models (Time-MoE, Time-LLM, MOMENT) on the top-50% data selected by TSRating yields lower MSE than training on the full dataset—quality surpasses quantity.

4. **Data pruning validates assessment quality**: Removing samples in descending quality order causes performance to degrade fastest under TSRating (PatchTST on Traffic shows RMSE increase >0.03 after removing the top 40%, versus 0.01–0.02 for KNNShapley).

5. **Multi-dimensional fusion outperforms individual dimensions**: Ablation studies show that individual dimensions exhibit unstable performance across datasets (e.g., amplitude is best on Weather but worst on Traffic); four-dimension fusion achieves consistent performance across all datasets.

6. **Encoder choice has limited impact**: MOMENT, Chronos, and TimeGPT yield comparable performance on the Weather dataset, indicating that TSRater's gains are primarily attributable to LLM supervision rather than the encoder.

## Highlights & Insights

- **First systematic validation of LLM-as-judge for TS quality**: The LLM-as-judge paradigm from NLP is innovatively transferred to the TS domain. The key contribution lies in a carefully designed 4-dimensional prompt that covers fundamental TS properties (trend, frequency, amplitude, pattern).
- **Knowledge distillation paradigm**: LLM pairwise judgment → Bradley-Terry scalar conversion → MLP distillation transfers expensive LLM judgment capability into a lightweight model with extremely efficient inference. New dataset inference requires only ~200 seconds.
- **Elegant application of signSGD**: Replacing standard gradient descent with signSGD in the MAML inner loop—updating only gradient signs—naturally circumvents second-order hypergradient computations and reduces the computational overhead of meta-learning.

## Limitations & Future Work

- LLM judgment quality is model-dependent—different LLMs (GPT-4o-mini vs. Claude vs. Gemini) yield varying judgments; while experiments suggest the differences are minor, consistency cannot be guaranteed.
- Whether the four dimensions (trend/frequency/amplitude/pattern) exhaustively cover all TS quality aspects is debatable—factors such as distributional shift and label noise are not addressed.
- The frozen MOMENT encoder constitutes an implicit performance ceiling for TSRater.
- Evaluation focuses primarily on forecasting and classification; anomaly detection is only briefly addressed in the appendix.
- The top-50% selection threshold is fixed—adaptive threshold strategies are not explored.

## Related Work & Insights

### vs. TimeInf (Zhang et al., 2024b)
TimeInf adapts influence functions to TS data to preserve temporal dependencies, but (1) requires per-domain Hessian and gradient computation from scratch (~4,938 seconds); (2) is effective only within a single domain and requires recomputation for cross-domain use; (3) exhibits significant accuracy degradation when transferred to new domains. TSRating achieves ~200-second adaptation to new domains via meta-learning pretraining with more stable cross-domain accuracy.

### vs. DataShapley (Ghorbani & Zou, 2019)
DataShapley uses cooperative game theory to assess individual sample contributions. While theoretically principled, it (1) incurs enormous computational costs (~210,000 seconds); (2) does not account for temporal characteristics; (3) requires per-dataset recomputation with no reusability. TSRating is more than 45× faster and incorporates built-in temporal understanding.

### vs. Qurating (Wettig et al., 2024)
Qurating pioneered the LLM-as-judge paradigm for data quality assessment, but is limited to text data (dimensions such as writing style and factual accuracy). TSRating extends this paradigm to the TS domain, with core innovations including (1) TS-specific 4-dimensional prompts; (2) pairwise-to-scalar conversion via the Bradley-Terry model; and (3) cross-domain generalization via meta-learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic framework for LLM-based TS quality assessment; innovatively transfers the LLM-as-judge paradigm from NLP to the TS domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 11 datasets × 3 tasks × multiple models + foundation model fine-tuning + data pruning + ablations across encoders, dimensions, and LLMs.
- **Writing Quality**: ⭐⭐⭐⭐ — Framework description is clear, validation is rigorous, and mathematical notation is well-defined.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to TS data curation and foundation model fine-tuning; reusable meta-trained model lowers deployment barriers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](tsrating_time_series_quality_llm.md)
- [\[AAAI 2026\] Finding Time Series Anomalies using Granular-ball Vector Data Description](../../AAAI2026/time_series/finding_time_series_anomalies_using_granular-ball_vector_data_description.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/time_series/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)

</div>

<!-- RELATED:END -->
