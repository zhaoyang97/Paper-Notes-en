---
title: >-
  [Paper Note] TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment
description: >-
  [ICLR 2026][Time Series][time series quality assessment] TSRating leverages the prior knowledge of LLMs to conduct pairwise quality judgments of time series data chunks across four dimensions—trend, frequency, amplitude…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "time series quality assessment"
  - "LLM judgment"
  - "meta-learning"
  - "data selection"
  - "Bradley-Terry model"
date: 2026-05-08
content_hash: 38e3240f06ebbd64
---

# TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment

**Conference**: ICLR 2026
**arXiv**: [2506.01290](https://arxiv.org/abs/2506.01290)  
**Code**: [https://github.com/clsr1008/TSRating](https://github.com/clsr1008/TSRating)  
**Area**: Time Series
**Keywords**: time series quality assessment, LLM judgment, meta-learning, data selection, Bradley-Terry model

## TL;DR

TSRating leverages the prior knowledge of LLMs to conduct pairwise quality judgments of time series data chunks across four dimensions—trend, frequency, amplitude, and pattern—converts these comparisons into scalar scores via the Bradley-Terry model, and trains a cross-domain generalizable TSRater via meta-learning, enabling efficient and accurate time series data quality assessment.

## Background & Motivation

**Background**: High-quality time series data is critical for model performance. Existing data quality assessment methods are primarily based on influence functions and Shapley values. While effective within a single domain, these approaches incur high computational costs (Hessian computation or exponential combinatorial overhead) and overlook the fact that real-world time series data originates from highly diverse domains.

**Limitations of Prior Work**: Influence functions require computationally intensive Hessian and gradient operations, while Shapley values face exponential computational costs. More critically, these methods are typically effective only within a single domain and generalize poorly to diverse time series data across domains.

**Key Challenge**: There is a need for a time series quality assessment method that is both cross-domain generalizable and computationally efficient. Traditional methods struggle to simultaneously achieve estimation fidelity and computational efficiency.

**Goal**: (1) Verify whether LLMs can understand and judge the quality of diverse time series data; (2) train a lightweight scoring model, TSRater, to replace costly LLM inference; (3) achieve cross-domain generalization through meta-learning.

**Key Insight**: LLMs have demonstrated strong capabilities in text quality assessment (e.g., Qurating, Ask-LLM), and the rich knowledge accumulated during pre-training may extend to the understanding of time series data. The authors verify that LLMs can indeed distinguish time series quality along the dimensions of trend, frequency, amplitude, and pattern with 92–99% accuracy.

**Core Idea**: LLMs serve as "teachers" performing pairwise comparisons of time series chunk quality; the Bradley-Terry model converts comparison results into scalar scores; meta-learning then trains a lightweight TSRater for efficient inference on new domains.

## Method

### Overall Architecture

Input time series → segment into chunks via sliding window → LLM pairwise quality judgment across four dimensions → Bradley-Terry MLE to generate scalar scores → scores assigned back to point-level and sample-level → TSRater trained on these annotations → TSRater meta-trained across 9 domains → TSRater deployed for rapid quality scoring.

### Key Designs

1. **LLM Pairwise Quality Judgment**:

    - Function: Leverages LLM prior knowledge to assess the relative quality of pairs of time series data chunks.
    - Mechanism: Prompt templates are designed for four evaluation criteria (trend, frequency, amplitude, pattern). For each pair of chunks $B_i$ and $B_j$, the LLM determines which is of higher quality; confidence $p_{i \succ j}$ is estimated via repeated sampling. To reduce positional bias, the positions of the two chunks are swapped and results are averaged. For multivariate series, judgments are made per channel and then averaged. The Bradley-Terry model $p_{i \succ j} = \sigma(s(B_i) - s(B_j))$ is then applied via MLE to convert pairwise preferences into scalar scores.
    - Design Motivation: Pairwise comparison is more stable and reliable than absolute scoring; the Bradley-Terry model provides a theoretically grounded scalar conversion.

2. **TSRater Scoring Model**:

    - Function: Distills LLM quality judgments into a lightweight model for efficient inference.
    - Mechanism: A pre-trained MOMENT encoder (~109M parameters) is used as a frozen time series feature extractor, followed by a 3-layer MLP (hidden dimension 256, with LayerNorm, ReLU, and residual connections) that maps embeddings to scalar quality scores. The training objective is a binary cross-entropy loss $\mathcal{L}_\theta$ aligned with LLM pairwise preference judgments.
    - Design Motivation: LLM inference is costly; once TSRater is trained, only lightweight inference is required to evaluate large volumes of data, amortizing the annotation cost effectively.

3. **Meta-learning for Cross-domain Training**:

    - Function: Enables TSRater to rapidly adapt to unseen new domains.
    - Mechanism: Based on the MAML strategy, meta-tasks are constructed from 22 data subsets across 9 major domains drawn from the Time-300B corpus. At each episode, tasks are sampled; inner-loop updates are performed on the support set using signSGD (avoiding second-order derivative computation), and the meta-parameters are updated based on query set loss. The objective is $\min_\theta \sum_{\mathcal{T}_i} \mathcal{L}^{query}_{\mathcal{T}_i}(\theta - \alpha \cdot \text{sign}(\nabla_\theta \mathcal{L}^{support}_{\mathcal{T}_i}(\theta)))$.
    - Design Motivation: signSGD replaces standard gradients to avoid higher-order derivative computation for meta-gradients, substantially reducing meta-learning training costs.

### Loss & Training

TSRater employs binary cross-entropy loss to align with the pairwise preferences of the Bradley-Terry model. In the meta-learning phase, signSGD is used as the inner-loop optimizer, while standard gradient descent is applied in the outer loop. A separate TSRater is trained for each quality dimension, and the normalized scores from all four dimensions are fused to produce the final quality assessment.

## Key Experimental Results

### Main Results

| Dataset (Long-term Forecasting RMSE) | Random | DataShapley | KNNShapley | TimeInf | TSRating |
|--------------------------------------|--------|-------------|------------|---------|----------|
| Electricity (Linear) | 1.601 | 1.580 | 1.325 | 1.391 | **1.390** |
| Weather (Linear) | 0.665 | 0.638 | 0.625 | 0.616 | **0.611** |
| Traffic (Linear) | 0.979 | 0.956 | 0.696 | **0.609** | 0.683 |
| ExRate (Linear) | 0.356 | 0.323 | 0.290 | 0.272 | 0.275 |

TSRating outperforms or matches methods based on Shapley values and influence functions in the majority of settings.

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| LLM synthesis validation | Trend/Frequency/Amplitude/Pattern accuracy: 94.5%/92.25%/98.75%/95.75% |
| Remove meta-learning | Significant drop in cross-domain generalization performance |
| Remove signSGD | Increased training time with comparable performance |
| Data pruning experiment | Removing high-quality samples selected by TSRating leads to significant performance degradation |

### Key Findings

- LLM quality judgment accuracy on synthetic data reaches 92–99%, validating the ability of LLMs to understand time series quality.
- TSRating is also effective for fine-tuning time series foundation models: fine-tuning on high-quality subsets significantly improves generalization performance.
- Meta-learning enables TSRater to adapt to new domains with very few fine-tuning steps, confirming cross-domain generalization capability.

## Highlights & Insights

- The idea of **using LLMs as quality judges for time series** is highly novel—it bypasses the computational bottleneck of traditional methods by reformulating the problem as a pairwise comparison task well-suited to LLMs.
- The **four-dimensional evaluation criteria** (trend, frequency, amplitude, pattern) are well-motivated, cover the core characteristics of time series, and are supported by classical literature.
- **Replacing standard gradients with signSGD** for the inner-loop of meta-learning is a practical engineering technique transferable to other meta-learning scenarios to reduce computational cost.

## Limitations & Future Work

- The reliability of LLM judgments depends on prompt design and the capability of the LLM itself; accuracy may degrade for extreme or rare time series patterns.
- Whether the four evaluation dimensions comprehensively cover quality characteristics across all domains is debatable (e.g., outlier density and signal-to-noise ratio are not included).
- TSRater relies on the frozen MOMENT encoder, and the encoder's domain coverage directly affects cross-domain performance.
- Experiments select the top 50% of data for training—the optimal selection ratio may vary across tasks.

## Related Work & Insights

- **vs. TimeInf (Zhang et al., 2024)**: TimeInf uses time-aware influence functions, which are more precise but computationally expensive and limited to a single domain; TSRating achieves cross-domain generalization via LLM and meta-learning.
- **vs. DataShapley/KNNShapley**: Shapley-based methods are theoretically rigorous but face exponentially growing computation; TSRating incurs near-zero inference cost after training.
- **vs. Qurating (Wettig et al., 2024)**: Qurating uses LLMs to assess text quality; TSRating extends this idea to the time series domain for the first time.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of LLM judgment to time series quality assessment; highly novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 11 datasets, 3 task types, and multiple baselines; experiments are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly described with intuitive framework diagrams.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for time series data management with practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)

</div>

<!-- RELATED:END -->
