---
title: >-
  [Paper Note] Synthetic Series-Symbol Data Generation for Time Series Foundation Models
description: >-
  [NeurIPS 2025][Time Series][time series foundation model] This paper proposes the Series-Symbol (S²) data generation mechanism and SymTime, a dual-modality foundation model. Grounded in Takens' theorem and symbolic dynam…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "time series foundation model"
  - "synthetic data generation"
  - "symbolic expressions"
  - "contrastive learning"
  - "pre-training"
date: 2026-05-08
content_hash: f0fd57bd6ea47fb8
---

# Synthetic Series-Symbol Data Generation for Time Series Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.08445](https://arxiv.org/abs/2510.08445)  
**Code**: [GitHub](https://github.com/wwhenxuan/SymTime)  
**Area**: Time Series
**Keywords**: time series foundation model, synthetic data generation, symbolic expressions, contrastive learning, pre-training

## TL;DR
This paper proposes the Series-Symbol (S²) data generation mechanism and SymTime, a dual-modality foundation model. Grounded in Takens' theorem and symbolic dynamics theory, the framework generates unlimited synthetic time series–symbol paired data (40M pairs / 50B tokens). Through cross-modal contrastive pre-training, SymTime achieves performance competitive with models pre-trained on real data across five time series tasks.

## Background & Motivation

**Background**: Time series foundation models (e.g., Moirai, Timer, TimeGPT) have made significant progress in recent years. However, compared to CV and NLP, the time series domain faces severe data scarcity and distributional imbalance. Existing large-scale time series datasets remain insufficient in domains such as finance and healthcare, and their scale is far smaller than benchmarks like ImageNet or WebText.

**Limitations of Prior Work**: According to neural scaling laws, imbalanced training data degrades out-of-distribution generalization, leading to performance bias. Current time series pre-training strategies predominantly rely on real data collection, facing dual bottlenecks of data privacy constraints and incomplete domain coverage. Among the few methods that use synthetic data (e.g., Chronos), the generation strategies lack a theoretical characterization of the intrinsic nature of time series.

**Core Idea**: Drawing on Takens' theorem (time series are low-dimensional projections of complex dynamical systems) and symbolic dynamics theory (complex systems can be abstractly represented by symbolic expressions), this work constructs a theory-driven synthetic data generation mechanism. Diverse symbolic expressions are randomly constructed to cover a broad range of dynamical system types, and the resulting time series data naturally exhibit rich temporal properties and semantic correspondences.

## Method

### Overall Architecture
The framework consists of two main components: (1) the Series-Symbol (S²) data generation mechanism, which constructs symbolic expressions and generates time series–symbol paired data via forward propagation at unlimited scale; and (2) the SymTime foundation model, comprising a time series encoder and a symbol encoder, pre-trained via masked modeling and cross-modal contrastive learning, followed by fine-tuning on downstream tasks.

### Key Designs

1. **S² Data Generation Mechanism**:

    - **Function**: Generates unlimited high-quality synthetic time series data alongside their corresponding symbolic expressions.
    - **Mechanism**: Multivariate symbolic expressions $f(\cdot)$ are constructed by randomly sampling binary trees — first selecting binary operators ($+, -, \times$) to build the tree skeleton, then inserting variables and constants into leaf nodes, adding unary operators (sin, cos, log, exp, pow2, etc.), and finally applying affine transformations to increase diversity. The input $X$ is sampled from a mixture distribution and ARMA processes; the output series is obtained via forward propagation $Y = f(X)$.
    - **Design Motivation**: Grounded in Takens' theorem and symbolic dynamics theory, there exists a rigorous semantic correspondence between symbolic expressions and time series. By enumerating all input/output dimension combinations ($M \in [1,6]$, $N \in [1,12]$), the generated data covers the full time series representation space. A total of 40M paired samples are generated, comprising 50B tokens.
    - **Novelty**: Unlike ForecastPFN and Chronos, whose synthetic strategies lack a theoretical characterization of time series generation, S² is more closely aligned with the generative mechanisms underlying time series.

2. **SymTime Dual-Modality Pre-training Architecture**:

    - **Function**: Enhances time series representation learning by incorporating symbolic information.
    - **Mechanism**: A time series encoder (6-layer Transformer) reconstructs masked patches via Masked Time Series Modeling (MTM); a symbol encoder (6-layer DistilBERT) learns symbolic representations via Masked Language Modeling (MLM). The two encoders are aligned in representation space through cross-modal contrastive learning using a MoCo-style momentum encoder, bringing semantically related time series–symbol pairs closer together.
    - **Design Motivation**: Pure MTM pre-training captures only statistical patterns in time series and cannot encode the underlying dynamical semantics. Cross-modal contrastive learning injects symbolic semantic information into the time series encoder, endowing it with a unique inductive bias.

3. **Momentum Distillation**:

    - **Function**: Aligns the encoder output on masked data with that of the momentum encoder, mitigating the adverse effects of masking noise.
    - **Mechanism**: Inspired by ALBEF, random masking is treated as noise; the momentum encoder generates soft pseudo-labels, and a KL divergence constraint is applied to encourage masked representations to approximate those of complete data.
    - **Design Motivation**: Performing contrastive learning directly on masked data may produce noisy gradients due to missing information; momentum distillation alleviates this issue through soft-label smoothing.

### Loss & Training
The overall pre-training objective is $\mathcal{L} = \mathcal{L}_\text{mtm} + \mathcal{L}_\text{mlm} + \alpha \cdot \mathcal{L}_\text{tsc} + (1-\alpha) \cdot \mathcal{L}_\text{tsc}^\text{mod}$, where $\mathcal{L}_\text{mtm}$ is the MSE loss for patch reconstruction, $\mathcal{L}_\text{mlm}$ is the cross-entropy loss for masked language modeling, $\mathcal{L}_\text{tsc}$ is the cross-modal contrastive loss, and $\mathcal{L}_\text{tsc}^\text{mod}$ is the KL divergence loss for momentum distillation. For downstream fine-tuning, a linear head is appended directly for classification tasks; for reconstruction tasks (forecasting, imputation, anomaly detection), trend and periodic components are first decomposed and processed separately.

## Key Experimental Results

### Main Results (Scaling Effect — Long-term Forecasting)

| Pre-training Scale | ETTm1 MSE | ETTm2 MSE | ETTh1 MSE | Weather MSE | Traffic MSE | Exchange MSE | Avg MSE |
|-------------------|-----------|-----------|-----------|-------------|-------------|--------------|---------|
| 0B (no pre-train) | 0.401 | 0.293 | 0.487 | 0.257 | 0.471 | 0.383 | 0.358 |
| 1B | 0.376 | 0.292 | 0.461 | 0.257 | 0.473 | 0.370 | 0.354 |
| 10B | 0.376 | 0.281 | 0.444 | 0.250 | 0.473 | 0.368 | 0.345 |
| 25B | 0.378 | 0.278 | 0.434 | 0.253 | 0.467 | 0.357 | 0.342 |
| **50B** | **0.371** | **0.274** | **0.430** | **0.247** | **0.457** | **0.359** | **0.336** |

### Ablation Study

| Configuration | ETTh1 MSE | ETTh2 MSE | Notes |
|---------------|-----------|-----------|-------|
| Full SymTime | Best | Best | Complete pre-training setup |
| w/o Pre-train | Significant drop | Significant drop | Direct fine-tuning without pre-training |
| w/o Symbol | Drop | Drop | Symbol encoder removed; MTM only |
| Real-Data | Drop | Drop | MTM only on equivalent-scale real data |
| w/o MTM | Drop | Drop | Masked time series modeling loss removed |
| w/o Distill | Drop | Drop | Momentum distillation removed |
| Freeze | Worst | Worst | Pre-trained parameters frozen during fine-tuning |

### Key Findings
- Scaling pre-training data from 0B to 50B tokens consistently reduces average long-term forecasting MSE from 0.358 to 0.336, validating the scaling effect of S² data.
- Short-term forecasting OWA decreases from 0.887 to 0.849; imputation MSE drops from 0.038 to 0.026 on ETTm2, demonstrating substantial gains.
- Ablations confirm that the symbol encoder and contrastive learning are critical — removing symbolic information degrades performance, demonstrating that symbolic semantics genuinely enhance time series representations.
- Complexity analysis shows that SymTime requires fewer parameters and less GPU memory than LLM-based models such as Time-LLM.
- t-SNE visualizations show that, after pre-training, the time series encoder forms distinct clusters corresponding to different operator types, confirming the effectiveness of cross-modal semantic alignment.

## Highlights & Insights
- The theoretical grounding in Takens' theorem and symbolic dynamics provides a rigorous mathematical foundation for synthetic data generation, rather than relying on heuristic design.
- Data can be generated at unlimited scale and covers the full representation space (validated by Radviz visualizations showing that S² data spans the statistical properties of the Monash real-world benchmark).
- Pre-training exclusively on synthetic data yields downstream performance competitive with real-data pre-training, entirely circumventing data privacy and scarcity issues.
- Cross-modal contrastive learning endows the time series encoder with symbolic semantics as a unique inductive bias, representing a meaningful exploration of new pre-training paradigms for time series.

## Limitations & Future Work
- The coverage of symbolic expressions is limited to the selected operator set, excluding stochastic differential equations and similar formulations.
- The model scale is modest (6-layer Transformer); scaling behavior of larger models remains unexplored.
- While fine-tuned performance is thoroughly evaluated, zero-shot forecasting capability is not demonstrated (comparison with Chronos and Moirai in zero-shot settings is insufficient).
- The current framework supports only deterministic symbolic expressions and does not account for the role of stochastic processes in time series generation.

## Related Work & Insights
- **Moirai / Timer / TimeGPT**: Time series foundation models pre-trained on real data; SymTime achieving competitive performance with purely synthetic data is a compelling result.
- **Chronos**: Combines synthetic and real data, but the synthetic strategy lacks theoretical motivation; the S² generation mechanism is more principled.
- **ALBEF / MoCo**: Methodological sources for cross-modal contrastive learning and momentum distillation.
- **Core Insight**: Time series are fundamentally projections of dynamical systems; generating time series from symbolic expressions constructs data from the ground up — an approach generalizable to other scientific data generation scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The symbol–time series dual-modality pre-training paradigm is novel, and the theory-driven synthetic data generation is convincing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers five tasks, scaling experiments, ablation studies, representation analysis, and complexity analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear integration of theory and experiments; the introduction of Takens' theorem feels natural.
- **Value**: ⭐⭐⭐⭐ Offers a new data paradigm for time series foundation models; the feasibility of purely synthetic pre-training carries significant implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
