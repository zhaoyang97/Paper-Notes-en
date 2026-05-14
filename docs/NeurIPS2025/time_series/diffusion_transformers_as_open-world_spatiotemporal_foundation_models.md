---
title: >-
  [Paper Note] Diffusion Transformers as Open-World Spatiotemporal Foundation Models
description: >-
  [NeurIPS 2025][Time Series][Diffusion Transformer] This paper proposes UrbanDiT, the first open-world urban spatiotemporal foundation model based on Diffusion Transformers. It integrates heterogeneous data types (grid/gr…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Diffusion Transformer"
  - "Spatiotemporal Foundation Model"
  - "Urban Computing"
  - "Prompt Learning"
  - "Zero-shot"
date: 2026-05-08
content_hash: dea2851f52c63cc5
---

# Diffusion Transformers as Open-World Spatiotemporal Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2411.12164](https://arxiv.org/abs/2411.12164)
**Code**: [tsinghua-fib-lab/UrbanDiT](https://github.com/tsinghua-fib-lab/UrbanDiT)
**Area**: Time Series
**Keywords**: Diffusion Transformer, Spatiotemporal Foundation Model, Urban Computing, Prompt Learning, Zero-shot

## TL;DR
This paper proposes UrbanDiT, the first open-world urban spatiotemporal foundation model based on Diffusion Transformers. It integrates heterogeneous data types (grid/graph) and diverse tasks (prediction, interpolation, extrapolation, imputation) through a unified prompt learning framework, achieving state-of-the-art performance across multiple cities and scenarios while demonstrating strong zero-shot generalization.

## Background & Motivation
Spatiotemporal dynamics in urban environments arise from diverse human activities and manifest in heterogeneous data formats: grid-based data (e.g., crowd flow) and graph-based data (e.g., road network traffic speed). Existing urban spatiotemporal models exhibit notable limitations:

- **UniST**: Supports only grid-based data and exclusively performs prediction tasks.
- **UrbanGPT**: LLM-based; supports only grid-based data with a narrow task scope.
- **GPD**: Handles only graph-based traffic data; does not support multi-source data or zero-shot transfer.
- **CityGPT**: LLM-based processing of language-form urban data; lacks multi-source and zero-shot support.

The core pain point is the **absence of a unified foundation model capable of simultaneously handling heterogeneous data types, supporting diverse spatiotemporal tasks, and generalizing to open-world scenarios**. Diffusion Transformers (e.g., Sora) combine the generative capability of diffusion models with the scalability of Transformers, providing an ideal backbone for addressing this challenge.

## Core Problem
Can a foundation model analogous to those in NLP/CV be constructed to learn universal spatiotemporal patterns and serve as a general-purpose model for urban spatiotemporal applications? Three sub-problems must be resolved:

1. How to unify heterogeneous grid-based and graph-based data formats?
2. How to support diverse tasks — prediction, interpolation, extrapolation, and imputation — within a single model?
3. How to achieve zero-shot generalization to unseen cities and datasets?

## Method

### Data Unification: Heterogeneous Spatiotemporal Data to Sequences
Three-dimensional structures (2D spatial + 1D temporal) are unified into a one-dimensional sequence format:

- **Temporal dimension**: Temporal patching (analogous to PatchTST) segments time series into patches.
- **Grid-based data**: 2D spatial patching (analogous to ViT) reshapes $H \times W \times T$ into a sequence of length $L = \frac{H \times W \times T}{p_s \times p_s \times p_t}$.
- **Graph-based data**: A GCN processes each node, which is then integrated with the temporal dimension into a one-dimensional sequence.

### Task Unification: Mask-Based Multi-Task Framework
All tasks are unified under a paradigm of "reconstructing masked regions," realized through different masking strategies:

| Task | Mask Strategy |
|------|--------------|
| Forward Prediction | Mask future time steps |
| Backward Prediction | Mask past time steps |
| Temporal Interpolation | Mask specific temporal positions in the sequence |
| Spatial Extrapolation | Mask spatial positions outside the observed region |
| Spatio-Temporal Imputation | Randomly mask spatiotemporal positions |

The input is represented as $X^t = X^t \ast (1-M) + X^0 \ast M$, where $M$ controls the conditional information for each task.

### Spatio-Temporal Transformer Block
The model stacks multiple spatiotemporal Transformer blocks, each containing separate temporal attention and spatial attention modules. This decoupled design avoids the quadratic complexity of joint attention.

### Unified Prompt Learning (Core Innovation)

**Data-Driven Prompt**: Three memory pools (key-value stores) are employed to capture:

- **Temporal patterns**: $(K_t, V_t)$, retrieved based on the temporal features of the input.
- **Frequency-domain patterns**: $(K_f, V_f)$, retrieved based on the spectral features of the input.
- **Spatial patterns**: $(K_s, V_s)$, retrieved based on the spatial features of the input.

The most relevant prompts are retrieved from each memory pool via softmax attention, analogous to retrieval-augmented generation (RAG), enabling the model to distinguish distributional differences across datasets.

**Task-Specific Prompt**: A task-specific prompt $P_m = \text{Attention}(\text{Flatten}(M))$ is generated from the mask map, enabling the model to perceive the type of task being performed.

All prompts are concatenated with the input sequence before being fed into the Transformer, leveraging its ability to handle variable-length sequences for flexible input processing.

### Loss & Training
- At each iteration, one dataset and one task are randomly selected for gradient descent.
- The rectified flow training approach from InstaFlow (an ODE framework that linearizes noise-to-data trajectories) is adopted to improve generation efficiency.
- Diffusion steps are set to 500 during training; only 20 steps are required at inference (25× speedup).

## Key Experimental Results

### Datasets
Multi-domain datasets spanning 5 cities (New York, Beijing, Shanghai, Nanjing, etc.) covering taxi demand, cellular network traffic, crowd flow, transportation, and dynamic population. Data is split temporally in a 6:2:2 ratio.

### Main Results

**Forward Prediction (Grid-based, Table 2)**

| Model | TaxiBJ MAE | FlowSH MAE | TaxiNYC MAE | CrowdNJ MAE | PopBJ MAE |
|-------|-----------|------------|------------|------------|----------|
| UniST | 14.04 | 9.10 | 5.85 | 0.119 | 0.106 |
| CSDI | 14.76 | 8.77 | 5.05 | 0.094 | 0.078 |
| **UrbanDiT** | **12.61** | **5.61** | **5.58** | **0.092** | **0.077** |

Overall relative improvement: **11.3%**; backward prediction relative improvement: **30.4%**.

**Spatial Extrapolation (Table 3, 50% spatial positions masked)**
UrbanDiT substantially outperforms all baselines; for example, on TaxiBJ, MAE drops from 36.66 (CSDI) to **8.10**.

**Zero-shot Performance**
On the PopSH dataset, UrbanDiT's zero-shot performance surpasses nearly all supervised baselines trained on target data, validating its open-world generalization capability.

### Ablation Study
Removing any individual prompt component leads to a significant performance drop; removing all prompts yields the worst performance. The **frequency-domain prompt has the greatest impact**.

**Scalability**
UrbanDiT-L exhibits the steepest performance improvement slope (0.011 vs. 0.0015/0.0019) as data volume increases from 0.8 to 1.0, indicating that larger models continue to benefit from additional data.

## Highlights & Insights
1. **Truly unified model**: The first to simultaneously support grid/graph data, 5 spatiotemporal tasks, and multi-city multi-domain data.
2. **Elegant prompt learning design**: The combination of data-driven prompts (three-domain memory pools) and task-specific prompts is both flexible and effective.
3. **Strong zero-shot capability**: Surpasses most supervised baselines without any target-domain training data.
4. **Efficient inference**: Rectified flow combined with 20-step inference achieves a 25× speedup while maintaining generation quality.
5. **Good scalability**: Larger models continue to benefit as data grows, consistent with the scaling laws of foundation models.

## Limitations & Future Work
1. **Limited data coverage**: Currently focused on human activity data (mobility, transportation); environmental variables (air pollution, climate indicators, microclimate dynamics) are not addressed.
2. **Computational cost**: Training and inference costs of Diffusion Transformers remain high, particularly at scale with large urban datasets.
3. **Prior-dependent prompt design**: The three-domain memory pool structure (temporal/frequency/spatial) is manually designed; automated prompt discovery warrants exploration.
4. **Ceiling of spatial extrapolation**: While performance under 50% masking is substantially better than baselines, extreme sparsity regimes have not been validated.
5. **Numerical data only**: Multimodal urban data such as text and imagery are not considered.

## Related Work & Insights

| Dimension | UniST | UrbanGPT | GPD | UrbanDiT |
|-----------|-------|----------|-----|----------|
| Model Initialization | From scratch | LLM | From scratch | From scratch |
| Data Type | Grid | Grid | Graph | Grid + Graph |
| Multi-source Data | ✓ | ✓ | ✗ | ✓ |
| Task Flexibility | ✗ | ✗ | ✗ | ✓ |
| Zero-shot | ✓ | ✓ | ✗ | ✓ |

UrbanDiT is the only model satisfying all five criteria. Compared to CSDI (the second-best baseline), UrbanDiT not only comprehensively outperforms CSDI on diffusion-based generation tasks where CSDI excels, but also supports heterogeneous data types and multi-task scenarios that CSDI cannot handle.

**Broader Implications**:
- **General value of prompt learning**: Data-driven prompts via memory pool retrieval are transferable to other foundation model settings requiring the unification of heterogeneous data.
- **Mask-based task unification**: Unifying prediction, interpolation, extrapolation, and imputation as masked reconstruction — analogous to MAE in vision — is a paradigm worth extending to additional domains.
- **Rectified flow acceleration**: The ODE-based linearized trajectory training strategy can substantially reduce inference steps for diffusion models, making them practical for compute-constrained deployments.
- **Urban computing × foundation models**: As urban data continues to accumulate, the value of domain-specific foundation models will increase considerably.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First to extend DiT to unified urban spatiotemporal modeling; the prompt learning framework is creatively designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 5 cities, multiple domains, 5 tasks, few-shot/zero-shot transfer, ablation, and scalability experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, richly illustrated, with complete methodological exposition.
- **Value**: ⭐⭐⭐⭐ — Establishes a new benchmark for urban spatiotemporal foundation models; open-sourced code and datasets facilitate follow-up research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [\[NeurIPS 2025\] BubbleFormer: Forecasting Boiling with Transformers](bubbleformer_forecasting_boiling_with_transformers.md)
- [\[ICLR 2026\] Relational Feature Caching for Accelerating Diffusion Transformers](../../ICLR2026/time_series/relational_feature_caching_for_accelerating_diffusion_transformers.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
