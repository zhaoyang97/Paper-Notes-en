---
title: >-
  [Paper Note] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting
description: >-
  [ICLR 2026][Image Restoration][Explainable forecasting] ProtoTS is proposed to achieve explainable time series forecasting via hierarchical prototype learning: a small number of coarse-grained prototypes provide a global…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "Explainable forecasting"
  - "hierarchical prototypes"
  - "exogenous variables"
  - "multi-channel embedding"
  - "expert-controllable"
date: 2026-05-08
content_hash: f0400a303bcc644b
---

# ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting

**Conference**: ICLR 2026
**arXiv**: [2509.23159](https://arxiv.org/abs/2509.23159)
**Code**: [Available](https://github.com/SKURA502/ProtoTS)
**Area**: Image Restoration
**Keywords**: Explainable forecasting, hierarchical prototypes, exogenous variables, multi-channel embedding, expert-controllable

## TL;DR

ProtoTS is proposed to achieve explainable time series forecasting via hierarchical prototype learning: a small number of coarse-grained prototypes provide a global pattern overview, while progressive refinement captures local variations. Heterogeneous exogenous variables are handled through multi-channel embedding and bottleneck fusion. On the LOF dataset, MSE is reduced by 48.3% and MAE by 20.9%. The framework additionally supports expert editing of prototypes to further improve performance.

## Background & Motivation

Time series forecasting is widely applied in high-stakes domains such as power dispatch, energy management, and weather prediction. In these scenarios, accurate predictions alone are insufficient—understanding the reasons behind predictions is equally critical to prevent significant financial losses and to establish trust.

Two core deficiencies of existing explainability methods:

- **C1 (output side)**: Methods such as TFT and DiPE-Linear explain predictions at individual time steps but fail to explain overall trend patterns (e.g., "why does the power load curve exhibit three decreasing peaks at noon, afternoon, and evening"). Power dispatch experts need to understand holistic patterns to decide whether to purchase external electricity.
- **C2 (input side)**: Existing explanations focus on only a subset of input variables (e.g., CycleNet considers only endogenous variables). However, forecasting outcomes are determined by the interaction of multiple heterogeneous variables (e.g., high temperature + summer → air-conditioning peak), necessitating an understanding of their joint effects.

**ProtoTS's approach**: Each prototype corresponds to a typical temporal pattern (e.g., "Spring Festival pattern," "summer weekday pattern"), and predictions are formed by matching instances to prototypes via similarity. A small number of prototypes provide a global overview, while the hierarchical structure enables progressive refinement and expert intervention.

## Method

### Overall Architecture

ProtoTS consists of two major modules:

1. **Multi-channel prototype similarity computation module**: Processes heterogeneous input variables and computes instance-prototype similarity.
2. **Hierarchical prototype learning module**: Organizes prototypes in a tree structure to learn temporal patterns from coarse to fine.

### Key Designs

**1. Multi-Channel Embedding**

Independent encoding channels are designed for endogenous variables, discrete exogenous variables, and continuous exogenous variables:

- **Endogenous channel**: $\gamma(\mathbf{y}_t)$, encoded via an MLP with activation functions.
- **Discrete exogenous channel**: $\mathbf{E}_j(\mathbf{x}_{t,j}^{\text{dis}})$, using independent embedding tables.
- **Continuous exogenous channel**: $\psi_j(\mathbf{x}_{t,j}^{\text{con}})$, using variable-specific nonlinear projections.

The complete embedding at time $t$ is obtained by additive aggregation:

$$\mathbf{Z}_t = \gamma(\mathbf{y}_t) + \sum_{j=1}^{C_{\text{dis}}} \mathbf{E}_j(\mathbf{x}_{t,j}^{\text{dis}}) + \sum_{j=1}^{C_{\text{con}}} \psi_j(\mathbf{x}_{t,j}^{\text{con}})$$

Within the prediction window, $\mathbf{y}_t$ is unavailable, so only exogenous variables are used.

**2. Bottleneck Channel Fusion**

Aggregating heterogeneous variables may introduce noise. ProtoTS incorporates a bottleneck layer $\mathbb{R}^d \to \mathbb{R}^{d_{\text{bottle}}} \to \mathbb{R}^d$ ($d_{\text{bottle}} \ll d$) within an MLP-Mixer architecture, performing fusion along both the feature and temporal dimensions:

$$\mathbf{Z}_{1:L+H}^{(l+1)} = \text{MLP}_{\text{time}}(\text{MLP}_{\text{feature}}(\mathbf{Z}_{1:L+H}^{(l)})^T)^T$$

The temporal dimension is then linearly aggregated: $\hat{\mathbf{Z}} = \mathbf{Z}_{1:L+H}^T \mathbf{W} \in \mathbb{R}^d$.

**3. Prototype Similarity Computation and Prediction**

Each prototype comprises a learnable embedding $\boldsymbol{\mu} \in \mathbb{R}^d$ and a temporal pattern $\mathbf{p} \in \mathbb{R}^T$. Similarity is computed via Euclidean distance followed by softmax:

$$f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_c) = \frac{\exp(-d(\hat{\mathbf{Z}}, \boldsymbol{\mu}_c))}{\sum_{i=1}^N \exp(-d(\hat{\mathbf{Z}}, \boldsymbol{\mu}_i))}$$

The prediction is a weighted combination of prototype temporal patterns: $\hat{\mathbf{Y}} = \sum_{i=1}^N f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_i) \cdot \mathbf{p}_i$.

**4. Hierarchical Prototype Learning**

- **Root level**: A small number of prototypes (e.g., 6) capture coarse-grained patterns (seasonality, holidays, etc.) and are trained to convergence first.
- **Splitting strategy**: Leaf prototypes requiring refinement are selected based on normalized loss (top $\alpha$%), and each is split into $M$ child prototypes.
- **Child-level prediction**:

$$\hat{\mathbf{Y}} = \sum_{i=1}^N f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_i) \sum_{j=1}^M f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_{i,j}) \cdot \mathbf{p}_{i,j}$$

The splitting criterion is based on the mean MAE loss over instances associated with each prototype: high-loss prototypes indicate that their temporal patterns insufficiently represent the associated instances and require further refinement.

**5. Expert Controllability**

- Selectively splitting specific prototypes (e.g., splitting the "Spring Festival" prototype into "pre-holiday" and "mid-holiday").
- Introducing new root-level prototypes.
- Directly editing the temporal patterns of prototypes.

### Loss & Training

$$\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_1 - \lambda \sum_{i=1}^N f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_i) \log(f(\hat{\mathbf{Z}}|\boldsymbol{\mu}_i))$$

- L1 prediction loss combined with entropy regularization (encouraging a small number of prototypes to cover the majority of predictions).
- Stage-wise training: root level trained first → prototypes split upon convergence → child level continues training.

## Key Experimental Results

### Main Results

**LOF dataset** (power load forecasting, 22 exogenous variables, 4 regions):

| Model | RE | YC | EA | PC | Avg MAE | vs ProtoTS |
|------|-----|-----|-----|-----|---------|-----------|
| **ProtoTS** | **0.198** | **0.055** | **0.059** | **0.112** | **0.106** | - |
| TiDE | 0.253 | 0.057 | 0.061 | 0.164 | 0.134 | +21% |
| iTransformer | 0.279 | 0.080 | 0.097 | 0.139 | 0.149 | +29% |
| TimeXer | 0.272 | 0.079 | 0.096 | 0.182 | 0.157 | +32% |
| XGBoost | 0.405 | 0.084 | 0.092 | 0.230 | 0.203 | +48% |

**EPF dataset** (electricity price forecasting, 5 markets):

| Model | NP | PJM | BE | FR | DE | Avg MAE |
|------|-----|-----|-----|-----|-----|---------|
| **ProtoTS** | **0.213** | **0.152** | **0.226** | **0.183** | **0.318** | **0.218** |
| TimeXer | 0.240 | 0.173 | 0.241 | 0.192 | 0.343 | 0.238 |

ProtoTS reduces MSE by 48.3% and MAE by 20.9% on LOF; both MSE and MAE are reduced by 8% on EPF.

### Ablation Study

| Component | PC MSE | YC MSE | RE MSE | EA MSE | Avg MAE |
|------|--------|--------|--------|--------|---------|
| w/o bottleneck | 0.044 | 0.013 | 0.089 | 0.129 | 0.143 |
| w/o multi-channel | 0.034 | 0.006 | 0.108 | 0.007 | 0.117 |
| w/o hierarchy | 0.026 | 0.006 | 0.089 | 0.007 | 0.110 |
| **ProtoTS (full)** | **0.025** | **0.006** | **0.085** | **0.007** | **0.106** |

### Key Findings

- **Data efficiency**: When training data is reduced from 100% to 50%, ProtoTS degrades only marginally, whereas TimeXer and iTransformer deteriorate substantially.
- **Number of root prototypes**: Performance saturates at 12–15 prototypes, suggesting a limited number of distinct temporal patterns.
- **Quantitative explainability evaluation**: In a user study with 24 participants, ProtoTS achieves a User Precision of 77% (vs. TFT 64%, NBEATSx 62%) and a SUS score of 73.36, substantially outperforming baselines.
- **Expert editing case study**: Manually splitting the "Spring Festival" prototype into "pre-holiday" and "mid-holiday" reduces MSE by 0.009 during the Spring Festival period.

## Highlights & Insights

1. **Prototypes as temporal patterns**: This work is the first to decode prototypes as output sequences (e.g., 96-step forecast curves) rather than single class labels.
2. **Global + local dual-level explanation**: Coarse-grained prototypes provide global understanding, while fine-grained prototypes supply local detail.
3. **Expert-in-the-loop**: Explainability goes beyond mere "visualization"—experts can actively edit prototypes to optimize the model.
4. **Handling heterogeneous exogenous variables**: Multi-channel embedding combined with bottleneck denoising prevents noisy variables from interfering with predictions.

## Limitations & Future Work

- Semantic naming of prototypes (e.g., "Spring Festival pattern") currently requires manual summarization; integration with LLMs for automatic semantic label generation is a promising direction.
- Hierarchy depth and splitting strategy rely on heuristic rules (top $\alpha$% loss); adaptive schemes merit further exploration.
- Evaluation is limited to power load and electricity price datasets; applicability to other high-stakes domains (healthcare, finance) remains to be validated.
- Integration with foundation models (e.g., using ProtoTS prototypes to explain large model predictions) is an interesting future direction.

## Related Work & Insights

- **Prototype network family**: Extending prototypes from classification to regression-based sequence output represents an important advancement of prototype methods.
- **CycleNet**: Discovers periodic patterns in endogenous variables only; ProtoTS jointly models interactions with exogenous variables.
- **TFT attention-based explanation**: Provides local step-wise explanations; ProtoTS prototypes offer a more intuitive global perspective.
- Insight: In time series forecasting, "explainability" is not merely an add-on feature—prototype learning can simultaneously improve predictive accuracy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Hierarchical prototypes decoded as temporal sequence outputs, expert-editable design, pioneering contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (LOF + EPF datasets, complete ablations, user study as a bonus)
- Writing Quality: ⭐⭐⭐⭐⭐ (Detailed power domain case studies, highly convincing hierarchical prototype visualizations)
- Value: ⭐⭐⭐⭐⭐ (Balances accuracy and explainability; expert-controllable design directly addresses industrial needs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](../../NeurIPS2025/image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)
- [\[ACL 2026\] Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning](../../ACL2026/image_restoration/understanding_and_mitigating_spurious_signal_amplification_in_test-time_reinforc.md)
- [\[ICLR 2026\] Mechanism of Task-oriented Information Removal in In-context Learning](mechanism_of_task-oriented_information_removal_in_in-context_learning.md)
- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](../../ICCV2025/image_restoration/learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)

</div>

<!-- RELATED:END -->
