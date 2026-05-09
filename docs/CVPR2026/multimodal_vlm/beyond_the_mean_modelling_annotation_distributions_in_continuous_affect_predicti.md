---
title: >-
  [Paper Note] Beyond the Mean: Modelling Annotation Distributions in Continuous Affect Prediction
description: >-
  [CVPR 2026][Multimodal VLM][affect prediction] This paper proposes a Beta distribution-based framework for modelling affective annotation consensus. The model predicts only the mean and standard deviation of the annotation distribution, from which higher-order descriptors—including skewness, kurtosis, and quantiles—are derived in closed form via moment matching. Experiments on SEWA and RECOLA demonstrate that Beta distributions effectively capture the full distributional characteristics of annotator disagreement.
tags:
  - CVPR 2026
  - Multimodal VLM
  - affect prediction
  - annotation distribution modelling
  - Beta distribution
  - annotator disagreement
  - uncertainty
date: 2026-05-08
content_hash: 5b41f3481eb802cc
---

# Beyond the Mean: Modelling Annotation Distributions in Continuous Affect Prediction

**Conference**: CVPR 2026
**arXiv**: [2604.07198](https://arxiv.org/abs/2604.07198)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: affect prediction, annotation distribution modelling, Beta distribution, annotator disagreement, uncertainty

## TL;DR
This paper proposes a Beta distribution-based framework for modelling affective annotation consensus. The model predicts only the mean and standard deviation of the annotation distribution, from which higher-order descriptors—including skewness, kurtosis, and quantiles—are derived in closed form via moment matching. Experiments on SEWA and RECOLA demonstrate that Beta distributions effectively capture the full distributional characteristics of annotator disagreement.

## Background & Motivation
**State of the Field**: In continuous affect prediction (valence-arousal), multiple annotators frequently disagree in their perception of the same behaviour. Such disagreement reflects the **intrinsic subjectivity** of affective signals rather than simple annotation noise.

**Limitations of Prior Work**: Mainstream methods collapse annotations into **point estimates** (mean or median), discarding rich information about annotator disagreement, uncertainty, and distributional shape. This implicitly treats disagreement as noise rather than as a meaningful signal.

**Root Cause**: Ignoring annotation variability may impair model generalisation and calibration, particularly in high-stakes applications such as healthcare and education.

**Core Idea**: Model the annotation distribution using a Beta distribution, which (1) is defined on $[0,1]$, naturally suiting normalised affective dimensions; (2) offers flexible parameterisation (capable of representing symmetric, skewed, and concentrated distributions); and (3) permits closed-form derivation of all higher-order statistics from $(μ, σ)$.

## Method

### Overall Architecture
Annotator signals → Compute empirical $(μ, σ)$ → Moment matching to Beta parameters $(α, β)$ → Multimodal features → ANN predicts $(μ, σ)$ → Recover Beta distribution → Closed-form derivation of skewness / kurtosis / quantiles

### Key Designs
1. **Beta Distribution Moment Matching**:
   Given the empirical annotator mean $μ$ and variance $σ^2$:
   $$\phi = \frac{\mu(1-\mu)}{\sigma^2} - 1, \quad \alpha = \mu\phi, \quad \beta = (1-\mu)\phi$$
   Constraints: $0 < \mu < 1$, $0 < \sigma^2 < \mu(1-\mu)$, ensuring $\alpha, \beta > 0$.
   - **Rationale for Beta distribution**: Its bounded-interval property matches the value range of affective dimensions; its shape flexibility captures diverse patterns of annotator behaviour.

2. **Closed-Form Derivation of Higher-Order Descriptors**:
   - Skewness: $\text{Skew}(X) = \frac{2(\beta-\alpha)\sqrt{\alpha+\beta+1}}{(\alpha+\beta+2)\sqrt{\alpha\beta}}$ → captures asymmetry in annotator disagreement
   - Kurtosis: measures the concentration of annotations
   - Quantiles: obtained via the inverse regularised incomplete Beta function
   - **Advantage**: The model need only learn to predict two scalars $(μ, σ)$; all higher-order statistics are obtained at no additional cost.

3. **Model Variants**:
   - $M_I$: Two independent networks predicting $μ$ and $σ$ separately
   - $M_S$: Shared first layer with branching second layer
   - $M_F$: Fully shared network with two output heads
   - Baseline $B$: Separate regression networks trained independently for each descriptor ($μ$, $σ$, skewness, kurtosis, quantiles)

### Loss & Training
- MSE loss for optimising $(μ, σ)$ predictions
- Adam optimiser, learning rate $1\text{e-}3$, batch size 128
- 5-fold subject-independent cross-validation, repeated 10 times with random seeds
- Early stopping (validation MSE, patience of 5 epochs)
- Evaluation metrics: CCC (Concordance Correlation Coefficient) and KL divergence

## Key Experimental Results

### Main Results (CCC Performance)

| Dataset | Modality | Model | Arousal μ | Arousal σ | Valence μ | Valence σ |
|---------|----------|-------|-----------|-----------|-----------|-----------|
| RECOLA | Audio | $M_I$ | 0.19 | 0.04 | **0.54** | 0.25 |
| RECOLA | Fusion | $M_I$ | **0.24** | 0.01 | 0.48 | 0.26 |
| SEWA | Visual | $M_F$ | **0.80** | **0.61** | 0.76 | 0.51 |
| SEWA | Fusion | $M_F$ | 0.76 | **0.65** | **0.78** | **0.57** |

### Ablation Study (Higher-Order Descriptor CCC: Beta-Derived M vs. Direct Regression B)

| Dataset / Modality | Descriptor | Baseline B | Beta Model M | Notes |
|--------------------|------------|------------|--------------|-------|
| RECOLA Audio | median | 0.16 | **0.18** | Beta derivation superior |
| RECOLA Audio | q25 | 0.12 | **0.18** | Substantially outperforms direct regression |
| RECOLA Fusion | median | 0.30 | **0.31** | Marginally superior |
| SEWA Visual | skew | **0.21** | 0.19 | Baseline slightly better on isolated metric |

### KL Divergence (Predicted Distribution vs. Ground-Truth Annotation Distribution)

| Dataset | Modality | vs. Uniform $\mathcal{U}$ | vs. True Beta $\mathcal{B}$ |
|---------|----------|--------------------------|----------------------------|
| RECOLA | Audio | 13.59 | **0.64** |
| SEWA | Visual | 2.40 | **0.78** |

### Key Findings
- The KL divergence of Beta distribution predictions is substantially lower than the uniform distribution baseline (0.64 vs. 13.59), confirming that the model successfully captures the shape of the annotation distribution.
- Higher-order descriptors derived from $(μ, σ)$ **match or surpass** regressors trained individually for each descriptor in most cases.
- $M_F$ (fully shared) performs best on SEWA, while $M_I$ (independent) is superior on RECOLA—dataset characteristics influence the optimal parameter-sharing strategy.
- The visual modality substantially outperforms audio on SEWA (CCC 0.80 vs. 0.02), consistent with dataset-specific properties (face-to-face interaction vs. remote collaboration).

## Highlights & Insights
- **Methodological contribution**: The first application of closed-form Beta distribution derivation to continuous affect prediction, offering an elegant and efficient solution.
- Modelling affective consensus as a probability distribution rather than a point estimate more faithfully reflects the subjective nature of affective annotation.
- Predicting only two scalars suffices to recover a complete distributional description, imposing minimal computational overhead.
- Establishes a research framework that treats affect as a probabilistic signal processing problem.

## Limitations & Future Work
- The Beta distribution assumption is restrictive—annotation distributions may be multimodal, whereas the Beta distribution can only model unimodal shapes.
- The lightweight ANN architecture may limit feature learning capacity; deeper models (LSTM / Transformer) warrant exploration.
- RECOLA has only 6 annotators and SEWA only 3—the limited number of annotators constrains the accuracy of distributional estimation.
- Temporal structure is not exploited, as predictions are made on independent windows.

## Related Work & Insights
- This work aligns with MBNet (modelling listener bias in MOS prediction) and DeePMOS (predicting the full MOS distribution).
- Beta distribution modelling is generalisable to other subjective evaluation tasks (image quality assessment, speech quality assessment, etc.).
- The paper provides a practical tool for the "annotator disagreement as signal" research paradigm in affective computing.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The closed-form Beta distribution derivation is concise and powerful, though probabilistic modelling of annotation disagreement has prior precedents.
- **Experimental Thoroughness**: ⭐⭐⭐ Only two datasets with small-scale models (two-layer ANN); larger-scale validation is needed.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and experimental design is rigorous.
- **Value**: ⭐⭐⭐⭐ Provides the affective computing community with a practical distribution-aware modelling framework.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](evolmm_self_evolving_lmm_continuous_rewards.md)
- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[CVPR 2026\] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)
- [\[CVPR 2026\] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)

<!-- RELATED:END -->
