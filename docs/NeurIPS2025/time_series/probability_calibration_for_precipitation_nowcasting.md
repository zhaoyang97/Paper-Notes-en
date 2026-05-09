---
title: >-
  [Paper Note] Probability Calibration for Precipitation Nowcasting
description: >-
  [NeurIPS 2025][Time Series][Probability Calibration] This paper proposes the Expected Threshold Calibration Error (ETCE) as a more appropriate metric for probability calibration in precipitation nowcasting, and extends post-hoc calibration techniques from computer vision to the forecasting domain. By incorporating a lead-time-conditioned Selective Scaling method, the proposed approach reduces model calibration error by up to 23.5%.
tags:
  - NeurIPS 2025
  - Time Series
  - Probability Calibration
  - Precipitation Nowcasting
  - Selective Scaling
  - Calibration Error
  - Neural Weather Models
date: 2026-05-08
content_hash: 474cca78457ce43f
---

# Probability Calibration for Precipitation Nowcasting

**Conference**: NeurIPS 2025
**arXiv**: [2510.00594](https://arxiv.org/abs/2510.00594)
**Code**: N/A
**Area**: Time Series / Weather Forecasting
**Keywords**: Probability Calibration, Precipitation Nowcasting, Selective Scaling, Calibration Error, Neural Weather Models

## TL;DR

This paper proposes the Expected Threshold Calibration Error (ETCE) as a more appropriate metric for probability calibration in precipitation nowcasting, and extends post-hoc calibration techniques from computer vision to the forecasting domain. By incorporating a lead-time-conditioned Selective Scaling method, the proposed approach reduces model calibration error by up to 23.5%.

## Background & Motivation

Precipitation nowcasting—high spatiotemporal resolution prediction of rainfall over the next four hours—is critical for time-sensitive decisions in disaster response, traffic safety, urban drainage management, and winter road maintenance. In recent years, deep neural network (DNN)-based neural weather models (NWMs) have achieved state-of-the-art performance and are being deployed operationally by industry and meteorological agencies.

However, many applications require not only accurate forecasts but also **probabilistic forecasts**—that is, predicted probabilities that faithfully reflect the likelihood of events. The central requirement of probabilistic forecasting is **calibration**: when a model predicts an event with probability $p$, that event should occur with frequency exactly $p$.

Existing standard calibration metrics such as the **Expected Calibration Error (ECE)** exhibit notable shortcomings:

- ECE considers only the highest-probability predicted class and its associated confidence, masking threshold-level calibration biases in multi-class ordinal problems.
- For precipitation forecasting—an ordinal classification problem where simultaneous knowledge of, e.g., exceedance probabilities at 1 mm and 10 mm is required—ECE fails to capture calibration quality across all precipitation intervals.
- Static Calibration Error (SCE), though extended to multi-class settings, still assumes class independence, which is inappropriate for ordinal precipitation amounts.

Furthermore, unlike computer vision tasks, weather forecasting introduces a unique dimension—**lead time**—that further complicates calibration.

## Method

### Overall Architecture

The paper makes two core contributions: (1) proposing ETCE as a calibration metric tailored to precipitation forecasting, and (2) extending and adapting post-hoc calibration techniques from computer vision to the meteorological forecasting domain. The overall pipeline uses a fixed base probabilistic model to generate logit outputs, which are then post-processed by a lightweight calibrator to improve calibration.

### Key Designs

**1. Expected Threshold Calibration Error (ETCE)**

The core idea of ETCE is to evaluate calibration separately at each precipitation threshold $R_k$. Given $K$ precipitation thresholds, the cumulative exceedance probability $\hat{P}(r > R_k)$ is first computed for each threshold. Predictions are then binned into $B$ equally spaced confidence intervals per threshold, and the following is computed:

$$\text{ETCE} = \frac{1}{K} \sum_{k=1}^{K} \sum_{b=1}^{B} w_b \left| \text{acc}(b, R_k) - \text{conf}(b, R_k) \right|$$

where $w_b$ denotes the bin weight. Since precipitation is a rare event, the authors adopt uniform weights $w_b = 1/B$ rather than sample-count-based weights, to prevent dry events from dominating the metric. The number of bins is set to $B=20$.

**2. Temperature Scaling (TS)**

The classical approach: a single scalar parameter $T \in \mathbb{R}^+$ is learned, and predicted probabilities are rescaled via $\hat{p} = \sigma_{\text{softmax}}(z/T)$. In segmentation tasks, a single temperature is applied uniformly across all pixels and samples.

**3. Local Temperature Scaling (LTS)**

A distinct temperature value is learned per pixel $x$ using a small hierarchical CNN that maps the logit vector to a temperature value. The authors' adaptation uses only logits as input and introduces lead-time conditioning via FiLM (Feature-wise Linear Modulation), applying affine transformations conditioned on lead-time information to intermediate feature maps.

**4. Selective Scaling (SS)**

Based on the key observation that the primary source of miscalibration in neural networks is **overconfidence on incorrect predictions**, SS proceeds as follows:
- A classifier (based on logits) is trained to detect incorrect predictions made by the base model.
- Temperature scaling ($T > 1$) is applied only to detected incorrect predictions, reducing their overconfidence.
- Original probabilities are preserved for correct predictions.

$$\hat{p} = \begin{cases} \sigma_{\text{softmax}}(z), & \text{if } \hat{y} = y \\ \sigma_{\text{softmax}}(z/T), & \text{if } \hat{y} \neq y \end{cases}$$

The authors incorporate lead-time embeddings into a 3-layer MLP classifier via FiLM conditioning, and additionally investigate a larger attention-based architecture using Segformer.

### Base Model and Data

**Base Model**: A fixed probabilistic model that produces predictions conditioned on lead time independently. The architecture comprises:
- Spatial encoder: a sequence of convolutional layers downsampling input from $512 \times 512$ to $64 \times 64$ (512 channels)
- Attention module: 4 axial attention layers (512 channels)
- Classification head: 12 output channels corresponding to 12 precipitation intensity intervals
- Total parameters: 21M

**Data**: Inputs include 7-step MRMS radar imagery, 2-step 16-channel GOES satellite data, 1-step HRRR numerical weather model precipitation forecasts, and terrain, latitude/longitude, temporal metadata, and lead time. The target is MRMS precipitation discretized into 12 intervals (0.2 to 10+ mm/h).

**Calibration Training Data**: Temporally non-overlapping with base model training data:
- 110K samples for training the incorrect-prediction classifier
- 1K samples for optimizing temperature parameters
- 47K samples for evaluation

## Key Experimental Results

### Main Results

| Calibration Method | Parameters | F1 Score | Mean ETCE | ΔETCE (%) |
|:---|:---:|:---:|:---:|:---:|
| Uncalibrated baseline | - | 0.565 | 0.079 | - |
| Temperature Scaling (TS) | 1 | 0.565 | 0.080 | −1.0 |
| LTS (w/o lead-time conditioning) | 2,107 | 0.573 | 0.096 | −21.3 |
| LTS (w/ lead-time conditioning) | 2,143 | 0.564 | 0.082 | −3.6 |
| **Selective Scaling w/ MLP** | **3,254** | **0.564** | **0.060** | **+23.5** |
| Selective Scaling w/ Segformer B0 | 3,728,550 | 0.567 | 0.062 | +21.6 |

### Ablation Study: Segformer Classifier Scale Comparison

| Classifier Architecture | Relative ETCE Improvement | Computational Cost |
|:---|:---:|:---:|
| MLP (3-layer) | 23.5% | Minimal (3,254 parameters) |
| Segformer-B0 | 21.6% | High (3.7M parameters) |
| Segformer-B1 | ~24% | Higher |
| Segformer-B2 | ~24.8% | Very high |

Larger Segformer classifiers yield only ~1.3% additional ETCE improvement at substantially greater computational cost, making the MLP the most cost-effective choice.

### Key Findings

1. **Selective Scaling achieves the best calibration**: The MLP-based variant reduces ETCE by 23.5% while preserving F1 score (0.564 vs. baseline 0.565), demonstrating that calibration improvement does not come at the expense of predictive quality.

2. **Temperature Scaling is ineffective**: The single-parameter temperature scaling approach not only fails to improve calibration in precipitation forecasting but slightly worsens ETCE by 1%, contradicting positive findings in certain computer vision contexts and underscoring the unique characteristics of forecasting problems.

3. **LTS requires lead-time conditioning**: LTS without lead-time conditioning severely degrades calibration (−21.3%); incorporating FiLM-based lead-time encoding mitigates the harm (−3.6%), though calibration remains worse than the baseline.

4. **Lead-time-dependent behavior**: At short lead times (≤150 minutes), MLP and Segformer-B0 perform comparably; at longer lead times, MLP outperforms, indicating that simpler models generalize better for long-horizon predictions.

5. **Calibration improvement primarily reduces overconfidence**: Analysis of miscalibration diagrams reveals that the uncalibrated model is systematically overconfident across lead times and precipitation thresholds; Selective Scaling effectively reduces calibration error in high-confidence bins.

## Highlights & Insights

- **Principled design of the ETCE metric**: Thresholding preserves the ordinal nature of precipitation amounts, and uniform bin weights prevent rare events from being dominated by dry-period samples.
- **Elegant formulation of Selective Scaling**: The strategy of "correcting confidence only for incorrect predictions" avoids disturbing already well-calibrated correct predictions.
- **FiLM conditioning as a domain adaptation mechanism**: Incorporating lead-time information into the calibrator is a key domain-specific innovation that benefits all methods.
- **Extreme parameter efficiency**: The best-performing method requires only 3,254 additional parameters, making deployment overhead negligible.

## Limitations & Future Work

- Experiments are conducted exclusively on MRMS data over North America; geographic generalizability remains unknown.
- Calibrators operate in logit space and do not exploit spatial context from raw inputs such as satellite imagery.
- Uniform ETCE weights may be suboptimal for applications that prioritize high-precipitation thresholds.
- The combination of Selective Scaling and LTS has not been explored.
- Future work could consider incorporating spatial and/or temporal information as additional conditioning signals.

## Related Work & Insights

- The transfer of calibration techniques from computer vision to meteorological forecasting warrants attention; analogous cross-domain transfers may prove effective in other scientific prediction tasks.
- FiLM conditioning serves as a general-purpose mechanism for injecting contextual information (e.g., time, location) into calibration models, with broad applicability.
- The "correct only the errors" principle underlying Selective Scaling is generalizable to other probabilistic prediction tasks.

## Rating

- **Novelty**: 3/5 — ETCE and FiLM-conditioned calibration represent reasonable incremental contributions.
- **Technical Depth**: 3/5 — Methods are relatively straightforward, though the experimental design is rigorous.
- **Experimental Thoroughness**: 3.5/5 — Ablation studies are comprehensive, but evaluation is limited to a single dataset.
- **Value**: 4/5 — The method is extremely lightweight and directly deployable in operational weather forecasting systems.
- **Writing Quality**: 4/5 — Well-structured with clear figures and tables.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[NeurIPS 2025\] Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)
- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)
- [\[NeurIPS 2025\] Diffusion Transformers as Open-World Spatiotemporal Foundation Models](diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)
- [\[NeurIPS 2025\] Improving Time Series Forecasting via Instance-aware Post-hoc Revision (PIR)](improving_time_series_forecasting_via_instance-aware_post-hoc_revision.md)

<!-- RELATED:END -->
