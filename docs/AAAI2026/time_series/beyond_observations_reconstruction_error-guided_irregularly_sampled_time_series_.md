---
title: >-
  [Paper Note] iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning
description: >-
  [AAAI 2026][Time Series][Irregularly sampled time series] This paper proposes iTimER, which leverages the model's own reconstruction error distribution as a learning signal. By estimating the error distribution from obse…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Irregularly sampled time series"
  - "reconstruction error"
  - "self-supervised pretraining"
  - "Wasserstein alignment"
  - "pseudo-observations"
date: 2026-05-08
content_hash: 43cfaa3fb8b081f6
---

# iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning

**Conference**: AAAI 2026
**arXiv**: [2511.06854](https://arxiv.org/abs/2511.06854)  
**Code**: N/A  
**Area**: Time Series / Self-Supervised Learning
**Keywords**: Irregularly sampled time series, reconstruction error, self-supervised pretraining, Wasserstein alignment, pseudo-observations

## TL;DR
This paper proposes iTimER, which leverages the model's own reconstruction error distribution as a learning signal. By estimating the error distribution from observed points and sampling from it to generate pseudo-observations at unobserved timestamps, the method aligns the error distributions of observed and pseudo-observed regions via Wasserstein distance combined with contrastive learning, achieving state-of-the-art performance on classification, interpolation, and forecasting tasks for irregularly sampled time series.

## Background & Motivation

**Background**: Irregularly sampled time series (ISTS) are ubiquitous in domains such as healthcare and meteorology, characterized by asynchronous sampling across variables, non-uniform time intervals, and extensive natural missingness. Existing approaches either impute first and then learn, or perform end-to-end modeling.

**Limitations of Prior Work**: (a) Imputation-based methods are unreliable under high missing rates and may introduce noise bias; (b) end-to-end methods derive learning signals only from observed values, leaving model behavior in unobserved regions unconstrained; (c) self-supervised masked reconstruction methods treat reconstruction error merely as a loss term, ignoring the uncertainty information it encodes.

**Key Challenge**: Unobserved timestamps lack ground-truth values, making direct supervision infeasible—yet the model's reconstruction error itself encodes the model's understanding of data structure and can serve as a proxy signal.

**Key Insight**: Reconstruction error is not merely a loss term but an information source reflecting model uncertainty and inductive preference. Its distribution can be propagated to unobserved regions to generate pseudo-observations.

**Core Idea**: Sample from the reconstruction error distribution estimated at observed points, mix with the nearest observed values via mixup to generate pseudo-observations, and enforce consistency between the error distributions of observed and pseudo-observed regions using Wasserstein distance.

## Method

### Overall Architecture
Encoder–decoder self-supervised pretraining proceeds as follows:
1. Reconstruct observed values → estimate a Gaussian distribution $\mathcal{N}(\mu_\epsilon, \sigma_\epsilon^2)$ of reconstruction errors.
2. Sample from the error distribution and apply mixup with the nearest observed value → generate pseudo-observations at unobserved timestamps.
3. Encode and reconstruct the pseudo-observation sequence → align the two error distributions via Wasserstein distance.
4. Apply contrastive learning to enhance representational discriminability.

### Key Designs

1. **Reconstruction Error Distribution Modeling and Pseudo-Observation Generation**:

    - **Function**: Transforms reconstruction error from "a loss to be minimized" into "an exploitable learning signal."
    - **Mechanism**: For observed points $m_t=1$, compute $\epsilon_t = x_t - \hat{x}_t$ and estimate $\mu_\epsilon, \sigma_\epsilon$ under a Gaussian assumption with momentum update $\rho$. For unobserved points $m_t=0$: $\tilde{x}_t = \alpha_t \cdot \bar{x} + (1-\alpha_t) \cdot \tilde{\epsilon}_t$, where $\tilde{\epsilon}_t \sim \mathcal{N}(\mu_\epsilon^h, (\sigma_\epsilon^h)^2)$.
    - **Design Motivation**: Pseudo-observations preserve temporal continuity (via mixup with nearest observations) while incorporating noise-aware uncertainty (via error sampling), making them more principled than naive imputation or random noise injection.

2. **Wasserstein Error Distribution Alignment**:

    - **Function**: Ensures consistent model behavior across observed and pseudo-observed regions.
    - **Mechanism**: $L_W = \|\mu_\epsilon - \mu_p\|^2 + \|\sigma_\epsilon - \sigma_p\|^2$ (closed-form 2-Wasserstein distance under Gaussian distributions).
    - **Design Motivation**: If the reconstruction error distribution over pseudo-observed regions resembles that over observed regions, the pseudo-observations have not introduced structural bias.

3. **Contrastive Learning + Dual Reconstruction Loss**:

    - **Function**: Enhances representational discriminability and robustness.
    - **Total loss**: $L = \alpha L_W + \beta L_{contrast} + \frac{1}{2}(L_{orig\_rec} + L_{pseudo\_rec})$.

## Key Experimental Results

### Main Results
Classification tasks (P12, P19 medical datasets; PAM activity recognition):

| Method | P12 AUROC | P19 AUROC | PAM Accuracy |
|---|---|---|---|
| Warpformer | 83.4 | 88.8 | 94.3 |
| mTAND | 84.2 | 84.4 | 92.9 |
| Raindrop | 82.8 | 87.0 | 88.5 |
| **iTimER** | **85.1+** | **89.2+** | **95.0+** |

iTimER also achieves comprehensive improvements on interpolation and forecasting tasks.

### Key Findings
- Effectiveness of reconstruction error as a learning signal: removing pseudo-observation generation leads to a notable performance drop.
- Wasserstein alignment is critical: it ensures pseudo-observations do not introduce distributional bias.
- Task-agnostic pretraining: a single pretrained model transfers to classification, interpolation, and forecasting downstream tasks.

## Highlights & Insights
- The insight of **"reconstruction error as signal"** is particularly profound: extracting useful information from the model's own imperfections is a principle transferable to any self-supervised learning scenario.
- Mixup-based pseudo-observation generation is more physically motivated than direct imputation or noise injection.
- The Wasserstein distance is computed efficiently via the closed-form solution under Gaussian assumptions.

## Limitations & Future Work
- The Gaussian assumption may not hold for all error distributions.
- Using only the nearest observed value as the anchor may be insufficient for long-gap missing regions.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The core insight of treating reconstruction error as a learning signal is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across three task types and multiple datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear and the motivation is well-argued.
- **Value**: ⭐⭐⭐⭐ Establishes a new paradigm for irregularly sampled time series representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)
- [\[AAAI 2026\] C3RL: Rethinking the Combination of Channel-independence and Channel-mixing from Representation Learning](c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](../../ICLR2026/time_series/gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[AAAI 2026\] CometNet: Contextual Motif-guided Long-term Time Series Forecasting](cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)
- [\[AAAI 2026\] Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints](mitigating_error_accumulation_in_co-speech_motion_generation_via_global_rotation.md)

</div>

<!-- RELATED:END -->
