---
title: >-
  [Paper Note] Time-O1: Time-Series Forecasting Needs Transformed Label Alignment
description: >-
  [NeurIPS 2025][Time Series][Learning Objective] This paper proposes Time-O1, which addresses the autocorrelation bias and task overload of the TMSE loss in time series forecasting by transforming label sequences into dec…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Learning Objective"
  - "Label Autocorrelation"
  - "SVD Transform"
  - "Decorrelation"
date: 2026-05-08
content_hash: 73e07601b0be8030
---

# Time-O1: Time-Series Forecasting Needs Transformed Label Alignment

**Conference**: NeurIPS 2025
**arXiv**: [2505.17847](https://arxiv.org/abs/2505.17847)
**Code**: [Available](https://github.com/Master-PLC/Time-o1)
**Area**: Time Series Forecasting
**Keywords**: Time Series, Learning Objective, Label Autocorrelation, SVD Transform, Decorrelation

## TL;DR

This paper proposes Time-O1, which addresses the autocorrelation bias and task overload of the TMSE loss in time series forecasting by transforming label sequences into decorrelated, importance-ranked principal components. The method achieves state-of-the-art performance while remaining compatible with a wide range of forecasting models.

## Background & Motivation

Training objectives for time series forecasting models typically rely on the temporal mean squared error (TMSE), which computes step-wise discrepancies between predictions and label sequences. However, TMSE suffers from two fundamental deficiencies:

**Deficiency 1: Bias induced by label autocorrelation.** Time series are inherently autocorrelated (adjacent steps are highly correlated), yet TMSE treats each step as an independent task, ignoring inter-step correlations. According to Theorem 3.1, the bias between TMSE and the true likelihood of the label sequence is:

$$\text{Bias} = \|Y - \hat{Y}\|_{\Sigma^{-1}}^2 - \|Y - \hat{Y}\|^2 - \frac{1}{2}\log|\Sigma|$$

This bias vanishes only when label steps are decorrelated.

**Deficiency 2: Optimization difficulty from increasing prediction steps.** In long-term forecasting, the number of prediction steps $T$ can reach 720. TMSE treats each step as an independent task, and gradient conflicts in multi-task learning intensify as the number of tasks grows, making convergence difficult.

The prior work FreDF proposes frequency-domain alignment to address the bias; however, frequency-domain components are fully decorrelated only as $T \to \infty$, and residual correlations persist under finite prediction horizons. Furthermore, frequency-domain transformations do not reduce the number of tasks, leaving the optimization difficulty unresolved.

## Method

### Overall Architecture

The core mechanism of Time-O1 is to transform label sequences via an optimal projection matrix into decorrelated, importance-ranked principal components, and then align only the top-$K$ most important components during training. The final loss is a weighted combination of the transformed-domain loss and TMSE.

### Key Designs

1. **Solving the Optimal Projection Matrix**: Given the normalized label matrix $\mathbf{Y} \in \mathbb{R}^{m \times T}$, the projection matrix $\mathbf{P}^*$ is obtained by constrained optimization: projection directions are identified sequentially to maximize component variance, subject to mutual orthogonality. Formally, $\mathbf{P}_p^* = \arg\max_{\mathbf{P}_p} (\mathbf{Y}\mathbf{P}_p)^\top(\mathbf{Y}\mathbf{P}_p)$, with constraints $\|\mathbf{P}_p\|^2 = 1$ and $\mathbf{P}_p^\top \mathbf{P}_j = 0$.

    By Lemma 3.3, $\mathbf{P}^*$ can be efficiently computed via SVD: $\mathbf{Y} = \mathbf{U}\mathbf{\Lambda}(\mathbf{P}^*)^\top$. The transformed components $\mathbf{Z} = \mathbf{Y}\mathbf{P}^*$ satisfy decorrelation (Lemma 3.2) and are ordered by descending variance.

2. **Salient Component Selection and Task Reduction**: The top $K = \text{round}(\gamma \cdot T)$ most important components are retained, where $\gamma$ controls the retention ratio. The transformed-domain loss aligns only these $K$ components: $\mathcal{L}_{\text{trans},\gamma} = \|\hat{\mathbf{Z}}_{\cdot,1:K} - \mathbf{Z}_{\cdot,1:K}\|_1$. The $\ell_1$ norm is used instead of $\ell_2$, as the large variance discrepancy across components renders $\ell_2$ unstable.

3. **Combined Loss**: The final learning objective is a weighted combination of the transformed-domain loss and the original TMSE: $\mathcal{L}_{\alpha,\gamma} = \alpha \cdot \mathcal{L}_{\text{trans},\gamma} + (1-\alpha) \cdot \mathcal{L}_{\text{tmse}}$, where $\alpha$ controls the relative weighting.

### Loss & Training

Time-O1 is model-agnostic — it can directly replace the training loss of any forecasting model. The implementation pipeline is straightforward: normalize labels → compute principal components via SVD → project predictions and labels → compute the combined loss. Only two hyperparameters, $\alpha$ and $\gamma$, require tuning.

## Key Experimental Results

### Main Results (Long-Term Forecasting, 8 Datasets)

| Model | ETTm1 MSE | ETTm2 MSE | ETTh1 MSE | ECL MSE | Weather MSE |
|-------|-----------|-----------|-----------|---------|-------------|
| **Time-O1** | **0.380** | **0.272** | **0.431** | **0.170** | **0.241** |
| Fredformer | 0.387 | 0.280 | 0.447 | 0.191 | 0.261 |
| iTransformer | 0.411 | 0.295 | 0.452 | 0.179 | 0.269 |
| DLinear | 0.403 | 0.342 | 0.456 | 0.212 | 0.265 |
| TimesNet | 0.438 | 0.302 | 0.472 | 0.212 | 0.271 |

### Ablation Study (Comparison with Alternative Learning Objectives)

| Loss Function | ETTm1 MSE | ETTh1 MSE | Weather MSE | Notes |
|--------------|-----------|-----------|-------------|-------|
| **Time-O1** | **0.379** | **0.431** | **—** | Transform + TMSE fusion |
| FreDF | 0.384 | 0.447 | — | Frequency-domain alignment |
| Koopman | 0.389 | 0.452 | — | Koopman operator |
| Dilate | 0.389 | — | — | Shape alignment |
| DF (TMSE) | 0.387 | — | — | Baseline TMSE |

### Key Findings

- Time-O1 consistently improves baseline model performance across all 8 long-term forecasting datasets.
- On ETTh1, the MSE of Fredformer is reduced from 0.447 to 0.431 (a 3.6% decrease).
- Modifying only the learning objective yields improvements comparable to or exceeding those from architectural innovations.
- Time-O1 achieves stronger decorrelation than FreDF, as frequency-domain components retain residual correlations under finite prediction horizons.
- On the ETTh1 dataset, approximately 50.5% of inter-step label correlation coefficients exceed 0.25, confirming the severity of the autocorrelation problem.
- After transformation, only a small number of components carry large variance, enabling effective task reduction with $\gamma < 1$.

## Highlights & Insights

- **Applying PCA to label space is the core innovation**: Conventional PCA is used for input feature dimensionality reduction; this paper innovatively applies it to label sequence decorrelation and saliency discrimination.
- The theoretical analysis is rigorous and complete: from Theorem 3.1 (quantitative bias analysis) to Lemma 3.2/3.3 (decorrelation guarantees and SVD implementation), a coherent theoretical chain is established.
- The method is extremely lightweight — requiring only one SVD and a matrix multiplication, with no additional parameters.
- Model-agnosticism enables broad applicability, making it a strong candidate for adoption as a standard training technique.

## Limitations & Future Work

- The SVD projection matrix is computed from training-set labels, assuming consistent label distribution at test time.
- In multivariate settings, each variable is processed independently, without accounting for cross-variable correlations.
- The choice of $\gamma$ may vary across datasets and currently requires validation-set search.
- Only the $\ell_1$ norm is evaluated; a systematic comparison with other robust loss functions has not been conducted.

## Related Work & Insights

- FreDF is the most direct predecessor, proposing frequency-domain loss to mitigate bias, albeit with incomplete decorrelation.
- This work suggests that in any task requiring sequence alignment, accounting for the correlation structure of the label space may yield consistent improvements.
- The application of PCA to label space is generalizable to other sequential prediction tasks such as speech and NLP.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying PCA for label-space decorrelation and saliency discrimination represents a genuinely novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 8 datasets, multiple baseline models, comparisons with 5+ learning objectives, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and motivations are clearly articulated.
- **Value**: ⭐⭐⭐⭐⭐ — A model-agnostic training technique that can be directly applied to existing systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)

</div>

<!-- RELATED:END -->
