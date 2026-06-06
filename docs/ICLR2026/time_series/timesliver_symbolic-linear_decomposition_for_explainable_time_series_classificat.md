---
title: >-
  [Paper Note] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification
description: >-
  [ICLR 2026][Time Series][Temporal Attribution] TimeSliver is an explainability-driven deep learning framework that jointly leverages raw time series data and symbolic abstractions (binning) to construct representations t…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Temporal Attribution"
  - "Symbolic Abstraction"
  - "Linear Combination"
  - "Explainable Classification"
  - "Positive/Negative Attribution"
date: 2026-05-08
content_hash: 88bdb473bfb2ebd6
---

# TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification

**Conference**: ICLR 2026
**arXiv**: [2601.21289](https://arxiv.org/abs/2601.21289)  
**Code**: [GitHub](https://github.com/pandeyakash23/TimeSliver)  
**Area**: Time Series / Explainability
**Keywords**: Temporal Attribution, Symbolic Abstraction, Linear Combination, Explainable Classification, Positive/Negative Attribution

## TL;DR
TimeSliver is an explainability-driven deep learning framework that jointly leverages raw time series data and symbolic abstractions (binning) to construct representations that preserve the original temporal structure. Each element linearly encodes the contribution of its corresponding temporal segment to the final prediction, yielding per-timestep positive/negative attribution scores. TimeSliver surpasses competing methods by 11% in temporal attribution accuracy across 7 datasets while achieving performance on par with SOTA on 26 UEA benchmarks.

## Background & Motivation

### State of the Field

**Background**: DL models (CNN/LSTM/Transformer) achieve strong classification performance but lack interpretability. Explainability is critical for high-stakes applications (healthcare/finance/law).

### Limitations of Prior Work

**Limitations of Prior Work**: (1) Post-hoc explanations (DeepLIFT/IntGrad/SHAP) are sensitive to baseline choice, assume feature independence, and do not generalize across datasets.

### Root Cause

**Key Challenge**: (2) Attention weights in Transformers are not faithful proxies for true attribution.

### Resolution Direction

**Resolution Direction**: (3) MIL-based methods have not been extended to multivariate settings.

### Additional Notes

**Additional Notes**: (4) Inability to distinguish positive from negative attribution — i.e., whether a segment "pushes toward" or "pushes away from" the predicted class.

**Key Insight**: Design an intrinsically interpretable architecture in which linear combination guarantees that attributions are directly computable without relying on post-hoc methods.

## Method

### TimeSliver Three-Module Architecture

**Module I: Temporal Segment Representation Learning (Q)**
- The time series is segmented; each segment is encoded to produce a segment-level representation vector.
- Segment-to-original-time-position correspondence is preserved.

**Module II: Symbolic Abstraction Latent Vector (Z)**
- The time series is symbolized via binning to produce a symbolic representation.
- An encoder maps this to a latent temporal vector $Z$.
- Symbolization suppresses high-frequency noise and captures pattern-level information.

**Module III: Linear Combination → Interpretable Representation**
- **Key operation**: $R = Z \odot Q$ (element-wise linear combination)
- $R$ is passed directly and linearly to the classification layer: $\hat{y} = W \cdot R + b$
- By linearity, the contribution of each temporal segment equals $W \cdot (Z_k \cdot Q_k)$.
- Positive contributions push toward the predicted class; negative contributions push away.

### Attribution Score Computation ($f_\text{att}$)
- Positive attribution $\phi_k^+$: the positively contributing portion of $W \cdot (Z_k \cdot Q_k)$ toward the predicted class.
- Negative attribution $\phi_k^-$: the negatively contributing portion toward the predicted class.
- Non-parametric operation — computed directly from representations and weights without approximation.

### Fundamental Distinction from Post-Hoc Methods

| Property | Post-Hoc Methods | **TimeSliver** |
|----------|-----------------|---------------|
| Attribution source | Gradient / perturbation | Architecturally intrinsic |
| Baseline dependency | Yes | No |
| Positive / negative attribution | Not distinguished | **Distinguished** |
| Faithfulness | Questionable | **Guaranteed (linear)** |

## Key Experimental Results

### Temporal Attribution Quality (7 Datasets)

### Main Results

| Method | Attribution Accuracy | Notes |
|--------|---------------------|-------|
| DeepLIFT | Baseline | Post-hoc |
| IntGrad | Medium | Post-hoc |
| Grad-CAM | Low | Unsuitable for time series |
| SHAP | Medium | Slow |
| Attention | Low (unfaithful) | Intrinsic |
| **TimeSliver** | **+11%** | Intrinsic linear |

### Predictive Performance (26 UEA Benchmarks)

### Ablation Study

| Method | Mean Accuracy | Explainability |
|--------|--------------|---------------|
| SOTA (various) | Best | None |
| **TimeSliver** | **−2% (on par)** | **Strong** |

### Key Findings
- Linear combination does not sacrifice predictive capacity — explainability and performance are not in conflict.
- Positive/negative attribution reveals which temporal segments "support" vs. "oppose" the prediction, providing richer information than scalar attribution.
- Symbolic abstraction helps the model ignore irrelevant fluctuations and focus on structural patterns.
- Cross-domain consistency — effective across audio, sleep staging, and fault diagnosis tasks.

## Highlights & Insights
- **"Linearity as a guarantee of interpretability"**: Rather than approximating attributions via complex post-hoc methods, TimeSliver uses a linear architecture to ensure exact attribution at the design level.
- **Information richness of positive + negative attribution**: Knowing that a segment "supports" the prediction is insufficient; knowing which segments "oppose" it provides a complete picture for decision-making.
- **Elegance of symbolic abstraction**: Binning is conceptually simple yet compresses unnecessary detail, directing the model toward structural patterns rather than raw numerical values — analogous to human temporal reasoning.
- **Pareto frontier of prediction and explainability**: TimeSliver performs well on both axes without sacrificing one for the other.

## Rating
- Novelty: ⭐⭐⭐⭐ Architectural innovation via symbolic-linear decomposition
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 attribution datasets + 26 UEA benchmarks + 12 baselines
- Writing Quality: ⭐⭐⭐⭐ Explainability concepts clearly articulated
- Value: ⭐⭐⭐⭐ Significant contribution to explainable time series analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](../../AAAI2026/time_series/counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[ICLR 2026\] WARP: Weight-Space Linear Recurrent Neural Networks](weight-space_linear_recurrent_neural_networks.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)

</div>

<!-- RELATED:END -->
