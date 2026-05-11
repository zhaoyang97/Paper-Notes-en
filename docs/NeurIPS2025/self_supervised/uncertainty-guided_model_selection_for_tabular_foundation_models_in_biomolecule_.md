---
title: >-
  [Paper Note] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction
description: >-
  [NeurIPS 2025 (Workshop: Multi-modal Foundation Models for Life Sciences)][Self-Supervised Learning][TabPFN] This paper proposes OligoICP, a method that leverages the interquartile range (IQR) of TabPFN's predicted distr…
tags:
  - "NeurIPS 2025 (Workshop: Multi-modal Foundation Models for Life Sciences)"
  - "Self-Supervised Learning"
  - "TabPFN"
  - "siRNA efficacy prediction"
  - "uncertainty-guided selection"
  - "posterior ensemble"
  - "model selection"
date: 2026-05-08
content_hash: ae8e00983828e773
---

# Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction

**Conference**: NeurIPS 2025 (Workshop: Multi-modal Foundation Models for Life Sciences)  
**arXiv**: [2510.02476](https://arxiv.org/abs/2510.02476)  
**Code**: None  
**Area**: Bioinformatics, Tabular Foundation Models  
**Keywords**: TabPFN, siRNA efficacy prediction, uncertainty-guided selection, posterior ensemble, model selection

## TL;DR

This paper proposes OligoICP, a method that leverages the interquartile range (IQR) of TabPFN's predicted distributions as an unlabeled model selection heuristic, achieving superior performance over both specialized SOTA models and naive ensembles on siRNA knockdown efficiency prediction.

## Background & Motivation

- siRNA silences target genes by cleaving mRNA transcripts and represents a promising therapeutic modality, yet designing siRNAs with high knockdown efficiency remains a key challenge.
- Biomolecule efficacy datasets are typically **small, heterogeneous, and derived from diverse experimental technologies**.
- In-context learners such as TabPFN excel on small tabular datasets, but their performance is **highly sensitive to the provided context**.
- Simply using more data does not guarantee improvement — large datasets may exceed the computational limits of ICL or diverge from the pretraining distribution.
- **Core Problem**: How can one select the best models for an ensemble without access to labeled validation data?

## Method

### Overall Architecture

The OligoICP pipeline:
1. Construct feature sets (one-hot + trimer counts + thermodynamic parameters = 574-dimensional features).
2. Train an ensemble of 400 TabPFN models, each on a randomly sampled training subset of size $k$, where $k \in [1, 20]$.
3. Use the IQR of each model's predicted distribution as an uncertainty measure.
4. Select the top 10% of models with the lowest mean IQR and aggregate their predictions by averaging.

### Key Designs

**Feature Engineering** (574 dimensions):
- siRNA 19-mer one-hot encoding ($4 \times 19 = 76$ dimensions)
- mRNA 57-nt one-hot encoding ($5 \times 57 = 285$ dimensions)
- siRNA trimer counts (64 dimensions) + mRNA trimer counts (125 dimensions)
- Thermodynamic parameters (Gibbs free energy change, enthalpy change, etc.)

**Uncertainty Measure — IQR**:
- TabPFN can output quantile estimates of the predictive distribution.
- $\text{IQR} = Q_{85\%} - Q_{15\%}$ (expected to cover 70% of in-distribution data).
- Key finding: IQR is **negatively correlated** with true prediction accuracy (higher IQR → lower accuracy).
- After model-level aggregation, the Pearson $r$ between mean IQR and model correlation is $-0.42$.

**Model Selection Strategy**:
- Full ensemble mean: average predictions from all 400 models.
- OligoICP: average only the top 10% of models with the lowest IQR (~40 models).
- Baseline: a single model trained on all available data.

### Datasets

- Huesken dataset: 2,361 data points, 29 mRNA targets.
- Target1: 295 + 366 + 9 data points (from patents of 3 institutions).
- Target2: 252 data points.

## Key Experimental Results

### TabPFN vs. Specialized SOTA (OligoFormer)

| Dataset | TabPFN MAE↓ | OligoFormer MAE↓ | TabPFN Corr↑ | OligoFormer Corr↑ |
|--------|-----------|----------------|-------------|------------------|
| Huesken (ID) | 0.087±0.004 | 0.096 | 0.677±0.042 | 0.630 |
| Target1 (A, OOD) | 0.245 | 0.251 | 0.244 | 0.158 |
| Target1 (B, OOD) | 0.159 | 0.180 | 0.200 | 0.082 |

### Model Selection Strategy Comparison

| Dataset | OligoICP MAE | Full Ensemble MAE | All Data Single MAE | Oracle Best MAE |
|--------|-------------|-------------------|--------------------|--------------------|
| Target1 (A) | 0.270±0.005 | 0.268±0.002 | 0.278 | 0.197 |
| Target1 (B) | 0.174±0.001 | 0.169±0.001 | 0.172 | 0.149 |
| Target2 | **0.185±0.001** | 0.189±0.001 | 0.186 | 0.161 |

| Dataset | OligoICP Corr | Full Ensemble Corr | All Data Single Corr | Oracle Best Corr |
|--------|-------------|-------------------|--------------------|--------------------|
| Target1 (A) | **0.278±0.015** | 0.257±0.012 | 0.051 | 0.544 |
| Target1 (B) | 0.072±0.005 | 0.086±0.020 | 0.112 | 0.430 |
| Target2 | **0.246±0.015** | 0.230±0.002 | 0.230 | 0.384 |

### Key Findings

- TabPFN with simple features **outperforms the specialized OligoFormer**, particularly in OOD settings.
- A discernible negative correlation exists between IQR and prediction error.
- OligoICP yields substantial improvements in correlation (Target1(A): $0.051 \to 0.278$, a 5×+ gain).
- A notable gap remains relative to the Oracle best model ($0.278$ vs. $0.544$), indicating room for improvement in the selection strategy.
- The correlation of a single full-data model can be extremely low ($0.051$), demonstrating that "more data ≠ better performance."

## Highlights & Insights

- **General-purpose tabular models can surpass domain-specific models** — challenging the assumption that specialized models are always superior.
- Using IQR as an unlabeled model selection criterion is conceptually simple yet effective.
- The approach provides a natural solution for handling large-scale contextual data that exceeds the single-forward-pass limit of ICL.
- The additional computational cost is manageable, as each model processes only a limited amount of data and inference can be parallelized.

## Limitations & Future Work

- Improvements in MAE are marginal; the primary gains are observed in correlation.
- Target1(B) remains challenging for all methods, and OligoICP fails to improve performance there.
- Oracle results indicate substantial remaining potential for better model selection strategies.
- Validation is limited to siRNA tasks; extension to broader biomolecule prediction tasks is needed.
- The feature dimensionality (574) exceeds TabPFN's pretraining limit, requiring the use of a flag that bypasses this constraint.
- No comparison is made against more sophisticated context selection strategies such as LoCalPFN.

## Related Work & Insights

- This work represents one of the first applications of TabPFN/TabPFNv2's success on small tabular datasets to the biomedical domain.
- Posterior ensembling with model selection is a classic strategy in AutoML systems such as Auto-sklearn, but IQR-guided selection constitutes a novel contribution.
- The approach has direct practical value for sequence design tasks in drug discovery.

## Rating

⭐⭐⭐ — Strongly practical; the finding that a general-purpose model outperforms specialized counterparts is valuable. However, methodological novelty is limited and the experimental scale is relatively small.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[ICLR 2026\] Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](../../ICLR2026/self_supervised/regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)

</div>

<!-- RELATED:END -->
