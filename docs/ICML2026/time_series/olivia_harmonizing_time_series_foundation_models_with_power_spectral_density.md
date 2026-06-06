---
title: >-
  [Paper Note] OLIVIA: Harmonizing Time Series Foundation Models with Power Spectral Density
description: >-
  [ICML 2026][Time Series][Power Spectral Density] OLIVIA significantly improves the pre-training of time series foundation models on heterogeneous data by introducing Power Spectral Density (PSD) driven coordination mecha…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Power Spectral Density"
  - "Time Series Foundation Models"
  - "Domain Adaptation"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: 42dd0ad67d555e58
---

# OLIVIA: Harmonizing Time Series Foundation Models with Power Spectral Density

**Conference**: ICML 2026  
**arXiv**: [2605.17340](https://arxiv.org/abs/2605.17340)  
**Code**: To be confirmed  
**Area**: Time Series / Foundation Models  
**Keywords**: Power Spectral Density, Time Series Foundation Models, Domain Adaptation, Attention Mechanism

## TL;DR
OLIVIA significantly improves the pre-training of time series foundation models on heterogeneous data by introducing Power Spectral Density (PSD) driven coordination mechanisms—the Harmonizer (orthogonal second-order coordination based on Householder reflections) and HarmonicAttention (low-dimensional interaction via resonators)—achieving SOTA across TSLib zero-shot, GIFT-Eval, and GluonTS benchmarks.

## Background & Motivation

**Background**: Time series foundation models learn unified general representations through pre-training on large-scale datasets from multiple domains—a paradigm proven effective in NLP and CV. However, existing models face severe challenges when handling heterogeneous time series.

**Limitations of Prior Work**: Time series from different domains exhibit significantly different temporal patterns (periodic structures, long-term dependencies). While this diversity is a prerequisite for learning broadly applicable temporal knowledge, it complicates pre-training—(1) at the optimization level: joint training on data with distinct temporal characteristics often leads to slow convergence and sub-optimality; (2) at the representation learning level: models must simultaneously adapt to incompatible temporal structures, making it difficult to form a unified transferable representation.

**Key Challenge**: Existing foundation models achieve domain adaptation through architectural modularity or capacity specialization (Mixture-of-Experts, frequency-aware patching), but **do not explicitly resolve the fundamental differences in temporal distributions**—namely, diagnosing and harmonizing cross-domain spectral disparities using the concept of PSD from signal processing.

**Goal**: (1) Understand and quantify cross-domain temporal heterogeneity in a principled manner; (2) efficiently achieve PSD consistency during large-scale pre-training without falling into direct, unstable divergence minimization.

**Key Insight**: Normalized PSD is a dataset-level descriptor that reflects the underlying second-order temporal correlation structure by capturing the distribution of temporal variations across frequencies. PSD is invariant to global temporal translations and relatively robust to local temporal misalignments, making it an ideal representation for comparing signals acquired under disparate conditions.

**Core Idea**: Reduce mismatches by harmonizing the PSD of each dataset in the spectral domain—reformulated from unfeasible direct divergence minimization into a structural coordination approach based on **shared reparameterization of second-order temporal correlations**.

## Method

### Overall Architecture
The encoder-decoder architecture integrates two core innovations—the **Harmonizer** (a PSD-driven transformation module that implicitly harmonizes spectral features by sharing reparameterized second-order correlations, aligning inputs before encoding and restoring to the original domain after decoding) and the **HarmonicFormer** (a Transformer-derived backbone using HarmonicAttention instead of dense token interactions, performing self-attention in a low-dimensional harmonic interaction space).

Information Flow: Raw Time Series $\to$ Harmonizer-Aligner (aligning to shared spectral space) $\to$ HarmonicFormer Encoding-Decoding $\to$ Harmonizer-Restorer (restoring to original domain) $\to$ Forecast Output.

### Key Designs

1. **Harmonizer: Orthogonal Second-Order Coordination based on Householder Reflections**:

    - **Function**: Projects time series into a shared canonical spectral space via a learned orthogonal matrix $Q$, implicitly harmonizing cross-dataset PSD consistency without explicit divergence computation.
    - **Mechanism**: Direct minimization of PSD Jensen-Shannon Divergence (JSD) is infeasible in large-scale pre-training (noisy gradients, unstable training). Starting from **Proposition 1** (there exists a shared orthogonal matrix $Q$ such that the subspace spanned by its first $r$ columns is invariant for the second-order moment matrices of all datasets, block-diagonalizing the covariance matrices), PSD coordination is achieved via shared reparameterized second-order correlations. Aligner $\mathcal{X} = X Q^\top$, where $Q$ is parameterized as a product of $K$ Householder reflections $Q = \prod_k H_k$, $H_k = I - 2 V_k V_k^\top$, ensuring $Q$ always resides in the orthogonal group. The Restorer performs the inverse mapping $Y = \mathcal{Y} Q$.
    - **Design Motivation**: Avoids direct unstable divergence minimization by using structural alignment through projection into a shared subspace; the bidirectional design maintains signal integrity and energy conservation; the step-by-step construction of Householder reflections ensures smooth gradient flow.

2. **HarmonicAttention: Low-Dimensional Harmonic Interaction via Resonators**:

    - **Function**: Reduces the $\mathcal{O}(L^2 P)$ dense token interaction of standard Transformers to $\mathcal{O}(L M P + M^2 P)$, where $M$ resonators ($M \ll L$) act as compact intermediates to efficiently conduct global dependencies.
    - **Mechanism**: **Proposition 2** states that the second-order moment matrix aligned by the Harmonizer has a block-diagonal structure $\Sigma_\mathcal{X} = \text{diag}(\Lambda, \Phi)$, where the token Gram matrix can be decomposed into a dominant low-rank term plus a bounded residual—providing a theoretical basis for approximating dense dependencies with compact harmonic patterns. Three stages: (1) Token-to-resonator aggregation $R^{(h)} = (A^{(h)})^\top \tilde{Z}^{(h)}$; (2) Inter-resonator interaction $\text{ResAct}(R^{(h)}) = \text{Softmax}_{\text{res}}(R^{(h)} (R^{(h)})^\top / \sqrt{P}) R^{(h)}$; (3) Global resonator projection $\text{Head}^{(h)} = A^{(h)} \text{ResAct}(R^{(h)})$.
    - **Design Motivation**: Mediates interactions through a resonator bottleneck to avoid dense $L \times L$ attention matrices, achieving linear scalability for long sequences; resonators act as a set of compact harmonic patterns that naturally encode dominant energy in the shared subspace.

3. **HarmonicFormer + Training Strategy**:

    - **Function**: Stacks HarmonicAttention encoders-decoders on top of Harmonizer alignment to form a scalable and expressive pre-training backbone.
    - **Mechanism**: Starting from PSD-consistent representations output by the Harmonizer, a Transformer-style encoder-decoder framework is constructed, replacing all standard multi-head self-attention with HarmonicAttention. Pre-training uses multi-task learning and a two-stage strategy (including masked modeling, regression, and other signals).
    - **Design Motivation**: Structural low-dimensional spectral interaction combined with Transformer expressive depth allows the theoretical elegance of the Harmonizer to be converted into practical performance gains through the efficient interaction mechanism of HarmonicFormer.

## Key Experimental Results

### Main Results (TSLib Zero-shot)

| Benchmark | Metric | Olivia | SEMPO | Time-MoE_B | Time-MoE_L | Moirai_B |
|-----------|--------|--------|-------|------------|------------|----------|
| ETTh1 | MSE | **0.399** | 0.410 | 0.445 | 0.435 | 0.433 |
| ETTh1 | MAE | **0.421** | 0.430 | 0.449 | 0.449 | 0.431 |
| Weather | MSE | **0.247** | 0.248 | 0.279 | 0.318 | 0.312 |
| Electricity | MSE | **0.188** | 0.196 | — | — | 0.207 |

### Ablation Study

| Configuration | ETTh1 MSE | Inference (s) | Model Size (M) |
|---------------|-----------|---------------|----------------|
| **HarmonicAttention** | **0.399** | 43.051 | **5.1** |
| w/o Harmonizer | 0.472 | — | — |
| Full Attention | 0.472 | — | — |
| Linear Attention | 0.412 | — | — |
| Nyström Attention | 0.488 | — | — |

### Key Findings
- Olivia achieves an average MSE reduction of 2.7% compared to SEMPO and 26.3% compared to the Time-MoE series on TSLib zero-shot.
- On GluonTS, improvements of 11.6–32% in NRMSE were observed compared to SEMPO, and over 86% compared to Time-MoE.
- The removal of the Harmonizer significantly worsens MSE (0.399 $\to$ 0.472), validating the core value of PSD coordination.
- HarmonicAttention gains performance from structural matching with PSD-consistent representations, rather than simple attention capacity.
- Olivia has the smallest parameter count (5.1M vs. SEMPO 6.5M, Time-MoE_B 113M).

## Highlights & Insights
- **PSD as a Fundamental Diagnostic Tool**: Systematically introducing Power Spectral Density for heterogeneity diagnosis in time series foundation models is more operational than generic "domain adaptation."
- **Elegant Transformation of Second-Order Moment Structure**: The dual theory of Propositions 1 and 2 skillfully transforms the seemingly un-optimizable PSD divergence problem into a block-diagonalization problem of second-order statistics—demonstrating how deep theoretical thinking guides model design.
- **Reusable Low-Dimensional Interaction Paradigm**: HarmonicAttention achieves efficient global dependency modeling via a "resonator bottleneck," offering potential value to any domain requiring self-attention on long sequences.
- **Unification of Representation Learning and Computational Efficiency**: Both are optimized simultaneously through PSD coordination—the Harmonizer improves representation learning while HarmonicAttention improves computational efficiency, serving as complements rather than trade-offs.

## Limitations & Future Work
- Inference latency is slightly high (constructing orthogonal matrices via Householder reflections introduces extra overhead, 43s vs. SEMPO 8s); more efficient orthogonal parameterizations (QR decomposition, Cayley transform) could be explored.
- The theoretical relationship between the resonator hyperparameter $M$ and the true signal rank $r$ is bounded, but how to align them and how they vary across datasets is not fully discussed.
- Applicability to heterogeneous downstream tasks (classification, anomaly detection) remains to be verified.

## Related Work & Insights
- **vs. SEMPO / Time-MoE / Moirai**: These achieve domain generalization through architectural modularity but lack explicit handling of fundamental cross-domain spectral differences; Olivia aligns specifically through PSD consistency constraints.
- **vs. ROSE**: ROSE uses spectral masking and adaptive registers to isolate domain-specific features; Olivia does the opposite—aggregating domain-specific variations into the orthogonal complement subspace through coordination.
- **vs. General Low-Rank Attention**: Linear / Nyström are general-purpose low-rank approximations; the resonators in HarmonicAttention are derived from PSD alignment structural principles, making them more sensitive to time-series-specific structures.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introduces PSD as a diagnostic tool for foundation model design; the integration of HarmonicAttention and Harmonizer is theoretically profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two large-scale benchmarks + 6 additional GluonTS datasets + comprehensive ablations + efficiency analysis, with consistent and significant results.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the correspondence between the two propositions and the method design is well-argued; some occasional details are omitted.
- Value: ⭐⭐⭐⭐⭐ Provides a theory-driven breakthrough in the frontier of time series foundation models; the PSD coordination idea is transferable to other multi-source heterogeneous data scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](../../NeurIPS2025/time_series/frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[ICML 2026\] FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](factorynet_a_large-scale_dataset_toward_industrial_time-series_foundation_models.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](../../NeurIPS2025/time_series/how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](../../NeurIPS2025/time_series/sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](../../ICLR2026/time_series/adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)

</div>

<!-- RELATED:END -->
