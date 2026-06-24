---
title: >-
  [Paper Note] Causal Foundation Models: Disentangling Physics from Instrument Properties
description: >-
  [ICML 2025][Remote Sensing][Causal Foundation Models] Introduces a causally-driven foundation model that disentangles physical signals and instrumental effects from astronomical time series using a dual-encoder architecture and structured contrastive learning. By leveraging naturally occurring observational triplets (the same target observed by different instruments, or different targets observed by the same instrument), the proposed model significantly outperforms single lat…
tags:
  - "ICML 2025"
  - "Remote Sensing"
  - "Causal Foundation Models"
  - "Physics-Instrument Disentangling"
  - "Contrastive Learning"
  - "Time Series"
  - "Astronomical Observations"
date: 2026-05-08
content_hash: 8f065c9747286410
---

# Causal Foundation Models: Disentangling Physics from Instrument Properties

**Conference**: ICML 2025  
**arXiv**: [2507.05333](https://arxiv.org/abs/2507.05333)  
**Code**: None  
**Area**: Remote Sensing  
**Keywords**: Causal Foundation Models, Physics-Instrument Disentangling, Contrastive Learning, Time Series, Astronomical Observations

## TL;DR
Introduces a causally-driven foundation model that disentangles physical signals and instrumental effects from astronomical time series using a dual-encoder architecture and structured contrastive learning. By leveraging naturally occurring observational triplets (the same target observed by different instruments, or different targets observed by the same instrument), the proposed model significantly outperforms single latent space approaches in low-data regimes.

## Background & Motivation

### Background

**Background**: Foundation models for astronomical time series are developing rapidly, but observational data inherently merges real physical signals (stellar variability) with systematic instrumental effects (sensor drift, calibration biases).

**Limitations of Prior Work**: Existing foundation models encode all sources of variation into a single latent space, leading to poor generalization capabilities across different instruments.

**Key Challenge**: Physical signals and instrumental effects are heavily entangled, whereas foundation models need to reason about both components independently.

**Goal**: To learn disentangled latent space representations for both physical and instrumental attributes.

**Key Insight**: Utilizing the natural structure of astronomical observations—where the same star is observed by different instruments (shared physics, different instruments) and different stars are observed by the same instrument (different physics, shared instrument)—to construct positive and negative sample pairs for contrastive learning.

**Core Idea**: A dual-encoder architecture combined with structured contrastive learning, where one encoder captures physical information and the other captures instrumental properties.

## Method

### Overall Architecture
1. Dual Encoders: $E_{\text{phys}}$ extracts physical representations, and $E_{\text{inst}}$ extracts instrumental representations.
2. Structured Contrastive Learning: Leveraging triplet relationships—observations of the same star under different instruments should yield similar physical representations, while observations with the same instrument on different stars should yield similar instrumental representations.
3. Downstream tasks are trained based on either of the representations.

### Key Designs

1. **Dual-Encoder Architecture**:

    - **Function**: To learn independent latent spaces for physics and instruments respectively.
    - **Mechanism**: Employs two Transformer encoders to map time series data onto $z_{\text{phys}}$ and $z_{\text{inst}}$ respectively.
    - **Design Motivation**: The underlying causal structure requires physical and instrumental variables to be mutually independent.

2. **Structured Contrastive Learning**:

    - **Function**: To construct positive and negative sample relationships using observational triplets.
    - **Mechanism**: For star $s$, observations under instruments $m_1$ and $m_2$ should share a similar $z_{\text{phys}}$; for instrument $m$, observations of different stars should share a similar $z_{\text{inst}}$.
    - **Design Motivation**: Eliminates the need for explicit labeling by exploiting the natural pairing structure of observations.

### Loss & Training
- InfoNCE contrastive losses applied to both physical and instrumental representations.
- Regularization to ensure the independence of the two latent spaces.

## Key Experimental Results

### Main Results
Downstream prediction on simulated TESS astronomical time series:

| Method | Stellar Parameters R² ↑ | Instrument Parameters R² ↑ | Cross-Instrument Gen. R² ↑ |
|------|-------------|-------------|---------------|
| Single Latent Space Baseline | 0.72 | 0.68 | 0.41 |
| Contrastive Learning Baseline | 0.78 | 0.74 | 0.52 |
| **Causal Model (Physics)** | **0.91** | 0.12 | **0.83** |
| **Causal Model (Instrument)** | 0.08 | **0.89** | 0.15 |

### Few-Shot Performance (Only 10% Labeled Data)

| Method | 10-shot R² | 50-shot R² | 100-shot R² |
|------|-----------|-----------|------------|
| Fine-Tuning Baseline | 0.35 | 0.52 | 0.61 |
| **Causal Model** | **0.62** | **0.78** | **0.85** |

### Ablation Study

| Configuration | Stellar Parameters R² | Disentanglement (MI↓) |
|------|-----------|-------------|
| Full Model | 0.91 | 0.03 |
| w/o Contrastive Loss | 0.79 | 0.21 |
| Single Encoder | 0.72 | 0.45 |

### Key Findings
- The physical encoder contains almost no instrumental information ($R^2=0.12$), and the instrumental encoder contains almost no physical information ($R^2=0.08$), demonstrating successful disentanglement.
- Under low-data scenarios (few-shot), the advantages of the causal model are more pronounced.
- Cross-instrument transfer is effectively supported.

## Highlights & Insights
- The combination of **causal structures + contrastive learning** is both natural and highly effective.
- Utilizing the "natural experiment" structure of astronomical observations represents an elegant way of data utilization.
- The methodology can be generalized to any observational data with similar causal structures (e.g., meteorology, medical sensors, etc.).

## Limitations & Future Work
- Evaluated only on simulated data; testing on real-world astronomical data is pending.
- Assumes that physical and instrumental effects are fully separable, whereas interactions may exist in reality.
- Calls for paired "multi-instrument observations of the same target" dataset.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant application of causal foundation models on scientific data.
- Experimental Thoroughness: ⭐⭐⭐ Simulated data only.
- Writing Quality: ⭐⭐⭐⭐ Clear causal motivations.
- Value: ⭐⭐⭐⭐ Significant methodological implications for scientific foundation models.


## Related Work & Insights
- **vs. Representative Methods in the Same Field**: This work makes unique contributions in methodological architecture, complementing existing approaches.
- **vs. Traditional Methods**: Compared to traditional methods, the proposed approach achieves significant improvements on key metrics.
- **Insights**: The technical pipeline of this work provides valuable insights for future research in related domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models](mapeval_a_map-based_evaluation_of_geo-spatial_reasoning_in_foundation_models.md)
- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](../../NeurIPS2025/remote_sensing/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)
- [\[NeurIPS 2025\] Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](../../NeurIPS2025/remote_sensing/mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)
- [\[ICML 2026\] The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench](../../ICML2026/remote_sensing/the_perception-physics_paradox_probing_scientific_alignment_with_tc-bench.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)

</div>

<!-- RELATED:END -->
