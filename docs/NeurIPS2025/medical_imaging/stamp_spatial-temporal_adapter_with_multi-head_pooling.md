---
title: >-
  [Paper Note] STAMP: Spatial-Temporal Adapter with Multi-Head Pooling
description: >-
  [NeurIPS 2025][Medical Imaging][EEG Classification] STAMP introduces a lightweight spatial-temporal adapter with only 750K parameters for Time Series Foundation Models (TSFMs). Through three sets of positional encodings (token/spatial/temporal), cross-gated MLP mixing, and multi-head attention pooling, it enables a frozen TSFM (e.g., MOMENT 385M) to compete with or surpass EEG-specific models with 29M parameters (CBraMod) across 8 EEG datasets, achieving 193% higher Kappa than CBraMod on BCIC-IV-2a.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - EEG Classification
  - TSFM Adapter
  - Spatial-Temporal Encoding
  - Multi-Head Pooling
  - Parameter Efficiency
date: 2026-05-08
content_hash: 84312c5cd5c62eeb
---

# STAMP: Spatial-Temporal Adapter with Multi-Head Pooling

**Conference**: NeurIPS 2025
**arXiv**: [2511.10848](https://arxiv.org/abs/2511.10848)
**Code**: [https://github.com/autonlab/STAMP](https://github.com/autonlab/STAMP)
**Area**: EEG Signals / Foundation Model Adaptation
**Keywords**: EEG Classification, TSFM Adapter, Spatial-Temporal Encoding, Multi-Head Pooling, Parameter Efficiency

## TL;DR
STAMP introduces a lightweight spatial-temporal adapter with only 750K parameters for Time Series Foundation Models (TSFMs). Through three sets of positional encodings (token/spatial/temporal), cross-gated MLP mixing, and multi-head attention pooling, it enables a frozen TSFM (e.g., MOMENT 385M) to compete with or surpass EEG-specific models with 29M parameters (CBraMod) across 8 EEG datasets, achieving 193% higher Kappa than CBraMod on BCIC-IV-2a.

## Background & Motivation

**State of the Field**: TSFMs (e.g., MOMENT, Chronos), pretrained across multiple domains, have demonstrated strong general-purpose representation capabilities. EEG-specific foundation models (CBraMod, LaBraM) perform well on EEG classification but require large parameter counts (29M/5.8M) and EEG-specific pretraining.

**Limitations of Prior Work**: TSFMs process univariate time series — EEG is spatiotemporal data with 64 channels × 1000+ time steps, making direct application infeasible for the spatial dimension. Feeding each channel independently into a TSFM discards inter-channel spatial relationships.

**Root Cause**: TSFMs possess strong temporal representations but lack spatial understanding; EEG models understand spatial structure but require large-scale EEG pretraining. The core challenge is enabling TSFMs to understand EEG's spatial dimension at minimal cost.

**Paper Goals**: Design a lightweight adapter that enables general-purpose TSFMs to efficiently process spatiotemporal EEG data.

**Starting Point**: Freeze the TSFM and train only a 750K-parameter adapter — three sets of positional encodings inject spatial information, cross-gated MLP mixes spatial and temporal features, and multi-head attention pooling aggregates representations.

**Core Idea**: Triple positional encoding (token + spatial + temporal) + cross-gated MLP spatiotemporal mixing + multi-head pooling = 750K parameters enabling a frozen TSFM to process EEG spatiotemporal data.

## Method

### Overall Architecture
EEG data ($S$ channels × $T$ time steps) → frozen TSFM (e.g., MOMENT-L 385M) encodes to $S \times T' \times D$ (downsampled 1024→128) → **Positional Encoding** ($\tilde{e}_{ij} = e'_{ij} + p_{ij} + s_i + t_j$) → **CC-GMLP** (decoupled spatial and temporal gated mixing) → **MHAP** (multi-head attention pooling to fixed-length vector) → Classification Head

### Key Designs

1. **Three-Set Positional Encoding (PE-NST)**:

    - Function: Injects spatial-temporal positional information into TSFM output tokens.
    - Mechanism: Token-wise PE $p_{ij} \in \mathbb{R}^D$ provides an independent embedding for each (channel, time) position; Spatial PE $s_i$ encodes channel identity (e.g., C3/C4/Oz); Temporal PE $t_j$ encodes temporal position. All three are summed.
    - Design Motivation: Ablation studies demonstrate that all three PE components are necessary — token PE alone is insufficient (lacking general spatial/temporal structure), and spatial + temporal PE alone is insufficient (lacking position-specific embeddings).

2. **Cross-Channel GMLP (CC-GMLP)**:

    - Function: Performs feature mixing along spatial and temporal dimensions respectively.
    - Mechanism: Spatial gating $g_S(Z) = Z_1 \odot (W \cdot Z_2)$ (mixing along the spatial dimension); temporal gating operates analogously along the temporal dimension. Both operate independently to preserve spatiotemporal disentanglement.
    - Design Motivation: Transformers are parameter-heavy for spatiotemporal sequences; GMLP is more efficient, and the CC (cross-channel) variant further reduces parameters (0.74M vs. GMLP 0.79M) while achieving better performance.

3. **Multi-Head Attention Pooling (MHAP)**:

    - Function: Aggregates variable-length spatiotemporal tokens into a fixed-length classification vector.
    - Mechanism: Multiple learnable query vectors aggregate token information via attention weights. Final classification: $\hat{y} = \text{softmax}(W(\lambda z + (1-\lambda)\hat{e}))$
    - Design Motivation: More flexible than mean pooling — capable of learning to attend to different temporal segments and spatial regions.

### Loss & Training
- Standard cross-entropy classification loss
- MOMENT-Large (385M) is frozen; adapter contains 750K trainable parameters
- Compatible with multiple TSFMs (MOMENT S/B/L, Chronos, TSPulse)

## Key Experimental Results

### Main Results (8 EEG Datasets)

| Dataset | STAMP (750K) | CBraMod (29M) | LaBraM (5.8M) | Result |
|--------|-------------|--------------|--------------|------|
| SHU-MI | 0.660 AUC | 0.657 | 0.660 | Tie |
| MentalArith | **0.811** | 0.749 | 0.772 | STAMP wins |
| BCIC-IV-2a | **0.409 Kappa** | 0.212 | 0.316 | **+193%** |
| TUEV | 0.662 | 0.618 | **0.664** | Tie |
| SEED-V | 0.208 | **0.259** | 0.239 | CBraMod wins |
| FACED | 0.278 | **0.508** | 0.470 | CBraMod wins |

STAMP is competitive or superior on 6/8 datasets, with relative weakness on emotion recognition tasks.

### Ablation Study

| Variant | Description |
|------|------|
| PE-NST (all three sets) | Optimal |
| PE-ST (no token PE) | Performance degrades |
| CC-GMLP vs. Transformer | CC-GMLP outperforms on all 4 datasets with fewer parameters |
| MHAP vs. Mean Pooling | MHAP substantially better on BCIC-IV-2a; comparable elsewhere |
| Different TSFM backbones | MOMENT L > B > S; Chronos slightly better on emotion tasks; TSPulse stronger on event tasks |

### Key Findings
- A 750K-parameter adapter enables a general-purpose TSFM to match a 29M-parameter EEG-specific model on most tasks — 39× improvement in parameter efficiency.
- Emotion recognition (SEED-V, FACED) is a weakness for TSFMs, likely due to the absence of emotion-relevant features in TSFM pretraining.
- CC-GMLP is more efficient than Transformer for spatiotemporal mixing — suggesting EEG spatiotemporal relationships are relatively simple and do not require full attention.
- The choice of TSFM backbone has limited impact — architectural design matters more than pretraining data.

## Highlights & Insights
- **Exceptional Parameter Efficiency**: The combination of a 750K adapter and a frozen 385M TSFM outperforms a 29M EEG-specific model, validating that general temporal representations + lightweight spatiotemporal adaptation constitute a more efficient paradigm.
- **Elegant CC-GMLP Design**: Decoupled spatial and temporal gated mixing avoids the curse of dimensionality while preserving spatiotemporal interaction.
- **Discovery of TSFM Capability Boundaries**: Failure on emotion recognition reveals that TSFM pretraining signals lack affective semantics — EEG-specific pretraining may be necessary for such tasks.

## Limitations & Future Work
- Poor performance on emotion recognition tasks — TSFM pretraining data does not contain emotion-related neural signal patterns.
- Evaluation is limited to classification; temporal forecasting and generation tasks — the primary use case of TSFMs — are not assessed, as EEG prediction tasks remain scarce.
- The complete design with five positional encoding components introduces additional hyperparameters requiring tuning for different EEG devices.
- Despite being lightweight, the 750K adapter still requires separate training per dataset — zero-shot cross-dataset generalization is not validated.
- CC-GMLP assumes separability of spatial and temporal dimensions, which may be insufficient for tasks requiring joint spatiotemporal modeling (e.g., rapid-response brain-computer interfaces).
- Dependence on patch-level feature extraction from the TSFM may result in information loss for fine-grained events in raw EEG waveforms.

## Related Work & Insights
- **vs. CBraMod**: A 29M-parameter EEG-specific model; STAMP achieves comparable performance with a 750K adapter and a general-purpose TSFM.
- **vs. LaBraM**: A 5.8M-parameter model; STAMP matches or exceeds it on most tasks.
- **Insight**: The paradigm of freezing large pretrained models and training lightweight adapters, already successful in NLP and CV, is validated here for the EEG domain.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to adapt TSFMs for EEG spatiotemporal data
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 datasets + multiple TSFMs + comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and well-organized
- Value: ⭐⭐⭐⭐ Validates the feasibility of general-purpose TSFM + lightweight adaptation for EEG

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](../../CVPR2026/medical_imaging/t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)
- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[NeurIPS 2025\] RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis](rad_towards_trustworthy_retrieval-augmented_multi-modal_clinical_diagnosis.md)
- [\[NeurIPS 2025\] A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking](a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)

<!-- RELATED:END -->
