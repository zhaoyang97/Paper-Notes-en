---
title: >-
  [Paper Note] OmniSat: Self-Supervised Modality Fusion for Earth Observation
description: >-
  [ECCV 2024][Time Series][multi-modal fusion] This paper proposes OmniSat, a unified framework that fuses heterogeneous remote sensing data—including multi-spectral time-series (S2), SAR time-series (S1), and high-resolution single-temporal images (SPOT/Aerial)—into a unified representation using modality-specific encoders and cross-modal contrastive self-supervised pre-training. It outperforms all unimodal and multimodal baselines on semantic segmentation and crop classificat…
tags:
  - "ECCV 2024"
  - "Time Series"
  - "multi-modal fusion"
  - "earth observation"
  - "self-supervised"
  - "Sentinel"
  - "PASTIS"
date: 2026-05-08
content_hash: ceb858f6f73af9af
---

# OmniSat: Self-Supervised Modality Fusion for Earth Observation

**Conference**: ECCV 2024  
**arXiv**: [2404.08351](https://arxiv.org/abs/2404.08351)  
**Code**: [GitHub](https://github.com/gastruc/OmniSat)  
**Area**: Self-Supervised  
**Keywords**: multi-modal fusion, earth observation, self-supervised, Sentinel, PASTIS

## TL;DR
This paper proposes OmniSat, a unified framework that fuses heterogeneous remote sensing data—including multi-spectral time-series (S2), SAR time-series (S1), and high-resolution single-temporal images (SPOT/Aerial)—into a unified representation using modality-specific encoders and cross-modal contrastive self-supervised pre-training. It outperforms all unimodal and multimodal baselines on semantic segmentation and crop classification.

## Background & Motivation

**Background**: Earth observation has abundant multi-source data, such as S2 (multi-spectral time-series), S1 (SAR time-series), and high-resolution single-temporal images (SPOT/aerial), with resolutions ranging from 10m to 0.2m. Existing methods mostly utilize only a single modality.

**Limitations of Prior Work**: The spatial-temporal resolution, number of bands, and acquisition frequency differ completely across various modalities. Direct concatenation or simple fusion fails to effectively exploit complementary information.

**Key Challenge**: High-resolution data has excellent spatial details but lacks temporal information; time-series data is temporally rich but has low spatial resolution.

**Goal**: Design a unified multimodal remote sensing architecture that can flexibly ingest any subset of modalities.

**Key Insight**: Dedicated encoders are used for each modality to align them into a shared semantic space using cross-modal contrastive learning.

**Core Idea**: Modality-specific encoders + cross-modal CLIP-style alignment + flexible dropout to achieve robust fusion of arbitrary modality combinations.

## Method

### Overall Architecture
Each modality has a dedicated encoder: time-series data utilizes a spatio-temporal attention Transformer (a U-TAE variant), while high-resolution data uses CNN/ViT. The output of each encoder is mapped to a shared space via projection heads. During inference, cross-attention is used to fuse multimodal tokens.

### Key Designs

1. **Modality-Specific Encoders**

    - Function: Handle heterogeneous data separately (time-series multi-spectral, SAR, single-temporal high-resolution).
    - Mechanism: S2 uses U-TAE (spatio-temporal attention), S1 is similar but adapted to 2-channel SAR, and high-resolution data uses ViT/ResNet.
    - Design Motivation: The input formats are too disparate to use a shared encoder.

2. **Cross-Modal Contrastive Alignment**

    - Function: Self-supervised pre-training maps different modalities of the same geographical plot to adjacent embeddings.
    - Mechanism: CLIP-style contrastive learning—bringing S1 and S2 embeddings of the same plot closer while pushing different plots apart.
    - Design Motivation: Labels are scarce; spatial co-occurrence is leveraged to obtain alignment signals for free.

3. **Modality Dropout + Flexible Inference**

    - Function: Randomly drop certain modalities during training, allowing the model to accept any subset during inference.
    - Mechanism: Randomly mask some modality inputs at each step to force the model to extract information from any subset.
    - Design Motivation: In practical deployment, not all plots have coverage from all modalities.

### Loss & Training
- Pre-training: Cross-modal InfoNCE; Fine-tuning: Cross-entropy + Dice
- Datasets: PASTIS (French S1+S2+SPOT), FLAIR (Aerial+S2)

## Key Experimental Results

### Main Results (PASTIS Semantic Segmentation mIoU%)

| Modality Combination | Method | mIoU |
|---------|------|------|
| S2 only | U-TAE | 63.1 |
| S1+S2 | Dual-stream | 65.2 |
| S1+S2+SPOT | **OmniSat** | **67.5** |

### Ablation Study

| Configuration | mIoU | Note |
|------|------|------|
| Full | 67.5 | Tri-modal |
| w/o Pre-training | 64.1 | Self-supervised +3.4 |
| w/o Dropout | 65.8 | Robustness +1.7 |
| w/o S1 | 66.2 | All-weather SAR |
| w/o SPOT | 65.9 | High-res details |

### Key Findings
- Tri-modal fusion improves by 4.4 mIoU compared to the best unimodal baseline.
- Self-supervised pre-training contributes 3.4 mIoU, acting as the largest source of performance gain.
- Modality dropout limits the performance degradation of missing any modality to within 1-2 mIoU.

## Highlights & Insights
- **First unified framework to handle optical time-series, SAR time-series, and high-resolution single-temporal images**
- **Geographical Co-occurrence Self-Supervision**: Aligns multi-source data by exploiting their inherent spatial correspondence.
- **Flexible Inference**: Ingests arbitrary modality subsets into a single model, making it highly practical for deployment.

## Limitations & Future Work
- Only validated on French agricultural regions; global generalization capability remains unverified.
- Contrastive learning may lead to confusion in geographically close but semantically distinct areas.
- Spatial coverage of the high-resolution modality is limited.

## Related Work & Insights
- **vs U-TAE**: Unimodal S2 baseline, which OmniSat extends to a multimodal setting.
- **vs SatCLIP**: Performs geographic coordinate alignment but does not handle temporal sequences.
- **vs SkySense**: Large-scale pre-training but does not address missing modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ First unified framework to handle three types of heterogeneous remote sensing data.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets + modality ablations + pre-training ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-defined problem definitions.
- Value: ⭐⭐⭐⭐⭐ Modality dropout is extremely valuable for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](../../NeurIPS2025/time_series/towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[ICML 2026\] Self-Supervised Dynamical System Representations for Physiological Time-Series](../../ICML2026/time_series/self-supervised_dynamical_system_representations_for_physiological_time-series.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](../../NeurIPS2025/time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[ICML 2026\] HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series](../../ICML2026/time_series/hepa_a_self-supervised_horizon-conditioned_event_predictive_architecture_for_tim.md)
- [\[ICML 2025\] TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning](../../ICML2025/time_series/timepoint_accelerated_time_series_alignment_via_self-supervised_keypoint_and_des.md)

</div>

<!-- RELATED:END -->
