---
title: >-
  [Paper Note] TDCNet: Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving IRSTD
description: >-
  [AAAI 2026][Remote Sensing][Infrared small target detection] This paper proposes TDCNet, which unifies temporal difference and 3D convolution into a single Temporal Difference Convolution (TDC). Through re-parameterization, TDC introduces zero additional inference overhead. Combined with TDC-guided spatio-temporal attention (TDCSTA), TDCNet achieves an F1 of 97.12% (AP50 93.83%) on the newly constructed IRSTD-UAV dataset, which contains 15,106 frames of real infrared UAV imag…
tags:
  - "AAAI 2026"
  - "Remote Sensing"
  - "Infrared small target detection"
  - "temporal difference convolution"
  - "re-parameterization"
  - "spatio-temporal attention"
  - "UAV"
date: 2026-05-08
content_hash: 224e5bd056cfcbfe
---

# TDCNet: Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving IRSTD

**Conference**: AAAI 2026  
**arXiv**: [2511.09352](https://arxiv.org/abs/2511.09352)  
**Code**: [https://github.com/IVPLaboratory/TDCNet](https://github.com/IVPLaboratory/TDCNet)  
**Area**: Remote Sensing / Infrared Small Target Detection  
**Keywords**: Infrared small target detection, temporal difference convolution, re-parameterization, spatio-temporal attention, UAV

## TL;DR
This paper proposes TDCNet, which unifies temporal difference and 3D convolution into a single Temporal Difference Convolution (TDC). Through re-parameterization, TDC introduces zero additional inference overhead. Combined with TDC-guided spatio-temporal attention (TDCSTA), TDCNet achieves an F1 of 97.12% (AP50 93.83%) on the newly constructed IRSTD-UAV dataset, which contains 15,106 frames of real infrared UAV imagery.

## Background & Motivation

**Background**: Moving infrared small target detection (IRSTD) requires the integration of temporal information. Existing methods either employ frame differencing (capturing motion but losing appearance) or 3D convolution (preserving appearance but lacking explicit motion cues).

**Limitations of Prior Work**: Frame differencing and 3D convolution are complementary yet have not been unified; publicly available infrared small target video datasets remain scarce.

**Key Challenge**: How can motion and appearance features be jointly exploited without increasing inference cost?

**Goal**: Implicitly embed frame differencing into the weight structure of 3D convolutions, enabling multi-branch multi-scale training with zero inference overhead.

**Key Insight**: Mathematically demonstrate, via 3D convolution weight decomposition, that temporal difference can be encoded as a special weight structure.

**Core Idea**: TDC = a unified representation of frame differencing ⊕ 3D convolution, re-parameterizable into a standard 3D convolution.

## Method

### Overall Architecture
Three parallel backbones: TDC backbone (temporal difference features) + 3D backbone (spatio-temporal features) + 2D backbone (spatial features of the current frame). Features are fused via TDCSTA attention before being passed to the detection head. A progressive training strategy is adopted—2D/3D backbones are pre-trained first, followed by joint training of TDC and TDCSTA.

### Key Designs

1. **Temporal Difference Convolution (TDC)**: Restructures 3D convolution weights along the temporal dimension to implicitly encode frame differencing. Three temporal scales are defined:

    - **S-TDC**: Short-term, consecutive frame difference ($F_t - F_{t-1}$)
    - **M-TDC**: Medium-term, skip-frame difference ($F_t - F_{t-2}$)
    - **L-TDC**: Long-term, difference between the current frame and all historical frames ($F_5 - F_t$)

2. **TDCR Re-parameterization Module**: Three parallel TDC branches each have independent batch normalization; at inference, they are mathematically merged into a single 3D convolution. FLOPs are reduced by 7.3G (102.96→95.67) with no loss in accuracy.

3. **TDCSTA Spatio-Temporal Attention**: (1) Each of the three streams performs 3D local window self-attention independently; (2) A cross-attention mechanism is applied where TDC features serve as Q, 3D features as K, and 2D features as V—TDC features thus "guide" the aggregation of spatial appearance information.

4. **IRSTD-UAV Dataset**: 17 sequences, 15,106 frames of real infrared UAV images, publicly released.

### Loss & Training
Detection loss (YOLOv8-style). Adam optimizer, lr=0.001. Input: 5 consecutive frames at 640×640. Hardware: RTX 3090.

## Key Experimental Results

### Main Results

| Method | Type | IRSTD-UAV F1/AP50 | IRDST F1/AP50 |
|------|------|------------------|---------------|
| YOLO11-L | Single-frame | 95.99/91.20 | 96.35/92.10 |
| MOCID | Multi-frame | 96.05/91.32 | 97.88/94.74 |
| **TDCNet** | **Multi-frame** | **97.12/93.83** | **97.91/94.79** |

### Ablation Study

| Method | F1 | AP50 | FLOPs(G) |
|------|-----|------|----------|
| Temporal difference only | 92.25 | 89.73 | 41.4 |
| 3D convolution only | 87.36 | 75.97 | 45.8 |
| TD + 3D (separate) | 93.87 | 89.81 | 45.9 |
| **TDC (unified)** | **96.76** | **92.50** | **45.7** |

| TDC Scale | F1 | AP50 |
|---------|-----|------|
| S-TDC only | 94.91 | 90.31 |
| M-TDC only | 95.65 | 90.46 |
| L-TDC only | 96.29 | 92.35 |
| **All scales** | **96.76** | **92.50** |

### Key Findings
- TDC outperforms separately applied TD+3D by 2.89 F1 at comparable FLOPs, demonstrating the advantage of the unified representation.
- L-TDC contributes the most individually (F1 96.29), but the combination of all three scales yields the best performance.
- Re-parameterization incurs zero accuracy loss while reducing FLOPs by 7%.
- Using TDC features as Q in cross-attention is critical; replacing them with SF as Q drops F1 to 91.26.

## Highlights & Insights
- **The mathematical unification of TDC** is particularly elegant—two complementary temporal modeling paradigms are fused into a single convolution kernel, enabling multi-branch training with zero inference overhead.
- **The IRSTD-UAV dataset** fills a gap in infrared small target video data and constitutes a direct contribution to the research community.

## Limitations & Future Work
- Fixed 5-frame input; longer sequences may yield further improvement.
- Validated only in small target scenarios; applicability to general object detection remains untested.
- The progressive training strategy increases training complexity.

## Related Work & Insights
- **vs. MOCID**: Both are multi-frame methods; TDCNet surpasses MOCID on both datasets.
- **vs. STMENet**: STMENet achieves only F1 87.36, far below TDCNet's 97.12.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified TDC representation and re-parameterization design are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets with detailed ablations covering modules, scales, frame counts, kernel sizes, and re-parameterization.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear and well-presented.
- Value: ⭐⭐⭐⭐⭐ Dual contribution of a practical method and a new dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting](../../NeurIPS2025/remote_sensing/ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)
- [\[CVPR 2026\] TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis](../../CVPR2026/remote_sensing/tessera_temporal_embeddings_of_surface_spectra_for_earth_representation_and_anal.md)
- [\[CVPR 2026\] Sparsely Timing the Change: A Spiking Temporal Framework for Remote Sensing Interpretation](../../CVPR2026/remote_sensing/sparsely_timing_the_change_a_spiking_temporal_framework_for_remote_sensing_inter.md)
- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)
- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](../../NeurIPS2025/remote_sensing/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)

</div>

<!-- RELATED:END -->
