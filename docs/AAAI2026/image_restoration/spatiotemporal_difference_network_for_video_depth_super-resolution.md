---
title: >-
  [Paper Note] SpatioTemporal Difference Network for Video Depth Super-Resolution
description: >-
  [AAAI 2026][Image Restoration][Video Depth Super-Resolution] Motivated by the statistical observation that spatially non-smooth regions and temporally varying regions in video depth super-resolution (VDSR) follow long-tail distributions, this paper proposes STDNet. The method incorporates a spatial difference branch (learning spatial difference representations for intra-frame RGB-D adaptive aggregation) and a temporal difference branch (exploiting temporal difference representations for motion compensation in changing regions). On the TarTanAir dataset at ×16 super-resolution, RMSE is reduced from 112.04 cm to 96.80 cm, outperforming state-of-the-art methods by an average of 27.6%–32.6%.
tags:
  - AAAI 2026
  - Image Restoration
  - Video Depth Super-Resolution
  - Long-tail Distribution
  - Spatial Difference
  - Temporal Difference
  - Deformable Convolution
date: 2026-05-08
content_hash: b013039b64f5ac14
---

# SpatioTemporal Difference Network for Video Depth Super-Resolution

**Conference**: AAAI 2026
**arXiv**: [2508.01259](https://arxiv.org/abs/2508.01259)
**Code**: [yanzq95/STDNet](https://github.com/yanzq95/STDNet)
**Area**: Image Restoration
**Keywords**: Video Depth Super-Resolution, Long-tail Distribution, Spatial Difference, Temporal Difference, Deformable Convolution

## TL;DR

Motivated by the statistical observation that spatially non-smooth regions and temporally varying regions in video depth super-resolution (VDSR) follow long-tail distributions, this paper proposes STDNet. The method incorporates a spatial difference branch (learning spatial difference representations for intra-frame RGB-D adaptive aggregation) and a temporal difference branch (exploiting temporal difference representations for motion compensation in changing regions). On the TarTanAir dataset at ×16 super-resolution, RMSE is reduced from 112.04 cm to 96.80 cm, outperforming state-of-the-art methods by an average of 27.6%–32.6%.

## Background & Motivation

### Development of Depth Super-Resolution

Depth data plays a critical role in 3D reconstruction, virtual reality, and augmented reality. Numerous depth super-resolution (DSR) methods have been proposed to reconstruct high-resolution (HR) depth maps from low-resolution (LR) inputs. Single-frame DSR has achieved notable progress through filtering-based, multimodal fusion, multi-task collaborative, and structure-guided approaches. Video depth super-resolution (VDSR) further improves reconstruction quality by aggregating multi-frame RGB-D features.

### Long-tail Distribution Problem

The authors conduct a statistical analysis of VDSR and identify long-tail distribution phenomena along two dimensions:

- **Spatial dimension**: Discrepancies between GT depth and upsampled LR depth are concentrated in non-smooth regions (edges, structural transitions), which constitute a small fraction of the overall data yet are substantially harder to reconstruct than the dominant smooth regions.
- **Temporal dimension**: Depth differences between consecutive and non-adjacent frames are concentrated in temporally varying regions (dynamic objects, edge contours, occluded areas), exhibiting similarly long-tail characteristics.

### Limitations of Prior Work

Existing VDSR methods such as DVSR incorporate multi-frame feature aggregation but do not explicitly model these long-tail distributional properties. Unimodal video RGB super-resolution methods have limited effectiveness in establishing multi-frame, multimodal RGB-D correspondences. A framework specifically designed to address spatially and temporally long-tail regions is therefore needed.

## Core Problem

How to selectively enhance reconstruction quality in spatially non-smooth regions and temporally varying regions—both exhibiting long-tail distributions—within video depth super-resolution, while maintaining temporal consistency?

## Method

### Overall Architecture

STDNet consists of two core branches:

1. **Spatial Difference Branch**: Predicts spatial difference representations $\boldsymbol{\sigma}$ from LR depth video to guide intra-frame RGB feature alignment and aggregation toward depth non-smooth regions.
2. **Temporal Difference Branch**: Estimates consecutive-frame differences $\boldsymbol{\varphi}$ and cross-frame differences $\hat{\boldsymbol{\varphi}}$ to prioritize multi-frame RGB-D aggregation in temporally varying regions.

A difference regularization loss is additionally introduced to supervise the learning of spatiotemporal difference representations.

### Spatial Difference Branch

**Spatial difference representation**: Non-smooth region information is captured via a downsampling–upsampling operation on depth features:

$$\boldsymbol{\sigma} = |\boldsymbol{F}_d - f_{bu}(f_{bd}(\boldsymbol{F}_d))|$$

**Spatial difference mechanism**:

1. A filter kernel $\boldsymbol{k}_t = \mathcal{G}(\boldsymbol{\sigma}_t)$ is generated from $\boldsymbol{\sigma}_t$ to align RGB features toward non-smooth depth regions.
2. Adaptive weights $\boldsymbol{w}_t = \mathcal{E}_w(\boldsymbol{\sigma}_t)$ are encoded from $\boldsymbol{\sigma}_t$ (via convolution, max, mean, and sigmoid operations).
3. Weighted aggregation: $\boldsymbol{F}_{sd}^t = f_c(\boldsymbol{F}_d^t, \boldsymbol{w}_t \otimes \boldsymbol{F}_r^t, \mathcal{F}(\boldsymbol{F}_r^t, \boldsymbol{k}_t))$

This mechanism drives RGB information to selectively propagate into depth non-smooth regions via spatial difference representations, effectively alleviating the long-tail effect.

### Temporal Difference Branch

**Temporal difference representations**:

$$\boldsymbol{\varphi}_t = |\boldsymbol{F}_{sd}^t - \boldsymbol{F}_{sd}^{t+1}|, \quad \hat{\boldsymbol{\varphi}}_t = |\boldsymbol{F}_{sd}^t - \boldsymbol{F}_{sd}^{t+2}|$$

These capture temporal variation information from consecutive and cross-frame pairs, respectively.

**Temporal difference strategy**: A bidirectional iterative scheme is employed, comprising neighboring-frame fusion and cross-frame fusion stages:

- **Neighboring-frame fusion**: The temporal difference $\boldsymbol{\varphi}_{t-1}$ is passed through encoder $\mathcal{E}_\varphi$ to produce offsets $\delta_{t-1}$ and modulation scalars $m_{t-1}$; deformable convolution $\mathcal{D}$ is then applied to dynamically sample temporally varying information. Spatial difference weights $\boldsymbol{w}_t$ are additionally used to mitigate cross-modal discrepancies:

$$\boldsymbol{F}_f^{t-1,t} = f_c(\boldsymbol{F}_f^t, \mathcal{D}(\boldsymbol{F}_f^{t-1}, \delta_{t-1}, m_{t-1}), \boldsymbol{w}_t \otimes \mathcal{D}(\boldsymbol{F}_r^{t-1}, \delta_{t-1}, m_{t-1}))$$

- **Cross-frame fusion**: $\hat{\boldsymbol{\varphi}}_{t-2}$ is used analogously to process frame $t-2$.
- **Final fusion**: $\hat{\boldsymbol{F}}_f^t = \boldsymbol{F}_f^{t-1,t} + \boldsymbol{F}_f^{t-2,t}$

### Difference Regularization Loss

The total loss combines a reconstruction loss with difference regularization:

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \beta \mathcal{L}_{diff}$$

where $\mathcal{L}_{rec}$ uses Charbonnier regularization and $\mathcal{L}_{diff} = \alpha_1 \mathcal{L}_{sd} + \alpha_2 \mathcal{L}_{td}$.

- **Spatial difference loss**: Introduces uncertainty constraints to impose larger reconstruction error penalties in non-smooth regions: $\mathcal{L}_{sd} = \sum_q (\boldsymbol{\sigma}^q - \min(\boldsymbol{\sigma}^q)) \|\boldsymbol{D}_{GT}^q - \boldsymbol{D}_{HR}^q\|_1$
- **Temporal difference loss**: Constrains the temporal difference representations to be consistent with the temporal variations in GT depth, covering both consecutive-frame and cross-frame terms.

Hyperparameter settings: $\alpha_1 = \alpha_2 = 0.5$, $\beta = 0.01$.

## Key Experimental Results

### Table 1: Quantitative Comparison on TarTanAir

| Method | Conference | ×4 RMSE↓ | ×4 MAE↓ | ×8 RMSE↓ | ×8 MAE↓ | ×16 RMSE↓ | ×16 MAE↓ | ×16 TEPE↓ |
|--------|------------|----------|---------|----------|---------|-----------|----------|-----------|
| DJFR | PAMI'19 | 75.56 | 10.59 | 105.45 | 18.43 | 141.14 | 31.22 | 20.27 |
| DKN | IJCV'21 | 82.69 | 11.73 | 110.10 | 18.78 | 153.56 | 33.21 | 21.93 |
| SGNet | AAAI'24 | 79.40 | 11.36 | 116.33 | 23.15 | 144.17 | 34.34 | 20.14 |
| DORNet | CVPR'25 | 63.38 | 8.60 | 93.75 | 13.96 | 123.24 | 23.59 | 16.40 |
| DVSR | CVPR'23 | 57.72 | 4.40 | 76.96 | 7.74 | 112.04 | 14.39 | 11.06 |
| **STDNet** | — | **50.28** | **3.73** | **72.03** | **6.75** | **96.80** | **12.01** | **8.90** |

At ×16 super-resolution, STDNet reduces RMSE by 15.24 cm, MAE by 2.38 cm, and TEPE by 2.16 cm compared to DVSR.

### Table 2: Generalization on DyDToF Dataset

| Method | ×4 RMSE↓ | ×8 RMSE↓ | ×16 RMSE↓ | ×16 MAE↓ |
|--------|----------|----------|-----------|----------|
| DVSR | 19.53 | 27.63 | 43.55 | 9.80 |
| **STDNet** | **18.23** | **26.87** | **39.24** | **8.72** |

Without fine-tuning, STDNet outperforms DVSR on DyDToF, reducing ×16 RMSE by 4.31 cm.

### Table 3: Generalization on DynamicReplica Dataset

| Method | ×4 RMSE↓ | ×8 RMSE↓ | ×16 RMSE↓ |
|--------|----------|----------|-----------|
| DVSR | 0.37 | 0.58 | 1.25 |
| **STDNet** | **0.32** | **0.53** | **1.10** |

### Ablation Study

| Variant | TarTanAir ×4 RMSE↓ | DyDToF ×4 RMSE↓ |
|---------|-------------------|-----------------|
| Baseline (concatenation replacing SD+TD) | ~60+ | ~36+ |
| +SD (spatial difference only) | RMSE reduced by 3.56 cm | — |
| +TD (temporal difference only) | RMSE reduced by 14.02 cm | — |
| **+SD+TD (full STDNet)** | **reduced by 17.94 cm** | — |

Removing the difference regularization loss increases TarTanAir ×16 RMSE by 7.08 cm, validating the effectiveness of the loss design.

### Model Complexity

Compared to single-frame methods, STDNet reduces parameter count by an average of 9.23 M and RMSE by 35.82 cm. Compared to the multi-frame method DVSR, inference speed improves by 47.35 ms and performance improves by 4.93 cm RMSE, with only 4.4 M additional parameters.

## Highlights & Insights

1. **Problem formulation grounded in statistical analysis**: Rather than designing a network without prior analysis, the authors first perform histogram-based statistical analysis on VDSR, identify long-tail distributional characteristics along both spatial and temporal dimensions, and then design targeted solutions accordingly—a data-driven problem discovery approach worth emulating.
2. **Simplicity and effectiveness of the spatial difference mechanism**: The spatial difference representation is obtained via a straightforward downsampling–upsampling difference (analogous to the Laplacian pyramid concept), which is then used to generate filter kernels and weights for guiding RGB-D aggregation. The design is concise yet yields significant gains.
3. **Elegant integration of temporal difference with deformable convolution**: Temporal difference representations are transformed into offsets and modulation scalars for deformable convolution, enabling motion compensation to naturally focus on long-tail regions with pronounced temporal variation.
4. **Consistent cross-dataset generalization**: Trained solely on TarTanAir, STDNet consistently improves performance on DyDToF and DynamicReplica without fine-tuning, demonstrating robustness.
5. **Substantial improvement in temporal consistency**: x-t slice visualizations clearly demonstrate that STDNet produces more stable depth predictions in temporally varying regions.

## Limitations & Future Work

1. **Synthetic data training**: Experiments are conducted exclusively on synthetic datasets (TarTanAir, DyDToF, DynamicReplica); performance on noisy depth videos captured by real sensors remains unvalidated.
2. **Fixed number of reference frames**: Experiments show that using 2 frames (1 neighboring + 1 cross-frame) yields optimal performance, but this empirical setting may not generalize well to scenes with large motion or prolonged occlusion; an adaptive frame selection mechanism is lacking.
3. **Absence of optical flow**: Spatial difference representations are computed via simple downsampling–upsampling differences, and temporal differences via inter-frame feature differences, both without explicit motion estimation using optical flow, which may limit performance under large-displacement scenarios.
4. **Computational overhead**: Although the parameter count is lower than single-frame methods such as DKN, the bidirectional iteration combined with deformable convolution increases computation, posing challenges for real-time applications.
5. **Scope limited to depth super-resolution**: The method is not extended to related tasks such as depth completion or depth denoising, leaving its generality insufficiently validated.

## Related Work & Insights

- **DVSR (Sun et al., CVPR 2023)**: The first dToF-based VDSR method, which mitigates spatial blurring through multi-frame fusion but does not explicitly address long-tail distributions. STDNet outperforms DVSR by an average of 32.6%/28.8%/27.6% on TarTanAir at ×4/×8/×16.
- **DORNet (Wang et al., CVPR 2025)**: A recent single-frame DSR method whose performance is substantially weaker than multi-frame methods (×16 RMSE 123.24 vs. STDNet 96.80), confirming the importance of multi-frame information.
- **BasicVSR++ (Chan et al.)**: A bidirectional recurrent framework for video RGB super-resolution. STDNet adopts its bidirectional iterative idea but designs a temporally difference-driven aggregation strategy tailored to VDSR.
- **SVDC (Zhu et al., 2025)**: A video depth completion framework that fuses multi-frame features via adaptive frequency selection. STDNet focuses specifically on the long-tail distribution problem in depth super-resolution and follows a distinct design philosophy.

The long-tail distribution perspective is generalizable to other video restoration tasks (e.g., video deblurring, video denoising), where edge and motion regions similarly follow non-uniform distributions. The spatial difference representation (downsampling–upsampling difference) serves as a lightweight, supervision-free detector for non-smooth regions and can be applied as an attention weight generator in other multimodal fusion tasks. The temporal difference-driven deformable convolution design can inspire motion modeling in video depth estimation and video optical flow estimation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The long-tail distribution perspective and the dual-branch design driven by spatiotemporal differences are novel; however, individual modules (deformable convolution, bidirectional iteration) are combinations of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, three scaling factors, 11 comparison methods, detailed ablation studies (SD/TD/loss/frame count), complexity analysis, and PCA visualizations constitute a comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — The narrative logic of statistical analysis → problem identification → method design is clear; figures and tables are of high quality; histogram comparisons intuitively illustrate the mitigation of long-tail effects.
- **Value**: ⭐⭐⭐⭐ — Achieves substantial improvements on the VDSR task (×16 average 27.6%), with a long-tail distribution perspective transferable to other restoration tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment](temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment.md)
- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](../../CVPR2026/image_restoration/ucan_unified_convolutional_attention_network_for_expansive_receptive_fields_in_l.md)
- [\[AAAI 2026\] SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining](sd-psfnet_sequential_and_dynamic_point_spread_function_netwo.md)
- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](../../CVPR2026/image_restoration/selfhvd_self-supervised_handheld_video_deblurring.md)

</div>

<!-- RELATED:END -->
