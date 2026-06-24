---
title: >-
  [Paper Note] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images
description: >-
  [CVPR 2025][Segmentation][Salient Object Detection] RDNet proposes a region proportion-aware dynamic adaptive salient object detection network to address the dramatic object scale variations in remote sensing images. By introducing the Dynamic Adaptive Detail-aware module (DAD, selecting combinations of different kernel sizes based on target region proportions), the Frequency-matching Context Enhancement module (FCE, performing feature interaction in the wavelet domain)…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Salient Object Detection"
  - "Remote Sensing Images"
  - "Dynamic Convolutional Kernels"
  - "Wavelet Transform"
  - "Region Proportion Guidance"
date: 2026-05-08
content_hash: 3860e7d6acd8a0f5
---

# RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images

**Conference**: CVPR 2025  
**arXiv**: [2603.12215](https://arxiv.org/abs/2603.12215)  
**Code**: To be confirmed  
**Area**: Segmentation / Remote Sensing  
**Keywords**: Salient Object Detection, Remote Sensing Images, Dynamic Convolutional Kernels, Wavelet Transform, Region Proportion Guidance

## TL;DR
RDNet proposes a region proportion-aware dynamic adaptive salient object detection network to address the dramatic object scale variations in remote sensing images. By introducing the Dynamic Adaptive Detail-aware module (DAD, selecting combinations of different kernel sizes based on target region proportions), the Frequency-matching Context Enhancement module (FCE, performing feature interaction in the wavelet domain), and the Region Proportion-aware Localization module (RPL, cross-attention + proportion guidance), the method achieves state-of-the-art (SOTA) performance on three datasets: EORSSD, ORSSD, and ORSI-4199.

## Background & Motivation
**Background**：Salient object detection in optical remote sensing images (ORSI-SOD) is an important task in remote sensing analysis, and CNN/Transformer-based methods have recently made significant progress.

**Limitations of Prior Work**：（1）Target scales in remote sensing images vary extremely. Fixed-size convolution kernel combinations either aggregate excess background (large kernels processing small objects) or lose the overall region (small kernels processing large objects);（2）current methods directly use full-resolution self-attention for adjacent layer feature interactions, which induces heavy computational overhead and mixes high- and low-frequency information;（3）CNN backbones lack the capability to model global context and long-range dependencies.

**Key Challenge**：Targets of different scales require convolution kernels with different receptive fields; however, existing methods rigidly apply the same convolution strategy to all targets.

**Goal** How to dynamically select convolution kernel combinations based on target region proportions and design computationally efficient cross-layer feature interaction mechanisms?

**Key Insight**：Guide dynamic convolution kernel selection via "region proportion prediction" inspired by classification tasks—first predicting the area proportion of the target in the image, and then selecting the suitable convolution kernel combination accordingly.

**Core Idea**：Predict target region proportion $\rightarrow$ Dynamically select multi-scale convolution kernel combinations $\rightarrow$ Cross-layer feature interaction in the wavelet domain.

## Method

### Overall Architecture
Input image of size $3 \times 384 \times 384$ $\rightarrow$ SwinTransformer backbone extracts 5-level features $\{F_i^R\}_{i=1}^5$ $\rightarrow$ RPL module processes high-level features $F_4^R, F_5^R$ (extracting location information $F^A$ + region proportion $F^G$) $\rightarrow$ DAD module utilizes $F^G$ to guide the dynamic convolution of low-level features $F_1^R$ for detail extraction $F^P$ $\rightarrow$ FCE module processes mid-level features $F_2^R, F_3^R$ (extracting context $F^W$ via wavelet-domain interaction) $\rightarrow$ Bottom-up fusion yields the final saliency map $S$.

### Key Designs

1. **Region Proportion-Aware Localization Module (RPL)**:

    - **Function**: Extract spatial semantic information from high-level features and predict the proportion of the target region relative to the entire image.
    - **Mechanism**: Successive channel-attention and spatial-attention cross-actions (cross-attention) are applied to $F_4^R, F_5^R$, which are then concatenated and processed by a $3 \times 3$ convolution to obtain the location feature $F^A$. Meanwhile, $F_5^R$ is processed by GAP + two FC layers to generate the proportion feature $F^G \in \mathbb{R}^{4 \times 1}$, which guides the DAD module.
    - **Design Motivation**: High-level features contain rich structural and semantic information; predicting the proportion provides a solid basis for downstream dynamic convolutions.

2. **Dynamic Adaptive Detail-Aware Module (DAD)**:

    - **Function**: Dynamically select combinations of convolution kernels of different sizes based on the target region proportion to extract fine details.
    - **Mechanism**: Three strategies are established: proportion $< 25\%$ uses primarily small kernels ($1 \times 1, 3 \times 3, 5 \times 5$), $25\% - 50\%$ uses a medium combination, and $> 50\%$ utilizes all five kernels (from $1 \times 1$ up to $9 \times 9$). A dual-branch architecture is adopted: the bottom branch acts as a detail extractor (summation of multi-kernel convolutions) and the top branch serves as a detail optimizer (multi-kernel convolutions on max-pooled features to generate spatial attention weights). The two branches are fused via element-wise multiplication and addition.
    - **Design Motivation**: Large targets require a expansive receptive field to capture the entire region, while small objects do not need large kernels to avoid background interference. Dynamic selection avoids the limitations of a "one-size-fits-all" approach.

3. **Frequency-Matching Context Enhancement Module (FCE)**:

    - **Function**: Perform cross-layer feature interaction in the wavelet domain to extract contextual information while avoiding the huge overhead of full-resolution self-attention.
    - **Mechanism**: Continuous Discrete Wavelet Transform (DWT) is applied to $F_2^R, F_3^R$ to obtain four frequency sub-bands (LL, LH, HL, HH) for each. Channel attention cross-interaction is performed between corresponding frequency sub-bands, followed by Inverse DWT (IDWT) reconstruction, and then enhancement through channel and spatial attention. The key is "frequency matching"—interacting low-frequency with low-frequency and high-frequency with high-frequency, rather than direct blending.
    - **Design Motivation**: Directly applying self-attention to blend high- and low-frequency components dilutes key details. Separating them in the wavelet domain followed by matching frequency interactions yields much finer details.

### Loss & Training
Binary Cross-Entropy (BCE) loss + IoU loss + MSE loss for region proportion prediction (supervising $F^G$ to approach the ground-truth proportion).

## Key Experimental Results

### Main Results (EORSSD / ORSSD / ORSI-4199)

| Method | EORSSD MAE↓ | EORSSD $F_\beta$↑ | ORSSD MAE↓ | ORSSD $F_\beta$↑ | ORSI-4199 MAE↓ | ORSI-4199 $F_\beta$↑ |
|------|-------------|-------------------|------------|-------------------|----------------|---------------------|
| HFCNet | 0.0051 | 0.7845 | 0.0073 | 0.8581 | 0.0270 | 0.8272 |
| GeleNet | 0.0066 | 0.8367 | 0.0083 | 0.8879 | 0.0266 | 0.8711 |
| ADSTNet | 0.0065 | 0.8321 | 0.0089 | 0.8856 | 0.0319 | 0.8615 |
| **RDNet** | **0.0049** | **0.8563** | **0.0066** | **0.9080** | **0.0254** | **0.8781** |

### Ablation Study

| Configuration | Description |
|------|------|
| SwinTransformer baseline | Backbone only + simple decoder |
| +RPL | With RPL |
| +RPL+DAD | With RPL and DAD |
| +RPL+DAD+FCE (Full) | Full RDNet |

Comparison of three proportion strategies: fixed kernels vs. dynamic kernel selection. Dynamic selection is consistently superior in both large- and small-target scenarios.

### Key Findings
- Outperforms over 15 SOTA methods across all three datasets; on EORSSD, the MAE drops from the runner-up of 0.0051 to 0.0049 (3.9%$\downarrow$), and $F_\beta$ increases from 0.8367 to 0.8563.
- t-test statistical validation shows significant performance differences compared to all other methods (all p-values are far below 0.05).
- The dynamic kernel selection of DAD is the primary source of performance gains, showing the most prominent improvement in large/small target scenarios.
- Wavelet-domain interaction in FCE reduces computational overhead compared to direct self-attention while maintaining performance.

## Highlights & Insights
- **Region Proportion Prediction $\rightarrow$ Dynamic Kernels**: Using high-level features to predict target area proportion to guide receptive field selection for low-level features is a clean, simple, and effective scale adaptation strategy.
- **Frequency-Matching Interaction**: Instead of directly performing cross-layer attention in the spatial domain, interacting corresponding frequency sub-bands in the wavelet domain reduces computation and avoids high-low frequency mixing. This concept is transferable to other multi-scale feature fusion tasks.
- **Three-Strategy Selector**: Although simple (categorized into three tiers: $<25\%$, $25\%-50\%$, and $>50\%$), experiments prove its effectiveness, bypassing the training difficulties of continuous regression.

## Limitations & Future Work
- The target region proportion is divided into only three discrete classes ($<25\%$, $25\%-50\%$, and $>50\%$). This relatively coarse granularity might be improved by a finer continuous-proportion prediction.
- Validated only on remote sensing images; generalization ability on natural scene SOD datasets remains untested.
- The inference speed is 13 FPS. The computational overhead from wavelet transforms and matrix operations is relatively high, and real-time performance needs improvement.
- The PG block uses GAP + FC for proportion prediction, highly relying on the semantic quality of high-level features. Predictions may be inaccurate in extreme small-target scenarios.

## Related Work & Insights
- **Ours vs. GeleNet**: GeleNet achieves an $F_\beta$ of 0.8367 on EORSSD, whereas the proposed method scores 0.8563 (a 2.3% gain). GeleNet uses graph reasoning to model region and boundary relationships, while this work uses dynamic kernels, which is more direct and efficient.
- **Ours vs. ADSTNet**: ADSTNet is a strong baseline on ORSSD ($F_\beta$ 0.8856). The proposed method significantly outperforms it at 0.9080 (+2.5%), showing that dynamic kernel selection is highly effective for multi-scale remote sensing scenarios.
- **Ours vs. HFCNet**: HFCNet achieved the best MAE on EORSSD previously (0.0051). This work lowers it further to 0.0049, validating that frequency-domain interaction is more delicate than spatial-domain interaction.
- **Transferable Ideas**: The idea of using region proportion prediction to drive dynamic receptive fields can be extended to medical image segmentation (where tumor sizes vary widely) or autonomous driving (where near/far targets have massive scale differences).

## Rating
- Novelty: ⭐⭐⭐⭐ Guiding dynamic kernel selection via region proportion is a clever design, and the frequency-matching interaction in the wavelet domain is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on three datasets, compared with 21 methods, validated by t-test, with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, though complex mathematical symbols exist and the conciseness can be improved.
- Value: ⭐⭐⭐⭐ A solid piece of work in the field of ORSI-SOD, and the dynamic kernel selection principle has high generalizability.
- SwinTransformer as a backbone is heavy, making deployment on edge devices difficult.
- How DAD handles mixed proportions within a batch during training/inference warrants further exploration.

## Related Work & Insights
- **vs. GeleNet**: GeleNet uses graph reasoning to model regions and boundaries, while RDNet uses dynamic convolutional kernels to adapt to scale more directly.
- **vs. ADSTNet**: ADSTNet utilizes a Transformer+CNN dual-branch, while RDNet employs a pure SwinTransformer backbone coupled with three dedicated modules.
- **vs. HFCNet**: HFCNet has a lower MAE, but its $F_\beta$ and $E_\xi$ are inferior to RDNet, indicating that RDNet is superior in overall detection quality.

## Rating
- Novelty: ⭐⭐⭐ The idea of dynamic kernel selection is intuitive but not highly novel; the wavelet interaction has some novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, 15+ comparison methods, t-test, ablation, and visualization.
- Writing Quality: ⭐⭐⭐ The mathematical formulations are written clearly, but the explanation is somewhat verbose.
- Value: ⭐⭐⭐ Incremental improvement in the remote sensing SOD field; the generalizability and transferability of the method are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[CVPR 2025\] SmartEraser: Remove Anything from Images using Masked-Region Guidance](smarteraser_remove_anything_from_images_using_masked-region_guidance.md)

</div>

<!-- RELATED:END -->
