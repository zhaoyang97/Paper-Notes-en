---
title: >-
  [Paper Note] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images
description: >-
  [CVPR 2026][Segmentation][Salient Object Detection] This paper proposes RDNet, which employs a region proportion-aware Proportion Guidance block to estimate the area ratio of salient objects and dynamically selects combi…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Salient Object Detection"
  - "Remote Sensing Images"
  - "Dynamic Convolutional Kernel Selection"
  - "Wavelet Transform"
  - "Region Proportion Awareness"
date: 2026-05-08
content_hash: 3a00324bfa4a2006
---

# RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images

**Conference**: CVPR 2026
**arXiv**: [2603.12215](https://arxiv.org/abs/2603.12215)
**Code**: None
**Area**: Image Segmentation
**Keywords**: Salient Object Detection, Remote Sensing Images, Dynamic Convolutional Kernel Selection, Wavelet Transform, Region Proportion Awareness

## TL;DR

This paper proposes RDNet, which employs a region proportion-aware Proportion Guidance block to estimate the area ratio of salient objects and dynamically selects combinations of 3/4/5 convolutional kernels of varying sizes for detail extraction. Combined with wavelet-domain frequency-matched context enhancement (reducing computation to 1/4) and a cross-attention localization module, RDNet comprehensively outperforms 21 state-of-the-art methods on three optical remote sensing SOD benchmarks: EORSSD, ORSSD, and ORSI-4199.

## Background & Motivation

**Background**: Optical Remote Sensing Image Salient Object Detection (ORSI-SOD) has increasingly relied on CNN/Transformer-based multi-level feature extraction and fusion pipelines, achieving continuous performance gains on standard benchmarks.

**Limitations of Prior Work**:

1. Object scales in remote sensing images vary drastically (from aircraft spanning only a few pixels to stadiums occupying half the image). Existing methods apply fixed convolutional kernel combinations — large kernels introduce excessive background noise for small objects, while small kernels fail to capture the complete extent of large objects.
2. Full-resolution self-attention for cross-layer feature interaction incurs high computational cost, and directly mixing high- and low-frequency information dilutes object features.
3. CNN backbones lack the capacity for global context modeling and long-range dependency capture.

**Key Challenge**: Object scale is inherently uncertain, yet feature extraction strategies remain static — without knowing "how large the object is," it is impossible to select "how wide a perspective to adopt."

**Goal**: Adaptively select appropriate feature extraction strategies according to the proportion of the image area occupied by the salient object, while performing multi-level feature interaction efficiently.

**Key Insight**: Estimate the object region proportion from high-level features, then use this estimate to guide dynamic kernel selection in low-level feature extraction; perform mid-level feature interaction via frequency decomposition in the wavelet domain for dimensionality reduction.

**Core Idea**: Determine approximately how large the object is before deciding how to perceive it — region proportion-guided dynamic convolutional kernel selection.

## Method

### Overall Architecture

Given an input of $4 \times 3 \times 384 \times 384$, a Swin Transformer backbone extracts five levels of features. High-level features $F_4$ and $F_5$ are fed into the RPL module for object localization and region proportion estimation; low-level feature $F_1$ is processed by the DAD module, which dynamically selects convolutional kernels guided by the estimated proportion for detail extraction; mid-level features $F_2$ and $F_3$ are enhanced by the FCE module through wavelet-domain frequency-matched context enhancement. The outputs of the three modules are fused in a bottom-up manner to generate the final saliency map.

### Key Designs

1. **RPL (Region Proportion-aware Localization Module)**

   - **Function**: Localizes salient objects and estimates their area proportion using high-level semantic features.
   - **Mechanism**: Applies alternating channel attention (GAP → two 1×1 Conv layers → Sigmoid) and spatial attention (Max Pooling → Sigmoid) on $F_4$ and $F_5$, then concatenates and applies a 3×3 convolution to obtain localization features.
   - **PG (Proportion Guidance) block**: Applies GAP followed by two fully connected layers on $F_5$ to output a per-sample object region proportion, supervised by ground-truth proportion via MSE loss.
   - **Design Motivation**: Knowing "how large the object is" in advance enables the subsequent DAD module to select appropriate convolutional kernels.

2. **DAD (Dynamic Adaptive Detail-perception Module)**

   - **Function**: Dynamically selects the number and size of convolutional kernels for object detail extraction based on the region proportion output by PG.
   - **Mechanism**: The region proportion is discretized into three ranges — $<25\%$ uses 3 kernel types ($1\times1$, $3\times3$, $5\times5$); $25\%$–$50\%$ uses 4 types (adding $7\times7$); $>50\%$ uses 5 types (adding $9\times9$). A dual-branch design is adopted: the lower branch performs detail extraction via summation of multi-scale convolutions, and the upper branch applies spatial attention weighting to suppress noise.
   - **Design Motivation**: Small objects do not require large receptive fields (which introduce background noise), while large objects require large receptive fields to capture complete regions — proportion-guided selection breaks the one-size-fits-all paradigm.

3. **FCE (Frequency-matched Context Enhancement Module)**

   - **Function**: Performs efficient cross-layer interaction among mid-level features, avoiding the high computational cost of full-resolution self-attention and the interference caused by mixing high- and low-frequency information.
   - **Mechanism**: DWT decomposes features into four frequency sub-bands (LL/LH/HL/HH) → attention-based interaction is performed between corresponding sub-bands → IDWT reconstructs the features → concatenated with original features → channel and spatial attention further suppress noise.
   - **Design Motivation**: Performing interaction in the frequency domain halves the spatial resolution in each dimension, reducing computation to 1/4 of full-resolution attention while preventing high- and low-frequency information from interfering with each other.

### Loss & Training

- Total loss: $\mathcal{L}_\text{total} = \mathcal{L}_\text{BCE} + \mathcal{L}_\text{IoU} + \mathcal{L}_{F\text{-measure}} + \mathcal{L}_\text{MSE}$, with equal weights.
- The first three terms supervise saliency map prediction (pixel-level BCE + region-level IoU + precision-recall-balanced F-measure).
- MSE supervises region proportion prediction.
- Optimizer: RMSprop; learning rate: $1\text{e}{-5}$; batch size: 4; input resolution: $384\times384$.

## Key Experimental Results

### Main Results

| Dataset | Metric | RDNet | GeleNet (Prev. SOTA) | ADSTNet | HFCNet | Gain |
|---------|--------|-------|----------------------|---------|--------|------|
| EORSSD | MAE↓ | **0.0049** | 0.0066 | 0.0065 | 0.0051 | −25.8% |
| EORSSD | Fβ↑ | **0.8563** | 0.8367 | 0.8321 | 0.7845 | +2.3% |
| EORSSD | Eξ↑ | **0.9718** | 0.9678 | 0.9633 | 0.9280 | +0.4% |
| ORSSD | MAE↓ | **0.0066** | 0.0083 | 0.0089 | 0.0073 | −20.5% |
| ORSSD | Fβ↑ | **0.9080** | 0.8879 | 0.8856 | 0.8581 | +2.3% |
| ORSI-4199 | MAE↓ | **0.0254** | 0.0266 | 0.0319 | 0.0270 | −4.5% |
| ORSI-4199 | Fβ↑ | **0.8781** | 0.8711 | 0.8615 | 0.8272 | +0.8% |

RDNet achieves the best performance on all metrics across comparisons with 21 methods. All pairwise t-test p-values are $<1\text{e}{-10}$, confirming statistical significance.

### Ablation Study

| Configuration | EORSSD MAE | EORSSD Fβ | Notes |
|---------------|-----------|----------|-------|
| Full RDNet | 0.0049 | 0.8563 | Baseline |
| w/o DAD | 0.0052 | 0.8550 | Dynamic kernel selection is effective |
| w/o FCE | 0.0061 | — | Largest impact; frequency-domain interaction is critical |
| w/o RPL | 0.0054 | — | Localization and proportion estimation are effective |
| No proportion guidance (fixed kernels) | ↓ | ↓ | Dynamic selection outperforms fixed strategy |
| Thresholds [25%, 50%] | **Best** | **Best** | Both wider and narrower thresholds degrade performance |

Backbone comparison: Swin Transformer achieves Fβ $0.8563 \gg$ ViT $0.5762 \gg$ ResNet-50 $0.7756$. The model runs at 48.7 GFLOPs and 13 FPS on an RTX 3090.

### Key Findings

- The FCE module contributes the most; frequency-domain cross-layer interaction is the core driver of performance improvement.
- Region proportion-guided dynamic kernel selection consistently outperforms fixed kernel strategies.
- Swin Transformer's global context modeling capability is critical for remote sensing SOD.
- Failure cases are concentrated on extremely small objects and scenes where object texture is highly similar to the background.

## Highlights & Insights

1. **Region proportion → dynamic kernel selection** is an intuitive and effective design — deciding "how to look" based on "how large the object is."
2. Wavelet-domain frequency-matched interaction reduces computation to 1/4 of full-resolution self-attention while preventing high- and low-frequency information from interfering with each other.
3. The PG block directly supervises proportion prediction with MSE loss, providing a well-defined learning objective for dynamic selection rather than relying on pure heuristics.
4. MAE is reduced by 4.5%–25.8% over the previous SOTA across three datasets, representing substantial improvements.

## Limitations & Future Work

1. The inference speed of 13 FPS is insufficient for real-time remote sensing detection requirements.
2. The three-tier proportion thresholds (25%/50%) are manually defined; end-to-end learnable soft thresholds warrant investigation.
3. Failure cases indicate persistent difficulties with extremely small objects and objects with background-similar textures.
4. Evaluation is limited to three remote sensing SOD benchmarks without extension to natural image SOD or general segmentation tasks.

## Related Work & Insights

- **vs. GeleNet**: Also employs Transformers for remote sensing SOD but adopts a fixed feature extraction strategy. RDNet's core advantage lies in the dynamic kernel selection mechanism that adapts to objects of varying scales.
- **vs. ADSTNet**: An adaptive dual-stream Transformer; Fβ of 0.8321 vs. RDNet's 0.8563, with the gap attributable to targeted handling of multi-scale objects.
- **vs. HFCNet**: The closest competitor in MAE (0.0051 vs. 0.0049), yet a substantial Fβ gap (0.7845 vs. 0.8563) indicates insufficient region completeness.
- **Insights**: The region proportion-guided paradigm could be transferred to anchor-free detectors for dynamic receptive field adjustment; wavelet-domain feature interaction could be applied to multimodal feature fusion.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Region proportion-guided dynamic kernel selection is a genuinely novel contribution, though the overall framework remains within the encoder-decoder + attention paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comparisons with 21 methods, multiple ablation groups, and t-test statistical significance verification are comprehensive.
- **Writing Quality**: ⭐⭐⭐ Formulas and structure are clear, though some descriptions are redundant.
- **Value**: ⭐⭐⭐⭐ Practically valuable within the remote sensing SOD subfield; the dynamic kernel selection idea has reasonable generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] AFRO: Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning](bootstrap_dynamic-aware_3d_visual_representation_for_scalable_robot_learning.md)

</div>

<!-- RELATED:END -->
