---
title: >-
  [Paper Note] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast
description: >-
  [ICCV 2025][Medical Imaging][MRI brain tumor segmentation] M-Net reinterprets the spatial continuity between adjacent MRI slices as "quasi-temporal" data…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "MRI brain tumor segmentation"
  - "sequential modeling"
  - "Mesh-Cast mechanism"
  - "spatiotemporal correlation"
  - "two-phase training"
date: 2026-05-08
content_hash: a0ad46e2996ca90c
---

# M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast

**Conference**: ICCV 2025
**arXiv**: [2507.20582](https://arxiv.org/abs/2507.20582)  
**Code**: N/A  
**Area**: Medical Imaging
**Keywords**: MRI brain tumor segmentation, sequential modeling, Mesh-Cast mechanism, spatiotemporal correlation, two-phase training

## TL;DR

M-Net reinterprets the spatial continuity between adjacent MRI slices as "quasi-temporal" data, and proposes the Mesh-Cast mechanism to seamlessly integrate arbitrary sequential models (LSTM, Transformer, Mamba SSM, etc.) into both channel and temporal information processing. Combined with a Two-Phase Sequential training strategy (TPS), M-Net achieves state-of-the-art segmentation performance on BraTS2019 and BraTS2023.

## Background & Motivation

**Background**: MRI segmentation of brain tumors is critical for disease diagnosis and treatment planning. Deep learning methods have continuously evolved from UNet, including attention-based architectures such as CANet and MIRAU-Net, KAN-based approaches such as UKAN, and hybrid architectures such as TransUNet and Swin UNETR. Mamba SSM-based methods such as Mamba UNet have also demonstrated strong performance in medical image segmentation.

**Limitations of Prior Work**: Most existing methods either process MRI slices independently (2D methods) or apply 3D convolutions to the entire volume. 2D methods fail to exploit spatial continuity between adjacent slices, resulting in poor segmentation consistency; 3D methods, while capable of capturing volumetric information, incur prohibitively high computational costs that limit practical deployment. Although sequential modules from language models have been introduced into the visual domain, they typically operate on patch sequences within a single image and do not fully exploit the "quasi-temporal" spatial correlation between MRI slices.

**Key Challenge**: Adjacent MRI slices exhibit clear spatial continuity—the size and location of lesion regions change smoothly across slices, analogous to the temporal relationships between video frames. However, within a 2D slice processing framework, this cross-slice "quasi-temporal" information is unobservable, imposing a significant performance ceiling on 2D methods relative to their 3D counterparts.

**Goal**: To design a segmentation framework that exploits the "quasi-temporal" spatial correlation between MRI slices while retaining the computational efficiency of 2D methods.

**Key Insight**: The paper draws an analogy between MRI slice sequences and video frame sequences, introducing sequential modeling into MRI slice segmentation. The key innovation is a general-purpose Mesh-Cast mechanism that flexibly embeds arbitrary sequential processing algorithms along both the temporal and channel dimensions.

**Core Idea**: By alternately propagating sequential information along both temporal and channel dimensions via the Mesh-Cast mechanism, 2D slice-based methods can capture 3D volumetric contextual information while maintaining computational efficiency.

## Method

### Overall Architecture

M-Net adopts a classic encoder–decoder structure with skip connections. The input is a set of multi-modal MRI sequences $X = \{x_1, x_2, \ldots, x_T\}$, where each slice $x_t \in \mathbb{R}^{H \times W \times C}$. Each layer contains a Vision Sequential Module (for intra-frame spatial information) and a Mesh-Cast Sequential Module (for cross-frame temporal and channel information). The output is a segmentation mask for each slice.

### Key Designs

1. **Mesh-Cast Sequential Module**:

    - **Function**: The core component, which alternately performs sequential modeling along the temporal and channel dimensions to capture "quasi-temporal" spatial correlations between MRI slices.
    - **Mechanism**: Given a feature sequence $X_{in}$, the module first treats channels $C$ as the batch dimension and applies a sequential model to capture temporal correlations across $T$ frames. A Mesh-Cast Forward operation then swaps the $C$ and $T$ dimensions, enabling the sequential model to operate along the channel dimension and capture complementary feature correlations across different MRI modalities (T1, T1c, T2, FLAIR). Mesh-Cast Backward restores the original dimensions. When multiple layers are stacked, a Squeeze-and-Excitation (SE) hierarchical attention mechanism is used to weight the outputs of each layer.
    - **Design Motivation**: Temporal modeling alone captures only spatial positional changes, while channel-dimension modeling exploits complementary information across MRI modalities. The dimension-swapping mechanism of Mesh-Cast enables the same sequential model to serve both dimensions simultaneously, and the sequential algorithm can be freely replaced with LSTM, Transformer, Mamba SSM, or any other model.

2. **Two-Phase Sequential Training Strategy (TPS)**:

    - **Function**: Enhances model generalization and robustness.
    - **Mechanism**: In Phase 1, frame-level shuffling is applied to the input sequences, randomly combining slices from different sequences and positions to form new sequences for training, encouraging the model to learn universal feature patterns across sequences. In Phase 2, the original ordered sequences are restored for fine-tuning, allowing the model to learn genuine temporal dependencies. Phase 1 shuffling provides data diversity and accelerates convergence, while Phase 2 ordered training enables fine-grained sequential correlation learning.
    - **Design Motivation**: Training directly on ordered sequences tends to overfit to specific sequential patterns. The shuffle-then-order two-phase strategy encourages the model to first learn general anatomical structural features before focusing on sequence-specific contextual dependencies. Experiments show that the shuffle-then-order ordering significantly outperforms order-then-shuffle.

3. **Vision Sequential Module**:

    - **Function**: Extracts spatial sequential features within a single frame.
    - **Mechanism**: A Cross-Scan strategy serializes 2D images along four directions (left→right, right→left, top→bottom, bottom→top), applies sequential modules to each direction to extract spatial correlations, and merges the resulting features. The module shares the same sequential module interface as the Mesh-Cast Sequential Module.
    - **Design Motivation**: Ensures that the model captures not only cross-slice temporal information but also rich spatial features within individual slices.

### Loss & Training

A combined loss function of BCE Loss and Dice Loss is employed: $L_{joint} = \sum_{i=1}^{3} (\lambda L_{Dice}^i + (1-\lambda) L_{BCE}^i)$. The multi-class segmentation task is reformulated as multi-channel binary segmentation, with losses computed separately for the three tumor sub-regions: WT, TC, and ET. Dice Loss addresses class imbalance, while BCE Loss provides pixel-level error computation.

## Key Experimental Results

### Main Results

Comparison with 12 state-of-the-art methods on BraTS 2019 and BraTS 2023 (format: BraTS2019/BraTS2023):

| Method | Year | FLOPs | Dice-WT↑ | Dice-TC↑ | Dice-ET↑ | Haus95-WT↓ |
|--------|------|-------|----------|----------|----------|-----------|
| UNet | 2015 | 321G | 87.36/90.71 | 88.59/93.05 | 90.69/93.36 | 1.358/1.186 |
| Swin UNETR | 2022 | 137G | 88.16/91.11 | 88.85/93.20 | 90.86/93.42 | 1.308/1.163 |
| Mamba UNet | 2024 | 72G | 88.21/91.03 | 90.11/93.32 | 90.86/93.31 | 1.306/1.173 |
| nnUNet | 2021 | 82G | 87.81/90.34 | 90.23/92.74 | 90.96/92.37 | 1.297/1.210 |
| **M-Net** | Ours | **91G** | **88.38/91.33** | **90.52/93.55** | **91.43/93.42** | **1.287/1.153** |

### Ablation Study

Ablation of different sequential models and training strategies on BraTS 2019:

| Configuration | Dice-WT | Dice-TC | Dice-ET | Notes |
|---------------|---------|---------|---------|-------|
| Backbone (Slices) | 87.17 | 89.29 | 90.41 | Baseline without sequential modules |
| Mamba SSM (Slices) | 88.05 | 90.21 | 90.65 | Slice input only |
| Mamba SSM (TPS) | **88.38** | **90.52** | **91.43** | Full TPS strategy |
| M-Net (T only) | 87.86 | 89.28 | 90.93 | Temporal modeling only |
| M-Net (T+C, TPS) | **88.38** | **90.52** | **91.43** | Full model with temporal + channel |
| M-Net (T+C, Ordered) | 88.05 | 90.21 | 90.65 | Ordered training only |
| M-Net (T+C, Shuffled) | 88.07 | 90.32 | 91.05 | Shuffle training only |

### Key Findings

- **Mamba SSM performs best**: Among all evaluated sequential models, Mamba SSM achieves the highest performance with the smallest additional FLOPs (91.29G vs. baseline 72.44G).
- **Channel modeling is indispensable**: Temporal-only modeling (T only) trails the full temporal-and-channel model (T+C) by 1.24% on TC, confirming the importance of cross-modal feature modeling in the channel dimension.
- **TPS strategy is effective**: The shuffle-then-order two-phase strategy outperforms any single-phase strategy and also surpasses the reverse order-then-shuffle configuration.
- **Inference efficiency is excellent**: M-Net requires only 15 minutes for inference, compared to 97 minutes for nnUNet (approximately 16% of nnUNet's inference time).

## Highlights & Insights

- **A novel quasi-temporal modeling perspective**: Redefining the spatial relationships between MRI slices as "quasi-temporal" data is a precise and inspiring analogy. This perspective is transferable to any sequentially structured data with spatial continuity, such as CT scans and ultrasound sequences.
- **Dimension-swapping design of Mesh-Cast**: Through a simple dimension transposition, the same sequential module alternately operates along both the temporal and channel dimensions—an elegant and efficient design. This principle can be generalized to other scenarios requiring multi-dimensional sequential modeling.
- **Debiasing effect of the TPS training strategy**: Pre-training with shuffled data forces the model to learn universal patterns, followed by fine-tuning on ordered data to capture specific dependencies. This "generalize-first, specialize-later" training paradigm offers a broadly applicable methodological reference.

## Limitations & Future Work

- Validation is currently limited to brain tumor MRI segmentation and has not been extended to other medical imaging modalities (CT, ultrasound, etc.).
- The Mesh-Cast mechanism requires the sequential module to execute once along each of the temporal and channel dimensions; computational cost may still scale when both $T$ and $C$ are large.
- Evaluation is limited to two datasets (BraTS 2019 and BraTS 2023), restricting data diversity.
- Whether sequential models can effectively capture long-range dependencies for highly irregular or spatially dispersed tumors remains to be verified.

## Related Work & Insights

- **vs. Swin UNETR**: Swin UNETR applies a Transformer architecture to 3D volumetric data at high computational cost (137G). M-Net processes 2D slices while capturing 3D information via Mesh-Cast, achieving higher efficiency and superior performance.
- **vs. Mamba UNet**: Mamba UNet applies Mamba SSM to sequential processing within a single frame. M-Net elevates Mamba SSM to the cross-frame temporal dimension, fully exploiting spatial continuity between slices.
- **vs. nnUNet**: nnUNet is a strong auto-configured baseline with extremely long inference time (97 minutes). M-Net outperforms nnUNet on all metrics while being more than six times faster at inference.

## Rating

- Novelty: ⭐⭐⭐⭐ The Mesh-Cast dimension-swapping mechanism is elegantly designed, and the quasi-temporal modeling perspective is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons against 12 methods on two datasets, with ablations covering sequential model selection, Mesh-Cast components, and the TPS strategy.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and detailed illustrations.
- Value: ⭐⭐⭐⭐ Provides a general MRI sequential segmentation framework; Mesh-Cast is extensible to other imaging modalities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](../../CVPR2026/medical_imaging/pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[ICCV 2025\] Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data](scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data.md)
- [\[ICCV 2025\] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities](mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](radgpt_constructing_3d_image-text_tumor_datasets.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)

</div>

<!-- RELATED:END -->
