---
title: >-
  [Paper Note] A Unified Image-Dense Annotation Generation Model for Underwater Scenes
description: >-
  [CVPR 2025][3D Vision][Underwater scenes] This paper proposes TIDE, a unified text-to-image and dense annotation generation method. Relying solely on text as input, it simultaneously generates highly consistent underwater images, depth maps, and semantic masks. By ensuring consistency across multimodal outputs through Implicit Layout Sharing (ILS) and Time Adaptive Normalization (TAN) mechanisms, the synthesized SynTIDE dataset significantly enhances the performance of underw…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Underwater scenes"
  - "data synthesis"
  - "diffusion models"
  - "depth estimation"
  - "semantic segmentation"
date: 2026-05-08
content_hash: 252351ec6f708a61
---

# A Unified Image-Dense Annotation Generation Model for Underwater Scenes

**Conference**: CVPR 2025  
**arXiv**: [2503.21771](https://arxiv.org/abs/2503.21771)  
**Code**: [https://github.com/HongkLin/TIDE](https://github.com/HongkLin/TIDE)  
**Area**: 3D Vision  
**Keywords**: Underwater scenes, data synthesis, diffusion models, depth estimation, semantic segmentation

## TL;DR
This paper proposes TIDE, a unified text-to-image and dense annotation generation method. Relying solely on text as input, it simultaneously generates highly consistent underwater images, depth maps, and semantic masks. By ensuring consistency across multimodal outputs through Implicit Layout Sharing (ILS) and Time Adaptive Normalization (TAN) mechanisms, the synthesized SynTIDE dataset significantly enhances the performance of underwater depth estimation and semantic segmentation.

## Background & Motivation
Underwater dense prediction (depth estimation and semantic segmentation) is a core technology for underwater exploration and environmental monitoring. However, high-quality, large-scale underwater dense annotation data is extremely scarce due to complex underwater environments and prohibitive data collection costs, representing a key bottleneck that constrains industrial and research development.

Prior work Atlantis utilized ControlNet conditioned on terrestrial depth maps to generate underwater depth data, achieving certain results. However, two core issues exist: 1) using terrestrial depth maps as conditions is suboptimal, and the generated data may not conform to the distribution of real underwater scenes; 2) it can only generate a single type of annotation (depth map), which fails to meet the needs of comprehensive underwater scene understanding.

The starting point of this work is an intuitive question: Can high-quality underwater images and multiple dense annotations be generated simultaneously using only text? The core challenge that needs to be addressed is how to maintain high consistency between the parallelly generated images and annotations.

## Method

### Overall Architecture
TIDE is built upon the pretrained PixArt-α text-to-image Transformer, incorporating three parallel denoising branches: text-to-image, text-to-depth, and text-to-mask. The three branches share a text encoder and achieve cross-modal alignment through the ILS and TAN mechanisms. During inference, consistent underwater images, depth maps, and semantic masks are output simultaneously using only a text description as input.

### Key Designs

1. **Implicit Layout Sharing (ILS)**:

    - Core Observation: In text-to-image models, cross-attention maps control the layout of the generated images.
    - The cross-attention map $\mathbf{M}_i = \text{softmax}(\mathbf{Q}_i \mathbf{K}_i^\top / \sqrt{c})$ calculated in the text-to-image branch is directly substituted into the depth and mask branches.
    - The cross-attention for the depth and mask branches is simplified to $\text{Attn}_d = \mathbf{M}_i \times \mathbf{V}_d$ and $\text{Attn}_m = \mathbf{M}_i \times \mathbf{V}_m$.
    - This design is elegant and efficient: it ensures layout consistency while reducing the cross-attention computation of the text-to-dense branches.
    - It leverages the strong layout control capability acquired by the text-to-image model during large-scale pretraining.

2. **Time Adaptive Normalization (TAN)**:

    - Considering the complementarity between features of different modalities, cross-modal feature interaction is introduced.
    - The cross-modal feature $\mathbf{x}_f$ is mapped via an MLP to two normalization parameters, $\gamma$ and $\beta$.
    - A time embedding $\mathbf{x}_t$ is introduced to generate an adaptive coefficient $\alpha$ (via linear transformation + Sigmoid) to control the intensity of cross-modal influence.
    - Normalization formula: $\mathbf{x}' = \alpha \cdot \gamma \mathbf{x} + \alpha \cdot \beta$, and $\mathbf{x}^* = \mathbf{x}' + \mathbf{x}$ (residual connection).
    - Interaction directions: bidirectional interaction between depth↔mask; bi-modal fusion of depth+mask→image (by taking the average $\bar{\gamma}$ and $\bar{\beta}$).
    - TAN and ILS complement each other: ILS guarantees macro layout consistency, while TAN further optimizes feature alignment at a detailed level.

3. **Data Preparation & Training Strategy**:

    - Approximately 14K quadruplets of {Image, Depth, Mask, Caption} are constructed based on existing underwater segmentation datasets (SUIM, UIIS, USIS10K).
    - Depth maps are generated by the pretrained Depth Anything model (pseudo-labels); captions are generated by BLIP2.
    - **Two-Stage Training**: (1) Mini-Transformer pretraining: initialized with the first 10 layers of PixArt-α and trained on the 14K image-text pairs for 60K iterations; (2) TIDE joint training: fine-tuning with LoRA for 200K iterations with a batch size of 4.
    - The LoRA ranks are 32 for text-to-image, 64 for text-to-depth, and 64 for text-to-mask.

4. **SynTIDE Dataset Synthesis**:

    - Approximately 5K non-redundant captions are obtained by deduplicating the 14K captions.
    - Ten samples are generated for each caption to build a large-scale synthetic dataset.
    - Underwater scenes unseen during training can be generated (zero-shot generation capability, benefiting from LoRA fine-tuning which preserves the generalization ability of the pretrained model).

### Loss & Training
The total loss is the sum of the denoising MSE losses of the three branches:
$$\mathcal{L} = \mathcal{L}_{mse}^I + \mathcal{L}_{mse}^D + \mathcal{L}_{mse}^M$$

Trainable parameters include only the TAN module and LoRA parameters, while the base Transformer weights are frozen.

## Key Experimental Results

### Main Results — Underwater Depth Estimation

| Model | Dataset | Metric | Atlantis | SynTIDE | Gain |
|------|--------|------|----------|---------|------|
| NewCRFs | Sea-thru D3+D5 | $SI_{log}$↓ | 37.10 | **22.37** | -14.73 |
| NewCRFs | Sea-thru D3+D5 | $\delta_1$↑ | 0.48 | **0.84** | +0.36 |
| AdaBins | Sea-thru D3+D5 | $SI_{log}$↓ | 38.24 | **26.92** | -11.32 |
| MIM | Sea-thru D3+D5 | $SI_{log}$↓ | 37.01 | **22.49** | -14.52 |
| PixelFormer | SQUID | $SI_{log}$↓ | 21.34 | **19.08** | -2.26 |

### Main Results — Underwater Semantic Segmentation

| Model | Training Data | UIIS mIoU | USIS10K mIoU |
|------|----------|-----------|-------------|
| Segformer | Real | 70.2 | 74.6 |
| Segformer | Real+SynTIDE | **75.4(+5.2)** | **76.1(+1.5)** |
| Mask2former | Real | 72.7 | 76.1 |
| Mask2former | Real+SynTIDE | **74.3(+1.6)** | **77.1(+1.0)** |
| ViT-Adapter | Real | 73.5 | 74.6 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| w/o ILS, w/o TAN | Low consistency | Baseline parallel generation |
| w/ ILS, w/o TAN | Consistent layout | Effective macro alignment |
| w/ ILS, w/ TAN | Highest consistency | ILS and TAN are complementary |

### Key Findings
- SynTIDE outperforms Atlantis across the board in depth estimation, especially on the NewCRFs model where the $SI_{log}$ improves by 14.73.
- The $\delta_1$ metric increases from 0.48 to 0.84 (by 36 percentage points), demonstrating that synthetic data significantly enhances the model's perception of underwater depth.
- Training semantic segmentation models with SynTIDE alone performs comparably to real data, and joint training with real data yields the best results.
- The zero-shot generation capability enables TIDE to generate underwater scenes not covered by the training set.

## Highlights & Insights
- The design concept of a unified framework is forward-looking — generating multiple annotations at once is more efficient and consistent than step-by-step generation.
- The ILS mechanism is designed ingeniously: directly reusing the attention maps of the text-to-image model achieves layout consistency with zero extra computational overhead.
- TAN introduces adaptive regulation in the temporal dimension, allowing cross-modal interaction to have varying intensities across different diffusion timesteps, which is highly reasonable.
- Achieving such a significant performance boost with only 14K training samples + LoRA fine-tuning demonstrates that the method effectively leverages pretrained knowledge.
- Key insight in underwater scene data synthesis: text conditioning is more flexible than depth map conditioning and can cover more scene variations.

## Limitations & Future Work
- The ground truth depth maps are generated by Depth Anything (pseudo-labels), so the depth accuracy is limited by the performance of the monocular depth estimation model.
- Currently, only depth and semantic mask annotations are supported, which could be extended to surface normals, etc.
- The small size of the training dataset (14K) may limit generation diversity.
- The quality and realism of the generated images still rely on the capability of the pretrained text-to-image model.
- Underperformance or minor performance drops in some models on the SQUID dataset (e.g., S.Rel) suggest that a gap still exists between the synthetic data distribution and certain real scenes.
- Currently, only underwater scenes have been validated; the effect of transferring the approach to other data-scarce domains remains to be verified.

## Related Work & Insights
- Atlantis pioneered generative methods for resolving underwater depth data scarcity but is limited to a single annotation type and terrestrial depth conditions.
- FreeMask and SegGen demonstrated the capability of text-conditioned segmentation data synthesis, but they are single-task models.
- The ControlNet series of methods uses image conditions to control generation; this work does the opposite, using text conditions to achieve multi-annotation generation.
- The Transformer architecture of PixArt-α provides space for ILS — block-level attention map sharing is natural.
- The data synthesis paradigm in this paper can be extended to other data-scarce domains (e.g., medical imaging, remote sensing, etc.).

## Rating
- **Novelty**: ⭐⭐⭐⭐ First method to simultaneously generate images and multiple dense annotations from text, featuring elegant designs of ILS and TAN.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Fully validated across two downstream tasks (depth estimation and semantic segmentation) with multiple models and datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly articulated motivation, intuitive descriptions of methods, and high-quality illustrations.
- **Value**: ⭐⭐⭐⭐ Provides an effective solution for data-scarce scenarios, showing strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)
- [\[ICCV 2025\] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](../../ICCV2025/3d_vision/relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [\[CVPR 2025\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)

</div>

<!-- RELATED:END -->
