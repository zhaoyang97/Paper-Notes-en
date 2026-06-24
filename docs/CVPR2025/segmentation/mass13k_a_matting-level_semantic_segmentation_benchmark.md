---
title: >-
  [Paper Note] MaSS13K: A Matting-level Semantic Segmentation Benchmark
description: >-
  [CVPR 2025][Segmentation][High-Resolution Segmentation] This work constructs MaSS13K, a matting-level semantic segmentation dataset containing 13,348 images at 4K resolution (with mask complexity 20-50 times higher than existing datasets), and proposes the MaSSFormer model, which utilizes a dual-branch pixel decoder (global semantics + local structure) to achieve high-quality segmentation of fine boundaries in high-resolution scenarios while maintaining computational efficien…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "High-Resolution Segmentation"
  - "Matting-Level Annotation"
  - "Semantic Segmentation Benchmark"
  - "Boundary Quality"
  - "Pixel Decoder"
date: 2026-05-08
content_hash: 8226e863b517277d
---

# MaSS13K: A Matting-level Semantic Segmentation Benchmark

**Conference**: CVPR 2025  
**arXiv**: [2503.18364](https://arxiv.org/abs/2503.18364)  
**Code**: [https://github.com/xiechenxi99/MaSS13K](https://github.com/xiechenxi99/MaSS13K)  
**Area**: Semantic Segmentation  
**Keywords**: High-Resolution Segmentation, Matting-Level Annotation, Semantic Segmentation Benchmark, Boundary Quality, Pixel Decoder

## TL;DR

This work constructs MaSS13K, a matting-level semantic segmentation dataset containing 13,348 images at 4K resolution (with mask complexity 20-50 times higher than existing datasets), and proposes the MaSSFormer model, which utilizes a dual-branch pixel decoder (global semantics + local structure) to achieve high-quality segmentation of fine boundaries in high-resolution scenarios while maintaining computational efficiency.

## Background & Motivation

**Background**: Semantic segmentation has made significant progress on datasets such as COCO-Stuff and ADE20K, and methods like Mask2Former have shifted the paradigm from pixel classification to mask classification. However, the resolution of these datasets is generally lower than 1000×1000, and the annotation quality is coarse, failing to meet the high demands for fine mask details in image editing, bokeh photography, and AR/VR.

**Limitations of Prior Work**: (1) High-resolution datasets (e.g., Mapillary Vistas, EntitySeg) are still below 2K resolution and lack precise annotations; (2) High-precision binary segmentation datasets (e.g., DIS5K, matting datasets) have fine annotations but only support foreground/background separation and cannot perform full-scene semantic parsing; (3) Existing segmentation methods under 4K input suffer from relatively diminished receptive fields and difficulty in global semantic aggregation, coupled with inaccurate boundary detail extraction.

**Key Challenge**: How to guarantee global semantic correctness while extracting fine boundaries and local details under 4K high-resolution inputs, all while maintaining controllable computational costs?

**Key Insight**: This work approaches the problem from both data and model perspectives: constructing the first large-scale matting-level semantic segmentation dataset to provide high-quality annotations, and designing a lightweight pixel decoder specifically for high-resolution inputs to balance semantics and details.

## Method

### Overall Architecture

Based on the Mask2Former architecture, MaSSFormer consists of a pixel encoder, a pixel decoder, and a Transformer decoder. The core innovation lies in the pixel decoder: it splits the encoder features {S1-S4} into two groups, utilizing a global semantic branch to process {S2, S3, S4} for high-level semantic extraction, and a local structure branch to process {S1, I} for low-level detail extraction. Finally, an Edge-Guided Fusion (EGF) module merges the features from both paths to generate the high-resolution mask feature D1.

### Key Designs

1. **Cross-Semantic Transmission Module (CST) + Receptive Field Boosting Module (RFB)**:
    - CST enhances the global context of S4 through global average pooling, and then leverages window cross-attention to transfer spatial details of S3 to the upsampled D4. Finally, D3 is generated through window self-attention + deformable convolution.
    - RFB comprises four parallel deformable convolutions (kernel sizes 1/3/5/7) + pointwise convolutions to capture multi-scale structures while expanding the receptive field.
    - **Design Motivation**: The receptive field of standard convolutions is relatively small under 4K inputs. CST computes attention on low-resolution features to efficiently capture global semantics, while RFB further supplements multi-scale information.

2. **Low-level Structure Extraction Module (LSE)**:
    - S1 is upsampled to $H/2 \times W/2$ and concatenated with the downsampled image, extracting edge-aware features through channel splitting + spatial attention.
    - To reduce the computational overhead at $H/2$ resolution, features are split into two groups to be processed separately and then concatenated.
    - **Design Motivation**: Conventional methods discard 1/4 resolution features for efficiency, but high-resolution segmentation requires these high-frequency structural details.

3. **Edge-Guided Fusion Module (EGF)**:
    - First, S_detail and D2 are fused via addition and then processed by convolution to predict the edge map P^edge (supervised by BCE loss).
    - Then, Sigmoid(S_detail) is used as attention weights to refine D2, which then flows through channel compression $\to$ deformable convolution $\to$ residual connection to generate the final D1.
    - **Design Motivation**: The edge detection task forces the network to learn low-level structures, allowing S_detail to focus on edge regions and guiding the effective fusion of low-level and high-level features.

### Loss & Training

Based on the original loss of Mask2Former (classification loss L_cls + BCE + Dice), two additional terms are introduced:  
- **Weighted BCE Loss**: Edges are weighted by $(1 + \lambda W^i_j)$ using the edge weight map $W^i = G^i - f_{avg}(G^i, k)$ to strengthen boundary supervision.  
- **Edge Detection Loss L_edge**: Supervises the edge prediction results of the EGF module.  
- Total Loss: $L_{total} = L_{BCE}^w + L_{Dice} + L_{cls} + L_{edge}$

## Key Experimental Results

### Main Results

| Method | Backbone | mIoU (val) | BIoU (val) | BF1 (val) | Param | FLOPs |
|------|----------|-----------|-----------|----------|-------|-------|
| Mask2Former | R50 | 88.28 | 47.40 | .5458 | 44.01M | 3123G |
| MPFormer | R50 | 87.76 | 47.81 | .5513 | 43.9M | 4155G |
| MaSSFormer | R50 | **88.97** | **48.97** | **.5639** | 37.42M | 2036G |
| MaSSFormer-Lite | R18 | 87.11 | 45.35 | .5137 | 15.07M | 771G |

- MaSSFormer achieves the best performance across all metrics, with mIoU +0.69% and BIoU +1.57% (vs. second best), while requiring fewer FLOPs.
- MaSSFormer-Lite (R18) surpasses multiple R50 models in BIoU with only 15M parameters.

### Ablation Study

| Component | mIoU | BIoU | FLOPs |
|------|------|------|-------|
| Baseline | 85.54 | 42.69 | 1298G |
| +CST | 88.11 | 44.42 | 1348G |
| +CST+RFB | 88.29 | 45.23 | 1692G |
| +CST+RFB+LSE | 88.02 | 47.36 | 1928G |
| +CST+RFB+LSE+EGF (Full) | **88.97** | **48.97** | 2036G |

- CST contributes the most to the mIoU improvement (+2.57%), while LSE contributes the most to the BIoU improvement (+2.13%).
- Adding LSE alone slightly decreases mIoU (low-level features misleading high-level ones), but with EGF fusion, both metrics improve simultaneously.

### Key Findings

- The mIPQ mask complexity of MaSS13K (383±818) is ~20 times higher than that of EntitySeg (20±29) and ~3 times higher than that of DIS5K (116±452).
- On a stronger backbone (Swin-B), the BIoU gap between MaSSFormer and Mask2Former widens to +1.91%.
- New category extension experiment: Through a label decoupling strategy (precise labels with edge weights + pseudo labels with reverse weights), the BIoU of the Car category improved from 20.44 to 35.68.

## Highlights & Insights

- **Significant Dataset Contribution**: MaSS13K is the first large-scale matting-level semantic segmentation dataset, filling the gap of "high-precision annotation + multi-category semantic parsing". Its "others" category is not a traditional background class but contains finely segmented unnamed objects.
- **Exquisite Dual-Branch Decoder Design**: The global semantic branch efficiently aggregates global information at low resolutions, while the local structure branch extracts details at high resolutions. EGF guides the two-path fusion through an edge prediction task, making it more suitable for 4K scenarios than traditional FPN.
- **Generalization Mechanism for New Categories**: By learning edge segmentation capabilities from precise annotations + learning new category semantics from pseudo-labels + balancing both via a label decoupling strategy, it proves that high-quality annotations can transfer fine-grained segmentation capabilities to new categories.

## Limitations & Future Work

- It only contains 7 semantic categories, offering limited coverage.
- Extending to new categories requires class-by-class processing, and large-scale simultaneous multi-category extension is not yet supported.
- Segmenting extremely thin structures (e.g., power lines) remains challenging.
- Dataset construction cost is high (Photoshop matting-level annotation), making rapid expansion difficult.

## Related Work & Insights

- **Relationship with DIS5K**: DIS5K provides high-precision class-agnostic binary segmentation annotations, whereas MaSS13K extends this precision to multi-class semantic segmentation.
- **Evaluation Metric Insights**: BIoU and BF1 are more discriminative for evaluating high-resolution segmentation, as standard mIoU is too coarse to reflect differences in boundary quality.
- **Concept Transfer**: The dual-branch (semantics + details) architecture can be transferred to other dense prediction tasks requiring high-resolution outputs (such as matting, super-resolution guided segmentation, etc.).

## Rating

⭐⭐⭐⭐ — Outstanding dataset contribution (first matting-level multi-class semantic segmentation benchmark), reasonable and efficient model design, and valuable new category generalization mechanism. The limited number of categories and high annotation costs restrict its short-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MatAnyone: Stable Video Matting with Consistent Memory Propagation](matanyone_stable_video_matting_with_consistent_memory_propagation.md)
- [\[CVPR 2025\] RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring](ripvis_rip_currents_video_instance_segmentation_benchmark_for_beach_monitoring_a.md)
- [\[ICLR 2026\] Matting Anything 2: Towards Video Matting for Anything](../../ICLR2026/segmentation/matting_anything_2_towards_video_matting_for_anything.md)
- [\[CVPR 2025\] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)
- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)

</div>

<!-- RELATED:END -->
