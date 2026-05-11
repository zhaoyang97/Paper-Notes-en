---
title: >-
  [Paper Note] Segment and Matte Anything in a Unified Model (SAMA)
description: >-
  [AAAI 2026][Segmentation][SAM Extension] This paper proposes SAMA — a lightweight extension of SAM that introduces a Multi-View Local Encoder (MVLE) to capture fine-grained local features…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "SAM Extension"
  - "Unified Segmentation and Matting"
  - "Multi-View Local Encoder"
  - "Local-Adapter"
  - "Interactive Segmentation"
  - "Alpha Matting"
date: 2026-05-08
content_hash: 7d8e981bebd62695
---

# Segment and Matte Anything in a Unified Model (SAMA)

**Conference**: AAAI 2026
**arXiv**: [2601.12147](https://arxiv.org/abs/2601.12147)
**Authors**: Zezhong Fan, Xiaohan Li, Topojoy Biswas, Kaushiki Nag, Kannan Achan
**Area**: Image Segmentation / Image Matting
**Keywords**: SAM Extension, Unified Segmentation and Matting, Multi-View Local Encoder, Local-Adapter, Interactive Segmentation, Alpha Matting

## TL;DR
This paper proposes SAMA — a lightweight extension of SAM that introduces a Multi-View Local Encoder (MVLE) to capture fine-grained local features, a Local-Adapter to inject local details into the decoding process, and dual task-specific prediction heads. With only a 1.8% parameter increase, SAMA achieves high-quality interactive segmentation and alpha matting simultaneously within a unified model, reaching state-of-the-art performance on DIS-5K and multiple matting benchmarks.

## Background & Motivation

Precise object segmentation is a core task in computer vision, encompassing semantic/instance segmentation (assigning category labels to each pixel) and natural image matting (generating continuous alpha mattes to capture semi-transparent boundaries such as hair and glass). SAM represents a milestone in segmentation, demonstrating strong zero-shot generalization after training on over one billion masks. However, SAM's raw masks frequently lack tight boundary adherence and sub-pixel precision.

Existing improvements such as HQ-SAM, DIS-SAM, and Pi-SAM enhance segmentation quality, yet two critical challenges remain:

**Insufficient fine-grained perception**: As an interactive segmentation model, SAM struggles to capture the fine structural details of target objects.

**Difficulty integrating high-resolution details**: Incorporating high-resolution information during decoding while preserving zero-shot generalization is non-trivial.

On the other hand, interactive matting — which estimates precise alpha mattes from sparse user guidance — achieves excellent boundary detail but lacks object-level reasoning. Recent studies reveal a **strong correlation between segmentation and matting**: segmentation provides global object cues while matting supplies local boundary precision. The potential of exploiting this synergy to build a unified model remains largely unexplored.

## Method

### Overall Architecture
SAMA augments the frozen SAM parameters with three lightweight components:
- **Multi-View Local Encoder (MVLE)**: Extracts high-resolution features from multiple local views.
- **Local-Adapter**: Injects local features into the SAM decoding process.
- **Dual task-specific prediction heads**: Separate heads for segmentation and matting output.

The overall pipeline treats the input image as a global view while simultaneously cropping it into four non-overlapping local patches as local views. Both global and local views are encoded through the SAM encoder; the resulting feature maps are then fused via MVLE, refined via the Local-Adapter, and passed to the prediction heads.

### Multi-View Local Encoder (MVLE)
SAM relies solely on a single global representation, limiting its ability to capture fine-grained visual details. MVLE is inspired by the human visual system, which distinguishes distant global context from close-up local detail:

1. The input image is uniformly cropped into four non-overlapping local patches.
2. Each patch is upsampled to the original resolution and passed through the shared encoder to extract high-resolution local feature maps.
3. Multi-scale average pooling (with receptive fields of 4/8/16) is applied to the global features to obtain multi-scale context representations.
4. Within each spatial region, cross-attention is performed with local features as queries and global pooled features as keys/values for alignment.

### Local-Adapter
The Local-Adapter injects high-resolution local features into the SAM decoder through three steps:

1. **First cross-attention layer**: Local features from MVLE and early-layer encoder features are fused via residual connection as keys/values, with decoder outputs as queries, enabling local–global information integration.
2. **Second cross-attention layer**: Query and key-value roles are swapped (inspired by GLIP and GroundingDINO), enabling bidirectional interaction so that the adapter acquires both global and local awareness.
3. **Confidence map fusion**: A confidence map $C$ is generated via a $1\times1$ convolution followed by Sigmoid, and element-wise multiplied with the cross-attention output before being added back to the global features. This mechanism protects SAM's zero-shot generalization capability and prevents overfitting and catastrophic forgetting.

### Dual Task-Specific Prediction Heads
- Two learnable SAMA tokens (a segmentation token and a matting token) replace the original SAM output tokens.
- Two lightweight task-specific prediction heads process segmentation and matting respectively, reconstructing fine details through interpolation upsampling followed by convolutional layers (BN + GeLU).
- The design enables simultaneous generation of high-resolution segmentation masks and alpha mattes.

### Loss & Training
- **Data**: Segmentation uses DIS-5K and ThinObject-5K (high-quality annotations); matting uses AIM and AIM-500.
- The **SAM backbone is frozen**; only the newly added modules are trained. The matting head is frozen during segmentation training and vice versa.
- **Segmentation loss**: $\mathcal{L}_{seg} = \mathcal{L}_{BCE} + \mathcal{L}_{IoU} + \mathcal{L}_{SSIM}$
- **Matting loss**: $\mathcal{L}_{matting} = \mathcal{L}_{l_1} + \mathcal{L}_{SSIM} + \mathcal{L}_{Grad} + \mathcal{L}_{Laplacian}$

## Key Experimental Results

### Main Results I: DIS-5K Segmentation Benchmark (Table 1)

Comparison with SAM variants and dedicated segmentation models on the fine-grained DIS-5K dataset:

| Method | DIS-VD $F_\beta^{max}$↑ | DIS-VD MAE↓ | DIS-VD $S_\alpha$↑ | DIS-TE(All) $F_\beta^{max}$↑ | DIS-TE(All) MAE↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| SAM | 0.835 | 0.069 | 0.808 | 0.773 | 0.096 |
| HQ-SAM | 0.851 | 0.045 | 0.848 | 0.859 | 0.045 |
| Pi-SAM | 0.883 | 0.035 | 0.889 | 0.893 | 0.033 |
| DIS-SAM | 0.920 | 0.031 | 0.909 | 0.917 | 0.029 |
| BiRefNet | 0.891 | 0.038 | 0.898 | 0.896 | 0.035 |
| **SAMA** | **0.942** | **0.021** | **0.930** | **0.926** | **0.026** |

**Key Findings**: SAMA outperforms all SAM-based models across all metrics, achieving $F_\beta^{max}$ of 0.942 and MAE of 0.021 on DIS-VD. SAMA also remains competitive against BiRefNet, which is specifically trained for DIS tasks.

### Main Results II: Matting Benchmarks (Table 2)

Comparison with trimap-based and trimap-free methods on Composition-1K and Distinction-646:

| Method | Type | Comp-1K SAD↓ | Comp-1K MSE↓ | Dist-646 SAD↓ | Dist-646 MSE↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| VITMatte | trimap-based | 21.5 | 3.3 | 21.22 | 2.1 |
| MODNet | trimap-free | 47.1 | 12.3 | 41.7 | 9.0 |
| MFC-Net | trimap-free | 35.6 | 8.7 | 34.5 | 7.8 |
| **SAMA** | **trimap-free** | **22.8** | **2.9** | **22.4** | **2.2** |

**Key Findings**: SAMA achieves state-of-the-art performance among trimap-free methods, substantially outperforming MODNet and MFC-Net. Notably, SAMA approaches the best trimap-based method, VITMatte, without requiring trimap input — and even surpasses it on Comp-1K MSE (2.9 vs. 3.3), demonstrating strong generalization capability.

### Ablation Study (Tables 3–5)
- **MVLE + Local-Adapter**: Both components are essential. On DIS-VD, the baseline $F_\beta^{max}$ is 0.872; adding MVLE alone yields 0.882, adding Local-Adapter alone yields 0.893, and combining both reaches 0.942 — an 8% overall gain.
- **Multi-task learning**: Joint training outperforms single-task training. Joint training reduces matting SAD from 62.70 to 25.69 on RefMatte-RW100; the boundary detail learned from matting data reciprocally improves segmentation accuracy.

## Highlights & Insights
- **Pioneer unified framework**: SAMA is the first SAM-based model that simultaneously performs interactive segmentation and matting, with only a 1.8% parameter overhead.
- **MVLE multi-view strategy**: Cropping inputs into local patches and upsampling them simulates the human visual system's differentiated processing of near and far views, effectively enhancing fine-grained perception.
- **Confidence map protection mechanism**: The Local-Adapter uses confidence-gated fusion of local information, elegantly balancing accuracy improvement against zero-shot generalization.
- **Task complementarity gains**: Experiments confirm that joint training of segmentation and matting is mutually beneficial — segmentation provides global semantics while matting supplies boundary precision.

## Limitations & Future Work
- Experiments are conducted exclusively on images; extension to video segmentation/matting scenarios is not explored.
- Multi-view cropping with separate encoding increases inference latency — while the paper claims marginal overhead, four encoder forward passes cannot be ignored.
- Training data is limited (DIS-5K and AIM); performance on larger-scale data is not validated.
- Compatibility with subsequent SAM variants such as SAM2/SAM3 is not discussed.

## Related Work & Insights
- **Interactive segmentation**: SAM variants including HQ-SAM, Pi-SAM, and DIS-SAM improve segmentation accuracy or extend functionality.
- **Image matting**: Trimap-based methods (DIM, VITMatte) and trimap-free methods (MODNet, MAM, MatAny).
- **Unified segmentation and matting**: Prior work has identified strong structural correlations between the two tasks (Wang & Cohen 2005; Zheng et al. 2024), but unified modeling remains underexplored.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](segment_anything_across_shots_a_method_and_benchmark.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [\[AAAI 2026\] Tracking and Segmenting Anything in Any Modality](tracking_and_segmenting_anything_in_any_modality.md)

</div>

<!-- RELATED:END -->
