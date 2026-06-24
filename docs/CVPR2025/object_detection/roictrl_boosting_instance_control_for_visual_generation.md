---
title: >-
  [Paper Note] ROICtrl: Boosting Instance Control for Visual Generation
description: >-
  [CVPR 2025][Object Detection][Instance Control] Inspired by ROI-Align in object detection, ROICtrl proposes a complementary operation, ROI-Unpool, to achieve efficient and precise ROI feature recovery. It constructs a diffusion model adapter compatible with community fine-tuned models and existing spatial/embedded plugins, achieving SOTA performance in multi-instance regional control generation while drastically reducing computational costs.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Instance Control"
  - "Regional Generation"
  - "ROI Operations"
  - "ControlNet Compatibility"
  - "Multi-Instance Image Generation"
date: 2026-05-08
content_hash: 6284820a47824c45
---

# ROICtrl: Boosting Instance Control for Visual Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.17949](https://arxiv.org/abs/2411.17949)  
**Code**: [https://roictrl.github.io/](https://roictrl.github.io/)  
**Area**: Diffusion Models / Object Detection  
**Keywords**: Instance Control, Regional Generation, ROI Operations, ControlNet Compatibility, Multi-Instance Image Generation

## TL;DR

Inspired by ROI-Align in object detection, ROICtrl proposes a complementary operation, ROI-Unpool, to achieve efficient and precise ROI feature recovery. It constructs a diffusion model adapter compatible with community fine-tuned models and existing spatial/embedded plugins, achieving SOTA performance in multi-instance regional control generation while drastically reducing computational costs.

## Background & Motivation

**Background**: Text-to-image diffusion models excel at simple compositions with few main objects, but natural language struggles to precisely associate spatial positions and attribute information with multiple instances. Instance control research aims to control the generation of each instance using bounding boxes and text descriptions.

**Limitations of Prior Work**: Existing instance control methods fall into two categories: (1) Implicit injection (e.g., GLIGEN) — encoding box coordinates as position embeddings and fusing them with instance descriptions to inject into global feature maps via self-attention, which suffers from severe attribute leakage and low spatial alignment accuracy; (2) Explicit injection (e.g., MIGC, Instance Diffusion) — using masked attention to isolate the injection of instance descriptions for each ROI, which achieves better spatial alignment but performs computations on full-resolution feature maps, leading to high computational overhead. In addition, coordinate quantization errors in mask creation also degrade spatial precision.

**Key Challenge**: Visual generation needs to process variable-sized ROIs on high-resolution feature maps ($64 \times 64$ or $128 \times 128$), whereas target detection ROI layer operations operate on low-resolution features and only connect to simple classification heads. In generation tasks, it is necessary to "paste back" the processed ROI features to their original positions—a challenge that prior methods bypassed using masked attention, but at the cost of massive redundant computations.

**Goal**: To design an efficient, precise, and ecosystem-compatible adapter for instance control.

**Key Insight**: Drawing inspiration from ROI-Align in object detection—if a complementary ROI-Unpool operation can be designed to precisely restore cropped ROI features to their original positions, it would enable explicit and efficient ROI processing, with computational costs decoupled from the original feature map size.

**Core Idea**: ROI-Align crops ROI features from spatial feature maps, and ROI-Unpool bilinearly interpolates the processed ROI features back to their original coordinates. Together, they enable processing ROIs at a fixed small size ($r \times r$) and pasting them back onto the high-resolution feature maps.

## Method

### Overall Architecture

ROICtrl adds a parallel instance description injection pathway to each cross-attention layer of the pretrained diffusion model. The global description generates global attention outputs through the pretrained cross-attention layer. The instance description pathway first extracts fixed-size ROI features from spatial features using ROI-Align, reuses the pretrained cross-attention to inject instance descriptions, refines them with ROI self-attention, and finally restores them to their original positions using ROI-Unpool to generate instance attention outputs. The two outputs are fused into the final output through a learnable attention blending mechanism.

### Key Designs

1. **ROI-Unpool Operation**:

    - **Function**: Precisely restores cropped fixed-size ROI features to their correct positions on the original high-resolution feature maps.
    - **Mechanism**: Symmetrical to ROI-Align—while ROI-Align computes ROI feature values by bilinearly sampling four nearest grid points on the original feature map, ROI-Unpool bilinearly distributes the ROI feature values back to the spatial feature map via the four nearest grid points. For boundary regions lacking four sampling points, partial values are calculated using the available points. Locations outside the ROI regions remain empty. The entire process does not involve coordinate quantization, avoiding the quantization errors of masked attention.
    - **Design Motivation**: Prior methods used masked attention to bypass the challenge of "pasting back" variable-sized ROIs, which introduced massive redundant computations (all operations executed at full resolution) and coordinate quantization errors. ROI-Unpool directly addresses this issue, making the computational cost proportional to the ROI size rather than the feature map size.

2. **Instance Description Injection and Learnables Attention Blending**:

    - **Function**: Precisely injects specific descriptions of each instance while maintaining global information.
    - **Mechanism**: The global attention output $A_g$ and $n$ instance attention outputs $A_r$ are concatenated along the ROI axis into $A \in \mathbb{R}^{b \times (n+1) \times c \times h \times w}$. A $1 \times 1$ convolution computes the learnable fusion weight $W$, which is applied along the ROI axis via softmax for weighted fusion to obtain the final output. The instance description injection reuses the pretrained cross-attention (without adding new learnable modules) to guarantee compatibility with embedded plugins (e.g., IP-Adapter, ED-LoRA). ROI self-attention is also introduced to adapt to the discrepancies between ROI features and original spatial resolutions.
    - **Design Motivation**: Reusing the pretrained cross-attention instead of building new modules is a key design choice to ensure compatibility with pretrained cross-attention-based embedded plugins like IP-Adapter. The fusion weight allows the model to dynamically decide whether each spatial location should focus more on global descriptions or instance descriptions.

3. **Bounding Box Coordinate Embedding Guidance**:

    - **Function**: Enhances the "objectness" of ROI regions and improves performance in occlusion scenarios.
    - **Mechanism**: Borrowed from GLIGEN's box embedding approach, but only using box coordinate embeddings without instance description embeddings to prevent attribute leakage. Coordinate conditioning enhances the object generation tendency in corresponding regions and improves spatial alignment.
    - **Design Motivation**: The mixed embedding of boxes and text in GLIGEN is a major source of attribute leakage. Using only box coordinates provides a spatial position prior without introducing semantic confusion across instances.

### Loss & Training

- Standard diffusion loss: $\mathcal{L}_{LDM} = \mathbb{E}[|\epsilon - \epsilon_\theta(z_t, t, \phi(p_g, p_r, c_r))|_2^2]$
- Fusion weight regularization: $\mathcal{L}_{reg} = |M \odot W_{:,1,:,:}|_1 / |M|_1$, which reduces the weight of global attention outputs in ROI foreground regions and promotes the alignment of instance descriptions.
- Total loss: $\mathcal{L} = \mathcal{L}_{LDM} + 0.01 \cdot \mathcal{L}_{reg}$
- Once trained on a base model, it can be directly migrated to all community fine-tuned versions.

## Key Experimental Results

### Main Results

**MIG-Bench (Template Prompts)**:

| Method | mIoU AVG | Instance Success Rate AVG |
|------|----------|---------------|
| GLIGEN | 0.27 | 0.30 |
| MIGC | 0.56 | 0.66 |
| Instance Diffusion | 0.46 | 0.51 |
| **ROICtrl** | **0.66** | **0.73** |

**ROICtrl-Bench (Free-form Prompts)**:

| Method | mIoU AVG | Acc AVG |
|------|----------|---------|
| GLIGEN | 0.537 | 30.2% |
| MIGC | 0.490 | 38.9% |
| Instance Diffusion | 0.607 | 45.6% |
| **ROICtrl** | **0.652** | **48.7%** |

### Ablation Study

| Method | mIoU | Acc | Train VRAM | Inference Speed | Supports Embedded Plugins |
|------|------|-----|---------|---------|------------|
| **Ours** (ROICtrl) | 0.652 | 48.7 | 34.3G | 13.1s/img | ✓ |
| Mask-Attn ROICtrl | 0.628 | 49.2 | 65.5G | 31.5s/img | ✓ |
| Instance Diffusion | 0.607 | 45.6 | - | - | × |

Inference speed tested on A100, $1024^2$ resolution, 25 ROIs, 50 DDIM steps.

### Key Findings

- ROICtrl outperforms all methods across all three benchmarks, particularly showing significant advantages on free-form prompts (Tracks 3 & 4).
- Compared to the masked attention version, ROICtrl using ROI-Align/Unpool halves VRAM usage (34.3G vs. 65.5G), speeds up execution by 2.4x (13.1s vs. 31.5s), and achieves better spatial alignment.
- Once trained, ROICtrl can be directly applied to community fine-tuned models (e.g., RealisticVision, DreamShaper) without retraining.
- Compatibility with ControlNet, T2I-Adapter, IP-Adapter, and ED-LoRA has been successfully verified.
- Implicit embedding injection (GLIGEN-style) performs significantly worse than explicit ROI injection across all metrics.

## Highlights & Insights

1. ROI-Unpool is a simple and elegant operation—perfectly symmetrical to ROI-Align, it solves the "pasting back" challenge that has plagued explicit ROI injection.
2. The design choice of reusing pretrained cross-attention instead of adding new modules is clever—trading zero extra parameters for compatibility with the entire ecosystem of embedded plugins.
3. Using only box coordinate embeddings (removing instance text embeddings) to avoid attribute leakage is an insightful trade-off.
4. The introduction of free-form prompt evaluation in ROICtrl-Bench fills the gap left by previous benchmarks that only covered template-based descriptions.

## Limitations & Future Work

- ROI-Unpool only supports rectangular bounding boxes and does not support arbitrary ROI shapes (such as segmentation masks).
- Performance shows a clear declining trend as the number of instances increases (at L5 and L6 levels).
- It does not directly support instance control in video generation.
- Future directions could explore generalizing ROI-Unpool to non-rectangular regions and video generation.

## Related Work & Insights

- The ROI-Align $\rightarrow$ ROI-Unpool concept bridges the ROI processing paradigms of both object detection and image generation.
- The comparison with GLIGEN clearly demonstrates the fundamental difference between implicit vs. explicit ROI injection regarding spatial alignment and attribute leakage.
- The comparison with Instance Diffusion proves that efficiency and performance can be improved simultaneously without a trade-off.

## Rating

- **Novelty**: 8/10 — The concept of ROI-Unpool is simple yet profound, connecting ROI processing directions in both recognition and generation.
- **Experimental Thoroughness**: 9/10 — Highly comprehensive, involving three benchmarks + a new benchmark + ablation studies + efficiency comparisons + compatibility validation.
- **Writing Quality**: 8/10 — The analogy from object detection to generation is naturally introduced, and the problem definition is precise.
- **Value**: 8/10 — High compatibility and efficiency make it a promising candidate for a standard solution in instance control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Unseen Visual Anomaly Generation](unseen_visual_anomaly_generation.md)
- [\[CVPR 2025\] Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)
- [\[CVPR 2025\] Interpreting Object-level Foundation Models via Visual Precision Search](interpreting_object-level_foundation_models_via_visual_precision_search.md)
- [\[CVPR 2026\] Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](../../CVPR2026/object_detection/visual_prototype_conditioned_focal_region_generation_for_uav-based_object_detect.md)
- [\[CVPR 2025\] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](univad_a_training-free_unified_model_for_few-shot_visual_anomaly_detection.md)

</div>

<!-- RELATED:END -->
