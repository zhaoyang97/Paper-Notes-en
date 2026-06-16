---
title: >-
  [Paper Note] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models
description: >-
  [ICCV 2025][Self-Supervised Learning][feature upsampling] LoftUp is proposed to map low-resolution VFM features to arbitrary high resolutions via a coordinate-cross-attention architecture…
tags:
  - "ICCV 2025"
  - "Self-Supervised Learning"
  - "feature upsampling"
  - "vision foundation models"
  - "coordinate-based"
  - "self-distillation"
  - "DINOv2"
  - "dense prediction"
date: 2026-05-08
content_hash: 2958934f2a73a8c6
---

# LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models

**Conference**: ICCV 2025
**arXiv**: [2504.14032](https://arxiv.org/abs/2504.14032)  
**Code**: [https://github.com/andrehuang/loftup](https://github.com/andrehuang/loftup)  
**Area**: Self-Supervised Learning / Vision Foundation Models / Feature Upsampling
**Keywords**: feature upsampling, vision foundation models, coordinate-based, self-distillation, DINOv2, dense prediction

## TL;DR
LoftUp is proposed to map low-resolution VFM features to arbitrary high resolutions via a coordinate-cross-attention architecture, with class-agnostic mask refinement and self-distillation to construct full-resolution pseudo-GT for training, achieving average improvements of 10–20% across 6 downstream tasks and nearly 50% on video object segmentation.

## Background & Motivation
Feature maps produced by VFMs such as DINOv2 and CLIP typically have a spatial resolution of only 1/16 of the input (e.g., 224-pixel input → 16×16 features), which severely limits dense prediction tasks requiring pixel-level understanding (segmentation, depth estimation, etc.). Existing approaches include: (1) increasing input resolution — leading to prohibitive computation; (2) training task-specific decoders — lacking generalizability; (3) general-purpose feature upsamplers such as FeatUp/LiFT — but these compute losses at low resolution, providing insufficient constraints on high-resolution details, and tend to produce blurry outputs and artifacts.

## Core Problem
How to design a task-agnostic, plug-and-play VFM feature upsampler capable of generating sharp, high-quality full-resolution feature maps? Two core challenges arise: (1) upsampler architecture — how to avoid cumulative blurring from multi-stage upsampling; (2) training objective — without high-resolution GT features, how to construct effective supervision signals.

## Method

### Overall Architecture
LoftUp is a lightweight 2-layer cross-attention transformer (< 20% of VFM parameters) trained in two stages. Stage 1 uses class-agnostic masks from SAM to refine bicubic-upsampled features as pseudo-GT. Stage 2 applies self-distillation (teacher processes high-resolution crops; student processes standard-resolution inputs) to generate higher-quality pseudo-GT. After training, LoftUp is plug-and-play with no test-time optimization required.

### Key Designs
1. **Coordinate-cross-attention architecture**: Inspired by implicit neural representations in 3D reconstruction (e.g., NeRF), feature upsampling is formulated as a coordinate-to-feature mapping. Each pixel coordinate of the high-resolution image (sinusoidally encoded) together with its RGB value serves as the query, while low-resolution VFM features serve as keys and values; cross-attention enables global interaction to produce high-resolution features. This approach is sharper than multi-layer deconvolution/bilinear upsampling (avoiding cumulative blur), more expressive than LIIF's local MLP (global attention), and supports arbitrary upsampling factors.

2. **Stage 1: SAM mask-refined pseudo-GT**: Low-resolution features are bicubically upsampled to full resolution, then class-agnostic masks from SAM are used to compute mean features per mask region and blend them with the original features ($F_{\text{Mask-Bicubic}}[m] = \alpha \cdot \overline{F_{\text{Bicubic}}[m]} + (1-\alpha) \cdot F_{\text{Bicubic}}[m]$, $\alpha=0.8$). This leverages SAM's boundary information to produce features that are smooth within objects and sharp at boundaries.

3. **Stage 2: Self-distillation refinement**: Both teacher and student are initialized from the Stage 1 pretrained model. The teacher processes 2–4× higher-resolution crops (an easier task yielding higher-quality features); its output is downsampled to supervise the student. The teacher is updated via EMA. This further reduces residual blurring and artifacts from Stage 1, producing geometrically more consistent pseudo-GT.

### Loss & Training
- Stage 1: L2 loss against mask-refined pseudo-GT
- Stage 2: Affinity matrix loss (outperforms L2); EMA decay = 0.99; teacher updated every 10 steps
- Trained on a 1M subset of SA1B; batch size 8; AdamW; Stage 1 lr = 1e-3; Stage 2 lr = 1e-4

## Key Experimental Results

| Task | Metric | Low-res | Bilinear | FeatUp | LiFT | **LoftUp** |
|------|--------|---------|----------|--------|------|------------|
| COCO Segmentation | mIoU | 51.21 | 56.15 | 56.30 | 53.35 | **61.11** |
| Cityscapes Segmentation | mIoU | 36.54 | 44.79 | 44.19 | 35.80 | **53.10** |
| NAVI Depth | δ3↑ | 89.08 | 87.68 | 88.57 | 88.71 | **91.35** |
| DAVIS VOS | J&F | 52.82 | 54.26 | 55.03 | 41.14 | **67.31** |
| OV Segmentation | mIoU | 25.70 | 25.78 | 26.61 | 25.96 | **27.82** |
| Interactive Segmentation | IoU@1 | 55.77 | 55.83 | 56.67 | 31.99 | **65.24** |

- Semantic segmentation: +7.3% over FeatUp on COCO and +15.6% on Cityscapes
- Video object segmentation: nearly **50% improvement** over the low-resolution baseline (J Mean: +39.6%; F Mean: +97.6%)
- Inference speed comparable to bilinear interpolation (0.089 s vs. 0.092 s), far faster than FeatUp-Implicit (54.3 s)
- Under the same training objective, the LoftUp architecture outperforms resize-conv/LIIF/FeatUp-JBU across all tasks
- Generalizes across VFMs: consistently outperforms all baselines on DINOv2, CLIP, and RADIO

### Ablation Study
- Sinusoidal positional encoding + 3×3 convolution for image input yields the optimal combination
- 2-layer cross-attention is sufficient; 3 layers provide no additional gain
- Pseudo-GT comparison: self-distillation > mask-bicubic > per-image optimization > 2× features
- Performance scales with training data from 50K to 1M, with diminishing returns
- Full-resolution loss is critical — the low-resolution losses used in FeatUp/LiFT are insufficient to constrain fine-grained details

## Highlights & Insights
- **Coordinate representation borrowed from 3D reconstruction**: Transferring the NeRF coordinate-to-color mapping paradigm to feature upsampling circumvents the fundamental limitations of conventional multi-stage upsampling
- **Elegant two-stage pseudo-GT construction**: SAM masks provide geometric boundaries → bicubic smoothing within regions → self-distillation for further refinement, progressively improving quality
- **Genuinely task-agnostic**: consistent improvements across 6 highly heterogeneous tasks (segmentation / depth / normals / VOS / open-vocabulary segmentation / interactive segmentation), demonstrating strong generalizability
- **< 20% additional parameters + bilinear-level inference speed**: the plug-and-play design is highly practical
- **Remarkable gains on VOS**: the ~50% improvement demonstrates that high-resolution features are critically important for temporal correspondence

## Limitations & Future Work
- Stage 1 relies on SAM for mask generation, introducing additional computational dependencies at training time
- Main experiments are conducted only with DINOv2-S/14; performance on larger models is not thoroughly validated
- Systematic evaluation at higher input resolutions (e.g., 512/1024) has not been performed
- The global cross-attention may face computational bottlenecks at very high resolutions

## Related Work & Insights
- **vs. FeatUp**: FeatUp employs a modified JBU architecture with multi-view reconstruction as low-resolution loss; LoftUp uses a coordinate cross-attention architecture with full-resolution self-distilled GT, advancing along both dimensions
- **vs. LiFT**: LiFT applies a U-Net for 2× upsampling with 2× features as GT; LoftUp directly maps to arbitrary resolutions with high-quality pseudo-GT
- **vs. LIIF**: LIIF is also coordinate-based but relies on local MLP interactions; LoftUp's cross-attention enables global interaction

The coordinate-based design confers resolution-agnostic flexibility, which is valuable for deployment across varying resolution scenarios. The pseudo-GT generation method combining SAM masks and self-distillation is transferable to other learning settings where GT is unavailable. This work is complementary to research on scaling language-free visual representations — SSL encoder features may hold even greater untapped potential at higher resolutions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Transferring coordinate representation from 3D to 2D upsampling, combined with the two-stage pseudo-GT design, constitutes two highly original contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 heterogeneous downstream tasks, multiple VFM backbones, component-wise ablations, pseudo-GT comparisons, and efficiency analysis
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem decomposition is clear (architecture vs. training objective); the pseudo-GT property comparison tables in Tables 1/2 are particularly instructive
- Value: ⭐⭐⭐⭐⭐ — Positioned as a universal enhancement module for all VFMs, with substantial impact on the dense prediction community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](../../NeurIPS2025/self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[CVPR 2026\] Robustness of Vision Foundation Models to Common Perturbations](../../CVPR2026/self_supervised/robustness_of_vision_foundation_models_to_common_perturbations.md)
- [\[ICCV 2025\] Improving Large Vision and Language Models by Learning from a Panel of Peers](improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](../../CVPR2026/self_supervised/com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](../../CVPR2026/self_supervised/talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)

</div>

<!-- RELATED:END -->
