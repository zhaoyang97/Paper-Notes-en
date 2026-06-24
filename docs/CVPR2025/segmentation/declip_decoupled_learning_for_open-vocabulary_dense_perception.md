---
title: >-
  [Paper Note] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception
description: >-
  [CVPR 2025][Segmentation][Open-Vocabulary Segmentation] DeCLIP identifies the "proxy token" phenomenon in CLIP's self-attention, which prevents image tokens from aggregating spatial correlation information. It proposes a framework that decouples the self-attention module into "content" and "context" features, optimizing them respectively through CLIP self-distillation and Vision Foundation Model (VFM) distillation. It out-performs existing methods across open-vocabulary objec…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Open-Vocabulary Segmentation"
  - "CLIP"
  - "Decoupled Attention"
  - "Knowledge Distillation"
  - "Dense Prediction"
date: 2026-05-08
content_hash: e4caeb325da5c495
---

# DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception

**Conference**: CVPR 2025  
**arXiv**: [2505.04410](https://arxiv.org/abs/2505.04410)  
**Code**: [https://github.com/xiaomoguhz/DeCLIP](https://github.com/xiaomoguhz/DeCLIP)  
**Area**: Image Segmentation  
**Keywords**: Open-Vocabulary Segmentation, CLIP, Decoupled Attention, Knowledge Distillation, Dense Prediction

## TL;DR
DeCLIP identifies the "proxy token" phenomenon in CLIP's self-attention, which prevents image tokens from aggregating spatial correlation information. It proposes a framework that decouples the self-attention module into "content" and "context" features, optimizing them respectively through CLIP self-distillation and Vision Foundation Model (VFM) distillation. It out-performs existing methods across open-vocabulary object detection and semantic segmentation.

## Background & Motivation
Traditional dense prediction methods (object detection, image segmentation) rely on predefined categories and cannot handle unbounded visual concepts. The emergence of vision-language models like CLIP enables open-vocabulary dense prediction, but directly applying CLIP to dense prediction suffers from domain shift.

**Why is CLIP not suitable for dense prediction?** Through analyzing the attention maps of different CLIP layers, the authors identified a critical issue — the **"proxy token" phenomenon**:
- At shallow layers, the attention of the CLS token is widely distributed across the entire image.
- At deep layers (after the 9th layer), the CLS token no longer focuses on the main objects, but is highly attentive to specific tokens in the background.
- More severely, **image tokens exhibit similar behavior to the CLS token** — regardless of their own positions, they highly attend to those "proxy" tokens.
- These "proxy" tokens act as information hubs for the CLS token; while helpful for image-level classification, they **disrupt the spatial and semantic relationships among image tokens**.

**Key Challenge**: Dense prediction requires image tokens to possess both **local discriminativeness** (ability to distinguish the semantics of different regions) and **spatial consistency** (association among tokens of the same semantic region). However, CLIP's "proxy token" phenomenon impairs both capacities. Concurrently applying self-distillation and VFM distillation on the same feature results in optimization conflict (causing a drop of 3.9 mAcc in regional classification).

**Core Idea**: **Decouple** the Q (context feature) and the output (content feature) in the last self-attention block of CLIP, and distill them separately using different teacher models to avoid optimization conflict.

## Method

### Overall Architecture
DeCLIP is an unsupervised pre-finetuning method that takes the original image and cropped sub-images as input. The self-attention of the last CLIP layer is re-interpreted: the Q projection output serves as the "context" feature, responsible for spatial consistency; the attention-weighted output serves as the "content" feature, responsible for local discriminativeness. CLIP itself (a frozen copy) acts as the teacher for self-distillation of the content feature, and a vision foundation model (such as DINOv2) acts as the teacher for the context feature.

### Key Designs
1. **Decoupled Attention**:

    - Extract two types of features from the self-attention of the last CLIP layer:
    - **Context feature** $\mathbf{X}_{context}$: directly taken as the Q projection output $\text{Proj}_q(\mathbf{X})$
    - **Content feature** $\mathbf{X}_{content}$: calculated by performing weight summation on V using the self-attention of the context feature ($\text{Attn}_{context} = \text{SoftMax}(\mathbf{X}_{context} \mathbf{X}_{context}^T / \sqrt{d})$), followed by projection.
    - Inspiration: Previous training-free OVS methods (such as SCLIP, ClearCLIP) change $\text{Attn}_{qk}$ to $\text{Attn}_{qq}$ and remove residual connections to improve segmentation. DeCLIP generalizes this insight into a trainable decoupled distillation framework.
    - Through decoupling, different constraints can be applied to both features without interference.

2. **Content Feature Distillation**:

    - Teacher model: frozen copy of CLIP (self-distillation).
    - Crop the input image into $k$ sub-regions as sub-images.
    - Student: extracts region features $\mathbf{f}_i^s$ from the content feature map using RoI Align.
    - Teacher: passes the cropped sub-images into the frozen CLIP to obtain the corresponding CLS tokens $\mathbf{f}_i^t$.
    - Loss: cosine similarity alignment $\mathcal{L}_{content} = \frac{1}{k} \sum_{i=1}^k (1 - \cos(\mathbf{f}_i^t, \mathbf{f}_i^s))$.
    - Intuition: Classifying cropped images using the CLS token of CLIP yields higher accuracy than using region features; hence, alignment enhances the discriminativeness of region features.

3. **Context Feature Distillation**:

    - Teacher model: vision foundation model (DINOv2 performs best).
    - Compute the cosine similarity matrix (correlation volume) between tokens for both the VFM and CLIP's context features.
    - Align the correlation volumes between the two using the L2 loss: $\mathcal{L}_{context} = \frac{1}{HW} \sum_i \sum_j \|r_{ij}^{VFM} - r_{ij}^{CLIP}\|_2$.
    - VFMs do not suffer from the "proxy token" phenomenon and exhibit superior correlation between semantically related tokens.
    - By distilling the correlation of VFM, the spatial consistency of CLIP tokens is improved.

### Loss & Training
- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{content} + \lambda \mathcal{L}_{context}$
- Unsupervised pre-finetuning: does not require annotated data.
- Once finetuning is completed, the enhanced CLIP can be plug-and-play integrated into various downstream dense prediction frameworks.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | DeCLIP | Baseline | Gain |
|--------|------|------|----------|------|
| OV-COCO Detection (F-ViT, ViT-B) | AP50 Novel | 41.1 | 37.6 (CLIPSelf) | +3.5 |
| OV-COCO Detection (OV-DQUO, ViT-B) | AP50 Novel | 46.1 | 39.2 (OV-DQUO) | +6.9 |
| OV-COCO Detection (OV-DQUO, ViT-L) | AP50 Novel | 48.3 | 45.6 (OV-DQUO) | +2.7 |
| OV-LVIS Detection (OV-DQUO, ViT-L) | mAP rare | 41.5 | 39.3 (OV-DQUO) | +2.2 |
| OV Semantic Segmentation (CAT-Seg+DeCLIP, ViT-B) | ADE150 mIoU | 36.3 | 31.8 (CAT-Seg) | +4.5 |
| OV Semantic Segmentation (CAT-Seg+DeCLIP, ViT-L) | ADE150 mIoU | 40.7 | 37.9 (CAT-Seg) | +2.8 |
| VLM Feature Segmentation (8 benchmarks) | Avg mIoU | 41.9 | 38.2 (SCLIP) | +3.7 |

### Ablation Study

| Configuration | Region Classification mAcc (Thing/Stuff) | Semantic Segmentation mIoU (Context59/CityScape) |
|------|---------|------|
| Self-distillation only (CLIPSelf) | 69.5 / 44.6 | 29.4 / 25.6 |
| Self-distillation + VFM distillation (without decoupling) | 65.6 / 41.3 (-3.9/-3.3) | 32.4 / 28.7 (+3.0/+3.1) |
| Self-distillation + VFM + decoupling (DeCLIP) | 75.0 / 51.8 (+5.5/+7.2) | 35.3 / 32.3 (+5.9/+6.7) |

| VFM Selection | Region Classification (Thing) | Context59 | ADE | Characteristics |
|----------|-----------------|-----------|-----|------|
| DINO ViT-B/16 | 67.6 | 38.1 | 20.4 | Moderate segmentation, weak classification |
| SAM ViT-B/16 | 75.0 | 35.3 | 18.5 | Strong classification, weak segmentation |
| DINOv2 ViT-B/14 | 77.2 | 39.2 | 21.9 | Best of both worlds |

### Key Findings
- **Direct distillation without decoupling causes optimization conflicts**: region classification performance drops by 3.9 mAcc, which is the core motivation for proposing the decoupling framework.
- Both capabilities improve significantly after decoupling: region classification by +5.5 and semantic segmentation by +5.9, verifying that decoupling effectively avoids conflicts.
- DINOv2 performs best as the VFM teacher: SAM lacks semantic correlation capabilities, whereas DINO focuses indiscriminatingly on all major objects.
- At the ViT-B scale, CAT-Seg+DeCLIP nearly outperforms all existing methods that employ larger encoders (such as ConvNeXt-L).

## Highlights & Insights
1. **Discovery of the "proxy token" phenomenon**: Conducts an in-depth analysis of why CLIP fails at dense prediction, revealing that it is not simply due to domain shift but rather specific tokens acting as information hubs in the attention mechanism, which disrupts the spatial correlation among image tokens.
2. **Decoupled distillation to avoid optimization conflict**: Without decoupling, self-distillation and VFM distillation interfere with each other; with decoupling, the two objectives can be independently optimized, an insight possessing significant methodological value.
3. **Unsupervised pre-finetuning paradigm**: DeCLIP requires no labeled data, and once finetuned, it can be directly applied to any downstream detection/segmentation framework, demonstrating high versatility.
4. **Analysis of VFM selection**: Uncovers the differences in spatial consistency guidance among various VFMs (DINO vs. SAM vs. DINOv2), identifying DINOv2 as the optimal choice that balances both semantic and spatial aspects.

## Limitations & Future Work
- Currently, only the self-attention of the last layer is decoupled. Whether decoupling more layers to obtain multi-scale information is worth exploring.
- Context distillation utilizes a simple L2 loss to align correlation volumes, and more sophisticated distillation strategies could yielded further improvements.
- The VFM teacher is fixed. Whether one can dynamically adjust distillation weights or employ an ensemble of multiple VFMs remains an open question.
- Content distillation relies on uniform grid cropping, which might lack flexibility for objects with non-grid distributions.
- The additional VFM forward pass introduces computational overhead, though this is only incurred during the pre-finetuning phase.

## Related Work & Insights
- CLIPSelf proposed the idea of self-distillation to enhance region classification. DeCLIP builds upon this by incorporating VFM guidance and resolving the optimization conflict.
- Training-free methods such as ClearCLIP and SCLIP improve segmentation results by modifying the attention mechanism, which inspired the Q-Q attention design of DeCLIP.
- CLIM uses mosaic patchworks as pseudo-regions, whereas DeCLIP uses actual crops which are more accurate.
- Insights for open-vocabulary segmentation: The representation quality of pretrained models represents the ceiling; decoupling and enhancement at the feature level is more effective than elaborate designs in downstream frameworks.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery of the "proxy token" phenomenon is insightful, and the decoupled distillation design is elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Detection (OV-COCO, OV-LVIS) + Segmentation (6 OVS benchmarks) + VLM feature segmentation (8 benchmarks) + Region classification + Cross-dataset transfer.
- Writing Quality: ⭐⭐⭐⭐⭐ Deep motivation analysis, intuitive figure presentations, and a clear logical progression from discovering "proxy tokens" to identifying optimization conflicts and proposing the decoupling solution.
- Value: ⭐⭐⭐⭐⭐ Highly practical in open-vocabulary dense prediction tasks, offering a versatile framework that can be seamlessly incorporated into various downstream methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Frequency Dynamic Convolution for Dense Image Prediction](frequency_dynamic_convolution_for_dense_image_prediction.md)

</div>

<!-- RELATED:END -->
