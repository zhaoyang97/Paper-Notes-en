---
title: >-
  [Paper Note] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection
description: >-
  [CVPR 2026][Object Detection][Open-Vocabulary Object Detection] NoOVD proposes a framework that, during frozen-VLM-based OVD training, employs a parameter-free K-FPN to preserve CLIP knowledge for discovering potential n…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Open-Vocabulary Object Detection"
  - "Novel Category Discovery"
  - "Self-Distillation"
  - "K-FPN"
  - "Frozen VLM"
date: 2026-05-08
content_hash: fc2273a8b598bda5
---

# NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.21069](https://arxiv.org/abs/2603.21069)
**Code**: N/A
**Area**: Object Detection
**Keywords**: Open-Vocabulary Object Detection, Novel Category Discovery, Self-Distillation, K-FPN, Frozen VLM

## TL;DR
NoOVD proposes a framework that, during frozen-VLM-based OVD training, employs a parameter-free K-FPN to preserve CLIP knowledge for discovering potential novel-category objects, applies self-distillation to embed novel-category knowledge into the detector, and introduces R-RPN at inference to improve novel-category recall, achieving SOTA on OV-LVIS, OV-COCO, and Objects365.

## Background & Motivation

1. **Background**: Open-vocabulary object detection (OVD) aims to enable detectors to recognize novel categories unseen during training. Dominant approaches build on frozen VLMs (e.g., CLIP), training only the detection modules (FPN, RPN, RoI head) and leveraging zero-shot transfer for novel category recognition.
2. **Limitations of Prior Work**: A significant train-test gap exists — during training, only base-category annotations are available, forcing all unannotated novel-category objects to be treated as background. Novel-category proposals receive low RPN scores and are filtered out; at the RoI stage, their features are forced to align with background text embeddings. At test time, these proposals similarly score low and are removed during post-processing, substantially reducing novel-category recall.
3. **Key Challenge**: Without novel-category annotations during training, the model is compelled to learn novel categories as background; yet at test time it is expected to recognize them. Existing remedies either rely on large-scale additional data (high cost) or use pseudo-labels (introducing noise).
4. **Goal**: (1) Discover potential novel-category objects without extra data or pseudo-label noise; (2) embed novel-category knowledge into the detector; (3) improve novel-category recall at inference.
5. **Key Insight**: Exploit the zero-shot recognition capability of the frozen CLIP itself to discover foreground objects, using category-agnostic generic foreground/background text descriptions instead of specific class names — enabling foreground–background discrimination without knowing novel category names.
6. **Core Idea**: Use frozen CLIP's zero-shot capability for category-agnostic foreground discovery, then inject novel-category knowledge into the detector via self-distillation, preventing novel-category features from being incorrectly aligned to background.

## Method

### Overall Architecture
A two-stage detection framework built on a frozen CLIP. During training: (1) K-FPN constructs a parameter-free feature pyramid from frozen CLIP multi-layer features to retain CLIP's novel-category knowledge; (2) category-agnostic foreground/background text descriptions combined with CLIP zero-shot capability are used to discover potential novel-category proposals; (3) self-distillation aligns CLIP features with RoI features for discovered novel-category proposals. At inference: (4) R-RPN applies the same discovery strategy to boost novel-category proposal confidence scores.

### Key Designs

1. **K-FPN (Knowledge-retentive FPN)**:

    - **Function**: Construct a feature pyramid from frozen CLIP that retains its original knowledge.
    - **Mechanism**: Using CLIP ViT-B/16 as an example, features from layers [5, 7, 11] are extracted as {P2, P3, P4} and fused top-down in FPN style. A frozen CLIPSelf projection head reduces the dimensionality from 768 to 512 (aligned with text embedding dimension) to obtain {C2, C3, C4}. The high-level C4 is upsampled and concatenated with C3 and C2 to produce high-resolution feature maps {F2, F3, F4}; two max-pooling operations on C4 yield {F5, F6}. The entire process contains no learnable parameters.
    - **Design Motivation**: Standard FPN has learnable parameters; training exclusively on base-category data causes CLIP features to drift after passing through FPN, losing novel-category knowledge. K-FPN employs entirely parameter-free operations to maximally preserve CLIP's original representational capability, providing a reliable feature basis for subsequent novel-category discovery.

2. **Novel Category Discovery and Embedding**:

    - **Function**: Discover potential novel-category objects during training and inject the corresponding knowledge into the detector.
    - **Mechanism**: (1) ChatGPT-o1 is used to generate 30 category-agnostic foreground descriptions (e.g., "This is an object, specifically a plant") and 30 background descriptions (e.g., "This is a background area"), from which frozen CLIP text embeddings are extracted. (2) RPN proposals are mapped onto K-FPN for RoI Align feature extraction; cosine similarities with foreground/background text embeddings are computed; proposals with higher foreground than background scores are retained, and those with high IoU overlap with GT bounding boxes are excluded, leaving potential novel-category objects. (3) These proposals are cropped from the original image and fed into frozen CLIP for feature extraction, then self-distillation is applied via an L2 loss between the CLIP features and the RoI head output features: $\mathcal{L}_{kd} = \|F_{proposals+}^{RoI} - F_{proposals+}^{Image}\|_2^2$.
    - **Design Motivation**: Novel-category objects can be discovered using generic "foreground vs. background" descriptions without relying on specific class names. Self-distillation enables the detector to learn novel-category knowledge during training rather than treating them as background. No additional data or pseudo image-text pairs are required, eliminating pseudo-label noise at the source.

3. **R-RPN (Re-weighted RPN)**:

    - **Function**: Improve novel-category proposal recall at inference.
    - **Mechanism**: Prior to RPN post-processing, the same foreground discovery strategy used during training is applied to classify NMS-filtered proposals as foreground. The K-FPN foreground score is then fused with the original RPN score via weighted combination: $S_{R-RPN} = \alpha \cdot S_{RPN} + (1-\alpha) \cdot S_{K-FPN}$, with $\alpha=0.5$. Proposals are re-ranked by the fused score, and the top-1000 are forwarded to the RoI head.
    - **Design Motivation**: Novel-category proposals receive low RPN scores (having been treated as background during training) and are filtered out during post-processing — a primary cause of novel-category detection failure. R-RPN rescues these overlooked proposals by injecting CLIP's foreground knowledge.

### Loss & Training
- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{cls-RPN} + \mathcal{L}_{reg-RPN} + \mathcal{L}_{reg-RoI} + \mathcal{L}_{cons} + \mathcal{L}_{kd}$
- $\mathcal{L}_{cons}$ is the contrastive loss for the RoI head; $\mathcal{L}_{kd}$ is the novel-category self-distillation L2 loss (weight = 1).
- The CLIP image/text encoder is frozen; only FPN, RPN, and RoI head are trained.
- Training: 5 epochs on OV-COCO, 50 epochs on OV-LVIS.
- Hardware: 16 NVIDIA 3090 GPUs, batch size 10/GPU, AdamW lr = $10^{-4}$.

## Key Experimental Results

### Main Results — OV-LVIS

| Method | Backbone | AP_r (rare/novel) | AP (overall) |
|--------|----------|-------------------|--------------|
| CLIPSelf + F-ViT | ViT-L/14 | 34.9 | 35.1 |
| DeCLIP + F-ViT | ViT-L/14 | 37.2 | 36.0 |
| **DeCLIP + NoOVD** | **ViT-L/14** | **39.2 (+2.0)** | **37.7 (+1.7)** |
| CLIPSelf + NoOVD | ViT-B/16 | 28.3 (+2.9) | 26.7 (+1.3) |
| YOLOE | YOLOv11-L | 29.1 | 35.2 |
| RO-ViT | ViT-H/16 | 34.1 | 35.1 |

### Main Results — OV-COCO

| Method | Backbone | AP_novel^50 | AP^50 |
|--------|----------|-------------|-------|
| DeCLIP + F-ViT | ViT-L/14 | 46.2 | 60.3 |
| **DeCLIP + NoOVD** | **ViT-L/14** | **47.5 (+1.3)** | **61.0 (+0.7)** |
| CORA+ | RN50x4 | 43.1 | 56.2 |

### Ablation Study

| Configuration | AP_r | AP |
|---------------|------|----|
| Baseline (F-ViT) | 25.4 | 25.4 |
| + CLIP-top (simple top-layer features) | 26.4 | 26.1 |
| + K-FPN | 27.5 | 26.4 |
| + R-RPN | 26.7 | 25.9 |
| + K-FPN + R-RPN (full) | **28.3** | **26.7** |

### Cross-Dataset Transfer (LVIS → Objects365)

| Method | Backbone | AP_r | AP50 |
|--------|----------|------|------|
| CLIPSelf + F-ViT | ViT-L/14 | 21.7 | 39.2 |
| CLIPSelf + NoOVD | ViT-L/14 | **22.8 (+1.1)** | **40.2 (+1.0)** |

### Key Findings
- **K-FPN vs. simple CLIP top-layer features**: K-FPN's multi-scale feature pyramid outperforms single-layer features by 1.1% on novel category detection, demonstrating that multi-scale representations are crucial for discovering novel objects of varying sizes.
- **K-FPN and R-RPN are complementary**: K-FPN addresses knowledge retention and injection during training, while R-RPN addresses recall at inference; their combination yields the best performance.
- **OV-LVIS gains are more consistent than OV-COCO**: OV-COCO annotations are incomplete; novel-category objects correctly detected by NoOVD during training are counted as false positives at test time, causing the gains to be underestimated.
- **Fusion weight W = 0.3 is optimal**: Balancing high-level semantics and low-level details in K-FPN feature fusion is important; deviations in either direction degrade performance.

## Highlights & Insights
- **Novel category discovery without extra data or pseudo-labels**: Generic foreground/background text descriptions combined with frozen CLIP's zero-shot capability enable novel category discovery without knowing specific class names or constructing image-text pseudo-pairs. This "category-agnostic yet foreground-aware" strategy is both elegant and concise.
- **Fully parameter-free K-FPN design**: A feature pyramid is constructed from frozen CLIP features using only interpolation, concatenation, and pooling, maximally preventing base-category training from corrupting novel-category knowledge. Despite its simplicity, the design directly addresses the root cause.
- **Consistent discovery strategy across training and inference**: Foreground discovery drives self-distillation during training and re-weights R-RPN scores at inference, yielding a logically coherent framework.

## Limitations & Future Work
- The foreground/background text descriptions are a fixed set of 30+30 sentences generated by ChatGPT, which may fail to cover all scene semantics; automated or adaptive prompt design could be more effective.
- Novel-category proposal selection for self-distillation relies on thresholds, potentially missing novel objects that are visually similar to background (e.g., manhole covers on roads).
- The fixed $\alpha=0.5$ in R-RPN may not be optimal across different datasets or varying novel-to-base category ratios.
- The current framework is a two-stage detector; integration with one-stage or DETR-based detectors remains unexplored.

## Related Work & Insights
- **vs. Detic**: Detic extends category coverage using ImageNet image-level labels, requiring large-scale additional data; NoOVD requires no extra data and relies solely on CLIP's inherent capability.
- **vs. CLIPSelf/DeCLIP**: These methods improve CLIP's region-level representations but still treat novel categories as background during training; NoOVD corrects this at the level of the training procedure itself.
- **vs. F-VLM**: F-VLM similarly builds on a frozen VLM but lacks a mechanism for actively discovering novel categories; NoOVD proactively mines and learns novel-category knowledge via K-FPN and self-distillation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Category-agnostic novel category discovery and parameter-free K-FPN design are conceptually novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on OV-LVIS, OV-COCO, and Objects365 with multiple backbones, cross-dataset transfer, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, method description is thorough, and figures are informative.
- **Value**: ⭐⭐⭐⭐ Offers a new OVD paradigm requiring no additional data, providing a valuable reference for the open-vocabulary detection community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [\[CVPR 2026\] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)
- [\[NeurIPS 2025\] CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](../../NeurIPS2025/object_detection/cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)
- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](detecting_unknown_objects_via_energy-based_separation.md)
- [\[CVPR 2026\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching.md)

</div>

<!-- RELATED:END -->
