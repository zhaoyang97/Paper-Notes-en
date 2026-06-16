---
title: >-
  [Paper Note] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection
description: >-
  [CVPR 2026][Object Detection][K-FPN] The NoOVD framework is proposed to discover potential novel category objects during frozen VLM-based OVD training by preserving CLIP knowledge with a parameter-free K-FPN. It embeds novel category knowledge into the detector via self-distillation and enhances the recall of novel categories during inference using R-RPN,
tags:
  - CVPR 2026
  - Object Detection
  - K-FPN
date: 2026-05-08
content_hash: 14ae7184507132fa
---
# NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection

**Conference**: CVPR 2026  
**arXiv**: [2603.21069](https://arxiv.org/abs/2603.21069)  
**Code**: None  
**Area**: Object Detection  
**Keywords**: Open-vocabulary object detection, Novel category discovery, Self-distillation, K-FPN, Frozen VLM

## TL;DR
The NoOVD framework is proposed to discover potential novel category objects during frozen VLM-based OVD training by preserving CLIP knowledge with a parameter-free K-FPN. It embeds novel category knowledge into the detector via self-distillation and enhances the recall of novel categories during inference using R-RPN, achieving SOTA results on OV-LVIS, OV-COCO, and Objects365.

## Background & Motivation

1. **Background**: Open-vocabulary object detection (OVD) aims to enable detectors to recognize novel categories not seen during training. Mainstream methods are built on frozen VLMs (e.g., CLIP), training only the detection modules (FPN, RPN, RoI head) to leverage the zero-shot transfer capabilities of the VLM for novel categories.
2. **Limitations of Prior Work**: A significant gap exists between training and testing. During training, only base-class annotations are available, and all unannotated novel category objects are forced to be treated as background. Consequently, novel proposals receive low scores and are filtered out during the RPN stage, while novel features are forced to align with background text embeddings in the RoI stage. These proposals are similarly removed during post-processing at test time, leading to a sharp decline in novel category recall.
3. **Key Challenge**: During training, there are no novel category annotations, forcing the model to learn novel categories as background; however, at test time, the model is required to recognize these categories. Existing solutions either rely on large-scale external data (high cost) or use pseudo-labels (introducing noise).
4. **Goal**: (1) Discover potential novel category objects without introducing extra data or pseudo-label noise; (2) Embed novel category knowledge into the detector; (3) Enhance the recall rate of novel categories during inference.
5. **Key Insight**: Leverage the zero-shot recognition capability of frozen CLIP itself to discover foreground objects. Use class-agnostic general foreground/background text descriptions instead of specific category names to distinguish foreground from background without knowing the names of the novel categories.
6. **Core Idea**: Utilize the zero-shot capability of frozen CLIP for class-agnostic foreground discovery, then inject novel category knowledge into the detector through self-distillation to prevent novel category features from being erroneously aligned with the background.

## Method

### Overall Architecture
NoOVD addresses the "train-test gap" in open-vocabulary detection. During training, the lack of novel annotations causes these objects to be suppressed as background. The framework, built on a two-stage detector with frozen CLIP, integrates CLIP's zero-shot foreground recognition throughout both training and inference. During training, a parameter-free K-FPN constructs a feature pyramid from multi-layer frozen CLIP features to prevent base-class training from polluting CLIP's original novel category knowledge. Class-agnostic "foreground/background" descriptions are used to retrieve potential novel proposals. These are then injected into the RoI head via self-distillation of CLIP region features. During inference, R-RPN reuses this discovery strategy to re-weight scores for novel proposals suppressed by the RPN, recovering them for the RoI head.

```mermaid
graph TD
    A["Input Image → Frozen CLIP Image Encoder"] --> B["K-FPN: Parameter-free Knowledge Preservation Pyramid<br/>Interpolation+Concat+Pooling, 5 layers of 512-dim features"]
    B --> C["RPN generates proposals"]
    subgraph DISC["Novel Category Discovery & Embedding"]
        direction TB
        D["Project to K-FPN for features<br/>Cosine similarity with FG/BG text"] --> E["Keep foreground, remove base GT boxes<br/>Obtain potential novel proposals"]
        E -->|Training| F["Crop original image for Frozen CLIP<br/>L2 Self-distillation into RoI head"]
    end
    C --> D
    E -->|Inference (Strategy Reuse)| G["R-RPN: Repropose Score Weighting<br/>α·RPN score + (1−α)·K-FPN FG score"]
    G --> H["Re-rank top-1000 → RoI head → OVD Output"]
```

### Key Designs

**1. K-FPN: Protecting CLIP Novel Knowledge with a Parameter-Free Pyramid**

Standard FPNs contain learnable parameters. Since training only involves base-class data, gradients pull CLIP features towards base classes, causing drift and loss of novel category representations. K-FPN eliminates all learnable parameters. Using CLIP ViT-B/16 as an example, features from layers [5, 7, 11] are taken as $\{P_2, P_3, P_4\}$. After top-down fusion, the frozen CLIPSelf dimensionality reduction head compresses the 768-dim features to 512-dim to align with text embeddings, resulting in $\{C_2, C_3, C_4\}$. High-resolution $\{F_2, F_3, F_4\}$ are formed by upsampling and concatenating, while $\{F_5, F_6\}$ are generated via max pooling. This path relies solely on interpolation, concatenation, and pooling, ensuring the original zero-shot discriminative power of CLIP remains intact.

**2. Novel Category Discovery and Embedding: Retrieving Novel Classes via "Foreground vs. Background"**

Since novel category names are unavailable during training, the strategy focuses on whether a region is "foreground." The authors generate 30 class-agnostic foreground descriptions (e.g., "This is an object, specifically a plant") and 30 background descriptions (e.g., "This is a background area") via ChatGPT. RoI Align is performed on K-FPN using RPN proposals to calculate cosine similarity with these embeddings. Proposals with foreground scores higher than background scores that do not overlap with base-class ground truths are identified as potential novel objects. These are cropped from the original image, passed through frozen CLIP, and the resulting features are used for L2 self-distillation with the RoI head output:

$$\mathcal{L}_{kd} = \|F_{proposals+}^{RoI} - F_{proposals+}^{Image}\|_2^2$$

This forces the detector to learn novel categories during training rather than treating them as background, without external data or noisy pseudo-labels.

**3. R-RPN: Recovering Suppressed Novel Proposals at Inference**

Even with knowledge injection, RPN scores remain biased towards base classes, causing novel proposals to be ranked low or filtered by NMS. R-RPN re-evaluates proposals after NMS using the same foreground discovery strategy. The final score is a weighted fusion of the K-FPN foreground score and the original RPN score:

$$S_{R-RPN} = \alpha \cdot S_{RPN} + (1-\alpha) \cdot S_{K-FPN}, \quad \alpha=0.5$$

Re-ranking by this fusion score ensures that novel boxes, previously at the bottom, are elevated to the top-1000 sent to the RoI head.

### Loss & Training
- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{cls-RPN} + \mathcal{L}_{reg-RPN} + \mathcal{L}_{reg-RoI} + \mathcal{L}_{cons} + \mathcal{L}_{kd}$
- $\mathcal{L}_{cons}$ is the RoI head contrastive loss; $\mathcal{L}_{kd}$ is the novel category self-distillation L2 loss (weight = 1).
- Frozen CLIP image/text encoder; training only FPN, RPN, and RoI head.
- OV-COCO trained for 5 epochs; OV-LVIS for 50 epochs.
- 16x NVIDIA 3090, batch size 10/GPU, AdamW lr=$10^{-4}$.

## Key Experimental Results

### Main Results - OV-LVIS

| Method | Backbone | AP_r (rare/novel) | AP (Total) |
|------|----------|---------------|---------|
| CLIPSelf + F-ViT | ViT-L/14 | 34.9 | 35.1 |
| DeCLIP + F-ViT | ViT-L/14 | 37.2 | 36.0 |
| **DeCLIP + NoOVD** | **ViT-L/14** | **39.2 (+2.0)** | **37.7 (+1.7)** |
| CLIPSelf + NoOVD | ViT-B/16 | 28.3 (+2.9) | 26.7 (+1.3) |
| YOLOE | YOLOv11-L | 29.1 | 35.2 |
| RO-ViT | ViT-H/16 | 34.1 | 35.1 |

### Main Results - OV-COCO

| Method | Backbone | AP_novel^50 | AP^50 |
|------|----------|-------------|-------|
| DeCLIP + F-ViT | ViT-L/14 | 46.2 | 60.3 |
| **DeCLIP + NoOVD** | **ViT-L/14** | **47.5 (+1.3)** | **61.0 (+0.7)** |
| CORA+ | RN50x4 | 43.1 | 56.2 |

### Ablation Study

| Configuration | AP_r | AP |
|------|------|-----|
| Baseline (F-ViT) | 25.4 | 25.4 |
| + CLIP-top (Simple Top Layer) | 26.4 | 26.1 |
| + K-FPN | 27.5 | 26.4 |
| + R-RPN | 26.7 | 25.9 |
| + K-FPN + R-RPN (Full) | **28.3** | **26.7** |

### Cross-Dataset Transfer (LVIS→Objects365)

| Method | Backbone | AP_r | AP50 |
|------|----------|------|------|
| CLIPSelf + F-ViT | ViT-L/14 | 21.7 | 39.2 |
| CLIPSelf + NoOVD | ViT-L/14 | **22.8 (+1.1)** | **40.2 (+1.0)** |

### Key Findings
- **K-FPN vs. Simple CLIP Top-layer Features**: The multi-scale feature pyramid of K-FPN provides a 1.1% higher gain in novel category detection than single-layer features, highlighting the importance of multi-scale discovery.
- **Complementarity**: K-FPN resolves knowledge preservation and injection during training, while R-RPN resolves recall during inference; the combination is optimal.
- **Dataset Stability**: OV-COCO annotations are incomplete; objects correctly detected by NoOVD during training might be counted as false positives during testing, potentially underestimating gains.
- **Fusion Weight**: A weight of $\alpha=0.3$ is optimal for K-FPN feature fusion; balancing high-level semantics and low-level details is crucial.

## Highlights & Insights
- **Novel Category Discovery without Extra Data or Pseudo-labels**: The use of general foreground/background descriptions and frozen CLIP's zero-shot capability to discover novel items without knowing category names is a concise and elegant strategy.
- **Fully Parameter-Free K-FPN**: Constructing a feature pyramid solely through interpolation, concatenation, and pooling prevents base-class training from corrupting novel category knowledge.
- **Consistent Discovery Strategy**: Reusing the discovery logic for both self-distillation during training and score re-weighting in R-RPN during inference ensures consistency.

## Limitations & Future Work
- Foreground/background descriptions are fixed (30+30) from ChatGPT; automated or adaptive prompt designs might improve coverage.
- Selection of novel proposals for distillation depends on a threshold, which might miss novel objects visually similar to the background (e.g., manhole covers on a road).
- The fixed value of $\alpha=0.5$ in R-RPN might not be optimal for all datasets or different base/novel ratios.
- The framework is currently based on two-stage detectors; integration with one-stage or DETR-based detectors remains unexplored.

## Related Work & Insights
- **vs. Detic**: Detic uses ImageNet image-level labels to expand categories, relying on massive data. NoOVD uses only CLIP's intrinsic capabilities.
- **vs. CLIPSelf/DeCLIP**: These optimize CLIP's region-level representation but still treat novel categories as background during training. NoOVD corrects the training flow itself.
- **vs. F-VLM**: While also based on frozen VLMs, F-VLM lacks an active mechanism to discover novel categories, whereas NoOVD actively mines and learns them via K-FPN and self-distillation.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach of category-agnostic discovery combined with parameter-free K-FPN is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive validation across OV-LVIS, OV-COCO, and Objects365 with multiple backbones and cross-dataset testing.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and detailed methodological explanation.
- Value: ⭐⭐⭐⭐ Provides a new OVD paradigm without requiring external data, offering significant reference for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SRA-Det: Learning Omni-Grained Open-Vocabulary Detection Beyond Category Names](sra-det_learning_omni-grained_open-vocabulary_detection_beyond_category_names.md)
- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [\[CVPR 2026\] WeDetect: Fast Open-Vocabulary Object Detection as Retrieval](wedetect_fast_open-vocabulary_object_detection_as_retrieval.md)
- [\[CVPR 2026\] Thermal-Det: Language-Guided Cross-Modal Distillation for Open-Vocabulary Thermal Object Detection](thermal-det_language-guided_cross-modal_distillation_for_open-vocabulary_thermal.md)
- [\[CVPR 2026\] Consistency Beyond Contrast: Enhancing Open-Vocabulary Object Detection Robustness via Contextual Consistency Learning](consistency_beyond_contrast_enhancing_open-vocabulary_object_detection_robustnes.md)

</div>

<!-- RELATED:END -->
