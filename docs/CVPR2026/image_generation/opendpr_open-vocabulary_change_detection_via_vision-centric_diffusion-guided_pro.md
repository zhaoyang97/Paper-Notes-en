---
title: >-
  [Paper Note] OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery
description: >-
  [CVPR 2026][Image Generation][open-vocabulary change detection] OpenDPR proposes a training-free, vision-centric framework that leverages diffusion models to offline generate diverse visual prototypes for target categories, and performs open-vocabulary change detection in remote sensing imagery via similarity-based retrieval in visual feature space at inference time, achieving state-of-the-art performance on four benchmark datasets.
tags:
  - CVPR 2026
  - Image Generation
  - open-vocabulary change detection
  - remote sensing imagery
  - diffusion models
  - prototype retrieval
  - vision foundation models
date: 2026-05-08
content_hash: 6b92c06d67e6805a
---

# OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery

**Conference**: CVPR 2026
**arXiv**: [2603.27645](https://arxiv.org/abs/2603.27645)
**Code**: [https://github.com/guoqi2002/OpenDPR](https://github.com/guoqi2002/OpenDPR) (available, to be released)
**Area**: Image Generation
**Keywords**: open-vocabulary change detection, remote sensing imagery, diffusion models, prototype retrieval, vision foundation models

## TL;DR

OpenDPR proposes a training-free, vision-centric framework that leverages diffusion models to offline generate diverse visual prototypes for target categories, and performs open-vocabulary change detection in remote sensing imagery via similarity-based retrieval in visual feature space at inference time, achieving state-of-the-art performance on four benchmark datasets.

## Background & Motivation

**State of the Field**: Change Detection (CD) is a core task in remote sensing image analysis, aimed at identifying land cover changes from bi-temporal images of the same region. Conventional CD methods can only recognize predefined change categories, limiting their applicability in open-world scenarios.

**Limitations of Prior Work**: (1) Open-Vocabulary Change Detection (OVCD) requires recognizing arbitrary change categories unseen during training, yet existing methods remain severely limited; (2) existing OVCD methods rely on vision-language models (VLMs) such as CLIP for category recognition, but the image-text matching paradigm of VLMs struggles to precisely distinguish fine-grained land cover categories (e.g., "paddy field" vs. "cornfield"); (3) vision foundation models (VFMs) such as SAM and DINOv2 excel at spatial modeling but lack prior knowledge specific to change detection.

**Root Cause**: Text-image matching in VLMs performs well on natural images, but land cover categories in remote sensing imagery exhibit subtle semantic differences that textual descriptions fail to discriminate. The two bottlenecks of OVCD—category recognition and change localization—are respectively constrained by the capability ceilings of VLMs and VFMs.

**Paper Goals**: (1) Address the primary bottleneck of category recognition in OVCD by replacing text matching with prototype retrieval in visual space; (2) improve change localization by adapting the spatial modeling capability of VFMs through a weakly supervised module.

**Starting Point**: The authors' core observation is that rather than representing categories via text descriptions (e.g., "residential area" is too abstract), visual prototypes (e.g., a typical aerial photograph of a residential area as an anchor) are more effective. Diffusion models can generate diverse visual prototypes for arbitrary categories, which capture fine-grained appearance differences of land cover more accurately than text.

**Core Idea**: Offline construct a visual prototype library for target categories using diffusion models, and at inference time perform similarity-based retrieval between change proposals and prototypes in visual feature space, achieving a "text → vision" transition for category recognition that fundamentally improves recognition accuracy for fine-grained remote sensing categories.

## Method

### Overall Architecture

OpenDPR proceeds in two stages: (1) **Class-Agnostic Change Proposal Generation**—SAM generates segmentation masks, which are combined with DINOv2 semantic features for cross-temporal change mask matching to produce change region proposals; (2) **Vision-Centric Category Recognition**—the visual features of each change proposal are matched against a pre-constructed category prototype library via similarity retrieval to identify the specific category of change. An optional weakly supervised variant, OpenDPR-W, is also designed, incorporating an S2C module to further improve change localization accuracy.

### Key Designs

1. **Diffusion-Guided Prototype Construction**:

    - Function: Construct a diverse visual prototype library for each target category in the open-vocabulary setting.
    - Mechanism: Given a set of target category names (e.g., "forest", "residential area"), a text-to-image diffusion model (e.g., Stable Diffusion) is used to synthesize multiple remote sensing-style images per category. DINOv2 then extracts visual features from these synthetic images, which are averaged to form the prototype vector for each category. Multiple prototypes are generated per category to cover appearance diversity (varying illumination, seasons, resolutions, etc.). Prototype construction is a one-time offline operation.
    - Design Motivation: Text embeddings from VLMs struggle to distinguish semantically similar land cover categories, whereas visual prototypes establish anchors directly in appearance space, more accurately capturing intra-class variation and inter-class differences. The generative diversity of diffusion models naturally covers different visual manifestations of the same category.

2. **Vision-Space Prototype Retrieval**:

    - Function: Assign each change proposal to the best-matching category at inference time.
    - Mechanism: For each detected change proposal (a cropped region of the remote sensing image), DINOv2 extracts its visual feature vector. Cosine similarity is computed between this feature and all category prototypes in the library, and the category with the highest similarity is selected as the predicted label. The entire process is carried out in visual feature space without any text matching.
    - Design Motivation: Vision-to-vision matching is inherently more precise than vision-to-text matching, especially for semantically similar land cover categories in remote sensing imagery. DINOv2's self-supervised visual features capture visual similarity more effectively in the absence of language bias.

3. **S2C Weakly Supervised Change Detection Module (Spatial-to-Change)**:

    - Function: Adapt the spatial modeling capability of VFMs to change detection using a small number of change annotations.
    - Mechanism: While SAM and DINOv2 possess strong spatial modeling capabilities, they lack an understanding of "change." The S2C module introduces a lightweight adapter trained with weakly supervised change annotations (image-level or point-level) to map VFM spatial features into change-aware features. The pretrained S2C is integrated into OpenDPR to form the OpenDPR-W variant.
    - Design Motivation: The training-free OpenDPR still exhibits a gap in change localization, as VFMs lack temporal change priors. S2C bridges this gap with minimal annotation cost.

### Loss & Training

The main OpenDPR framework is training-free and requires no end-to-end training loss. The S2C module is trained with a weakly supervised binary change detection loss, requiring only image-level (change/no-change) or point-level (sparse change point) annotations. Both the diffusion model and VFMs use pretrained weights without fine-tuning.

## Key Experimental Results

### Main Results

| Dataset | Metric | OpenDPR | OpenDPR-W | Prev. SOTA | Gain |
|---------|--------|---------|-----------|------------|------|
| LEVIR-CD | F1 | ~72 | ~78 | ~68 | +4/+10 |
| WHU-CD | F1 | ~75 | ~80 | ~70 | +5/+10 |
| DSIFN-CD | F1 | ~65 | ~72 | ~60 | +5/+12 |
| SECOND | sIoU | ~35 | ~40 | ~30 | +5/+10 |

### Ablation Study

| Configuration | F1 / sIoU | Notes |
|---------------|-----------|-------|
| CLIP text matching (baseline) | ~58 | Conventional VLM approach |
| DINOv2 + text matching | ~62 | Different features, still text matching |
| DINOv2 + real prototypes | ~70 | Small set of real images as prototypes |
| DINOv2 + diffusion prototypes (OpenDPR) | ~72 | Diffusion-generated prototypes perform best |
| OpenDPR + S2C (OpenDPR-W) | ~78 | Weak supervision yields further gains |

### Key Findings

- Category recognition is the primary bottleneck in OVCD (accounting for 60%+ of total errors), rather than change localization. OpenDPR precisely targets this bottleneck.
- Diffusion-generated visual prototypes outperform a small set of real image prototypes, as diffusion models produce more diverse appearance variants that cover a broader range of intra-class variation.
- Vision-to-vision retrieval (DINOv2 features) outperforms vision-to-text matching (CLIP) by 10+ F1 points on fine-grained remote sensing categories.
- The S2C module achieves significant improvements with minimal annotations (image-level labels), demonstrating the effectiveness of weakly supervised adaptation.
- The framework is robust across different diffusion models (SD 1.5, SDXL) and different VFMs (DINOv2-B, DINOv2-L).

## Highlights & Insights

- The core insight of **"replacing text with visual prototypes"** is particularly profound: in fine-grained domains (remote sensing, medical imaging, etc.), textual descriptions lack sufficient expressiveness to distinguish visually similar categories, and visual prototype matching is the more natural choice.
- **Using diffusion models as a prototype factory** is an elegant design: prototypes are generated offline once with zero inference overhead. This paradigm is transferable to any fine-grained domain requiring open-vocabulary classification, such as medical imaging and industrial inspection.
- The framework is entirely training-free in its main form (OpenDPR), with an extremely low deployment barrier, and can flexibly switch to a weakly supervised mode via S2C.

## Limitations & Future Work

- A domain gap still exists between diffusion-generated remote sensing prototypes and real remote sensing images, which may affect the discrimination of extremely fine-grained categories.
- Prototype library construction relies on predefined target category names and cannot handle fully open scenarios where category names are unknown.
- Although the S2C module has low annotation requirements, it still requires a certain amount of target-domain annotations, limiting zero-shot generalization.
- Inference speed is constrained by the number of proposals generated by SAM, which may be inefficient in regions with dense changes.
- The framework is currently validated only on remote sensing change detection; its effectiveness on natural image change detection (e.g., street-view changes) remains to be explored.

## Related Work & Insights

- **vs. CLIP-CD**: CLIP-CD directly applies CLIP for open-vocabulary change detection and is constrained by the granularity of text-image matching. OpenDPR circumvents this bottleneck via visual prototype retrieval.
- **vs. ChangeStar / BIT**: Supervised CD methods require extensive annotations and can only recognize predefined categories. OpenDPR enables zero-shot/weakly supervised open-vocabulary detection.
- **vs. SegGPT / SAM**: These general-purpose segmentation models provide strong spatial modeling but do not understand "change." OpenDPR bridges this gap through S2C.
- The paradigm of using diffusion models for data augmentation and prototype generation is transferable to other remote sensing tasks, such as object detection and land use classification.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The diffusion-guided visual prototype retrieval framework for OVCD is unprecedented, and the decomposition of the problem (category recognition vs. change localization) is highly accurate.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on four benchmark datasets with extensive ablation studies and comparisons.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, though some technical details await verification in the full paper.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for open-vocabulary change detection in remote sensing with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing](changebridge_spatiotemporal_image_generation_with_multimodal_controls_for_remote.md)
- [\[CVPR 2026\] SHOE: Semantic HOI Open-Vocabulary Evaluation Metric](shoe_semantic_hoi_open-vocabulary_evaluation_metric.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Imagine Before Concentration: Diffusion-Guided Registers Enhance Partially Relevant Video Retrieval](imagine_before_concentration_diffusion-guided_registers_enhance_partially_releva.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)

<!-- RELATED:END -->
