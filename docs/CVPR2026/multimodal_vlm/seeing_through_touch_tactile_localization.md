---
title: >-
  [Paper Note] Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions
description: >-
  [CVPR 2026][Multimodal VLM][tactile localization] This paper introduces the tactile localization task—identifying regions in an image that share the same material properties as a given tactile input—and addresses it via local visual-tactile alignment and a material-diversity pairing strategy for learning dense cross-modal features. Two new tactile-material segmentation datasets are also constructed.
tags:
  - CVPR 2026
  - Multimodal VLM
  - tactile localization
  - visual-tactile alignment
  - material segmentation
  - cross-modal learning
  - dataset
date: 2026-05-08
content_hash: 2dc9223a91e19578
---

# Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions

**Conference**: CVPR 2026
**arXiv**: [2604.11579](https://arxiv.org/abs/2604.11579)
**Code**: [https://mm.kaist.ac.kr/projects/SeeingThroughTouch/](https://mm.kaist.ac.kr/projects/SeeingThroughTouch/)
**Area**: Multimodal VLM
**Keywords**: tactile localization, visual-tactile alignment, material segmentation, cross-modal learning, dataset

## TL;DR
This paper introduces the tactile localization task—identifying regions in an image that share the same material properties as a given tactile input—and addresses it via local visual-tactile alignment and a material-diversity pairing strategy for learning dense cross-modal features. Two new tactile-material segmentation datasets are also constructed.

## Background & Motivation

**Background**: Visual-tactile learning has primarily focused on global alignment (determining whether an image and a tactile signal correspond to the same material), lacking spatial localization capability—i.e., the ability to identify regions in a visual scene that "feel the same."

**Limitations of Prior Work**: (1) Global alignment methods cannot localize material regions; (2) existing datasets predominantly consist of close-up captures where visual frames exhibit minimal variation and a single material fills the entire frame, with no scene-level multi-material images; (3) no evaluation benchmark exists for tactile-material segmentation.

**Key Challenge**: Tactile localization requires fine-grained local cross-modal correspondence, whereas existing methods and data only provide coarse-grained global alignment.

**Core Idea**: Learn local visual-tactile alignment to produce tactile saliency maps, and expand effective training pairs through material-diversity pairing.

## Method

### Overall Architecture
A tactile encoder extracts tactile features (global pooling) and a visual encoder extracts spatial feature maps → a dense similarity map is computed as $M[h,w] = \bar{f}_t \cdot f_v[h,w]$ → max-pooling yields a similarity score for contrastive learning → at inference, the similarity map is directly used for tactile localization.

### Key Designs

1. **Local Visual-Tactile Alignment**:

    - Function: Learn spatially resolved cross-modal features.
    - Mechanism: The tactile feature is globally pooled into a 1D vector, which is dot-producted with each spatial position of the visual feature map to produce a similarity map; max-pooling is then applied for contrastive learning. DINOv2 serves as the shared encoder backbone, with the visual backbone frozen and only the aligners trained.
    - Design Motivation: Max-pooling directs the model to attend to the best-matching region in the image rather than averaging responses over all regions, making it naturally suited for localization.

2. **Material-Diversity Pairing Strategy**:

    - Function: Expand effective training pairs and enhance cross-instance generalization.
    - Mechanism: *In-domain pairing*—different tactile instances and different visual frames from the same material category can be combined across instances as positive pairs; *out-of-domain pairing*—scene-level web images are collected and matched to tactile samples based on material category, exploiting the assumption that "similar materials produce similar tactile signals."
    - Design Motivation: In Touch-and-Go, visual frames from the same instance are nearly identical, resulting in very few effective training pairs; cross-instance and cross-domain pairing substantially increases diversity.

3. **In-the-Wild Image Collection and Filtering**:

    - Function: Supplement scene-level multi-material images.
    - Mechanism: An LLM generates diverse search phrases for each material category (e.g., "brick chimney in a cozy living room"); images are collected from search engines, filtered for misclassified samples using CLIP similarity, and supplemented with images from the MINC material dataset.
    - Design Motivation: Images in the TG dataset are too close-range and single-material to support training for scene-level localization.

### Loss & Training
Symmetric contrastive learning loss (InfoNCE); the visual backbone is frozen while the tactile encoder and two aligner modules are trained.

## Key Experimental Results

### Main Results

| Dataset | Metric | STT | Prev. SOTA | Gain |
|--------|------|-----|---------|------|
| TG-Seg (new) | mIoU | Significantly better | ImageBind | Large |
| Web-Mat-Seg (new) | mIoU | Significantly better | UniTouch | Large |
| OpenSurfaces | F1 | Better | Baseline | Improved |

### Ablation Study

| Configuration | mIoU | Notes |
|------|------|------|
| Full (in-domain + out-of-domain) | Best | Full model |
| In-domain pairing only | Second | Lacks scene-level generalization |
| Standard pairing only | Poor | Too few effective training pairs |
| Global alignment substitute | Poor | No spatial localization capability |

### Key Findings
- Local alignment substantially outperforms global alignment, confirming that spatially resolved cross-modal features are critical for localization.
- Material-diversity pairing—especially out-of-domain images—is the key factor for generalizing to scene-level images.
- The model's ability to handle weak tactile signals (e.g., light touch or ambiguous material) improves significantly with the addition of out-of-domain data.

## Highlights & Insights
- **New Task Definition**: Tactile localization is a natural yet formally unstudied problem that can inspire broader research into sensory interaction.
- **Exploiting "Similar Materials, Similar Touch"**: This simple assumption substantially expands training data and represents a general strategy for addressing data scarcity in cross-modal learning.

## Limitations & Future Work
- The tactile sensor type is fixed (GelSight); cross-sensor generalization has not been verified.
- Material category granularity is limited (18 classes).
- Future work could extend to finer-grained material attributes (e.g., roughness, hardness).

## Related Work & Insights
- **vs. ImageBind/UniTouch**: Global alignment methods with no localization capability.
- **vs. TaRF**: Performs tactile localization in 3D NeRF, restricted to reconstructed scenes.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel task definition with a practical data strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two new datasets constructed for evaluation.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly described.
- Value: ⭐⭐⭐⭐ Opens a new direction for tactile localization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios](../../AAAI2026/multimodal_vlm/stola_self-adaptive_touch-language_framework_with_tactile_commonsense_reasoning_.md)
- [\[CVPR 2026\] EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)
- [\[CVPR 2026\] Rethinking VLMs for Image Forgery Detection and Localization](rethinking_vlms_for_image_forgery_detection_and_localization.md)
- [\[CVPR 2026\] PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding](focus_dont_prune_identifying_instruction-relevant_regions_for_information-rich_i.md)
- [\[CVPR 2026\] VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models](vlm-loc_localization_in_point_cloud_maps_via_vision-language_models.md)

</div>

<!-- RELATED:END -->
