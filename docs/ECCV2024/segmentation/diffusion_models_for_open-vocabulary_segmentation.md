---
title: >-
  [Paper Note] Diffusion Models for Open-Vocabulary Segmentation
description: >-
  [ECCV 2024][Segmentation][Open-Vocabulary Segmentation] This paper proposes OVDiff, which leverages a pre-trained text-to-image diffusion model to generate support image sets for arbitrary textual categories. From these, multi-level prototypes (class-level, instance-level, and part-level) are extracted and combined with background prototypes to achieve training-free open-vocabulary semantic segmentation, outperforming prior methods by over 10% on PASCAL VOC.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Open-Vocabulary Segmentation"
  - "Diffusion Models"
  - "Prototype Learning"
  - "Training-Free Segmentation"
  - "Text-to-Image Generation"
date: 2026-05-08
content_hash: bab768d86e274c2d
---

# Diffusion Models for Open-Vocabulary Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2306.09316](https://arxiv.org/abs/2306.09316)  
**Code**: None  
**Area**: Segmentation / Open-Vocabulary Segmentation  
**Keywords**: Open-Vocabulary Segmentation, Diffusion Models, Prototype Learning, Training-Free Segmentation, Text-to-Image Generation

## TL;DR

This paper proposes OVDiff, which leverages a pre-trained text-to-image diffusion model to generate support image sets for arbitrary textual categories. From these, multi-level prototypes (class-level, instance-level, and part-level) are extracted and combined with background prototypes to achieve training-free open-vocabulary semantic segmentation, outperforming prior methods by over 10% on PASCAL VOC.

## Background & Motivation

**Background**: Open-vocabulary segmentation requires pixel-level segmentation of any object in an image described in natural language. Recently, large-scale vision-language models (such as CLIP) have driven significant progress in this field, but these methods require expensive contrastive training on massive image-text pairs.

**Limitations of Prior Work**: (1) Contrastive training-based methods (e.g., GroupViT, TCL) require extra training on massive image-text pairs, which introduces noise because the text might not fully describe the image content. (2) Cross-modal feature alignment is inherently ambiguous—images with similar visual appearances may correspond to different text descriptions, and vice versa. (3) Background segmentation remains a long-standing challenge, usually requiring manual similarity thresholds, but the optimal threshold is difficult to determine and varies across different images.

**Key Challenge**: Open-vocabulary segmentation requires mapping arbitrary linguistically-expressed categories to precise pixel-level segmentations, but directly learning cross-modal alignment is both expensive and prone to ambiguity. Can existing pre-trained foundation models be utilized to synthesize segmenters on-demand, thereby avoiding extra training?

**Key Insight**: The authors observe that feature comparison within the same modality is easier and more reliable than cross-modal comparison. Therefore, they propose using a generative diffusion model to "translate" language queries into visual prototypes, thereby transforming the cross-modal problem into an intra-modal comparison. Diffusion models not only encode the visual appearance distribution of objects but also provide contextual priors (e.g., background), which are crucial for segmentation quality.

**Core Idea**: Utilizing diffusion models to generate support images for textual categories, extracting multi-level visual prototypes, and achieving training-free open-vocabulary segmentation via intra-modal nearest-neighbor search.

## Method

### Overall Architecture

OVDiff consists of two phases: (1) Prototype sampling phase—given a text query, Stable Diffusion is used to generate a support image set, which is decomposed into positive (foreground) and negative (background) prototypes using a pre-trained feature extractor and an unsupervised segmenter; (2) Segmentation phase—features of the target image are extracted and compared via cosine similarity with the pre-computed prototypes to assign a category to each pixel. The entire workflow requires no training and relies solely on pre-trained components.

### Key Designs

1. **Multi-level Prototype Extraction**:

    - **Function**: Constructing rich visual prototype representations for arbitrary textual categories.
    - **Mechanism**: For each query $c_i$, Stable Diffusion is used to generate $N=64$ support images. Cross-attention maps from the diffusion process are used to obtain attribution maps, combined with unsupervised instance segmentation methods like CutLER to generate foreground/background masks. Features are extracted within the masked regions using a pre-trained feature extractor to construct three types of prototypes: class-level prototypes (weighted average of all instances), instance-level prototypes (one per support image), and part-level prototypes ($K$-means clustering to obtain $K=32$ cluster centers).
    - **Design Motivation**: A single prototype cannot capture intra-class variation. Class-level prototypes provide global representation, instance-level prototypes cover the appearance of diverse samples, and part-level prototypes focus on local object features (e.g., a dog's nose, neck). The combination of all three ensures high-quality segmentation.

2. **Background Prototypes**:

    - **Function**: Directly segmenting background regions to bypass the threshold-setting problem.
    - **Mechanism**: In addition to foreground prototypes, prototypes of background regions are also extracted from the support images. During segmentation, an auxiliary "background" class is introduced, whose foreground prototype is defined as the union of background prototypes from all categories: $\mathcal{P}_{c_{\text{bg}}}^{\text{fg}} = \bigcup_{c_i \in \mathcal{C}} \mathcal{P}_{c_i}^{\text{bg}}$. In this way, background regions can be directly identified through prototype matching without manual thresholding.
    - **Design Motivation**: Images generated by diffusion models naturally contain category-related background contexts (e.g., boat images typically contain water and sky), which provides valuable contextual priors. Ablation studies show that removing background prototypes leads to a 10% drop in mIoU on PASCAL VOC.

3. **Category Pre-filtering & Stuff Filtering**:

    - **Function**: Reducing false matches and enhancing segmentation accuracy.
    - **Mechanism**: CLIP is used for multi-label classification pre-filtering on the target image, retaining candidate categories and combining them into multi-label prompts to select the best-matching category set $\mathcal{C}' \subseteq \mathcal{C}$. For "stuff" classes (e.g., sky, water), prototypes that might conflict with the background prototypes of other classes are additionally filtered. The "thing/stuff" classification is automatically handled by ChatGPT.
    - **Design Motivation**: When there is a large number of categories, prototypes of different categories may trigger false correspondences. Pre-filtering limits the candidate categories to a reasonable range. Stuff filtering addresses the issue where the background of one category might act as the foreground of another.

### Loss & Training

OVDiff is an entirely training-free method. Segmentation is achieved via cosine similarity nearest-neighbor lookup: $M = \arg\max_{c \in \hat{\mathcal{C}}} \max_{P \in \mathcal{P}_c^{\text{fg}}} s(\Phi_v(I), P)$. Support images are generated using Stable Diffusion v1.5, with a classifier-free guidance scale of 8.0 and 30 denoising steps. The feature extractor utilizes an ensemble of SD + DINO + CLIP, calculating the average of the three cosine distances.

## Key Experimental Results

### Main Results

| Dataset | Metric | OVDiff | OVDiff+PAMR | Prev. SOTA (TCL) | Prev. SOTA+PAMR | Gain |
|--------|------|--------|------------|--------|--------|------|
| PASCAL VOC | mIoU | 67.1 | 69.0 | 51.2 | 55.0 | +14.0 |
| Pascal Context | mIoU | 30.1 | 31.4 | 24.3 | 30.4 | +1.0 |
| COCO-Object | mIoU | 34.8 | 36.3 | 30.4 | 31.6 | +4.7 |

### Ablation Study

| Configuration | VOC mIoU | Context mIoU | Description |
|------|---------|-------------|------|
| Full (SD only) | 63.6 | 29.8 | Full method |
| w/o Background Prototypes | 53.6 (-10.0) | 28.3 (-1.5) | Background prototypes contribute the most |
| w/o Category Pre-filtering | 54.9 (-8.7) | 26.4 (-2.4) | Pre-filtering is also critical |
| w/o Stuff Filtering | n/a | 27.3 (-2.5) | Important for datasets with many stuff classes |
| w/o CutLER | 60.6 (-3.0) | 27.9 (-1.9) | Unsupervised segmentation improves mask quality |
| Only Average Prototypes | 62.5 (-1.1) | 29.0 (-0.8) | Multi-level prototypes make positive contributions |

### Key Findings

- Background prototypes and category pre-filtering are the two largest contributors to performance improvements.
- Different feature extractors are complementary: SD (63.6), CLIP keys (63.2), and DINO (59.6); their integration reaches 67.0.
- Support set performance saturates at 64-128 images; more samples only yield redundant prototypes.
- Part-level prototypes can be traced back to specific visual regions in support images, offering interpretability.

## Highlights & Insights

- **Shifting Cross-Modal Alignment to Intra-Modal Comparison**: Using generative models as a language-to-vision bridge ingeniously avoids the difficulties of cross-modal training.
- **Contextual Priors of Background Prototypes**: Leveraging the contextual background naturally contained in generated images solves the long-standing thresholding challenge in background segmentation.
- **Entirely Training-Free**: All components are pre-trained; no extra training, annotation, or fine-tuning is required, drastically lowering deployment barriers.
- **Interpretability**: Segmentation results can be traced back to specific regions in the support set, offering interpretability lacked by traditional end-to-end methods.

## Limitations & Future Work

- Constrained by the resolution of the feature extractors, potentially missing tiny objects.
- Incapable of segmenting targets that the diffusion model fails to generate (e.g., crisp text).
- Sampling support images introduces computational overhead (approx. 210s per class), though this is amortized across the entire image set segmentation.
- Inherits biases and limitations of pre-trained components (e.g., potential biases in Stable Diffusion).

## Related Work & Insights

- **TCL / GroupViT / OVSegmentor**: Representative works of contrastive training-based open-vocabulary segmentation.
- **CutLER**: An unsupervised instance segmentation method that provides high-quality masks for prototype extraction.
- **ReCO**: Leverages CLIP to retrieve exemplar images from ImageNet for co-segmentation; this paper replaces database retrieval with a generative model.
- *Insight*: Generative models can act as "domain experts", mapping textual semantics into visual representations. In the future, performance will naturally scale with advancements in both generative models and feature extractors.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Very novel approach of utilizing diffusion models for training-free semantic segmentation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets, detailed ablation studies, comparisons across multiple feature extractors)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (The training-free paradigm is highly significant for lowering implementation barriers of segmentation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[ECCV 2024\] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](../../CVPR2025/segmentation/mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](../../ICCV2025/segmentation/floss_free_lunch_in_openvocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
