---
title: >-
  [Paper Note] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection
description: >-
  [ICCV 2025][Multimodal VLM][OOD Detection] This paper proposes NegRefine, which leverages an LLM to filter proper nouns and subcategory labels from the negative label set, and designs a multi-label matching scoring function to handle cases where an image simultaneously matches both in-distribution and negative labels. On the ImageNet-1K benchmark, NegRefine achieves an average AUROC improvement of 1.82% and FPR95 reduction of 4.35%, establishing a new state of the art in zero-shot OOD detection.
tags:
  - ICCV 2025
  - Multimodal VLM
  - OOD Detection
  - Zero-Shot
  - CLIP
  - Negative Labels
  - Multi-Label Matching
date: 2026-05-08
content_hash: f8689a44e0264fc9
---

# NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection

**Conference**: ICCV 2025  
**arXiv**: [2507.09795](https://arxiv.org/abs/2507.09795)  
**Code**: [https://github.com/ah-ansari/NegRefine](https://github.com/ah-ansari/NegRefine)  
**Area**: Multimodal VLM  
**Keywords**: OOD Detection, Zero-Shot, CLIP, Negative Labels, Multi-Label Matching

## TL;DR
This paper proposes NegRefine, which leverages an LLM to filter proper nouns and subcategory labels from the negative label set, and designs a multi-label matching scoring function to handle cases where an image simultaneously matches both in-distribution and negative labels. On the ImageNet-1K benchmark, NegRefine achieves an average AUROC improvement of 1.82% and FPR95 reduction of 4.35%, establishing a new state of the art in zero-shot OOD detection.

## Background & Motivation

**State of the Field**: CLIP-based zero-shot OOD detection has seen significant progress in recent years. Negative label methods (e.g., NegLabel, CSP) select words from WordNet that are semantically distant from in-distribution classes as "negative labels," and exploit CLIP's image-text similarity to distinguish in-distribution from OOD samples—representing the most promising direction in this area.

**Limitations of Prior Work**: Negative label methods suffer from three critical issues: (a) **Subcategory overlap**—negative labels may contain subcategories of in-distribution labels (e.g., "african daisy" is a subcategory of "daisy"), and CLIP tends to assign higher scores to finer-grained labels, causing in-distribution samples to be misclassified as OOD; (b) **Proper noun interference**—WordNet contains many proper nouns (e.g., "costa rica") that yield unexpectedly high similarity scores for certain in-distribution images; (c) **Multi-label matching**—real-world images often contain multiple objects or match multiple descriptions, such that in-distribution images may simultaneously match certain negative labels with high confidence.

**Root Cause**: Existing methods rely solely on text-semantic similarity thresholds to select negative labels, without explicitly modeling lexical hierarchical relationships. Scoring functions treat each label's match independently, ignoring the reality that a single image may simultaneously match multiple labels.

**Paper Goals**: (a) How to clean the negative label set by removing subcategory and proper noun entries that cause misclassification; (b) How to design a scoring function that is robust to images simultaneously matching both in-distribution and negative labels.

**Starting Point**: Leveraging the semantic understanding capabilities of LLMs to identify hierarchical lexical relationships and proper noun attributes; exploiting CLIP's capacity—learned during training on multi-object descriptions—to detect multi-label matching by constructing caption-like texts through label concatenation.

**Core Idea**: Filter harmful negative labels using an LLM + construct a multi-matching score via label concatenation, jointly improving the reliability of zero-shot OOD detection.

## Method

### Overall Architecture
NegRefine builds upon the NegLabel/CSP framework. Given a test image and an in-distribution label set $Y_{in}$, the method first applies the NegFilter mechanism to remove proper nouns and subcategory labels from the negative label set $Y_{neg}$, yielding a refined set $Y'_{neg}$. At inference time, both the original NegLabel score $S_{NegLabel}$ and the multi-label matching score $S_{MM}$ are computed for each image and combined as: $S(x) = S_{NegLabel}(x) + \alpha \times S_{MM}(x)$.

### Key Designs

1. **NegFilter — Negative Label Filtering Mechanism**:

    - **Function**: Removes proper nouns and subcategories of in-distribution labels from the initial negative label set.
    - **Mechanism**: For each negative label $w$, the LLM first determines whether it is a proper noun ("Is $w$ a proper noun, like the name of an entity?"). If not, the $n=10$ labels in $Y_{in}$ most similar to $w$ are identified, and the LLM is queried for each regarding subcategory relationships. Labels confirmed as proper nouns or subcategories are removed from $Y'_{neg}$.
    - **Design Motivation**: WordNet's hierarchical structure is unreliable (e.g., "african daisy" and "daisy" appear at the same level), whereas LLMs are more accurate and flexible in judging semantic relationships. In experiments, proper noun filtering removed 1,749 labels (20.6%) and subcategory filtering removed 307 (3.6%).

2. **Multi-Matching Score ($S_{MM}$)**:

    - **Function**: Provides an additional in-distribution score compensation for images that simultaneously match both in-distribution and negative labels.
    - **Mechanism**: The top-$k$ labels most matched by the image from both $Y_{in}$ and $Y'_{neg}$ are retrieved, and concatenated texts $t_{i,j}$ = "$y_i$ and $\tilde{y}_j$" are constructed. The score is computed as: $S_{MM}(x) = \max_{i,j} \frac{e^{\text{sim}(x, t_{i,j})/\tau}}{e^{\text{sim}(x, t_{i,j})/\tau} + e^{\text{sim}(x, \tilde{y}_j)/\tau}}$. If the image genuinely contains both an in-distribution object and a negative-label object, the concatenated text similarity is significantly higher than that of the negative label alone, yielding a larger $S_{MM}$; if the in-distribution label is irrelevant (OOD sample), the concatenated text similarity does not increase, yielding a smaller $S_{MM}$.
    - **Design Motivation**: CLIP is trained on (image, caption) pairs where captions frequently describe multiple objects. This exploits CLIP's understanding of compositional descriptions to determine whether multi-label matching is genuine.

3. **Final Scoring Function**:

    - $S(x) = S_{NegLabel}(x) + \alpha \times S_{MM}(x)$, where $\alpha = 2$.
    - $S_{NegLabel}$ captures the overall in-distribution/OOD tendency; $S_{MM}$ compensates for in-distribution samples exhibiting multi-label matching.

### Loss & Training
This is a zero-shot method requiring no training. NegFilter uses Qwen2.5-14B-Instruct as the LLM, and filtering is performed only once as an offline preprocessing step.

## Key Experimental Results

### Main Results
ImageNet-1K is used as in-distribution data, evaluated on 4 reliable OOD datasets (excluding SUN, Places, and Textures, which have been shown to contain significant in-distribution contamination):

| Method | iNaturalist AUROC↑ | OpenImage-O AUROC↑ | Clean AUROC↑ | NINCO AUROC↑ | Avg. AUROC↑ | Avg. FPR95↓ |
|------|-------|-------|-------|-------|-------|-------|
| MCM | 94.59 | 92.00 | 83.24 | 74.34 | 86.04 | 52.59 |
| GL-MCM | 96.44 | 92.91 | 84.78 | 76.03 | 87.54 | 44.46 |
| CLIPN | 96.20 | 92.22 | 87.31 | 78.72 | 88.61 | 39.39 |
| NegLabel | 99.49 | 93.74 | 86.79 | 77.30 | 89.33 | 35.11 |
| CSP | 99.60 | 94.09 | 88.32 | 77.88 | 89.97 | 34.47 |
| **NegRefine** | **99.57** | **95.00** | **90.65** | **81.92** | **91.79** | **30.12** |

### Ablation Study

| NegFilter | $S_{MM}$ | Avg. AUROC↑ | Avg. FPR95↓ | Note |
|-----------|----------|-----------|-----------|------|
| ✗ | ✗ | 89.97 | 34.47 | CSP baseline |
| ✓ | ✗ | 90.95 | 31.88 | Filter only: FPR95 −2.59% |
| ✗ | ✓ | 91.35 | 31.70 | MM only: FPR95 −2.77% |
| ✓ | ✓ | **91.79** | **30.12** | Full method: FPR95 −4.35% |

### Key Findings
- **Complementary gains from both components**: NegFilter and $S_{MM}$ each independently contribute approximately 2.6–2.8% FPR95 improvement; combined, they achieve 4.35%.
- **Subcategory filtering has greater impact than proper noun filtering**: Although subcategory filtering removes only 307 labels (vs. 1,749 for proper nouns), its standalone FPR95 improvement is larger (0.88% vs. 0.61%), indicating that subcategory overlap is the more severe issue.
- **$S_{MM}$ outperforms local feature methods**: GL-MCM uses CLIP patch tokens to address multi-object scenarios, but the proposed label concatenation strategy is more effective, as even single-object images may match multiple labels (e.g., a necklace matching a leaf shape in Fig. 1(d))—a case that patch features cannot handle.
- **OOD dataset selection matters**: Following the findings of [Bitterwolf 2023], the authors exclude contaminated OOD datasets such as SUN (26.2% in-distribution overlap) and Places (59.5%).

## Highlights & Insights
- **LLM as a semantic filter**: Using an LLM to assess hierarchical lexical relationships and proper noun attributes is more reliable than relying on WordNet's own structure. This approach is extensible to other tasks requiring fine-grained semantic judgment.
- **Elegant label concatenation design**: Constructing "$y_i$ and $\tilde{y}_j$" as caption-like text exploits CLIP's understanding of multi-object descriptions to detect genuine multi-label matching—a simple intuition with notable empirical effect.
- **Re-examining OOD benchmark reliability**: The paper highlights severe in-distribution contamination in widely used OOD datasets (SUN, Places, Textures) and adopts a more reliable evaluation protocol, offering a valuable reference for the broader OOD detection community.

## Limitations & Future Work
- **LLM dependency**: NegFilter relies on the accuracy of LLM judgments; different LLMs may produce different results.
- **Hyperparameter sensitivity**: $\alpha=2$ and $k=5$ are fixed values; while ablations are provided, no automated tuning is performed.
- **Computational overhead**: Inference requires computing similarity for $k^2=25$ concatenated texts, introducing additional latency.
- **Tied to the negative label paradigm**: The design of $S_{MM}$ is inherently coupled to the negative label–in-distribution label contrastive framework.
- **Future directions**: One could explore using VLMs to perform direct multi-label detection on image content as an alternative to label concatenation, and investigate adaptive $\alpha$ weighting rather than a fixed value.

## Related Work & Insights
- **vs. NegLabel [ICLR'24]**: NegLabel pioneered the negative label concept but neglected label set quality and multi-label matching; NegRefine directly addresses these shortcomings within the same framework.
- **vs. CSP [AAAI'24]**: CSP constructs superclass labels using adjectives to extend OOD coverage of negative labels, but remains affected by subcategory overlap and multi-label matching issues.
- **vs. GL-MCM [NeurIPS'23]**: GL-MCM uses CLIP local features to handle multi-object images, but is effective only for spatially separable objects; the proposed method is more general.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Each component is individually straightforward, but their combination is effective and the label concatenation idea is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Detailed ablations, comprehensive comparisons, and adoption of a more reliable evaluation benchmark.
- **Writing Quality**: ⭐⭐⭐⭐ Three pain points are clearly illustrated with motivating examples; the logical chain is complete.
- **Value**: ⭐⭐⭐⭐ Achieves substantial improvement in zero-shot OOD detection with a method that is directly transferable.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](../../CVPR2026/multimodal_vlm/mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)
- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](../../AAAI2026/multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](../../CVPR2026/multimodal_vlm/activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)

<!-- RELATED:END -->
