---
title: >-
  [Paper Note] Weakly-Supervised Referring Video Object Segmentation through Text Supervision
description: >-
  [CVPR 2026][Segmentation][Weakly supervised] This paper proposes WSRVOS, the first weakly supervised referring video object segmentation framework that uses only text expressions as supervision signals. It achieves signi…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Weakly supervised"
  - "video object segmentation"
  - "referring expression"
  - "text supervision"
  - "multimodal alignment"
date: 2026-05-08
content_hash: 3ea14689f94df048
---

# Weakly-Supervised Referring Video Object Segmentation through Text Supervision

**Conference**: CVPR 2026
**arXiv**: [2604.17797](https://arxiv.org/abs/2604.17797)  
**Code**: [https://github.com/viscom-tongji/WSRVOS](https://github.com/viscom-tongji/WSRVOS)  
**Area**: Segmentation
**Keywords**: Weakly supervised, video object segmentation, referring expression, text supervision, multimodal alignment

## TL;DR

This paper proposes WSRVOS, the first weakly supervised referring video object segmentation framework that uses only text expressions as supervision signals. It achieves significant reduction in reliance on pixel-level annotations through MLLM-driven contrastive expression augmentation, bidirectional visual-language feature selection, instance-aware expression classification, and temporal segment ranking constraints.

## Background & Motivation

**Background**: Referring video object segmentation (RVOS) segments target instances in video according to text expressions. Mainstream methods (e.g., ReferFormer, SAMWISE) rely on pixel-level mask annotations, achieving strong performance at the cost of prohibitively expensive annotation.

**Limitations of Prior Work**: Exploration of weakly supervised RVOS is still in its early stages. Existing works such as WRVOS use the first-frame mask with bounding boxes for subsequent frames, while OCPG generates pseudo-masks from bbox/point annotations. However, bbox and point annotations still require substantial per-frame manual effort, particularly costly for long videos.

**Key Challenge**: How can a model learn to localize and segment target instances in video using only text expressions, without any spatial annotations (mask, bbox, or point)? The key challenges are: (1) heterogeneity between visual and language features makes semantic alignment difficult; (2) temporal dynamics and occlusions in video further complicate the alignment process.

**Goal**: Design an end-to-end weakly supervised RVOS framework that uses only text expressions as supervision during training, requiring no spatial annotations of any kind.

**Key Insight**: The captioning capability of multimodal large language models (MLLMs) such as Qwen3-VL can generate rich positive and negative textual descriptions for video, providing supervision signals far richer than the original brief expressions. Contrastive learning between correct and incorrect descriptions enables the model to indirectly acquire localization ability.

**Core Idea**: Use MLLMs to generate contrastive expression augmentation data (richly detailed positive expressions and hard negative expressions), then train the segmentation model through instance-aware classification and pseudo-mask fusion, without using any spatial annotations throughout.

## Method

### Overall Architecture

The framework consists of five components: (1) contrastive expression augmentation — generating positive and negative text expressions via Qwen3-VL; (2) multimodal feature selection and interaction — bidirectional selection of relevant visual and language features; (3) instance-aware expression classification — distinguishing positive from negative expressions; (4) positive prediction fusion — generating pseudo-masks as additional supervision; and (5) temporal segment ranking constraints — constraining mask overlap relationships between temporally adjacent frames.

### Key Designs

1. **Contrastive Referring Expression Augmentation**:

    - **Function**: Expands simple original expressions into rich positive and hard negative textual supervision signals.
    - **Mechanism**: *Positive expressions*: Qwen3-VL generates $P$ more detailed descriptions (focusing on appearance, actions, and interaction relationships) conditioned on the video and the original expression. InternVideo2 computes video-text similarity to filter out low-confidence descriptions ($c^k < 0.8$), and the retained descriptions are concatenated with the original expression to preserve original information. *Negative expressions*: Qwen3-VL modifies the category, attributes, and actions of the target instance to generate $N$ semantically plausible but target-inconsistent descriptions.
    - **Design Motivation**: Original dataset expressions are too simplistic and lack fine-grained semantic detail. Positive augmentation provides richer alignment signals, while hard negatives force the model to learn more discriminative representations. The MLLM is used only offline for data augmentation and does not participate in inference.

2. **Bidirectional Visual-Language Feature Selection and Instance-Aware Classification**:

    - **Function**: Filters out visual information irrelevant to the expression and non-informative words in the text, enabling precise multimodal alignment.
    - **Mechanism**: Bidirectional selection retains mutually highly relevant subsets of visual and language features, followed by proposal aggregation and expression matching under a Multiple Instance Learning paradigm, enabling the model to distinguish positive from negative expressions.
    - **Design Motivation**: Temporal dynamics in video cause visual features to contain substantial redundant information irrelevant to the referring expression, while text may include uninformative tokens such as prepositions. The refined features after bidirectional filtering facilitate more precise alignment.

3. **Positive Prediction Fusion and Temporal Ranking Constraints**:

    - **Function**: Generates high-quality pseudo-masks to provide spatial supervision while constraining temporal consistency.
    - **Mechanism**: Predictions from multiple positive expressions are fused into reliable pseudo-masks, which serve as additional supervision signals for segmentation training. The temporal segment ranking constraint requires that mask overlap between temporally adjacent frames exceeds that between distant frames, encouraging temporal smoothness: $\text{IoU}(m_t, m_{t+\delta_1}) > \text{IoU}(m_t, m_{t+\delta_2})$ when $\delta_1 < \delta_2$.
    - **Design Motivation**: Classification loss alone provides insufficient spatial localization signals. Pseudo-mask fusion exploits the intuition that predictions from multiple correct descriptions should be consistent, extracting reliable regions through prediction agreement. The temporal ranking constraint leverages the prior of temporal continuity in video.

### Loss & Training

The training objective comprises three components: instance-aware expression classification loss (distinguishing positive from negative text), pseudo-mask supervision loss (spatial localization), and temporal segment ranking loss (temporal consistency). The MLLM is used only offline during training data preprocessing.

## Key Experimental Results

### Main Results

| Dataset | Metric | WSRVOS (Ours) | OCPG (Point Sup.) | Gap |
|--------|------|-------------|-------------|------|
| A2D-Sentences | mAP | Best | Baseline | Significant improvement |
| J-HMDB Sentences | J&F | Best | Baseline | Significant improvement |
| Ref-YouTube-VOS | J&F | Best | Baseline | Significant improvement |
| Ref-DAVIS17 | J&F | Best | Baseline | Significant improvement |

### Ablation Study

| Configuration | Performance Change | Note |
|------|---------|------|
| Full WSRVOS | Best | Complete model |
| w/o contrastive expression augmentation | Degraded | Insufficient supervision signals |
| w/o bidirectional feature selection | Degraded | Reduced alignment precision |
| w/o positive prediction fusion | Degraded | Lack of spatial supervision signal |
| w/o temporal ranking constraint | Degraded | Degraded temporal consistency |

### Key Findings

- WSRVOS, supervised by text alone, outperforms weakly supervised methods using bbox/point annotations (e.g., OCPG), demonstrating that rich textual supervision can be more effective than sparse spatial annotations.
- Contrastive expression augmentation contributes most — both the discriminativeness of hard negatives and the richness of positive descriptions are critical.
- The temporal ranking constraint yields greater gains on long videos; its contribution is limited on short videos where adjacent frames differ minimally.

## Highlights & Insights

- The setting of requiring no spatial annotations of any kind represents a significant step forward in the RVOS field. Leveraging MLLMs' captioning capability to transform text from "weak supervision" into "rich supervision" is a highly forward-looking idea.
- The positive prediction fusion strategy is elegant: if the model's predictions for multiple correct descriptions are highly consistent, those regions very likely correspond to the target — using "prediction consistency" as a reliability measure for pseudo-labels.
- The temporal ranking constraint is simple yet effective: rather than requiring precise inter-frame mask propagation, it imposes only a soft constraint that "closer frames should be more similar."

## Limitations & Future Work

- Performance depends on the quality of expressions generated by the MLLM (Qwen3-VL); errors in the MLLM's video understanding may introduce noise.
- The InternVideo2 filtering threshold of 0.8 is manually set and may require adjustment for different domains.
- In scenarios with small or heavily occluded target instances, localization ability under pure text supervision may be insufficient.
- Future work could explore adaptive expression generation and filtering strategies, or incorporate visual grounding pretraining to enhance localization.

## Related Work & Insights

- **vs. WRVOS**: Requires the first-frame mask and bounding boxes; WSRVOS requires no spatial annotations whatsoever.
- **vs. OCPG**: Uses bbox/point annotations to generate pseudo-masks, yet WSRVOS achieves superior performance using only text.
- **vs. TRIS/PCNet (image-level)**: Image-level weakly supervised referring segmentation methods; WSRVOS extends to the more challenging video setting.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First purely text-supervised RVOS method; paradigm-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on four datasets with comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; method description is systematic.
- Value: ⭐⭐⭐⭐⭐ Substantially reduces annotation costs for RVOS; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[AAAI 2026\] SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation](../../AAAI2026/segmentation/ssr_semantic_and_spatial_rectification_for_clip-based_weakly_supervised_segmenta.md)
- [\[CVPR 2026\] Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning](follow_the_saliency_supervised_saliency_for_retrieval-augmented_dense_video_capt.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](../../ICCV2025/segmentation/referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[CVPR 2026\] Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](boundary_segment_action_segmentation.md)

</div>

<!-- RELATED:END -->
