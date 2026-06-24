---
title: >-
  [Paper Note] APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension
description: >-
  [ECCV 2024][LLM (Other)][Weakly Supervised Referring Expression Comprehension] This paper proposes APL (Anchor-based Prompt Learning), which designs an Anchor-based Prompt Encoder (APE) to generate distinctive prompts across three categories: location, color, and category. By dynamically integrating these prompts into anchor features to enrich visual semantics, alongside text reconstruction and visual alignment losses, APL achieves precise vision-language alignment. It outper…
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "Weakly Supervised Referring Expression Comprehension"
  - "Prompt Learning"
  - "Anchor Features"
  - "Vision-Language Alignment"
  - "One-stage Detection"
date: 2026-05-08
content_hash: 7db44f2485ec78fd
---

# APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension

**Conference**: ECCV 2024  
**Code**: [https://github.com/Yaxin9Luo/APL](https://github.com/Yaxin9Luo/APL)  
**Area**: LLM/NLP  
**Keywords**: Weakly Supervised Referring Expression Comprehension, Prompt Learning, Anchor Features, Vision-Language Alignment, One-stage Detection

## TL;DR

This paper proposes APL (Anchor-based Prompt Learning), which designs an Anchor-based Prompt Encoder (APE) to generate distinctive prompts across three categories: location, color, and category. By dynamically integrating these prompts into anchor features to enrich visual semantics, alongside text reconstruction and visual alignment losses, APL achieves precise vision-language alignment. It outperforms existing weakly supervised methods on four REC benchmarks (e.g., exceeding RefCLIP by 6.44% on RefCOCO).

## Background & Motivation

**Background**: Referring Expression Comprehension (REC) aims to localize a target object according to a given natural language description. Traditional methods rely on expensive instance-level annotations (i.e., explicit correspondences between bounding boxes and textual descriptions), which suffer from high training costs. Recently, weakly supervised REC has emerged as a research hotspot. Among them, RefCLIP is a representative one-stage weakly supervised REC method, which utilizes the anchor features of a pre-trained one-stage detection network to represent candidate objects and localizes targets through anchor-text ranking.

**Limitations of Prior Work**: Although RefCLIP demonstrates certain effectiveness, its visual semantic representation remains ambiguous and insufficient. Specifically, anchor features only carry coarse-grained spatial information, lacking fine-grained discriminative attributes such as location, color, and category. This makes it difficult to accurately distinguish similar candidate objects under weakly supervised settings. Additionally, semantic alignment between visual features and textual descriptions is insufficiently precise.

**Key Challenge**: The core challenge of weakly supervised REC lies in the absence of explicit object-text annotation pairs, requiring the model to learn fine-grained localization capabilities solely under image-text-pair-level supervision. The anchor features in existing methods are too coarse-grained to effectively encode the multi-dimensional visual attributes mentioned in textual descriptions (such as position, color, and category information in "the red cup on the left").

**Goal**: (1) How to enrich the visual semantic representation capability of anchors under weakly supervised settings; (2) How to achieve more precise vision-language alignment.

**Key Insight**: The authors observe that referring expressions naturally consist of descriptions across three dimensions: location (left, top), color (red, blue), and category (cup, person). Explicitly encoding these three types of information as prompts and injecting them into anchor features can substantially enhance the discriminative power of visual semantics.

**Core Idea**: Generate three-dimensional prompt information (location, color, category) using an Anchor-based Prompt Encoder and dynamically merge it into anchor features to enrich the visual semantic representation of weakly supervised REC.

## Method

### Overall Architecture

APL adopts a one-stage detection framework, taking images and referring expression texts as input. After extracting multi-scale features through a YOLOv3 backbone, the image generates a large number of anchors, each corresponding to a visual feature. The text is encoded into textual features using a language encoder (such as BERT or GloVe). The core of APL lies in: (1) using the Anchor-based Prompt Encoder (APE) to generate three types of prompts (location, color, category) for each anchor and dynamically fusing them into anchor features; (2) ranking the similarity between the enhanced anchor features and text features, selecting the top-scoring anchor as the localization result; (3) employing text reconstruction loss and visual alignment loss as auxiliary objectives to achieve precise vision-language alignment.

### Key Designs

1. **Anchor-based Prompt Encoder (APE)**:

    - **Function**: Generates discriminative prompts covering three aspects—location, color, and category—for each anchor.
    - **Mechanism**: APE comprises three sub-encoders. The location prompt encoder generates location embeddings based on the spatial coordinates (center point, width, height) of the anchor. The color prompt encoder extracts color histogram features from the anchor's corresponding image region and encodes them into color embeddings. The category prompt encoder utilizes the category prediction distribution of the pre-trained detector to generate category semantic embeddings. The three types of prompts are dynamically injected into the original anchor features through a learnable fusion mechanism (such as attention weighting) to form semantically-enhanced anchor representations.
    - **Design Motivation**: Descriptive information in referring expressions can naturally be decomposed into location, color, and category dimensions. Explicitly modeling these three types of information and injecting them as prompts can effectively compensate for the semantic deficiency of anchor features under weak supervision.

2. **Text Reconstruction Loss**:

    - **Function**: Serves as an auxiliary training objective to reconstruct corresponding textual descriptions from the enhanced anchor features.
    - **Mechanism**: Given the prompt-enhanced anchor features, a text decoder reconstructs the original referring expression. This loss encourages anchor features to fully encode the various attribute details mentioned in the text. The reconstruction loss is computed using a cross-entropy format to measure the difference between predicted tokens and ground-truth tokens.
    - **Design Motivation**: If the enhanced anchor features can accurately reconstruct the original text description, it indicates that these features have successfully captured key information such as location, color, and category mentioned in the text. This acts as a self-supervised constraint without requiring extra instance-level annotations.

3. **Visual Alignment Loss**:

    - **Function**: Ensures consistency between the enhanced anchor features and the original visual features.
    - **Mechanism**: Visual alignment is constrained in a contrastive learning manner—the enhanced feature and original feature of the same anchor should be close (positive pairs), whereas features of different anchors should be far away (negative pairs). This prevents semantic drift of features after prompt injection.
    - **Design Motivation**: Direct injection of prompts into anchor features might degrade the quality of raw visual representation. The visual alignment loss acts as a regularizer, ensuring that prompt enhancement adds extra discriminative info on top of preserving the original visual semantics, rather than overwriting existing representations.

### Loss & Training

The overall training loss consists of three components: (1) the main loss, which is the anchor-text contrastive ranking loss that requires the correct anchor to have a higher similarity score with the text than other anchors; (2) the text reconstruction loss $L_{rec}$ for text generation; (3) the visual alignment loss $L_{align}$ to maintain feature consistency. The total loss is formulated as: $L = L_{rank} + \lambda_1 L_{rec} + \lambda_2 L_{align}$. Only image-text pair-level supervision (weak supervision) is required during training, without instance-level bounding box annotations. The model utilizes a pre-trained YOLOv3 as the detection backbone, whose weights are pre-trained on the COCO dataset (excluding images in the RefCOCO validation/test sets).

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (APL) | Prev. SOTA (RefCLIP) | Gain |
|--------|------|----------|-----------------|------|
| RefCOCO val | Acc@0.5 | 64.51 | ~58.07 | +6.44% |
| RefCOCO testA | Acc@0.5 | 61.91 | - | - |
| RefCOCO testB | Acc@0.5 | 63.57 | - | - |
| RefCOCO+ val | Acc@0.5 | 42.70 | - | - |
| RefCOCO+ testA | Acc@0.5 | 42.84 | - | - |
| RefCOCO+ testB | Acc@0.5 | 39.80 | - | - |
| RefCOCOg val | Acc@0.5 | 50.22 | - | - |

### Weakly Supervised Referring Expression Segmentation (Weakly RES)

| Dataset | Metric | Ours (APL) |
|--------|------|----------|
| RefCOCO val | mIoU | 55.92 |
| RefCOCO testA | mIoU | 54.84 |
| RefCOCO testB | mIoU | 55.64 |
| RefCOCO+ val | mIoU | 34.92 |
| RefCOCOg val | mIoU | 40.13 |

### Training Other Models with Pseudo Labels

| Configuration | RefCOCO val | RefCOCO+ val | RefCOCOg val | Description |
|------|------------|-------------|-------------|------|
| APL_SimREC | 63.94 | 42.11 | 48.35 | Training SimREC with pseudo labels generated by APL |
| APL_TransVG | 64.86 | 39.28 | 46.11 | Training TransVG with pseudo labels generated by APL |

### Key Findings

- APL achieves a 6.44% improvement over RefCLIP on RefCOCO, proving that prompt learning strategies are highly effective in weakly supervised REC.
- The pseudo labels generated by APL can be used to train other fully supervised REC models (such as SimREC, TransVG), achieving performance close to that of using APL directly.
- The method also exhibits strong generalization capability on the weakly supervised referring expression segmentation task.

## Highlights & Insights

- Introducing prompt learning into weakly supervised REC is an innovative direction, effectively compensating for the visual semantic deficiency in weak supervision through three-dimensional prompts (location, color, category).
- The design of the text reconstruction loss is clever—by requiring anchor features to "speak" the corresponding textual descriptions, it implicitly forces the features to encode rich semantic information.
- The pseudo-label training experiments with other models demonstrate the practical value of the method, serving as a bridge from weak supervision to full supervision.

## Limitations & Future Work

- The method relies on a pre-trained YOLOv3 detector, where anchor quality directly limits the final performance. Utilizing stronger detectors (such as the DETR series) might yield further improvements.
- The color prompt encoder is based on color histograms, whose robustness to complex textures and lighting changes remains to be validated.
- The three types of prompts (location, color, category) are manually defined; future work could explore automatic discovery of multi-dimensional prompt types.
- Performance on RefCOCO+ is notably lower than on RefCOCO, indicating that handling pure attribute descriptions (without location keywords) still has room for improvement.

## Related Work & Insights

- **RefCLIP** is the primary baseline of this work, pioneering the one-stage weakly supervised REC paradigm.
- The vision-language contrastive learning framework of **CLIP** provides the theoretical foundation for the anchor-text ranking in this paper.
- Prompt Learning has been widely adopted in NLP (e.g., GPT-3, P-tuning) and multimodal learning. This paper innovatively applies it to weakly supervised object localization.
- The prompt-injection concept of this method could inspire other weakly supervised visual tasks, such as weakly supervised detection and weakly supervised segmentation.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining prompt learning with weakly supervised REC is highly novel, and the three-dimensional prompt design is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across four datasets with rich ablation and pseudo-labeling experiments.
- Writing Quality: ⭐⭐⭐ The methodology is clearly described, though some experimental details could be elaborated further.
- Value: ⭐⭐⭐⭐ Provides excellent reference value for weakly supervised vision-language tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FunQA: Towards Surprising Video Comprehension](funqa_towards_surprising_video_comprehension.md)
- [\[CVPR 2025\] Learning Textual Prompts for Open-World Semi-Supervised Learning](../../CVPR2025/llm_nlp/learning_textual_prompts_for_open-world_semi-supervised_learning.md)
- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](../../NeurIPS2025/llm_nlp/system_prompt_optimization_with_meta-learning.md)
- [\[ECCV 2024\] Cultural Value Differences of LLMs: Prompt, Language, and Model Size](cultural_value_differences_llms.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)

</div>

<!-- RELATED:END -->
