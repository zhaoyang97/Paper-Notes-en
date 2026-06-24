---
title: >-
  [Paper Note] Learning Textual Prompts for Open-World Semi-Supervised Learning
description: >-
  [CVPR 2025][LLM (Other)][Open-world semi-supervised learning] This paper proposes a new method for open-world semi-supervised learning (OWSSL) that enhances vision-language alignment via a global-and-local textual prompt learning strategy, and designs a forward-and-backward strategy to reduce noise in vision-language matching for unlabeled samples, outperforming the SOTA significantly on multiple fine-grained datasets.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Open-world semi-supervised learning"
  - "textual prompt learning"
  - "vision-language alignment"
  - "fine-grained recognition"
  - "noise suppression"
date: 2026-05-08
content_hash: 4de1ded2e0b85a90
---

# Learning Textual Prompts for Open-World Semi-Supervised Learning

**Conference**: CVPR 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Open-world semi-supervised learning, textual prompt learning, vision-language alignment, fine-grained recognition, noise suppression

## TL;DR

This paper proposes a new method for open-world semi-supervised learning (OWSSL) that enhances vision-language alignment via a global-and-local textual prompt learning strategy, and designs a forward-and-backward strategy to reduce noise in vision-language matching for unlabeled samples, outperforming the SOTA significantly on multiple fine-grained datasets.

## Background & Motivation

**Background**: Traditional semi-supervised learning has achieved significant success under the closed-set assumption—where unlabeled data is assumed to share the same category set as labeled data. However, the real world is open, and unlabeled data often contains novel categories never seen in the labeled set. To address this, researchers have proposed Open-World Semi-Supervised Learning (OWSSL), which requires the model to accurately recognize known classes while discovering and clustering unknown classes.

**Limitations of Prior Work**: (1) Visually similar fine-grained categories are difficult to distinguish—such as different bird species or aircraft models, making it hard to rely solely on visual features for reliable classification. (2) Prior methods have attempted to introduce textual information to help distinguish visually similar classes, but the alignment between images and text is sub-optimal, leading to limited performance gains. (3) Pseudo-labels for unlabeled samples contain noise, and using these noisy labels for vision-language matching further degrades alignment quality.

**Key Challenge**: OWSSL needs to simultaneously handle both known class recognition and unknown class discovery. Textual information can theoretically provide discriminative semantic features to distinguish visually similar classes, but existing methods fail to effectively align images and text, particularly on unlabeled data contaminated with label noise.

**Goal**: (1) How to align images and text more effectively to extract discriminative cross-category features? (2) How to reduce the noise introduced during vision-language matching for unlabeled samples?

**Key Insight**: The authors approach this from two perspectives: prompt learning and noise mitigation. For prompt learning, a global prompt is designed to capture cross-category commonalities, while local prompts focus on category specificity. For noise mitigation, a bidirectional forward-and-backward strategy is employed to filter out unreliable vision-language matches.

**Core Idea**: To improve vision-language alignment quality via global-and-local dual-level prompt learning, and suppress unlabeled matching noise through a bidirectional forward-and-backward strategy, thereby enhancing the fine-grained discrimination capability in open-world semi-supervised learning.

## Method

### Overall Architecture

The method is built upon pre-trained vision-language models like CLIP. The input consists of a small set of labeled images and a large set of unlabeled images. The overall pipeline is: images extract features through a visual encoder $\rightarrow$ text goes through a text encoder (with learnable prompts) to generate class textual embeddings $\rightarrow$ global-and-local prompt strategy enhances vision-language alignment $\rightarrow$ forward-and-backward strategy generates reliable pseudo-labels for unlabeled samples $\rightarrow$ joint optimization of known class classification and unknown class clustering.

### Key Designs

1. **Global-and-Local Textual Prompt Learning Strategy**:

    - **Function**: To enhance the alignment between images and text, capturing global commonalities as well as category-specific traits.
    - **Mechanism**: Design a two-level learnable text prompt. The Global Prompt consists of context tokens shared among all classes to capture dataset-level common features, helping the model understand the overall task. The Local Prompt consists of category-specific learnable tokens for each class, encoding distinct semantic information of that specific class (e.g., distinguishing features like the neck color or beak shape of specific birds). Prompts from both levels are concatenated and fed into the text encoder to generate more detailed category text representations.
    - **Design Motivation**: Standard CLIP uses fixed templates such as "a photo of a [class]", which lacks the capacity to express fine-grained distinctions. While global prompts learn general domain knowledge, local prompts capture intra-class unique features. The combination of both significantly increases the discriminability between fine-grained categories.

2. **Forward-and-Backward Strategy**:

    - **Function**: To reduce the noise generated during vision-language matching for unlabeled samples.
    - **Mechanism**: Divided into forward and backward steps. Forward step: Predict classes (pseudo-labels) for unlabeled images using the current model, and select samples whose prediction confidence is higher than a threshold for training. Backward step: Perform backward verification on the selected samples from the forward step—using textual features to retrieve from the image database and checking whether the highly similar image set retrieved via text is consistent with the forward prediction. Only samples that pass both forward and backward verification are considered reliable matches.
    - **Design Motivation**: Relying solely on forward matching from image to text easily generates many noisy labels, especially in scenarios where fine-grained categories and unknown categories are mixed. Backward verification serves as a redundant check from text to image, drastically reducing the noise rate.

3. **Joint Learning of Known and Unknown Classes**:

    - **Function**: To simultaneously optimize known class classification and unknown class discovery.
    - **Mechanism**: For known classes, use supervised classification loss on labeled data and pseudo-label loss on high-confidence unlabeled data filtered by the forward-and-backward strategy. For unknown classes, use a contrastive learning framework to cluster similar unlabeled samples. Text prompts serve both known and unknown classes—known class prompts are learned directly via supervised signals, while unknown class prompts are learned indirectly via the alignment between cluster centers and text.
    - **Design Motivation**: The core challenge of the open world is to successfully accomplish both classification and discovery simultaneously.

### Loss & Training

A combination of multiple loss functions is used: cross-entropy classification loss for known classes, pseudo-label loss for unlabeled samples (filtered by the forward-and-backward strategy), vision-language alignment contrastive loss, and clustering loss for unknown classes. Training proceeds in stages: first train text prompts to build a solid foundation for vision-language alignment, then jointly optimize all components.

## Key Experimental Results

### Main Results

Experiments on multiple fine-grained datasets (CUB-200 Birds, Stanford Cars, FGVC Aircraft, etc.):

| Method | CUB-200 Known | CUB-200 Unknown | CUB-200 Overall | FGVC Overall |
|------|-------|-------|---------|---------|
| ORCA | Baseline | Baseline | Baseline | Baseline |
| PromptCAL | Medium | Medium | Medium | Medium |
| Ours | **Best** | **Best** | **Best** | **Best** |

The paper achieves significant performance gains across multiple fine-grained datasets.

### Ablation Study

| Component | Known Acc | Unknown Acc | Overall Acc |
|------|-----------|-----------|-------------|
| Baseline (w/o text prompt) | Low | Low | Low |
| + Global Prompt | Improved | Improved | Improved |
| + Global + Local Prompt | Significant Improvement | Significant Improvement | Significant Improvement |
| + Forward-and-Backward Strategy | **Best** | **Best** | **Best** |

### Key Findings

- The combination of global and local prompts is significantly superior to only using global prompts; local prompts provide crucial class-specific information for fine-grained category differentiation.
- The bidirectional forward-and-backward verification reduces the noise rate significantly more than using only forward pseudo-labels.
- Textual information brings the most notable improvements on fine-grained datasets.

## Highlights & Insights

- **Clear division of labor between global and local prompts**: Global prompts establish a baseline for domain understanding, while local prompts provide class-differentiation capabilities.
- **Simple and effective Forward-and-Backward Strategy**: Achieves reliable noise filtering via bidirectional cross-validation.
- **Validated in fine-grained scenarios**: This is where text information is most valuable, making the validation much more convincing than on coarse-grained classification tasks.

## Limitations & Future Work

- Relies on pre-trained models like CLIP, which may suffer from degraded performance in highly specialized domains.
- The number of local prompts equals the number of known classes, meaning there are no explicit prompts for unknown classes.
- The threshold for the forward-and-backward strategy needs to be set manually.
- Future work could explore using LLMs to generate richer category descriptions to enhance prompts.

## Related Work & Insights

- **ORCA**: A representative method for open-world semi-supervised learning.
- **CoOp/CoCoOp**: Pioneering works in prompt learning.
- Insight: The combination strategy of prompt learning and bidirectional verification can be integrated into other open-world scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of global-local prompts and forward-backward strategy is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Evaluated on multiple fine-grained datasets with extensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (OWSSL is a promising direction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification](../../ACL2025/llm_nlp/a_semi-supervised_scalable_unified_framework_for_e-commerce_query_classification.md)
- [\[ECCV 2024\] APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](../../ECCV2024/llm_nlp/apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)
- [\[CVPR 2025\] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)
- [\[ACL 2025\] Retrospective Learning from Interactions](../../ACL2025/llm_nlp/retrospective_learning_from_interactions.md)
- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](../../NeurIPS2025/llm_nlp/system_prompt_optimization_with_meta-learning.md)

</div>

<!-- RELATED:END -->
