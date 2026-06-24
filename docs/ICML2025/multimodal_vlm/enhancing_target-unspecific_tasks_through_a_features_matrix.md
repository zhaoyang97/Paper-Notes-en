---
title: >-
  [Paper Note] Enhancing Target-unspecific Tasks through a Features Matrix
description: >-
  [ICML 2025][Multimodal VLM][CLIP] Proposes the Features Matrix (FM) method, which leverages multiple hand-crafted prompt templates to extract general knowledge from frozen CLIP to construct a features matrix. By aligning unexpected features with fine-tuned visual features, it enhances the model's performance on target-unspecific tasks (e.g., base-to-novel generalization, cross-dataset generalization, domain generalization).
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "CLIP"
  - "Features Matrix"
  - "Prompt Learning"
  - "Base-to-Novel Generalization"
  - "Generalization Ability"
date: 2026-05-08
content_hash: fa4173ff28fed165
---

# Enhancing Target-unspecific Tasks through a Features Matrix

**Conference**: ICML 2025  
**arXiv**: [2505.03414](https://arxiv.org/abs/2505.03414)  
**Code**: Unreleased  
**Area**: Multimodal/Vision-Language Models (VLM), Prompt Learning  
**Keywords**: CLIP, Features Matrix, Prompt Learning, Base-to-Novel Generalization, Generalization Ability

## TL;DR

Proposes the Features Matrix (FM) method, which leverages multiple hand-crafted prompt templates to extract general knowledge from frozen CLIP to construct a features matrix. By aligning unexpected features with fine-tuned visual features, it enhances the model's performance on target-unspecific tasks (e.g., base-to-novel generalization, cross-dataset generalization, domain generalization).

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: In recent years, large-scale vision-language models such as CLIP have achieved remarkable success in zero-shot inference. Prompt learning methods (such as CoOp, CoCoOp) adapt to downstream tasks by learning trainable prompt embeddings, performing well on base classes but often underperforming zero-shot CLIP with hand-crafted prompts on novel classes.

The core issue is: **prompt tuning is prone to overfitting to the downstream data distribution, causing the model to lose the general generalization capabilities acquired during pre-training.** For instance, CoOp achieves an accuracy of only 67.96% on novel classes, which is far lower than the 74.22% of zero-shot CLIP. Even when methods like KgCoOp introduce regularization constraints, their performance on novel classes (72.70%) still falls short of zero-shot CLIP.

The authors argue that the reason is: **regularization via a single hand-crafted prompt cannot fully exploit and utilize the diverse semantic general knowledge within CLIP.**

## Method

### Overall Architecture

The FM method is a general plug-and-play module that is compatible with existing prompt learning frameworks such as CoOp, CoCoOp, MaPLe, and PromptSRC.

Mechanism:
1. Feed 60 different hand-crafted prompt templates (such as "a photo of one", "a picture of a", "a drawing of a", etc.) into the text encoder of a frozen CLIP model.
2. For each class, extract text features from all templates to construct a **Features Matrix**.
3. Calculate the matching scores between the fine-tuned visual features and the features within the Features Matrix to form a **Scores Matrix**.

### Key Designs: Unexpected Features

Identify two types of "unexpected" features from the Scores Matrix:

- **Designated unexpected features** $F^k_{un}$: Features among the designated features of the current class (label $k$) that have low cosine similarity scores (low-$\beta$).
- **Non-designated unexpected features** $F^{\hat{k}}_{un}$: Features among the non-designated classes that have high cosine similarity scores (top-$\beta$).

These unexpected features represent the general semantic information that the model is likely to confuse or ignore.

### Loss & Training

Contrastive loss $\mathcal{L}_{CL}$:

$$\mathcal{L}_{CL} = -\log \frac{\exp\{\cos(t_k, v^{tun})\}}{\exp\{\cos(t_k, v^{tun})\} + \exp\{\cos(t_{\hat{k}}, v^{tun})\}}$$

where $t_k \in F^k_{un}$, $t_{\hat{k}} \in F^{\hat{k}}_{un}$.

The total loss is:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \gamma \mathcal{L}_{CL}$$

where $\mathcal{L}_{CE}$ is the standard cross-entropy loss for bi-modal alignment, and $\gamma$ is a hyperparameter.

## Key Experimental Results

### Main Results: Base-to-Novel Generalization (Average of 11 Datasets)

| Method | Base | Novel | HM |
|------|------|-------|-----|
| CoOp | 82.69 | 63.22 | 71.66 |
| CoOp+DePT | 83.66 | 71.82 | 77.29 |
| **CoOp+Ours** | **81.15** | **74.66** | **77.79** |
| MaPLe | 82.28 | 75.14 | 78.55 |
| MaPLe+DePT | 84.85 | 74.82 | 79.52 |
| **MaPLe+Ours** | **84.45** | **76.53** | **80.32** |
| PromptSRC | 84.26 | 76.10 | 79.97 |
| **PromptSRC+Ours** | **85.70** | **77.35** | **81.32** |

### Key Findings

- FM performance improves significantly on novel classes: the novel accuracy of CoOp+FM increases from 63.22% to 74.66%, surpassing the zero-shot CLIP performance for the first time.
- Combined with MaPLe and PromptSRC, FM consistently outperforms DePT (CVPR 2024) across all representative baselines.
- On ImageNet, PromptSRC+FM achieves a Harmonic Mean (HM) of 75.07%, outperforming all comparison methods.

## Highlights & Insights

1. **Plug-and-play design**: As a general module, FM can be seamlessly integrated into existing textual or multi-modal prompt learning frameworks.
2. **Excavating the value of hand-crafted prompts**: Deeply mining semantic information from frozen CLIP using multiple hand-crafted prompt templates.
3. **Clever concept of unexpected features**: Effectively preserving general knowledge through contrastive learning that focuses on feature pairs easily confused by the model.
4. **Training-free feature matrix**: FM is extracted once from the pre-trained CLIP, adding no extra training overhead.

## Limitations & Future Work

- Using 60 prompt templates in FM increases the computational overhead during inference (which can be mitigated by pre-calculating the features matrix).
- Hyperparameters such as $\beta$ and $\gamma$ require tuning.
- Slightly sacrificing accuracy on some base classes may occur to trade for novel class generalization.
- Validated only in classification scenarios; its effectiveness on downstream tasks such as detection/segmentation remains unknown.
- The size of the feature matrix grows linearly with the number of classes and templates.

## Related Work & Insights

- CoOp/CoCoOp (Text prompt learning)
- MaPLe/PromptSRC (Multi-modal prompt learning)
- DePT (CVPR 2024, Deformable Prompt Tuning)
- KgCoOp (Knowledge-guided prompt constraints)

## Rating

⭐⭐⭐⭐ — The method is simple yet effective, with experiments comprehensively covering 11 datasets and multiple frameworks, achieving significant improvements in novel classes. The core idea is clear and easy to understand, but the technical novelty is relatively limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[NeurIPS 2025\] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](../../NeurIPS2025/multimodal_vlm/sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)
- [\[ICLR 2026\] Enhancing Multi-Image Understanding through Delimiter Token Scaling](../../ICLR2026/multimodal_vlm/enhancing_multi-image_understanding_through_delimiter_token_scaling.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[CVPR 2025\] Mimic In-Context Learning for Multimodal Tasks](../../CVPR2025/multimodal_vlm/mimic_in-context_learning_for_multimodal_tasks.md)

</div>

<!-- RELATED:END -->
