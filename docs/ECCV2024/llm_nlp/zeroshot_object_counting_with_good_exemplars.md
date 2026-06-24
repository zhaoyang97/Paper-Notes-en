---
title: >-
  [Paper Note] Zero-Shot Object Counting with Good Exemplars (VA-Count)
description: >-
  [ECCV 2024][LLM (Other)][Zero-Shot Counting] Proposes VA-Count, a vision-association-based zero-shot object counting framework. It establishes robust visual associations between high-quality exemplars and images for arbitrary categories through a Grounding DINO-driven exemplar enhancement module and a contrastive learning noise suppression module.
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "Zero-Shot Counting"
  - "Vision-Language Pre-training"
  - "Exemplar Enhancement"
  - "Noise Suppression"
  - "Grounding DINO"
date: 2026-05-08
content_hash: 2a1b09f070246209
---

# Zero-Shot Object Counting with Good Exemplars (VA-Count)

**Conference**: ECCV 2024  
**arXiv**: [2407.04948](https://arxiv.org/abs/2407.04948)  
**Code**: [https://github.com/HopooLinZ/VA-Count](https://github.com/HopooLinZ/VA-Count)  
**Area**: LLM/NLP  
**Keywords**: Zero-Shot Counting, Vision-Language Pre-training, Exemplar Enhancement, Noise Suppression, Grounding DINO

## TL;DR

Proposes VA-Count, a vision-association-based zero-shot object counting framework. It establishes robust visual associations between high-quality exemplars and images for arbitrary categories through a Grounding DINO-driven exemplar enhancement module and a contrastive learning noise suppression module.

## Background & Motivation

- Zero-shot object counting (ZOC) aims to count objects in an image knowing only the category name, without requiring any annotations.
- The core problem of existing ZOC methods is their inability to effectively identify high-quality exemplars.
- Two types of existing methods each have limitations:
  1. **Vision-language alignment methods** (CLIP-Count, VLCount): Rely on direct image-text associations, resulting in insufficient representation for categories with atypical shapes.
  2. **Category-relevant exemplar search** (ZSC): Rely on arbitrary patch selection, failing to accurately delineate complete objects, and are constrained by predefined categories.
- **Goal**: Introduce a detection-driven exemplar discovery method (Grounding DINO) while simultaneously fusing textual and visual representations.

## Method

### Overall Architecture

Two core modules work collaboratively:
1. **Exemplar Enhancement Module (EEM)**: Employs Grounding DINO to discover positive and negative exemplars, filtered by a single-object classifier.
2. **Noise Suppression Module (NSM)**: Uses contrastive learning to distinguish between density maps of positive and negative exemplars, reducing the impact of erroneous exemplars.

### Key Designs

**Exemplar Enhancement Module (EEM)**:

*Grounding DINO-guided Box Selection*:
- Positive samples: Input image + positive text label (specific category name) $\rightarrow$ obtain positive candidate boxes $B^p$.
- Negative samples: Input image + negative text label ("object") $\rightarrow$ obtain negative candidate boxes $B^n$.
- Logit threshold $\tau_l = 0.02$.

*De-duplication Filtering*:
- For negative bounding boxes, remove parts that overlap with positive bounding boxes using an IoU threshold $\tau_{\text{iou}} = 0.5$.

*Single-Object Exemplar Filtering*:
- Binary classifier $\delta(\cdot) = \text{FFN}(\text{CLIP-ViT}(b))$.
- Uses a frozen CLIP-ViT-B/16 + a trainable FFN to judge whether a candidate box contains exactly one object.
- Training data: positive samples = single-object exemplars annotated in the training set; negative samples = randomly cropped patches + the whole image.
- Ensures the cleanliness of exemplars.

**Noise Suppression Module (NSM)**:

*Counter $\Gamma(\cdot)$*:
- Based on the CounTR architecture: Image encoder + exemplar-image interaction module + decoder.
- Image features act as Query; exemplar features are linearly projected as Key/Value.
- Generates positive/negative density maps using positive/negative exemplars, respectively.

*Contrastive Learning*:
- $$L_C = -\log\left(\frac{\exp(\text{sim}(D^p, D^g))}{\exp(\text{sim}(D^p, D^g)) + \exp(\text{sim}(D^n, D^g))}\right)$$
- Maximizes the similarity between the positive density map and the ground truth, while minimizing the similarity between the negative density map and the ground truth.
- Total loss: $$L_{\text{total}} = L_C + L_D$$

### Loss & Training

- Density loss $L_D$: $\text{MSE}(D^p, D^g)$
- Contrastive loss $L_C$: contrastive learning between positive/negative density maps and the ground truth.
- Optimizer: AdamW, learning rate $10^{-5}$, batch size 8.
- Two-stage training: MAE pre-training + fine-tuning.
- Select top-3 positive exemplars and top-3 negative exemplars for each image.

## Key Experimental Results

### Main Results

FSC-147 Dataset:

| Method | Type | Val MAE | Val RMSE | Test MAE | Test RMSE |
|------|------|---------|----------|----------|-----------|
| CLIP-Count | Zero-Shot | 17.78 | 55.43 | 18.97 | 95.93 |
| VLCount | Zero-Shot | 18.20 | 60.63 | 19.18 | 103.29 |
| **VA-Count** | **Zero-Shot** | **Best** | **Best** | **Best** | **Best** |

CARPK Dataset (Cross-dataset generalization): VA-Count likewise outperforms existing zero-shot and few-shot methods.

### Ablation Study

| Component | MAE Change |
|------|---------|
| Without EEM (random patches) | Significantly degraded |
| Without Single-Object Filter | Degraded (multi-object exemplars introduce noise) |
| Without NSM (no contrastive learning) | Degraded (erroneous exemplars not suppressed) |
| Full EEM + NSM | Best |

### Key Findings

1. Grounding DINO provides exemplar quality significantly superior to random patch selection.
2. Single-object filtering is crucial—bounding boxes with high Grounding DINO confidence can still contain multiple objects.
3. Contrastive learning effectively distinguishes the impact of positive and negative exemplars on density maps.
4. "object" as a general negative text label can detect objects of non-target categories.

## Highlights & Insights

- **Detection-driven exemplar discovery**: Introduces the universal detection capability of Grounding DINO into zero-shot counting, achieving high-quality exemplar acquisition for arbitrary categories.
- **Positive-negative exemplar contrastive learning**: Learns not only "what the target is" but also "what the target is not", facilitating a more robust bi-directional constraint.
- **Modular design**: EEM and NSM are independently controllable, making them easy to understand and extend.
- **Practical negative sample strategy**: Uses "object" to detect all objects, and removes those overlapping with positive classes to obtain negative samples.

## Limitations & Future Work

- Relies heavily on the detection quality of Grounding DINO, and may fail for objects that are difficult for DINO to detect (extremely small or heavily occluded).
- The single-object classifier requires additional training data and training processes.
- Only 3 positive/negative exemplars are selected per image; whether more exemplars would yield further improvements remains to be explored.
- Relatively high computational cost (Grounding DINO + CLIP + CounTR).
- Future directions: End-to-end training of the entire pipeline, and utilizing SAM to replace/enhance exemplar localization.

## Related Work & Insights

- The MAE pre-training + exemplar matching framework of CounTR serves as the base architecture of this work.
- The introduction of Grounding DINO demonstrates the plug-and-play value of large-scale VLP models in downstream tasks.
- The concept of positive-negative exemplar contrastive learning can be generalized to other tasks requiring exemplar matching (e.g., few-shot segmentation).

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 3.5 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 4 |
| Overall Score | 3.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection](adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](../../ACL2025/llm_nlp/zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Bilingual Zero-Shot Stance Detection](../../ACL2025/llm_nlp/bilingual_zero-shot_stance_detection.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[NeurIPS 2025\] MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention](../../NeurIPS2025/llm_nlp/monarchattention_zero-shot_conversion_to_fast_hardware-aware_structured_attentio.md)

</div>

<!-- RELATED:END -->
