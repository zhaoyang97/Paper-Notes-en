---
title: >-
  [Paper Note] Locality-Aware Zero-Shot Human-Object Interaction Detection
description: >-
  [CVPR 2025][Multimodal VLM][Zero-Shot HOI Detection] This paper proposes the LAIN framework, which enhances the local fine-grained perception and interaction reasoning capabilities of CLIP representations through Locality Adapters (LA) and Interaction Adapters (IA), achieving state-of-the-art performance across various zero-shot HOI detection settings.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Zero-Shot HOI Detection"
  - "CLIP Adaptation"
  - "Locality-Awareness"
  - "Interaction Reasoning"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 193f7cb07be3e54f
---

# Locality-Aware Zero-Shot Human-Object Interaction Detection

**Conference**: CVPR 2025  
**arXiv**: [2505.19503](https://arxiv.org/abs/2505.19503)  
**Code**: [http://cvlab.postech.ac.kr/research/LAIN](http://cvlab.postech.ac.kr/research/LAIN)  
**Area**: Multimodal VLMs  
**Keywords**: Zero-Shot HOI Detection, CLIP Adaptation, Locality-Awareness, Interaction Reasoning, Vision-Language Models

## TL;DR

This paper proposes the LAIN framework, which enhances the local fine-grained perception and interaction reasoning capabilities of CLIP representations through Locality Adapters (LA) and Interaction Adapters (IA), achieving state-of-the-art performance across various zero-shot HOI detection settings.

## Background & Motivation

The goal of zero-shot HOI detection is to recognize human-object interaction categories unseen during training. Although existing methods widely exploit the generalization capability of CLIP, they face key adaptation challenges:

1. **CLIP favors global information**: Its image-level pre-training makes CLIP adept at encoding global semantics, but it fails to capture fine-grained local details in region-level tasks. For example, CLIP's judgment of "riding a bicycle" does not depend on whether a person is actually on the bicycle region, but is based on the overall scene context.
2. **Adaptation impairs generalization**: Existing methods, when adapted to CLIP for HOI detection, often exhibit unseen category performance lower than CLIP's native zero-shot performance (e.g., under UC-RF and UV settings).
3. **Lack of interaction awareness**: Knowing only the local details of objects is insufficient; understanding the **interaction patterns** between humans and objects is also crucial (e.g., distinguishing "riding" vs. "repairing" a bicycle depends on the relationship between hands and handlebars).

## Method

### Overall Architecture

LAIN is a two-stage HOI detection framework:

1. A pre-trained DETR is utilized to detect objects in the image.
2. All valid human-object pairs are constructed to generate HO tokens.
3. HO tokens and image patch tokens are processed together through the $L$ layers of the CLIP vision encoder.
4. LA and IA are inserted at the front end of each CLIP layer to enhance local and interaction awareness.
5. Finally, the cosine similarity between the HO token and text embeddings is calculated to obtain the HOI score.

### Key Designs

1. **Locality Adapter (LA)** — Enhancing CLIP's local fine-grained perception:
    - Project patch tokens $F$ to a low-dimensional space $\tilde{F} \in \mathbb{R}^{H \times W \times D_a}$ ($D_a \ll D_{clip}$).
    - Construct spatial layout embeddings $L_{i,j} = \text{FFN}([b_t; c_t; e_t])$, integrating bounding box coordinates, confidence scores, and object text embeddings.
    - Aggregate neighborhood information using **multi-scale convolutions** (with different kernel sizes $k_n$): $L^{k_n} = \text{Conv}^{k_n}(\hat{F})$, $P = \text{FFN}(L^{k_1} + ... + L^{k_{N_c}})$.
    - Inject back into the original features via a learnable parameter $\gamma_{LA}$: $F' = F + \gamma_{LA} \cdot \text{FFN}(P)$.

2. **Interaction Adapter (IA)** — Capturing human-object interaction patterns:
    - Extract human and object region features $R_i^\tau$ from the updated patch tokens $F'$ using ROIAlign.
    - **Interaction Pattern Reasoning Module (IPRM)**: Extract interaction-related context $\tilde{R}_i^\tau$ using cross-attention with learnable queries $Q$, and perform mutual attention between human and object contexts: $\hat{R}_i^h = \text{CrossAttn}(\tilde{R}_i^h, \tilde{R}_i^o, \tilde{R}_i^o)$.
    - Project the HO token as queries to extract information from interaction-aware features and update: $T_i' = T_i + \gamma_{IA} \cdot \text{FFN}([\bar{R}_i^h; \bar{R}_i^o])$.

3. **HOI Scoring and Text Matching**:
    - Text template: "A photo of a person [verb-ing] a [object]" with learnable tokens prepended.
    - HOI score: $S = \text{Sigmoid}(T_{(L)} E^\top / \tau)$; Sigmoid is adopted instead of Softmax because a single person can interact with an object in multiple ways simultaneously.
    - Detector confidence is fused during inference: $S_{infer} = S \cdot S_H^\lambda \cdot S_O^\lambda$.

### Loss & Training

- Binary focal loss is employed: $\mathcal{L} = \text{FocalBCE}(S, Y)$
- Positive sample assignment is based on IoU thresholds.
- The CLIP vision encoder is frozen, and only the adapter parameters in LA and IA are trained (parameter-efficient).

## Key Experimental Results

### Main Results

| Zero-Shot Setting | Metric (Full mAP) | LAIN | LAIN† (ViT-L) | Prev. SOTA | Gain |
|-----------|-----------------|------|---------------|----------|------|
| RF-UC | Full | **34.41** | 38.13 | 33.17 (LogicHOI) | +1.24 |
| NF-UC | Full | **33.23** | 36.22 | 31.39 (ADA-CM) | +1.84 |
| UO (Unseen Obj) | Full | **34.27** | 37.60 | 28.53 (HOICLIP) | +5.74 |
| UV (Unseen Verb) | Full | **33.12** | 37.20 | 31.09 (HOICLIP) | +2.03 |
| UC (Unseen Comp) | Full | **34.36** | 36.81 | 32.11 (CLIP4HOI) | +2.25 |
| HICO-DET Fully Supervised | Full | **36.02** | - | 35.33 (CLIP4HOI) | +0.69 |

### Ablation Study

| Configuration | Unseen | Seen | Full | Description |
|------|--------|------|------|------|
| Baseline (w/o adapters) | 24.88 | 31.06 | 30.19 | No LA/IA |
| + LA only | 27.71 | 32.55 | 31.95 | Locality awareness is effective |
| + IA only | 27.37 | 33.57 | 32.70 | Interaction reasoning is effective |
| + LA + IA | **30.50** | **34.80** | **33.95** | Best collaborative effect |
| LA: w/o visual info | 26.77 | 32.18 | 31.40 | Visual context is important |
| LA: w/o spatial layout | 26.52 | 32.07 | 31.31 | Spatial prior is important |
| LA: Local Attention | 26.46 | 32.39 | 31.56 | Inferior to convolution |
| IA: w/o IPRM | 24.32 | 32.76 | 31.57 | IPRM is critical |
| IA: w/o context extraction | 25.64 | 32.41 | 31.40 | Noise filtering is effective |

### Key Findings

- **LAs and IAs are complementary**: LAs provide fine-grained object details, which are then leveraged by IAs for interaction reasoning. The joint application yields a larger improvement than using either individually (Unseen: +5.62 vs +2.83/+2.49).
- **Adapting CLIP via prior methods impairs generalization**: Under the RF-UC and UV settings, several existing methods perform worse on unseen categories than native zero-shot CLIP.
- LAIN achieved with ViT-B outperforms BCOM† using ViT-L, indicating that methodological improvements are more crucial than simply scaling up the model.
- Under the fully-supervised setting, LAIN shows a particularly significant improvement on rare HOI categories (35.70), demonstrating its strong generalization capability.
- Multi-scale convolution is better suited for capturing neighborhood-level local information than Local/Window Attention.

## Highlights & Insights

- **Precise problem formulation**: It clearly highlights the gap between CLIP's global encoding and region-level tasks in HOI adaptation, as well as the counter-intuitive phenomenon where adaptation degrades generalization.
- **Elegant complementary design of LA + IA**: LA preserves the CLIP patch token dimension via residual connections, and IA preserves the HO token dimension, ensuring compatibility with frozen CLIP layers.
- **Parameter-efficient**: Only lightweight adapters are inserted into each CLIP layer while the main CLIP backbone remains frozen, minimizing training cost.
- Spatial layout embedding integrates detector outputs (bounding box coordinates, classes, and confidence scores) without requiring extra annotations.

## Limitations & Future Work

- It relies on the detection quality of the pre-trained DETR, where missed or false detections directly impact subsequent HOI detection.
- It is validated only on image-level HOI detection, without extension to video HOI or temporal scenarios.
- The selection of convolution kernel sizes ($\mathbb{K}$) must be manually configured, which may depend heavily on dataset characteristics.
- The fixed text templates may limit generalization to broader open-vocabulary scenarios.
- The number of unseen categories in certain zero-shot settings is limited, requiring validation on a larger scale.

## Related Work & Insights

- The limitation of CLIP's global representation in region-level tasks like HOI/segmentation is a common challenge; the multi-scale convolution + spatial layout approach of LA is transferable to other tasks.
- The interaction reasoning mechanism in IPRM shares similarities with relation networks but is much more lightweight.
- The learnable gating parameters $\gamma_{LA}, \gamma_{IA}$ ensure that the adapters do not disrupt the pre-trained representations in early stages.

## Rating

- Novelty: ⭐⭐⭐⭐ The LA + IA adapter design is novel and the problem analysis is in-depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 zero-shot settings + fully supervised + extensive ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation figures, standard mathematical formulations, and rigorous logic.
- Value: ⭐⭐⭐⭐ The CLIP adaptation paradigm offers broad reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](../../ICLR2026/multimodal_vlm/zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[ACL 2025\] RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](../../ACL2025/multimodal_vlm/rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)

</div>

<!-- RELATED:END -->
