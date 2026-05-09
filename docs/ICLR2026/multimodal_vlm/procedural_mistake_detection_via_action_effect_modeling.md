---
title: >-
  [Paper Note] Procedural Mistake Detection via Action Effect Modeling
description: >-
  [ICLR 2026][Multimodal VLM][procedural mistake detection] This paper proposes a dual-branch multimodal supervision framework for action effect modeling, combining a visual branch (object state and spatial relation features) with a text branch (GPT-4o-generated scene graphs). Learnable effect tokens distill external supervision signals, achieving state-of-the-art mistake detection on egocentric procedural videos.
tags:
  - ICLR 2026
  - Multimodal VLM
  - procedural mistake detection
  - action effect modeling
  - egocentric video
  - scene graph
  - multimodal supervision
date: 2026-05-08
content_hash: ad93354b2236ae85
---

# Procedural Mistake Detection via Action Effect Modeling

**Conference**: ICLR 2026
**arXiv**: [2512.03474](https://arxiv.org/abs/2512.03474)
**Code**: [https://wenliangguo.github.io/Mistake_Detection](https://wenliangguo.github.io/Mistake_Detection) (project page)
**Area**: Multimodal VLM
**Keywords**: procedural mistake detection, action effect modeling, egocentric video, scene graph, multimodal supervision

## TL;DR
This paper proposes a dual-branch multimodal supervision framework for action effect modeling, combining a visual branch (object state and spatial relation features) with a text branch (GPT-4o-generated scene graphs). Learnable effect tokens distill external supervision signals, achieving state-of-the-art mistake detection on egocentric procedural videos.

## Background & Motivation

**Background**: Procedural mistake detection aims to identify whether an operator has correctly executed steps from egocentric video (e.g., adding the wrong seasoning while cooking). Existing methods primarily focus on the execution process of actions (how-to-do), but neglect the consequences of those actions (what-happened-after).

**Limitations of Prior Work**: Modeling only the action process fails to distinguish cases where "the correct action was performed but the outcome was wrong." For instance, the act of "flipping" looks identical in execution, yet if the food is burnt afterward, the step is erroneous.

**Key Challenge**: Whether an action is correct depends on its outcome, which is reflected in changes to object states and spatial relations after action completion. This requires understanding the causal "before-after" relationship.

**Goal**: How can action effects (object state changes + spatial relation changes) be effectively modeled to enhance mistake detection?

**Key Insight**: Extract object state and spatial relation information from effect frames (key frames after action completion), and learn effect representations through visual and textual dual-path multimodal supervision.

**Core Idea**: Select the effect frame that best reflects the action outcome, extract visual and textual representations of object states and spatial relations from it, and distill these into learnable effect tokens via alignment learning.

## Method

### Overall Architecture
An Action Effect Modeling (AEM) module is added on top of the ActionFormer backbone: (1) effect frame selection, (2) visual branch for extracting object state/relation features, (3) text branch using GPT-4o to generate and encode scene graphs, (4) learnable effect tokens for distilling dual-path information, and (5) prompt-based mistake detection.

### Key Designs

1. **Effect Frame Sampling**:

    - Function: Select the key frame from an action segment that best reflects the action outcome.
    - Mechanism: Jointly considers semantic relevance (cosine similarity between segment features and GPT-4o description embeddings) and visual sharpness (measured via Laplacian operator), selecting the top-1 ranked frame.
    - Design Motivation: The quality of the effect frame directly affects subsequent feature extraction. The naive baseline of using the last frame yields AUC = 70.6, while this method achieves 73.8, a gain of +3.2.

2. **Visual Branch (Dual-Path)**:

    - Function: Extract object state features and spatial relation features from the effect frame.
    - Mechanism: (a) State path: Grounding DINO detects objects; an image encoder extracts RoI features that are concatenated to form $F_s$. (b) Relation path: Object position encodings are concatenated to form $F_r$. Both feature streams are mapped through separate MLPs.
    - Design Motivation: Object appearance change (state) and positional change (relation) are two independent dimensions of action effects; modeling them separately yields greater precision.

3. **Text Branch (Scene Graph)**:

    - Function: Use GPT-4o to generate scene graphs from effect frames, providing structured effect descriptions.
    - Mechanism: Scene graph $G=(V,E)$ contains object, relation, and attribute nodes, decomposed into a state subgraph and a relation subgraph. GNN encoding followed by pooling produces text-side features $t_s$ and $t_r$.
    - Design Motivation: Scene graphs provide structured semantic information complementary to visual features. Experiments show that adding the text branch improves AUC from 68.4 to 71.7.

4. **Effect-Aware Learning**:

    - Function: Distill visual and text dual-path supervision signals into learnable effect tokens.
    - Mechanism: Effect token $e$ is mapped through an MLP and aligned with visual/text features via $L_2$ loss, while visual-text contrastive alignment is applied between the two branches. The distilled effect tokens are concatenated with action features and fed into the detector.
    - Design Motivation: External models (GPT-4o, Grounding DINO) are only required during training; at inference, the learned tokens are used directly with no additional overhead.

### Loss & Training
$$L = L_{\text{seg}} + L_{\text{eff}} + L_{\text{CL}} + L_{\text{det}}$$

where $L_{\text{seg}}$ is action segmentation loss, $L_{\text{eff}}$ is the effect alignment $L_2$ loss, $L_{\text{CL}}$ is visual-text contrastive loss, and $L_{\text{det}}$ is mistake detection contrastive loss.

## Key Experimental Results

### Main Results (EgoPER Dataset)

| Method | AUC | EDA |
|--------|-----|-----|
| HF2-VAD | 59.9 | 27.1 |
| EgoPED | 62.0 | 57.0 |
| AMNAR | 68.5 | 64.4 |
| **Ours** | **73.8** | **66.7** |

### Ablation Study

| Component | AUC | EDA |
|-----------|-----|-----|
| Baseline (no AEM) | 67.6 | 65.6 |
| + Visual effect supervision | 68.4 | 66.1 |
| + Text supervision | 69.4 | 66.3 |
| + Visual + Text (no alignment) | 71.7 | 66.4 |
| **+ Aligned Visual + Text** | **73.8** | **66.7** |

### Key Findings
- Compared to AMNAR (Prev. SOTA), AUC improves by 5.3 points.
- The effect frame sampling strategy outperforms the naive last-frame baseline by 3.2 AUC.
- Spatial relation features (AUC = 72.6) contribute more than object state features (AUC = 69.9).
- Visual-text alignment yields an additional 2.1 AUC gain over simple fusion (71.7 → 73.8).
- Open-source MLLM (Qwen3-VL) for scene graph generation achieves AUC = 73.3, approaching GPT-4o (73.8).

## Highlights & Insights
- **Action Effect Modeling**: Shifting mistake detection from "whether the action was correctly performed" to "whether the action outcome is correct" represents a highly insightful change of perspective.
- **Distillation-based Design**: GPT-4o and Grounding DINO provide supervision only during training; these models are not required at inference. The effect token serves as a knowledge distillation bridge.
- **Decomposition of State vs. Relation**: Decomposing action effects into object state changes and spatial relation changes is a transferable design principle for broader causal reasoning tasks.

## Limitations & Future Work
- The effect frame assumption requires that outcomes are immediately observable after action completion, which may not hold for delayed effects (e.g., slow cooking).
- Generating scene graphs with GPT-4o incurs high cost; although not needed at inference, data preparation during training is time-consuming.
- Evaluation is limited to constrained scenarios such as kitchen operations; generalization to more complex domains (e.g., industrial assembly) remains unknown.
- The quality of Grounding DINO object detection directly affects visual branch performance.

## Related Work & Insights
- **vs. AMNAR**: The previous SOTA, which adopts an anomaly detection paradigm. The proposed method explicitly models action effects, offering greater interpretability.
- **vs. EgoPED**: An earlier method that does not model effects; the proposed method substantially outperforms it.
- **vs. ActionFormer**: Serves as the backbone network; the proposed AEM module is built on top of it.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The action effect modeling perspective is highly novel and convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets with detailed ablations, though the scenarios are limited (kitchen only).
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear and the probabilistic framework is elegant.
- Value: ⭐⭐⭐⭐ Provides a new methodology for procedural video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](../../ACL2026/multimodal_vlm/don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)
- [\[ICLR 2026\] Unified Vision-Language Modeling via Concept Space Alignment](unified_vision-language_modeling_via_concept_space_alignment.md)
- [\[ICLR 2026\] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)
- [\[ICLR 2026\] Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)
- [\[ICLR 2026\] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and.md)

</div>

<!-- RELATED:END -->
