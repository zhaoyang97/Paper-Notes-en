---
title: >-
  [Paper Note] From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations
description: >-
  [AAAI 2026][Robotics][Image Manipulation Localization] This paper proposes BoxPromptIML, a weakly supervised image manipulation localization (IML) framework based on coarse-grained bounding box annotations. It leverages…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Image Manipulation Localization"
  - "Weak Supervision"
  - "Knowledge Distillation"
  - "Memory Guidance"
  - "SAM"
date: 2026-05-08
content_hash: 3d219aade2a9f16a
---

# From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations

**Conference**: AAAI 2026
**arXiv**: [2511.20359](https://arxiv.org/abs/2511.20359)
**Code**: [https://github.com/vpsg-research/BoxPromtIML](https://github.com/vpsg-research/BoxPromtIML)
**Area**: Robotics
**Keywords**: Image Manipulation Localization, Weak Supervision, Knowledge Distillation, Memory Guidance, SAM

## TL;DR
This paper proposes BoxPromptIML, a weakly supervised image manipulation localization (IML) framework based on coarse-grained bounding box annotations. It leverages a frozen SAM teacher model to convert rough bounding boxes into high-quality pseudo-masks, and trains a lightweight student model via a memory-guided gated fusion module (MGFM), achieving performance comparable to or surpassing fully supervised methods with an annotation cost of only 7 seconds per image.

## Background & Motivation

The IML field faces a fundamental **annotation cost vs. localization accuracy** trade-off:

- **Fully supervised methods** (TruFor, PIM, Mesorch, etc.): rely on pixel-level mask annotations, averaging **23 minutes** per image, which limits dataset scale and practical deployment.
- **Weakly supervised methods** (WSCL, etc.): use only image-level labels (4 seconds/image), but lack spatial localization capability, achieving an F1 of only 0.239.
- **The gap between the two**: a solution is needed that retains spatial localization information while substantially reducing annotation cost.

The authors quantify the time cost of different annotation strategies through a rigorous user study (10 volunteers annotating 100 images), finding that **coarse bounding box annotation requires only 7 seconds per image**—reducing cost by over 98% compared to full supervision while preserving critical spatial information.

**Core Idea**: Exploit SAM's zero-shot segmentation capability to "upgrade" low-cost box annotations into high-quality pseudo-masks, then train a lightweight student model via knowledge distillation that can perform independent inference without any prompts, further enhanced by a memory-guided mechanism.

## Method

### Overall Architecture

BoxPromptIML adopts a teacher-student knowledge distillation paradigm:
1. **Teacher model**: Frozen SAM, receiving the input image and coarse bounding box to generate high-quality pseudo-masks.
2. **Student model**: A lightweight network (TinyViT backbone) that receives only the original image and learns to predict manipulated regions.
3. **Core module**: The Memory-Guided Gated Fusion Module (MGFM) enhances the student model's feature fusion capability.

### Key Designs

1. **Teacher Model: Prompt-Based High-Fidelity Mask Generation**

    - **Function**: Converts coarse bounding box annotations into fine-grained pixel-level pseudo-masks.
    - **Mechanism**: Exploits the zero-shot segmentation capability of pretrained SAM; given image $I$ and bounding box $B$, generates binary mask $M_{teacher}$.
    - **Design Motivation**: SAM possesses strong general-purpose segmentation capability and can recover fine boundaries from minimal spatial cues; freezing the teacher avoids fine-tuning overhead.

2. **Memory-Guided Gated Fusion Module (MGFM)**

    - **Function**: Enhances multi-scale feature fusion in the student model by combining real-time observations with historical memory prototypes.
    - **Mechanism**: Inspired by the human concept of "collective unconscious," the module incorporates a dual guidance design:
      * **Gated Integration (GI)**: Generates a gating map $G_i = \sigma(\text{Conv}(F'_i))$ for each of the 4 scale features extracted by TinyViT, producing fused feature $F_{fused}$ via weighted aggregation.
      * **Memory Bank Guidance**: Maintains a learnable memory bank storing prototype manipulation patterns $A_{mem}$.
      * **Dual-guided refinement**: $A_{final} = \alpha(A'_{base} \odot G_{avg}) + (1-\alpha) \cdot A_{mem}$
    - **Design Motivation**: Single-image analysis may miss subtle manipulation traces; the memory bank provides cross-sample "historical experience" as a prior. Unlike standard attention mechanisms, the memory bank decouples knowledge aggregation from network weights, offering a more stable global prior.

3. **Coarse-Grained Annotation Strategy**

    - **Function**: Designs an annotation paradigm intermediate between pixel-level masks and image-level labels.
    - **Mechanism**: Annotators need only draw a coarse bounding box around the manipulated region.
    - **Design Motivation**: A 7-second vs. 23-minute annotation time difference, without sacrificing spatial localization information.

### Loss & Training

Standard binary cross-entropy loss is used for pseudo-supervised distillation:
$$\mathcal{L}_{loss} = BCE(A_{refined}, \hat{M}_{teacher})$$

- The student model predicts $A_{refined} \in [0,1]^{H \times W}$; the teacher pseudo-mask $M_{teacher}$ serves as soft supervision.
- At inference, only the student model is used, requiring no prompts or bounding boxes.
- Training set: a mixture of CASIAv2, Coverage, Columbia, and NIST16.

## Key Experimental Results

### Main Results

**Comparison with fully supervised methods (F1, threshold 0.5, 20 epochs)**:

| Method | Supervision | IND Avg. | OOD Avg. |
|--------|-------------|----------|----------|
| PSCC-Net | Full | 0.535 | 0.334 |
| TruFor | Full | 0.538 | 0.236 |
| Mesorch | Full | 0.544 | 0.219 |
| SparseViT | Full | 0.562 | 0.255 |
| PIM | Full | 0.648 | 0.357 |
| **Ours** | **Weak (Box)** | **0.619** | **0.285** |

**Comparison with weakly supervised methods**:

| Method | NC16 | C1 | Col | Cov | Avg. |
|--------|------|-----|------|------|------|
| WSCL | 0.111 | 0.140 | 0.524 | 0.180 | 0.239 |
| SCAF (scribble) | 0.226 | 0.530 | 0.442 | 0.400 | 0.400 |
| **Ours (box)** | **0.618** | **0.552** | **0.903** | **0.403** | **0.619** |

### Ablation Study

| Configuration | Params (M) | FLOPs (G) | Key Metric |
|---------------|-----------|-----------|------------|
| Mesorch | 85.8 | 124.9 | IND=0.544 (full supervision) |
| PIM | 152.5 | 682.9 | IND=0.648 (full supervision) |
| **Ours** | **5.5** | **1.4** | IND=0.619 (weak supervision) |
| SCAF | 27.57 | 35.39 | IND=0.400 (weak supervision) |

**Social network robustness test**:

| Method | No Compression | Facebook | WeiBo | WeChat | WhatsApp |
|--------|---------------|----------|-------|--------|----------|
| WSCL | 0.349 | 0.124 | 0.133 | 0.066 | 0.123 |
| Mesorch | 0.703 | 0.671 | 0.655 | 0.583 | 0.677 |
| Ours | 0.552 | 0.532 | 0.536 | 0.477 | 0.530 |

### Key Findings

- **Generalization**: Fully supervised methods (TruFor, PSCC-Net) show declining OOD performance as training epochs increase (overfitting), whereas the proposed method shows steady OOD improvement (10ep: 0.253 → 20ep: 0.285).
- **Extreme efficiency**: 5.5M parameters + 1.4G FLOPs, the most lightweight among all methods (1/28 of PIM's parameters, 1/488 of its FLOPs).
- **Fast convergence**: Competitive performance is achieved in only 20 epochs, without the need for full convergence as required by SCAF.
- **Annotation cost**: 7 seconds/image vs. 23 minutes/image for full supervision, reducing cost by over 98%.

## Highlights & Insights

- **Innovation in "coarse-to-fine" weak supervision paradigm**: The first IML pipeline combining box annotations, SAM pseudo-masks, and knowledge distillation.
- **Regularization effect of the memory bank**: The memory bank not only provides historical priors but also acts as a strong regularizer against overfitting, explaining the method's consistent improvement in OOD settings.
- **High practical utility**: Low annotation cost + lightweight model + fast convergence + no prompt required at inference.
- **In-depth analysis of overfitting in fully supervised methods**: Reveals that pixel-level mask annotations may introduce semantic bias.

## Limitations & Future Work

- Performance under social network compression scenarios remains below fully supervised methods such as Mesorch (e.g., WeChat: 0.477 vs. 0.583).
- The input resolution is relatively low (224×224 vs. 512×512 for other methods), potentially limiting fine-grained localization.
- The impact of memory bank size and update strategy has not been thoroughly investigated.
- SAM as a teacher model has inherent limitations—it may fail to generate accurate pseudo-masks for highly subtle manipulations (e.g., color adjustments).
- The effect of varying bounding box annotation quality (very coarse vs. relatively precise) on final performance has not been evaluated.

## Related Work & Insights

- SAM's role as a "universal teacher" for general-purpose segmentation is worth drawing upon in other domains.
- The memory bank + gating mechanism design in MGFM is generalizable to tasks that rely on prior knowledge, such as anomaly detection and defect inspection.
- The box annotation → pseudo-mask → distillation paradigm can be transferred to annotation-expensive domains such as medical image segmentation and remote sensing change detection.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2026/robotics/sapave_active_perception_manipulation_vla_roboti.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](../../ICCV2025/robotics/selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[AAAI 2026\] H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation](h-gar_a_hierarchical_interaction_framework_via_goal-driven_observation-action_re.md)
- [\[AAAI 2026\] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](affordance-guided_coarse-to-fine_exploration_for_base_placem.md)

</div>

<!-- RELATED:END -->
