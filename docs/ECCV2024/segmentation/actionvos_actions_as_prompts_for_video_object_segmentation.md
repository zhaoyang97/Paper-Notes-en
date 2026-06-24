---
title: >-
  [Paper Note] ActionVOS: Actions as Prompts for Video Object Segmentation
description: >-
  [ECCV 2024][Segmentation][Referring Video Object Segmentation] ActionVOS is proposed—a new setting for Referring Video Object Segmentation that uses human action narratives as additional linguistic prompts. It generates pseudo-labels via a parameter-free action-aware labeling module and designs an action-guided focal loss to suppress false positives, reducing the false segmentation of inactive objects by 35.6% mIoU on VISOR, while improving the segmentation of state-changing…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Referring Video Object Segmentation"
  - "Action Prompts"
  - "Active Object Segmentation"
  - "Egocentric Vision"
  - "Pseudo-labels"
date: 2026-05-08
content_hash: 01c36694a6d64cb5
---

# ActionVOS: Actions as Prompts for Video Object Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.07402](https://arxiv.org/abs/2407.07402)  
**Code**: [https://github.com/ut-vision/ActionVOS](https://github.com/ut-vision/ActionVOS)  
**Area**: Video Understanding / Video Object Segmentation  
**Keywords**: Referring Video Object Segmentation, Action Prompts, Active Object Segmentation, Egocentric Vision, Pseudo-labels  

## TL;DR
ActionVOS is proposed—a new setting for Referring Video Object Segmentation that uses human action narratives as additional linguistic prompts. It generates pseudo-labels via a parameter-free action-aware labeling module and designs an action-guided focal loss to suppress false positives, reducing the false segmentation of inactive objects by 35.6% mIoU on VISOR, while improving the segmentation of state-changing objects by 3.0% mIoU on VOST/VSCOS.

## Background & Motivation
Existing Referring VOS (RVOS) tasks mainly rely on static attributes (such as object name, color) to describe target objects. While sufficient in simple scenes, when the scene becomes complex—for example, when multiple instances of the same category exist (multiple knives, multiple bowls) or the object undergoes a state change (fingernails being painted blue from pink)—static attributes alone cannot accurately distinguish target objects from background objects. In egocentric vision, understanding human activities requires distinguishing which objects are actually involved in the interaction (active objects), which serves as the foundation for machine understanding of human actions.

## Core Problem
How to segment only active objects truly related to human actions in egocentric videos without segmenting redundant objects present in the scene but not involved in the interaction? The challenges lie in: (1) Existing datasets lack annotations indicating whether an object is involved in an action; (2) Static attributes easily fail under redundant instances or state changes.

## Method

### Overall Architecture
The input consists of a video clip, an action narrative (e.g., "cut apple"), and N object names, and the output is the segmentation mask for each object. Unlike traditional RVOS, ActionVOS introduces the action narrative as an additional linguistic prompt. The framework consists of three parts: (1) ActionVOS model—adding a classification head on top of the RVOS model to distinguish positive (active) and negative (inactive) samples; (2) Action-aware labeling module—automatically generating pseudo-labels using existing annotations; (3) Action-guided focal loss—reducing the impact of false positives in pseudo-labels.

### Key Designs
1. **ActionVOS Model (Classification Head)**: A linear classification head `nn.Linear(256,1)` is added to RVOS models like ReferFormer to predict the probability of each object's involvement in the action. During inference, a threshold $\theta=0.75$ is used for decision: segmentation results are retained if the probability $\ge \theta$, and an all-zero mask is outputted if $< \theta$. This is a minimalist yet effective modification.

2. **Action-aware Labeling Module**: To address the lack of active/inactive annotations in existing datasets, action narratives, semantic segmentation annotations, and hand-object segmentation annotations are leveraged to automatically label three categories of objects as positive samples: (a) objects mentioned in the action narrative; (b) objects inside the hand-object mask; (c) objects intersecting with the hand-object bounding box. The third category is designed loosely and introduces false positives, but it can capture indirectly related objects such as containers and contents. The entire module contains no trainable parameters.

3. **Action-Guided Focal Loss**: A pixel-level weighting mechanism is designed to handle false positives in pseudo-labels. Weight assignment follows three rules: (a) objects both mentioned in the narrative and inside the hand-object bbox > objects satisfying only one condition ($\lambda_{pos}=5$); (b) objects mentioned in the narrative or in contact with the hand > objects only within the bbox intersection ($\lambda_{nar}=\lambda_{h-obj}=2$); (c) negative objects are assigned a high weight ($\lambda_{neg}=5$) within their mask area to penalize false segmentations of negative samples.

### Loss & Training
- Based on focal loss, introducing the aforementioned action-guided weight $W$, with focal loss parameters $\alpha=0.25, \gamma=2$
- The classification head uses cross-entropy loss with a weight coefficient of 2
- Fine-tuned from the best checkpoint pre-trained on Refer-YouTube-VOS
- The text encoder RoBERTa is retrained during training
- The linguistic prompt uses a natural sentence format: e.g., "knife used in the action of cut apple"
- Hand-object masks are only utilized to generate pseudo-labels during training and are not required during inference

## Key Experimental Results

| Dataset | Metric | ActionVOS (RF-R101+AP) | RVOS Baseline | Change |
|--------|------|------------------------|----------|------|
| VISOR | p-mIoU | 65.4 | 67.7* | -2.3 (Slight) |
| VISOR | n-mIoU | 19.0 | 54.2 | **-35.2** (Large reduction in false segmentation) |
| VISOR | gIoU | 70.9 | 43.8 | **+27.1** |
| VISOR | Acc | 82.4 | 59.1 | **+23.3** |
| VOST | mIoU | 32.3 | 29.3 | +3.0 |
| VSCOS | mIoU | 49.4 | 46.4 | +3.0 |

*The p-mIoU of RVOS represents the upper bound (treating all objects as positive samples)

Effective across different backbones: Swin-L (gIoU 70.3, Acc 80.7), Video-Swin-B (gIoU 70.6, Acc 81.2).

### Ablation Study
- **Classification Head**: After adding the classification head, gIoU improved from 61.8 to 70.9, where threshold $\theta=0.75$ represents the optimal trade-off.
- **Prompt Format**: Natural sentence format (+sAction) > comma-concatenated (+, Action) > object name only (NoAction); fine-tuning the text encoder yields further improvement.
- **Action-Guided Focal Loss**: Compared to standard focal loss, gIoU improves from 70.6 to 70.9, n-mIoU decreases from 20.5 to 19.0, and Acc increases from 82.1 to 82.4. Under visualization, the model correctly distinguishes different objects held by each hand.
- **Models trained without action prompts** are misled by negative pseudo-labels, resulting in a p-mIoU of only 56.3 vs 65.4.
- **Generalization to Unseen Actions**: Performance on unseen action categories still outperforms RVOS and HOS baselines.

## Highlights & Insights
- **Minimalist yet Elegant Design**: The core of the proposed method (labeling module + loss weights) is parameter-free, requiring only one additional linear classification head, which is plug-and-play for any RVOS model.
- **Thoughtfully-defined "Positive Samples"**: Not only includes objects mentioned in the narrative but also extends to held tools, containers, and contents, fully modeling the hierarchical nature of human-object interaction.
- **Clear Design Logic of Action-guided Weights**: Elegantly handles pseudo-label noise through a multi-tier hierarchy of priorities (narrative+contact > narrative > contact > bbox intersection > negative).
- **No Additional Annotation Required**: Entirely leverages existing annotations of action narratives, semantic segmentations, and hand-object segmentations.
- **Cross-scenario Action Understanding**: Different actions in the same scene generate distinct segmentation results, demonstrating a genuine understanding of hand-object interaction semantics.

## Limitations & Future Work
- **Dependence on Dense Annotations**: Requires hand-object segmentation masks for training, preventing direct application on datasets lacking such annotations.
- **Poor Performance when Hands are Out-of-view**: Failure cases show that when the hand is outside the frame, the model struggles to determine object involvement.
- **Sensitivity to Object Name Ambiguity**: Vague terms like "package" can lead to erroneous segmentations.
- **Difficulty handling Drastic Shape Changes**: Actions causing major changes in object shape (e.g., "divide dough") remain challenging.
- **Upper Bound in Pseudo-label Quality**: The loose definition of the labeling module inevitably introduces false positives, although focal loss partially mitigates this effect.
- **Scalability to Open World**: The authors mentioned that incorporating more action-object relationships and reducing reliance on dense annotations are promising future directions.

## Related Work & Insights
- **vs ReferFormer (RVOS)**: RVOS does not distinguish between active and inactive objects, segmenting all mentioned objects. ActionVOS achieves selective segmentation of active objects through action prompts and the classification head, improving gIoU by 27.1 points.
- **vs HOS (Hand-Object Segmentation)**: HOS only segments hands and held objects, which is too narrow to cover indirectly related objects such as containers and contents. ActionVOS offers a more comprehensive definition of positive samples.
- **vs MeViS**: MeViS focuses on segmentation described by motion expressions but does not involve the distinction between active and inactive objects. The core difference of ActionVOS lies in introducing action semantics to filter target objects.

## Related Work & Insights
- **Transferability of Action Prompts**: The idea of action prompts can be transferred to other tasks, such as object discovery in action recognition, target object identification in robotic grasping, and interacting object localization in video QA.
- **Minimalist Head + Thresholding Scheme**: This scheme can serve as a general design pattern for task switching—applicable to any scenario requiring transitioning from "all segmentation" to "selective segmentation".

## Rating
- **Novelty**: ⭐⭐⭐⭐ The setting of actions as VOS prompts is novel, though the method itself is relatively simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets + multiple backbones + detailed ablations + evaluation on unseen categories.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition, highly informative figures and tables, and thorough discussion on the definition of positive samples.
- **Value**: ⭐⭐⭐⭐ Solid push for egocentric vision understanding, with highly compatible plug-and-play methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](../../ICCV2025/segmentation/referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[ECCV 2024\] General and Task-Oriented Video Segmentation](general_and_task-oriented_video_segmentation.md)
- [\[ECCV 2024\] SOS: Segment Object System for Open-World Instance Segmentation With Object Priors](sos_segment_object_system_for_open-world_instance_segmentation_with_object_prior.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)

</div>

<!-- RELATED:END -->
