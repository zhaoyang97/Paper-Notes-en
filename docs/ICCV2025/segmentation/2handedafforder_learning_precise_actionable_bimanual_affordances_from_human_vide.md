---
title: >-
  [Paper Note] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos
description: >-
  [ICCV 2025][Segmentation][bimanual affordance] This paper proposes an automated pipeline to extract precise bimanual affordance annotations from human activity videos, yielding the 2HANDS dataset, and trains a VLM-based 2HandedAfforder model that predicts precise object region segmentation masks for left and right hand grasps conditioned on text prompts. The approach significantly outperforms existing methods on the newly introduced ActAffordance benchmark.
tags:
  - ICCV 2025
  - Segmentation
  - bimanual affordance
  - affordance segmentation
  - VLM
  - hand-object interaction
  - egocentric video
date: 2026-05-08
content_hash: 40c01dc90658bfd2
---

# 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos

**Conference**: ICCV 2025
**arXiv**: [2503.09320](https://arxiv.org/abs/2503.09320)
**Code**: [https://sites.google.com/view/2handedafforder](https://sites.google.com/view/2handedafforder)
**Area**: Segmentation / Robotic Manipulation / Affordance
**Keywords**: bimanual affordance, affordance segmentation, VLM, hand-object interaction, egocentric video

## TL;DR

This paper proposes an automated pipeline to extract precise bimanual affordance annotations from human activity videos, yielding the 2HANDS dataset, and trains a VLM-based 2HandedAfforder model that predicts precise object region segmentation masks for left and right hand grasps conditioned on text prompts. The approach significantly outperforms existing methods on the newly introduced ActAffordance benchmark.

## Background & Motivation

**State of the Field**: Affordance grounding is a critical capability for robotic manipulation — robots must identify which regions of an object can be used for a specific task (e.g., where to grasp a bottle when pouring water). Existing methods typically rely on manually annotated datasets whose label quality resembles coarse object-part segmentation, lacking action-oriented precision.

**Limitations of Prior Work**: (a) Existing affordance datasets (IIT-AFF, AGD20K, etc.) provide imprecise annotations that often degrade into rough object-part segmentation; (b) most methods are task-agnostic, predicting generic "hotspot" regions without considering task context; (c) bimanual affordance — an important class of interactions — is entirely overlooked.

**Root Cause**: During hand-object interaction, the hand itself occludes the critical affordance region, making it difficult to extract precise contact areas directly from interaction images.

**Paper Goals**: (a) How to automatically extract precise, task-oriented bimanual affordance segmentation masks from video; (b) how to train a model that predicts the interaction regions for each hand separately, conditioned on text prompts.

**Starting Point**: Leveraging video-level hand inpainting to first remove occluding hands and obtain a complete object view, then recovering precise hand-object contact regions via mask completion.

**Core Idea**: Precise affordance masks are automatically extracted via video hand inpainting combined with mask completion, enabling a VLM-based model for text-driven bimanual affordance prediction.

## Method

### Overall Architecture

The system operates in two stages: (1) **Data extraction** — automatically generating 278K annotated affordance samples (the 2HANDS dataset) from EPIC-KITCHENS egocentric videos using hand inpainting and mask completion; (2) **Model training** — the VLM-based 2HandedAfforder network that takes an image and a task text prompt as input and outputs affordance segmentation masks for each hand along with a bimanual/unimanual classification.

### Key Designs

1. **Affordance Extraction Pipeline**:

    - Function: Automatically extract precise affordance regions on objects from human activity videos.
    - Mechanism: (a) Sparse hand-object masks from VISOR annotations are densified across full sequences via a video mask propagation network; (b) the video hand inpainting model VIDM restores hand regions to complete object appearances using 4 frames as input, with unoccluded frames providing visual cues; (c) SAM2 propagates the original object mask onto the inpainted image to obtain the complete object mask; (d) the final affordance region is defined as the complete object mask $\cap$ hand mask.
    - Design Motivation: Because the hand occludes the critical interaction region, the inpainting-plus-completion strategy elegantly circumvents the occlusion problem, yielding affordance regions more precise than manual annotation.
    - Additional Advantage: Narration text from videos is used as affordance category labels, naturally yielding 73 affordance classes and 163 object classes without any predefined taxonomy.

2. **VLM-based 2HandedAfforder Network**:

    - Function: Predict bimanual affordance masks in an image conditioned on a text prompt.
    - Mechanism: A text prompt (e.g., "pour tea from kettle") and an image are fed to a VLM (LLaVA-13B), which produces language tokens and a [SEG] token; a SAM image encoder extracts visual features; two SAM-style mask decoders independently generate affordance masks for the left and right hands respectively.
    - Design Motivation: VLMs excel at semantic reasoning but are ill-suited for pixel-level tasks, while the SAM encoder provides strong visual features — the two are complementary. The dual-decoder design naturally handles bimanual scenarios.

3. **Taxonomy Classifier (Bimanual Classification Head)**:

    - Function: Predict whether the interaction involves the left hand only, right hand only, or both hands.
    - Mechanism: An MLP applied to the output token of the left-hand mask decoder produces a three-way classification; at test time, the classification result determines which mask output(s) to use.
    - Design Motivation: Prevents spurious mask predictions for the unused hand in unimanual tasks.

### Loss & Training

- Mask prediction uses a combination of Dice Loss and Focal Cross-Entropy Loss.
- Classification prediction uses standard Cross-Entropy Loss.
- The VLM is fine-tuned with LoRA to preserve pretrained knowledge; the SAM image encoder is frozen.
- For unimanual tasks, the mask loss weight for the inactive hand is set to 0.

## Key Experimental Results

### Main Results

The ActAffordance benchmark is introduced, comprising 400 activities with human-annotated multimodal affordance masks.

| Method | IoU ↑ | Precision ↑ | Dir. HD ↓ | mAP ↑ |
|---|---|---|---|---|
| 2HandedAfforder | **0.058** | **0.130** | **202** | **0.104** |
| LISA | 0.044 | 0.050 | 255 | 0.047 |
| 2HAff-CLIP | 0.026 | 0.064 | 292 | 0.059 |
| AffordanceLLM | 0.012 | 0.013 | 225 | 0.012 |

Cropped variant (eliminating the influence of object localization):

| Method | IoU ↑ | Precision ↑ | Dir. HD ↓ | mAP ↑ |
|---|---|---|---|---|
| 2HandedAfforder | **0.086** | **0.269** | **100** | **0.240** |
| 3DOI | 0.082 | 0.224 | 109 | 0.180 |
| LISA | 0.082 | 0.122 | 130 | 0.116 |
| AffordanceLLM | 0.076 | 0.112 | 76 | 0.103 |

### Ablation Study

| Configuration | Description |
|---|---|
| AffExtract (data extraction) | Precision=0.420, IoU=0.185; validates extraction quality |
| 2HAff vs. 2HAff-CLIP | VLM variant achieves ~2× higher precision than CLIP variant, demonstrating the importance of reasoning capability |
| Ego4D generalization test | Despite no Ego4D training data, performance is comparable to or better than EPIC-KITCHENS |

### Key Findings

- **Reasoning capability is critical**: 2HAff (VLM) achieves approximately 2× higher precision than 2HAff-CLIP, confirming that VLM semantic reasoning substantially outperforms CLIP feature matching.
- The data extraction quality aligns reasonably with human annotation (Precision 0.42), while the low IoU (0.185) reflects the multimodal nature of affordances — multiple plausible interaction regions exist for the same task.
- Robot demonstrations confirm that predicted regions can be directly used for grasp planning, outperforming generic object segmentation.

## Highlights & Insights

- **Hand inpainting for affordance extraction**: Using video hand inpainting combined with mask completion to bypass occlusion is a novel and generalizable idea — any video containing hand-object interaction can serve as a data source.
- **Narration as natural category labels**: Avoiding a predefined fixed category taxonomy allows affordance classes to emerge naturally from task descriptions, achieving broader coverage.
- **Dual-decoder architecture**: Elegantly decomposes the bimanual problem into two parallel mask predictions augmented by a classification head for selection — a clean and principled design.
- **Transferability to robotic tasks**: Affordance regions can be directly converted into 6-DOF grasp point clouds, validated on a physical Tiago++ robot.

## Limitations & Future Work

- Overall IoU metrics remain low (below 0.1 for all methods), reflecting the inherent difficulty of the task.
- Training data is limited to kitchen scenes (EPIC-KITCHENS); generalization to other environments requires additional data.
- For tasks requiring precise force control (e.g., unscrewing a cap), region segmentation alone is insufficient.
- The multimodal nature of affordances is not addressed — multiple plausible grasp locations may exist for the same task, yet the model predicts only one.

## Related Work & Insights

- **vs. LISA**: LISA performs whole-object reasoning segmentation without considering precise affordance regions; the proposed method achieves finer region predictions through training on dedicated affordance data.
- **vs. VRB/ACP**: These methods predict task-agnostic hotspots or heatmaps; the proposed approach enables task-aware precise mask prediction via text prompts.
- **vs. AffordanceLLM**: Although AffordanceLLM also employs an LLM, its training data (AGD20K) provides coarser annotations compared to the automatically extracted 2HANDS dataset.

## Rating

- Novelty: ⭐⭐⭐⭐ Hand inpainting for affordance extraction is a novel idea; bimanual affordance is pioneered.
- Experimental Thoroughness: ⭐⭐⭐⭐ New benchmark, multiple baselines, ablations, and real-robot validation — comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; pipeline visualization is effective.
- Value: ⭐⭐⭐⭐ Direct practical applicability to robotic manipulation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)
- [\[ICCV 2025\] ReferEverything: Towards Segmenting Everything We Can Speak of in Videos](refereverything_towards_segmenting_everything_we_can_speak_of_in_videos.md)
- [\[ICCV 2025\] Open-World Skill Discovery from Unsegmented Demonstration Videos](open-world_skill_discovery_from_unsegmented_demonstration_videos.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)

<!-- RELATED:END -->
