---
title: >-
  [Paper Note] STORM: End-to-End Referring Multi-Object Tracking in Videos
description: >-
  [CVPR 2026][Video Understanding][Referring Multi-Object Tracking] STORM is the first end-to-end multimodal large language model framework for Referring Multi-Object Tracking (RMOT). It substantially reduces reliance on R…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Referring Multi-Object Tracking"
  - "Multimodal Large Language Models"
  - "Task Composition Learning"
  - "Dataset"
date: 2026-05-08
content_hash: 590345234442351c
---

# STORM: End-to-End Referring Multi-Object Tracking in Videos

**Conference**: CVPR 2026 Findings  
**arXiv**: [2604.10527](https://arxiv.org/abs/2604.10527)  
**Code**: [https://github.com/amazon-science/storm-referring-multi-object-grounding](https://github.com/amazon-science/storm-referring-multi-object-grounding)  
**Area**: Video Understanding
**Keywords**: Referring Multi-Object Tracking, Multimodal Large Language Models, Task Composition Learning, Video Understanding, Dataset

## TL;DR
STORM is the first end-to-end multimodal large language model framework for Referring Multi-Object Tracking (RMOT). It substantially reduces reliance on RMOT-annotated data through a task composition learning strategy and introduces the high-quality STORM-Bench dataset.

## Background & Motivation

**Background**: Referring multi-object tracking requires models to track all targets matching a textual description in a video. Existing RMOT methods decompose target localization and tracking into separate modules and rely on external detectors.

**Limitations of Prior Work**: (1) Training videos for RMOT are extremely scarce; (2) existing datasets contain ambiguous annotations and are domain-restricted; (3) modular approaches struggle to interpret complex referring expressions and reason over causal or relational dependencies.

**Key Challenge**: RMOT is a complex task requiring joint visual-language understanding and temporal tracking, yet annotation costs are prohibitively high, making it infeasible to obtain sufficient training data.

**Goal**: Unify localization and tracking, eliminate dependency on external modules, and address data scarcity.

**Key Insight**: Drawing on the LLM pretraining paradigm of "learn foundational capabilities first, then fine-tune," RMOT is decomposed into two fundamental subtasks: image-level grounding and single-object tracking.

**Core Idea**: Task Composition Learning decomposes RMOT into data-rich subtasks, first acquiring grounding and tracking capabilities, then fine-tuning on a small amount of RMOT data.

## Method

### Overall Architecture
STORM adopts a LLaVA-style MLLM architecture: a ViT visual encoder extracts frame-level visual features → an MLP projector maps them to the text embedding space → a LLaMA-based LLM autoregressively generates target bounding box sequences. The output is structured text of the form: `Object 1: Frame 1: [x1,y1,x2,y2], ...`.

### Key Designs

1. **Task Composition Learning (TCL)**:

    - **Function**: Reduces reliance on large-scale RMOT annotations by decomposing the task into subtasks.
    - **Mechanism**: Stage 1 pretrains on large-scale image grounding and single-object tracking data to learn cross-modal alignment and temporal consistency; Stage 2 fine-tunes on STORM-Bench using a Chain-of-Thought training strategy that guides the model to first localize targets in the first frame and then track them across frames.
    - **Design Motivation**: RMOT annotations are extremely costly, whereas image grounding and SOT data are abundant; progressive learning effectively combines these capabilities.

2. **End-to-End Unified Architecture**:

    - **Function**: Performs localization and tracking within a single MLLM framework.
    - **Mechanism**: Bounding boxes for all targets are output directly as plain text, leveraging the reasoning capacity of pretrained language models to handle complex referring expressions. Long videos are segmented into short clips, with predicted boxes from the previous clip used as prompts to stitch trajectories across clips.
    - **Design Motivation**: Eliminating external detectors/trackers avoids information loss between modules and enables the model to learn a unified spatiotemporal representation.

3. **STORM-Bench Dataset (Bottom-Up Annotation)**:

    - **Function**: Provides high-quality training and evaluation data for RMOT.
    - **Mechanism**: Targets are first localized and descriptions are generated (using an MLLM with three distinct visual input types for verification), then an LLM composes multi-target referring expressions with secondary validation. The dataset comprises 15.7K videos, 251K images, and 200K referring expressions.
    - **Design Motivation**: Existing RMOT datasets suffer from ambiguous annotations and small scale; bottom-up annotation is more robust than top-down approaches.

### Loss & Training
Standard next-token prediction cross-entropy loss.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | STORM | Prev. SOTA | Gain |
|----------------|--------|-------|------------|------|
| RefCOCO val | Acc@0.5 | 89.1 | 88.7 (M-GPT2) | +0.4 |
| Elysium RSOT | AUC | 84.1 | 83.3 (Elysium) | +0.8 |
| STORM-Bench RMOT | HOTA | 42.9 | 37.9 (Qwen2.5-VL) | +5.0 |

### Ablation Study

| Configuration | HOTA | Notes |
|---------------|------|-------|
| Full STORM | 42.9 | Complete model |
| w/o Stage 1 Pretraining | 35.2 | Subtask pretraining contributes substantially |
| w/o CoT Reasoning | 39.6 | Chain-of-Thought improves tracking consistency |

### Key Findings
- The TCL strategy significantly reduces the need for RMOT-annotated data; training on the image grounding subtask also improves RMOT performance.
- Longer and more comprehensive prompts further boost tracking performance (AUC: 87.4 → 87.5).
- The end-to-end approach demonstrates a clear advantage on complex referring expressions; the pipeline approach of Grounding DINO combined with a tracker achieves only 31.7 HOTA.

## Highlights & Insights
- **Practicality of Task Decomposition Learning**: Decomposing complex tasks into data-rich subtasks is a general strategy for alleviating annotation bottlenecks, transferable to other video tasks requiring complex annotations.
- **Bottom-Up Annotation Pipeline**: A more robust annotation approach than top-down methods, exploiting the asymmetry that description is easier than localization.

## Limitations & Future Work
- Built on an 8B-parameter model; inference efficiency remains a deployment bottleneck.
- Segment-based processing of long videos may result in loss of tracking consistency across clip boundaries.
- Free-form text output occasionally produces malformed bounding boxes.

## Related Work & Insights
- **vs. ReferGPT**: ReferGPT augments an MLLM with a matching module; STORM is fully end-to-end.
- **vs. Elysium**: Elysium relies on top-down annotation, which introduces noise; STORM's bottom-up annotation is more reliable.

## Rating
- Novelty: ⭐⭐⭐⭐ First end-to-end MLLM framework for RMOT; the TCL strategy is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three-level evaluation (image / SOT / RMOT) is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; method description is thorough.
- Value: ⭐⭐⭐⭐ Both the dataset and the method offer substantial contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](rethinking_two-stage_referring-by-tracking_in_referring_multi-object_tracking_ma.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[ICML 2026\] STORM: Segment, Track, and Object Re-Localization from a Single Image](../../ICML2026/video_understanding/storm_segment_track_and_object_re-localization_from_a_single_image.md)

</div>

<!-- RELATED:END -->
