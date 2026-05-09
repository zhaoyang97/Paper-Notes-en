---
title: >-
  [Paper Note] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs
description: >-
  [CVPR 2026][Multimodal VLM][video segmentation] This paper proposes the SPARROW framework, which injects temporal referential consistency via Target-Specific Tracked Features (TSF) and stabilizes pixel-level localization through dual-prompt (BOX+SEG) initialization. As a plug-and-play module, SPARROW consistently improves performance across three video MLLM baselines on six benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - video segmentation
  - MLLM grounding
  - temporal consistency
  - dual-prompt
  - referring video object segmentation
date: 2026-05-08
content_hash: 8e6c75a539bc9c18
---

# SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs

**Conference**: CVPR 2026
**arXiv**: [2603.12382](https://arxiv.org/abs/2603.12382)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: video segmentation, MLLM grounding, temporal consistency, dual-prompt, referring video object segmentation

## TL;DR
This paper proposes the SPARROW framework, which injects temporal referential consistency via Target-Specific Tracked Features (TSF) and stabilizes pixel-level localization through dual-prompt (BOX+SEG) initialization. As a plug-and-play module, SPARROW consistently improves performance across three video MLLM baselines on six benchmarks.

## Background & Motivation

**Temporal drift in video MLLMs**: Existing video MLLMs rely on static textual grounding tokens (e.g., [SEG]) to indicate objects to be segmented. However, [SEG] provides only semantic cues about "what to do," without encoding how an object's position or appearance evolves over time. The model must infer motion and appearance changes entirely from visual cues, leading to spatial drift, identity switching, and inconsistent segmentation.

**Unstable first-frame initialization**: The [SEG] token carries semantic information but no spatial prior; consequently, the first-frame mask frequently misaligns with the target, and such errors propagate and accumulate throughout the sequence. Once drift begins, object identity switching and referential inconsistency ensue.

**Common limitations of prior methods**: Methods such as VideoGLaMM, UniPixel, and GLUS rely on per-frame semantics and propagated masks rather than sequence-level referential cues, lacking explicit mechanisms for temporal identity maintenance.

## Method

### Overall Architecture
SPARROW augments a baseline video MLLM with two complementary modules: (1) Target-Specific Tracked Feature (TSF), which injects temporally aligned referential features during training to teach the model identity persistence; and (2) Dual-Prompt Grounding, which jointly decodes [BOX] and [SEG] tokens for coarse-to-fine spatial localization and semantic segmentation.

The architecture consists of: dual-branch visual encoder (spatial SigLIP + temporal InternVideo2) → V→L adapter → LoRA-finetuned LLM → L→V adapter (BOX/SEG) → SAM2 pixel decoder. All newly introduced modules are plug-and-play and do not modify the underlying backbone.

### Key Designs

1. **Target-Specific Tracked Feature (TSF)**:
    - **Function**: Injects temporally aligned reference object features during training, enabling the model to learn cross-frame identity persistence.
    - **Mechanism**: Given a text query, GroundingDINO detects the object in one frame → CLDTracker propagates detections across the sequence → K-means clustering ($K=4$) selects representative samples in the joint visual-spatial feature space → samples are encoded as TSF tokens $Z_{\text{TSF}}$ and concatenated to the LLM input. TSF is disabled by default at test time (no external detector/tracker required).
    - **Design Motivation**: TSF provides supervision signals during training that convey "what this object looks like across different frames," allowing the model to internalize temporal referential ability. Offline precomputation decouples the heavy modules from the training loop, and K-means selection ensures diverse appearance representation.

2. **Dual-Prompt Grounding (BOX + SEG)**:
    - **Function**: The LLM simultaneously emits [BOX] and [SEG] tokens; the former provides a spatial prior while the latter provides semantic segmentation.
    - **Mechanism**: The [BOX] embedding $e_{\text{BOX}}$ conditions a lightweight regression head—a class-agnostic proposer (Deformable-DETR) built on SAM2's Hiera features—to generate 300 candidate boxes. Language-conditioned filtering is then applied via cross-attention $A_i = \text{softmax}((W_q e_{\text{BOX}})(W_k F_i)^T/\sqrt{d})$, followed by bounding box coordinate refinement. The [SEG] embedding $e_{\text{SEG}}$ is fed together with the filtered box $\hat{b}_i$ into the SAM2 prompt encoder to produce the final mask.
    - **Design Motivation**: The coarse spatial prior from [BOX] constrains the search space for [SEG], stabilizing first-frame initialization and enabling drift correction. The independent scoring mechanism naturally supports multi-instance queries (e.g., "two players"). Re-issuing [BOX]+[SEG] at arbitrary frames enables drift correction without an external tracker.

3. **Two-Stage Training Strategy**:
    - **Function**: Stage 1 trains TSF injection (multimodal adapters + LoRA); Stage 2 trains BOX prompting (proposer pretraining → filtration head fine-tuning).
    - **Mechanism**: Stage 1 trains on 30,646 videos / 45,231 QA pairs with $\mathcal{L}_{total} = \mathcal{L}_{CE} + \mathcal{L}_{BCE} + \mathcal{L}_{DICE}$, updating only the V→L adapter, L→V SEG adapter, and LLM LoRA. Stage 2 first pretrains the class-agnostic proposer on COCO/Objects365/OpenImages/V3Det, then fine-tunes the filtration head with $\mathcal{L}_{filter} = \lambda_{cls}\mathcal{L}_{BCE} + \lambda_{box}(\mathcal{L}_{\ell_1} + \mathcal{L}_{GIoU})$.
    - **Design Motivation**: The two-stage decoupling allows Stage 1 to focus on temporal and semantic alignment, and Stage 2 to focus on spatial precision, avoiding multi-objective conflicts through progressive training.

### Loss & Training
- Stage 1: $\mathcal{L}_{CE}$ (semantic alignment) + $\mathcal{L}_{BCE} + \mathcal{L}_{DICE}$ (mask supervision)
- Stage 2 proposer pretraining: $\mathcal{L}_{obj} + \lambda_1\mathcal{L}_{\ell_1} + \lambda_2\mathcal{L}_{GIoU}$
- Stage 2 filter fine-tuning: IoU > 0.5 as positive samples, < 0.2 as negative samples; $\lambda_{cls}=1.0,\ \lambda_{box}=2.0$
- TSF dataset: 30,646 videos + 45,231 QA pairs, unified from HC-STVG, VID-Sentence, A2D Sentences, LaSOT, MeViS, GOT-10k, and Ref-SAV

## Key Experimental Results

### Main Results (RVOS task adapted to three baselines)

SPARROW is applied as a plug-and-play module to three state-of-the-art video MLLMs—VideoGLaMM, UniPixel, and GLUS—yielding consistent and significant improvements across six benchmarks including MeViS, Ref-DAVIS17, and Ref-YouTube-VOS.

| Baseline → +SPARROW | Improvement |
|---------------------|-------------|
| VideoGLaMM → +SPARROW | Consistent gains in temporal consistency and spatial precision |
| UniPixel → +SPARROW | Significant reduction in identity switching |
| GLUS → +SPARROW | Improved first-frame localization stability |

### Ablation Study

| Component | Effect |
|-----------|--------|
| Baseline (no TSF, no Dual-Prompt) | Baseline |
| + TSF (training-time) | Significant improvement in temporal consistency |
| + Dual-Prompt (BOX+SEG) | Significant improvement in spatial precision |
| + TSF + Dual-Prompt | **Best overall** |
| TSF applied at inference | Further marginal improvement (with additional overhead) |

### Key Findings
- TSF is disabled by default at test time after training—the model has internalized temporal referential ability and does not depend on external trackers.
- The spatial prior from [BOX] in Dual-Prompt yields the largest gains in first-frame stability, reducing error cascades caused by misinitialization.
- Consistent improvements across three architecturally distinct baselines validate the universality of the modular design.
- The independent scoring mechanism naturally handles multi-instance scenarios (e.g., "two dogs") without requiring additional annotations.

## Highlights & Insights
- The TSF design of **train-time injection, test-time removal** is elegant—offline tracking data is used to teach the model temporal awareness, incurring no additional deployment overhead.
- The coarse-to-fine BOX+SEG dual-prompt paradigm establishes a new direction for precise localization in video MLLMs.
- The plug-and-play design allows immediate application to any existing video MLLM, lowering the barrier to adoption.

## Limitations & Future Work
- Offline TSF data construction depends on the quality of GroundingDINO and CLDTracker; detection or tracking failures introduce noisy supervision.
- The Stage 2 proposer pretraining relies on large-scale detection datasets, which may hinder reproducibility in resource-constrained settings.
- Evaluation is limited to RVOS and GCG tasks; the effectiveness of the approach on video QA, moment retrieval, and other tasks remains to be verified.
- The K-means value of $K=4$ is empirically chosen; videos of varying complexity may require different values of $K$.

## Related Work & Insights
- **VideoGLaMM**: Performs per-frame SAM decoding via [SEG] tokens; SPARROW augments this with temporal and spatial enhancements.
- **Artemis**: Inspires the idea of injecting tracked features through TSF.
- **Groma**: Inspires the BOX-prompt grounding approach for images; SPARROW extends this to the video domain.
- **Takeaway**: The "grounding" and "tracking" capabilities of video MLLMs can be independently enhanced through plug-and-play modules, without redesigning the underlying architecture from scratch.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of TSF training-time injection and Dual-Prompt is novel; the plug-and-play modular design is impactful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across three baselines × six benchmarks with clear modular ablations.
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis, detailed method description, and clear illustrations.
- Value: ⭐⭐⭐⭐ Provides a general-purpose enhancement for temporal consistency and spatial precision in video MLLMs.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatiotemporal_collaborative_network.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_codegrounded_visual_stem_perception_fo.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)

<!-- RELATED:END -->
