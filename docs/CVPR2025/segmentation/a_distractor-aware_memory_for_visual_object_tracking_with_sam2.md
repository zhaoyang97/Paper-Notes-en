---
title: >-
  [Paper Note] A Distractor-Aware Memory for Visual Object Tracking with SAM2
description: >-
  [CVPR 2025][Segmentation][SAM2] A Distractor-Aware Memory (DAM) model is proposed for SAM2.1++, splitting the memory of SAM2 into Recent Appearance Memory (RAM, ensuring segmentation accuracy) and Distractor Resolution Memory (DRM, ensuring tracking robustness). Through an introspective update strategy, DAM detects distractors and automatically stores anchor frames, setting a new SOTA on 7 benchmarks.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "SAM2"
  - "Distractor-Aware Memory"
  - "Visual Object Tracking"
  - "Memory Management"
  - "Robustness"
date: 2026-05-08
content_hash: c438870623b24613
---

# A Distractor-Aware Memory for Visual Object Tracking with SAM2

**Conference**: CVPR 2025  
**arXiv**: [2411.17576](https://arxiv.org/abs/2411.17576)  
**Code**: [https://github.com/jovanavidenovic/DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM)  
**Area**: Segmentation / Visual Object Tracking  
**Keywords**: SAM2, Distractor-Aware Memory, Visual Object Tracking, Memory Management, Robustness

## TL;DR

A Distractor-Aware Memory (DAM) model is proposed for SAM2.1++, splitting the memory of SAM2 into Recent Appearance Memory (RAM, ensuring segmentation accuracy) and Distractor Resolution Memory (DRM, ensuring tracking robustness). Through an introspective update strategy, DAM detects distractors and automatically stores anchor frames, setting a new SOTA on 7 benchmarks.

## Background & Motivation

**Background**: Memory-based trackers (such as SAM2) have achieved SOTA performance on several benchmarks by storing recent frames in a memory buffer and localizing the target using cross-attention. SAM2 utilizes a FIFO memory containing the initial frame and the 6 most recent frames.

**Limitations of Prior Work**: Distractors (regions visually similar to the target) are a major cause of tracking failures. Existing memory management strategies only store recent frames, failing to effectively distinguish the target from distractors. When the target briefly leaves the field of view, the memory becomes filled with empty-mask frames, leading to re-detection failures.

**Key Challenge**: Accurate segmentation requires the latest target appearance (temporally correlated), whereas robust distractor handling requires anchor frames containing distractors (temporally uncorrelated). These two requirements are fundamentally different and should not be handled by the same memory strategy.

**Core Idea**: Divide memory by function into RAM (recent appearance, with temporal encoding, FIFO update) and DRM (distractor resolution, without temporal encoding, updated only when distractors are detected and tracking is reliable).

## Method

### Overall Architecture

The memory model is redesigned based on SAM2.1 without requiring any extra training. The total 6 memory slots are split in half: 3 slots for RAM (recent frames) and 3 slots for DRM (anchor frames + initial frame). A distractor detection mechanism and a reliability guard strategy based on SAM2's multi-hypothesis outputs are proposed.

### Key Designs

1. **Recent Appearance Memory (RAM) Management**:

    - Function: Stores the most recent target appearance to ensure segmentation accuracy.
    - Mechanism: Updated every $\Delta=5$ frames (preventing redundancy from highly correlated frames), always including the latest frame. A key improvement is that it is not updated when the target is absent (skipped when the predicted mask is empty), preventing empty-mask frames from contaminating the memory.
    - Design Motivation: [51] demonstrated that high-frequency updates lead to visual redundancy, reducing localization capability. Mitigating update frequency + skipping target-absent frames maintains appearance diversity.

2. **Distractor Resolution Memory (DRM) Management**:

    - Function: Stores anchor frames containing critical distractors to ensure tracking robustness and re-detection capability.
    - Mechanism: Leverages SAM2's multi-hypothesis output—SAM2 predicts 3 masks and selects the one with the highest IoU. Key finding: Before tracking fails, distractor information actually appears in the alternative masks. By calculating the bounding box area ratio of the joint region between the output mask and the alternative masks, a distractor is detected when the ratio is below $\theta_{anc}=0.7$. It is only updated when tracking is reliable (IoU > $\theta_{IoU}=0.8$ and area within median $\pm 20\%$) to prevent erroneously segmented frames from contaminating the memory.
    - Design Motivation: DRM does not use temporal encoding—since distractor resolution information should not be biased by temporal proximity and should act as a "time-agnostic prior."

3. **DiDi Dataset**:

    - Function: Distills a distractor-dense sequence subset from multiple standard benchmarks.
    - Mechanism: Uses DINOv2 features to compute the distractor ratio per frame—a frame is considered to contain distractors if the proportion of pixels outside the target region with high feature similarity to the target region exceeds 0.5. Sequences with $\ge 1/3$ of their frames containing distractors are selected.
    - Result: Eventually, 180 sequences totaling 274K frames were obtained, focusing on exposing and challenging distractor handling capabilities.

### Loss & Training

Completely training-free, implementing the DAM architecture directly by utilizing existing components of the pre-trained SAM2.1. This is because SAM2 already supports flexible configurations of variable memory length and temporal/non-temporal encoding.

## Key Experimental Results

### Main Results: SOTA Comparison on DiDi Dataset

| Method | Quality | Accuracy | Robustness |
|------|---------|----------|------------|
| TransT | 0.465 | 0.669 | 0.678 |
| SeqTrack | 0.529 | 0.714 | 0.718 |
| KeepTrack | 0.502 | 0.646 | 0.748 |
| ODTrack | 0.608 | 0.740 | 0.809 |
| Cutie | 0.575 | 0.704 | 0.776 |
| SAM2.1 | 0.649 | 0.720 | 0.887 |
| SAMURAI | 0.680 | 0.722 | 0.930 |
| SAM2.1Long | 0.646 | 0.719 | 0.883 |
| **SAM2.1++** | **0.694** | **0.727** | **0.944** |

### Ablation Study: Step-by-Step Validation of Memory Design

| Config | Quality | Accuracy | Robustness |
|------|---------|----------|------------|
| SAM2.1 (Baseline) | 0.649 | 0.720 | 0.887 |
| +Update only when target is present | 0.665 | 0.723 | 0.903 |
| +Decrease update frequency $\Delta=5$ | 0.667 | 0.718 | 0.914 |
| +DRM update only when reliable | 0.672 | 0.710 | 0.932 |
| +DRM update only when distractor detected | 0.644 | 0.691 | 0.913 |
| **Full DAM (Both conditions simultaneously)** | **0.694** | **0.727** | **0.944** |
| DAM+DRM temporal encoding | 0.669 | 0.711 | 0.925 |

### Key Findings

- **The two update conditions of DRM are both indispensable**: Updating only on distractor detection (without reliability guard) actually decreases performance (0.644 < 0.667), since erroneous segmentation during unreliable periods contaminates the DRM.
- **Not using temporal encoding is crucial for DRM**: Adding temporal encoding drops Quality by 3.6%, proving that distractor resolution information should indeed not carry temporal bias.
- **Robustness improvement is key**: Compared to the SAM2.1 baseline, Quality is improved by +7%, primarily originating from a +6% gain in Robustness (0.887 $\rightarrow$ 0.944), demonstrating that DAM effectively reduces tracking losses.
- On VOT2022, EAO reaches 0.753 (a 12% improvement over the challenge winner MS_AOT's 0.673), and on VOT2020, EAO reaches 0.729.

## Highlights & Insights

- **Functional Memory Splitting**: For the first time, tracking memory is partitioned functionally (accuracy vs. robustness), delivering a conceptually clear and elegant design. RAM uses temporal encoding because recent frames are highly relevant, while DRM omits it because distractor information should be agnostic to time.
- **Leveraging Existing Multi-Hypothesis Outputs**: In SAM2's three output masks, alternative masks contain distractor information—this is a valuable cue completely overlooked by prior work. This idea of "mining hidden information in existing outputs" is highly inspiring.
- **Zero Training Cost**: Seamlessly integrates with pre-trained SAM2 components without any additional training to achieve significant performance improvements. Extremely practical.

## Limitations & Future Work

- Distractor detection relies on a simple heuristic rule regarding bounding box area ratios, which may miss distractor types that cannot be captured by area changes.
- The DRM uses a fixed count of 3 slots, which may be insufficient to handle multiple different distractors in long sequences.
- Validated only in single-object tracking scenarios; distractor handling in multi-object scenes remains more complex.

## Related Work & Insights

- **vs SAMURAI**: SAMURAI also improves SAM2's memory management, but uses motion cues for memory selection. SAM2.1++ achieves 2% higher Quality on DiDi with a simpler design through functional splitting and introspective updates.
- **vs SAM2Long**: SAM2Long optimizes long sequences using constrained tree search but achieves on-par performance with baseline SAM2.1 in distractor scenarios, indicating that long-term memory is not equivalent to distractor resolution capability.
- **vs KeepTrack**: KeepTrack explicitly models association networks for multi-candidate detection, which leads to a complex architecture. DAM is more lightweight and achieves superior performance.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of functional memory splitting is novel, and leveraging multi-hypothesis outputs for distractor detection is highly clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensively evaluated on 7 benchmarks; ablation studies step-by-step validate every design decision.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically rigorous; every design choice is backed by experimental validation, and the contribution of the DiDi dataset is highly valuable.
- Value: ⭐⭐⭐⭐⭐ A zero-training enhancement of SAM2 tracking performance with significant improvements, setting new SOTAs on multiple benchmarks with open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](../../ICLR2026/segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [\[CVPR 2025\] SAMWise: Infusing Wisdom in SAM2 for Text-Driven Video Segmentation](samwise_infusing_wisdom_in_sam2_for_text-driven_video_segmentation.md)
- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[CVPR 2025\] MatAnyone: Stable Video Matting with Consistent Memory Propagation](matanyone_stable_video_matting_with_consistent_memory_propagation.md)

</div>

<!-- RELATED:END -->
