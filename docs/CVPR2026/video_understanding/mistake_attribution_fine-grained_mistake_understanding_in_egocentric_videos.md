---
title: >-
  [Paper Note] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos
description: >-
  [CVPR 2026][Video Understanding][mistake attribution] This paper introduces the Mistake Attribution (MATT) task, which attributes action mistakes in egocentric videos along three dimensions: semantic (which component of the instruction was violated), temporal (at which frame the point of no return, PNR, occurs), and spatial (which region in the PNR frame contains the error). A data engine called MisEngine automatically constructs large-scale mistake samples from existing action datasets, and a unified Transformer model, MisFormer, simultaneously addresses all three attribution sub-tasks, surpassing task-specific SOTA methods across multiple benchmarks.
tags:
  - CVPR 2026
  - Video Understanding
  - mistake attribution
  - egocentric video
  - semantic role labeling
  - spatiotemporal localization
  - instruction alignment
date: 2026-05-08
content_hash: 7f05b6051f613461
---

# Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos

**Conference**: CVPR 2026
**arXiv**: [2511.20525](https://arxiv.org/abs/2511.20525)
**Code**: [https://yayuanli.github.io/MATT](https://yayuanli.github.io/MATT)
**Area**: Video Understanding
**Keywords**: mistake attribution, egocentric video, semantic role labeling, spatiotemporal localization, instruction alignment

## TL;DR

This paper introduces the Mistake Attribution (MATT) task, which attributes action mistakes in egocentric videos along three dimensions: semantic (which component of the instruction was violated), temporal (at which frame the point of no return, PNR, occurs), and spatial (which region in the PNR frame contains the error). A data engine called MisEngine automatically constructs large-scale mistake samples from existing action datasets, and a unified Transformer model, MisFormer, simultaneously addresses all three attribution sub-tasks, surpassing task-specific SOTA methods across multiple benchmarks.

## Background & Motivation

1. **Background**: AI-assisted systems in physical environments (e.g., cooking guidance, assembly instruction) need to understand mistakes made by humans during instruction execution. Existing methods focus primarily on mistake detection—determining whether a step is erroneous—or provide coarse-grained error categories such as "missing step" or "action deviation."
2. **Limitations of Prior Work**: Coarse-grained detection fails to inform users *which part of the instruction was not correctly executed* (semantic dimension), *when the mistake became irreversible* (temporal dimension), and *which spatial region in the PNR frame contains the error* (spatial dimension). For instance, if the instruction is "pick up the hammer" but the user picks up a bolt, existing methods can only report "an error occurred," without identifying that the *object* role is wrong, the error manifests at frame 17, or the erroneous region is the bolt within a bounding box.
3. **Key Challenge**: Constructing fine-grained mistake datasets is extremely difficult—genuine mistakes become increasingly rare as annotators gain experience, while artificially injected mistakes introduce visual bias. Existing mistake datasets (EgoPER: 599 samples; Assembly101: 707 samples) are two orders of magnitude smaller than general action recognition datasets.
4. **Goal**: (a) How to automatically construct large-scale mistake datasets with semantic–temporal–spatial triplet annotations; (b) how to address all three attribution tasks within a single unified model.
5. **Key Insight**: Semantic Role Labeling (SRL) is applied to parse action descriptions into structured role groups, followed by cross-matching across samples in existing action recognition datasets. An instruction text such as "pick up the sieve" is paired with a video of "pick up the pan," automatically generating semantic attribution labels while inheriting PNR timestamps and hand/object spatial annotations from the original datasets.
6. **Core Idea**: Automatically construct mistake samples from large-scale action corpora via semantic role cross-matching, and perform joint semantic–temporal–spatial attribution with a unified Transformer.

## Method

### Overall Architecture

The input consists of an instruction text $T$ (e.g., "cut the apple") and a user execution video $V$. The output is a triplet: (1) error labels $\{y_r\}$ for each semantic role, (2) PNR frame timestamp $t_{PNR}$, and (3) an error region bounding box $B_{t_{PNR}}$ in the PNR frame. The system comprises two main components: the MisEngine data engine for automatic training data construction, and the MisFormer model for attribution inference.

### Key Designs

1. **MisEngine Data Engine**:

    - **Function**: Automatically constructs training samples with three-dimensional attribution annotations from existing action recognition datasets.
    - **Mechanism**: A three-step pipeline — (1) AllenNLP SRL parses each action description into semantic role groups (e.g., predicate "Pick up," object "the sieve"); (2) cross-sample comparison identifies role-level mismatches between pairs of action descriptions, yielding $C=|\mathcal{R}|^2$ mismatch categories (wrong predicate, wrong object, both wrong, both correct); (3) action descriptions and their corresponding videos are sampled from each mismatch category as erroneous attempts. Semantic labels are derived directly from cross-matching; temporal labels inherit PNR frame annotations; spatial labels inherit hand/object bounding boxes from the original datasets. This process yields 257K samples from Ego4D and 221K from EPIC-KITCHENS, surpassing the largest existing mistake dataset by two orders of magnitude.
    - **Design Motivation**: Circumvents the scarcity of genuine mistakes and the visual bias introduced by injected errors by reformulating mistake construction as a combinatorial problem over existing data.

2. **MisFormer Feature Extraction and Projection**:

    - **Function**: Extracts shared multimodal features from video and text.
    - **Mechanism**: InternVideo2's text encoder separately encodes each semantic role substring to obtain $F_R^T \in \mathbb{R}^{|\mathcal{R}| \times d}$, while the video encoder extracts $F^V \in \mathbb{R}^{L \times K \times d}$. A projection block $\mathcal{P}$ (2-layer Transformer decoder, no causal masking) first applies self-attention over role text features to exchange inter-role information, then cross-attention over video features to inject visual context, producing projected features $F_R^{T'} \in \mathbb{R}^{|\mathcal{R}| \times d}$.
    - **Design Motivation**: InternVideo2 pretraining aligns text and video features in a shared embedding space; the projection block further adapts these representations for the mistake understanding task.

3. **Three Attribution Heads (Semantic / Temporal / Spatial)**:

    - **Function**: Respectively output semantic role error labels, PNR frame localization, and error region bounding boxes.
    - **Mechanism**:
        - **Semantic head**: Each role's projected feature $F_r^{T'}$ is passed through an FFN + sigmoid for binary classification of whether the role is erroneous; trained with BCE loss.
        - **Temporal head**: Frame-level video features $F^V$ are aggregated via 2-layer self-attention into $F^{V'} \in \mathbb{R}^{L \times d}$; a 2-layer Transformer decoder (with $F^{V'}$ as query and $F_R^{T'}$ as key/value) generates per-frame probability distributions; the argmax identifies the PNR frame; trained with cross-entropy loss.
        - **Spatial head**: Cross-attention weights from the final layer of the projection block corresponding to the PNR frame are extracted and concatenated with projected text features; two self-attention layers generate a spatial saliency map, which is upsampled and concatenated with the PNR frame RGB to form a 4-channel input; a lightweight CNN regresses bounding box coordinates; trained with Huber loss.
    - **Design Motivation**: At inference time, the temporal and spatial heads are activated via a gating mechanism only when the semantic head detects at least one erroneous role, reducing unnecessary computation.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_S + \mathcal{L}_T + \mathcal{L}_{spatial}$, where $\mathcal{L}_S$ is binary cross-entropy (semantic), $\mathcal{L}_T$ is cross-entropy (temporal, computed only on erroneous samples), and $\mathcal{L}_{spatial}$ is Huber loss (spatial).

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | MisFormer | Prev. SOTA | Gain |
|--------|------|------|-----------|----------|------|
| EPIC-KITCHENS-M | Semantic Attribution | F1@0.5 | **83.89** | 77.23 (ChatGPT-4o) | +6.66% |
| Ego4D-M | Semantic Attribution | F1@0.5 | **56.24** | 50.95 (ChatGPT-4o) | +5.29% |
| Ego4D-M | Temporal Attribution | MAE(s) | **0.638** | 0.816 (EgoT2) | −21.81% |
| Ego4D-M | Spatial Attribution | mIoU | **59.21** | 49.88 (MediaPipe-U) | +18.70% |
| Ego4D-M | Mistake Detection | F1@0.5 | **57.55** | 15.62 (EgoPED) | +41.93% |

### Ablation Study

| Configuration | Semantic F1 | Temporal MAE(s) | Spatial mIoU | Detection F1 |
|------|---------|-------------|-----------|---------|
| MisFormer (full) | 56.24 | 0.438 | 59.21 | 57.55 |
| Replace with LaViLa backbone | 49.16 | 0.561 | 51.37 | 46.05 |
| Remove projection block $\mathcal{P}$ | 51.34 | 0.457 | 55.43 | 52.75 |
| Remove temporal attribution training | 51.29 | 0.623 | 57.78 | 57.46 |
| Replace attention map with GradCAM | 55.52 | 0.482 | 55.03 | 57.51 |

### Key Findings
- The InternVideo2 backbone (multimodal pretraining) is critical for MATT; replacing it with LaViLa causes consistent degradation across all sub-tasks.
- The projection block $\mathcal{P}$ is indispensable—raw text embeddings are insufficient to capture subtle discrepancies between instructions and video.
- Attribution of the *object* role is consistently easier than the *predicate* role, indicating that fine-grained action understanding remains a core challenge in egocentric video.
- Training from scratch on the small-scale EgoPER dataset yields poor results, but pretraining on EPIC-KITCHENS-M followed by fine-tuning achieves competitive performance.

## Highlights & Insights
- **MisEngine's "zero-cost annotation" design** is particularly elegant: by applying semantic role cross-matching, existing action recognition data is transformed into mistake understanding data, with all three-dimensional annotations inherited at no additional labeling cost. This paradigm of "simulating mistakes by combining correct samples rather than collecting genuine errors" is broadly transferable to other tasks requiring scarce negative samples.
- **Unified model vs. ensemble of specialists**: MisFormer achieves superior or competitive performance relative to task-specific SOTA methods across all four sub-tasks within a single unified model, with only 41M+ projection head parameters and high runtime efficiency (spatial head at 68.9 FPS).
- Formalizing mistake understanding as a three-dimensional attribution framework over instruction–execution discrepancies provides AI assistants with interpretable and actionable feedback.

## Limitations & Future Work
- The current framework supports only short instructions (predicate + object); real-world long-form and multi-step instructions require richer role sets.
- Spatial attribution underperforms dedicated hand–object interaction detectors (SSDA: mIoU 64.54 vs. 59.21); integrating task-specific object detection priors could improve results.
- The data engine assumes mistakes arise solely from role cross-matching, and cannot capture continuous deviations such as "degree errors" (e.g., slicing too thick).
- Current pretraining relies on general video–language alignment objectives; designing pretraining objectives specifically targeting mistake understanding may yield further gains.

## Related Work & Insights
- **vs. EgoPED / AMNAR**: These methods treat mistake detection as an out-of-distribution detection problem, do not leverage mistake supervision signals, and fail in large-scale multi-activity scenarios. MisFormer scales through large-scale supervised learning enabled by MisEngine.
- **vs. MistScene**: MistScene generates natural language explanations without structured attribution and lacks instruction alignment. MATT provides structured semantic–temporal–spatial triplets.
- **vs. ChatGPT-4o**: MisFormer surpasses this closed-source commercial model by 6.66% on semantic attribution, demonstrating the advantage of task-specific design over general-purpose large models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Proposes a new task definition, data engine, and unified model, with complete three-dimensional contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers four sub-tasks with human validation and ablations; spatial attribution comparisons could be further deepened.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, figures are excellent, and the logical progression is smooth.
- Value: ⭐⭐⭐⭐ — Provides a complete methodology for mistake feedback in egocentric AI assistants; the data engine paradigm is broadly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] LensWalk: Agentic Video Understanding by Planning How You See in Videos](lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)

</div>

<!-- RELATED:END -->
