---
title: >-
  [Paper Note] Explainable Forensics of Manipulated Segments in Untrimmed Long Videos
description: >-
  [ICML 2026][Video Generation][AI-generated video detection] Ours proposes the task of **temporal localization and explainable analysis of AI-generated segments in long videos**…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "AI-generated video detection"
  - "temporal localization"
  - "explainability"
  - "long video forensics"
  - "boundary-awareness"
date: 2026-05-08
content_hash: 0fc8cd94799fe0ef
---

# Explainable Forensics of Manipulated Segments in Untrimmed Long Videos

**Conference**: ICML 2026  
**arXiv**: [2606.02402](https://arxiv.org/abs/2606.02402)  
**Code**: TBD  
**Area**: AIGC Detection / AI Safety / Video Forensics  
**Keywords**: AI-generated video detection, temporal localization, explainability, long video forensics, boundary-awareness

## TL;DR
Ours proposes the task of **temporal localization and explainable analysis of AI-generated segments in long videos**, introducing the **TASLE large-scale dataset** and the **two-stage MSLoc baseline method**. By employing boundary-aware proposal generation and MLLM-based refinement, it achieves precise localization and explainable reasoning for manipulated segments in mixed real-fake videos.

## Background & Motivation

**Background**: Current AI-generated video detection methods primarily focus on binary classification (real vs. fake) of short video clips. Representative works such as DeMamba and BusterX++ are trained and evaluated on independent video segments lasting only a few seconds. Furthermore, existing AIGC detection datasets (e.g., GenVideo, GenVidBench) consist almost entirely of short or fully generated videos, **lacking annotations for mixed scenarios** where real and generated content coexist.

**Limitations of Prior Work**: Real-world video tampering often follows a "**sparse embedding**" pattern—a small amount of AI-generated content is interspersed within predominantly real video, rather than the entire video being fabricated. In this setting, current short-video detectors face two major challenges: (1) **Loss of boundary information**: Models are insensitive to subtle anomalies at the junction of real and fake content, failing to capture the smooth transition from real to generated sequences; (2) **Long-tail interference**: Large amounts of irrelevant real content introduce noise. Directly applying sliding-window inference across long videos is computationally prohibitive, while uniform sampling dilutes critical boundary cues.

**Key Challenge**: The design assumption of short-video detectors—that "each input clip is either entirely real or entirely fake"—collapses in mixed long-video scenarios. Existing MLLM temporal localization models (e.g., Trace) possess reasoning capabilities but often fail on long videos (tens of seconds) as uniform sampling causes critical frames to be overwhelmed by irrelevant content.

**Goal**: Establish the new task of "temporal localization and explainable analysis of AI-generated segments in long videos," accompanied by a corresponding large-scale dataset and baseline method.

**Key Insight**: A core observation is that manipulated segments in long videos typically appear in the form of "**boundaries**"—subtle inconsistencies at the transition between real and generated content serve as the strongest discriminative cues. The authors propose using "multi-class classification" instead of "binary classification" to capture this boundary information, designing a two-stage pipeline: a lightweight model for coarse proposals (focusing on boundaries), followed by an MLLM for fine localization and explanation (understanding semantics).

**Core Idea**: Transform long-video forensics from a single-stage clip-level binary classification task into a two-stage framework consisting of boundary-aware proposals and MLLM refinement. This explicitly models anomalies at real-fake junctions through boundary classification and adaptive sampling.

## Method

### Overall Architecture
MSLoc adopts a typical "coarse-to-fine" two-stage design:
- **Stage 1 (MSLoc-PG: Proposal Generation)**: Performs an efficient initial scan of long videos to quickly filter coarse locations of suspicious manipulated regions. A sliding window strategy (2-second window, 8-frame sampling) is used to avoid computational explosion from frame-by-frame processing. The key innovation is refomulating detection as a **four-class classification task** (real, fake, real-to-fake boundary, fake-to-real boundary), enabling the model to explicitly learn boundary features.
- **Stage 2 (MSLoc-PR: Refinement Module)**: Receives proposals from the first stage and refines each candidate region. An **adaptive sampling strategy** decomposes proposals into "boundary regions" ($\phi\%$ at both ends) and "event regions" (the internal segment). These are subjected to dense sampling (to capture subtle inconsistencies) and sparse sampling (to extract semantic context), respectively.

### Key Designs

1.  **Boundary-Aware Four-Class Classification**:
    - **Function**: Replaces traditional binary classification during the proposal stage to explicitly capture anomalies at transitions.
    - **Mechanism**: Defines a label space $\mathcal{Y} = \{y_{\text{real}}, y_{\text{fake}}, y_{\text{r2f}}, y_{\text{f2r}}\}$, where $y_{\text{r2f}}$ and $y_{\text{f2r}}$ represent real-to-fake and fake-to-real boundaries. Within each 2-second sliding window, 8 frames are sampled uniformly, and the model is optimized via cross-entropy loss $\mathcal{L}_{\text{ce}} = -\frac{1}{N_b} \sum_{i=1}^{N_b} \log(p_{i, t_i})$. The model learns to capture boundary signals between consecutive frames rather than just judging global veracity.
    - **Design Motivation**: Transitions in long videos often involve visual discontinuities (sudden motion changes, lighting mismatches). The four-class objective encourages sensitivity to temporal changes in these "transition zones"—ablation studies show F1Loc improves from 54.0 to 64.8.

2.  **Adaptive Sampling + Difference-Aware Modeling (DAM + EAM)**:
    - **Function**: Applies differentiated processing strategies to different parts of a proposal during the refinement stage.
    - **Mechanism**: Each coarse proposal $P_i$ is divided into boundary regions (the outer $\phi\%$, typically $\phi=20\%$) and an event region. Boundary regions undergo **dense sampling** (32 frames for $N_b=16$) to capture fine inter-frame anomalies; event regions undergo **规sparse sampling** (8 frames) for high-level semantics. Boundary features are compressed via Q-Former, specifically calculating inter-frame variation and invariant tokens using pixel-wise similarity priors. Event features utilize joint spatial-temporal compression.
    - **Design Motivation**: Transitions often occupy a small fraction of a proposal but contain the highest information density. Event regions primarily serve to generate explanatory text and do not require fine-grained frame processing. This asymmetric sampling improves localization accuracy (2-3% over uniform sampling) and reduces MLLM computation through compression; out-of-domain F1Loc improves by 17.6%.

3.  **Anomaly-Aware Loss**:
    - **Function**: Guides the MLLM to focus on genuine generation artifacts rather than making generic statements when generating explanations.
    - **Mechanism**: Injects three special "anomaly-aware tokens" into the MLLM input, encoded in a format understandable by the LLM. The output embeddings of these tokens predict anomaly categories (e.g., "boundary start explanation," "object anomaly") through a classification head, optimized by cross-entropy loss $\mathcal{L}_{\text{AA}}$.
    - **Design Motivation**: Since AI-generated content in the TASLE dataset shows high similarity to reference frames, models are prone to hallucinating false explanations. Anomaly-aware tokens force the reasoning process to bind with specific anomaly categories, improving the truthfulness of explanations—Table 3 shows RQ (explainability score) increases from 3.79 to 3.99 with this loss.

## Key Experimental Results

### Main Results

| Method | Data | F1Det | F1Loc | RQ |
|------|------|-------|-------|-----|
| D3 | Seen AIGC | 34.6 | 31.1 | ✗ |
| BusterX++* (FT) | TASLE | 33.6 | 36.4 | ✗ |
| DeMamba* (Binary) | TASLE | 54.9 | 54.0 | ✗ |
| MSLoc-PG (4-Class) | TASLE | 67.5 | 64.8 | ✗ |
| Trace* + DeMamba | TASLE | 55.7 | 59.1 | 3.45 |
| Trace* + MSLoc-PG | TASLE | 69.0 | 70.9 | 3.91 |
| **MSLoc (Full)** | **TASLE** | **70.1** | **72.2** | **4.05** |

### Out-of-Domain Evaluation

| Setting | MSLoc-PG | MSLoc | Gain |
|------|----------|-------|------|
| Seen Generation Types | 62.7 F1Det | 67.0 F1Det | +4.3% |
| Unseen Generation Types | 50.0 F1Loc | 62.8 F1Loc | +25.6% |
| Out-of-Domain (TVSum) | 38.7 F1Loc | 56.3 F1Loc | +45.5% |

### Key Findings
- Four-class classification yields significant gains over binary classification: MSLoc-PG (67.5 F1Det) vs. DeMamba (54.9), proving the efficacy of explicit boundary modeling.
- The two-stage design offers a clear advantage in generalization—MSLoc improves by 25.6% on unseen AIGC types.
- Boundary sampling is critical—Table 4 shows RQ drops from 4.01 to 3.80 when boundary sampling frames are reduced from 16 to 8.
- Computational efficiency is manageable: Compared to Trace (9 min), MSLoc (12 min) adds only 33% inference overhead but improves F1Loc from 37.3 to 63.8.

## Highlights & Insights
- **Boundary Classification Paradigm Shift**: Moving from "short video binary classification" to "long video four-class classification" is a simple yet powerful improvement. Traditional AIGC detection ignores temporal cues at real-fake junctions; this is transferable to other temporal anomaly detection tasks (deepfake detection, action anomaly recognition).
- **Progressive Refinement in Two-Stage Architecture**: While MSLoc follows a classic coarse-to-fine design, its innovation lies in Stage 1 being "boundary-aware coarse localization" rather than mere "candidate generation," while Stage 2 achieves boundary refinement and explainability via adaptive sampling and multimodal reasoning. This is particularly effective for long-tail problems.
- **Inspiration from Adaptive Sampling**: The idea of dense sampling for boundaries and sparse sampling for events reflects a deep understanding of the "problem structure." This logic can be applied to other long-sequence processing tasks (long-document comprehension, video event localization).

## Limitations & Future Work
- **Error Propagation in Cascaded Architecture**: MSLoc uses a cascaded two-stage design; misses in the first stage cannot be recovered later. Future work will explore end-to-end joint training.
- **Rapid Evolution of Generation Artifacts**: Current detection capability relies on the visibility of visual artifacts in AI-generated content. As video generation technology advances, artifacts from new generators will become more subtle—the authors commit to continuously updating the TASLE dataset.
- **Multimodal Cue Fusion**: Currently, only visual information is utilized. Future research could explore multimodal cues such as audio-visual synchronization, lip-syncing of speakers, and background consistency.

## Related Work & Insights
- **vs. Short-Video AIGC Detection** (DeMamba, BusterX++): Existing works focus on binary classification and assume input clips are independent. Ours extends to mixed long-video scenarios, introducing four-class classification and boundary modeling.
- **vs. Video Temporal Localization** (Trace, TimeChat): These models were originally designed for semantic event localization and perform poorly when directly applied to generation artifact detection (Trace alone achieves only 37.5 F1Loc). MSLoc addresses this pain point through its two-stage design and adaptive sampling.
- **vs. Explainability Methods** (FakeShield, IVY-FAKE): These methods provide natural language explanations but are based on short videos. TASLE provides finer annotation granularity with dual-level explanations (boundary-level and object-level).

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Systematically proposes a new task for long-video AIGC localization and explainable analysis, achieving a comprehensive upgrade over traditional short-video methods through four-class classification and two-stage design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Introduces a 12.5K large-scale dataset, detailed ablation analysis, and multiple generalization evaluation scenarios with robust baselines.
- Writing Quality: ⭐⭐⭐⭐  The problem statement is clear, the technical solution is logically sound, and the methodology chapters are well-structured.
- Value: ⭐⭐⭐⭐⭐  Both the dataset and the methodology have high practical value, directly addressing real-world video forensics needs. Application prospects include content moderation, judicial forensics, and autonomous driving safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](../../CVPR2026/video_generation/activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[ICML 2026\] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)
- [\[NeurIPS 2025\] Scaling RL to Long Videos](../../NeurIPS2025/video_generation/scaling_rl_to_long_videos.md)
- [\[ICML 2026\] Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories](rays_as_pixels_learning_a_joint_distribution_of_videos_and_camera_trajectories.md)
- [\[ICML 2026\] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)

</div>

<!-- RELATED:END -->
