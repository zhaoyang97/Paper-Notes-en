---
title: >-
  [Paper Note] AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes
description: >-
  [ICML 2026][Video Understanding][Audio-Visual Tracking] Proposes the AVTrack dataset and AVTracker baseline for Audio-Visual Instance Segmentation and Tracking (AVIS) in complex human-centric scenarios. By defining 8 cha…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Audio-Visual Tracking"
  - "Instance Segmentation"
  - "Human-centric"
  - "Multi-modal Reasoning"
  - "Dataset Benchmark"
date: 2026-05-08
content_hash: e6a15547768c5496
---

# AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes

**Conference**: ICML 2026  
**arXiv**: [2606.02724](https://arxiv.org/abs/2606.02724)  
**Code**: https://github.com/FudanCVL/AVTrack (Yes)  
**Area**: Video Understanding  
**Keywords**: Audio-Visual Tracking, Instance Segmentation, Human-centric, Multi-modal Reasoning, Dataset Benchmark

## TL;DR

Proposes the AVTrack dataset and AVTracker baseline for Audio-Visual Instance Segmentation and Tracking (AVIS) in complex human-centric scenarios. By defining 8 challenging conditions, a high-difficulty evaluation benchmark is constructed. A three-stage divide-and-conquer framework (ASR segment aggregation → local speaker localization → global identity association) is designed, outperforming existing state-of-the-art methods by approximately 8 percentage points in the HOTA metric.

## Background & Motivation

**Background**: Audio-visual speaker tracking aims to locate and track active speakers using auditory and visual cues. Recently, Audio-Visual Segmentation (AVS) has evolved from pixel-level segmentation to instance-level segmentation and tracking (AVIS). Representative methods like AVISM utilize Mask2Former and VITA to achieve memory-efficient instance-level audio-visual segmentation under window attention.

**Limitations of Prior Work**: Existing datasets and benchmarks have significant limitations—early speaker tracking datasets (AV16.3, CAV3D) were collected in controlled laboratory environments with simple scenes; AVA-ActiveSpeaker only provides bounding boxes and lacks cross-frame identity consistency; the AVSBench series primarily consists of 5-10 second short clips, failing to evaluate long-range temporal modeling; although AVISeg extended the duration to about 60 seconds, the scenes remain simple, lacking complex conditions such as camera motion, occlusion, and position changes, particularly underperforming in human-centric scenarios.

**Key Challenge**: Existing benchmarks lean toward evaluating static audio-visual co-occurrence rather than truly testing the model's robust spatio-temporal modeling and cross-modal reasoning capabilities in complex dynamic scenes. Simplified evaluation environments mask the true limitations of methods.

**Goal**: (1) Construct a high-quality AVIS evaluation benchmark for complex human-centric scenes; (2) Provide a modular and extensible strong baseline method to support follow-up research.

**Key Insight**: The authors systematically define 8 complex audio-visual scene conditions (occlusion, position change, background switching, camera motion, multiple instances, multi-turn speaking, audio-visual inconsistency, and scale dynamics). These serve as filtering criteria to construct the dataset from diverse video sources (TV series, movies, vlogs, animation, variety shows, interviews, stage performances), ensuring scene diversity and challenge.

**Core Idea**: Construct a high-difficulty AVIS benchmark dataset, AVTrack, by strictly defining complex scene standards, and propose a three-stage divide-and-conquer baseline, AVTracker, which organically combines ASR-driven dynamic window aggregation, VLM local reasoning, and global identity association.

## Method

### Overall Architecture

AVTracker adopts a divide-and-conquer strategy, decomposing human-centric AVIS into three stages. The input consists of synchronized video frame sequences and audio streams, and the output is the global instance trajectory for each speaker (including frame-by-frame masks). Workflow: Stage 1 transcribes audio via ASR and aggregates them into semantically complete Speaker Chunks based on speaker embedding similarity; Stage 2 utilize a VLM within the local time window of each chunk to associate speech content with visible characters, combined with SAM3 to generate frame-by-frame instance masks, forming Local Tracklets; Stage 3 associates local tracklets of the same speaker across segments via a global reasoner to output complete global speaker trajectories.

### Key Designs

1.  **Speaker Chunks Aggregation**:
    - **Function**: Merges fragmented short transcriptions from ASR into semantically complete speaker chunks as the basic units for subsequent local processing.
    - **Mechanism**: Use Whisper to transcribe audio into timestamped text segments $\mathcal{C} = \{c_i = (t_i^s, t_i^e, x_i)\}$. For each segment, speech separation (e.g., via MossFormer2) is optionally performed to obtain an enhanced signal $\hat{\mathbf{a}}_i$, followed by extracting speaker embeddings $\mathbf{e}_i = \mathcal{E}(\hat{\mathbf{a}}_i)$ using an ECAPA-TDNN encoder. Adjacent segments are merged into a chunk when the cosine similarity $\text{sim}(c_i, c_{i+1}) > \tau$, preserving complete semantic context. Compared to fixed windows, dynamic windows maintain complete semantic units and temporal continuity.
    - **Design Motivation**: Processing short ASR segments independently is inefficient and costly; merging adjacent segments of the same speaker reduces the number of local windows and improves global association efficiency.

2.  **VLM-driven Local Speaker Localization (Local Window Process)**:
    - **Function**: Precisely associates speech content with visible characters in the video within the timeframe of each speaker chunk, generating frame-by-frame instance masks.
    - **Mechanism**: Convert the time boundaries of chunk $s_k$ into frame indices $f_k^s, f_k^e$. Use Qwen3-VL as a local reasoner $\mathcal{R}^{local}$, taking speech text $x_k$ and video frames as input to predict the active speaker's bounding box $\mathbf{b}_{\mathcal{R}}^{(f)}$. Simultaneously, SAM3 generates candidate person detection boxes and masks for each frame. VLM predictions are aligned with SAM3 detections through IoU maximization: $\mathbf{b}^{(f)} = \arg\max_{\mathbf{b} \in \mathcal{B}_{\text{SAM3}}^{(f)}} \text{IoU}(\mathbf{b}_{\mathcal{R}}^{(f)}, \mathbf{b})$. The frame with the largest mask area is selected as the keyframe for subsequent global association.
    - **Design Motivation**: Leveraging the linguistic-visual reasoning capability of VLMs to establish a semantic bridge between speech content and visual characters is more robust than traditional audio feature matching; SAM3 provides high-quality instance masks, making the two complementary.

3.  **Global Identity Association (Global Window Process)**:
    - **Function**: Aggregates local tracklets scattered across different time windows into coherent global trajectories based on speaker identity.
    - **Mechanism**: Collect keyframes $\mathcal{F}_{\text{key}} = \{I_{f_k^{\text{key}}}\}_{k=1}^{K}$ from all local tracklets. A global reasoner $\mathcal{R}^{global}$ (also VLM-based) groups keyframes based on character appearance, establishing an identity mapping $\mathcal{G}: p \mapsto \{k_1, k_2, \ldots\}$. Finally, all local tracklets belonging to the same identity $p$ are merged into a global trajectory $\mathcal{T}_p = \bigcup_{k \in \mathcal{G}(p)} \mathcal{T}_k^{\text{local}}$, with empty masks inserted for unobserved frames to maintain temporal continuity.
    - **Design Motivation**: The same speaker may appear in multiple discontinuous segments; local processing alone cannot establish identity consistency across segments. Global reasoning compensates for long-range association capabilities.

## Key Experimental Results

### Main Results

Comparison of VIS and AVIS methods on the AVTrack benchmark (all values in percentage):

| Method | Type | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ |
|------|------|--------|--------|--------|--------|--------|
| VITA | VIS | 9.70 | 10.54 | 9.35 | 12.32 | 1.91 |
| LBVQ | VIS | 10.29 | 11.77 | 9.36 | 12.87 | 1.98 |
| CAVIS | VIS | 11.46 | 12.10 | 10.07 | 12.95 | 1.96 |
| AVISM | AVIS | 20.84 | 23.22 | 19.53 | 26.57 | 3.95 |
| ACVIS | AVIS | 20.60 | 22.59 | 19.66 | 26.23 | 4.23 |
| AVTrackFormer | AVIS | 21.47 | 22.51 | 20.26 | 26.41 | 4.11 |
| **AVTracker** | **AVIS** | **29.08** | **31.18** | **28.47** | **34.55** | **16.20** |

### Ablation Study

| Config | Description | HOTA | DetA | AssA | IDF1 | MOTA |
|------|------|------|------|------|------|------|
| Base | Whisper-large + Qwen3-VL-8B | 28.85 | 31.75 | 27.39 | 34.45 | 16.39 |
| M1 | Whisper-small (Speech model downscaled) | 25.19 | 27.33 | 24.25 | 29.92 | 14.88 |
| M2 | Qwen3-VL-4B (VLM downscaled) | 24.47 | 25.85 | 24.37 | 28.86 | 14.48 |
| M3 | Both models downscaled | 24.01 | 25.49 | 23.69 | 28.47 | 13.52 |
| M4 | VLM → Replaced by Face Detection | 23.62 | 24.80 | 21.31 | 27.16 | 11.03 |
| S1 | + SepFormer Separation | 28.41 | 30.81 | 27.54 | 33.65 | 15.99 |
| S2 | + MossFormer2 Separation | 29.08 | 31.18 | 28.47 | 34.55 | 16.20 |
| C1 | W/o Local Segment Compression | 16.88 | 18.34 | 16.33 | 19.99 | 9.34 |
| C2 | Fixed Window instead of Dynamic | 27.45 | 29.57 | 26.64 | 32.97 | 13.49 |

### Key Findings

- VIS methods achieve HOTA scores below 12 on AVTrack, indicating that purely visual cues are insufficient in complex human-centric scenes. Introducing audio doubles the AVIS performance, yet it remains suboptimal (approx. 20-21).
- AVTracker improves approximately 8 HOTA points over the strongest AVIS baseline. Its core advantage stems from VLM-driven cross-modal reasoning and the local-global divide-and-conquer strategy.
- Local segment compression is critical: removing it causes HOTA to plummet from 24.01 to 16.88 (-7.13), suggesting that compact local representations are key to the scalability of global associations.
- Speech separation requires sufficiently high quality for positive gains: MossFormer2 improves HOTA by 0.23, while SepFormer decreases it by 0.44. Low-quality separation introduces noise and temporal misalignment.
- Model scale is sensitive for both modalities: Reducing VLM from 8B to 4B drops HOTA by 4.38; reducing the speech model from large to small drops it by 3.66.

## Highlights & Insights

- **Text as an Audio-Visual Semantic Bridge**: Instead of directly matching audio and visual features, ASR converts audio to text, and VLM then associates text semantics with visual characters. This indirect alignment is more robust in complex scenes than end-to-end audio-visual feature matching. This approach is transferable to other cross-modal alignment tasks.
- **Dynamic vs. Fixed Windows**: Dividing processing windows based on semantic completeness (speaker chunk boundaries) rather than mechanical fixed-duration segmentation avoids truncating semantic units. This design philosophy is applicable to any video understanding task requiring temporal segmentation.
- **Dataset Design Methodology**: Systematically defining 8 complex conditions and filtering videos accordingly is a "definition-first, data-second" methodology that serves as a valuable reference for constructing other benchmarks.

## Limitations & Future Work

- AVTrack is only released as a test set (871 videos) without training data, which limits the fair evaluation of training-based methods.
- AVTracker relies on several large pre-trained models (Whisper-large, Qwen3-VL-8B, SAM3), resulting in high inference costs and difficulty for real-time deployment.
- The current baseline combines modules in a cascaded manner, where errors accumulate across stages (ASR error → chunk aggregation error → local localization error → global association error). An end-to-end differentiable unified framework might further improve performance.
- Speech overlap remains a bottleneck—performance is harmed if speech separation quality is insufficient, necessitating more robust multi-speaker solutions.
- Future work could explore memory-augmented agentic reasoning and dynamic scene graph representations to handle more complex long-range audio-visual understanding.

## Related Work & Insights

- **AVISM** (CVPR 2025): An AVIS method combining Mask2Former + VITA, using window attention for memory-efficient instance-level audio-visual segmentation.
- **COMBO** (2024): A cross-modal fusion method that explicitly models bilateral audio-visual relationships.
- **SAM3** (2025): Segment Anything with Concepts, providing general instance segmentation capabilities, used here as a mask sampler.
- **Qwen3-VL**: A vision-language model used as the core reasoning engine, capable of speaker-speech association in a zero-shot setting.
- **OVIS / MOSE**: Video instance segmentation benchmarks for occlusion/complex scenes; AVTrack extends similar challenge definitions into the audio-visual dimension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](../../AAAI2026/video_understanding/r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)
- [\[ICML 2026\] Unified Multimodal Visual Tracking with Dual Mixture-of-Experts](unified_multimodal_visual_tracking_with_dual_mixture-of-experts.md)
- [\[ICML 2026\] RELO: Reinforcement Learning to Localize for Visual Object Tracking](relo_reinforcement_learning_to_localize_for_visual_object_tracking.md)
- [\[CVPR 2026\] Drift-Resilient Temporal Priors for Visual Tracking](../../CVPR2026/video_understanding/drift-resilient_temporal_priors_for_visual_tracking.md)
- [\[NeurIPS 2025\] PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling](../../NeurIPS2025/video_understanding/prefm_online_audio-visual_event_parsing_via_predictive_future_modeling.md)

</div>

<!-- RELATED:END -->
