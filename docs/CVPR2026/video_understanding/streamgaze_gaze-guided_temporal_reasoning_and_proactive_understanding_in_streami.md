---
title: >-
  [Paper Note] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos
description: >-
  [CVPR 2026][Video Understanding][Gaze signals] This work presents StreamGaze, the first gaze-guided streaming video understanding benchmark, comprising 8,521 QA pairs covering three task categories — past, present, and proactive prediction. A gaze trajectory–video alignment pipeline is proposed to generate spatiotemporally grounded QA pairs, revealing a substantial gap in current MLLMs' ability to leverage gaze signals for temporal reasoning.
tags:
  - CVPR 2026
  - Video Understanding
  - Gaze signals
  - streaming video understanding
  - temporal reasoning
  - proactive prediction
  - egocentric video
date: 2026-05-08
content_hash: 7166406c7d87c4fa
---

# StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos

**Conference**: CVPR 2026
**arXiv**: [2512.01707](https://arxiv.org/abs/2512.01707)
**Code**: Available (Project page + Code + Dataset)
**Area**: Video Understanding / Streaming Video / Gaze Guidance
**Keywords**: Gaze signals, streaming video understanding, temporal reasoning, proactive prediction, egocentric video

## TL;DR
This work presents StreamGaze, the first gaze-guided streaming video understanding benchmark, comprising 8,521 QA pairs covering three task categories — past, present, and proactive prediction. A gaze trajectory–video alignment pipeline is proposed to generate spatiotemporally grounded QA pairs, revealing a substantial gap in current MLLMs' ability to leverage gaze signals for temporal reasoning.

## Background & Motivation

1. **Background**: Streaming video understanding requires models to process temporally sequential input frames in real time, which is critical for applications such as AR glasses and robotics. Existing streaming video benchmarks (e.g., StreamingBench, OVO-Bench) evaluate temporal reasoning capabilities.
2. **Limitations of Prior Work**: (a) Existing benchmarks rarely incorporate human perceptual signals — particularly gaze — even when they use egocentric video and implicitly target AR scenarios; (b) few benchmarks simultaneously cover past, present, and proactive tasks; (c) integrating gaze signals into video understanding is inherently difficult due to noisy raw gaze streams, persistent camera motion in egocentric footage, and the need for spatiotemporal grounding.
3. **Key Challenge**: Gaze is the most direct and reliable indicator of human visual attention, yet existing benchmarks and models entirely overlook this critical perceptual signal, creating a disconnect between evaluation and real-world deployment.
4. **Goal**: (1) How to construct gaze-guided streaming video QA data? (2) How to design gaze-relevant tasks spanning past, present, and proactive settings? (3) Can current MLLMs effectively exploit gaze signals?
5. **Key Insight**: Leveraging the temporal structure of gaze behavior — extracting fixations, constructing scanpaths, and distinguishing in-FOV from out-of-FOV regions — to build spatiotemporally grounded QA pairs.
6. **Core Idea**: The first benchmark to align gaze trajectories with egocentric streaming video, enabling gaze-guided evaluation of past, present, and proactive tasks through fixation extraction, region-specific visual prompting, and scanpath construction.

## Method

### Overall Architecture
The StreamGaze construction pipeline consists of four steps: (1) preprocessing — projecting raw gaze data onto the 2D image plane; (2) fixation extraction — identifying stable gaze moments; (3) region-based object extraction — distinguishing in-FOV from out-of-FOV objects; and (4) scanpath construction and QA generation. The final benchmark includes 10 tasks across three categories: past, present, and proactive.

### Key Designs

1. **Fixation Extraction**:

    - **Function**: Identify semantically meaningful and stable gaze moments from continuous noisy gaze streams.
    - **Mechanism**: Two conditions are applied. (a) Spatiotemporal stability: the spatial dispersion of gaze points within a fixation window must satisfy $d_t = \|(x_t, y_t) - (\bar{x}_i, \bar{y}_i)\|_2 \leq r_{thresh}$, and the duration must satisfy $t_i^e - t_i^s \geq \tau_{dur}$; (b) Scene consistency: the minimum Pearson correlation coefficient of hue-saturation histograms between consecutive frames within the window must satisfy $S_{min} \geq \tau_{scene}$, filtering out abrupt scene changes caused by camera motion.
    - **Design Motivation**: Saccades do not represent meaningful attention; only fixations reliably reflect a user's visual focus. Scene consistency checking addresses frequent camera motion in egocentric video.

2. **Region-specific Visual Prompting**:

    - **Function**: Precisely extract objects within and outside the gaze region.
    - **Mechanism**: For each frame, an FOV region is defined as a circular area centered at the gaze point with radius $\tau_{fov}$, and the remainder is treated as out-of-FOV. For the FOV region, a circular patch is cropped with a red dot overlaid at the gaze center and fed into an MLLM (InternVL3.5-38B) to extract $\mathcal{O}_i^{fov}$. For the out-of-FOV region, the FOV area is masked with a black disk before feeding into the MLLM to extract $\mathcal{O}_i^{out}$.
    - **Design Motivation**: Physical masking ensures the two object sets remain mutually exclusive, providing a foundation for constructing QA pairs of varying difficulty (e.g., distractors in easy vs. hard modes are sourced differently).

3. **Scanpath Construction and Task Classification**:

    - **Function**: Capture how gaze shifts over time across different spatial regions and semantic contexts.
    - **Mechanism**: All fixations are organized chronologically into a scanpath $\mathcal{S} = \{(\mathcal{O}_i^{fov}, \mathcal{O}_i^{out})\}_{i=1}^N$. Ten tasks are constructed from the scanpath: Past tasks (NFI: non-fixated item recognition; OTP: object transition prediction; GSM: gaze sequence matching; SR: scene recall), Present tasks (OI: object identification Easy/Hard; OAR: object attribute recognition; FAP: future action prediction), and Proactive tasks (GTA: gaze-triggered alarm; OAA: object appearance alarm).
    - **Design Motivation**: Scanpaths preserve the temporal dynamics of gaze. Different tasks target distinct aspects of gaze understanding — Past tasks probe temporal reasoning, Present tasks probe perceptual state, and Proactive tasks probe proactive intervention capability.

### Data Quality Assurance
All scanpaths and object extraction results are manually verified, achieving an average accuracy of approximately 83%. QA pairs undergo dual filtering via Qwen3-VL-30B validation and human review.

## Key Experimental Results

### Main Results

| Model | Params | Past Avg. | Present Avg. | Proactive Avg. | Overall |
|-------|--------|-----------|--------------|----------------|---------|
| Human | - | 0.800 | 0.880 | 0.773 | 0.827 |
| GPT-4o | - | 0.541 | 0.606 | 0.373 | 0.535 |
| Qwen2.5-VL | 7B | 0.450 | 0.522 | 0.447 | 0.478 |
| InternVL3.5 | 8B | 0.481 | 0.523 | 0.212 | 0.444 |
| ViSpeak | 7B | 0.428 | 0.467 | 0.547 | 0.467 |
| EgoGPT | 7B | 0.479 | 0.496 | 0.222 | 0.436 |
| AssistGaze | 26M | 0.257 | 0.223 | N/A | 0.223 |

### Ablation Study
Effect of gaze input modality on Qwen2.5-VL:

| Strategy | Past | Present | Proactive | Avg |
|----------|------|---------|-----------|-----|
| No gaze | 0.423 | 0.500 | 0.384 | 0.446 |
| Text prompt | 0.403 | 0.499 | 0.341 | 0.429 |
| Visual prompt | 0.398 | 0.503 | 0.342 | 0.429 |
| Saliency map | 0.394 | 0.546 | 0.386 | 0.454 |

### Key Findings
- **Large human–model gap**: Humans achieve 0.827 vs. the best model GPT-4o at 0.535 — a gap of nearly 30 percentage points — indicating that current MLLMs are far from capable of effectively leveraging gaze signals.
- **General-purpose MLLMs fail to exploit gaze**: Providing gaze information does not consistently improve model performance and can even degrade it on certain tasks (e.g., gaze prompts in the NFI task restrict exploration of non-fixated objects).
- **Streaming MLLMs excel at proactive tasks**: ViSpeak's frame-by-frame online processing mechanism gives it an advantage over non-streaming models on proactive tasks.
- **Gaze-specialized models generalize poorly**: AssistGaze, despite being designed specifically for gaze, fails to generalize to long streaming scenarios (Overall: 0.223).
- **Saliency maps are the optimal gaze input format**: Aggregating gaze trajectories into heatmaps is more amenable to current models than raw coordinates or frame-level overlays.

## Highlights & Insights
- **End-to-end pipeline from perceptual signal to benchmark design**: The pipeline — fixation extraction → region-specific visual prompting → scanpath construction → task generation — is grounded at each step in psychology and eye-tracking research, offering a replicable paradigm for integrating human perceptual science with AI evaluation.
- **Cognitive hierarchy in task design**: The Past → Present → Proactive progression represents not only a temporal dimension but also an increasing cognitive demand — from memory retrieval, to current perception, to intent inference and proactive intervention.
- **Elegant use of FOV/out-of-FOV distinction**: The FOV/out-of-FOV separation naturally enables difficulty control in QA construction (Easy mode draws distractors from other timesteps; Hard mode from the same frame but outside the FOV), requiring no additional annotation effort.

## Limitations & Future Work
- The benchmark relies exclusively on egocentric video datasets (EGTEA+, EgoExoLearn, HoloAssist), limiting scene diversity to cooking, laboratory, and assembly settings.
- No current model genuinely benefits from gaze signals — this is both a finding and a limitation, suggesting the need for gaze-aware model architectures.
- Proactive task evaluation simulates online decision-making via per-timestep querying, which still differs from truly real-time streaming inference.
- Gaze data collection depends on specific devices; variability in gaze accuracy across devices may limit generalizability.

## Related Work & Insights
- **vs. GazeVQA**: GazeVQA is the first gaze VQA dataset but is confined to static assembly-scenario interactions; StreamGaze extends to streaming settings and covers richer temporal dimensions.
- **vs. EgoGazeVQA**: EgoGazeVQA uses per-frame gaze points without fixation extraction or spatiotemporal grounding; StreamGaze models full scanpath dynamics.
- **vs. StreamingBench / OVO-Bench**: These benchmarks cover past/present/future tasks but entirely omit gaze signals; StreamGaze fills the gap at the intersection of gaze and streaming video.
- **vs. ViSpeak**: ViSpeak's strong proactive performance highlights the importance of frame-by-frame online processing architectures for streaming understanding; future gaze-aware models should prioritize streaming-compatible designs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First benchmark to introduce gaze signals into streaming video understanding; problem formulation is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 16 baselines across 4 model categories with in-depth ablation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, coherent method and task design, high-quality figures and tables.
- Value: ⭐⭐⭐⭐ — Opens a new direction for gaze-aware video understanding, though near-term practical impact is constrained by current model limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamReady: Learning What to Answer and When in Long Streaming Videos](streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)
- [\[CVPR 2026\] Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing](color_when_it_counts_grayscale-guided_online_triggering_for_always-on_streaming_.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_video.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)

</div>

<!-- RELATED:END -->
