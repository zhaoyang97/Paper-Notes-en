---
title: >-
  [Paper Note] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs
description: >-
  [CVPR 2025][Video Understanding][Video Quality Assessment] The first benchmark, Q-Bench-Video, to systematically evaluate the video quality understanding capabilities of Large Multimodal Models (LMMs), covering natural/AIGC/CG videos, four-dimensional quality focus, and multiple question types.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Quality Assessment"
  - "LMM Benchmark"
  - "LMM"
  - "AIGC Distortion"
  - "Temporal Consistency"
date: 2026-05-08
content_hash: b082be3bb8d6e0ff
---

# Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs

**Conference**: CVPR 2025  
**arXiv**: [2409.20063](https://arxiv.org/abs/2409.20063)  
**Code**: [https://github.com/Q-Future/Q-Bench-Video](https://github.com/Q-Future/Q-Bench-Video)  
**Area**: Video Understanding / Quality Assessment  
**Keywords**: Video Quality Assessment, LMM Benchmark, LMM, AIGC Distortion, Temporal Consistency

## TL;DR

The first benchmark, Q-Bench-Video, to systematically evaluate the video quality understanding capabilities of Large Multimodal Models (LMMs), covering natural/AIGC/CG videos, four-dimensional quality focus, and multiple question types.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Large Multimodal Models (LMMs) have made remarkable progress in high-level semantic video understanding tasks, but a systematic evaluation of video quality understanding is severely lacking. Video quality is crucial for compression optimization, user experience enhancement, and the establishment of video generation standards. The low-level information involved (blur, noise, compression artifacts, etc.) is fundamentally different from high-level semantic understanding. Existing LMM video benchmarks (e.g., MVBench, Video-MME) focus on semantic understanding, leaving out the quality perception dimension. On the other hand, the explosive development of AIGC video generation has introduced brand-new distortion types (unnatural textures, illumination inconsistency, etc.), urgently calling for a specialized evaluation framework. This paper systematically fills this gap.

### Proposed Solution

**Goal**: ### Overall Architecture

The construction of Q-Bench-Video follows three principles: (1) broad video content coverage—1,000 natural scenes, 600 AIGC, and 200 CG videos for a total of 1,800 videos; (2) uniform sampling based on quality annotations to ensure a balanced quality distribution; (3) focusing on four-dimensional quality dimensions that affect the viewing experience.


## Method

### Overall Architecture

The construction of Q-Bench-Video follows three principles: (1) broad video content coverage—1,000 natural scenes, 600 AIGC, and 200 CG videos for a total of 1,800 videos; (2) uniform sampling based on quality annotations to ensure a balanced quality distribution; (3) focusing on three-dimensional quality dimensions that affect the viewing experience. Each data entry is a meta-structure (V, Q, A, C), totaling 2,378 question-answering pairs. Twelve open-source and five closed-source LMMs are evaluated.

### Key Designs

1. **Three Question Types Design**: (a) Yes-or-No questions: binary judgment of video quality, with annotations adjusted to ensure a balanced 50:50 ratio of correct answers to avoid the bias of LMMs; (b) What-How questions: "What" identifies specific distortion types, while "How" distinguishes fine-grained differences in distortion severity; (c) Open-ended questions: without limiting the answer set, evaluating LMMs' ability to perceive video quality in real-world scenarios, such as "Please list and explain the possible factors causing the low clarity of this video." Additionally, a video pair comparison task is added to evaluate relative quality judgment capabilities.

2. **Four Dimensions of Quality Focus**: (a) Technical distortion: low-level degradations like blur, noise, and compression artifacts; (b) Aesthetic distortion: subjective aesthetic deviations in composition, color, illumination, etc.; (c) Temporal distortion: temporal issues such as camera shake, flickering, inconsistent motion, and stuttering; (d) AIGC distortion: unnatural textures, eerie faces, unrealistic object behaviors, and other artifacts unique to AI-generated content. A single question can cover multiple dimensions simultaneously.

3. **Diversity of Video Sources**: Natural videos are from LSVQ (600 sampled from 39K), MaxWell (350 sampled from 4.5K), and the WaterlooSQoE series; AIGC videos are from T2VQA-DB (200 sampled from 10K) and VideoFeedback (400 sampled from 37.6K); CG videos are sourced from LIVE-YT-Gaming (200 sampled from 600). Most datasets contain ITU-standard MOS annotations, ensuring the scientific rigour of quality sampling.

### Loss & Training

- Pure evaluation benchmark with no training component
- Open-ended questions use GPT-4 as an auxiliary scorer
- Multiple-choice questions use accuracy
- Video pair comparison uses consistency rate

## Key Experimental Results

### Main Results

| Model | Yes-or-No↑ | What-How↑ | Open-ended↑ | Average↑ |
|------|-----------|----------|-------------|------|
| GPT-4o | Highest | Highest | Highest | Highest |
| InternVL2 | Second Highest | Second Highest | - | Second Highest |
| VideoLLaMA2 | Medium | Medium | - | Medium |
| Human Performance | **Far higher than all LMMs** | **Far higher than all LMMs** | **Far higher than all LMMs** | **Significant Lead** |

### Ablation Study

| Dimension | LMM Performance Variance |
|------|-------------|
| Technical Distortion | Relatively Good (LMMs have basic perception of blur/noise) |
| Aesthetic Distortion | Medium |
| Temporal Distortion | Poor (LMMs struggle to capture temporal issues) |
| AIGC Distortion | Poor (LMMs are insensitive to AI-generated artifacts) |

### Key Findings

- LMMs have a basic but incomplete and imprecise understanding of video quality, with a significant gap compared to human performance.
- Closed-source models (e.g., GPT-4o) significantly outperform open-source models.
- LMMs perform worst on the temporal distortion and AIGC distortion dimensions—which are precisely the two most unique aspects of video quality.
- The video pair comparison task is more challenging than single-video evaluation.
- Open-ended questions expose the limitations of LMMs in explaining the causes of quality issues.

## Highlights & Insights

- The first work to propose LMM video quality understanding as an independent research direction, filling an important gap.
- The introduction of the AIGC distortion dimension is highly timely—with the popularity of video generation models, the demand for such evaluation is surging.
- The balanced design of Yes-or-No questions and the introduction of open-ended questions enhance the comprehensiveness and authenticity of the evaluation.
- The benchmark reveals the fundamental limitations of LMMs in low-level information perception.

## Limitations & Future Work

- The scale of 2,378 QA pairs can be further expanded.
- Open-ended evaluation relying on GPT-4 may introduce bias.
- The video quality scoring capability of LMMs (quantitative scoring vs. qualitative description) was not evaluated.
- The framework can be extended to evaluate outputs from more diverse video generation models.

## Related Work & Insights

- **vs Video-MME/MVBench**: Focus on semantic understanding; Q-Bench-Video focuses on low-level quality understanding, complementing each other.
- **vs Traditional VQA Methods**: Traditional methods output quality scores; Q-Bench-Video evaluates the quality understanding and explanation capabilities of LMMs.
- **vs Q-Bench (Image Version)**: Extends the image quality benchmark paradigm to videos, adding temporal and AIGC dimensions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first LMM benchmark for video quality, a pioneering direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — A comprehensive evaluation with 17 models, 4 dimensions, and 3 question types.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear benchmark design principles and a complete taxonomic system.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a standardized evaluation platform for video quality understanding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[CVPR 2026\] FPS-Bench: A Benchmark for High Frame-Rate Video Understanding](../../CVPR2026/video_understanding/fps-bench_a_benchmark_for_high_frame-rate_video_understanding.md)
- [\[CVPR 2025\] SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding](seriesbench_a_benchmark_for_narrative-driven_drama_series_understanding.md)
- [\[CVPR 2025\] FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding](fsbench_a_figure_skating_benchmark_for_advancing_artistic_sports_understanding.md)
- [\[CVPR 2026\] CoCoVideo: The High-Quality Commercial-Model-Based Contrastive Benchmark for AI-Generated Video Detection](../../CVPR2026/video_understanding/cocovideo_the_high-quality_commercial-model-based_contrastive_benchmark_for_ai-g.md)

</div>

<!-- RELATED:END -->
