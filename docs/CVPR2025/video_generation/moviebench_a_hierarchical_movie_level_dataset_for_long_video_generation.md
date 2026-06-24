---
title: >-
  [Paper Note] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation
description: >-
  [CVPR 2025][Video Generation][Long Video Generation] This paper introduces MovieBench, the first hierarchical dataset designed for movie-level long video generation. It provides a three-level annotation structure (movie-scene-shot) that includes character portraits, subtitles, and audio. Based on this, four benchmark tasks are defined (text-to-keyframe, identity-customized long video, keyframe-conditioned video, and audio-driven speaking head generation)…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Long Video Generation"
  - "Movie-level Dataset"
  - "Hierarchical Annotation"
  - "Character Consistency"
  - "Multi-scene Narrative"
date: 2026-05-08
content_hash: a683bde5f52b2398
---

# MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.15262](https://arxiv.org/abs/2411.15262)  
**Code**: [https://github.com/showlab/MovieBench](https://github.com/showlab/MovieBench)  
**Area**: Video Generation  
**Keywords**: Long Video Generation, Movie-level Dataset, Hierarchical Annotation, Character Consistency, Multi-scene Narrative

## TL;DR
This paper introduces MovieBench, the first hierarchical dataset designed for movie-level long video generation. It provides a three-level annotation structure (movie-scene-shot) that includes character portraits, subtitles, and audio. Based on this, four benchmark tasks are defined (text-to-keyframe, identity-customized long video, keyframe-conditioned video, and audio-driven speaking head generation), which reveal significant challenges for existing models in multi-scene narrative consistency.

## Background & Motivation

**Background**: The field of video generation has made significant progress in recent years. Models such as Stable Video Diffusion, CogVideo, and SORA can generate high-quality short video clips from text prompts. On the data side, large-scale video-text datasets like WebVid-10M, Panda-70M, and InternVid have driven model training. However, these datasets are primarily tailored for short videos (typically lasting from a few seconds to tens of seconds).

**Limitations of Prior Work**: Current video generation research focuses on single-scene short videos. Movie-level long video generation still faces three major unsolved problems: (1) narrative coherence across multiple scenes—different scenes need to unfold a complete story; (2) character appearance consistency—the same character must maintain a consistent appearance across different scenes; (3) audio continuity—dialogue and background audio must cooperate across scenes. The key obstacle is that **no public dataset** provides all three types of annotations simultaneously.

**Key Challenge**: Existing video datasets are either large-scale but coarsely annotated (providing only clip-level descriptions), or finely annotated but designed for video understanding rather than generation. Movie datasets like MAD and AutoAD focus on video understanding (retrieval, captioning), and their annotation formats are unsuitable for training generative models.

**Goal**: To build a hierarchical dataset specifically designed for movie-level video generation that includes rich character information, coherent storylines, and layered annotation structures, and to establish standardized evaluation benchmarks based on this.

**Key Insight**: Movies naturally exhibit multi-scene narratives, character consistency, and audio synchronization, making them ideal materials for studying long video generation. The authors utilize an automated pipeline to extract hierarchical annotations from movies, which significantly reduces human annotation costs.

**Core Idea**: Organize video data and annotations using a "movie-scene-shot" three-level hierarchical structure to provide a complete information chain from coarse to fine for long video generation research.

## Method

### Overall Architecture
MovieBench is not a modeling framework, but rather a dataset and benchmarking framework. The core pipeline includes: (1) movie data collection and preprocessing; (2) hierarchical annotation generation at three levels; (3) character bank construction; and (4) formulation and evaluation of four benchmark tasks. The input consists of full-length movie videos and subtitle files, while the output is structured hierarchical annotation data.

### Key Designs

1. **三级层次化数据结构 (Hierarchical Data Structure)**:

    - **Function**: To provide multi-granularity video description information from global to local levels.
    - **Mechanism**: The movie-level provides movie overviews (~43.4K words per movie, lasting ~45.6 minutes), including global information such as synopsis, main characters, and styles. The scene-level corresponds to scene segments in movies (averaging 263.6 words, 15.4 seconds), annotating scene themes, involved characters, and spatial relationships. The shot-level is the finest granularity (averaging 66.3 words, 4.09 seconds), offering detailed actions, dialogues, and visual descriptions of individual shots. These three levels of information are generated through an automated pipeline utilizing subtitle alignment and scene detection technologies.
    - **Design Motivation**: Generation tasks of different granularities require different levels of information. Keyframe generation needs scene-level information, shot transitions require shot-level details, and character consistency maintenance relies on movie-level character banks. Existing datasets provide only a single granularity, failing to support these demands simultaneously.

2. **角色信息库 (Character Bank)**:

    - **Function**: To establish complete character profiles for each movie, including character names, portrait images, and voices.
    - **Mechanism**: Facial images (covering multiple angles and expressions) of major characters are automatically extracted from movies. Different scenes featuring the same character are associated using face detection and clustering algorithms to construct a character identity library. Simultaneously, dialogue audio segments of characters are extracted for audio-driven generation tasks. The standard version of Ours contains portraits and audio, and the Ours++ extended version further expands the data scale (116.8 hours vs. 69.2 hours).
    - **Design Motivation**: Character consistency is one of the biggest challenges in movie-level video generation. Existing datasets lack character-level annotations, preventing models from learning the critical constraint that the same character should maintain a consistent appearance across different scenes.

3. **四个基准任务定义 (Four Benchmark Tasks)**:

    - **Function**: To comprehensively evaluate the different dimensional capabilities of long video generation.
    - **Mechanism**: Task 1 (Text→Keyframe/Storyboard) generates keyframe sequences from movie-level text descriptions, testing narrative understanding and visual planning capabilities. Task 2 (Identity-Customized Long Video) generates multi-scene videos that maintain character consistency given character portraits and scene descriptions. Task 3 (Keyframe-conditioned Video) generates coherent videos from keyframe sequences, testing inter-frame transition and dynamic generation capabilities. Task 4 (Audio-driven Talking Human) generates character speaking videos driven by audio.
    - **Design Motivation**: Movie-level video generation is not a single task but a synthesis of multiple sub-capabilities. Decomposing it into four tasks allows independent evaluation of each capability and facilitates step-by-step research and development.

### Dataset Statistics & Comparison
MovieBench complements existing datasets in key dimensions: total video duration is 69.2 hours (116.8 hours for Ours++), resolution is 1080p, and it provides triple annotations of character portraits, audio, and subtitles. It is the only dataset that covers all three hierarchical levels: movie-level, scene-level, and shot-level. Compared to WebVid-10M (360p, no character information) and Panda-70M (720p, no character information), MovieBench features a much higher annotation density despite its smaller scale.

## Key Experimental Results

### Dataset Scale Comparison

| Dataset | Total Duration | Resolution | Character Info | Hierarchical Structure | Text Source |
|--------|--------|--------|---------|---------|---------|
| WebVid-10M | 52Khr | 360p | ✗ | Shot-level only | Alt-Text |
| Panda-70M | 167Khr | 720p | ✗ | Shot-level only | Generated |
| InternVid | 371.5Khr | 720p | ✗ | Shot-level only | Generated |
| MiraData | 16Khr | 1080p | ✗ | Shot-level only | Generated |
| MovieBench | 69.2hr | 1080p | ✓ (Portrait + Audio) | Three-level Hierarchical | Generated |
| MovieBench++ | 116.8hr | 1080p | ✓ (Portrait + Audio) | Three-level Hierarchical | Generated |

### Benchmark Task Experimental Results

| Task | Method | Character Consistency | Narrative Coherence | Visual Quality |
|------|------|----------|----------|---------|
| Task 2 Identity-Customized | DreamVideo | Low | Medium | Medium |
| Task 2 Identity-Customized | Magic-Me | Low | Medium | Medium |
| Task 3 Keyframe-conditioned | StoryDiffusion+CogVideoX | Low | Low-Medium | Medium |
| Task 3 Keyframe-conditioned | StoryDiffusion+Kling 1.5 | Medium | Medium | High |
| Task 4 Audio-driven | Hallo2 (Source Image) | High | - | High |
| Task 4 Audio-driven | Hallo2 (Text Conditional) | Low | - | Medium |

### Key Findings
- All existing methods perform poorly in multi-scene character consistency: Even when utilizing identity-customization methods (DreamVideo, Magic-Me), stabilizing character appearances across scenes remains difficult, especially under dramatic changes in character poses and illumination.
- Closed-source commercial models (Kling 1.5) significantly outperform open-source models (CogVideoX) in visual quality but still face difficulties maintaining character consistency.
- Multi-character scenes present the greatest challenge: When three or more characters appear in the same scene, all methods are prone to character confusion or appearance drift.
- Character consistency in text-conditioned generation is significantly lower than in image-conditioned generation, highlighting that current models lack sufficient mapping capabilities from text to character features.

## Highlights & Insights
- **The hierarchical organization scheme is highly inspiring**: Organizing movie content across three levels ("movie-scene-shot") provides corresponding supervision signals for generation and understanding tasks of different granularities. This hierarchical design can be generalized to other long-sequence generation tasks (such as long documents and long dialogues).
- **The construction of the Character Bank** is a key contribution: Organizing data around characters fills a crucial gap in existing datasets, creating the data foundation for generating consistent characters. The pipeline of facial clustering and cross-scene association is highly reusable.
- **Revealing the fundamental flaws of the "long video = concatenating multiple short videos" paradigm**: The experiments clearly demonstrate that simply cascading short-video generative models cannot address movie-level challenges. There is an essential need to redesign generative architectures that natively consider global consistency.

## Limitations & Future Work
- The dataset scale is relatively small (69.2 hours), which limits its utility for large-scale training compared to massive datasets like WebVid-10M.
- The quality of the automatic annotation pipeline relies heavily on underlying modules (scene detection, face recognition, etc.), potentially introducing noise.
- Evaluation is primarily qualitative, lacking standard definitions for quantitative consistency and narrative quality metrics.
- Due to copyright constraints involving commercial films, the academic use of this dataset must navigate potential legal risks.
- The baselines for the four tasks are relatively straightforward, lacking specialized movie generation methodologies as strong baselines.

## Related Work & Insights
- **vs MAD / AutoAD**: These datasets are also movie-based but are geared toward understanding tasks (retrieval, description). MovieBench is designed for generation, with an annotation format better suited as training signals for generative models.
- **vs MiraData**: MiraData is also designed for video generation with high-quality annotations, but lacks character information and hierarchical structures. MovieBench significantly exceeds it in annotation richness.
- **vs WebVid-10M / Panda-70M**: These large-scale datasets are suitable for pre-training but are coarsely structured. MovieBench serves as a highly complementary fine-tuning or evaluation dataset.

## Rating
- Novelty: ⭐⭐⭐⭐ The first hierarchical dataset designed for movie-level long video generation, with four clearly defined tasks.
- Experimental Thoroughness: ⭐⭐⭐ Primarily qualitative analysis; quantitative evaluation is not fully compiled.
- Writing Quality: ⭐⭐⭐⭐ Transparent dataset description and well-defined tasks.
- Value: ⭐⭐⭐⭐ Fills a crucial gap in datasets and benchmarks for long video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation](long_video_diffusion_generation_with_segmented_cross-attention_and_content-rich_.md)
- [\[CVPR 2025\] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)
- [\[ICLR 2026\] Captain Cinema: Towards Short Movie Generation](../../ICLR2026/video_generation/captain_cinema_towards_short_movie_generation.md)

</div>

<!-- RELATED:END -->
