---
title: >-
  [Paper Note] VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?
description: >-
  [CVPR 2025][Multimodal VLM][Video Understanding] This paper proposes the VidComposition benchmark, specifically designed to evaluate MLLMs' composition understanding capabilities on **compiled videos** (movies, animations, etc.). It encompasses **5 major categories and 15 subtasks** (shot motion, narrative structure, character understanding, etc.). Evaluation of 33 MLLMs reveals a huge gap between current models and humans in cinematic video understanding (the best model achi…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Video Understanding"
  - "Video Composition Analysis"
  - "MLLM Benchmark"
  - "Movie Analysis"
  - "Shot Language"
date: 2026-05-08
content_hash: 3938c2d7532abc1d
---

# VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?

**Conference**: CVPR 2025  
**arXiv**: [2411.10979](https://arxiv.org/abs/2411.10979)  
**Code**: [https://yunlong10.github.io/VidComposition/](https://yunlong10.github.io/VidComposition/)  
**Area**: Multimodal VLM  
**Keywords**: Video Understanding, Video Composition Analysis, MLLM Benchmark, Movie Analysis, Shot Language

## TL;DR
This paper proposes the VidComposition benchmark, specifically designed to evaluate MLLMs' composition understanding capabilities on **compiled videos** (movies, animations, etc.). It encompasses **5 major categories and 15 subtasks** (shot motion, narrative structure, character understanding, etc.). Evaluation of 33 MLLMs reveals a huge gap between current models and humans in cinematic video understanding (the best model achieves $63.3\%$ vs. $86.3\%$ for humans).

## Background & Motivation
Existing video evaluation benchmarks for MLLMs suffer from three key deficiencies. First, most benchmarks focus on abstract understanding of naturally shot videos (e.g., action recognition, scene description), ignoring **compiled videos**—namely, cinematic and artistic works created by editing and combining multiple clips—which require shot-by-shot analysis. Second, existing benchmarks lack fine-grained evaluation of video "**composition**", such as shot motion perception, shot scale judgment, and narrative structure understanding, which are core abilities in cinematic analysis. Third, although benchmarks like TVQA include compiled videos, their compositional question-answering granularity is coarse, limited to basic questions like "who, when, and where". Key Challenge: **compiled videos dominate modern video platforms (movies, short videos, vlogs), but whether MLLMs truly understand the compositional language of these videos remains unknown**.

## Method

### Overall Architecture
VidComposition is a high-quality, human-annotated benchmark containing 982 compiled videos and 1,706 multiple-choice questions, covering 5 major evaluation dimensions and 15 subtasks. The video sources are video commentaries of movies, TV dramas, animations, etc. (copyright-free), with an average duration of around 20 minutes, segmented into clips averaging 794 frames, and with audio removed to prevent speech-based cheating.

### Key Designs
1. **Five-Dimension, Fifteen-Subtask Evaluation System**:
    - Function: Systematically cover all aspects of cinematic video composition understanding.
    - Mechanism:
        - **Camera Analysis (CA)**: Shot motion perception (push/pull/pan/tilt/static), shot scale perception (extreme long/long/medium/close-up/extreme close-up), camera angle perception (high/low/eye-level).
        - **Character Understanding (CU)**: Emotion perception, action perception, costume/makeup/props perception, character counting.
        - **Narrative Understanding (NU)**: Script matching (matching commentary text with video), plot ordering (restoring shuffled plot sequences).
        - **Scene Perception (SP)**: Background perception, scene counting, lighting perception.
        - **Production Analysis (MA)**: Artistic style perception, cut counting, special effects perception.
    - Design Motivation: Cover the complete video composition analysis capability from the technical level (camera parameters) to the narrative level (plot structure), which is completely blank in existing benchmarks.

2. **High-Quality Human Annotation Process**:
    - Function: Ensure the accuracy and difficulty control of the benchmark data.
    - Mechanism: Multi-round annotation and verification system. For perception tasks (action, emotion, etc.), annotators write correct answers and distractors directly after watching the video. Professional tasks (shot motion, shot scale, etc.) utilize predefined label sets. Script matching uses the commentary text from subtitle files, with distractors sourced from neighboring segments. Plot ordering requires restoring shuffled segments of the commentary script.
    - Design Motivation: Automated QA pair generation cannot guarantee the quality of professional questions in the cinematic domain. Though expensive, human annotation ensures the reliability of the benchmark.

3. **Difficulty Grading and Analytical Framework**:
    - Function: Reveal the capability distribution of MLLMs across different difficulties and dimensions.
    - Mechanism: Grading is based on the proportion of correct answers from models: $>60\%$ models answering correctly is graded as Easy, and $<10\%$ as Super Hard. Additional analysis covers influencing factors: number of input frames, visual encoder resolution, language decoder scale, and fine-tuning data volume.
    - Design Motivation: Not only to know how well the models perform but also to understand why, providing directions for model improvement.

## Key Experimental Results

### Main Results (Performance of 33 MLLMs on VidComposition, Selected Top Models)

| Model | Overall | Camera Analysis | Character Understanding | Narrative Understanding | Scene Perception | Production Analysis |
|------|------|----------|----------|----------|----------|----------|
| Human | **86.3** | 83.2 | 90.5 | 97.3 | 85.5 | 89.0 |
| LLaVA-OneVision-72B | 63.3 | 61.3 | 79.5 | 78.6 | 59.7 | 66.0 |
| InternVL2-40B | 60.7 | 55.2 | 75.3 | 65.8 | 64.0 | 66.2 |
| GPT-4o | 52.9 | 45.6 | 68.6 | 66.9 | 54.2 | 64.0 |
| Gemini-1.5-Pro | 49.4 | 45.7 | 68.1 | 42.0 | 60.7 | 72.9 |

### Dimensional Analysis

| Subtask | Human Accuracy | Best MLLM | Gap |
|--------|-----------|----------|------|
| Shot Motion Perception | $84.1\%$ | $57.1\%$ (LLaVA-OV-72B) | $-27.0\%$ |
| Script Matching | $97.0\%$ | $90.6\%$ (GPT-4o) | $-6.4\%$ |
| Scene Counting | $80.2\%$ | $53.6\%$ (Qwen2-VL-72B) | $-26.6\%$ |
| Cut Counting | $87.5\%$ | $58.6\%$ (Gemini-1.5-Pro) | $-28.9\%$ |
| Action Perception | $92.3\%$ | $90.0\%$ (Multi-model) | $-2.3\%$ |

### Key Findings
- **Camera Analysis is the hardest dimension**: Models lag significantly behind humans in understanding shot motion, angle, and scale, indicating that current MLLMs lack professional visual analysis capabilities.
- **Action perception is close to human level** ($90.0\%$ vs. $92.3\%$), but this aligns with the focus of existing benchmarks, indicating that models have saturated on simple perception tasks.
- **Models generally perform poorly on counting tasks**: Scene counting (highest $53.6\%$), cut counting (highest $58.6\%$), showing that precise temporal segmentation is a major weakness of MLLMs.
- **More frame inputs do not necessarily yield better results**: Some models perform better with 32 frames than with 64 frames, indicating a non-monotonic relationship between frame count and information utilization efficiency.

## Highlights & Insights
- **Fills an important gap**: The first benchmark to systematically evaluate MLLMs' comprehension of cinematic video composition. The design of 15 subtasks is extremely comprehensive.
- **Reveals a profound capability gap**: Models understand "what is happening" but not "how it was shot"—lacking comprehension of video production techniques (camera shots, editing, special effects).
- **Implications for video generation evaluation**: VidComposition can be used to automatically evaluate the compositional quality of generated videos, bridging video understanding and generation.

## Limitations & Future Work
- The video sources are commentary videos (rather than original movie clips), which may introduce secondary editing bias from the commentators.
- Removing audio prevents models from taking speech shortcuts but also loses critical information for certain tasks (e.g., emotion perception).
- Label definitions for some subtasks (e.g., camera angle judgment) may be subjective; different annotators might hold different boundary standards between "high angle" and "eye-level shot".
- It only evaluates using multiple-choice questions without involving open-ended generative evaluation, which may underestimate the depth of understanding in certain models.
- The dataset size (982 videos / 1706 questions) is relatively small, and the sample size for some subtasks is insufficient to draw statistically reliable conclusions.

## Related Work & Insights
- Compared to general video benchmarks like Video-MME, the key difference of VidComposition lies in its focus on "how" rather than "what"—not what happened in the video, but how the video was constructed.
- It aligns with the image compositionality evaluation approach of Winoground and MMComposition, but extends it to more complex video dimensions.
- Insight: Video MLLMs lack cinematic production-related annotation data in their training sets. Introducing professional movie analysis corpora might be key to enhancing such capabilities.
- The evaluation dimensions of VidComposition can be applied to video generation quality assessment, such as evaluating the shot language quality of videos generated by Sora.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to propose cinematic video composition understanding evaluation, filling an important gap in MLLM benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 33 models + influence factor analysis + difficulty grading, yielding extremely thorough analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined classification system, although some subtask definitions could be more precise.
- Value: ⭐⭐⭐⭐⭐ Provides a brand-new direction for video understanding research and directly exposes the core limitations of current models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](../../NeurIPS2025/multimodal_vlm/learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[ACL 2025\] Can MLLMs Understand the Deep Implication Behind Chinese Images?](../../ACL2025/multimodal_vlm/can_mllms_understand_the_deep_implication_behind_chinese_images.md)
- [\[ICLR 2026\] ODI-Bench: Can MLLMs Understand Immersive Omnidirectional Environments?](../../ICLR2026/multimodal_vlm/odi-bench_can_mllms_understand_immersive_omnidirectional_environments.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)

</div>

<!-- RELATED:END -->
