---
title: >-
  [Paper Note] SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding
description: >-
  [CVPR 2025][Video Understanding][Video understanding benchmark] Proposes SeriesBench, the first video benchmark for narrative-driven drama series understanding, covering 105 series, 28 tasks, and 5 major dimensions, and introduces the PC-DCoT (Plot-Character Double-Chain-of-Thought) framework, which boosts MLLM performance by over 10%.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video understanding benchmark"
  - "narrative understanding"
  - "TV series understanding"
  - "multimodal large language models"
  - "chain-of-thought"
date: 2026-05-08
content_hash: 509d6063b20f81d8
---

# SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding

**Conference**: CVPR 2025  
**arXiv**: [2504.21435](https://arxiv.org/abs/2504.21435)  
**Code**: [https://github.com/](https://github.com/) (GitHub Repository)  
**Area**: Video Understanding / Multimodal Benchmark  
**Keywords**: Video understanding benchmark, narrative understanding, TV series understanding, multimodal large language models, chain-of-thought

## TL;DR

Proposes SeriesBench, the first video benchmark for narrative-driven drama series understanding, covering 105 series, 28 tasks, and 5 major dimensions, and introduces the PC-DCoT (Plot-Character Double-Chain-of-Thought) framework, which boosts MLLM performance by over 10%.

## Background & Motivation

**Background**: With the rapid development of Multimodal Large Language Models (MLLMs), many video understanding benchmarks (e.g., Video-MME, MVBench, TempCompass) have emerged to evaluate models' video comprehension capabilities. These benchmarks primarily focus on "visual elements" (such as character actions and object states) in isolated video clips.

**Limitations of Prior Work**: Existing benchmarks suffer from three significant limitations: (1) they only focus on isolated videos, ignoring the coherent narrative structure and character development in multi-episode series; (2) they primarily evaluate visual elements (actions, states), overlooking the multimodal nature of modern videos (scripts, audio, effects); (3) they rarely involve deep narrative reasoning (e.g., "Why did this person hand over this object?"), staying only at the surface level of visual recognition.

**Key Challenge**: In reality, video content is typically presented in the form of continuous narrative episodes, requiring cross-video character tracking and plot reasoning. However, existing benchmarks completely overlook the evaluation of this serialized narrative understanding capability.

**Goal**: (1) Establish the first comprehensive benchmark for narrative-driven drama series understanding; (2) cover multimodal elements including subtitles, audio, and special effects; (3) propose a reasoning framework to enhance the narrative understanding capabilities of MLLMs.

**Key Insight**: Inspired by human binge-watching behavior—people naturally track plot lines and character relationships when watching a show—this process is simulated by constructing plot event chains and character temporal chains.

**Core Idea**: Decompose series understanding into 5 major task dimensions (video/script/audio/augmented/comprehension), construct the benchmark through long-span narrative annotation and full-information conversion methods, and use the PC-DCoT double-chain-of-thought framework to enhance the narrative understanding capability of models.

## Method

### Overall Architecture

SeriesBench consists of three components: (1) Dataset—105 curated series, 1,072 video clips, and 29,196 task samples; (2) Task System—28 fine-grained sub-tasks across 5 major dimensions, supporting multiple-choice, boolean, and open-ended question formats; (3) PC-DCoT Reasoning Framework—enhances the narrative understanding capability of MLLMs through a double-chain structure. The inputs are serial videos and associated subtitles/character information, and the output is the accurate answer to narrative-related questions.

### Key Designs

1. **Five-Dimension 28-Subtask System**:

    - Function: Provides systematic evaluation coverage across various components of modern videos.
    - Mechanism: Designs 5 major task dimensions around core elements of modern videos: [Video] analyzes visual elements like character actions, scene transitions, and object states across 6 sub-tasks; [Script] focuses on narrative elements such as world-building, plot development, and character motivation across 7 sub-tasks; [Audio] evaluates dialogue attribution, tone/emotion, and sound effect impact across 5 sub-tasks; [Augmented] checks subtitle recognition, tag interpretation, and special effects understanding across 3 sub-tasks; [Comprehension] integrates all elements for high-level tasks like future prediction and character empathy across 3 sub-tasks.
    - Design Motivation: Prior benchmarks only evaluated vision, whereas modern videos are multimodal compositions. The five-dimension design aligns the evaluation with actual video understanding demands, particularly the [Script] and [Comprehension] dimensions, which fill the gap in narrative reasoning evaluation.

2. **Long-Span Narrative Annotation + Full-Information Conversion Method**:

    - Function: Efficiently constructs high-quality narrative understanding QA data.
    - Mechanism: The annotation workflow consists of two steps. The first step, "long-span narrative annotation," involves 32 professional annotators fully understanding the video narrative first, then identifying key segments (critical events, character actions) and summarizing the annotations as declarative sentences integrating all relevant content. The second step, "full-information conversion," leverages GPT-4o to transform the manually annotated declarative sentences under the complete video context (subtitles, theme, character background) into various question types (boolean, multiple-choice, open-ended). The annotated contents naturally serve as the question stems and correct answers, while related video information forms distractor options. Random sampling of 500 annotations shows that 96% meet the quality standards.
    - Design Motivation: Directly asking annotators to write questions easily leads to shallow visual questions. By first annotating narrative events and then converting them into questions, it ensures the questions require deep narrative reasoning instead of surface-level recognition. The full-information conversion method efficiently generates diverse question types using rich existing video metadata.

3. **PC-DCoT (Plot-Character Double-Chain-of-Thought) Reasoning Framework**:

    - Function: Enhances MLLMs' capability to understand complex plots and character relationships in narrative-driven drama series.
    - Mechanism: A three-step workflow. (1) Event and Character Extraction: The MLLM receives raw video frames and questions, extracting key events and characters that need to be tracked. (2) Double-Chain Construction: A trained video clip model is used to retrieve frames corresponding to events and aggregate them into independent event sequences, constructing the "plot event chain"; simultaneously, character portraits are utilized to retrieve all frames where the character appears in the video, constructing the "character temporal chain." (3) Double-Chain Synthesis and Reasoning: The two chains are aligned based on precise timestamps to determine which characters are involved in each event interval. The synthesized unified representation then assists the MLLM in performing more accurate narrative reasoning.
    - Design Motivation: Characters appear discontinuously in narrative videos, whereas events develop coherently. Tracking them separately and then merging them is more effective than directly processing the entire video. This design simulates the human cognitive process of simultaneously focusing on "what happened" and "who was involved" while watching a show.

### Loss & Training

PC-DCoT is an inference-time framework and does not require training. However, a trained video clip matching model is used during the construction process to retrieve key frames. The annotation process uses GPT-4o for full-information conversion. Evaluation uses accuracy (for multiple-choice/boolean) and BLEU-2/METEOR/BERTScore F1 (for open-ended questions).

## Key Experimental Results

### Main Results

| Model | Overall | VS(Video) | SC(Script) | AU(Audio) | AG(Augmented) | CO(Comprehension) |
|------|---------|---------|---------|---------|---------|---------|
| Random Choice | 37.7 | 39.3 | 38.2 | 35.5 | 36.5 | 38.8 |
| Qwen2-VL (7B) | 60.3 | 55.7 | 57.5 | 58.6 | 75.3 | 59.6 |
| GPT-4o | **62.8** | 55.8 | **62.8** | **60.6** | **79.9** | 59.6 |
| GPT-4o + PC-DCoT | **76.2** (+13.4) | **78.6** | **76.1** | **73.8** | **82.1** | 61.7 |
| InternVL2 + PC-DCoT | 73.3 (+14.1) | 76.5 | 71.4 | 67.3 | 81.1 | 66.7 |
| Human | 95.8 | 98.2 | 94.4 | 94.6 | 97.2 | 92.6 |

### Ablation Study

| Configuration | Accuracy Gain | Explanation |
|------|----------|------|
| InternVL2 baseline | 59.2 | w/o PC-DCoT |
| + PC-DCoT | 73.3 (+14.1%) | Largest improvement |
| Qwen2-VL baseline | 60.3 | w/o PC-DCoT |
| + PC-DCoT | 73.9 (+13.6%) | Improvement comparable to InternVL2 |
| MiniCPM-V 2.6 baseline | 59.1 | w/o PC-DCoT |
| + PC-DCoT | 72.0 (+12.9%) | Generally effective across all architectures |
| GPT-4o baseline | 62.8 | Commercial model |
| + PC-DCoT | 76.2 (+13.4%) | Commercial model also benefits significantly |

### Key Findings

- **Significant Performance Drop of SOTA Models on SeriesBench**: Models that achieve over 80% on benchmarks like Video-MME drop to only around 60% on SeriesBench, indicating a severe deficiency in narrative understanding capabilities. This is mainly because discrete frame sampling disrupts visual continuity, making models over-reliant on image-level descriptions rather than narrative comprehension.
- **PC-DCoT is Universally Effective Across Models**: An average improvement of 13.5% across 4 models indicates that the double-chain structure indeed helps models better organize and exploit narrative information.
- **Gap Between Open-Source and Commercial Models Remains Significant**: GPT-4o leads in [Script] tasks, but open-source models can approach or even surpass GPT-4o in visual tasks.
- **Interesting Finding**: VideoLLaMA2.1-AV exhibits a performance drop after adding audio, indicating that current audio-visual alignment capabilities are still deficient; textual information (subtitles) is more effective than audio.
- **Inconsistent Impact of Multi-Episode Context**: Adding preceding or succeeding episode content does not always improve performance and sometimes even degrades it due to the increased token count.

## Highlights & Insights

- **Long-Span Narrative Annotation Method**: Demanding annotators to comprehend the complete narrative before annotating (rather than annotating frame-by-frame) ensures the narrative depth of questions. This "understand-first, annotate-second" paradigm can be extended to other annotation scenarios requiring deep understanding.
- **Double-Chain Design of PC-DCoT**: The plot chain tracks "what happened" while the character chain tracks "who was involved," merging them via timeline alignment. This decompose-and-reconstitute reasoning strategy solves complex narrative understanding problems with a straightforward structure.
- **Full-Information Conversion Method**: Utilizing existing metadata to automatically transform annotated declarative sentences into diverse question types significantly reduces annotation costs while ensuring question quality.

## Limitations & Future Work

- Although covering multiple genres, 105 series is still limited, and they are mainly Chinese short dramas from the Kuaishou platform, lacking cultural and linguistic diversity.
- PC-DCoT relies on a pre-trained video clip matching model to retrieve key frames, which directly affects the quality of the double-chain construction.
- Even with PC-DCoT, the best model (76.2%) still lags behind the human level (95.8%) by nearly 20%, especially in complex causal reasoning and multi-character plot analysis.
- Evaluation of the audio dimension is limited by the fact that most Video-MLLMs do not support audio inputs, meaning what is actually tested is the capability to understand audio indirectly through subtitles.

## Related Work & Insights

- **vs Video-MME/MVBench**: These benchmarks only evaluate the recognition of visual elements in isolated videos. SeriesBench extends this to serialized narrative understanding with richer task dimensions (5 dimensions vs 1 dimension), requiring deeper reasoning capabilities.
- **vs EgoSchema/MLVU**: Although they focus on long video understanding, they are still limited to single videos and lack cross-video narrative associations. The multi-episode design of SeriesBench is closer to real-world video consumption scenarios.
- **vs TVBench**: Focuses on TV shows but still remains at the level of visual understanding. SeriesBench is the first to systematically evaluate multimodal understanding capabilities of narratives, audio, special effects, etc.

## Rating

- Novelty: ⭐⭐⭐⭐ The first benchmark targeting drama narrative understanding, with a simple and effective PC-DCoT mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on 10 models, verifying PC-DCoT across 4 models, with in-depth multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Complete structure, clear task definitions, and informative tables.
- Value: ⭐⭐⭐⭐ Fills the gap in narrative understanding evaluation, though the dataset scale and diversity still have room for growth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding](fsbench_a_figure_skating_benchmark_for_advancing_artistic_sports_understanding.md)
- [\[CVPR 2025\] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)
- [\[ICCV 2025\] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](../../ICCV2025/video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)
- [\[AAAI 2026\] LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding](../../AAAI2026/video_understanding/livibench_an_omnimodal_benchmark_for_interactive_livestream_video_understanding.md)
- [\[ICLR 2026\] ScaleLong: A Multi-Timescale Benchmark for Long Video Understanding](../../ICLR2026/video_understanding/scalelong_a_multi-timescale_benchmark_for_long_video_understanding.md)

</div>

<!-- RELATED:END -->
