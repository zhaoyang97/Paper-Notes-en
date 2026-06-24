---
title: >-
  [Paper Note] Video-Bench: Human-Aligned Video Generation Benchmark
description: >-
  [CVPR 2025][Video Generation][Video Generation Evaluation] This paper proposes Video-Bench, a comprehensive benchmark for video generation evaluation, which systematically leverages Multimodal Large Language Models (MLLMs) to automatically evaluate generated videos through two techniques, Chain-of-Query and Few-Shot Scoring, achieving the highest alignment with human preferences across all evaluation dimensions.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Generation Evaluation"
  - "Benchmark"
  - "MLLM Evaluation"
  - "Human Preference Alignment"
  - "Chain-of-Query"
date: 2026-05-08
content_hash: a430ba33bcf95720
---

# Video-Bench: Human-Aligned Video Generation Benchmark

**Conference**: CVPR 2025  
**arXiv**: [2504.04907](https://arxiv.org/abs/2504.04907)  
**Code**: [https://github.com/Video-Bench/Video-Bench.git](https://github.com/Video-Bench/Video-Bench.git)  
**Area**: Diffusion Models  
**Keywords**: Video Generation Evaluation, Benchmark, MLLM Evaluation, Human Preference Alignment, Chain-of-Query

## TL;DR

This paper proposes Video-Bench, a comprehensive benchmark for video generation evaluation, which systematically leverages Multimodal Large Language Models (MLLMs) to automatically evaluate generated videos through two techniques, Chain-of-Query and Few-Shot Scoring, achieving the highest alignment with human preferences across all evaluation dimensions.

## Background & Motivation

**Background**: Video generation evaluation benchmarks are mainly divided into two categories: metric-and-embedding-based methods (e.g., FID, FVD, CLIP score) which provide quantitative evaluations but often conflict with human judgments; and LLM-based methods, which possess reasoning capabilities but face two main limitations: difficulty in cross-modal comparison and vague textual grading standards.

**Limitations of Prior Work**: Metric-based methods (VBench, EvalCrafter) evaluate video quality by combining computed metrics, but their evaluation results significantly deviate from human preferences. LLM-based methods (CompBench, T2VScore) attempt to introduce reasoning capabilities, but only use LLMs in the text-video alignment dimension, while still relying on traditional metrics for other aspects. Specifically, there are two bottlenecks: (1) in video-text alignment evaluation, MLLMs are prone to textual bias and hallucination, making it difficult to accurately detect cross-modal inconsistencies; (2) in video quality evaluation, the vagueness of textual standards causes models to bias towards "average scores".

**Key Challenge**: There is a significant gap between the ratings of auto-evaluation methods and human perception, especially in dimensions requiring cross-modal understanding and fine-grained quality judgment.

**Goal**: Build a fully-dimensional, auto-evaluation framework for video generation that is highly aligned with human preferences.

**Key Insight**: The authors point out two inherent difficulties in direct MLLM scoring: cross-modal comparison and vague evaluation standards. They propose to decompose the problem: first converting video to text before comparison for alignment dimensions, and utilizing multi-video references to calibrate scoring for quality dimensions.

**Core Idea**: Systematically solve the two major bottlenecks of MLLMs in video evaluation through two strategies: Chain-of-Query (iterative multi-turn video-to-text conversion and comparison) and Few-Shot Scoring (batch video comparison for score calibration).

## Method

### Overall Architecture

Video-Bench consists of two layers: (1) an evaluation dimension system, divided into video-condition alignment (5 dimensions) and video quality (4 dimensions), totaling 9 dimensions; (2) an MLLM auto-evaluation framework, which designs different evaluation strategies for different types of dimensions—Chain-of-Query for alignment dimensions and Few-Shot Scoring for quality dimensions. It is accompanied by 419 prompts and 35,196 human annotations.

### Key Designs

1. **Chain-of-Query (Alignment Dimension Evaluation)**:

    - **Function**: Converts cross-modal comparison into uni-modal textual comparison through multi-turn iterations, resolving the cross-modal hallucination problem of MLLMs.
    - **Mechanism**: Divided into four steps: (1) the MLLM generates an initial textual description and summary of the video; (2) the LLM generates $N$ sets of targeted question chains based on the description and the original prompt (e.g., "Is the color of the koala in the video consistent with the prompt?"); (3) the MLLM answers the questions one by one and regenerates the description, adding dimension-related details; (4) the MLLM aggregates the video content and multi-turn dialogue history to issue a final score. This process avoids direct cross-modal comparison.
    - **Design Motivation**: Direct cross-modal comparison by MLLMs is hallucination-prone. Converting video information into text before comparison significantly reduces misjudgments caused by modal disparity.

2. **Few-Shot Scoring (Quality Dimension Evaluation)**:

    - **Function**: Solves the issue where MLLMs tend to output "average scores" when grading video quality.
    - **Mechanism**: Multiple videos generated from the same prompt are grouped into a batch. During evaluation, each video leverages other videos in the batch as references. When evaluating the second video, the rating of the first video and all other in-batch videos serve as implicit references, forming a comparative framework. This resembles "grading by comparing the reference answer side-by-side with the student's work".
    - **Design Motivation**: Textual standards alone (such as the boundary between "slightly blurry" and "highly clear") are too vague, making it difficult for MLLMs to distinguish different levels. Providing concrete video references serves as "anchors," making the scoring more discriminative.

3. **Dimension System Design**:

    - **Function**: Provides comprehensive coverage of evaluation dimensions.
    - **Mechanism**: Video-condition alignment contains object class consistency, action consistency, color consistency, scene consistency (3-point scale), and overall video-text consistency (5-point scale); video quality contains imaging quality, aesthetic quality, temporal consistency, and motion quality (all 5-point scale).
    - **Design Motivation**: Existing benchmark dimensions are incomplete, and different dimensions vary in difficulty, requiring customized scoring scales.

### Loss & Training

This work does not involve model training. The evaluation framework is based on GPT-4o (for multimodal input) and GPT-4o-mini (for text-only reasoning). The prompt suite design combines human action data from Kinetics-400 and related prompts from VBench, featuring 70-90 prompts per dimension, with each prompt sampled 3 times to reduce stochastic bias.

## Key Experimental Results

### Main Results

| Evaluation Method | Imaging Quality | Aesthetic Quality | Temporal Consistency | Motion Quality | Overall Alignment | Object Class | Color | Action | Scene |
|---------|---------|---------|----------|---------|---------|---------|------|------|------|
| MUSIQ | 0.363 | - | - | - | - | - | - | - | - |
| CLIP | - | - | 0.260 | - | - | - | - | - | - |
| CompBench* | - | - | - | - | 0.633 | 0.611 | 0.696 | 0.633 | 0.631 |
| **Video-Bench** | **0.733** | **0.702** | **0.402** | **0.514** | **0.732** | **0.735** | **0.750** | **0.718** | **0.733** |

### Ablation Study

| Configuration | Alignment Avg Spearman | Quality Avg Spearman | Description |
|------|-------------------|-------------------|------|
| W/o Chain-of-Query | 0.679 | - | Single-turn evaluation |
| + Chain-of-Query | 0.7336 | - | Multi-turn iteration gain +0.054 |
| W/o Few-Shot Scoring | - | 0.561 | Independent scoring |
| + Few-Shot Scoring | - | 0.620 | Batch reference gain +0.059 |

### Key Findings

- The human-machine consistency of Video-Bench (Krippendorff $\alpha = 0.50$) is nearly on par with inter-annotator human consistency ($\alpha = 0.52$), showing that automatic evaluation has approached human evaluation levels.
- Video-Bench outperforms existing methods across all 9 dimensions, with Chain-of-Query contributing the largest improvement to alignment-related dimensions (e.g., Color Consistency increases from 0.699 to 0.750).
- The model leaderboard shows that Gen3 is optimal in video quality, while CogVideoX achieves the best performance in condition alignment.
- In certain cases where Video-Bench ratings diverge from human evaluation, Video-Bench actually provides a more objective and accurate judgment.

## Highlights & Insights

- **The "modality conversion" strategy of Chain-of-Query** is highly clever—instead of letting the MLLM perform direct cross-modal comparison, it first "translates" the video into text and then conducts textual comparison, effectively avoiding the hallucination problems inherent in multimodal models. This approach can be generalized to any scenario involving cross-modal alignment evaluation.
- **The "batch mutual reference" mechanism of Few-Shot Scoring** essentially introduces the idea of relative scoring, solving the problem of vague absolute scoring criteria. This trick is reusable in any LLM-based grading scenario.
- The proposed 9-dimension evaluation system is comprehensive, covering layers hierarchically from the object level to the video level.

## Limitations & Future Work

- Relying on GPT-4o as the evaluation model incurs high API costs, and inconsistency issues may arise due to model version updates.
- The prompt suite scale is relatively limited (419 prompts), which provides insufficient coverage for some long-tail scenarios.
- It only evaluates text-to-video generation, leaving emerging tasks such as image-to-video and video editing unaddressed.
- Few-Shot Scoring requires generating multiple videos from the same prompt, which escalates evaluation overhead.
- The Spearman correlation coefficient for the motion quality dimension remains relatively low (0.514), suggesting that dynamic quality evaluation is still a challenging bottleneck.

## Related Work & Insights

- **vs VBench**: VBench primarily relies on a combination of computational metrics, yielding lower alignment with human preferences; Video-Bench fully utilizes MLLMs and significantly boosts alignment through Chain-of-Query and Few-Shot Scoring.
- **vs CompBench**: CompBench only uses single-turn LLM evaluation for the alignment dimension, whereas the multi-turn iteration of Video-Bench improves performance on Video-Condition Alignment by an average of 0.093.
- **vs EvalCrafter**: EvalCrafter fits the relationship between user ratings and metrics via linear regression, which is a more indirect approach.

## Rating

- Novelty: ⭐⭐⭐⭐ Chain-of-Query and Few-Shot Scoring represent effective methodological innovations, though the overall framework is compositional.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, utilizing 35K human annotations, 7 generative models, 9 dimensions, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich visualizations, though some notation is slightly heavy.
- Value: ⭐⭐⭐⭐ Provides a robust baseline for video generation evaluation, with the Chain-of-Query concept demonstrating strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VMBench: A Benchmark for Perception-Aligned Video Motion Generation](../../ICCV2025/video_generation/vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)
- [\[CVPR 2025\] VEU-Bench: Towards Comprehensive Understanding of Video Editing](veu-bench_towards_comprehensive_understanding_of_video_editing.md)
- [\[ICML 2026\] V2V-Bench: A Comprehensive Benchmark for Video-to-Video Generation Evaluation](../../ICML2026/video_generation/v2v-bench_a_comprehensive_benchmark_for_video-to-video_generation_evaluation.md)
- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation](../../CVPR2026/video_generation/vga-bench_a_unified_benchmark_and_multi-model_framework_for_video_aesthetics_and.md)
- [\[CVPR 2025\] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation](tokenmotion_decoupled_motion_control_via_token_disentanglement_for_human-centric.md)

</div>

<!-- RELATED:END -->
