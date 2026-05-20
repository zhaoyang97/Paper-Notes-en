---
title: >-
  [Paper Note] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark
description: >-
  [CVPR 2026][Image Generation][unified models] This paper introduces RealUnify, the first benchmark specifically designed to evaluate the bidirectional synergy between understanding and generation capabilities in unified…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "unified models"
  - "capability synergy"
  - "understanding and generation"
  - "benchmark"
  - "bidirectional evaluation"
date: 2026-05-08
content_hash: 45504145685f49cc
---

# RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark

**Conference**: CVPR 2026
**arXiv**: [2509.24897](https://arxiv.org/abs/2509.24897)  
**Code**: [https://github.com/FrankYang-17/RealUnify](https://github.com/FrankYang-17/RealUnify)  
**Area**: Image Generation
**Keywords**: unified models, capability synergy, understanding and generation, benchmark, bidirectional evaluation

## TL;DR
This paper introduces RealUnify, the first benchmark specifically designed to evaluate the bidirectional synergy between understanding and generation capabilities in unified models. Through 1,000 manually annotated instances and a dual evaluation protocol (direct and stepwise), it reveals that current unified models, despite possessing both understanding and generation capabilities, still fail to achieve genuine capability synergy in end-to-end scenarios.

## Background & Motivation
1. **Background**: Multimodal unified models (e.g., BAGEL, Janus-Pro) that integrate visual understanding (VQA) and visual generation (T2I) into a single architecture have emerged as a major direction toward general-purpose AI.
2. **Limitations of Prior Work**: Existing evaluation frameworks (e.g., MME-Unify, UniEval) primarily assess understanding and generation separately, or simply combine the two task types, making it impossible to determine whether unified models truly achieve a "1+1>2" synergistic effect.
3. **Key Challenge**: The greatest value of unified models lies in the bidirectional gain between understanding and generation—using understanding to guide generation and using generation to assist understanding. However, no rigorous benchmark currently exists to verify whether this bidirectional synergy is genuine.
4. **Goal**: To design a benchmark that precisely measures the degree of capability synergy in unified models, answering the question of whether unification truly yields stronger performance than isolated capabilities.
5. **Key Insight**: Tasks are designed such that they can only be completed by leveraging understanding–generation synergy, with stepwise evaluation protocols used to diagnose the source of performance bottlenecks.
6. **Core Idea**: Through carefully designed bidirectional synergy tasks and a direct/stepwise dual evaluation protocol, this work systematically examines for the first time whether unified models achieve genuine synergy between understanding and generation.

## Method

### Overall Architecture
RealUnify comprises 1,000 manually annotated instances spanning 10 categories and 32 subtasks. The core design is organized along two axes: **Understanding-Enhanced Generation (UEG)**—requiring reasoning (commonsense, logical, etc.) to guide image generation; and **Generation-Enhanced Understanding (GEU)**—requiring mental simulation or reconstruction to solve reasoning tasks. Evaluation employs both direct and stepwise protocols.

### Key Designs

1. **Understanding-Enhanced Generation (UEG) Task Design**:

    - **Function**: Evaluates whether a model can leverage understanding capabilities to improve generation quality.
    - **Mechanism**: Encompasses six task types—world knowledge (generating images requiring objective knowledge), commonsense reasoning (generating images consistent with everyday phenomena), mathematical reasoning (generating correct results after computation), logical reasoning (generation satisfying logical constraints), scientific reasoning (applying principles from physics/chemistry/biology), and code-to-image (generating corresponding images after parsing code logic). Each task requires the model to first "understand" and then "generate."
    - **Design Motivation**: Existing T2I benchmarks primarily focus on aesthetics and text alignment, rather than on whether models can apply knowledge and reasoning to complete complex generation tasks.

2. **Generation-Enhanced Understanding (GEU) Task Design**:

    - **Function**: Evaluates whether a model can leverage generation capabilities to assist visual understanding.
    - **Mechanism**: Encompasses four task types—mental reconstruction (reasoning about shuffled image patches to reconstruct and answer questions), mental tracking (tracking the state of colored line segments through multi-step transformations), attention focusing (highlighting key regions via generative means to aid recognition), and cognitive navigation (maze/map navigation requiring intermediate visual outputs to assist understanding).
    - **Design Motivation**: Tests whether models can improve understanding by "thinking in images" rather than relying solely on language-based reasoning.

3. **Dual Evaluation Protocol (Direct + Stepwise)**:

    - **Function**: Diagnoses the source of performance bottlenecks—whether failures stem from insufficient basic capabilities or from synergy integration failures.
    - **Mechanism**: Direct evaluation requires end-to-end task completion; stepwise evaluation decomposes tasks into independent understanding and generation stages (UEG: understand then generate; GEU: generate then understand). Comparing results across the two protocols reveals whether a model "lacks capability" or "has capability but cannot integrate it."
    - **Design Motivation**: End-to-end results alone cannot distinguish capability deficits from synergy failures. Stepwise evaluation reveals whether models possess the requisite capabilities but cannot spontaneously integrate them.

4. **Polling Evaluation for Generation Assessment**:

    - **Function**: Verifies the correctness of generated image content.
    - **Mechanism**: For images generated in UEG tasks, a list of verification questions is designed and Gemini 2.5 Pro is used as the judge model for polling-based evaluation, ensuring generated content aligns with the target.
    - **Design Motivation**: Directly assessing the correctness of generated images is more challenging than evaluating aesthetics, necessitating an automated, content-based verification mechanism.

### Loss & Training
This paper presents a benchmark study and does not involve model training. Regarding data construction: UEG tasks were manually designed by 10 human experts and cross-validated by 3 reviewers; GEU tasks were partially auto-generated and subsequently annotated by experts. The reliability of Gemini 2.5 Pro as the judge model was verified through agreement with human expert ratings.

## Key Experimental Results

### Main Results

| Model | UEG Direct | UEG Step | GEU Direct | GEU Step | Overall |
|-------|-----------|----------|-----------|----------|---------|
| Nano Banana (closed-source) | 63.0 | - | 31.8 | - | 50.5 |
| BAGEL (best open-source) | 32.7 | 47.7 | 39.3 | 35.8 | 35.3/42.9 |
| UniPic2 | 37.5 | 40.5 | 24.0 | 23.8 | 32.1/33.8 |
| OneCAT | 37.5 | 39.0 | 31.3 | 29.3 | 35.0/35.1 |
| Oracle (Gemini+GPT-Image) | - | 72.7 | - | 31.8 | - |

### Ablation Study

| Evaluation Protocol | BAGEL UEG | BAGEL GEU | Notes |
|--------------------|----------|----------|-------|
| Direct | 32.7 | 39.3 | End-to-end; spontaneous integration fails |
| Stepwise | 47.7 | 35.8 | UEG improves significantly; GEU degrades |
| Oracle (GT intermediate results) | Higher | Higher | Indicates basic capabilities exist but integration is insufficient |

### Key Findings
- **Stepwise evaluation yields substantial UEG gains**: BAGEL improves from 32.7% to 47.7%, indicating that the model possesses internal knowledge but cannot spontaneously integrate it into generation.
- **Stepwise evaluation degrades GEU performance**: Decomposition leads to lower performance, suggesting that models rely on understanding shortcuts in direct evaluation rather than genuinely leveraging generation capabilities.
- **Large open-source vs. closed-source gap**: On UEG, the best open-source model achieves 37.5% vs. 63.0% for the closed-source model; however, on GEU, open-source models (BAGEL: 39.3%) outperform the closed-source counterpart (31.8%).
- **Oracle upper bound remains far out of reach**: The combined expert model achieves 72.7% on UEG, while the best unified model reaches only 47.7% (stepwise), revealing a substantial gap.

## Highlights & Insights
- **Stepwise evaluation uncovers the "capability without utilization" phenomenon**: The most central finding is that models possess both understanding and generation capabilities but cannot spontaneously integrate them in end-to-end scenarios. This diagnostic evaluation design is transferable to other AI systems that require multi-capability synergy.
- **Discovery of "understanding shortcuts" in GEU tasks**: On tasks requiring "generate first, then understand," models bypass generation and directly answer via understanding; forcing generation in the stepwise setting leads to worse performance. This reveals a severe underutilization of generation capabilities in current models.
- **Polling evaluation mechanism**: Using a question list combined with LLM-based judgment to verify the correctness of generated images is better suited to knowledge-intensive generation evaluation than traditional FID/CLIP metrics.

## Limitations & Future Work
- Evaluation relies on Gemini 2.5 Pro as the judge model, introducing potential evaluation bias (despite reasonable agreement with human ratings).
- With only 1,000 instances, certain subtasks may have insufficient sample sizes to support statistical significance.
- No exploration of training methods to improve synergy capabilities—the work diagnoses the problem but proposes no solutions.
- Future work could explore specific training strategies (e.g., alternating training, synergy rewards) to promote genuine capability integration.

## Related Work & Insights
- **vs. MME-Unify**: The latter evaluates both understanding and generation but does not test their synergy; RealUnify specifically designs tasks that require synergy to complete.
- **vs. T2I-CoReBench/WISE**: These benchmarks preliminarily explore the benefit of understanding for generation, but are neither systematic nor bidirectional, and lack stepwise diagnostics.
- **vs. Expert model combinations**: The Oracle experiment demonstrates that simply combining the best expert models (Gemini + GPT-Image) achieves 72.7%, far surpassing any unified model, suggesting that the unified architecture itself is not the key factor—training strategies and inductive biases are.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First benchmark to systematically evaluate capability synergy in unified models; the stepwise evaluation protocol is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 unified models + 6 expert baselines, dual evaluation protocols, and judge reliability validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, rich figures and tables, convincing conclusions.
- **Value**: ⭐⭐⭐⭐ Points unified model research toward "what truly needs to be optimized."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [\[CVPR 2026\] Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
