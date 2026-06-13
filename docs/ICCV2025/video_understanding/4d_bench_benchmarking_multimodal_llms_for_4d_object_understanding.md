---
title: >-
  [Paper Note] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding
description: >-
  [ICCV 2025][Video Understanding][4D understanding] This paper introduces 4D-Bench, the first benchmark for evaluating multi-modal large language models (MLLMs) on 4D object (dynamic 3D object) understanding…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "4D understanding"
  - "MLLM evaluation"
  - "spatiotemporal reasoning"
  - "benchmark"
  - "video question answering"
date: 2026-05-08
content_hash: 993b0b54c00e2150
---

# 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding

**Conference**: ICCV 2025
**arXiv**: N/A (CVF Open Access)  
**Code**: [https://4dbench.github.io/](https://4dbench.github.io/)  
**Area**: Video Understanding
**Keywords**: 4D understanding, MLLM evaluation, spatiotemporal reasoning, benchmark, video question answering

## TL;DR

This paper introduces 4D-Bench, the first benchmark for evaluating multi-modal large language models (MLLMs) on 4D object (dynamic 3D object) understanding, comprising two tasks: 4D object question answering and 4D object captioning. The benchmark reveals that even GPT-4o achieves only 63% accuracy on simple 4D objects (vs. 91% human baseline), with particularly weak performance on object counting and temporal understanding.

## Background & Motivation

1. **Background**: MLLMs such as GPT-4o and Qwen2-VL have achieved remarkable progress in 2D image and video understanding, yet 4D (3D + time) object understanding remains almost entirely unevaluated in a systematic manner. 4D digital assets are increasingly important in digital twins, augmented reality, gaming, and related domains.

2. **Limitations of Prior Work**: No publicly available standardized benchmark exists for evaluating MLLMs on 4D object understanding. Existing 3D language understanding benchmarks (e.g., ScanQA) focus on static 3D scenes and ignore motion; 2D video benchmarks ignore multi-view understanding. Neither captures the joint multi-view spatial-temporal reasoning required for 4D object understanding.

3. **Key Challenge**: 4D object understanding demands simultaneous multi-view spatial reasoning (observing objects from multiple perspectives to resolve occlusion and ambiguity) and temporal reasoning (tracking dynamic changes over time), posing entirely new challenges to MLLMs trained solely on 2D images or videos. Furthermore, large-scale 4D–text paired data is extremely scarce, making direct training of 4D understanding models difficult.

4. **Goal**: To construct a high-quality 4D object understanding benchmark that systematically evaluates the capabilities and shortcomings of existing MLLMs across multiple dimensions of 4D understanding, thereby providing directions for future improvement.

5. **Key Insight**: Rather than building a dedicated 4D understanding model, the paper leverages existing MLLM image/video understanding capabilities by rendering 4D objects as multi-view videos. The benchmark also incorporates synthetic counterfactual data (e.g., spiders with six legs) to test whether models genuinely understand the input rather than relying on prior knowledge.

6. **Core Idea**: By designing questions that require joint multi-view and temporal reasoning, and by collecting high-quality human-annotated descriptions, the paper constructs a comprehensive benchmark capable of diagnosing MLLMs across multiple dimensions of 4D understanding: appearance, action, counting, spatial relationships, and temporal relationships.

## Method

### Overall Architecture

The construction pipeline of 4D-Bench consists of four stages: (1) **4D data collection** — rendering multi-view videos of tens of thousands of dynamic 3D objects from Objaverse-XL; (2) **data cleaning** — filtering low-quality samples via motion analysis and visual quality assessment; (3) **annotation** — designing multi-view temporal reasoning questions for the QA task (MLLM-assisted generation with human verification) and collecting five independent human annotations per object for the captioning task; (4) **evaluation** — a hybrid evaluation framework combining traditional metrics with GPT-4o scoring.

### Key Designs

1. **4D Object QA**:

    - **Function**: Evaluates MLLMs across five sub-tasks — Appearance, Action, Object Counting, Spatial Relationship, and Temporal Relationship.
    - **Mechanism**: A total of 751 four-choice questions covering 736 4D objects. Questions are carefully designed so that answering them requires integrating both multi-view and temporal information. The annotation pipeline combines MLLM generation with multi-round filtering (Qwen2-VL 7B format checking → text-only blind filtering → human review).
    - **Design Motivation**: Each sub-task targets a distinct dimension of 4D understanding; for instance, counting requires cross-view fusion to resolve occlusion, while temporal tasks require tracking object evolution over time.

2. **4D Object Captioning**:

    - **Function**: Evaluates MLLMs' ability to generate descriptions of 4D object appearance and actions.
    - **Mechanism**: 580 representative 4D objects are curated from approximately 8,000 candidates, each annotated independently by five annotators. Evaluation employs traditional n-gram metrics (BLEU, ROUGE, etc.), embedding-based metrics (BERTScore), and GPT-4o scoring along two dimensions (GPT-Appearance and GPT-Action, each scored 0–5).
    - **Design Motivation**: The captioning task requires integrating multi-view appearance information with temporal action information, making it more comprehensive than QA while also more challenging to evaluate.

3. **Counterfactual Data and Robustness Evaluation**:

    - **Function**: Tests whether MLLMs genuinely understand visual input or merely rely on prior knowledge.
    - **Mechanism**: The dataset includes synthetic objects that violate common sense (e.g., spiders with six legs). All evaluated MLLMs, including GPT-4o, incorrectly report eight legs, indicating reliance on training-data priors rather than genuine visual understanding. Ablations on input ordering robustness (temporal-first vs. view-first) and timestamp information are also conducted.
    - **Design Motivation**: Exposes the fundamental problem of MLLMs that "appear to understand but are actually memorizing."

### Loss & Training

This paper is a benchmarking study and does not involve model training. Evaluation uses a standard input configuration of $K=3$ views with $N=6$ frames per view.

## Key Experimental Results

### Main Results

| Model | Counting (%) | Temporal (%) | Action (%) | Spatial (%) | Appearance (%) | Overall (%) |
|---|---|---|---|---|---|---|
| GPT-4o | 44.09 | 59.29 | 63.55 | 69.40 | 77.21 | **62.98** |
| Gemini 1.5 Pro | 46.46 | 58.57 | 59.35 | 64.18 | 68.38 | 59.52 |
| Qwen2-VL 72B | 45.67 | 55.71 | 58.41 | 61.19 | 72.06 | 58.72 |
| LLaVA-Video 7B | 42.52 | 55.00 | 52.80 | 56.72 | 78.68 | 56.86 |
| **Human Baseline** | **88.98** | **89.29** | **94.39** | **91.04** | **89.71** | **91.08** |
| Model Average | 37.29 | 49.29 | 49.37 | 53.57 | 63.92 | 50.69 |

### Ablation Study

| Configuration | GPT-Appearance | GPT-Action | GPT-Eval |
|---|---|---|---|
| GPT-4o Captioning | 3.507/5 | 3.258/5 | 3.382/5 |
| Human Captioning | 3.772/5 | 3.879/5 | 3.826/5 |
| 1 view → 3 views | 2.79 → 2.98 | — | ↑0.19 |
| 1 frame → 6 frames | — | — | 2.48 → 2.96 |

### Key Findings

- **Object counting is the greatest weakness**: All models average only 37.29% on the counting sub-task, far below other sub-tasks, reflecting the difficulty of cross-view fusion and correspondence establishment for MLLMs.
- **Appearance understanding >> Action understanding**: Models perform reasonably on appearance and spatial understanding (63.92% and 53.57%), but are notably weaker on action and temporal understanding (49.37% and 49.29%).
- **Open-source models approach closed-source on appearance but lag on action**: LLaVA-Video 7B achieves 78.68% on appearance, surpassing GPT-4o, but only 52.80% on action.
- **CoT degrades performance**: Chain-of-Thought prompting leads to a 9.72% accuracy drop for Qwen2-VL 7B, indicating that conventional language-based CoT is not suitable for visual reasoning.

## Highlights & Insights

- **First 4D understanding benchmark**: Fills a gap in MLLM evaluation along the 4D dimension and provides a systematic diagnostic framework, offering significant value for understanding multi-view spatiotemporal reasoning capabilities.
- **Clever counterfactual design**: Using out-of-distribution synthetic objects to distinguish genuine visual comprehension from statistical guessing — an OOD evaluation paradigm transferable to other benchmark designs.
- **Analysis of view and frame count effects**: Performance plateaus or even degrades beyond three views and six frames, suggesting that long-context processing and information selection are key bottlenecks for current MLLMs.

## Limitations & Future Work

- The dataset scale is limited (751 QA + 580 Caption), which may affect statistical significance.
- Only multi-view video is used as the 4D representation; native 3D representations such as point clouds are not explored.
- The synthetic data distribution may diverge from real-world scenarios, potentially limiting the generalizability of evaluation results.
- Future directions include visual chain-of-thought reasoning, adaptive view selection, and multi-modal inputs combining point clouds with video.

## Related Work & Insights

- **vs. VSI-Bench**: VSI-Bench evaluates spatial intelligence; 4D-Bench adds the temporal dimension, making the two complementary.
- **vs. MVBench/Video-MME**: These 2D video benchmarks do not require multi-view reasoning, whereas 4D-Bench demands cross-view information fusion.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic 4D object understanding benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 14 models with multi-faceted analysis
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and in-depth analysis
- Value: ⭐⭐⭐⭐ Reveals significant deficiencies of MLLMs in 4D understanding

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)

</div>

<!-- RELATED:END -->
