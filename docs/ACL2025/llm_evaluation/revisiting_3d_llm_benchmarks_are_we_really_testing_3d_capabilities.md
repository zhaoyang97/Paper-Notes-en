---
title: >-
  [Paper Note] Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?
description: >-
  [ACL 2025][LLM Evaluation][3D LLM] This work reveals the "2D-Cheating" issue in 3D LLM evaluations—where 2D VLMs outperform 3D SOTA on certain benchmarks after rendering point clouds into images. This demonstrates that these benchmarks fail to effectively evaluate genuine 3D understanding capabilities, based on which design principles for effective 3D evaluation are proposed.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "3D LLM"
  - "2D-Cheating"
  - "Benchmark Evaluation"
  - "Point Cloud"
  - "Vision-Language Model"
date: 2026-05-08
content_hash: b285125dcf7a01f0
---

# Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?

**Conference**: ACL 2025  
**arXiv**: [2502.08503](https://arxiv.org/abs/2502.08503)  
**Code**: [https://github.com/JiaheJin/VLM3D](https://github.com/JiaheJin/VLM3D)  
**Area**: LLM Evaluation  
**Keywords**: 3D LLM, 2D-Cheating, Benchmark Evaluation, Point Cloud, Vision-Language Model

## TL;DR

This work reveals the "2D-Cheating" issue in 3D LLM evaluations—where 2D VLMs outperform 3D SOTA on certain benchmarks after rendering point clouds into images. This demonstrates that these benchmarks fail to effectively evaluate genuine 3D understanding capabilities, based on which design principles for effective 3D evaluation are proposed.

## Background & Motivation

The development of 3D LLMs aims to enable language models to understand the real 3D physical world. However, due to the extreme scarcity of 3D training data, many methods rely on existing 2D VLMs and LLMs to generate annotations for 3D data. This raises a fundamental question: **What actual capabilities do 3D LLMs possess that truly distinguish them from 2D VLMs?**

The authors propose the concept of "2D-Cheating": if ordinary 2D VLMs can easily solve certain "3D tasks" once the point cloud is rendered into images, then these tasks are not actually evaluating genuine 3D capabilities. The existence of this issue implies that current 3D LLM benchmarks may create a false sense of progress in the community—some seemingly excellent 3D models might just be performing tasks that can be fully resolved in 2D.

## Method

### Overall Architecture

The VLM3D pipeline is proposed: Point cloud rendered as images $\rightarrow$ few-shot enhanced query $\rightarrow$ VLM input. By testing the performance of VLM3D on multiple 3D LLM benchmarks and comparing it with 3D SOTA, benchmarks suffering from the 2D-Cheating problem are identified.

### Key Designs

1. **Viewpoint Selection**: Viewpoints are a critical bottleneck for 2D models to understand 3D scenes. The authors hypothesize that the key limitations of 2D models stem from viewpoint dependency: (i) blind spots outside the viewpoint; (ii) occlusion and overlap; (iii) the lack of multi-faceted geometry in single-surface capture. Three strategies are designed for different complexities:

    - **Single View**: Objects use a fixed viewpoint, scenes use the Bird's-Eye View (BEV), serving as the base configuration.
    - **Multi View**: Images are rendered from four directions (East, West, South, North) and combined, theoretically providing more complete 3D information.
    - **Oracle View**: Uses the Best-of-N method—sampling 20 responses from each of 5 viewpoints, using the viewpoint with the highest average score as the oracle. This design removes the randomness of "guessing correctly" (by averaging multiple samplings) and explores the performance upper bound of VLMs.

2. **Human-Intuition-Selection (HIS)**: Computes the centroid of relevant objects and uses heuristic algorithms to select the optimal viewpoint. Although this is a more realistic viewpoint selection method, experiments show that its performance is far inferior to Best-of-N. Reasons include the difficulty of intuitive viewpoint selection in complex scenes, occlusion issues, and the fact that many questions rely on common sense rather than 3D details.

3. **Design Principles for Effective 3D Evaluation**: Based on experimental analysis, four principles are proposed:

    - Choose complex point clouds (scenes rather than simple objects)
    - Tasks should go beyond surface-level information and delve into 3D structural details
    - Avoid overly general questions, testing genuine understanding of the current 3D input
    - Evaluation methodology: Include multiple reasonable answers, using LLM-based evaluation instead of text similarity metrics

## Key Experimental Results

### Main Results—Object Point Cloud Benchmarks

| Benchmark | Metric | 3D SOTA | VLM3D (GPT-4o) | Gain |
|------|------|---------|----------------|------|
| 3D MM-Vet | LLM-eval | 43.2 | **58.1** | +14.9 |
| ObjaverseXL-LVIS Caption | BLEU-1 | 32.2 | **36.2** | +4.0 |
| ObjaverseXL-LVIS Caption | ROUGE-L | 35.5 | **36.8** | +1.3 |
| ObjaverseXL-LVIS Caption | CIDEr | 78.0 | **79.3** | +1.3 |

VLM-only using a single-view image comprehensively outperforms 3D SOTA $\rightarrow$ these object benchmarks suffer from severe 2D-Cheating.

### Main Results—Scene Point Cloud Benchmarks

| Benchmark | Metric | 3D Baseline | 3D SOTA | VLM3D Single View | VLM3D Oracle |
|------|------|-------------|---------|-------------|-------------|
| ScanQA | METEOR | 13.1 | 20.0 | 12.8 (-0.3) | 28.2 (+15.1) |
| ScanQA | CIDEr | 64.9 | 101.4 | 51.2 (-13.7) | 71.2 (+6.3) |
| SQA3D | EM | 47.2 | 52.6 | 42.2 (-5.0) | - |

In scene benchmarks, VLM single-view is inferior to 3D models $\rightarrow$ these benchmarks can better evaluate 3D capabilities.

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Single View vs Multi View | CIDEr: 51.2 $\rightarrow$ 54.0 | Multi-view brings only minor improvement |
| Single View vs Oracle | CIDEr: 51.2 $\rightarrow$ 71.2 | Oracle significantly outperforms 3D Baseline |
| HIS vs Best-of-N | HIS is far worse than BoN | Intuitive viewpoint selection is unreliable in complex scenes |

### Principle Validation

| Benchmark | Pass Rate: Point Cloud Selection | Pass Rate: Task Focus | Pass Rate: All Three Passed |
|------|-----------------|-----------------|-------------------|
| 3D MM-Vet | 53.9% | 25.9% | 16.0% |
| ObjaverseXL-LVIS | 36.3% | 19.0% | 7.3% |
| ScanQA | 69.7% | 76.5% | 62.9% |
| SQA3D | 86.2% | 91.6% | 81.9% |

Benchmarks with low pass rates (object benchmarks) are exactly those where VLMs outperform 3D SOTA, validating the effectiveness of the principles.

### Key Findings

1. **Object point cloud tasks basically do not require 3D representation**: VLMs can outperform 3D SOTA relying solely on rendered images, exposing the invalidity of these benchmarks due to simple object structures and tasks only requiring surface-level information.
2. **VLMs struggle to form a unified 3D understanding from multiple views**: A combination of 4 views theoretically contains sufficient information, but VLMs actually achieve only marginal gains, indicating that current VLMs lack multi-view fusion capabilities.
3. **A good viewpoint can significantly boost VLM performance**: The Oracle viewpoint allows VLMs to outperform the 3D Baseline, though still falling short of 3D SOTA, showing that even with the best viewpoint, VLMs still face deficiencies in certain tasks.
4. **Many 3D questions are actually answered using common sense**: Part of the reason HIS failed is that many questions do not require 3D details; using world knowledge + random viewpoint actually scores higher.

## Highlights & Insights

- **Accurate concept of "2D-Cheating"**: Captures the core issue in 3D evaluation with a concise concept that is easy to understand and widely influential.
- **Valuable negative results**: Revealing the deficiencies of existing benchmarks is just as important as proposing new ones, helping the community avoid investing resources in ineffective directions.
- **Actionable design principles**: The four principles are highly executable and quantitatively validated by automatically evaluating pass rates using LLMs, rather than remaining at a qualitative discussion level.
- **Clever methodology**: The Best-of-N design for Oracle View controls randomness by averaging multiple samplings, ensuring both the reliability of the upper-bound estimation and avoiding overly optimistic conclusions.

## Limitations & Future Work

- Only two VLMs, GPT-4o and Qwen2-VL-72B, are used, which may not fully represent the capabilities of all VLMs.
- VLMs are only adapted to 3D tasks in a few-shot manner, whereas 3D LLMs are trained on the corresponding training set, making the comparison not completely fair.
- Rendering point clouds into images inherently loses information (compared to real photos), which might underestimate the actual capabilities of VLMs.
- Only principles are proposed without constructing a new benchmark that complies with them, which is the most direct subsequent work.
- The analysis of 2D-Cheating is mainly focused on QA/Caption tasks, and does not cover downstream tasks such as detection and segmentation.

## Related Work & Insights

- Relation to 3D QA benchmarks like ScanQA and SQA3D: Rather than proposing a new benchmark, this paper reflects on the validity of existing benchmarks.
- Insight 1: In any "high-dimensional input" domain (3D, video, etc.), one should verify whether low-dimensional proxies (images, sampled frames) can achieve equivalent performance to ensure evaluation validity.
- Insight 2: The 3D LLM community needs to more clearly decouple and separately evaluate 1D (language), 2D (visual), and 3D capabilities.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The "2D-Cheating" concept is novel and far-reaching, directly addressing the core pain points of 3D LLM evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers object/scene benchmarks, multiple viewpoint strategies, and quantitative validation of principles, though the variety of VLMs is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear concepts, rigorous argumentation logic, and intuitive charts.
- **Value**: ⭐⭐⭐⭐⭐ — Plays an important corrective role in the evaluation methodology of the 3D LLM community; the four principles will influence future benchmark designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](../../ICCV2025/llm_evaluation/3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)
- [\[ICCV 2025\] SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](../../ICCV2025/llm_evaluation/sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)
- [\[ACL 2025\] ChatBench: From Static Benchmarks to Human-AI Evaluation](chatbench_from_static_benchmarks_to_human-ai_evaluation.md)
- [\[ACL 2025\] ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities](onebench_to_test_them_all_sample-level_benchmarking_over_open-ended_capabilities.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)

</div>

<!-- RELATED:END -->
