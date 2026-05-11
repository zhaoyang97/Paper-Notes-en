---
title: >-
  [Paper Note] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Spatial logical reasoning] This paper proposes SpatiaLQA, a benchmark comprising 9,605 QA pairs across 241 real-world indoor scenes…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Spatial logical reasoning"
  - "VLM benchmark"
  - "scene graph"
  - "indoor scene understanding"
  - "multi-step reasoning"
date: 2026-05-08
content_hash: b21a64b7435cdada
---

# SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models

**Conference**: CVPR 2026  
**arXiv**: [2602.20901](https://arxiv.org/abs/2602.20901)  
**Code**: [https://github.com/xieyc99/SpatiaLQA](https://github.com/xieyc99/SpatiaLQA)  
**Area**: Multimodal VLM  
**Keywords**: Spatial logical reasoning, VLM benchmark, scene graph, indoor scene understanding, multi-step reasoning

## TL;DR
This paper proposes SpatiaLQA, a benchmark comprising 9,605 QA pairs across 241 real-world indoor scenes, systematically evaluates 41 VLMs on spatial logical reasoning, and introduces a recursive scene graph-assisted reasoning method to enhance VLMs' spatial logical reasoning capabilities.

## Background & Motivation
**Background**: VLMs have achieved strong performance on general VQA and logical reasoning tasks, yet remain inadequate in complex real-world scenarios that require the integration of spatial understanding and multi-step logical reasoning.

**Limitations of Prior Work**: Existing benchmarks focus either on spatial understanding (e.g., SpatialRGPT-Bench) or logical reasoning (e.g., MathVista) in isolation, lacking an evaluation framework that integrates both. Furthermore, EQA tasks target action execution rather than purely visual-semantic reasoning.

**Key Challenge**: Spatial logical reasoning demands that a model simultaneously possess precise spatial perception and rigorous multi-step causal reasoning; the fusion of these two capabilities has not been systematically studied in existing VLMs.

**Goal**: (a) Construct a comprehensive spatial logical reasoning benchmark; (b) systematically evaluate existing VLMs on this task; (c) propose improvement methods.

**Key Insight**: Decompose complex scenes into task-relevant scene graphs, enabling VLMs to focus on the spatial context surrounding target objects.

**Core Idea**: A recursive scene graph construction method progressively decomposes complex indoor scenes into task-relevant spatial relation graphs, thereby enhancing VLMs' multi-step spatial reasoning capabilities.

## Method

### Overall Architecture
Given an indoor scene image and a question requiring multi-step spatial reasoning, the method produces a sequence of logically coherent operation steps. The pipeline consists of three stages: (1) obtaining depth maps and segmentation maps via visual foundation models; (2) recursively constructing a scene graph centered on the target object; (3) feeding the scene graph together with the question into a VLM to generate the final answer.

### Key Designs

1. **SpatiaLQA Benchmark Construction**:

    - **Function**: Constructs 9,605 QA pairs derived from 241 real-world indoor scenes.
    - **Mechanism**: Three-stage data collection — 2,401 pairs annotated manually, 2,251 pairs obtained via subgraph extraction augmentation, and 4,953 pairs generated via graph expansion augmentation.
    - **Design Motivation**: Directly constructing large-scale spatial logical reasoning data is prohibitively costly; subgraph extraction based on logical dependencies and graph expansion enable efficient augmentation.

2. **Evaluation Metric Design**:

    - **Function**: Step-level matching based on GPT-4o and the Hungarian algorithm.
    - **Mechanism**: GPT-4o first generates a matching matrix between predicted steps and annotated steps; the Hungarian algorithm then determines the optimal one-to-one assignment; precision and recall are finally computed for both content and preconditions.
    - **Design Motivation**: Open-ended multi-step answers cannot be evaluated with conventional accuracy metrics, necessitating step-level semantic matching.

3. **Recursive Scene Graph-Assisted Reasoning (RSGAR)**:

    - **Function**: Leverages Depth Anything V2 and SAM to obtain depth and segmentation information, then recursively constructs a scene graph centered on the target object.
    - **Mechanism**: The task-specified object serves as the initial source node; the VLM identifies objects in direct contact with it along with their spatial relations, forming scene graph nodes and edges; the process iterates until a maximum number of iterations is reached.
    - **Design Motivation**: Directly processing complex scenes often causes VLMs to overlook critical spatial relations; progressive decomposition directs the model's attention to local spatial contexts.

### Loss & Training
RSGAR is an inference-time method that requires no additional training, augmenting reasoning directly using pretrained VLMs and visual foundation models.

## Key Experimental Results

### Main Results

| Model | $F_c$ (Content F1) | $F_p$ (Precondition F1) |
|------|---------------|-------------------|
| Human | 97.6 | 92.5 |
| GPT-4o | 52.5 | 19.2 |
| Claude 3.5 Sonnet | 46.3 | 15.8 |
| Gemini 2.0 Flash | 44.1 | 14.7 |
| GPT-4o + RSGAR | **56.8** | **22.4** |
| InternVL2-26B | 38.2 | 12.1 |

### Ablation Study

| Configuration | $F_c$ | $F_p$ | Note |
|------|-------|-------|------|
| GPT-4o (baseline) | 52.5 | 19.2 | No scene graph assistance |
| + Depth map | 53.8 | 20.1 | Depth information only |
| + Segmentation map | 54.2 | 20.5 | Segmentation information only |
| + RSGAR (1 round) | 55.1 | 21.3 | Single-round scene graph |
| + RSGAR (3 rounds) | 56.8 | 22.4 | 3 recursive rounds, best performance |

### Key Findings
- Even the strongest model, GPT-4o, achieves only approximately $F_c = 52.5\%$ on spatial logical reasoning, a substantial gap from the human score of 97.6%.
- All VLMs perform considerably worse on precondition reasoning $F_p$, indicating that understanding inter-step dependencies is the core challenge.
- Model performance degrades sharply as the number of answer steps increases, confirming that multi-step reasoning is the primary bottleneck.
- RSGAR yields consistent improvements across multiple VLMs, validating the effectiveness of scene graph decomposition.

## Highlights & Insights
- **Evaluation Framework Design**: Combining GPT-4o for semantic matching with the Hungarian algorithm for optimal alignment elegantly addresses the challenge of evaluating open-ended multi-step responses. This two-stage evaluation paradigm is transferable to other multi-step reasoning tasks.
- **Recursive Scene Graph Decomposition**: Transforming end-to-end complex spatial reasoning into progressively focused sub-problem solving is analogous to a spatial counterpart of chain-of-thought reasoning, and cleverly exploits the complementary strengths of visual foundation models.
- **Data Augmentation Strategy**: Subgraph extraction and graph expansion efficiently generate large volumes of training data from limited annotations while preserving logical consistency.

## Limitations & Future Work
- The benchmark covers only indoor scenes; complex outdoor environments (e.g., traffic scenarios, construction sites) are not addressed.
- RSGAR depends on external visual models (SAM, Depth Anything), introducing additional computational overhead and error propagation.
- The maximum number of iterations in scene graph construction is fixed, lacking an adaptive termination mechanism.
- Incorporating spatial logical reasoning capabilities into VLM training has not been explored; the proposed method enhances reasoning only at inference time.

## Related Work & Insights
- **vs. SpatialRGPT-Bench**: SpatialRGPT focuses solely on spatial understanding without involving multi-step logical reasoning; SpatiaLQA extends this by incorporating step-level dependency relations.
- **vs. EmbodiedBench**: EmbodiedBench targets embodied execution with a predefined action primitive output space; SpatiaLQA focuses on open-vocabulary reasoning processes.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes a new task definition and large-scale benchmark, filling the gap in spatial logical reasoning evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 41 VLMs covering mainstream models with comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, though some descriptions are verbose.
- Value: ⭐⭐⭐⭐ The benchmark resource is of significant value to the community; the proposed method has considerable room for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[CVPR 2026\] HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)
- [\[CVPR 2026\] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)
- [\[CVPR 2026\] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)

</div>

<!-- RELATED:END -->
