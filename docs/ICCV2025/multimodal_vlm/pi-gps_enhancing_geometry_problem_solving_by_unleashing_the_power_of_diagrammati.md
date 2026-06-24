---
title: >-
  [Paper Note] Pi-GPS: Enhancing Geometry Problem Solving by Unleashing the Power of Diagrammatic Information
description: >-
  [ICCV 2025][Multimodal VLM][geometry problem solving] Pi-GPS leverages diagrammatic information to resolve ambiguities in textual descriptions. By introducing a lightweight Rectifier–Verifier module, it addresses a previously overlooked problem of textual ambiguity, achieving nearly 10% improvement over prior state-of-the-art neuro-symbolic methods on Geometry3K.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "geometry problem solving"
  - "textual ambiguity resolution"
  - "multimodal reasoning"
  - "neuro-symbolic methods"
  - "MLLM"
date: 2026-05-08
content_hash: 5d0f93d4b7a841b3
---

# Pi-GPS: Enhancing Geometry Problem Solving by Unleashing the Power of Diagrammatic Information

**Conference**: ICCV 2025
**arXiv**: [2503.05543](https://arxiv.org/abs/2503.05543)  
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: geometry problem solving, textual ambiguity resolution, multimodal reasoning, neuro-symbolic methods, MLLM

## TL;DR

Pi-GPS leverages diagrammatic information to resolve ambiguities in textual descriptions. By introducing a lightweight Rectifier–Verifier module, it addresses a previously overlooked problem of textual ambiguity, achieving nearly 10% improvement over prior state-of-the-art neuro-symbolic methods on Geometry3K.

## Background & Motivation

Geometry Problem Solving (GPS) requires models to derive correct answers from textual descriptions and corresponding geometric diagrams, making it one of the most challenging tasks in multimodal mathematical reasoning. It demands a comprehensive understanding of visual shapes, spatial relationships, symbolic abstraction, and logical inference.

**Taxonomy of existing approaches**:
- **Symbolic methods**: Rely on formal logic and theorem libraries for inference; highly interpretable but limited in rule coverage.
- **Neural methods**: Generate solution paths in a data-driven manner; flexible but lacking correctness guarantees.
- **Hybrid methods** (e.g., Inter-GPS, GeoDRL, E-GPS): Combine both paradigms via a parse-then-reason pipeline and represent the current mainstream direction.

**Core limitation — textual ambiguity neglected**: In existing hybrid methods, the text parser and diagram parser operate independently during the parsing stage, making it impossible to resolve textual ambiguities using diagrammatic information. For example:
- A text mentions "a shape" without specifying whether it is a triangle, rectangle, or circle.
- A text refers to "the area of the shaded region" without specifying which elements constitute the shaded region.
- A text describes an altitude relation `IsAltitudeOf(Line(C,P), Shape($))` without specifying the corresponding shape.

Such ambiguities are difficult to resolve from text alone but can often be trivially resolved with the aid of the diagram. Nevertheless, prior work has almost entirely overlooked this issue.

**Key Insight**: Design a lightweight micro-module that systematically resolves textual ambiguities using diagrammatic information, then feeds the disambiguated formal language into the subsequent reasoning stage. This approach is orthogonal to existing frameworks and can be seamlessly integrated as a plug-and-play component to boost the performance of any neuro-symbolic method.

## Method

### Overall Architecture

Pi-GPS consists of two major components: a **Parser** and a **Reasoner**. The core contribution of this paper — the Text Disambiguation Module — is embedded within the Parser. The overall pipeline proceeds as follows:

1. **Text Parser**: A regex-based rule parser that extracts a proposition set $P_T$ and the goal $t^*$ from the problem text.
2. **Diagram Parser**: PGDPNet is used to end-to-end parse the geometric diagram, extracting elements such as points, lines, and circles along with their logical relationships.
3. **Text Disambiguation Module** (core contribution): Resolves ambiguities in the text parsing output using diagrammatic information.
4. **Theorem Predictor**: An LLM (o3-mini) predicts the optimal theorem application sequence.
5. **Solver**: Executes symbolic reasoning following the predicted theorem order to produce the final answer.

### Key Designs

#### 1. Rectifier — MLLM-Based Ambiguity Resolution

- **Function**: Identifies unknown identifiers (unspecified elements denoted by `$`) in the text parsing output and resolves them by prompting an MLLM with diagrammatic information.
- **Mechanism**: Regex-based classification first identifies three types of ambiguity; type-specific prompts are then constructed and fed to an MLLM (e.g., GPT-4o) for resolution.
- **Three ambiguity types**:
    - **Unspecified points**: e.g., `CircumscribedTo(Square($), Circle($))` — a circumscribed square and inscribed circle are identified, but the specific vertices are unspecified.
    - **Unspecified shapes**: e.g., `IsAltitudeOf(Line(C,P), Shape($))` — an altitude is identified, but the corresponding shape is unspecified.
    - **Unspecified regions**: e.g., `Find(AreaOf(Shaded(Shape($))))` — the shaded area is to be computed, but the constituent elements of the shaded region are unspecified.
- **Design Motivation**: Tailored prompts (rather than generic ones) help the MLLM focus on specific ambiguity types; experiments show that generic prompts introduce additional errors.

#### 2. Verifier — Geometric Heuristic Consistency Checking

- **Function**: Validates the geometric consistency of the disambiguation output produced by the MLLM, filtering out results that violate geometric rules.
- **Mechanism**: Three heuristic rules are applied:
    - **Entity existence verification**: Checks whether the points and lines generated by the MLLM exist in the original diagram.
    - **Shape closure and ordering verification**: Checks whether the polygon vertices form a closed figure in the correct order, using a graph data structure to verify connectivity and node degrees.
    - **Geometric consistency verification**: Uses analytic geometry to verify whether the generated vertex coordinates are consistent with the target geometric shape.
- **Design Motivation**: MLLMs are prone to hallucination, and generating precise formal language is notoriously difficult (a single character or bracket error can invalidate the output). The Verifier employs an iterative feedback mechanism that returns failure information to the Rectifier for correction.

#### 3. Theorem Predictor — LLM-Based Theorem Ordering

- **Function**: Employs an LLM (o3-mini) to predict the optimal theorem application sequence.
- **Mechanism**: Theorem library knowledge is provided as part of the prompt; the model generates the most appropriate theorem application order based on the disambiguated text and diagram formal language.
- **Design Motivation**: Prior methods rely on annotated theorem sequences for training, which are costly to label and limited in coverage. Leveraging the generalization capability of LLMs reduces dependence on annotated data.

### Loss & Training

This method is primarily a rule-based and prompt-driven framework with no end-to-end training. The diagram parser PGDPNet is pretrained, and both the MLLM and LLM are used in a zero-shot manner.

## Key Experimental Results

### Main Results

| Category | Method | Geometry3K (Choice) | Geometry3K (Completion) | PGPS9K (Choice) | PGPS9K (Completion) |
|----------|--------|---------------------|------------------------|-----------------|---------------------|
| MLLM | GPT-4o | 58.6 | 34.8 | 51.0 | 33.3 |
| MLLM | Gemini 2 | 60.7 | 38.9 | 56.8 | 38.2 |
| Neuro-symbolic | Inter-GPS | 57.5 | 43.4 | — | — |
| Neuro-symbolic | GeoDRL | 68.4 | 57.9 | 66.7 | 55.6 |
| Neuro-symbolic | E-GPS | 67.9 | — | — | — |
| **Neuro-symbolic** | **Pi-GPS (Ours)** | **77.8** | **70.6** | **69.8** | **61.4** |

Pi-GPS outperforms GeoDRL by 9.4% (Choice) and E-GPS by 9.9% (Choice) on Geometry3K, and even surpasses human experts on certain subcategories (ratio).

### Ablation Study

| Configuration | Completion | Choice | Notes |
|---------------|-----------|--------|-------|
| Baseline (no module) | 60.7 | 70.6 | Parsing + exhaustive solving only |
| + Text Disambiguation Module | 68.9 | 76.6 | Disambiguation is the primary gain source (+6%) |
| + Theorem Predictor | 63.2 | 72.3 | Reduces inference steps |
| + Both | 70.6 | 77.8 | Full Pi-GPS |
| Rectifier (generic prompt) | 62.4 | 71.9 | Generic prompt degrades performance |
| Rectifier (tailored prompt) | 64.2 | 73.3 | Tailored prompt is effective |
| + Verifier | 70.6 | 77.8 | Verifier contributes 4–6% gain |

### Key Findings

- The Text Disambiguation Module is the primary driver of performance improvement, contributing a consistent gain of approximately 5–6%.
- The Verifier is critical: without it, the Rectifier may degrade performance due to MLLM hallucinations.
- Tailored prompts outperform generic prompts; generic prompts may introduce unnecessary modifications.
- Theorem prediction yields similar results across all tested LLMs, suggesting that this task is sufficiently tractable for modern LLMs.
- Using LLMs for theorem prediction is more accurate and interpretable than using them for direct end-to-end problem solving.

## Highlights & Insights

- **Value of problem identification**: Textual ambiguity is a long-overlooked yet critical issue; this paper systematically demonstrates its significant impact on GPS performance.
- **Modular design**: The Rectifier–Verifier paradigm generalizes to other tasks requiring multimodal ambiguity resolution.
- **Necessity of the Verifier**: Directly trusting MLLM outputs is unreliable; domain-knowledge-driven validation is essential to constrain generation.
- **Insight on LLM usage**: Employing LLMs for intermediate reasoning steps (theorem prediction) is more appropriate than direct end-to-end problem solving.

## Limitations & Future Work

- The text parser remains rule-based and handles certain syntactic variations poorly.
- The diagram parser struggles to recognize complex geometric relationships (e.g., tangent lines).
- The theorem library is incomplete, lacking certain critical theorems (e.g., interior angles of a regular hexagon are 120°).
- The disambiguation module relies on the zero-shot capability of MLLMs and is not sufficiently robust to formatting errors.
- Validation on larger-scale or higher-difficulty geometry datasets remains to be conducted.

## Related Work & Insights

- **AlphaGeometry2**: Demonstrates the potential of LLMs in geometric reasoning, but focuses primarily on language processing while neglecting diagrams.
- **GeoDRL**: Introduces deep reinforcement learning for theorem prediction and serves as an important baseline.
- **E-GPS**: Integrates top-down and bottom-up reasoning to reduce the number of solution steps.
- Research on ambiguity resolution in multimodal reasoning offers transferable insights for related areas such as VQA and chart understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ — Problem identification is novel, though the solution is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablations and sufficient comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with intuitive examples.
- Value: ⭐⭐⭐⭐ — Reveals a previously overlooked critical factor and provides reproducible improvement strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Survey of Deep Learning for Geometry Problem Solving](../../ACL2026/multimodal_vlm/a_survey_of_deep_learning_for_geometry_problem_solving.md)
- [\[ICLR 2026\] GeoBench: Rethinking Multimodal Geometric Problem-Solving via Hierarchical Evaluation](../../ICLR2026/multimodal_vlm/geobench_rethinking_multimodal_geometric_problem-solving_via_hierarchical_evalua.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](../../NeurIPS2025/multimodal_vlm/learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[ICCV 2025\] Information Density Principle for MLLM Benchmarks](information_density_principle_for_mllm_benchmarks.md)
- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)

</div>

<!-- RELATED:END -->
