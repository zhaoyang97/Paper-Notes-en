---
title: >-
  [Paper Note] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs
description: >-
  [ICLR 2026][LLM Reasoning][Geometric Reasoning] This paper formalizes the Program-to-Geometry task and proposes GeoGramBench (500 problems)…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Geometric Reasoning"
  - "Program-to-Geometry"
  - "Benchmark"
  - "Spatial Reasoning"
  - "Asymptote Code"
date: 2026-05-08
content_hash: 31382a635e9f3b2b
---

# GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs

**Conference**: ICLR 2026
**arXiv**: [2505.17653](https://arxiv.org/abs/2505.17653)  
**Code**: [GitHub](https://github.com/LiAuto-DSR/GeoGramBench)  
**Area**: LLM Reasoning
**Keywords**: Geometric Reasoning, Program-to-Geometry, Benchmark, Spatial Reasoning, Asymptote Code

## TL;DR

This paper formalizes the Program-to-Geometry task and proposes GeoGramBench (500 problems), evaluating 19 frontier LLMs on their ability to construct geometric representations from procedural drawing code and reason over them using a three-level geometric complexity taxonomy. Even GPT-5 achieves only 39.26% accuracy at the highest abstraction level, revealing fundamental limitations in LLM spatial abstraction.

## Background & Motivation

**Background**: Spatial reasoning is a foundational capability for both human cognition and AI, underpinning applications such as robotics, autonomous navigation, and automated design. LLMs have attracted broad attention for interpreting geometric transformations and spatial relations, yet their ability to perform geometric reasoning from procedural code remains largely overlooked.

**Limitations of Prior Work**: Existing benchmarks (e.g., MathVerse, GeoSense, Euclid) focus on visual geometric understanding; while MATH-500 and AIME24 include a small number of Asymptote-based problems, they lack systematic Program-to-Geometry evaluation. More critically, existing benchmarks fail to identify **answer leakage** in code—where code parameters directly or indirectly expose the answer.

**Key Challenge**: Preliminary studies indicate a significant performance drop when LLMs transition from code to spatial reasoning. DeepSeek-R1 shows accuracy drops of 23.5% (AIME24) and 10.9% (MATH-500) on geometry problems containing Asymptote code ($\mathbb{P}_{TC}$) compared to pure-text problems ($\mathbb{P}_T$). GPT-o1 and QwQ-32B exhibit similar trends.

**Goal**: This paper formalizes the Program-to-Geometry task definition, proposes GeoGramBench—a curated benchmark of 500 geometry problems with procedural drawing code—and introduces a three-level geometric complexity taxonomy in place of traditional reasoning-difficulty classifications.

## Method

### Overall Architecture

**Task Definition**: Given a textual description and geometric drawing code (Asymptote/Matplotlib), the model must parse the code to construct an internal geometric representation and reason over it to produce a numerical answer (length/area/volume/angle/ratio/count).

**Taxonomy** based on geometric complexity rather than reasoning steps:
1. **Primitive Recognition**: 1–2 geometric primitives (points/lines/arcs/circles/polygons), focusing on basic attributes such as length, area, and angle.
2. **Local Relation Composition**: Multiple local geometric elements requiring identification and integration of spatial relations among sub-components.
3. **Global Abstract Integration**: Involves spatial orientation, parameterization, recursion, 3D objects, composite structures, and advanced geometric operations (rotation/folding/projection).

### Key Design 1: Answer Leakage Prevention

**Function**: Ensures that models cannot directly obtain the answer by inspecting the code, requiring genuine geometric reasoning.

**Mechanism**: Two categories of leakage are identified:
- **Direct leakage**: Answers are explicitly encoded as coordinate values (e.g., circle radius, segment length); addressed by rescaling coordinates while preserving geometric shape.
- **Indirect leakage**: Answers can be derived from code parameters or formulae; addressed by modifying or obfuscating key code parameters.

**Design Motivation**: A large number of Asymptote problems in MATH-500 were found to directly embed the answer in code; without remediation, evaluation validity is compromised. Two rounds of expert verification by four annotators (master's degree or above in mathematics) ensured that no answer can be obtained through code inspection.

### Key Design 2: Validation of the Geometric Complexity Taxonomy

**Function**: Demonstrates that geometric complexity—rather than the number of reasoning steps—is the primary challenge in Program-to-Geometry tasks.

**Mechanism**: QwQ-32B is evaluated on MATH-500 problems stratified by both reasoning complexity (per MATH-500 annotations) and geometric complexity:
- Pure-text problems ($\mathbb{P}_T$): accuracy decreases as reasoning complexity increases—consistent with traditional benchmarks.
- Code-augmented problems ($\mathbb{P}_{TC}$): accuracy is **largely independent of reasoning complexity** but decreases significantly as geometric complexity increases.

**Design Motivation**: Traditional step-count-based taxonomies (high school → competition level) are unsuitable for this task. The geometric complexity taxonomy more accurately captures model bottlenecks.

### Data Construction Pipeline

Approximately 905K candidate problems are aggregated from three open-source mathematical datasets (NuminaMath-1.5, HARP, Omni-MATH) → filtered to 9,260 problems containing Asymptote code → n-gram deduplication yields 1,782 problems → GPT-4o screening for geometry problems yields 1,247 → two rounds of human verification (format normalization + quality enhancement: decontamination / answer leakage prevention / accuracy checking) → 392 problems → supplemented with AIME24 (5 problems) / MATH-500 (42 problems) / MathVerse (61 solid geometry problems with hand-written Matplotlib code) → **final benchmark of 500 problems**.

## Key Experimental Results

### Main Results: Performance of 19 LLMs on GeoGramBench

| Model | Primitive | Compositional | Abstract | Overall |
|-------|-----------|--------------|----------|---------|
| **GPT-5** | **90.44%** | **84.59%** | 39.26% | **75.01%** |
| Qwen3-235B-Think | 89.09% | 79.12% | **49.05%** | 74.00% |
| GPT-o1 | 85.92% | 76.12% | 44.67% | 70.92% |
| GPT-o3-mini | 83.49% | 76.10% | 42.67% | 70.00% |
| DeepSeek-R1 | 84.68% | 75.13% | 40.86% | 69.17% |
| QwQ-32B | 85.17% | 73.12% | 37.92% | 67.12% |
| GPT-4o | 40.02% | 21.36% | 4.51% | 21.40% |
| DeepScaleR-1.5B | 65.44% | 47.89% | 15.76% | 43.83% |

**All models score below 50% at the Abstract level**; GPT-5 achieves only 39.26%.

### Ablation Study: Effect of Drawing Language

| Benchmark | Asymptote Code | Matplotlib Code | Difference |
|-----------|---------------|----------------|------------|
| AIME24 (QwQ-32B) | ~X% | ~X% | < 1% |
| MATH-500 (QwQ-32B) | ~X% | ~X% | < 1% |

The choice of drawing language has negligible impact on performance; the bottleneck lies in spatial abstraction rather than code syntax comprehension.

### Key Findings

- **Hardest subtypes**: Among Primitive and Compositional levels, angle problems are the most difficult (requiring reconstruction and reasoning over implicit spatial relations); at the Abstract level, area and volume problems are hardest (requiring complete 3D spatial understanding).
- **Limited effectiveness of CoT reasoning**: Token Budget Forcing increases token count by 77.4% (10,544→18,710) but improves accuracy by only 0.30% (54.60%→54.90%), indicating that the bottleneck lies in spatial representation construction rather than reasoning length.
- **Saturation effect in domain-specific fine-tuning**: Adding 100 GeoGramBench training samples improves performance by 3.02%, but scaling from 100 to 300 samples yields only an additional 0.58% gain, suggesting the bottleneck is architectural rather than data-driven.
- **Common failure modes**: (1) preference for algebraic over geometric construction approaches; (2) rarely introducing auxiliary lines or points; (3) difficulty judging spatial orientation (clockwise vs. counterclockwise); (4) confusion in symbol-to-geometric-element mapping.

## Highlights & Insights

- This is the first work to formally define the Program-to-Geometry task and construct a dedicated large-scale benchmark.
- The validation experiments for the geometric complexity taxonomy are highly compelling—demonstrating that the source of difficulty in this task differs fundamentally from that in traditional mathematical reasoning.
- The identification and systematic prevention of answer leakage is a significant contribution that substantially improves evaluation validity.
- The behavioral analyses (RQ1–3) provide deep insight into the internal geometric reasoning mechanisms of LLMs.
- The hypothesized "multi-stage internal geometric representation process" (Appendix H) offers a valuable framework for future research.

## Limitations & Future Work

- Coverage is limited to 2D and simple 3D geometry; real-world 3D scenes are not addressed.
- Failure mode analysis is primarily qualitative, lacking automated and systematic diagnostic tools.
- Although GeoGramBench represents the largest Program-to-Geometry evaluation set, the distribution across subtypes is uneven (e.g., only 27 volume problems).
- Only zero-shot settings are tested; few-shot and in-context learning strategies remain unexplored.
- Fine-tuning experiments are conducted on a single model (s1.1-32B).

## Related Work & Insights

- **SGP-Bench** (Qiu et al., 2024) and **SVGenius** (Chen et al., 2025): focus on SVG code comprehension; GeoGramBench further targets geometric reasoning rather than code parsing alone.
- **s1: Simple Test-time Scaling** (Muennighoff et al., 2025): Token Budget Forcing shows limited effectiveness on GeoGramBench, suggesting that test-time scaling offers little benefit for spatial reasoning.
- Implications for multimodal model design: the fundamental bottleneck is LLMs' spatial abstraction capacity, which cannot be resolved by scaling data or reasoning length alone—architectural-level innovation is required.

## Rating

- Novelty: ⭐⭐⭐⭐ First dedicated Program-to-Geometry evaluation benchmark with a clearly defined task and theoretically grounded taxonomy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 19 models, including behavioral analysis, fine-tuning ablations, CoT analysis, and drawing language comparison.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and research-question-driven, with rich and clear figures and tables.
- Value: ⭐⭐⭐⭐ Reveals fundamental limitations in LLM spatial reasoning with important implications for future model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TopoBench: Benchmarking LLMs on Hard Topological Reasoning](topobench_benchmarking_llms_on_hard_topological_reasoning.md)
- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics](from_abstract_to_contextual_what_llms_still_cannot_do_in_math_word_problem_solvi.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)

</div>

<!-- RELATED:END -->
