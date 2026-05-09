---
title: >-
  [Paper Note] AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy
description: >-
  [NeurIPS 2025][Scientific Visualization] AstroVisBench introduces the first code benchmark for evaluating LLMs on astronomical scientific computing and visualization. It extracts 864 tasks (processing + visualization) from 110 Jupyter Notebooks, and designs a dual evaluation pipeline (execution-based variable inspection + VLM-as-Judge visualization scoring, achieving Spearman ρ=0.822 with expert ratings). Evaluation of 8 state-of-the-art models reveals that Gemini 2.5 Pro performs best, yet attains only a 15.7% error-free rate, with FileNotFoundError accounting for 43% of all errors.
tags:
  - NeurIPS 2025
  - Scientific Visualization
  - Astronomical Code Generation
  - Domain-Specific Benchmark
  - VLM Evaluation
  - Jupyter Notebook
date: 2026-05-08
content_hash: ec345a33d151a893
---

# AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy

**Conference**: NeurIPS 2025
**arXiv**: [2505.20538](https://arxiv.org/abs/2505.20538)
**Code**: [https://astrovisbench.github.io](https://astrovisbench.github.io)
**Area**: LLM Code Generation / Astronomy
**Keywords**: Scientific Visualization, Astronomical Code Generation, Domain-Specific Benchmark, VLM Evaluation, Jupyter Notebook

## TL;DR
AstroVisBench introduces the first code benchmark for evaluating LLMs on astronomical scientific computing and visualization. It extracts 864 tasks (processing + visualization) from 110 Jupyter Notebooks, and designs a dual evaluation pipeline (execution-based variable inspection + VLM-as-Judge visualization scoring, achieving Spearman ρ=0.822 with expert ratings). Evaluation of 8 state-of-the-art models reveals that Gemini 2.5 Pro performs best, yet attains only a 15.7% error-free rate, with FileNotFoundError accounting for 43% of all errors.

## Background & Motivation

**Background**: Existing LLM code generation benchmarks (SWE-bench, BigCodeBench) focus on general-purpose programming and do not assess the correctness of scientific visualizations. Astronomy research workflows involve specialized APIs (astropy, specutils, photutils) and complex visualizations (color-magnitude diagrams, light curves, all-sky projections).

**Limitations of Prior Work**: (a) Astronomical APIs are highly specialized (38 libraries, 26 astronomy-specific)—LLMs may have had insufficient exposure during training; (b) Visualization evaluation is difficult to automate—correctness requires not only executability but also visual accuracy and aesthetic quality; (c) Underspecification of queries—astronomers' requests are often vague (e.g., "plot a color-magnitude diagram" without specifying exact parameters).

**Key Challenge**: Astronomical visualization requires deep domain knowledge (e.g., magnitude axes should be reversed, axis labels require domain-specific formatting), yet the depth of LLMs' astronomical knowledge remains largely unknown.

**Goal**: Construct a standardized benchmark for evaluating LLMs on end-to-end astronomical research workflows (data processing → visualization).

**Key Insight**: Tasks are extracted from real astronomers' Jupyter Notebooks to ensure authenticity. Execution-based evaluation (variable inspection) is designed for processing tasks, and VLM-based evaluation (Claude 3.5 Sonnet) is applied to visualization tasks.

**Core Idea**: 110 real astronomical Notebooks → 864 processing + visualization tasks + execution-based + VLM dual evaluation = the first benchmark for astronomical scientific visualization code generation.

## Method

### Overall Architecture
110 Notebooks (Astro Data Lab + STScI) → dependency tracking to extract visualization cells → split into processing tasks $t_{process}$ and visualization tasks $t_{visualize}$ → GPT-4o generates natural language queries → **Dual Evaluation**: processing tasks compare executed variable values (VI score); visualization tasks are judged by Claude 3.5 (No Error / Minor / Major).

The benchmark covers 38 libraries (26 astronomy-specific) across 6 astronomical subfields (spectroscopy / photometry / image processing / time series / cosmology / simulation modeling).

### Key Designs

1. **Task Extraction and Query Generation**:

    - Function: Systematically extract evaluable code tasks from Notebooks.
    - Mechanism: Upstream dependencies of visualization cells are traced and split into processing and visualization stages. GPT-4o generates natural language queries reflecting typical astronomer workflows without leaking expert knowledge. Underspecification is handled by mapping ambiguous constants to ground-truth values.
    - Design Motivation: Using raw Notebook cells as evaluation inputs would be unfair—models require reasonable natural language prompts.

2. **Dual Evaluation Pipeline**:

    - Function: Independently assess data processing accuracy and visualization quality.
    - Mechanism: **Processing**: Variable Inspection score $VI = |\mathbf{V}_M \cap \mathcal{V}_G| / |\mathcal{V}_G|$—directly compares post-execution variable values. **Visualization**: Claude 3.5 Sonnet rates outputs relative to reference images as No Error (1) / Minor (2) / Major (3), achieving Spearman ρ=0.822 with five astronomy experts.
    - Design Motivation: Exact matching suffices for processing tasks, but visualization "correctness" involves aesthetics, readability, and domain conventions—necessitating VLM-based judgment.

3. **Error Taxonomy**:

    - Function: Fine-grained analysis of LLM code generation failure modes.
    - Mechanism: Execution errors (FileNotFoundError 43%, QueryClientError, ValueError/TypeError) + visualization errors (domain convention violations, improper axis scaling, poor readability).
    - Design Motivation: Understanding failure modes is a prerequisite for improvement—FileNotFoundError indicates that LLMs hallucinate file paths.

### Loss & Training
- This work presents an evaluation benchmark; no model training is involved.
- 8 models evaluated: Gemini 2.5 Pro, Claude 3.7 / Opus 4, GPT-o3-mini / 4o, QwQ, Qwen-2.5, Llama-4.

## Key Experimental Results

### Main Results

| Metric | Best Model | Value | Runner-up |
|--------|-----------|-------|-----------|
| Processing Crash Rate | Gemini 2.5 Pro | **30.8%** | Claude Sonnet 50.9% |
| VI Score | o3-mini | **0.694** | Claude Opus 0.644 |
| Visualization Crash Rate | Gemini 2.5 Pro | **20.1%** | o3-mini 30.3% |
| Error-Free Rate | Gemini 2.5 Pro | **15.7%** | Claude Sonnet 9.5% |
| Major Error Rate | o3-mini | 29.6% | Gemini 28.5% |

### Error Analysis

| Error Type | Proportion | Description |
|------------|-----------|-------------|
| FileNotFoundError | **43%** | Hallucinated file paths |
| QueryClientError | — | ADQL query failures |
| ValueError / TypeError | — | Misuse of specialized APIs |
| Domain Convention Violations | Common | e.g., incorrect magnitude axis direction, non-standard axis labels |

### Key Findings
- The best model achieves only a 15.7% error-free rate—astronomical scientific visualization remains highly challenging for LLMs.
- FileNotFoundError accounts for 43% of errors—LLMs extensively hallucinate file paths and data locations.
- Processing vs. visualization: processing tasks are more sensitive to API knowledge (larger VI variance), while visualization tasks are more sensitive to domain conventions.
- VLM-as-Judge (Claude 3.5) correlates strongly with human experts (ρ=0.822), enabling scalable automated evaluation.
- All models perform substantially worse on specialized astronomical libraries (e.g., wfc3tools, lightkurve) than on general-purpose libraries, reflecting insufficient training coverage.

## Highlights & Insights
- **First scientific visualization code benchmark**: addresses a critical gap in LLM evaluation for scientific computing.
- **VLM-as-Judge is viable**: a high correlation of ρ=0.822 makes large-scale automated evaluation feasible.
- **FileNotFoundError at 43%** exposes a fundamental weakness in LLM code generation—severely insufficient environmental awareness (file systems, data paths).

## Limitations & Future Work
- Automated LLM-based preprocessing of Notebooks may introduce hallucination noise—expert validation was conducted on a subset but not the full collection.
- VLM judgments may still diverge from expert assessments, particularly on subtle domain conventions (e.g., magnitude axis direction).
- Execution-based evaluation cannot pickle generators, lambdas, or OS resources, limiting automated assessment of certain complex tasks.
- The evaluation represents a snapshot of 8 models—rapid model updates may render results outdated.
- Coverage is limited to astronomy—extending to other scientific domains such as chemistry, physics, and biology requires analogous efforts.
- The handling of query underspecification (mapping constants to ground truth) may be overly permissive.

## Related Work & Insights
- **vs. SWE-bench**: Focuses on general-purpose code repair; does not evaluate scientific visualization.
- **vs. BigCodeBench**: Focuses on general-purpose code generation; does not address domain specialization.
- **vs. ML-Bench**: Evaluates machine learning workflows; does not address astronomical science.
- **Insights**: Each scientific domain requires analogous domain-specific code benchmarks—AstroVisBench provides a methodological template for constructing such benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First scientific visualization code benchmark, filling a significant gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 models + 864 tasks + VLM validation + error taxonomy.
- Writing Quality: ⭐⭐⭐⭐ Task design and evaluation pipeline are clearly described; the error taxonomy is valuable.
- Value: ⭐⭐⭐⭐⭐ Advances standardized evaluation of LLMs in scientific computing and provides a template for other scientific domains.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](../../ACL2026/code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](../../ICLR2026/code_intelligence/paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](../../ACL2026/code_intelligence/precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](embedding_alignment_in_code_generation_for_audio.md)
- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)

<!-- RELATED:END -->
