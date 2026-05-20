---
title: >-
  [Paper Note] Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs
description: >-
  [NeurIPS 2025][Robotics][Engineering Design] This paper introduces EngDesign—the first LLM engineering design benchmark spanning 9 engineering domains (operating systems, computer architecture, control systems…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Engineering Design"
  - "LLM Benchmarking"
  - "Simulation-Based Evaluation"
  - "Multi-Domain Engineering"
  - "AGI"
date: 2026-05-08
content_hash: 88b8d498e19999a0
---

# Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2509.16204](https://arxiv.org/abs/2509.16204)  
**Code**: [https://agi4engineering.github.io/Eng-Design/](https://agi4engineering.github.io/Eng-Design/)  
**Area**: Robotics
**Keywords**: Engineering Design, LLM Benchmarking, Simulation-Based Evaluation, Multi-Domain Engineering, AGI

## TL;DR

This paper introduces EngDesign—the first LLM engineering design benchmark spanning 9 engineering domains (operating systems, computer architecture, control systems, mechanical engineering, structural engineering, digital hardware, analog circuits, robotics, and signal processing)—replacing conventional QA matching with a simulation-driven evaluation pipeline. The benchmark reveals that even the most capable reasoning model, o3, achieves only a 34% pass rate.

## Background & Motivation

**Background**: LLMs have demonstrated strong performance on conventional QA benchmarks (MMLU, HumanEval, GPQA, etc.) and have been explored for textbook-level engineering problem answering. However, existing evaluations focus on knowledge recall and exact-match answer checking, leaving genuine engineering design capabilities—which require integrating domain knowledge, constraint reasoning, design trade-offs, and functional verification through professional simulation tools—largely unexamined.

**Limitations of Prior Work**: (1) Existing engineering benchmarks are limited to factual QA within single domains and fail to capture the complexity of real design work; (2) Evaluation methods rely on exact matching or LLM-as-judge, yet engineering design problems typically admit no unique correct answer (e.g., infinitely many valid controller designs); (3) No unified evaluation platform exists across multiple engineering disciplines.

**Key Challenge**: Real-world engineering design is inherently open-ended—given performance specifications and constraints, multiple valid solutions may exist. Yet existing benchmarks assess this open-ended capability using closed-form QA, producing unreliable conclusions. What is needed is an evaluation method that verifies whether a design *works*, rather than whether an answer is *correct*.

**Goal**: To construct a multi-domain engineering benchmark that objectively measures the practical engineering design capabilities of LLMs through a simulation-driven evaluation pipeline.

**Key Insight**: Rather than presenting LLMs with multiple-choice questions, the paper requires them to "design like an engineer"—producing controller parameters, circuit designs, GPU architecture code, and similar artifacts—and then uses professional tools such as SPICE simulation, the MATLAB Control System Toolbox, and finite element analysis to automatically verify whether designs satisfy performance requirements.

**Core Idea**: Replace static answer matching with a simulation-driven evaluation pipeline, and assess the genuine engineering design capabilities of LLMs across 101 open-ended design tasks spanning 9 engineering domains.

## Method

### Overall Architecture

EngDesign comprises 101 design tasks across 9 engineering domains, with a total of 473 scoring items. Each task consists of four components: (1) a task description (serving as the LLM prompt, averaging 779 tokens); (2) a scoring rubric (multi-item, 100 points total, supporting partial credit); (3) an evaluation pipeline (automated simulation scripts); and (4) a reference design (a validated solution confirming feasibility). LLM outputs are parsed into a structured format and fed into task-specific simulation evaluation pipelines, which produce pass/fail judgments, scores from 0 to 100, and detailed logs.

### Key Designs

1. **Simulation-Driven Evaluation Pipeline**:

    - **Function**: Objectively verifies whether LLM-generated engineering designs satisfy functional requirements using professional simulation tools.
    - **Mechanism**: Each task is equipped with a domain-specific evaluation script. Control systems are evaluated via MATLAB simulation of closed-loop responses (checking rise time, overshoot, phase margin, etc.); analog circuits are verified through SPICE simulation of gain, bandwidth, and related metrics; structural designs undergo finite element analysis; digital hardware undergoes functional simulation. The pipeline produces three outputs: a binary pass/fail judgment, a fine-grained score from 0 to 100, and simulation logs. 67 tasks are fully open-source (EngDesign-Open), while 34 require commercial simulation tools (MATLAB, Cadence, etc.).
    - **Design Motivation**: Traditional benchmarks rely on string matching or LLM-based judgment, which cannot reliably assess the functional correctness of open-ended designs. Simulation-based evaluation is the gold standard in engineering practice, elevating assessment from "does the text look correct" to "does the design actually work."

2. **Structured LLM Output Mechanism**:

    - **Function**: Ensures uniform output formatting across different LLMs so that outputs can be automatically parsed by the evaluation pipeline.
    - **Mechanism**: The Python `instructor` library (built on Pydantic) is used to define schema templates specifying expected fields (e.g., design parameters, code snippets). LLM outputs are constrained to two components: a `reasoning` field (capturing the inference process) and a `ConfigFile` class (encoding the design result), the latter being automatically parsed to trigger simulation.
    - **Design Motivation**: The evaluation pipeline requires programmatic extraction of design parameters; free-form LLM outputs cannot be parsed reliably.

3. **Partial Credit and Multi-Stage Quality Control**:

    - **Function**: Quantifies design quality at fine granularity and identifies dimensions on which models partially succeed.
    - **Mechanism**: Each task is decomposed into multiple scoring items (averaging approximately 4.7 items per task). For example, a controller design task may include "stability 20 pts + rise time 20 pts + overshoot 20 pts + steady-state error 20 pts + robustness 20 pts." Even when a design fails overall, partial successes on individual sub-metrics are recorded. Benchmark construction follows five stages: initial task design → LLM-based filtering → first-round review → domain expert review → final integration.
    - **Design Motivation**: Binary scoring obscures incremental progress. Multi-dimensional scoring more precisely reveals capability gaps and provides actionable guidance for model improvement.

### Loss & Training

No training is involved. The evaluation protocol runs each LLM three times per task and reports pass rates and average scores. An iterative evaluation protocol is also designed, in which the previous round's design output and simulation feedback are provided as a new prompt to the LLM, simulating an engineer's iterative refinement process.

## Key Experimental Results

### Main Results

| Model | Type | Overall Pass% | OS | Ctrl | DHD | Robo | SigP | Stru |
|------|------|---------------|-----|------|-----|------|------|------|
| GPT-4o | Chat | 15.7 | 4.2 | 18.5 | 10.3 | 26.7 | 17.7 | 25.6 |
| Claude-3.7-Sonnet | Chat | 22.6 | 0.0 | 16.7 | 33.3 | 33.3 | 21.6 | 30.8 |
| o1 | Reasoning | 29.2 | 37.5 | 24.1 | 41.0 | 50.0 | 25.5 | 23.1 |
| **o3** | Reasoning | **34.4** | 25.0 | 35.2 | 20.5 | **63.3** | **41.2** | 30.8 |
| o4-mini-high | Reasoning | 34.0 | 37.5 | 27.8 | **47.2** | 46.7 | 35.3 | 35.9 |
| DeepSeek-R1 | Reasoning | 25.5 | 5.3 | 36.4 | 38.5 | 26.7 | 20.5 | 41.7 |
| Gemini-2.5-Pro | Reasoning | 29.5 | 9.5 | 33.3 | 43.6 | 56.7 | 12.8 | **50.0** |

### Ablation Study (Iterative Design)

| Model | Round 1 Pass% | Round 5 Pass% | Round 10 Pass% |
|------|-----------|-----------|------------|
| GPT-4o | ~14 | ~25 | ~30 |
| o1 | ~26 | ~40 | ~48 |
| o3 | ~30 | ~48 | ~58 |
| o4-mini | ~28 | ~42 | ~50 |

### Key Findings

- **Analog IC design is a universal blind spot**: All 12 evaluated models achieve a 0% pass rate on Analog IC Design, with no improvement even after 10 iterative rounds. This reflects the fact that analog circuit design demands extremely fine-grained physical intuition and parameter tuning expertise that current LLMs entirely lack.
- **Reasoning models substantially outperform chat models**: o3 (34.4%) vs. GPT-4o (15.7%), a gap exceeding 2×. Reasoning models also exhibit greater robustness—o1 achieves a reasoning robustness score of 0.62, compared to only 0.20 for Gemini-2.0-Flash.
- **Iterative design substantially improves performance**: o3 improves from ~30% in a single round to ~58% after 10 rounds, indicating that LLMs can learn from simulation feedback and refine designs, mirroring an engineer's iterative workflow.
- **Primary failure modes**: Domain knowledge errors (DKE) and constraint violations (CVE) account for 55–67% of failures; over-reliance on prior knowledge (PKO) and hallucinations (HAL) account for 25–30%; computational errors (CE) account for less than 9%.
- Domain-specific strengths vary considerably across models: Claude outperforms o3 on digital hardware but falls substantially behind on signal processing.

## Highlights & Insights

- **A pioneering paradigm shift in evaluation**: Moving from "is the answer correct" to "does the design work" represents arguably the most important methodological contribution to LLM evaluation in the engineering domain. Simulation-driven evaluation is fully objective and reproducible, eliminating the subjectivity inherent in LLM-as-judge approaches.
- **Fine granularity via partial credit**: The binary scoring of conventional benchmarks is insufficiently discriminative. EngDesign's multi-dimensional scoring precisely localizes which specific engineering steps a model fails at—information that is highly valuable for guiding model improvement.
- **Iterative design protocol**: By simulating a real engineer's workflow, the protocol shows that o3 reaches ~58% after 10 rounds, suggesting that feedback-driven agentic paradigms may be the appropriate deployment mode for LLMs on engineering tasks.

## Limitations & Future Work

- 34 tasks require commercial software such as MATLAB and Cadence, limiting full open reproducibility.
- The benchmark scale of 101 tasks is relatively modest, with some domains (e.g., Analog IC Design with only 5 tasks) having insufficient sample sizes.
- The uneven distribution of tasks reflects contributor research interests rather than the true distribution of engineering workloads across disciplines.
- 23 tasks include image inputs, placing text-only models (e.g., DeepSeek-R1/v3) at an inherent disadvantage on these tasks.
- Multi-model collaboration (e.g., agent frameworks) is not evaluated; only single-model single-pass or iterative generation is assessed.

## Related Work & Insights

- **vs. MMLU/GPQA and similar QA benchmarks**: These benchmarks test knowledge recall, whereas EngDesign tests knowledge application and design synthesis—two entirely distinct dimensions of the LLM capability spectrum.
- **vs. HumanEval**: HumanEval assesses the functional correctness of code generation; EngDesign extends a similar programmatic verification philosophy to engineering design, where problems are considerably more open-ended than programming tasks.
- **vs. domain-specific works such as ControlAgent and AnalogCoder**: These works apply LLMs to assist design within specific domains; EngDesign provides a unified cross-domain evaluation framework.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first simulation-driven engineering design benchmark spanning multiple domains, with a pioneering evaluation paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 models × 101 tasks × 3 rounds + iterative experiments; task scale could be further expanded.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, result analysis is thorough, and the failure taxonomy is insightful.
- Value: ⭐⭐⭐⭐⭐ Provides a rigorous assessment of the capability boundaries of LLMs in engineering, revealing the substantial gap between current models and a truly "AI engineer."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy](../../AAAI2026/robotics/do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)
- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[NeurIPS 2025\] Predicting the Performance of Black-Box LLMs through Follow-Up Queries](predicting_the_performance_of_black-box_llms_through_follow-up_queries.md)
- [\[ICLR 2026\] Token Taxes: Mitigating AGI's Economic Risks](../../ICLR2026/robotics/token_taxes_mitigating_agis_economic_risks.md)
- [\[NeurIPS 2025\] Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs](zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)

</div>

<!-- RELATED:END -->
