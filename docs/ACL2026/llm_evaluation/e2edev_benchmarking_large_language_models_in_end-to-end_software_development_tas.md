---
title: >-
  [Paper Note] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task
description: >-
  [ACL 2026][LLM Evaluation][end-to-end software development] This paper proposes E2EDev, an end-to-end software development benchmark grounded in Behavior-Driven Development (BDD) principles. It comprises 46 real-world web projects, 244 fine-grained requirements, and 703 executable BDD tests. Evaluation reveals that even the strongest LLMs (Claude series) achieve no more than 60% requirement accuracy, and that the interaction overhead of multi-agent frameworks is disproportionate to their performance gains.
tags:
  - ACL 2026
  - LLM Evaluation
  - end-to-end software development
  - behavior-driven development
  - benchmark
  - multi-agent coding
  - requirements verification
date: 2026-05-08
content_hash: 39028b0ad212b16c
---

# E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task

**Conference**: ACL 2026
**arXiv**: [2510.14509](https://arxiv.org/abs/2510.14509)
**Code**: [https://github.com/SCUNLP/E2EDev](https://github.com/SCUNLP/E2EDev)
**Area**: LLM Evaluation
**Keywords**: end-to-end software development, behavior-driven development, benchmark, multi-agent coding, requirements verification

## TL;DR

This paper proposes E2EDev, an end-to-end software development benchmark grounded in Behavior-Driven Development (BDD) principles. It comprises 46 real-world web projects, 244 fine-grained requirements, and 703 executable BDD tests. Evaluation reveals that even the strongest LLMs (Claude series) achieve no more than 60% requirement accuracy, and that the interaction overhead of multi-agent frameworks is disproportionate to their performance gains.

## Background & Motivation

**State of the Field**: LLM-driven end-to-end software development (E2ESD) is evolving from function-level code generation toward full-project automation. Existing frameworks span multi-agent approaches (ChatDev, MetaGPT) and single-agent approaches (GPT-Engineer), yet evaluation methodology lags significantly behind framework development.

**Limitations of Prior Work**: (1) Existing benchmarks (SoftwareDev, SRDD) rely on coarse-grained requirement descriptions as input — vague phrasings such as "manage words" fail to specify whether users need editing, bookmarking, or deletion functionality; (2) Evaluation depends on subjective human judgment or heuristic metrics, lacking a systematic methodology grounded in software engineering standards, rendering cross-framework comparisons inconsistent and unreliable.

**Root Cause**: E2ESD tasks require simultaneously performing high-level planning (deciding what to build) and fine-grained functional implementation (precisely satisfying requirement details). Ambiguous requirements and unreliable evaluation in existing benchmarks prevent a genuine understanding of where framework capabilities break down.

**Paper Goals**: (1) Construct an E2ESD benchmark with fine-grained requirement specifications; (2) Design an automated BDD-based evaluation pipeline; (3) Systematically analyze the actual capabilities and failure modes of various frameworks and LLMs on E2ESD tasks.

**Starting Point**: Drawing on Behavior-Driven Development (BDD) principles from software engineering, this work employs Given-When-Then Gherkin scenarios to simulate real user interactions, enabling requirement verification from a user perspective.

**Core Idea**: Transform E2ESD evaluation from subjective human scoring to executable BDD tests grounded in fine-grained requirements, deterministically verifying the requirement conformance of generated code by simulating authentic user interactions.

## Method

### Overall Architecture

E2EDev consists of three components: (1) a fine-grained user requirement list for each software project; (2) multiple BDD test scenarios with corresponding Python step implementations for each requirement; and (3) a fully automated testing pipeline based on the Behave framework. The dataset is constructed from 46 real-world GitHub web projects via HITL-MAA (Human-In-The-Loop Multi-Agent Annotation framework).

### Key Designs

1. **HITL-MAA Annotation Framework**:

    - **Function**: Semi-automatically extracts fine-grained requirements and executable tests from source code.
    - **Mechanism**: A three-stage pipeline — (a) a Code Analyzer Agent analyzes core functionalities and UI element interactions, a Requirement Extractor Agent generates candidate requirements, and human reviewers ensure accuracy; (b) a Test Case Generation Agent produces Gherkin-format BDD scenarios for each requirement, reviewed collaboratively by five software testing experts; (c) a Test Automation Engineer Agent generates Python step implementations, with iterative self-correction via a Dry Run Verifier and Test Runner resolving over 80% of logical errors without human intervention.
    - **Design Motivation**: Pure manual annotation is prohibitively expensive, while pure LLM generation yields unstable quality; human-machine collaboration balances efficiency and quality.

2. **Test ID Anchor System**:

    - **Function**: Assigns unique test IDs to UI components as structurally invariant DOM anchors.
    - **Mechanism**: Prior to requirement and test generation, GPT-4o assigns unique test IDs to key UI components, ensuring consistent cross-project reference regardless of DOM structural variation.
    - **Design Motivation**: Projects generated by different frameworks exhibit substantial HTML structural differences, necessitating stable anchors to execute tests consistently across projects.

3. **Multi-Level Evaluation Metric Suite**:

    - **Function**: Assesses code validity at both the requirement level and test level, while measuring generation efficiency.
    - **Mechanism**: Req. Acc measures the proportion of fully satisfied requirements (all test cases pass); Test Acc measures the proportion of passing tests; Balanced Score weights both to eliminate test granularity bias. Efficiency metrics include API cost, carbon emissions, and wall-clock time.
    - **Design Motivation**: Test pass rates alone may be biased by uneven test counts per requirement; requirement-level metrics more faithfully reflect real user experience.

### Loss & Training

E2EDev is an evaluation benchmark and does not involve model training. The evaluation pipeline automatically executes Python steps corresponding to Gherkin scenarios via the Behave framework, performing deterministic pass/fail verification on each generated project.

## Key Experimental Results

### Main Results

**Requirement Accuracy (Req. Acc %) across Frameworks and LLM Backbones**

| LLM Backbone | Vanilla LLM | GPT-Engineer | Self-Collab. | MapCoder | ChatDev | MetaGPT |
|---|---|---|---|---|---|---|
| Claude-Haiku 4.5 | 48.69 | **53.75** | 49.01 | 49.61 | 44.73 | 5.39 |
| GPT-4o | 45.95 | **50.83** | 46.83 | 47.70 | 42.71 | 0.00 |
| GPT-4o-mini | **44.82** | 42.13 | 37.90 | 41.30 | 33.16 | 0.00 |
| Qwen-Max | 43.33 | **49.61** | 42.30 | 48.83 | 43.93 | 1.65 |
| Qwen-7B | 22.37 | **24.03** | 20.65 | 11.90 | 10.96 | 0.00 |

### Ablation Study

**Failure Mode Analysis (Manual Evaluation of 360 Projects)**

| Failure Type | Description | Primarily Affected Frameworks |
|---|---|---|
| Code Inconsistency | Missing/conflicting/empty functions | MetaGPT (44% attributable) |
| Requirement Omission | Required functionality not implemented | Vanilla LLM, ChatDev |
| Requirement Deviation | Implementation logic diverges from requirements | All frameworks (notable improvement in multi-agent) |
| Detail Mismatch | Mostly correct but edge-case errors | Most severe in Self-Collaboration |

### Key Findings

- Even the strongest combination of Claude-Haiku 4.5 + GPT-Engineer achieves only 53.75% Req. Acc, indicating that E2ESD remains a formidable challenge.
- MetaGPT approaches 0% success across nearly all LLM backbones, rooted in inter-agent communication breakdown — programmers ignore architects' file structures, and product managers overwrite and compress original requirements.
- Multi-agent frameworks incur substantial interaction overhead (ChatDev averages 15.72 dialogue turns) with limited performance gains, sometimes underperforming Vanilla LLM.
- The gap between Soft Req. Acc and Req. Acc exceeds 25%, indicating that models can implement basic functionality but fail to handle complex edge cases.
- Framework performance is heavily dependent on backbone capability; weaker models can be further degraded by framework overhead.

## Highlights & Insights

- Introducing BDD methodology into LLM evaluation represents an insightful cross-domain transfer — applying the mature software engineering practice of Given-When-Then to the verification of AI-generated code.
- The iterative self-correction mechanism in HITL-MAA (Dry Run + Test Runner) resolves 80% of logical errors, demonstrating the practical utility of LLMs within annotation pipelines.
- Failure mode analysis exposes a fundamental issue in multi-agent architectures: information is progressively diluted as it passes between agents, preserving high-level functionality while losing fine-grained details.

## Limitations & Future Work

- Coverage is limited to web applications; while the authors argue this constitutes a "lower-bound test," challenges in desktop, mobile, and backend applications may differ substantially.
- The benchmark scale of 46 projects is constrained by the high cost of repository-level benchmark construction.
- CI/CD integration and deep backend verification are excluded in favor of browser-automation-based black-box testing.
- Future work may extend the benchmark into a continuously updated public leaderboard supporting longitudinal evaluation.

## Related Work & Insights

- **vs. rSDE-Bench**: rSDE-Bench uses function-level unit tests to verify outputs; E2EDev uses BDD tests to verify behavior from a user perspective, at a granularity more aligned with real usage scenarios.
- **vs. SoftwareDev/SRDD**: These rely on vague descriptions and human evaluation; E2EDev provides fine-grained requirements and automated deterministic assessment.
- **vs. Mle-Bench/GitTaskBench**: These focus on ML pipelines and repository operations, respectively; E2EDev targets the complete pipeline from requirements to executable projects.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing BDD into LLM E2ESD evaluation is a meaningful contribution, though the benchmark construction methodology itself is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 6 LLM backbones × 6 frameworks with additional manual failure mode analysis — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with intuitive figures and in-depth analysis.
- **Value**: ⭐⭐⭐⭐ Fills a gap in reliable E2ESD evaluation; failure mode analysis provides direct guidance for framework design.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)

<!-- RELATED:END -->
