---
title: >-
  [Paper Note] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task
description: >-
  [ACL 2026][LLM Evaluation][End-to-End Software Development] Ours proposes E2EDev, an end-to-end software development benchmark based on Behavior-Driven Development (BDD) principles. It contains 46 real-world Web projects…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "End-to-End Software Development"
  - "Behavior-Driven Development"
  - "Benchmarking"
  - "Multi-agent Coding"
  - "Requirement Verification"
date: 2026-05-08
content_hash: 135a49fd6e7c99f1
---

# E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task

**Conference**: ACL 2026  
**arXiv**: [2510.14509](https://arxiv.org/abs/2510.14509)  
**Code**: [https://github.com/SCUNLP/E2EDev](https://github.com/SCUNLP/E2EDev)  
**Area**: LLM Evaluation  
**Keywords**: End-to-End Software Development, Behavior-Driven Development, Benchmarking, Multi-agent Coding, Requirement Verification

## TL;DR

Ours proposes E2EDev, an end-to-end software development benchmark based on Behavior-Driven Development (BDD) principles. It contains 46 real-world Web projects, 244 fine-grained requirements, and 703 executable BDD tests. Evaluation reveals that even the strongest LLMs (Claude series) achieve no more than 60% requirement accuracy, and the complex interaction costs of multi-agent frameworks are disproportionate to their performance gains.

## Background & Motivation

**Background**: LLM-driven End-to-End Software Development (E2ESD) is evolving from function-level code generation to the automatic generation of complete projects. Existing frameworks are divided into multi-agent methods (ChatDev, MetaGPT) and single-agent methods (GPT-Engineer), but evaluation systems lag significantly behind framework development.

**Limitations of Prior Work**: (1) Existing benchmarks (SoftwareDev, SRDD) use coarse-grained requirement descriptions as input; vague descriptions like "manage words" fail to clarify whether the user needs editing, bookmarking, or deletion functions. (2) Evaluation relies on subjective human judgment or heuristic metrics, lacking a systematic methodology based on software engineering standards, leading to inconsistent and unreliable cross-framework comparisons.

**Key Challenge**: E2ESD tasks require simultaneous high-level planning (deciding what to build) and fine-grained functional implementation (precisely satisfying requirement details). The vague requirements and unreliable evaluation of existing benchmarks prevent a true understanding of the performance bottlenecks in these frameworks.

**Goal**: (1) Construct an E2ESD benchmark with fine-grained requirement specifications; (2) Design an automated evaluation pipeline based on BDD; (3) Systematically analyze the real capabilities and failure modes of various frameworks and LLMs in E2ESD tasks.

**Key Insight**: Drawing from Behavior-Driven Development (BDD) principles in software engineering, Gherkin scenario descriptions in the Given-When-Then format are used to simulate real user interactions, enabling verification from the user's perspective on whether the generated software meets requirements.

**Core Idea**: Shift E2ESD evaluation from vague manual scoring to executable BDD tests based on fine-grained requirements, deterministically verifying the requirement compliance of generated code by simulating real user interactions.

## Method

### Overall Architecture

E2EDev consists of three parts: (1) a list of fine-grained user requirements for each software project; (2) multiple BDD test scenarios and their Python step implementations for each requirement; (3) a fully automated test pipeline based on the Behave framework. The dataset is constructed from 46 real GitHub Web projects using HITL-MAA (Human-In-The-Loop Multi-Agent Annotation framework).

### Key Designs

1.  **HITL-MAA Annotation Framework**:
    - **Function**: Semi-automatically extracts fine-grained requirements and executable tests from source code.
    - **Mechanism**: A three-stage pipeline—(a) Code Analyzer Agent analyzes core functions and UI element interactions, Requirement Extractor Agent generates candidate requirements, and human review ensures accuracy; (b) Test Case Generation Agent generates Gherkin-format BDD scenarios for each requirement, reviewed by five software testing experts; (c) Test Automation Engineer Agent generates Python step implementations, ensuring executability through iterative self-correction by Dry Run Verifier and Test Runner, with over 80% of logical errors resolved without human intervention.
    - **Design Motivation**: Pure manual annotation is too costly, while pure LLM generation quality is unstable; human-AI collaboration balances efficiency and quality.

2.  **Test ID Anchor System**:
    - **Function**: Assigns unique test IDs to UI components as structurally invariant DOM anchors.
    - **Mechanism**: Before generating requirements and tests, GPT-4o assigns unique test IDs to key UI components, ensuring that different projects can consistently reference the same component even if DOM structures differ.
    - **Design Motivation**: HTML structures generated by different frameworks vary greatly; stable anchors are needed for consistent cross-project test execution.

3.  **Multi-level Evaluation Metric System**:
    - **Function**: Evaluates code effectiveness at both the requirement and test levels while measuring generation efficiency.
    - **Mechanism**: Req. Acc measures the percentage of requirements fully met (all test cases passed), Test Acc measures the percentage of passed tests, and Balanced Score weights both to eliminate test granularity bias. Efficiency metrics include API costs, carbon emissions, and time.
    - **Design Motivation**: Relying solely on test pass rates may be biased due to uneven test counts per requirement; requirement-level metrics align better with the actual user experience.

### Loss & Training

E2EDev is an evaluation benchmark and does not involve model training. The evaluation process is automatically executed based on the Behave framework for Python steps corresponding to Gherkin scenarios, performing deterministic pass/fail verification for each generated project.

## Key Experimental Results

### Main Results

**Requirement Accuracy (Req. Acc %) of different frameworks and LLM backbones**

| LLM Backbone | Vanilla LLM | GPT-Engineer | Self-Collab. | MapCoder | ChatDev | MetaGPT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Claude-Haiku 4.5 | 48.69 | **53.75** | 49.01 | 49.61 | 44.73 | 5.39 |
| GPT-4o | 45.95 | **50.83** | 46.83 | 47.70 | 42.71 | 0.00 |
| GPT-4o-mini | **44.82** | 42.13 | 37.90 | 41.30 | 33.16 | 0.00 |
| Qwen-Max | 43.33 | **49.61** | 42.30 | 48.83 | 43.93 | 1.65 |
| Qwen-7B | 22.37 | **24.03** | 20.65 | 11.90 | 10.96 | 0.00 |

### Ablation Study

**Failure Mode Analysis (Manual evaluation of 360 projects)**

| Failure Type | Description | Primary Affected Frameworks |
| :--- | :--- | :--- |
| Code Inconsistency | Missing/conflicting/empty functions | MetaGPT (44% from here) |
| Requirement Omission | Mandatory features not implemented | Vanilla LLM, ChatDev |
| Requirement Deviation | Implementation logic diverges from requirements | All frameworks (Multi-agent shows improvement) |
| Detail Mismatch | Mostly correct but with edge case errors | Self-Collaboration (most severe) |

### Key Findings

- Even with the strongest combination of Claude-Haiku 4.5 + GPT-Engineer, Req. Acc is only 53.75%, indicating that E2ESD remains a significant challenge.
- MetaGPT has near 0% success rates across almost all LLM backbones, primarily due to communication breakdown between agents—programmers ignore the architect's file structure and product managers rewrite/compress original requirements.
- The interaction costs of multi-agent frameworks are high (ChatDev averages 15.72 dialogue rounds), but the performance gains are limited, sometimes even underperforming compared to Vanilla LLM.
- The gap between Soft Req. Acc and Req. Acc exceeds 25%: models can implement basic functions but struggle with complex edge cases.
- Frameworks are highly dependent on LLM backbone capabilities; on weaker models, frameworks may even degrade performance.

## Highlights & Insights

- Introducing BDD testing methodology to LLM evaluation is a clever cross-domain transfer—applying mature software engineering practices (Given-When-Then) to the verification of AI-generated code.
- The iterative self-correction mechanism (Dry Run + Test Runner) in HITL-MAA resolved 80% of logical errors, demonstrating the practical value of LLMs in annotation pipelines.
- Failure mode analysis reveals a fundamental problem with multi-agent architectures: information is diluted layer by layer during agent communication, where high-level functions are retained but details are lost.

## Limitations & Future Work

- Currently only covers the Web application domain; although the authors argue this is a "lower bound test," challenges in desktop, mobile, or backend applications may differ.
- The scale of 46 projects is limited due to the extremely high cost of constructing repository-level benchmarks.
- Excludes CI/CD and deep backend detection, focusing on black-box testing via browser automation.
- Future work could expand this into a continuously updated public leaderboard supporting longitudinal evaluation.

## Related Work & Insights

- **vs rSDE-Bench**: rSDE-Bench uses function-level unit tests for output verification; E2EDev uses BDD tests to verify behavior from a user perspective, making its granularity closer to real-world usage scenarios.
- **vs SoftwareDev/SRDD**: These rely on vague descriptions and manual evaluation; E2EDev provides fine-grained requirements and automated deterministic evaluation.
- **vs Mle-Bench/GitTaskBench**: These focus on ML pipelines and repository operations, respectively; E2EDev focuses on the complete workflow from requirements to executable projects.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing BDD to LLM E2ESD evaluation is a meaningful innovation, though the benchmark construction methodology itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 LLM backbones × 6 frameworks, plus manual failure mode analysis, making it very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive charts, and in-depth analysis.
- Value: ⭐⭐⭐⭐ Fills the gap in reliable E2ESD evaluation; failure mode analysis provides direct guidance for framework design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CUB: Benchmarking Context Utilisation Techniques for Language Models](cub_benchmarking_context_utilisation_techniques_for_language_models.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](../../ICML2026/llm_evaluation/politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)

</div>

<!-- RELATED:END -->
