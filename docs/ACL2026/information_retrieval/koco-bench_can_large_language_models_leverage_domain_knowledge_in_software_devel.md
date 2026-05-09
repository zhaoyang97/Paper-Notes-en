---
title: >-
  [Paper Note] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?
description: >-
  [ACL 2026][Domain-specific code generation] KoCo-Bench introduces the first code benchmark with an explicit domain knowledge corpus, covering 11 frameworks and 25 projects across 6 emerging domains (RL, Agent, RAG, etc.). It evaluates LLMs' ability to acquire and apply domain knowledge for code generation and knowledge comprehension, revealing that even the strongest coding agent, Claude Code, achieves only 34.2%.
tags:
  - ACL 2026
  - Domain-specific code generation
  - benchmark
  - domain specialization
  - knowledge corpus
  - software engineering
date: 2026-05-08
content_hash: b2b2f03306cb5178
---

# KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?

**Conference**: ACL 2026
**arXiv**: [2601.13240](https://arxiv.org/abs/2601.13240)
**Code**: [https://github.com/jiangxxxue/KOCO-bench](https://github.com/jiangxxxue/KOCO-bench)
**Area**: Information Retrieval
**Keywords**: Domain-specific code generation, benchmark, domain specialization, knowledge corpus, software engineering

## TL;DR

KoCo-Bench introduces the first code benchmark with an explicit domain knowledge corpus, covering 11 frameworks and 25 projects across 6 emerging domains (RL, Agent, RAG, etc.). It evaluates LLMs' ability to acquire and apply domain knowledge for code generation and knowledge comprehension, revealing that even the strongest coding agent, Claude Code, achieves only 34.2%.

## Background & Motivation

**Background**: LLMs perform well on general programming tasks, but domain-specific software development requires specialized domain knowledge (APIs, rules, constraints, etc.). Domain specialization methods (SFT, RAG, kNN-LM) have been employed to help LLMs learn and utilize domain knowledge.

**Limitations of Prior Work**: Existing domain-specific code benchmarks (e.g., EvoCodeBench, DomainEval) evaluate only what LLMs already know, not how they acquire and apply new knowledge. They provide test sets without explicit knowledge corpora, making it impossible to support research on domain knowledge learning and modeling.

**Key Challenge**: Research on domain specialization methods requires benchmarks to evaluate effectiveness, yet existing benchmarks lack a knowledge corpus component, preventing the systematic development of this research direction.

**Goal**: Construct a complete benchmark comprising a knowledge corpus and a test set to support evaluation of domain specialization methods in realistic software development settings.

**Key Insight**: Leveraging the natural ecosystem of software frameworks—frameworks come with documentation, source code, and examples (knowledge corpus), while projects built on these frameworks serve as evaluation tasks—forming a complete pipeline from knowledge acquisition to knowledge application.

**Core Idea**: Using 11 emerging frameworks released after 2024 as the foundation, the benchmark constructs a multi-source knowledge corpus (documentation + source code + examples), paired with multi-granularity code generation tasks (function-level to project-level, with unit/integration tests) and domain knowledge comprehension QA, simulating the realistic scenario of developers working with unfamiliar frameworks.

## Method

### Overall Architecture

KoCo-Bench = Knowledge Corpus + Evaluation Tasks. The knowledge corpus is derived from framework documentation, source code, and usage examples. Evaluation comprises two tasks: (1) *Domain Code Generation*—providing three-level requirement descriptions (project/module/function), with correctness verified via unit and integration tests; (2) *Domain Knowledge Comprehension*—multiple-choice QA to assess mastery of knowledge points in the corpus.

### Key Designs

1. **Multi-source Knowledge Corpus Construction**:

    - Function: Simulates the knowledge sources available to developers when learning a new framework.
    - Mechanism: Selects Python frameworks created after March 2024 (ensuring they fall outside LLM training data) with well-maintained documentation. Covers 6 domains: RL, Agent, RAG, model optimization, embodied AI, and Ascend ecosystem. The corpus includes framework documentation (averaging 77K lines), source code, and usage examples.
    - Design Motivation: Selecting emerging frameworks avoids data leakage; multi-source coverage ensures completeness of knowledge.

2. **Multi-granularity Code Generation Evaluation**:

    - Function: Evaluates domain code generation capability from function-level to project-level.
    - Mechanism: Provides three-level requirement descriptions (project overview → module decomposition → core functions). A total of 131 core functions are paired with 978 tests (averaging 8.6 unit tests per function plus integration tests). Requirements undergo multiple rounds of multi-agent ambiguity resolution and human review. Docker environments ensure test reproducibility.
    - Design Motivation: Multi-granularity supports evaluation of diverse code generation techniques; rigorous test suites prevent false positives.

3. **Domain Knowledge Comprehension QA**:

    - Function: Precisely evaluates LLMs' mastery of specific knowledge points.
    - Mechanism: Atomic multiple-choice questions (one knowledge point per question, supporting multiple correct answers), pre-filtered by 3 LLMs (to exclude trivial questions) and manually reviewed. A total of 107 questions.
    - Design Motivation: Code generation tasks cannot precisely pinpoint knowledge gaps; QA directly assesses knowledge comprehension.

### Loss & Training

KoCo-Bench is a benchmark rather than a model, requiring 28.5 person-months to construct. Evaluated methods include direct generation, SFT, RAG, kNN-LM, and Claude Code (agent).

## Key Experimental Results

### Main Results

| Method | Function-level Pass@1 | Project-level Pass | QA Accuracy |
|---|---|---|---|
| Claude Sonnet 4.5 (direct) | ~20% | Very low | ~60% |
| + RAG | Marginal gain | Marginal gain | — |
| + SFT | Marginal gain | Marginal gain | — |
| **Claude Code (agent)** | **34.2%** | — | — |

### Ablation Study

| Configuration | Result | Notes |
|---|---|---|
| Larger knowledge corpus | Diminishing returns | SFT gains diminish on larger corpora |
| Cross-domain continual learning | Catastrophic forgetting | Performance on prior domains degrades after learning new ones |
| No knowledge corpus (direct generation) | Very poor | Confirms domain knowledge is absent from pretraining |

### Key Findings

- Even SOTA closed-source LLMs struggle with domain code generation; Claude Code achieves only 34.2%.
- Existing domain specialization methods (SFT, RAG, kNN-LM) yield only marginal improvements with inconsistent cross-domain effects.
- Agent-based approaches (Claude Code) are currently the most effective, yet substantial room for improvement remains.
- The most common errors are misuse of domain APIs and violations of domain data constraints.
- Larger knowledge corpora lead to diminishing learning returns, indicating that existing methods cannot effectively assimilate large-scale domain knowledge.

## Highlights & Insights

- The dual-component design of "knowledge corpus + test set" represents a paradigm innovation in benchmark design—enabling the benchmark to not only assess performance but also support the development of domain specialization methods.
- Selecting emerging frameworks released after 2024 to prevent data leakage; this temporal control strategy ensures evaluation fairness.
- The multi-round, agent-assisted requirement ambiguity resolution pipeline is worth adopting by other benchmark construction efforts.

## Limitations & Future Work

- Coverage is limited to 6 AI-related domains; non-AI domains (finance, healthcare, etc.) remain to be incorporated.
- The scale of 131 core functions is relatively small.
- Framework selection is biased toward the Python ecosystem; other programming languages await coverage.
- Over time, framework knowledge may gradually enter LLM training data.

## Related Work & Insights

- **vs. EvoCodeBench/DomainEval**: These provide only test sets without knowledge corpora, evaluating existing knowledge rather than knowledge acquisition capability.
- **vs. SWE-bench**: Focuses on issue resolution without involving domain knowledge learning. KoCo-Bench simulates the realistic scenario of "learning a new framework and developing a new project."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First domain code benchmark with a knowledge corpus, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers multiple methods (SFT/RAG/Agent), multiple LLMs, and multi-dimensional analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with thorough construction details.
- Value: ⭐⭐⭐⭐⭐ — Provides critical infrastructure for research on domain specialization methods.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](../../ICLR2026/information_retrieval/tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[AAAI 2026\] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description](../../AAAI2026/information_retrieval/oad-promoter_enhancing_zero-shot_vqa_using_large_language_models_with_object_att.md)
- [\[ACL 2026\] Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation](conjecture_and_inquiry_quantifying_software_performance_requirements_via_interac.md)

<!-- RELATED:END -->
