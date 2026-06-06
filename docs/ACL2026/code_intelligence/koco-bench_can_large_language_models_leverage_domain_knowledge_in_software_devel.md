---
title: >-
  [Paper Note] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?
description: >-
  [ACL 2026][Code Intelligence][Domain code generation] KoCo-Bench proposes the first code benchmark featuring an explicit domain knowledge corpus, covering 11 frameworks and 25 projects across 6 emerging domains (RL…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Domain code generation"
  - "benchmark"
  - "domain specialization"
  - "knowledge corpus"
  - "software engineering"
date: 2026-05-08
content_hash: afe6b38be55b5f06
---

# KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?

**Conference**: ACL 2026  
**arXiv**: [2601.13240](https://arxiv.org/abs/2601.13240)  
**Code**: [https://github.com/jiangxxxue/KOCO-bench](https://github.com/jiangxxxue/KOCO-bench)  
**Area**: Information Retrieval  
**Keywords**: Domain code generation, benchmark, domain specialization, knowledge corpus, software engineering

## TL;DR

KoCo-Bench proposes the first code benchmark featuring an explicit domain knowledge corpus, covering 11 frameworks and 25 projects across 6 emerging domains (RL, Agent, RAG, etc.). It evaluates LLMs' capabilities in acquiring and applying domain knowledge for code generation and understanding, revealing that even the strongest coding agent, Claude Code, achieves only 34.2%.

## Background & Motivation

**Background**: LLMs demonstrate excellent performance in general programming tasks but require specialized domain knowledge (APIs, rules, constraints, etc.) for domain-specific software development. Domain specialization methods such as SFT, RAG, and kNN-LM are employed to assist LLMs in learning and utilizing domain knowledge.

**Limitations of Prior Work**: Existing domain-specific code benchmarks (e.g., EvoCodeBench, DomainEval) only evaluate what LLMs already know rather than how they acquire and apply new knowledge. These benchmarks provide only test sets without explicit knowledge corpora, failing to support research on domain knowledge learning and modeling.

**Key Challenge**: Research on domain specialization methods requires benchmarks to evaluate effectiveness; however, the lack of a knowledge corpus component in existing benchmarks hinders the standardized development of this field.

**Goal**: To construct a complete benchmark comprising a knowledge corpus and a test set to support the evaluation of domain specialization methods in real-world software development.

**Key Insight**: Leveraging the natural ecosystem of software frameworks—specifically the accompanying documentation, source code, and examples (knowledge corpus)—alongside project implementations based on these frameworks (evaluation tasks) creates a complete pipeline from knowledge acquisition to application.

**Core Idea**: Based on 11 emerging frameworks created after 2024, the authors construct a multi-source knowledge corpus (docs + source code + examples). This is paired with multi-granularity code generation tasks (from function-level to project-level, including unit/integration tests) and domain knowledge understanding QA to simulate real-world scenarios where developers build applications using unfamiliar frameworks.

## Method

### Overall Architecture

KoCo-Bench = Knowledge Corpus + Evaluation Tasks. The knowledge corpus is derived from framework documentation, source code, and use cases. The evaluation consists of two tasks: (1) Domain Code Generation—providing project/module/function three-tier requirement descriptions verified via unit and integration tests; (2) Domain Knowledge Understanding—multiple-choice QA to assess mastery of knowledge points within the corpus.

### Key Designs

1.  **Multi-source Knowledge Corpus Construction**:
    - **Function**: Simulates the knowledge sources available to developers when learning a new framework.
    - **Mechanism**: Python frameworks created after March 2024 were selected (to ensure they are not in the LLM training data) that possess comprehensive documentation. These cover 6 domains: RL, Agent, RAG, Model Optimization, Embodied AI, and the Ascend ecosystem. The corpus includes framework documentation (averaging 77K lines), source code, and examples.
    - **Design Motivation**: Selecting emerging frameworks avoids data leakage, and multi-source data ensures knowledge completeness.

2.  **Multi-granularity Code Generation Evaluation**:
    - **Function**: Evaluates domain code generation capabilities from the function level to the project level.
    - **Mechanism**: The benchmark provides three-tier requirement descriptions (project overview → module division → core functions). It includes 131 core functions with 978 tests (averaging 8.6 unit tests per function + integration tests). Requirements underwent multi-round multi-agent disambiguation and manual audit. A Docker environment is used to ensure test reproducibility.
    - **Design Motivation**: Multi-granularity supports the evaluation of various code generation techniques, while strict test suites prevent misjudgment.

3.  **Domain Knowledge Understanding QA**:
    - **Function**: Precisely evaluates LLM mastery of specific knowledge points.
    - **Mechanism**: Atomic multiple-choice questions were designed (one knowledge point per question, supporting multiple correct answers). These were pre-filtered by three LLMs (to remove overly simple questions) and manually audited, resulting in 107 questions.
    - **Design Motivation**: Since code generation tasks make it difficult to pinpoint specific knowledge gaps, QA directly evaluates knowledge understanding.

### Loss & Training

KoCo-Bench is a benchmark rather than a model; its construction required 28.5 person-months. Evaluations were conducted using methods such as direct generation, SFT, RAG, kNN-LM, and Claude Code.

## Key Experimental Results

### Main Results

| Method | Function-level Pass@1 | Project-level Pass | QA Accuracy |
| :--- | :--- | :--- | :--- |
| Claude Sonnet 4.5 (Direct) | ~20% | Extremely Low | ~60% |
| + RAG | Marginal Gain | Marginal Gain | - |
| + SFT | Marginal Gain | Marginal Gain | - |
| **Claude Code (agent)** | **34.2%** | - | - |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Increased corpus scale | Diminishing learning returns | SFT gains decrease on large corpora |
| Cross-domain continuous learning | Catastrophic forgetting | Performance on old domains degrades after learning new ones |
| No knowledge corpus (Direct) | Extremely poor | Proves domain knowledge is not in pre-training data |

### Key Findings

- Even SOTA closed-source LLMs struggle with domain code generation; Claude Code reached only 34.2%.
- Existing domain specialization methods (SFT, RAG, kNN-LM) provide only marginal gains, and cross-domain effectiveness is inconsistent.
- Agent-based methods (Claude Code) are currently the most effective, but significant room for improvement remains.
- The most common errors involve the misuse of domain APIs and violations of domain data constraints.
- Larger knowledge corpora can lead to diminishing returns in learning—existing methods cannot effectively digest large-scale domain knowledge.

## Highlights & Insights

- The "Knowledge Corpus + Test Set" dual-component design is a paradigm innovation in benchmark design, allowing benchmarks to support the development of specialization methods rather than just evaluating performance.
- Selecting post-2024 emerging frameworks avoids data leakage; this temporal control strategy ensures evaluation fairness.
- The multi-round agent-assisted requirement disambiguation process serves as a valuable reference for constructing other benchmarks.

## Limitations & Future Work

- The benchmark currently covers only 6 AI-related domains; non-AI domains (finance, healthcare, etc.) remain to be expanded.
- The scale of 131 core functions is relatively small.
- Framework selection is biased towards the Python ecosystem; other languages need coverage.
- Over time, framework knowledge may gradually enter LLM training data.

## Related Work & Insights

- **vs EvoCodeBench/DomainEval**: These provide only test sets without knowledge corpora, and thus only evaluate existing knowledge rather than knowledge acquisition capabilities.
- **vs SWE-bench**: SWE-bench focuses on issue fixing and does not involve domain knowledge learning. KoCo-Bench simulates the realistic scenario of "learning a new framework + developing a new project."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First domain code benchmark including a knowledge corpus; fills a significant gap)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Covers multiple methods (SFT/RAG/Agent), multiple LLMs, and multi-dimensional analysis)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure and detailed construction details)
- **Value**: ⭐⭐⭐⭐⭐ (Provides critical infrastructure for research into domain specialization methods)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](../../ICLR2026/code_intelligence/dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)
- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](../../ICML2026/code_intelligence/poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)
- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](../../ICLR2026/code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)

</div>

<!-- RELATED:END -->
