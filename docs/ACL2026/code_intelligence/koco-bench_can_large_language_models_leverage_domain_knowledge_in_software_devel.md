---
title: >-
  [Paper Note] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?
description: >-
  [ACL 2026][Code Intelligence][Paper Note] KoCo-Bench introduces the first code benchmark featuring an explicit domain knowledge corpus, covering 11 frameworks and 25 projects across 6 emerging areas (RL, Agent, RAG, etc.). It evaluates the ability of LLMs to acquire and apply domain knowledge from a corpus for code generation and understanding, revealing that
tags:
  - ACL 2026
  - Code Intelligence
date: 2026-05-08
content_hash: 59a6ed1f8f44c99a
---
# KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?

**Conference**: ACL 2026  
**arXiv**: [2601.13240](https://arxiv.org/abs/2601.13240)  
**Code**: [https://github.com/jiangxxxue/KOCO-bench](https://github.com/jiangxxxue/KOCO-bench)  
**Area**: Information Retrieval  
**Keywords**: Domain code generation, Benchmark, Domain specialization, Knowledge corpus, Software engineering

## TL;DR

KoCo-Bench introduces the first code benchmark featuring an explicit domain knowledge corpus, covering 11 frameworks and 25 projects across 6 emerging areas (RL, Agent, RAG, etc.). It evaluates the ability of LLMs to acquire and apply domain knowledge from a corpus for code generation and understanding, revealing that even the strongest coding agent, Claude Code, achieves only 34.2%.

## Background & Motivation

**Background**: LLMs perform excellently on general programming tasks but require specific domain knowledge (APIs, rules, constraints) for domain-specific software development. Domain specialization methods (SFT, RAG, kNN-LM) are employed to assist LLMs in learning and utilizing such knowledge.

**Limitations of Prior Work**: Existing domain-specific code benchmarks (e.g., EvoCodeBench, DomainEval) only assess existing knowledge within LLMs rather than the acquisition and application of new knowledge. They provide test sets without explicit knowledge corpora, failing to support research on domain knowledge learning and modeling.

**Key Challenge**: Research on domain specialization requires benchmarks to evaluate effectiveness, yet the lack of a knowledge corpus component in existing benchmarks hinders the standardized development of this field.

**Goal**: Construct a complete benchmark comprising a knowledge corpus and a test set to support the evaluation of domain specialization methods in real-world software development.

**Key Insight**: Utilize the natural ecosystem of software frameworks—including official documentation, source code, and examples (knowledge corpus)—alongside project implementations based on these frameworks (evaluation tasks) to form a complete pipeline of knowledge acquisition → knowledge application.

**Core Idea**: Based on 11 emerging frameworks post-2024, a multi-source knowledge corpus (docs + source + examples) is constructed. This is paired with multi-granularity code generation tasks (function-level to project-level, including unit/integration tests) and domain knowledge understanding QA to simulate real-world scenarios where developers work with unfamiliar frameworks.

## Method

### Overall Architecture

KoCo-Bench decomposes the scenario of "a developer learning an unfamiliar framework" into two parts: a **knowledge corpus** for learning, aggregated from official documents, source code, and use cases of 11 emerging frameworks; and **evaluation tasks** to verify learning effectiveness. Given a development requirement, the model first retrieves domain knowledge from the corpus and then produces specific outputs—either writing code that passes unit and integration tests across three levels (project/module/function) or answering multiple-choice questions targeting knowledge points in the corpus. This pipeline covers the closed loop of "knowledge acquisition → knowledge application," rather than testing what the model happens to remember from pre-training.

### Key Designs

**1. Multi-source Knowledge Corpus: Preventing Data Leakage via Time Windows**

The primary concern in domain code evaluation is data leakage, where answers are already present in model weights. KoCo-Bench addresses this by selecting only Python frameworks created after March 2024 with comprehensive documentation, ensuring they are unlikely to appear in the training corpora of mainstream LLMs. These cover six emerging areas: RL, Agent, RAG, Model Optimization, Embodied AI, and the Ascend ecosystem. For each framework, the corpus includes official documentation (averaging 77K lines), source code, and examples. These three source types are complementary, providing normative specifications and practical coding examples.

**2. Multi-granularity Code Generation: Three-tiered Requirements and Rigorous Test Suites**

Domain development involves both micro-tasks (implementing a function) and macro-tasks (building a project). A single granularity cannot characterize the capability boundaries of different code generation techniques. Thus, the benchmark provides tiered requirement descriptions: project overview → module division → core functions. Macro intentions are mapped to 131 core functions equipped with 978 tests (averaging 8.6 unit tests per function plus integration tests). Requirement texts undergo multi-agent disambiguation and human audit to ensure failures are not due to vague descriptions. All evaluations are executed in Docker environments for reproducibility.

**3. Domain Knowledge Understanding QA: Pinpointing Knowledge Gaps via Atomic Questions**

Code generation presents a mixed signal; errors may stem from missing knowledge or implementation mistakes. The QA task addresses this with 107 atomic multiple-choice questions. Each question tests a single knowledge point. Questions are pre-filtered by 3 LLMs to remove trivial items and then manually audited. Incorrect answers map directly to a lack of domain knowledge mastery, complementing code generation by distinguishing between "knowing" and "applying."

## Key Experimental Results

### Main Results

| Method | Function-level Pass@1 | Project-level Pass | QA Accuracy |
|------|-------------|-----------|----------|
| Claude Sonnet 4.5 (Direct) | ~20% | Extremely Low | ~60% |
| + RAG | Marginal Gain | Marginal Gain | - |
| + SFT | Marginal Gain | Marginal Gain | - |
| **Claude Code (agent)** | **34.2%** | - | - |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Increased Corpus Scale | Diminishing Learning Returns | SFT benefits decrease with large corpora |
| Cross-domain Continual Learning | Catastrophic Forgetting | Performance on old domains degrades after learning new ones |
| No Knowledge Corpus (Direct) | Extremely Poor | Proves domain knowledge is not in pre-training |

### Key Findings

- Even SOTA closed-source LLMs struggle with domain-specific code generation; Claude Code reaches only 34.2%.
- Existing domain specialization methods (SFT, RAG, kNN-LM) provide only marginal gains with inconsistent cross-domain performance.
- Agent-based methods (Claude Code) are currently the most effective, yet significant room for improvement remains.
- Common errors include misuse of domain APIs and violation of domain data constraints.
- Larger knowledge corpora lead to diminishing returns, suggesting existing methods cannot effectively digest large-scale domain knowledge.

## Highlights & Insights

- The "knowledge corpus + test set" dual-component design is a paradigm innovation for benchmarks, enabling both performance evaluation and development of specialization methods.
- Selecting frameworks post-2024 creates a temporal control strategy that ensures fairness by preventing data leakage.
- The multi-agent assisted requirement disambiguation process serves as a valuable reference for constructing other benchmarks.

## Limitations & Future Work

- Currently covers only 6 AI-related domains; expansion to non-AI domains (finance, healthcare, etc.) is needed.
- The scale of 131 core functions is relatively small.
- Framework selection is biased towards the Python ecosystem; other languages remain to be covered.
- Over time, framework knowledge may eventually enter LLM training data.

## Related Work & Insights

- **vs. EvoCodeBench/DomainEval**: These provide only test sets without knowledge corpora, assessing existing knowledge rather than acquisition capabilities.
- **vs. SWE-bench**: Focuses on issue fixing without involving domain knowledge learning. KoCo-Bench simulates the real scenario of "learning a new framework + developing a new project."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First domain code benchmark to include a knowledge corpus, filling a critical gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple methods (SFT/RAG/Agent), various LLMs, and multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Structured clearly with detailed construction specifics.
- **Value**: ⭐⭐⭐⭐⭐ Provides essential infrastructure for research into domain specialization methods.

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
