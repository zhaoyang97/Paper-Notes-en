---
title: >-
  [Paper Note] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?
description: >-
  [ACL 2026][Code Intelligence][Paper Note] KoCo-Bench proposes the first code benchmark containing an explicit domain knowledge corpus, covering $11$ frameworks and $25$ projects across $6$ emerging domains (RL, Agent, RAG, etc.). It evaluates the ability of LLMs to acquire and apply domain knowledge for code generation and knowledge understanding from a knowle
tags:
  - ACL 2026
  - Code Intelligence
date: 2026-05-08
content_hash: c5f2cd1461c1f298
---
# KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?

**Conference**: ACL 2026  
**arXiv**: [2601.13240](https://arxiv.org/abs/2601.13240)  
**Code**: [https://github.com/jiangxxxue/KOCO-bench](https://github.com/jiangxxxue/KOCO-bench)  
**Area**: Information Retrieval  
**Keywords**: domain code generation, benchmark, domain specialization, knowledge corpus, software engineering

## TL;DR

KoCo-Bench proposes the first code benchmark containing an explicit domain knowledge corpus, covering $11$ frameworks and $25$ projects across $6$ emerging domains (RL, Agent, RAG, etc.). It evaluates the ability of LLMs to acquire and apply domain knowledge for code generation and knowledge understanding from a knowledge corpus, revealing that even the strongest coding agent, Claude Code, only achieves $34.2\%$.

## Background & Motivation

**Background**: LLMs perform excellently on general programming tasks, but require specialized domain knowledge (APIs, rules, constraints, etc.) in domain-specific software development. Domain specialization methods (SFT, RAG, kNN-LM) are used to help LLMs learn and utilize domain knowledge.

**Limitations of Prior Work**: Existing domain-specific code benchmarks (such as EvoCodeBench, DomainEval) only evaluate what knowledge the LLM already knows, rather than how it acquires and applies new knowledge. They provide only test sets without explicit knowledge corpora, failing to support research on domain knowledge learning and modeling.

**Key Challenge**: Research on domain specialization methods requires benchmarks to evaluate effectiveness, but the lack of knowledge corpus components in existing benchmarks prevents the standardized development of research in this direction.

**Goal**: Construct a complete benchmark containing a "knowledge corpus + test set" to support the evaluation of domain specialization methods in real-world software development.

**Key Insight**: Utilize the natural ecosystem of software frameworks—the documentation, source code, and examples provided by the framework (knowledge corpus)—and the project implementation based on the framework (evaluation tasks) to form a complete chain of "knowledge acquisition → knowledge application."

**Core Idea**: Based on $11$ emerging frameworks from after 2024, construct a multi-source knowledge corpus (documentation + source code + examples), combined with multi-granularity code generation tasks (function-level to project-level, including unit/integration tests) and domain knowledge understanding QA, to simulate the real-world scenario of a developer performing development based on an unfamiliar framework.

## Method

### Overall Architecture

KoCo-Bench deconstructs the real-world scenario of "a developer starting with an unfamiliar framework" into two halves: one half is the **knowledge corpus** available for learning, aggregated from official documentation, source code, and use cases of $11$ emerging frameworks; the other half is the **assessment tasks** to verify learning achievement. Given a development requirement, the model first acquires domain knowledge from the corpus and then applies it to specific outputs—either writing code that passes unit and integration tests across three layers (project/module/function) or answering multiple-choice questions targeting knowledge points in the corpus. The entire chain covers a complete closed loop of "knowledge acquisition → knowledge application," rather than just testing what the model happens to remember from pre-training.

### Key Designs

**1. Multi-source Knowledge Corpus: Closing Data Leakage via Time Windows**

The biggest fear in domain code evaluation is not the difficulty of the questions, but that the answers have already been memorized into the model weights. KoCo-Bench's strategy is to select only Python frameworks created after March 2024 with complete documentation, ensuring chronologically that they cannot appear in the training corpora of mainstream LLMs. These cover six emerging domains: RL, Agent, RAG, model optimization, embodied AI, and the Ascend ecosystem. For each framework, the corpus includes not only official documentation (averaging up to $77K$ lines) but also source code and use cases—these three sources are complementary, providing both normative specifications on "how to use" and examples of "how to write," simulating all learning materials a developer can actually obtain.

**2. Multi-granularity Code Generation: Three-layer Requirements and Strict Test Suites**

Domain development involves both micro-tasks like "implementing a function" and macro-tasks like "building an entire project"; a single granularity cannot characterize the capability boundaries of different code generation technologies. To this end, the benchmark provides a three-layer requirement description: project overview → module division → core functions, mapping macro intentions step-by-step to $131$ core functions, equipped with $978$ tests (an average of $8.6$ unit tests per function, plus integration tests). Requirement texts undergo multiple rounds of multi-agent disambiguation and manual auditing to prevent models from being misjudged due to vague descriptions. All evaluations are executed in Docker environments to ensure reproducibility. Dense test coverage makes it difficult to "accidentally pass compilation" by luck.

**3. Domain Knowledge Understanding QA: Pinpointing Knowledge Gaps via Atomic Multiple-Choice Questions**

Code generation is a mixed signal—errors might stem from missing knowledge or simply engineering implementation failures, making it difficult to attribute them solely to a specific knowledge point. The QA task fills this gap with $107$ atomic multiple-choice questions: each question tests only one knowledge point (supporting multiple selections). Questions are pre-filtered by $3$ LLMs to remove overly simple items and then manually audited, so that an incorrect answer directly maps to "the model does not master this domain knowledge." It complements the code generation task—the former locates "whether it knows," and the latter tests "whether it can use it."

## Key Experimental Results

### Main Results

| Method | Function-level Pass@1 | Project-level Pass | QA Accuracy |
|------|-------------|-----------|----------|
| Claude Sonnet 4.5 Direct Gen | ~$20\%$ | Extremely Low | ~$60\%$ |
| + RAG | Marginal Gain | Marginal Gain | - |
| + SFT | Marginal Gain | Marginal Gain | - |
| **Claude Code (agent)** | **$34.2\%$** | - | - |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Increased Knowledge Corpus Scale | Diminishing learning returns | SFT gains decrease on large corpora |
| Cross-domain Continual Learning | Catastrophic forgetting | Degradation in old domains after learning new ones |
| No Knowledge Corpus (Direct Gen) | Extremely poor | Proves domain knowledge is not in pre-training |

### Key Findings

- Even SOTA closed-source LLMs struggle with domain code generation; Claude Code achieves only $34.2\%$.
- Existing domain specialization methods (SFT, RAG, kNN-LM) yield only marginal improvements, with inconsistent effects across domains.
- Agent methods (Claude Code) are currently the most effective, but there is still significant room for improvement.
- The most common errors are the misuse of domain APIs and violations of domain data constraints.
- As the knowledge corpus grows larger, learning effectiveness actually diminishes—existing methods cannot effectively digest large-scale domain knowledge.

## Highlights & Insights

- The dual-component design of "knowledge corpus + test set" is a paradigm innovation in benchmark design—enabling the benchmark to not only evaluate performance but also support the development of domain specialization methods.
- Selecting emerging frameworks after 2024 to avoid data leakage; this time-control strategy ensures the fairness of the evaluation.
- The multi-round agent-assisted requirement disambiguation process is worth emulating by other benchmark constructions.

## Limitations & Future Work

- Only covers $6$ AI-related domains; non-AI domains (finance, healthcare, etc.) remain to be expanded.
- The scale of $131$ core functions is relatively small.
- Framework selection leans toward the Python ecosystem; other languages are yet to be covered.
- Over time, framework knowledge may gradually enter LLM training data.

## Related Work & Insights

- **vs EvoCodeBench/DomainEval**: These provide only test sets without knowledge corpora, and can only evaluate existing knowledge rather than knowledge acquisition capabilities.
- **vs SWE-bench**: Focuses on issue fixing and does not involve domain knowledge learning. KoCo-Bench simulates the real-world scenario of "learning a new framework + developing a new project."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first domain code benchmark to include a knowledge corpus, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple methods (SFT/RAG/Agent), multiple LLMs, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed construction details.
- Value: ⭐⭐⭐⭐⭐ Provides critical infrastructure for the study of domain specialization methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ICML 2026\] Physics Is All You Need? A Case Study in Physicist-Supervised AI Development of Scientific Software](../../ICML2026/code_intelligence/physics_is_all_you_need_a_case_study_in_physicist-supervised_ai_development_of_s.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)
- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](../../ICLR2026/code_intelligence/dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)
- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](../../ICML2026/code_intelligence/poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)

</div>

<!-- RELATED:END -->
