---
title: >-
  [Paper Note] SWE-QA: Can Language Models Answer Repository-level Code Questions?
description: >-
  [ACL 2026][Code Intelligence][Repository-level Code Understanding] SWE-QA constructs a repository-level code QA benchmark covering 15 real-world Python repositories and 720 high-quality QA pairs. The benchmark uses GitHu…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Repository-level Code Understanding"
  - "Code QA"
  - "RAG"
  - "Software Engineering Agent"
  - "SWE-Bench"
date: 2026-05-08
content_hash: d0087375d82c50c4
---

# SWE-QA: Can Language Models Answer Repository-level Code Questions?

**Conference**: ACL 2026  
**arXiv**: [2509.14635](https://arxiv.org/abs/2509.14635)  
**Code**: https://github.com/peng-weihan/SWE-QA-Bench  
**Area**: Code Intelligence / Repository-level Question Answering  
**Keywords**: Repository-level Code Understanding, Code QA, RAG, Software Engineering Agent, SWE-Bench

## TL;DR
SWE-QA constructs a repository-level code QA benchmark covering 15 real-world Python repositories and 720 high-quality QA pairs. The benchmark uses GitHub issues to induce question types and employs human verification for answers. Experiments demonstrate that vanilla LLM direct prompting is insufficient; RAG and tool-integrated agents like OpenHands or SWE-agent are required to meet real-world development QA needs.

## Background & Motivation
**Background**: Evaluation of code QA has long skewed toward localized problems such as functions, code snippets, API documentation, or StackOverflow-style queries. Benchmarks like CoSQA, CodeQA, and CodeQueries primarily test the ability to explain isolated code blocks. Although repository-level datasets like CodeRepoQA, CoreQA, and Spyder-CodeQA have recently emerged, they remain unsystematic in question coverage, cross-file dependencies, and human verification.

**Limitations of Prior Work**: In real-world software development, developers rarely ask "what does this line of code mean." Instead, they ask questions such as "where is a specific feature implemented," "why does this class access a certain attribute lazily," or "how does a test take effect across routes, configurations, and request contexts." These questions require models to navigate across multiple files, classes, functions, and control flows; relying solely on parametric memory or a single retrieved snippet frequently misses critical dependencies.

**Key Challenge**: While code LLMs are increasingly proficient at writing local code, evaluation systems still do not adequately test the understanding of "the repository as a system." Snippet-level benchmarks might make models appear strong without demonstrating their capability to answer questions about system design, dependency tracking, or feature localization asked by actual maintainers.

**Goal**: The authors aim to fill this gap by providing an evaluation perspective closer to real software engineering. This involves abstracting a repository-level question taxonomy from developer issues and constructing a reusable QA generation and human verification pipeline to compare direct prompting, RAG, agents, and commercial code assistants.

**Key Insight**: Instead of generating questions from synthetic templates, the paper crawls GitHub issues from repositories associated with SWE-Bench to observe how developers actually ask questions. These are categorized into four major types (What / Why / Where / How) across 12 fine-grained intents, ensuring the benchmark distribution resembles real-world development environments rather than a collection of academic tasks.

**Core Idea**: Use a GitHub issue-driven question taxonomy combined with static code structure and human verification to construct an extensible benchmark that evaluates cross-file, multi-hop, repository-level reasoning.

## Method
SWE-QA serves as a benchmark study. Its methodology encompasses a complete repository-level QA production pipeline: understanding developer questions from issues, templatizing question types, parsing target repositories to instantiate questions around specific elements, generating initial answers via RAG, and finally having experienced developers revise, cross-verify, and filter the data.

### Overall Architecture
The input consists of raw open-source Python repositories and developer questions extracted from GitHub issues. The output is 720 repository-level QA pairs, where each question is bound to a target repository, a question type, a context requiring multi-hop reasoning, and a human-verified long-form answer.

The pipeline comprises four steps:
1. Crawling and analyzing GitHub issues to establish a question taxonomy and seed templates.
2. Parsing repository structures using tree-sitter to instantiate questions around focal code elements.
3. Generating initial reference answers using Retrieval-Augmented Generation (RAG).
4. Expert revision of answers, filtering of low-quality samples, and category balancing.

### Key Designs
1. **Repository-level Question Taxonomy Induced from Real Issues**:
	- **Function**: Compresses questions raised by real developers in issues into a reusable taxonomy.
	- **Mechanism**: The authors crawled 77,100 GitHub issues from 12 popular SWE-Bench repositories, filtered 41,955 issues with at least 1,000 characters, and used an LLM to extract 127,415 explicit code-understanding questions. 1,000 samples were manually coded to form four major categories (What, Why, Where, How) and 12 fine-grained intents, such as Architecture exploration, Dependency tracing, Design rationale, Feature Location, and Algorithm Implementation.
	- **Design Motivation**: Subjectively designed question types often favor "easy-to-label" local issues. Inducing the taxonomy from issues preserves systemic problems found in maintenance and ensures templates align with real development scenarios.

2. **Structure-Based Question Instantiation and Answer Generation**:
	- **Function**: Converts abstract seed questions into specific, multi-hop questions within a real repository.
	- **Mechanism**: Tree-sitter is used to parse repositories and extract classes, functions, methods, and dependencies. A compact subgraph is selected around a focal element to fill the seed template. For answer generation, a code element index is built, combining semantic similarity and structural dependencies to retrieve relevant code, documentation, and architecture info. A strong model generates initial answers based on this context, required to cite code locations and avoid hallucinations.
	- **Design Motivation**: Repository-level context is long and sparse. Neither stuffing the entire repository into a model nor using single functions is feasible. Using structural subgraphs and retrieved contexts balances realism with cost.

3. **Expert Revision and Data Validation**:
	- **Function**: Ensures each QA pair is correct, comprehensive, and balanced across repositories.
	- **Mechanism**: Two experts with at least three years of experience independently check facts, completeness, and phrasing. Disagreements are resolved by a third expert. Vague questions, factual errors, or answers unsupported by the repository are filtered. Each repository is mandated to have 48 samples, balanced across the four major categories.
	- **Design Motivation**: Repository-level answers are long (mean 266.64 words) and involve multiple entities (mean 8.71 functions, 3.19 files). LLM generation often yields locally correct but logically incomplete answers; human cross-verification is essential for benchmark reliability.

### Loss & Training
This work does not train a new model. The evaluation strategy uses SWE-QA as a test set to compare the answer quality of six LLMs under different enhancement methods: direct prompting, Function Chunking RAG, Sliding Window RAG, SWE-agent, and OpenHands. Automated evaluation uses Claude Sonnet 4.5 as an LLM-as-Judge, scoring across five dimensions: correctness, completeness, relevance, clarity, and coherence (20 points each, 100 total). Judge bias is mitigated via system anonymization, answer randomization, and human evaluation slices.

## Key Experimental Results

### Main Results
SWE-QA includes 720 questions across 15 Python repositories, 13,300 files, 22,522 classes, 142,404 functions, and over 3.4 million lines of code. On average, each question requires 8.71 functions, 3.19 files, a reasoning chain depth of 4.72, and a dependency chain depth of 2.96. 90.9% of questions have a reasoning chain depth $> 1$, and 77.6% require cross-file knowledge.

| System / Method | Overall | Key Info | Conclusion |
|-----------------|---------|----------|------------|
| Qwen3-Coder-30B direct | 50.80 | No repo context | Weakest performance |
| Qwen3-Coder-30B + Sliding Window RAG | 64.86 | +14.06 Gain | Retrieval significantly improves results |
| Qwen3-Coder-30B + OpenHands | 65.88 | +15.08 Gain | Agent use helps but is unstable for small models |
| GLM-4.6 + OpenHands | 70.15 | Near best | Strong models with agents are highly competitive |
| GPT-5.1 direct | 61.41 | Strongest base | Still lower than tool-augmented systems |
| GPT-5.1 + OpenHands | 70.79 | Best in table | Agent framework yields highest score |
| Cursor | 70.66 | Commercial tool | Comparable to best open combinations |
| Tongyi Lingma | 69.07 | Commercial tool | Validates end-to-end engineering retrieval |

### Ablation Study
The paper analyzes question types, repository sources, and evaluation protocols as a breakdown of benchmark difficulty.

| Analysis Dimension | Key Results | Description |
|--------------------|-------------|-------------|
| Why-type questions | Avg 69.77 | Design rationale/purpose usually have comments/semantic cues, easier to answer |
| How-type questions | Avg 69.13 | Requires process understanding but supported by structural context |
| Where-type questions| Avg 66.76 | Requires precise localization; demanding for retrieval and cross-file tracking |
| What-type questions | Avg 65.81 | Architecture exploration scroes only 61.84; one of the hardest subcategories |
| SWE-Bench repos | Avg 68.59 | Easier compared to SWE-Bench-Live |
| SWE-Bench-Live repos| Avg 64.98 | -3.61 points; likely less affected by data leakage |
| Human Eval (GPT-5.1+OpenHands) | 82.33 | Consistent with LLM-as-Judge ranking, supports auto-eval reliability |

### Key Findings
- **Context acquisition determines the ceiling**: Direct prompting is insufficient for repository-level questions. RAG provides a stable boost, while agents (OpenHands/SWE-agent) further improve performance with strong models.
- **What and Where questions are more difficult**: They require precise implementation localization and reconstruction of architectural relationships rather than general explanations of intent.
- **Agent cost is high**: OpenHands averages ~87,045 input tokens and ~1,930 output tokens per question, with SWE-agent even higher, indicating performance gains come with significant overhead.
- **Complexity Matters**: Large complex repositories like Pylint are significantly harder than smaller ones like Flask or Requests; repository scale and architectural complexity directly impact QA difficulty.

## Highlights & Insights
- The paper concretizes "repository-level understanding" into an evaluable data structure. By inducing taxonomy from issues, it shows that real-world questions are not just API lookups but involve cross-file localization, design rationale, and system behavior.
- Data statistics are compelling: questions involving an average of 3.19 files and 8.71 functions prove that SWE-QA is at a different difficulty tier than traditional snippet-based Code QA.
- The RAG vs. Agent comparison is practical. Results show that even GPT-5.1 direct only achieves 61.41, requiring retrieval and tool execution to reach 70+, which serves as a guide for designing code assistants.
- This benchmark can be adapted for internal documentation QA or test failure explanation by inducing templates from private tickets and using structural parsing for local datasets.

## Limitations & Future Work
- Currently covers only Python repositories primarily from SWE-Bench. The language and ecosystem coverage remain narrow; Java, TypeScript, or C++ would introduce different build systems and dependency challenges.
- The scale of 720 QA pairs is high-quality but small. Larger-scale, continuously updated data is needed for training or fine-tuning repository-level models.
- Automated evaluation relies on LLM-as-Judge. Despite human validation, fine-grained factual errors in complex code answers might still be missed.
- Answers are primarily natural language. Future work could integrate QA with repository operations, such as executing tests or running static analysis to provide verifiable patches.

## Related Work & Insights
- **vs CoSQA / CodeQA**: These focus on snippet/function-level QA. SWE-QA evaluates repository-level, multi-hop, and cross-file reasoning.
- **vs CodeRepoQA / CoreQA**: These are repo-level but SWE-QA emphasizes the combination of question taxonomy, modular info, and human validation.
- **vs SWE-Bench**: SWE-Bench focuses on issue resolution/patch generation, while SWE-QA focuses on understanding. They are complementary: one tests "can you fix it," the other "do you understand the system."
- **Insight for Code Assistants**: Simple embedding-based retrieval of function blocks is insufficient. Robust assistants need structural indexing, cross-file dependency tracking, and iterative tool use to answer Why/Where questions.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Inducing taxonomy from real issues is solid; more grounded in software engineering than typical Code QA.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers 6 LLMs, 5 enhancement types, and commercial tools, though lacks more languages and execution-based settings.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear pipeline and evaluation metrics; dense information in tables.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant for code assistants, repo-level RAG, and SE agents; a directly reusable benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)
- [\[ACL 2026\] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)
- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](../../ICML2026/code_intelligence/matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)
- [\[ACL 2026\] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)
- [\[ICML 2026\] SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale](../../ICML2026/code_intelligence/swe-rebench_v2_language-agnostic_swe_task_collection_at_scale.md)

</div>

<!-- RELATED:END -->
