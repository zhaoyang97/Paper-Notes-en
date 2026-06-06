---
title: >-
  [Paper Note] Improving Code Localization with Repository Memory
description: >-
  [ICLR 2026][Code Intelligence][code localization] By leveraging a repository's commit history to construct episodic memory (past commits) and semantic memory (summaries of active code functionality)…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "code localization"
  - "repository memory"
  - "commit history"
  - "language agent"
  - "SWE-bench"
date: 2026-05-08
content_hash: 8ddb4b629505c043
---

# Improving Code Localization with Repository Memory

**Conference**: ICLR 2026
**arXiv**: [2510.01003](https://arxiv.org/abs/2510.01003)  
**Code**: N/A  
**Area**: Software Engineering / LLM Agent
**Keywords**: code localization, repository memory, commit history, language agent, SWE-bench

## TL;DR

By leveraging a repository's commit history to construct episodic memory (past commits) and semantic memory (summaries of active code functionality), this work enhances the code localization capability of language agents, achieving significant improvements on SWE-bench.

## Background & Motivation

Code localization—identifying the files and code snippets that need to be modified—is a critical first step in repository-level software engineering tasks such as bug fixing. Existing approaches, including retrieval-based (CodeRankEmbed), procedural (Agentless), and agent-based (LocAgent) methods, share a common limitation: they treat each issue as a novel puzzle to be solved from scratch, without leveraging any prior knowledge of the repository.

In contrast, human developers accumulate long-term repository memory over time—including an understanding of core module functionality and associations between bug patterns and their fix locations. This accumulated experience enables developers to become experts in a codebase. The paper illustrates this point through an analysis of LocAgent's failure cases on the django repository: without repository knowledge, the agent must construct complex reasoning chains to trace data flow and function calls, making it prone to premature termination or reasoning errors. An experienced developer, however, can leverage memories of past commits to quickly identify relevant modules.

The core insight of this paper is that commit history is a rich yet underutilized resource that can naturally serve as a basis for constructing repository memory for agents.

## Method

### Overall Architecture

RepoMem extends the existing LocAgent framework with memory tools, constructing two complementary memory stores: episodic memory and semantic memory. These memory tools are integrated in a modular fashion into LocAgent's ReAct loop, working in concert with the original code navigation tools.

### Key Designs

1. **Episodic Memory of Past Commits**:

    - **Construction**: A structured corpus is crawled and preprocessed from the repository's recent commit history (7,000 commits prior to issue creation), storing code patches, commit messages, timestamps, and associated issue links.
    - **Filtering Mechanism**: Issues with textual overlap with test instances—along with their associated commits—are excluded to prevent data leakage.
    - **Tool Interface**:
        - `SearchCommit(query, top_k)`: Uses BM25 to retrieve historical commits matching a query, returning commit SHAs, messages, and lists of modified files.
        - `ExamineCommit(id)`: Retrieves the full context for a given commit ID, including the diff patch and associated issue content.
    - **Design Motivation**: Simulates a developer's episodic memory by providing concrete exemplars of past problem resolution.

2. **Semantic Memory of Active Code Functionality**:

    - **Construction**: Commit frequency is analyzed to identify the top 200 most frequently modified files (development hotspots), and an LLM is used to generate high-level functional summaries for each file.
    - **Tool Interface**:
        - `ViewSummary(file_name)`: Retrieves the cached summary for a specific file.
        - `SearchSummary(query, top_k)`: Performs keyword search over all file summaries, returning the most relevant (file, summary) pairs.
    - **Design Motivation**: Provides architectural-level contextual knowledge of the codebase, guiding the agent toward the most active and relevant code areas and preventing it from getting lost in a large repository.

3. **Integration with LocAgent**:

    - LocAgent's three core tools are: SearchEntity (search for code entities), TraverseGraph (multi-hop graph traversal), and RetrieveEntity (retrieve source code).
    - RepoMem directly adds memory tools to the action space, creating a synergy between high-level memory-guided reasoning and low-level structured code analysis: the agent first uses memory to form hypotheses, then employs LocAgent's tools for detailed verification.

### Retrieval Method Selection

Three retrieval methods are compared experimentally:
- BM25 + LLM tokenizer (capable of recognizing code entity names such as `MigrationWriter`): best performance.
- BM25 + whitespace tokenizer: second best.
- Dense retrieval (GritLM-7B): worst performance.

This outcome stems from the "rigid" vocabulary characteristic of code repositories—semantically similar but functionally distinct entities (e.g., `MigrationWriter` vs. `OperationWriter`) require exact matching rather than semantic matching.

## Key Experimental Results

### Main Results

All experiments use GPT-4o (2024-05-13) as the backbone LLM.

| Method | SWE-bench-verified Acc@1 | Acc@3 | Acc@5 | SWE-bench-live Acc@1 | Acc@3 | Acc@5 |
|--------|--------------------------|-------|-------|----------------------|-------|-------|
| CodeRankEmbed | 29.6 | 45.1 | 54.3 | 26.2 | 44.6 | 52.3 |
| Agentless | 53.3 | 67.8 | 71.4 | 40.0 | 60.0 | 62.3 |
| LocAgent | 64.8 | 70.4 | 71.6 | 59.2 | 60.8 | 63.1 |
| RepoMem (episodic-only) | 67.8 | 72.4 | 74.3 | 60.0 | 61.5 | 64.6 |
| RepoMem (semantic-only) | 65.0 | 71.0 | 72.8 | 56.9 | 61.5 | 63.9 |
| **RepoMem (full)** | **68.6** | **74.5** | **76.5** | **60.8** | **63.9** | **66.2** |

Acc@5 improves by 4.9% on SWE-bench-verified and by 3.1% on SWE-bench-live.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Episodic memory only | Acc@5 = 74.3 | Referencing historical commits yields significant gains |
| Semantic memory only | Acc@5 = 72.8 | Helps focus on active code areas |
| Both combined | Acc@5 = 76.5 | Complementary information yields best performance |
| BM25 (LLM tokenizer) | django Acc@5 = 79.7 | Outperforms dense retrieval at 65.8 |

### Key Findings

- **Per-repo analysis**: Repositories with richer commit histories benefit more (e.g., sympy improves by 16.7%), while repositories with sparse histories may see performance degradation (the "others" group drops by 13.1%).
- **Shift in agent behavior**: With memory introduced, the agent significantly reduces its reliance on TraverseGraph and RetrieveEntity, shifting toward a more targeted, hypothesis-driven exploration strategy.
- **Cost-efficiency analysis**: Average cost increases, but exhibits high variance at the per-example level—some issues are resolved much more cheaply when memory directly identifies the target, while others incur additional overhead when memory proves unhelpful. Extra costs are primarily concentrated on difficult instances where LocAgent itself fails, indicating that the agent strategically allocates more resources to harder problems.

## Highlights & Insights

1. **Natural advantage of repository memory**: Commit history is a natural record of a repository's evolution; it enables the construction of high-quality memory without additional annotation, representing an elegant and practical design choice.
2. **Analogy to cognitive science**: Episodic memory corresponds to a developer's "recall of past experiences," while semantic memory corresponds to "understanding of module functionality"—their synergy reflects the actual working patterns of human developers.
3. **Sparse retrieval > dense retrieval**: In the code domain, exact keyword matching outperforms semantic matching, a finding with important implications for the application of RAG to code tasks.
4. **Modular design**: The memory tools can be readily integrated into any ReAct-based agent framework.

## Limitations & Future Work

1. **Poor performance on repositories with sparse history**: When a repository's commit history is limited, memory retrieval may return irrelevant information that interferes with reasoning.
2. **Lack of adaptive memory usage strategy**: The agent currently cannot determine when to leverage memory versus when to reason from scratch; future work could train agents to make dynamic decisions based on issue novelty.
3. **Validation limited to file-level localization**: Performance at the function level or line level is not demonstrated.
4. **Restricted to bug-fixing scenarios**: Applicability to other repository-level tasks (e.g., feature development, refactoring) remains unexplored.

## Related Work & Insights

- **LocAgent** (Chen et al., 2025): The foundational framework of this work, which represents code structure and dependencies via a heterogeneous graph.
- **Agentless** (Xia et al., 2025): A representative procedural approach that directly leverages LLMs and repository structure for localization.
- **SWE-Exp** (Chen et al., 2025): Distills procedural knowledge from an agent's past success/failure trajectories; orthogonal to this work, which derives memory from commit history.
- **Agent Workflow Memory** (Wang et al., 2025): A broader line of research on memory-augmented agents; this paper focuses on the code-specific setting.

## Rating

- Novelty: ⭐⭐⭐⭐ — Using commit history as a memory source is a natural yet novel idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two benchmarks, multi-faceted analysis, per-repo analysis, and cost analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, vivid case studies, and in-depth analysis.
- Value: ⭐⭐⭐⭐ — Practical improvements to code agents via a simple, effective, and easily generalizable method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](../../ACL2026/code_intelligence/learning_adaptive_parallel_execution_for_efficient_code_localization.md)
- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](../../ACL2026/code_intelligence/swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](../../ACL2026/code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](../../ACL2026/code_intelligence/reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)

</div>

<!-- RELATED:END -->
