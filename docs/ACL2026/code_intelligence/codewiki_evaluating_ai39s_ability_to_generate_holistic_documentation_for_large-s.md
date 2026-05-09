---
title: >-
  [Paper Note] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases
description: >-
  [ACL 2026][Code documentation generation] This paper presents CodeWiki, an open-source framework based on hierarchical decomposition and recursive multi-agent processing for automated repository-level code documentation generation. It also introduces the CodeWikiBench benchmark, achieving a quality score of 68.79% across seven programming languages, surpassing the closed-source system DeepWiki (64.06%).
tags:
  - ACL 2026
  - Code documentation generation
  - repository-level understanding
  - multi-agent systems
  - hierarchical decomposition
  - code benchmarks
date: 2026-05-08
content_hash: 1e3a07582e48251c
---

# CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases

**Conference**: ACL 2026
**arXiv**: [2510.24428](https://arxiv.org/abs/2510.24428)
**Code**: [GitHub](https://github.com/FSoft-AI4Code/CodeWiki)
**Area**: Code Intelligence
**Keywords**: Code documentation generation, repository-level understanding, multi-agent systems, hierarchical decomposition, code benchmarks

## TL;DR

This paper presents CodeWiki, an open-source framework based on hierarchical decomposition and recursive multi-agent processing for automated repository-level code documentation generation. It also introduces the CodeWikiBench benchmark, achieving a quality score of 68.79% across seven programming languages, surpassing the closed-source system DeepWiki (64.06%).

## Background & Motivation

**State of the Field**: As codebases continue to grow in scale and complexity, maintaining comprehensive and up-to-date documentation has become a critical bottleneck in software development. Approximately 31% of developers already rely heavily on AI for code documentation, reflecting the urgent demand for automated documentation generation.

**Limitations of Prior Work**: Existing approaches primarily focus on function-level and file-level documentation generation (e.g., CodeBERT, DocAgent), and struggle to scale to the repository level. Repository-level documentation must capture architectural patterns, cross-module interactions, data flows, and system-level design decisions — capabilities that current tools lack due to their inability to model semantic dependencies and hierarchical structures. Evaluation methodology is also inadequate: traditional BLEU/ROUGE metrics fail to capture the multidimensional nature of documentation quality, and no systematic benchmark exists for repository-level documentation.

**Root Cause**: Repository-level documentation generation requires simultaneous understanding of both local implementation details and global architectural relationships, yet LLMs are constrained by limited context windows and cannot process large codebases in a single pass. Most existing approaches also suffer from severely insufficient multilingual support, focusing primarily on Python.

**Paper Goals**: To construct a scalable, multilingual framework for automated repository-level documentation generation, accompanied by a reliable evaluation methodology.

**Starting Point**: Drawing inspiration from dynamic programming, the framework decomposes large repositories into manageable modules via hierarchical partitioning, then recursively generates and synthesizes documentation in a bottom-up manner.

**Core Idea**: Repository-level documentation generation is decomposed into three stages — static analysis and module decomposition, recursive agent-based documentation generation, and hierarchical assembly and synthesis — with a dynamic delegation mechanism enabling adaptive processing of repositories at arbitrary scale.

## Method

### Overall Architecture

The CodeWiki framework consists of three main stages: (1) **Repository Analysis** — constructing a dependency graph via AST/LLM parsing and identifying high-level components, followed by hierarchical module decomposition; (2) **Recursive Documentation Generation** — assigning a dedicated agent to each leaf module, equipped with capabilities for source code access, module tree browsal, documentation workspace manipulation, and dependency graph traversal, with dynamic delegation to sub-agents when module complexity exceeds single-pass capacity; (3) **Hierarchical Assembly** — synthesizing sub-module documentation into parent-module architectural overviews in a bottom-up manner, ultimately producing comprehensive documentation with multimodal elements including architecture diagrams and data flow visualizations.

### Key Designs

1. **Hierarchical Module Decomposition**:

    - **Function**: Partitions large repositories into manageable modular units.
    - **Mechanism**: Employs Tree-Sitter parsers to extract ASTs and constructs a directed dependency graph $G=(V,E)$. Topological sorting identifies zero-indegree entry components (e.g., main functions, API endpoints), which are then recursively partitioned into a module tree. For scalability, the module tree uses only component IDs as input.
    - **Design Motivation**: Applies the divide-and-conquer principle from dynamic programming to resolve the tension between limited LLM context windows and large repository scales, while preserving architectural coherence.

2. **Dynamic Delegation Recursive Agents**:

    - **Function**: Adaptively handles modules of varying complexity.
    - **Mechanism**: Each dedicated leaf-module agent evaluates whether delegation is necessary based on code complexity metrics (cyclomatic complexity, nesting depth), semantic diversity, and context window utilization. When module complexity exceeds a threshold, the agent delegates sub-modules to newly spawned child agents for recursive processing.
    - **Design Motivation**: Ensures the framework can handle repositories of arbitrary scale while maintaining per-module documentation quality, achieving bounded complexity and architectural consistency.

3. **Cross-Module Reference Management & Hierarchical Synthesis**:

    - **Function**: Maintains cross-module documentation consistency and generates global architectural documentation.
    - **Mechanism**: A global registry tracks documented components and their locations; when external components are encountered, cross-references are created rather than duplicating content. Parent-module synthesis follows a multi-stage LLM pipeline: analyzing thematic patterns across child documents → generating architectural overviews → creating feature summaries → developing usage guides → generating visualizations.
    - **Design Motivation**: Eliminates content redundancy and produces an interconnected documentation system that accurately reflects the actual structure and interactions of the codebase.

### Evaluation Framework: CodeWikiBench

The core innovation of CodeWikiBench lies in its **Hierarchical Rubric** design: evaluation criteria are automatically generated from official documentation of open-source projects, mirroring project architecture in a hierarchical structure. Multiple Judge Agents (drawn from different model families) independently assess leaf-level requirements, and final scores along with reliability metrics are computed via weighted bottom-up aggregation. The multi-model consensus mechanism effectively reduces single-model bias.

## Key Experimental Results

### Main Results

| Repository | Language | LOC | CodeWiki | DeepWiki | Gain |
|---|---|---|---|---|---|
| OpenHands | Python | 229K | 82.45% | 73.04% | +9.41% |
| svelte | JavaScript | 125K | 71.96% | 68.51% | +3.45% |
| puppeteer | TypeScript | 136K | 83.00% | 64.46% | +18.54% |
| ml-agents | C# | 86K | 79.78% | 74.80% | +4.98% |
| logstash | Java | 117K | 57.90% | 54.80% | +3.10% |
| wazuh | C | 1.4M | 64.17% | 68.68% | -4.51% |
| electron | C++ | 184K | 42.30% | 44.10% | -1.80% |
| **Average** | | | **68.79%** | **64.06%** | **+4.73%** |

### Cross-Language Analysis

| Language Category | CodeWiki | DeepWiki | Gain |
|---|---|---|---|
| Scripting languages (Python/JS/TS) | 79.14% | 68.67% | +10.47% |
| Managed languages (C#/Java) | 68.84% | 64.80% | +4.04% |
| Systems languages (C/C++) | 53.24% | 56.39% | -3.15% |

### Key Findings
- CodeWiki outperforms all baselines on 5 of 7 repositories, with the largest gain on the TypeScript repository (+18.54%).
- The advantage is most pronounced on high-level scripting languages (+10.47%), but CodeWiki falls slightly below DeepWiki on systems programming languages (C/C++).
- Performance differences are primarily attributable to language characteristics rather than repository scale.
- A preliminary human study indicates CodeWiki is preferred in 7 out of 9 evaluations.

## Highlights & Insights
- **Elegant hierarchical decomposition**: Applying dynamic programming principles to documentation generation elegantly addresses scalability while preserving architectural semantic consistency.
- **Evaluation methodology innovation**: CodeWikiBench's hierarchical rubric generation and multi-model consensus evaluation mechanism provide a systematic solution for repository-level documentation assessment.
- **Open-source transparency**: In a landscape dominated by closed-source systems, the open-source release of CodeWiki carries significant community value.

## Limitations & Future Work
- **Underperformance on systems programming languages**: CodeWiki falls below DeepWiki on C/C++, with insufficient parsing capability for low-level constructs such as pointer arithmetic and template metaprogramming.
- **Evaluation rubrics lack thorough human validation**: Semantic reliability stands at 73.65% and structural reliability at 70.84%.
- **Limited scale of human evaluation**: Only 3 participants × 3 repositories.
- Future directions include developing specialized parsing modules for systems languages, multi-version documentation tracking, and leveraging documentation to support downstream tasks.

## Related Work & Insights
- **vs. DocAgent**: DocAgent employs multi-agent collaboration for function-level documentation generation, whereas CodeWiki focuses on repository-level hierarchical synthesis.
- **vs. DeepWiki**: A closed-source commercial system with generally strong performance but lacking hierarchical decomposition capabilities.
- **vs. OpenDeepWiki/deepwiki-open**: Open-source alternatives that adopt a whole-repository direct prompting approach, resulting in noticeably lower performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The hierarchical decomposition combined with dynamic delegation is a novel design; the CodeWikiBench evaluation methodology offers meaningful methodological contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 7 languages and 7 repositories with cross-language and scalability analyses; human evaluation is limited in scale.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with three clearly organized research questions.
- **Value**: ⭐⭐⭐⭐ Addresses an important gap in automated repository-level documentation generation and evaluation; the open-source release positively impacts the community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)
- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](../../ICLR2026/code_intelligence/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)

<!-- RELATED:END -->
