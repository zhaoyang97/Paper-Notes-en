---
title: >-
  [Paper Note] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases
description: >-
  [ACL 2026][Code Intelligence][Code Documentation Generation] CodeWiki is proposed as an open-source framework based on hierarchical decomposition and recursive multi-agent processing for the automated generation of repos…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code Documentation Generation"
  - "Repository-level Understanding"
  - "Multi-agent Systems"
  - "Hierarchical Decomposition"
  - "Code Benchmark"
date: 2026-05-08
content_hash: 518211df338c4c5b
---

# CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases

**Conference**: ACL 2026  
**arXiv**: [2510.24428](https://arxiv.org/abs/2510.24428)  
**Code**: [GitHub](https://github.com/FSoft-AI4Code/CodeWiki)  
**Area**: Code Intelligence  
**Keywords**: Code Documentation Generation, Repository-level Understanding, Multi-agent Systems, Hierarchical Decomposition, Code Benchmark

## TL;DR

CodeWiki is proposed as an open-source framework based on hierarchical decomposition and recursive multi-agent processing for the automated generation of repository-level code documentation. The authors also establish the CodeWikiBench benchmark, where the framework achieves a quality score of 68.79% across seven programming languages, outperforming the closed-source system DeepWiki (64.06%).

## Background & Motivation

**Background**: As the scale and complexity of codebases continue to grow, maintaining comprehensive and timely documentation has become a core bottleneck in software development. Approximately 31% of developers already extensively use AI to assist in code documentation, reflecting an urgent demand for automated solutions.

**Limitations of Prior Work**: Existing methods primarily focus on function-level and file-level documentation generation (e.g., CodeBERT, DocAgent), which are difficult to scale to the repository level. Repository-level documentation requires capturing architectural patterns, cross-module interactions, data flows, and system-level design decisions, yet existing tools lack the capability to model these semantic dependencies and hierarchical structures. Furthermore, evaluation systems are inadequate—traditional BLEU/ROUGE metrics fail to capture the multi-dimensional features of documentation quality, and systematic benchmarks for repository-level documentation are missing.

**Key Challenge**: Repository-level documentation generation requires simultaneous understanding of local implementation details and global architectural relationships. However, the limited context window of LLMs prevents processing large codebases at once. Additionally, existing multi-language support is severely lacking, with most research focusing solely on Python.

**Goal**: To build a scalable, multi-language automated framework for repository-level documentation generation while providing a reliable evaluation methodology.

**Key Insight**: Drawing inspiration from dynamic programming, the framework utilizes hierarchical decomposition to break down large repositories into manageable modules, followed by recursive bottom-up generation and synthesis of documentation.

**Core Idea**: The repository-level documentation generation is divided into three stages: static analysis and module decomposition, recursive agent documentation generation, and hierarchical assembly and synthesis. A dynamic delegation mechanism is employed to achieve adaptive processing for repositories of any scale.

## Method

### Overall Architecture

The CodeWiki framework consists of three main stages: (1) Repository Analysis Stage—parsing via AST/LLM to construct dependency graphs and identify high-level components, followed by hierarchical module decomposition; (2) Recursive Documentation Generation Stage—assigning dedicated agents to each leaf module, equipped with source code access, module tree navigation, documentation workspace operations, and dependency graph traversal capabilities. If module complexity exceeds single-session capacity, tasks are dynamically delegated to sub-agents; (3) Hierarchical Assembly Stage—synthesizing sub-module documentation bottom-up into architectural overviews for parent modules, ultimately producing comprehensive multimodal documentation including architecture diagrams and data flow visualizations.

### Key Designs

1.  **Hierarchical Module Decomposition**:
    - **Function**: Decomposes large repositories into manageable modular units.
    - **Mechanism**: Tree-Sitter parsers are used to extract ASTs and construct a directed dependency graph $G=(V,E)$. Entry components with zero in-degree (e.g., main functions, API endpoints) are identified via topological sorting, then recursively partitioned into a module tree. To ensure scalability, the module tree uses only component IDs as input.
    - **Design Motivation**: Leveraging the "divide and conquer" strategy of dynamic programming to resolve the conflict between limited LLM context windows and massive repository scales while maintaining architectural consistency.

2.  **Dynamic Delegation Recursive Agents**:
    - **Function**: Adaptively handles modules of varying complexity.
    - **Mechanism**: A dedicated agent for each leaf module determines the need for delegation based on code complexity metrics (cyclomatic complexity, nesting depth), semantic diversity, and context window utilization. When complexity exceeds a threshold, the agent delegates sub-modules to new sub-agents for recursive processing.
    - **Design Motivation**: To ensure the framework can handle repositories of any size while maintaining the documentation quality of each module, achieving bounded complexity and architectural coherence.

3.  **Cross-Module Reference & Hierarchical Synthesis**:
    - **Function**: Maintains documentation consistency across modules and generates global architectural documentation.
    - **Mechanism**: A global registry tracks documented components and locations; when external components are encountered, cross-references are created instead of redundant content. Hierarchical synthesis for parent modules involves multi-stage LLM processing: analyzing thematic patterns in sub-documentation, generating architectural overviews, creating feature summaries, developing usage guides, and generating visualization charts.
    - **Design Motivation**: To avoid content redundancy and generate an interconnected documentation system that truly reflects the actual structure and interactions of the codebase.

### Evaluation Framework: CodeWikiBench

The core innovation of CodeWikiBench lies in the design of a Hierarchical Rubric: evaluation criteria are automatically generated from the official documentation of open-source projects, mirroring the project architecture. The evaluation process involves multiple Judge Agents (using different model families) independently judging leaf-level requirements, followed by weighted aggregation to calculate final scores and reliability metrics bottom-up. The multi-model consensus mechanism effectively reduces single-model bias.

## Key Experimental Results

### Main Results

| Repository | Language | LOC | CodeWiki | DeepWiki | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
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
| :--- | :--- | :--- | :--- |
| Scripting (Python/JS/TS) | 79.14% | 68.67% | +10.47% |
| Managed (C#/Java) | 68.84% | 64.80% | +4.04% |
| Systems (C/C++) | 53.24% | 56.39% | -3.15% |

### Key Findings
- CodeWiki outperforms all baselines in 5 out of 7 repositories, with the largest gain seen in the TypeScript repository (+18.54%).
- The advantage is most significant in high-level scripting languages (+10.47%), though it slightly trails DeepWiki in system programming languages (C/C++).
- Performance variations are primarily attributed to language characteristics rather than repository scale.
- Preliminary human studies show CodeWiki was preferred in 7 out of 9 evaluations.

## Highlights & Insights
- **Ingenious Hierarchical Decomposition**: Applying dynamic programming concepts to documentation generation effectively solves scalability issues while preserving architectural semantics.
- **Innovative Evaluation Methodology**: The hierarchical rubric generation and multi-model consensus judging in CodeWikiBench provide a systematic solution for repository-level evaluation.
- **Open-Source Transparency**: In a landscape dominated by closed-source systems, the open-source release of CodeWiki holds significant community value.

## Limitations & Future Work
- **Suboptimal Performance in Systems Languages**: Performance in C/C++ is lower than DeepWiki due to insufficient parsing capabilities for low-level constructs like pointer operations and template metaprogramming.
- **Rubrics Lack Extensive Human Validation**: Semantic reliability is at 73.65% and structural reliability at 70.84%.
- **Limited Scale of Human Evaluation**: Only 3 participants across 3 repositories were involved.
- **Future Directions**: Developing specialized parsing modules for systems languages, multi-version documentation tracking, and utilizing documentation to support downstream tasks.

## Related Work & Insights
- **vs DocAgent**: DocAgent uses multi-agent collaboration for function-level documentation, whereas CodeWiki focuses on repository-level hierarchical synthesis.
- **vs DeepWiki**: A closed-source commercial system that performs well overall but lacks explicit hierarchical decomposition capabilities.
- **vs OpenDeepWiki/deepwiki-open**: Open-source alternatives that use direct whole-repository prompting, resulting in significantly lagging performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of hierarchical decomposition and dynamic delegation is novel; the CodeWikiBench methodology is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 languages and 7 repositories with cross-language and scalability analysis, though the human evaluation scale is small.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with well-organized research questions.
- Value: ⭐⭐⭐⭐ Fills a critical gap in automated repository-level documentation generation and evaluation; open-sourcing has a positive community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)
- [\[ACL 2026\] AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor](automonitor-bench_evaluating_the_reliability_of_llm-based_misbehavior_monitor.md)
- [\[ACL 2026\] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)

</div>

<!-- RELATED:END -->
