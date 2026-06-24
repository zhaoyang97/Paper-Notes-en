---
title: >-
  [Paper Note] DocAgent: A Multi-Agent System for Automated Code Documentation Generation
description: >-
  [ACL 2025][Multi-Agent][Code documentation generation] Proposes DocAgent, an automated code documentation generation system based on topological dependency sorting. Through a collaborative Reader-Searcher-Writer-Verifier workflow, it incrementally constructs context, significantly outperforming FIM and Chat baselines across completeness, helpfulness, and truthfulness.
tags:
  - "ACL 2025"
  - "Multi-Agent"
  - "Code documentation generation"
  - "multi-agent system"
  - "topological sorting"
  - "LLM-as-judge"
  - "code understanding"
date: 2026-05-08
content_hash: 49f6a04a389e24c6
---

# DocAgent: A Multi-Agent System for Automated Code Documentation Generation

**Conference**: ACL 2025  
**arXiv**: [2504.08725](https://arxiv.org/abs/2504.08725)  
**Code**: [Yes](https://github.com/facebookresearch/DocAgent)  
**Area**: Other  
**Keywords**: Code documentation generation, multi-agent system, topological sorting, LLM-as-judge, code understanding

## TL;DR

Proposes DocAgent, an automated code documentation generation system based on topological dependency sorting. Through a collaborative Reader-Searcher-Writer-Verifier workflow, it incrementally constructs context, significantly outperforming FIM and Chat baselines across completeness, helpfulness, and truthfulness.

## Background & Motivation

High-quality code documentation is crucial for software development, especially in the AI era, where accurate docstrings are increasingly essential for code understanding tasks. However, existing LLM methods (such as FIM predictors and Chat) suffer from three core issues when automatically generating documentation:

**Incompleteness**: Missing necessary information such as descriptions of parameters and return values.

**Unhelpfulness**: Merely repeating code elements without presenting design motivations and usage scenarios.

**Hallucination**: Fabricating non-existent components, especially in large or private repositories.

The root causes of these problems lie in:
- The difficulty of precisely locating relevant context in large codebases.
- Dependency chains easily exceeding the LLM context window.
- The lack of a robust automated evaluation framework (as BLEU/ROUGE is inapplicable, and manual evaluation is expensive).

The authors analyzed 164 high-star Python repositories and discovered that only 27.28% of documentable nodes contain documentation, and 62.25% of the repositories average less than 30 words per documentation block.

## Method

### Overall Architecture

DocAgent consists of two phases: the Navigator module determines the dependency-aware processing order, and the Multi-Agent system incrementally generates documentation.

### Key Designs

1. **Navigator (Dependency Graph + Topological Sorting)**:

    - Performs AST static analysis on the entire repository to identify functions, methods, classes, and their dependency relationships (calls, inheritance, attribute access, and imports).
    - Constructs a directed dependency graph, and uses Tarjan's algorithm to detect and compress cycles into supernodes, yielding a DAG.
    - Topological sorting ensures a "dependency-first" order: a component is processed only after all of its one-hop dependencies have completed documentation generation.
    - Core advantage: Each component only requires one-hop dependency information, eliminating the need to pull infinitely growing background chains.

2. **Reader Agent (Information Needs Analysis)**:

    - Analyzes the complexity, visibility, and implementation details of the target component.
    - Decides whether additional context is needed, and what context is required.
    - Outputs structured XML information requests: internal dependency code + external knowledge (algorithms, third-party libraries).
    - Interacts with the Searcher over multiple rounds to iteratively supplement context.

3. **Searcher Agent (Information Retrieval)**:

    - Internal code analysis tool: Utilizes static analysis to retrieve internal component source code, call sites, and class hierarchies.
    - External knowledge retrieval tool: Obtains domain knowledge (e.g., DPO algorithm principles) via general search APIs.
    - Integrates retrieval results into a structured context for subsequent agents to use.

4. **Writer Agent (Documentation Generation)**:

    - Generates documentation using different templates based on the component type (function/method/class).
    - For functions: Summary, description, Args, Returns, Raises, Examples.
    - For classes: Summary, description, initialization examples, constructor arguments, public attributes.

5. **Verifier Agent (Quality Verification)**:

    - Evaluates the informativeness, level of detail, and completeness of the documentation.
    - Directly provides feedback to the Writer to revise formatting issues.
    - Requests more context from the Reader when information is insufficient, triggering a new interaction loop.

6. **Orchestrator (Workflow Management)**:

    - Manages the Reader $\rightarrow$ Searcher $\rightarrow$ Writer $\rightarrow$ Verifier iterative workflow.
    - Adaptive context truncation: Monitors total token counts and selectively removes the largest paragraphs to control length.

### Evaluation Framework (Three Dimensions)

- **Completeness**: Automatically checks the structural completeness of documentation based on AST analysis and regular expressions (scoring $0$-$1$).
- **Helpfulness**: Decomposed evaluation and LLM-as-judge using a 5-point Likert scale with scoring rubrics and examples.
- **Truthfulness**: Extracts code entities mentioned in the documentation, cross-validates them against the dependency graph, and calculates the Existence Ratio.

## Key Experimental Results

### Main Results — Completeness (Table)

| System | Overall | Function | Method | Class |
|------|---------|----------|--------|-------|
| DA-GPT | 0.934 | 0.945 | 0.935 | 0.914 |
| DA-CL | 0.953 | 0.985 | 0.982 | 0.816 |
| Chat-GPT | 0.815 | 0.828 | 0.823 | 0.773 |
| Chat-CL | 0.724 | 0.726 | 0.744 | 0.667 |
| FIM-CL | 0.314 | 0.291 | 0.345 | 0.277 |

### Main Results — Truthfulness (Table)

| System | Extracted | Verified | Existence Ratio |
|------|-----------|----------|-----------------|
| DA-GPT | 305 | 265 | **95.74%** |
| DA-CL | 600 | 354 | 88.17% |
| Chat-GPT | 347 | 366 | 61.10% |
| Chat-CL | 488 | 366 | 68.03% |
| FIM-CL | 131 | 338 | 45.04% |

### Ablation Study (Table)

| System | Overall Helpfulness | Summary | Description | Parameters |
|------|---------------------|---------|-------------|------------|
| DA-GPT | 3.88 | 4.32 | 3.60 | 2.71 |
| DA-Rand-GPT | 3.44(-0.44) | 3.62(-0.70) | 3.30(-0.30) | 2.20(-0.51) |
| DA-CL | 2.35 | 2.36 | 2.43 | 2.00 |
| DA-Rand-CL | 2.18(-0.17) | 1.88(-0.48) | 2.42(-0.10) | 2.00(0.00) |

Removing topological sorting significantly degrades both helpfulness and truthfulness (GPT-4o mini's Existence Ratio decreases from 94.64% to 86.75%).

### Key Findings

- DocAgent (GPT-4o mini) consistently leads across all three dimensions with 0.934 Completeness, 3.88 Helpfulness, and 95.74% Truthfulness.
- FIM performs the worst (Completeness is merely 0.314, Truthfulness 45.04%), indicating that fill-in-the-middle approaches are not suitable for documentation generation.
- Parameter description is the most challenging task for all systems (Chat baselines scored below 2.2).
- Topological sorting benefits Summary generation the most (gain of 0.48-0.70).

## Highlights & Insights

- **Topological Sorting + Incremental Context Construction** is the core innovation: It solves long dependency issues through processing sequence design rather than brute-force extension of context windows.
- Multi-Agent role design simulates human team collaboration; the multi-round interaction of Reader-Searcher ensures contextual adequacy.
- The three-dimensional evaluation framework (Completeness/Helpfulness/Truthfulness) is more comprehensive than traditional BLEU/ROUGE, and can be reused for other code generation tasks.
- The design of external knowledge retrieval is practical: New algorithms prior to LLM knowledge cutoff dates require documentation from the internet.

## Limitations & Future Work

- Extremely large codebases may still exceed LLM context window limits.
- Reliant only on static analysis, lacking dynamic behaviors understanding.
- Currently supports only Python; adapting to other languages requires additional effort.
- The computational cost and environmental impact of LLM multi-agent systems cannot be ignored.
- Generated documentation may still exhibit hallucinations and requires manual review.

## Related Work & Insights

- Introduces topological processing order based on past multi-agent code toolchains (MapCoder, ChatDev, AutoGen).
- The robust design of LLM-as-judge in the evaluation framework (decomposed evaluation + structured prompt + few-shot calibration) is worth learning from.
- Documentation coverage statistics on 164 repositories reveal the true plight of code documentation.

## Rating

- **Novelty**: 7/10 — The combination of topological sorting + Multi-Agent is creative, but individual components are not fundamentally new.
- **Experimental Thoroughness**: 8/10 — The three-dimensional evaluation + ablation validation are thorough, but the dataset size (366 components) is on the smaller side.
- **Writing Quality**: 8/10 — Clear structure, and the evaluation framework is described in detail.
- **Value**: 8/10 — Code documentation generation is a high-value application scenario; the framework has practical deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](../../ACL2026/multi_agent/memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ICLR 2026\] Aegis: Automated Error Generation and Attribution for Multi-Agent Systems](../../ICLR2026/multi_agent/aegis_automated_error_generation_and_attribution_for_multi-agent_systems.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](../../AAAI2026/multi_agent/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)
- [\[NeurIPS 2025\] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](../../NeurIPS2025/multi_agent/masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](../../ACL2026/multi_agent/roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)

</div>

<!-- RELATED:END -->
