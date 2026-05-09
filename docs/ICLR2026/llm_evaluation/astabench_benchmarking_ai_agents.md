---
title: >-
  [Paper Note] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite
description: >-
  [ICLR 2026 Oral][LLM Evaluation][Agent Benchmark] The AI2 team identifies five methodological flaws in existing scientific research agent benchmarks and introduces AstaBench, the first agent evaluation suite covering the full scientific research pipeline. AstaBench comprises 4 categories and 11 sub-benchmarks with 2,400+ questions, a production-grade controllable search tool backed by Semantic Scholar, and 9 research-optimized Asta Agent baselines. It conducts the largest systematic evaluation to date across 57 agents (22 types), finding that despite progress on individual tasks such as literature retrieval, AI remains far from meeting the demands of end-to-end scientific research assistance.
tags:
  - ICLR 2026 Oral
  - LLM Evaluation
  - Agent Benchmark
  - Scientific Research Automation
  - Reproducible Evaluation
  - AI for Science
date: 2026-05-08
content_hash: da3f222a800604dd
---

# AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite

**Conference**: ICLR 2026 Oral
**arXiv**: [2510.21652](https://arxiv.org/abs/2510.21652)
**Code**: [allenai/asta-bench](https://github.com/allenai/asta-bench)
**Area**: LLM Evaluation
**Keywords**: Agent Benchmark, Scientific Research Automation, Reproducible Evaluation, AI for Science

## TL;DR

The AI2 team identifies five methodological flaws in existing scientific research agent benchmarks and introduces AstaBench, the first agent evaluation suite covering the full scientific research pipeline. AstaBench comprises 4 categories and 11 sub-benchmarks with 2,400+ questions, a production-grade controllable search tool backed by Semantic Scholar, and 9 research-optimized Asta Agent baselines. It conducts the largest systematic evaluation to date across 57 agents (22 types), finding that despite progress on individual tasks such as literature retrieval, AI remains far from meeting the demands of end-to-end scientific research assistance.

## Background & Motivation

**Background**: AI agents have demonstrated significant potential for automating scientific research—covering automated literature reviews, experiment reproduction, data analysis, and even proposing new research directions. Numerous systems have emerged, including general-purpose agents such as Google/OpenAI Deep Research and domain-specific systems such as AI Scientist and AIGS. However, rigorous evaluation of these agents is a prerequisite for genuine progress.

**Limitations of Prior Work**: The authors systematically identify five methodological flaws in existing benchmarks. First, **lack of full-pipeline coverage**: most benchmarks evaluate only a single sub-task (e.g., QA or retrieval), failing to reflect the comprehensive demands placed on agents in real research scenarios. Second, **non-reproducible tools**: different agents rely on different search engines and tool chains, meaning evaluations fundamentally compare tool differences rather than agent capabilities. Third, **uncontrolled confounding variables**: model cost, API call counts, and tool access permissions are not standardized, making it impossible to disentangle model strength from tool strength. Fourth, **absence of standardized interfaces**: no unified agent construction and evaluation framework exists, hampering rapid prototyping and fair comparison. Fifth, **insufficient baselines**: the lack of diverse and numerous baseline agents makes it difficult for the community to assess how real any reported progress is.

**Key Challenge**: Evaluating scientific research agents requires measuring both point capabilities (e.g., retrieval, coding) and full-pipeline capabilities (from literature survey to end-to-end scientific discovery). The latter is far more complex to evaluate and requires a controlled tool environment to eliminate confounding factors.

**Key Insight**: The authors' team, backed by the deployed Semantic Scholar/Asta system (108M+ abstracts, 12M+ full-text papers), holds a unique advantage: (1) a production-grade literature search API available as a controlled tool, and (2) a deployed Asta Agent that has accumulated large volumes of real user requests, enabling the construction of question sets closely aligned with actual research needs.

**Core Idea**: Systematically address the five methodological flaws in scientific research agent evaluation at the methodological level, constructing a standardized agent evaluation platform with full-pipeline coverage, controlled tools, and sufficient baselines.

## Method

### Overall Architecture

AstaBench rests on three pillars: (1) a 2,400+ question set spanning the full scientific research lifecycle (4 categories, 11 sub-benchmarks); (2) a production-grade reproducible search tool environment based on Semantic Scholar (Asta MCP tools); and (3) 9 research-optimized Asta Agent baselines together with a standardized evaluation protocol (including cost metrics and leaderboard submission). The entire framework is built on the InspectAI evaluation infrastructure, supporting Docker-sandboxed code execution and MCP tool invocation.

### Key Designs

1. **Question Set Architecture: 4 Categories and 11 Sub-benchmarks**

    AstaBench decomposes research capability into four progressive levels, each comprising multiple independent sub-benchmarks. The Literature category contains 5 sub-benchmarks: PaperFindingBench (paper retrieval), ScholarQABench2 (scientific QA), LitQA2-FT and its search variant (literature QA), and ArxivDIGESTables-Clean (structured summary table generation). The Code category contains 3 sub-benchmarks: CORE-Bench-Hard (repository-level code questions), DS-1000 (data science programming), and SUPER-Expert (complex programming tasks with trajectory tracking). The Data Analysis category contains DiscoveryBench (data-driven scientific discovery). The End-to-End Discovery category contains E2E-Bench and its hard variant (complete scientific discovery workflows).

    A notable design feature is that many questions are drawn directly from real user requests received by the deployed Asta Agent, ensuring that the question distribution reflects actual research needs rather than academic conjecture. Questions span multiple domains including computer science and biomedicine.

    **Design Motivation**: Existing benchmarks cover only a single link in the research chain (e.g., only retrieval or only coding), and cannot reflect an agent's true performance across the complete research pipeline.

2. **Controllable Tool Environment Based on Asta MCP**

    All literature tasks use a unified Asta MCP tool interface (backed by the Semantic Scholar API covering 108M+ abstracts and 12M+ full-text papers), with date and corpus restrictions to ensure evaluation validity. Coding tasks are provided with a standardized Docker sandbox environment containing a stateful Python session (similar to a Jupyter notebook) to ensure reproducible code execution.

    Tools are classified into three categories: Standard (only pre-defined evaluation environment tools), Custom Interface (custom but capability-equivalent or restricted tools), and Custom (tools beyond standard constraints). Each leaderboard submission is annotated with its tool category.

    **Design Motivation**: This eliminates confounding factors such as "an agent scores higher because it uses a better search engine." A continuously maintained API (rather than a one-time crawl) guarantees long-term reproducibility.

3. **9 Asta Agent Baselines and Standardized Evaluation Protocol**

    Nine research-optimized Asta Agent architectures are provided (including ReAct, code-execution-based, context-compression-based, and others), forming a complete baseline spectrum from simple to complex, all open-sourced for community comparison. Agents are classified into four openness levels: open-source + open weights, open-source + closed weights, closed-source + API, and closed-source + UI only.

    The evaluation protocol standardizes cost metrics: model call counts, token consumption, and API expenditure are recorded for each agent, with performance–cost trade-offs displayed on the leaderboard. Validation and test splits are supported; the validation set is used for development and the test set for final evaluation.

    **Design Motivation**: Insufficient baselines create the illusion that any new method represents progress. The 9 baselines and large-scale comparison across 57 agents provide the community with a reliable performance reference.

### Evaluation Metric Framework

Each sub-benchmark uses metrics designed for its task characteristics: literature retrieval tasks use retrieval precision and recall; QA tasks use answer correctness (combining automated evaluation and LLM-based judgment); coding tasks use execution pass rate and result match; end-to-end discovery tasks use multi-dimensional research report quality assessment. The scorer code version for all metrics can be independently locked (supporting solve–score decoupling) to ensure scoring consistency across versions.

## Key Experimental Results

### Main Results: AstaBench Task Composition

| Category | Sub-benchmark | Capability Assessed | Scale |
|----------|--------------|---------------------|-------|
| Literature (lit) | PaperFindingBench | Paper retrieval | Hundreds |
| Literature (lit) | ScholarQABench2 | Scientific literature QA | Hundreds |
| Literature (lit) | LitQA2-FT / FT-Search | Literature QA + search | Hundreds |
| Literature (lit) | ArxivDIGESTables-Clean | Structured table generation | Hundreds |
| Code (code) | CORE-Bench-Hard | Repository-level code questions | Hundreds |
| Code (code) | DS-1000 | Data science programming | Hundreds |
| Code (code) | SUPER-Expert | Complex programming + trajectory tracking | Hundreds |
| Data Analysis (data) | DiscoveryBench | Data-driven discovery | Hundreds |
| Discovery (discovery) | E2E-Bench / E2E-Hard | End-to-end scientific discovery | Hundreds |
| **Total** | **11 sub-benchmarks** | **Full-pipeline coverage** | **2,400+** |

### Methodological Flaw Remediation Comparison

| Evaluation Dimension | Problem in Prior Benchmarks | AstaBench Solution |
|---------------------|----------------------------|-------------------|
| Metric comprehensiveness | Only a single sub-task evaluated (e.g., retrieval or coding only); fragmented coverage | 4 categories and 11 sub-benchmarks covering the full pipeline from literature retrieval to end-to-end scientific discovery |
| Tool reproducibility | Agents bring their own search tools; large performance gaps across tools | Unified Asta MCP tools (Semantic Scholar API), continuously maintained for long-term reproducibility |
| Confounding variable control | Model cost and tool permissions not standardized; unfair comparison | Token/API costs recorded; tool category annotated (Standard / Custom Interface / Custom) |
| Standardized interface | No common agent construction framework; each system self-contained | Unified solver interface based on InspectAI + ToolsetConfig tool management |
| Baseline sufficiency | Insufficient variety and number of baseline agents | 9 Asta Agent baselines + comprehensive comparison across 57 agents / 22 types |
| Openness classification | No distinction among open-source / closed-source / API-only model characteristics | 4-level openness classification + 3-level toolset classification; leaderboard filterable by category |

### Key Findings

- **Progress on individual tasks but large gaps in full-pipeline performance**: AI agents achieve relatively good performance on individual tasks such as literature retrieval and simple QA, but fall significantly short of human researchers on end-to-end scientific discovery tasks (E2E-Bench) requiring multi-step reasoning and cross-modal collaboration. This indicates that excelling at sub-tasks does not equate to excelling at research.
- **Tool differences are the primary confounding factor**: When tool variables are controlled (all agents using the same Asta MCP tools), performance differences among agents primarily stem from reasoning strategies and context management rather than tool quality. This validates the necessity of a controlled tool environment.
- **Large variation in cost–performance trade-offs**: Agents of similar performance levels can differ by several-fold in token consumption, indicating that accuracy alone is insufficient for agent evaluation—cost efficiency is an important dimension.
- **Inconsistent capabilities across tasks**: Agents proficient in literature retrieval are not necessarily proficient in code generation, and agents strong in data analysis may perform poorly on end-to-end discovery tasks. Scientific research agents require more balanced multi-dimensional capabilities.

## Highlights & Insights

- **Methodological contribution over technical contribution**: The core value of AstaBench is not simply "yet another benchmark," but rather the systematic definition at the methodological level of how to correctly evaluate scientific research agents. The five evaluation principles (comprehensiveness, tool controllability, confounding control, standardized interface, and sufficient baselines) are generalizable to other agent evaluation contexts.
- **Question set driven by real user needs**: Many questions are drawn directly from user requests received by the production-deployed Asta Agent, ensuring that the benchmark measures capabilities users actually need rather than capabilities researchers assume are important. This product-informed benchmark design philosophy is worth emulating.
- **Elegant tool environment design**: Using the continuously maintained Semantic Scholar API (rather than a one-time data snapshot) as the search backend, combined with date/corpus restrictions to ensure evaluation validity, balances reproducibility and realism. The date restriction on search tools is particularly elegant—even as new papers are published, evaluation results for existing questions remain unaffected.
- **Cost-visible evaluation paradigm**: Displaying performance–cost trade-offs on the leaderboard rather than accuracy rankings alone compels agent developers to consider efficiency as well as performance—a more expensive agent is only worthwhile if its performance is substantially better.

## Limitations & Future Work

- **Disciplinary bias toward CS**: Although the benchmark claims multi-disciplinary coverage, reliance on Semantic Scholar as the literature backend may result in insufficient coverage of fields beyond computer science and biomedicine (e.g., physics, chemistry, social sciences). Tool support for domains requiring patent databases or clinical trial databases is especially lacking.
- **Difficulty evaluating experimental capabilities**: The current benchmark is weighted toward information retrieval and textual reasoning, and is largely unable to evaluate "hands-on" research skills such as experimental design, execution, and instrument operation. Although E2E-Bench tests end-to-end discovery, it remains confined to computational experiments.
- **Difficulty measuring creativity**: The core of scientific research lies in proposing novel hypotheses and discovering unexpected relationships, yet such creative dimensions are extremely difficult to measure with automated metrics. Current evaluation still primarily assesses whether a predefined answer is reached, potentially undervaluing agents' creative outputs.
- **High hardware requirements**: Running the complete evaluation suite requires 128 GB+ of memory and a powerful CPU (the authors used a 128 GB / 8-core machine with $N=8$ parallelism), and the API costs for a single full evaluation are considerable, limiting participation by smaller teams.

## Related Work & Insights

- **vs. AI Scientist / AIGS**: These are the objects of evaluation (scientific research agent systems); AstaBench provides a standardized platform for fairly evaluating them. AI Scientist focuses on "writing papers," whereas AstaBench has broader coverage.
- **vs. SWE-bench / HumanEval / DS-1000**: These code generation and repair benchmarks are integrated into AstaBench as sub-tasks (the Code category includes DS-1000, CORE-Bench, etc.), but AstaBench covers a broader scientific research task chain.
- **vs. GAIA / AgentBench**: General-purpose agent evaluation benchmarks test general tool-use capabilities, whereas AstaBench makes deep customizations for the scientific research vertical domain (built-in literature search tools, research-specific evaluation metrics).
- **vs. Deep Research systems**: General research agents can be directly evaluated on AstaBench and fairly compared with research-specific agents—a major advantage of AstaBench.

## Rating

- Novelty: ⭐⭐⭐⭐ The first benchmark to systematically address methodological flaws in scientific research agent evaluation; outstanding methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 57 agents / 22 types / 11 sub-benchmarks / 2,400+ questions; unprecedented in scale.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; the five-flaw framework is logically structured; contributions are presented systematically.
- Value: ⭐⭐⭐⭐⭐ Establishes standardized evaluation infrastructure for AI-for-Science agent research; impact is already emerging.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](../../ACL2026/llm_evaluation/researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[NeurIPS 2025\] Benchmarking is Broken — Don't Let AI be its Own Judge](../../NeurIPS2025/llm_evaluation/benchmarking_is_broken_--_dont_let_ai_be_its_own_judge.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[AAAI 2026\] Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](../../AAAI2026/llm_evaluation/towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)

<!-- RELATED:END -->
