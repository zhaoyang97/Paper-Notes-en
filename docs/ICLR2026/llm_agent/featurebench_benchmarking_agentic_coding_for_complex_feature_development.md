---
title: >-
  [Paper Note] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development
description: >-
  [ICLR 2026][LLM Agent][agentic coding] This paper introduces FeatureBench—a benchmark for feature-level software development targeting code agents, comprising 200 tasks across 24 open-source repositories, with each task requiring an average of 790 lines of code spanning 15.7 files. Even Claude Opus 4.5 (74.4% on SWE-bench) resolves only 11.0% of tasks, revealing a substantial capability gap in realistic feature development scenarios.
tags:
  - ICLR 2026
  - LLM Agent
  - agentic coding
  - software development
  - feature development
  - benchmark
  - code agent
date: 2026-05-08
content_hash: 175f4946b6d3cb04
---

# FeatureBench: Benchmarking Agentic Coding for Complex Feature Development

**Conference**: ICLR 2026
**arXiv**: [2602.10975](https://arxiv.org/abs/2602.10975)
**Code**: [https://github.com/LiberCoders/FeatureBench](https://github.com/LiberCoders/FeatureBench)
**Area**: Agent
**Keywords**: agentic coding, software development, feature development, benchmark, code agent

## TL;DR
This paper introduces FeatureBench—a benchmark for feature-level software development targeting code agents, comprising 200 tasks across 24 open-source repositories, with each task requiring an average of 790 lines of code spanning 15.7 files. Even Claude Opus 4.5 (74.4% on SWE-bench) resolves only 11.0% of tasks, revealing a substantial capability gap in realistic feature development scenarios.

## Background & Motivation

**State of the Field**: LLM-driven coding agents (Claude Code, Qwen Code, etc.) are transitioning from assistive tools to autonomous developers. SWE-bench is the most widely used evaluation benchmark, but resolution rates have rapidly risen from <10% to >70%, leaving it with insufficient discriminative power.

**Limitations of Prior Work**: (a) SWE-bench focuses predominantly on single-PR bug fixes (78–82% of tasks), averaging only 32.8 lines of code across 1.7 files—far from representative of real-world feature development complexity; (b) feature development typically spans multiple PRs distributed across a project timeline, which PR-based collection methods cannot fully capture; (c) many benchmarks rely on manual annotation or cannot be automatically scaled and updated.

**Root Cause**: Agents are approaching saturation on bug fixing, yet the core of real-world software development is feature implementation—requiring global architectural understanding, cross-file dependency management, and substantial code generation. This capability is entirely unaddressed by existing benchmarks.

**Paper Goals**: (a) Construct a benchmark targeting feature-level development; (b) provide an automated, scalable task collection toolkit; (c) ensure execution-based evaluation (rather than LLM-as-judge).

**Starting Point**: Reverse-tracing from unit tests—by executing F2P tests and dynamically tracking the dependency graph, the framework automatically identifies feature-related code and extracts it from the codebase as the portion to be implemented.

**Core Idea**: Test-driven feature extraction—starting from unit tests, the runtime dependency graph is traced to automatically isolate complete feature implementations from the repository as tasks for the agent to complete.

## Method

### Overall Architecture
A four-stage automated pipeline: (1) Environment setup—Dockerized repositories, requiring only manual specification of installation commands (~3 minutes per repository); (2) Test construction—validating and selecting F2P and P2P tests; (3) Dependency graph tracing—dynamically tracing and constructing an object dependency graph with BFS traversal for node classification; (4) Code extraction and problem generation—extracting feature implementations from the codebase and automatically generating problem descriptions with interface definitions.

### Key Designs

1. **Test-Driven Code Patch Extraction**:

    - Function: Automatically isolates feature-related code from the complete repository.
    - Mechanism: F2P and P2P tests are executed; Python `trace` captures function call events and dependencies to construct an object dependency graph. An LLM analyzes functions imported by F2P tests to distinguish target feature functions from auxiliary test functions. BFS traversal begins from the feature entry point: nodes appearing in P2P tests are labeled "remained" (depended upon by other functionality, cannot be removed), while those absent are labeled "extracted" (feature code to be implemented).
    - Design Motivation: The inclusion of P2P tests ensures that extracting feature code does not break existing functionality—a critical engineering constraint.

2. **Automated Problem Description Generation**:

    - Function: Generates explicit interface definitions and functional descriptions.
    - Mechanism: Function signatures, input/output types, and import paths are extracted from the extracted code snippets. Existing docstrings are used directly; otherwise an LLM generates them from the code. It explicitly specifies that "the solution must be directly callable."
    - Design Motivation: Eliminating implementation ambiguity—explicit interface definitions ensure that correct implementations pass all tests, enabling automated execution-based evaluation.

3. **Two-Level Difficulty Design**:

    - L1 (Incremental Development): Extending and implementing new features on top of the existing codebase.
    - L2 (From-Scratch Development): Implementing the same functionality entirely from scratch.
    - Design Motivation: Reflects two common scenarios encountered in real-world software development.

4. **Post-Validation Pipeline**:

    - The modified codebase must pass all P2P tests (existing functionality intact) and fail all F2P tests (target feature removed).
    - All tests must pass after restoring the patch.
    - All utility functions required by F2P tests must remain accessible in the modified codebase.

### Loss & Training
FeatureBench is an evaluation benchmark rather than a training methodology. Evaluation metrics:
- **Resolved Rate**: Proportion of tasks completely solved.
- **Passed Rate**: Average proportion of F2P tests passed (soft metric).
- **Token I/O**: Input/output token consumption (efficiency metric).

## Key Experimental Results

### Main Results

| Scaffold + Model | Lite Resolved | Full Resolved | Token I/O |
|------------------|--------------|---------------|-----------|
| OpenHands + Qwen3-Coder-480B | 6.7% | 3.5% | 2.0M/14k |
| OpenHands + DeepSeek-V3.2 | 6.7% | 5.5% | 3.1M/23k |
| Gemini-CLI + Gemini-3-Pro | 10.0% | 5.0% | 2.5M/12k |
| Claude Code + Claude Opus 4.5 | 20.0% | 11.0% | 7.5M/34k |
| **Codex + GPT-5.1-Codex** | **20.0%** | **12.5%** | 6.3M/39k |
| OpenHands + Claude Opus 4.5 | 20.0% | 10.5% | 8.1M/29k |

### Comparison with SWE-bench

| Dimension | SWE-bench | FeatureBench |
|-----------|-----------|-------------|
| Problem description length | 195 words | **4,818 words** |
| Gold solution lines | 32.8 | **790.2** |
| Gold solution files | 1.7 | **15.7** |
| Number of functions | 3 | **29.2** |
| Test cases | 6.3 | **62.7** |

### Key Findings
- **Claude Opus 4.5 achieves 74.4% on SWE-bench yet only 11.0% on FeatureBench**, indicating that feature-level development is an order of magnitude harder than bug fixing.
- **Passed Rate substantially exceeds Resolved Rate** (e.g., 45.53% vs. 11.0% for Claude), suggesting that agents produce code that appears plausible but is functionally incomplete, requiring extensive debugging.
- **Token consumption is enormous**: all models consume over one million input tokens, yet resolution rates remain low—efficiency is a critical bottleneck.
- **Common failure modes**: `NameError` (cross-file dependencies not correctly imported) and `AttributeError` (missing prototype methods)—both manifestations of insufficient cross-file comprehension.
- **L2 (from-scratch development) is substantially harder than L1 (incremental development)**, suggesting that agents rely heavily on contextual cues from existing code.

## Highlights & Insights
- **Test-driven feature extraction**: Tracing dependency graphs backward from tests to isolate feature code is an elegant engineering approach—simultaneously enabling automation and ensuring codebase integrity after extraction.
- **Sustainable updating**: Built on an automated toolkit, FeatureBench can continuously generate tasks from new repositories, mitigating data leakage concerns.
- **Revealing the true capability boundary of agents**: Saturation on SWE-bench does not imply near-human performance—feature development is the genuine litmus test.
- **3,825 executable environments available for training**: Beyond serving as an evaluation benchmark, this constitutes a high-quality data source for agent reinforcement learning training.

## Limitations & Future Work
- Limited to Python repositories; software development in other languages is not covered.
- The scale of 200 tasks, while larger than manually curated benchmarks, remains smaller than SWE-bench's 500 instances.
- L2 from-scratch evaluation may be overly strict—in practice, complete from-scratch implementations are rare.
- The benchmark does not assess agents' iterative debugging capabilities—only final outcomes are measured, not intermediate processes.

## Related Work & Insights
- **vs. SWE-bench**: SWE-bench is dominated by bug fixes (78–82%), averaging 32 lines; FeatureBench focuses on feature development averaging 790 lines—an order of magnitude harder.
- **vs. PaperBench/MLE-bench**: These benchmarks target the ML domain, are manually curated, and cannot be automatically scaled; FeatureBench covers general Python development and is fully automated.
- **vs. SWE-Smith/SWE-Flow**: These generate synthetic tasks but do not guarantee feature-level complexity or existing functionality integrity; FeatureBench's P2P test constraints ensure fidelity to realistic development scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fills the gap in feature-level programming benchmarks; the test-driven feature extraction methodology is both novel and practically useful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven scaffold–model configurations, 24 real-world repositories, and in-depth comparison with SWE-bench.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ Reveals the true capability gap in agentic coding and provides the community with a more discriminative evaluation tool.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](livenewsbench_evaluating_llm_web_search_capabilities_with_fresh_news.md)
- [\[ICLR 2026\] Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)

<!-- RELATED:END -->
