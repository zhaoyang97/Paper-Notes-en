---
title: >-
  [Paper Note] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development
description: >-
  [ICLR 2026][LLM Agent][agentic coding] This paper introduces FeatureBench, a benchmark for evaluating code agents on feature-level software development. Through a test-driven automated pipeline…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "agentic coding"
  - "benchmark"
  - "feature development"
  - "test-driven"
  - "SWE-bench"
  - "code agent"
date: 2026-05-08
content_hash: 7f4ac60758ea273e
---

# FeatureBench: Benchmarking Agentic Coding for Complex Feature Development

**Conference**: ICLR 2026
**arXiv**: [2602.10975](https://arxiv.org/abs/2602.10975)
**Code**: [github.com/LiberCoders/FeatureBench](https://github.com/LiberCoders/FeatureBench)
**Area**: LLM Agent
**Keywords**: agentic coding, benchmark, feature development, test-driven, SWE-bench, code agent

## TL;DR
This paper introduces FeatureBench, a benchmark for evaluating code agents on feature-level software development. Through a test-driven automated pipeline, it extracts verifiable feature implementation tasks from open-source repositories. The strongest model, Claude Opus 4.5, resolves only 11.0% of tasks, revealing a substantial gap between current agents and the demands of complex feature development.

## Background & Motivation
Existing code agent benchmarks (e.g., SWE-bench) primarily focus on bug-fixing scenarios, with feature request tasks comprising only 18–22% of instances. As end-to-end coding systems such as Claude Code and Qwen Code emerge, evaluating agents on **feature development**—rather than merely bug fixing—has become critically important.

The PR-based methodology of SWE-bench has a structural limitation: a complete feature often spans multiple PRs distributed across a project's timeline, making single-PR granularity insufficient for capturing complete functional patches. Moreover, many PRs lack labels, making it difficult to reliably distinguish feature contributions from bug fixes.

Existing benchmarks also suffer from evaluation limitations. PaperBench and Paper2Coder rely on human review or LLM judgment, lacking automated execution-level verification. Manually curated benchmarks (e.g., GitTaskBench with 54 instances, DevEval with 22) are too small to comprehensively assess agent capabilities.

Data contamination is a growing concern—as training data coverage expands, static benchmarks risk being memorized, necessitating a continuously updatable evaluation framework.

The authors argue that an ideal benchmark must simultaneously satisfy four conditions: (1) targeting real feature-level development; (2) enabling executable verification; (3) supporting automated and scalable collection; and (4) being continuously updatable to mitigate data leakage. No prior work satisfies all four criteria.

## Method

### Overall Architecture
FeatureBench comprises two core components: (1) a **feature-oriented task definition** that requires agents to implement directly callable functional modules given high-level descriptions and interface signatures; and (2) a **test-driven automated collection toolkit** that extracts feature-level tasks from open-source repositories and generates verifiable environments with F2P/P2P tests.

### Task Design

Each instance provides: a high-level task description, function signatures (with input/output types), import paths, a list of prohibited URLs, and a Dockerfile execution environment. Agents must produce **directly callable** solutions.

Two difficulty levels are defined:
- $L_1$ (incremental development): adding new functionality to an existing codebase
- $L_2$ (from-scratch construction): implementing equivalent functionality entirely from scratch

Evaluation metrics:
$$\text{Resolved Rate} = \frac{\#\text{tasks fully solved}}{\#\text{total tasks}}$$
$$\text{Passed Rate} = \frac{1}{N}\sum_{i=1}^{N}\frac{\#\text{F2P tests passed}_i}{\#\text{F2P tests total}_i}$$

### Key Design: Test-Driven Code Patch Extraction

1. **Dependency graph construction**: During F2P and P2P test execution, function call events are captured via Python's built-in tracing facilities to construct an object dependency graph (nodes = functions, with P2P labels).

2. **LLM classification of top-level objects**: An LLM distinguishes directly tested top-level objects from auxiliary utility functions in test files, achieving F1 of 84.94% and accuracy of 91.74%.

3. **BFS traversal and node classification**: Starting from top-level objects, breadth-first search is performed; nodes appearing in P2P tests are labeled **remained**, while those absent are labeled **extracted**—ensuring existing functionality is not broken.

4. **Code extraction and post-validation**: Extracted nodes are removed from the codebase to produce a "feature-absent" codebase and a corresponding patch. Post-validation requires: (a) the feature-absent codebase passes all P2P tests and fails all F2P tests; (b) after applying the patch, all tests pass.

5. **Automated problem statement generation**: Task descriptions are constructed automatically from code docstrings or LLM-generated summaries, combined with interface signatures.

### Benchmark Configuration
- **Full Set**: 200 high-quality instances, each with >100 lines of implementation code, ≥10 F2P test points, and P2P spanning 5 files
- **Lite Set**: 30 randomly sampled instances for low-cost rapid evaluation
- Sources: 24 open-source PyPI repositories, spanning May 2022–September 2025

## Key Experimental Results

### Main Results: Model Performance on Full Set

| Scaffold | Model | Passed Rate | Resolved Rate | Token I/O |
|----------|-------|-------------|---------------|-----------|
| Codex | GPT-5.1-Codex (medium) | 41.66% | **12.5%** | 6.3M / 39k |
| Claude Code | Claude Opus 4.5 | 43.29% | 11.0% | 7.5M / 34k |
| OpenHands | Claude Opus 4.5 | 45.53% | 10.5% | 8.1M / 29k |
| Gemini-CLI | Gemini-3-Pro (low) | 32.43% | 5.0% | 2.5M / 12k |
| OpenHands | DeepSeek-V3.2 | 26.30% | 5.5% | 3.1M / 23k |
| OpenHands | Qwen3-Coder-480B | 24.55% | 3.5% | 2.0M / 14k |
| OpenHands | Gemini-3-Pro (low) | 30.08% | 4.5% | 6.2M / 40k |

### Comparison with SWE-bench (Shared Repository Subset)

| Model | SWE-bench Verified Resolved | FeatureBench Resolved | FeatureBench Passed |
|-------|---------------------------|----------------------|---------------------|
| Claude Opus 4.5 | **74.40%** | 5.2% | 41.08% |
| Gemini-3-Pro | 74.20% | 0.0% | 30.05% |
| Qwen3-Coder-480B | 55.40% (OpenHands: 69.60%) | 0.0% | 23.46% |
| DeepSeek-V3.2 | 60.00% | 0.0% | 22.98% |

### Task Complexity Comparison

| Attribute | SWE-bench | FeatureBench ($L_1$) |
|-----------|-----------|---------------------|
| Problem description length (words) | 195.1 | **4818.0** |
| Gold solution lines | 32.8 | **790.2** |
| Files involved | 1.7 | **15.7** |
| Functions involved | 3 | **29.2** |
| F2P test points | 9.1 | **62.7** |
| Total test points | 120.8 | **302.0** |

## Key Findings
- **Even the strongest agents resolve only ~11–12.5%**: Claude Opus 4.5 and GPT-5.1-Codex achieve only 11.0% and 12.5% on the Full Set, respectively, while the same models reach 74.4% on SWE-bench—a performance gap of nearly an order of magnitude.
- **Passed Rate far exceeds Resolved Rate** (~45% vs. ~12%): Agents produce code that appears reasonable but fails to pass all tests, reflecting the reality that AI-generated code in practical development requires extensive debugging.
- **Token consumption is substantial**: All models consume over one million input tokens, yielding extremely poor efficiency given the low success rates—agent efficiency emerges as an important research direction.
- **Failure mode analysis**: NameError is the most frequent error, indicating fundamental difficulties in cross-file dependency resolution; TypeError/AttributeError stem from LLMs' "lazy" tendency to guess rather than read actual interface definitions.
- **$L_2$ is significantly harder**: From-scratch construction yields universally lower Resolved Rates with smaller inter-model differences, suggesting that the absence of codebase structure is a common bottleneck for multi-step reasoning.
- **Interface specifications are critical**: Removing function signatures causes substantial performance degradation (GPT-5.1-Codex: 20.0% → 16.7%), while providing ground-truth unit tests can boost success rates to 60%+.
- **Diminishing returns with more steps**: Increasing from 50 to 100 steps yields noticeable gains, but gains from 100 to 500 steps are marginal.

## Highlights & Insights
- **Fills an evaluation gap**: FeatureBench is the first coding benchmark to simultaneously satisfy all four criteria—feature-oriented, execution-based, scalable, and continuously updatable—addressing the bug-fixing bias of SWE-bench.
- **The test-driven automated collection pipeline** is elegantly designed: feature isolation is achieved via BFS traversal of the dependency graph with P2P protection, requiring no manual annotation of feature boundaries; human effort per repository is approximately 3 minutes.
- **Reveals structural deficiencies in agents**: The bottleneck is not model scale but architectural capabilities—cross-file reasoning, long-horizon planning, and efficient context utilization—providing direct guidance for next-generation agent architecture design.
- **Lite Set and Full Set rankings are highly consistent**, validating the representativeness of small-scale rapid evaluation.

## Limitations & Future Work
- Coverage is limited to **Python** repositories; evaluation of code agents for Java, C++, Rust, and other languages is absent.
- The 24 repositories are concentrated primarily in AI/ML toolchains (e.g., Transformers, FlashAttention), with insufficient coverage of web development, systems software, and other domains.
- LLM classification of top-level objects (F1 = 84.94%) introduces classification errors that propagate to task construction quality.
- The relationship between $L_2$ difficulty and "build-from-scratch" benchmarks such as Commit0 is not sufficiently discussed.
- Agent performance when using browser tools or search engines has not been evaluated.

## Related Work & Insights

### vs. SWE-bench
SWE-bench operates at PR granularity with a focus on bug fixing, where feature tasks constitute only 18–22%. FeatureBench targets feature-level tasks with an average implementation size of 790 lines vs. 33 lines—roughly an order of magnitude more complex. On the shared repository subset, Claude Opus 4.5's Resolved Rate drops from 74.4% to 5.2%.

### vs. SWE-Smith / SWE-Flow
SWE-Smith synthesizes tasks heuristically with limited quality guarantees; SWE-Flow relies on F2P tests but neglects P2P validation, failing to ensure that feature extraction does not break existing functionality. FeatureBench's P2P protection mechanism and post-validation pipeline are the key differentiators.

### vs. PaperBench / DevEval
PaperBench (20 instances) and DevEval (22 instances) are too small in scale and depend on expert curation; FeatureBench provides 200 instances with 3,825 executable environments and supports automatic expansion.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic feature-level coding benchmark; test-driven extraction methodology is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven model–scaffold combinations, multi-dimensional ablations, direct comparison with SWE-bench
- Writing Quality: ⭐⭐⭐⭐ Clear structure and in-depth analysis; quantitative task complexity comparisons are intuitive
- Value: ⭐⭐⭐⭐⭐ Exposes a substantial gap between current agents and feature development demands, pointing toward directions for next-generation architectures

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](../../ACL2026/llm_agent/securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)
- [\[ICLR 2026\] PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement](physcensis_physics-augmented_llm_agents_for_complex_physical_scene_arrangement.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)

</div>

<!-- RELATED:END -->
