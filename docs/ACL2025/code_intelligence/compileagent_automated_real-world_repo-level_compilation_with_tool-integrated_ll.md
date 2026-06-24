---
title: >-
  [Paper Note] CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System
description: >-
  [ACL 2025][Code Intelligence][Automated Compilation] Proposes CompileAgent, the first LLM agent framework designed for repository-level code compilation. By integrating five specialized tools and a flow-based agent strategy, it improves the compilation success rate by up to 71% on CompileAgentBench (consisting of 100 real-world C/C++ projects), costing only $0.22 per project on average.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Automated Compilation"
  - "LLM Agent"
  - "Repository-Level Compilation"
  - "Tool Integration"
  - "Multi-Agent Discussion"
date: 2026-05-08
content_hash: 2e76465682e4f731
---

# CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System

**Conference**: ACL 2025  
**arXiv**: [2505.04254](https://arxiv.org/abs/2505.04254)  
**Code**: None  
**Area**: Code Intelligence  
**Keywords**: Automated Compilation, LLM Agent, Repository-Level Compilation, Tool Integration, Multi-Agent Discussion

## TL;DR
Proposes CompileAgent, the first LLM agent framework designed for repository-level code compilation. By integrating five specialized tools and a flow-based agent strategy, it improves the compilation success rate by up to 71% on CompileAgentBench (consisting of 100 real-world C/C++ projects), costing only $0.22 per project on average.

## Background & Motivation

### Background
As open-source projects grow increasingly complex, compiling source code into executables or libraries has become a common requirement in software development. Compilation artifacts are not only directly usable but also support downstream tasks such as dataset construction, performance testing, and security vulnerability analysis. However, repository-level compilation is far more complex than single-file compilation, involving challenges in environment adaptation, dependency management, build configuration, and more.

### Limitations of Prior Work

**Scattered and hard-to-find compilation instructions**: Build guides can be hidden in various documents such as README, doc.html, install.txt, or even on external websites, requiring significant manual effort from developers to locate them.

**Difficult-to-resolve compilation errors**: Issues like dependency conflicts, environment mismatches, and code compatibility require rich experience and iterative debugging.

**Limited capabilities of existing tools**: Tools like Oss-Fuzz-Gen only execute predefined compilation commands based on specific filenames, failing on projects with non-standard file names and struggling to adapt to dynamically changing environments.

### Key Challenge
Repository-level compilation requires understanding the entire codebase structure, documentation, dependencies, and dynamically interacting with an interactive environment. This exceeds the capabilities of traditional rule-based methods but perfectly aligns with the strengths of LLM Agents.

### Key Insight
This work introduces LLM Agents to repository-level compilation—an unexplored territory—and designs specialized tools and flow-based strategies to simulate the real-world compilation process of human developers.

## Method

### Overall Architecture
CompileAgent consists of two core modules: **CompileNavigator** (for searching build instructions) and **ErrorSolver** (for resolving compilation errors), integrating five distinct tools scheduled by a MasterAgent using a flow-based strategy.

Overall Compilation Process:
1. Download the code repository and mount it in a Docker container.
2. Retrieve the repository structure (using the `tree` command).
3. Use FileNavigator to locate files containing compilation instructions.
4. Use InstructionExtractor to extract and execute build instructions.
5. If compilation succeeds, terminate; if errors occur, try resolving them internally first, and invoke ErrorSolver if that fails.

### Key Designs

#### 1. **CompileNavigator Module** — Finding Build Instructions
**Function**: Locates and extracts compilation instructions from complex repository structures.

It includes three tools:
- **Shell**: Isolates the compilation environment (Ubuntu 22.04) via Docker containers to protect the host machine, allowing the LLM to execute arbitrary commands over SSH.
- **File Navigator**: Deploys two collaborative agents (SearchAgent I and II) that take the repository structure as input and discuss to identify the files most likely containing compilation instructions.
- **Instruction Extractor**: A SummarizeAgent reads the content of specified files, searches for compilation-related URLs within them, crawls web pages if necessary, and finally summarizes and outputs the compilation instructions.

**Design Motivation**: Simulates the human developer's process of finding compilation guides—inspecting the repository structure first, locating specific files, and then extracting instructions. The collaborative discussion between two agents improves file localization accuracy.

#### 2. **ErrorSolver Module** — Solving Compilation Errors
**Function**: Automatically analyzes and debugs when errors occur during compilation.

It includes two tools:
- **Website Search**: Wraps Google search, prioritizing solutions from reliable open-source platforms like GitHub and StackOverflow, and aggregates relevant information.
- **Multi-Agent Discussion**: Three agents analyze the compilation error and generate their own initial design solutions, followed by up to 3 rounds of discussion. After each round, command-line solutions are segmented and counted; if duplicate solutions exceed a threshold, a consensus is reached to generate the final solution.

**Design Motivation**: Compilation errors typically carry clear error logs (path issues, environment configuration, compatibility problems, etc.) that do not require complex reasoning. However, multi-agent discussion integrates diverse perspectives to enhance solution quality.

#### 3. **Flow-based Agent Strategy**
**Mechanism**: Defines a fixed execution order for tools, seamlessly chaining them through engineered prompts.

Unlike ReAct (reasoning and acting at each step) and Plan-and-Execute (planning followed by execution), the flow-based strategy mimics the actual compilation workflow of human developers: locate build guide $\rightarrow$ execute compilation commands $\rightarrow$ attempt self-correction on error $\rightarrow$ query external resources and discuss upon failure. This structured flow minimizes the decision-making burden of LLMs during compilation tasks.

### Loss & Training
This work does not involve model training. CompileAgent is a test-time inference agent framework. All agents are powered by existing off-the-shelf LLMs (e.g., GPT-4o, Claude-3.5-sonnet), completing tasks via prompt engineering and tool invocation.

## Key Experimental Results

### CompileAgentBench Benchmark
- 100 real-world C/C++ projects on GitHub.
- Covers 14 different domains (cryptography, audio, neural networks, etc.).
- Manually compiled and verified by three developers with 3–4 years of experience, costing a total of 46 person-hours.
- Categorized by the acquisition difficulty of compilation instructions.

### Main Results

| Model | Oss-Fuzz-Gen | Readme-AI | RAG | CompileAgent | Gain |
|------|-------------|-----------|-----|-------------|---------|
| GPT-4o | 25% | 70% | 71% | 89% | +18~64% |
| Claude-3.5-sonnet | 25% | 74% | 73% | 91% | +17~66% |
| Gemini-1.5-flash | 25% | 56% | 57% | 77% | +20~52% |
| Qwen2.5-32B | 25% | 57% | 55% | 72% | +15~47% |
| Mixtral-8×7B | 25% | 35% | 37% | 47% | +10~22% |
| LLaMA3.1-70B | 25% | 52% | 61% | 71% | +10~46% |
| DeepSeek-v2.5 | 25% | 56% | 59% | 80% | +21~55% |

CompileAgent significantly outperforms all baselines across all 7 LLMs. Claude-3.5-sonnet achieves the highest success rate of 91%, representing a 66% improvement over Oss-Fuzz-Gen. The average compilation cost per project is only $0.22.

### Ablation Study

| Configuration | Success Rate | Time (h) | Cost ($) | Description |
|------|--------|---------|---------|------|
| CompileAgent (Full) | 89% | 8.38 | 16.53 | Based on GPT-4o |
| - File Navigator | 81% | 6.93 | 17.32 | Decreased by 8%, fails to precisely locate compilation files |
| - Instruction Extractor | 77% | 7.18 | 18.26 | Decreased by 12%, fails to extract compilation instructions from files |
| - Website Search | 84% | 7.25 | 16.53 | Decreased by 5%, fails to query web search for error solutions |
| - Multi-Agent Discussion | 71% | 8.77 | 18.89 | Decreased by 18%; the most critical component, significantly weakening the capability to handle complex errors |

Multi-Agent Discussion is the most frequently called tool (averaging 1.87 times per project) and the most critical component. Removing it results in the largest drop in success rate.

### Strategy Comparison

| Strategy | Claude-3.5-sonnet | GPT-4o | Qwen2.5-32B |
|------|-------------------|--------|-------------|
| ReAct | ~60% | ~55% | ~40% |
| Plan-and-Execute | Lowest | Lowest | Lowest |
| OpenAIFunc | - | Medium | - |
| **Flow-based (Ours)** | **91%** | **89%** | **72%** |

The flow-based strategy significantly outperforms other strategies across all evaluated models, maintaining a 30% to 53% performance advantage.

### Key Findings
- **Stronger LLMs lead to better compilation results**: Model capabilities are positively correlated with compilation success rates, though Mixtral-8×7B performs poorly, potential due to its mixture-of-experts architecture.
- **Multi-Agent Discussion is the core component**: Resolving compilation errors heavily relies on multi-agent collaboration, as single agents struggle with complex dependency logic.
- **Flow-based strategies outperform flexible strategies**: For structured verification tasks like compilation, a fixed workflow is more effective than free-form decision-making.
- **Extremely low cost**: Costing $0.22 per project on average, which is orders of magnitude lower than the manual time cost (46 person-hours/100 projects).
- **Common failure causes**: Complex build dependency chains (specific library versions), toolchain mismatches, and complex configuration setups.

## Highlights & Insights
- **First work to apply LLM Agents to repository-level compilation**, filling a gap in automated compilation research.
- **Tool designs highly align with the actual human developer workflow**: Inspecting documents $\rightarrow$ extracting instructions $\rightarrow$ executing $\rightarrow$ web searching on error $\rightarrow$ discussion-based debugging. This reflects a deep understanding of real-world compilation settings.
- **The Docker-isolated design** guarantees host-machine security while providing independent build environments, representing a sound engineering choice.
- **The consensus mechanism of Multi-Agent Discussion** (counting segmented commands) is simple yet effective, avoiding the overhead of heavy-weight reasoning frameworks.
- **High cost-efficiency**: $0.22/project versus approximately 0.46 person-hours/project for manual efforts.

## Limitations & Future Work
- **Dependence on LLM comprehension**: Agents may misunderstand prompts or commands, leading to repetitive or erroneous operations. Future work should explore fine-tuning to improve instruction-following capabilities.
- **Basic toolsets**: Advanced programming and debugging tools (e.g., GDB, Valgrind) are not yet integrated. Expanding the toolset could improve the capability to resolve complex bugs.
- **High dependence on prompt engineering**: System performance is tightly coupled with prompt quality; more automated prompt optimization techniques need to be explored.
- **Limited to C/C++**: While the possibility of cross-language extension is discussed, experiments are currently only conducted on C/C++ datasets.
- **Limited benchmark scale**: 100 projects may not be sufficient for comprehensive evaluations; the benchmark should be scaled up to larger and more diverse projects.

## Related Work & Insights
- **Oss-Fuzz-Gen**: A rule-based method relying on file name matching, which fails when standard filenames are absent, highlighting the flexibility advantage of LLM Agents.
- **SWE-Agent / OpenHands**: Agent frameworks for code debugging and repair. The design of CompileAgent could be generalized to support broader software engineering tasks.
- **ReAct vs Flow-based**: For tasks with explicit, predefined workflows, structured flows surpass free-form reasoning, offering valuable guidance for agent strategy designs.
- **Multi-Agent Collaboration**: The round-table discussion mechanism from ReConcile has been successfully adapted to resolve compilation bugs.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing LLM Agents to repository-level compilation represents a critical innovation, though the agent framework architecture itself is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested with 7 LLMs, 3 baselines, 4 strategies, and a complete ablation study, though the benchmark is restricted to 100 C/C++ projects.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the method is detailed, though some paragraphs are slightly repetitive.
- Value: ⭐⭐⭐⭐ Practical value for software engineering automation, with a low cost of $0.22 per project indicating strong real-world utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](../../ACL2026/code_intelligence/logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ICLR 2026\] EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits](../../ICLR2026/code_intelligence/edit-bench_evaluating_llm_abilities_to_perform_real-world_instructed_code_edits.md)
- [\[NeurIPS 2025\] VeriMaAS: Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/code_intelligence/automated_multi-agent_workflows_for_rtl_design.md)
- [\[ICML 2026\] Pull Requests as a Training Signal for Repo-Level Code Editing](../../ICML2026/code_intelligence/pull_requests_as_a_training_signal_for_repo-level_code_editing.md)
- [\[ICLR 2026\] CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning](../../ICLR2026/code_intelligence/codesense_a_real-world_benchmark_and_dataset_for_code_semantic_reasoning.md)

</div>

<!-- RELATED:END -->
