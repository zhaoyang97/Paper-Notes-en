---
title: >-
  [Paper Note] MaintainCoder: Maintainable Code Generation Under Dynamic Requirements
description: >-
  [NeurIPS 2025][maintainable code] This work is the first to systematically define and address the **maintainability** problem in LLM-based code generation, contributing both a benchmark and a method. MaintainBench evaluates code maintainability under requirement evolution using 4 change patterns and dynamic metrics; MaintainCoder integrates the Waterfall model, design patterns, and 6 specialized agents, achieving 60%+ improvement on dynamic maintainability metrics while also improving initial code correctness.
tags:
  - NeurIPS 2025
  - maintainable code
  - design patterns
  - waterfall model
  - multi-agent
  - dynamic requirements
  - MaintainBench
date: 2026-05-08
content_hash: e795f532d2a20b4f
---

# MaintainCoder: Maintainable Code Generation Under Dynamic Requirements

**Conference**: NeurIPS 2025
**arXiv**: [2503.24260](https://arxiv.org/abs/2503.24260)
**Code**: [https://github.com/IAAR-Shanghai/MaintainCoder](https://github.com/IAAR-Shanghai/MaintainCoder)
**Area**: LLM Agent / Code Generation / Software Engineering
**Keywords**: maintainable code, design patterns, waterfall model, multi-agent, dynamic requirements, MaintainBench

## TL;DR
This work is the first to systematically define and address the **maintainability** problem in LLM-based code generation, contributing both a benchmark and a method. MaintainBench evaluates code maintainability under requirement evolution using 4 change patterns and dynamic metrics; MaintainCoder integrates the Waterfall model, design patterns, and 6 specialized agents, achieving 60%+ improvement on dynamic maintainability metrics while also improving initial code correctness.

## Background & Motivation

LLM-based code generation has made significant progress on functional correctness (HumanEval, MBPP, SWE-Bench), yet it neglects the most critical dimension in real-world software development: **maintainability**—the ability of code to adapt to evolving requirements with minimal modification effort. As Heraclitus observed, "the only constant is change," and this is especially true in software engineering.

**Real-world cases of the maintainability crisis**:
- **Knight Capital**: Unmaintained legacy code evaporated $440 million in minutes.
- **Y2K Problem**: Short-sighted design decisions caused $100 billion in global remediation costs.
- Industry research consistently shows that **60–80% of software lifecycle costs** arise from post-deployment maintenance, rooted in high coupling and low cohesion.

**Three fundamental gaps**:
1. No benchmark quantifies maintainability over requirement evolution cycles.
2. No method systematically applies software engineering principles such as design patterns to LLM code generation.
3. Traditional static metrics (cyclomatic complexity CC, maintainability index MI) analyze only code structure and fail to capture dynamic maintenance scenarios.

**Evidence supporting the approach**: Dong et al. demonstrated that the Waterfall model improves code correctness by 29.9–47.1%; Hegedűs et al. found a Pearson correlation of 0.89 between design patterns and software maintainability.

## Core Problem
How can LLM-generated code $C_0 = \mathcal{G}(P_0)$ adapt to a dynamic requirement sequence $\{P_0, P_1, \ldots, P_n\}$ with minimal cumulative maintenance cost $\mathcal{M}(C_0) = \mathbb{E}[\sum_{i=0}^{n-1} \gamma^i \mathcal{M}(C_i \to C_{i+1})]$?

## Method

### I. MaintainBench: The First Dynamic Maintainability Benchmark

**Data construction**—extends 5 classic benchmarks across three difficulty levels (500+ Python samples total):
- **Introductory**: HumanEval-Dyn + MBPP-Dyn (30 problems each → 120 problems)
- **Mixed**: APPS-Dyn (50 problems → 200+ problems, spanning introductory to competitive difficulty)
- **Competitive**: CodeContests-Dyn + xCodeEval-Dyn (30 problems each → 120+ problems)

**Four requirement change patterns** (corresponding to the four maintenance types in ISO/IEC/IEEE 14764:2022):
1. **Functional extension $\pi_{ext}$**: Adding new functional requirements that invoke existing functions (adaptive + perfective maintenance).
2. **Interface modification $\pi_{int}$**: Changing API contracts such as input parameters and return types, e.g., due to external library upgrades (adaptive maintenance).
3. **Data structure transformation $\pi_{dst}$**: Replacing data representations to meet efficiency or scalability needs (perfective maintenance).
4. **Error handling enhancement $\pi_{err}$**: Introducing exception handling for ZeroDivisionError, IndexError, etc. (corrective + preventive maintenance).

**Construction pipeline**: Original problem $P_0$ → GPT-4o generates variants $(P_1, S_1', T_1')$ per change pattern → Python interpreter executes tests automatically → failures trigger expert diagnosis and correction → iteration until all tests pass → multi-stage expert review for quality assurance.

**Dynamic metric design** (via Monte Carlo estimation + first-order truncation approximation of dynamic maintainability):
- **Pass@k**: Functional correctness after requirement modification ($k$ up to 5).
- **$\text{Code}_{diff}^{per}$**: Modified lines as a percentage of the original code (relative change).
- **$\text{Code}_{diff}^{abs}$**: Absolute number of modified lines.
- **$\text{AST}_{sim}$**: Structural similarity of abstract syntax trees before and after modification.

### II. MaintainCoder: A Multi-Agent Maintainable Code Generation System

The overall architecture mirrors the human software development lifecycle (Waterfall model), implemented via the AutoGen framework for inter-agent communication. It comprises two modules and 6 specialized agents:

**Code Framework Module**—transforms requirements into a maintainable architectural blueprint:
1. **Requirements Analysis Agent**: Applies chain-of-thought to decompose problems step by step, extracting core functionality, identifying key challenges, and proposing high-level solutions while avoiding unnecessary complexity.
2. **Design Pattern Selection Agent**: Acts as a software architect, selecting the most appropriate design pattern (Strategy, Factory, Observer, etc.) for each functional module, prioritizing modularity, reduced coupling, extensibility, and reusability, with alternative options provided.
3. **Architecture Design Agent**: Constructs class structures based on design patterns, adheres to the Single Responsibility Principle, explicitly defines inter-class dependencies, and iteratively revises based on review feedback.
4. **Architecture Review Agent**: Examines design clarity, extensibility, performance, and best practices; identifies coupling, cohesion, and reuse issues; **prevents over-fragmentation**; and provides actionable improvement recommendations.

**Code Generation Module**—implements the blueprint as executable code:
5. **Code Generation Agent**: Translates the architectural design into PEP 8/PEP 257 compliant code; comments focus on design intent and non-obvious logic rather than redundant descriptions; inserts test samples for iterative debugging.
6. **Code Optimization Agent**: Executes code → collects syntax errors, test failures, and anomalous behavior → applies chain-of-thought to diagnose root causes → fixes and re-tests → iterates until all tests pass.

## Key Experimental Results

Experiments proceed in two phases: Phase I generates initial code $C_0$ using MaintainCoder or a baseline; Phase II uses a fixed generator (e.g., GPT-4o-mini) to apply requirement modifications to $C_0$ and evaluates dynamic maintainability.

### Main Results on Mixed-Level APPS-Dyn

| Method | MI↑ | CC↓ | Pass@5↑ | AST_sim↑ | Code_diff^per↓ |
|--------|-----|-----|---------|----------|----------------|
| GPT-4o-mini | 63.3 | 5.10 | 35.5% | 0.589 | 140% |
| AgentCoder (4o-mini) | 63.3 | 5.81 | 21.0% | 0.510 | 66.3% |
| MapCoder (4o-mini) | 67.8 | 5.98 | 30.5% | 0.583 | 73.8% |
| **MaintainCoder (4o-mini)** | **69.5** | **2.75** | **50.5%** | **0.797** | **29.4%** |
| DeepSeek-V3 | 61.8 | 7.59 | 59.5% | 0.598 | 131% |
| **MaintainCoder (DS-V3)** | **62.4** | **3.21** | **62.5%** | **0.828** | **29.2%** |
| GPT-4o | 63.0 | 4.58 | 39.5% | 0.556 | 140% |
| Claude-3.7-Sonnet | 59.3 | 6.65 | 48.5% | 0.620 | 85.2% |
| Gemini-2.5-Flash | 59.7 | 9.00 | 51.0% | 0.631 | 108% |

Key finding: MaintainCoder's CC is approximately **1/2 to 1/3** of baselines, AST similarity is **28%+ higher**, and code modification volume is only **1/5** of baselines.

### Competitive-Level Datasets
- **CodeContests-Dyn**: MaintainCoder (4o-mini) achieves Pass@5 32.6%, CC 2.68, AST_sim 0.833, Code_diff^per 23.2%—best across all metrics.
- **xCodeEval-Dyn**: MaintainCoder (DS-V3) achieves Pass@5 **36.7%** (surpassing second-place GPT-4o at 32.8%, a **60% relative improvement**), AST_sim 0.785, Code_diff^per 33.0%.
- On competitive-level problems, CC for baseline multi-agent systems (AgentCoder/MapCoder) inflates to 15–20, while MaintainCoder maintains CC around 3.

### Correctness and Maintainability as Complementary Goals

| Method | APPS | CodeContests | xCodeEval |
|--------|------|-------------|-----------|
| GPT-4o-mini | 44% | 18% | 46% |
| **MaintainCoder (4o-mini)** | **48%** | **23%** | **57%** |
| DeepSeek-V3 | 66% | 48% | 75% |
| **MaintainCoder (DS-V3)** | **69%** | **51%** | **77%** |

Gains increase with problem complexity: GPT-4o-mini improves from 46%→57% (+11%) on xCodeEval, compared to only +4% on APPS. Improvements persist even over the already strong DeepSeek-V3.

### Human Baseline and Reasoning Model Comparison (CodeContests-Dyn)

| Method | CC↓ | Pass@5↑ | AST_sim↑ | Code_diff^per↓ |
|--------|-----|---------|----------|----------------|
| Human programmers (CF 1700–2300) | 8.17 | 23.5% | 0.541 | 112.3% |
| o3-mini | 11.3 | 30.3% | 0.661 | 101.6% |
| EvoMAC (4o-mini) | 5.18 | 26.5% | 0.685 | 60.1% |
| MetaGPT (4.1-mini) | 7.63 | 30.3% | 0.760 | 44.3% |
| **MaintainCoder (4o-mini)** | **2.68** | **32.6%** | **0.833** | **23.2%** |
| **MaintainCoder (o3-mini)** | **3.85** | **36.4%** | **0.794** | **27.8%** |

Human programmers with competitive experience (Codeforces 1700–2300) produce code under time pressure with lower maintainability metrics than AI-generated code. MaintainCoder yields substantial improvements even over the strong reasoning model o3-mini.

### Static vs. Dynamic Metrics
- AgentCoder/MapCoder achieve significantly higher MI than baselines, yet their Pass@k **decreases**—static metrics yield misleading conclusions.
- MI and CC frequently **contradict** each other (e.g., MapCoder's MI=66.1 and CC=7.32 versus GPT-4o-mini's MI=57.8 and CC=6.06, where both metrics simultaneously worsen).
- Dynamic metrics (Pass@k, AST_sim, Code_diff) are **highly consistent** with one another—confirming that dynamic metrics more accurately reflect true maintainability.

### Ablation Study

| Configuration | APPS-Dyn | xCodeEval-Dyn |
|---------------|----------|--------------|
| Full MaintainCoder (4o-mini) | 50.5% | 27.3% |
| w/o Architecture Review | 49.0% (−3.0%) | 20.3% (−25.7%) |
| w/o Code Optimization | 40.0% (−20.8%) | 17.2% (−37.1%) |

The Code Optimization Agent contributes more overall, but Architecture Review is also critical on high-difficulty tasks (−25.7% on xCodeEval). Architecture Review typically requires only one iteration, making it a lightweight yet effective component.

### Computational Cost

| Dataset | MaintainCoder | MapCoder | o3-mini | MetaGPT/ChatDev | GPT-4o-mini |
|---------|--------------|----------|---------|-----------------|-------------|
| CodeContests | 33.1k | 38.7k | 20.8k | 50k+ | 2.5k |
| xCodeEval | 29.6k | 23.5k | 21.2k | 50k+ | 2.3k |

MaintainCoder's token consumption is comparable to MapCoder and far below MetaGPT/ChatDev; the upfront cost is offset by substantially reduced downstream maintenance overhead.

## Highlights & Insights
- **Paradigm shift in problem framing**: The first work to systematically introduce "maintainability" from a software engineering perspective into LLM code generation, replacing static test cases with dynamic requirement evolution.
- **MaintainBench fills an evaluation gap**: 4 requirement change patterns × 3 difficulty levels × 500+ samples + a dynamic metric framework.
- **Counterintuitive finding**: Multi-agent systems (AgentCoder/MapCoder) optimized for single-round correctness actually **degrade** long-term maintainability; the effect of CoT/Self-Planning on maintainability is unstable and near-random.
- **Dual benefit, not a trade-off**: Good architectural design not only improves maintainability but also promotes initial code correctness through structural clarity.
- **Static metrics falsified**: MI and CC are mutually contradictory and disconnected from actual maintenance costs.
- **Surpassing human programmers**: Code produced by competitive programmers under time pressure exhibits lower maintainability than AI-generated code.

## Limitations & Future Work
- The multi-agent pipeline incurs higher upfront API costs (~30k tokens/problem), making it unsuitable for real-time code assistance.
- Evaluation is limited to Python; generalization to other languages and repository-scale projects (e.g., SWE-Bench scale) remains to be validated.
- Design pattern selection relies on the LLM's intrinsic knowledge and may fail for complex architectural decisions.
- Only one round of requirement change ($P_0 \to P_1$) is simulated; multi-round continuous evolution evaluation is left for future work.
- MaintainBench's requirement variations are generated by GPT-4o and may introduce distributional bias.

## Related Work & Insights
- **vs. AgentCoder/MapCoder/EvoMAC**: These systems optimize single-round correctness while ignoring structural quality; CC inflates to 15–20 on competitive-level problems.
- **vs. SWE-Bench**: SWE-Bench evaluates the ability to resolve GitHub issues, not whether the generated code itself is maintainable.
- **vs. MetaGPT/ChatDev**: Role-based multi-agent systems that do not emphasize design patterns or the Waterfall process, and incur higher token costs (50k+).
- **vs. CoT/Self-Planning**: Prompt engineering approaches have unstable effects on maintainability; MaintainCoder's structured pipeline is significantly more robust.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to systematically define and address maintainable code generation; contributes both a benchmark and a method with a highly forward-looking problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks × 10+ baselines (including human, reasoning model, and multi-agent) + ablation + static vs. dynamic metric analysis + cost analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation grounded in Knight Capital/Y2K cases is impactful; formal definitions are complete; methodology is systematic.
- Value: ⭐⭐⭐⭐⭐ Paradigm shift—from "generating correct code" to "generating maintainable code"; MaintainBench has the potential to become a standard benchmark.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](embedding_alignment_in_code_generation_for_audio.md)
- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[NeurIPS 2025\] Text-to-Code Generation for Modular Building Layouts in Building Information Modeling](text-to-code_generation_for_modular_building_layouts_in_building_information_mod.md)
- [\[NeurIPS 2025\] Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models](table2latex-rl_high-fidelity_latex_code_generation_from_table_images_via_reinfor.md)
- [\[AAAI 2026\] TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](../../AAAI2026/code_intelligence/tapas_are_free_training-free_adaptation_of_programmatic_agen.md)

<!-- RELATED:END -->
