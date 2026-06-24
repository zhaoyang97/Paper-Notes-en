---
title: >-
  [Paper Note] DevOps-Gym: Benchmarking AI Agents in Software DevOps Cycle
description: >-
  [ICLR 2026][Code Intelligence][DevOps] DevOps-Gym is the first end-to-end agent evaluation benchmark covering the full software DevOps lifecycle (build configuration, runtime monitoring, issue resolution, test generation). It semi-automatically collects 700+ tasks from 30+ real Java/Go projects and provides a dynamic execution environment with tool-calling interfaces. The evaluation reveals that even the strongest Claude Code + Claude-4-Sonnet only achieves 20%~50% success ra…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "DevOps"
  - "AI Agent"
  - "software engineering benchmark"
  - "tool use"
  - "long-horizon tasks"
date: 2026-05-08
content_hash: 302a153bf9f8a6a2
---

# DevOps-Gym: Benchmarking AI Agents in Software DevOps Cycle

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bP48r4dt7Z](https://openreview.net/forum?id=bP48r4dt7Z)  
**Code**: TBD (Paper promises open source, including evaluation framework and baseline implementation)  
**Area**: Code Intelligence / AI Agent / Software Engineering Evaluation  
**Keywords**: DevOps, AI Agent, software engineering benchmark, tool use, long-horizon tasks

## TL;DR
DevOps-Gym is the first end-to-end agent evaluation benchmark covering the full software DevOps lifecycle (build configuration, runtime monitoring, issue resolution, test generation). It semi-automatically collects 700+ tasks from 30+ real Java/Go projects and provides a dynamic execution environment with tool-calling interfaces. The evaluation reveals that even the strongest Claude Code + Claude-4-Sonnet only achieves 20%~50% success rates on operations tasks like monitoring and build configuration, while the success rate for end-to-end pipeline tasks is 0% across all evaluated agents.

## Background & Motivation
**Background**: With the development of LLMs and agents, software development stages (coding, fixing GitHub issues) have been significantly automated. Benchmarks like SWE-bench, HumanEval, and SWT-bench have matured in evaluating "code generation / issue resolution / test generation."

**Limitations of Prior Work**: However, DevOps involves more than just writing code. A complete software delivery cycle includes "post-development" stages such as **build/deployment, runtime monitoring, and operations management**, which still rely heavily on manual labor. Existing benchmarks focus almost entirely on isolated development tasks, evaluating either single-point capabilities or operations tasks in simulated environments or narrow infrastructure scenarios (e.g., IT-bench's simulation, AIOpsLab's microservice cloud). **No benchmark evaluates end-to-end DevOps capabilities on real repositories.**

**Key Challenge**: Operations tasks differ fundamentally from pure code generation. Diagnosing a memory leak requires repeatedly calling domain tools like `ps` and `iostat` to observe process/IO states, interpreting output for anomaly signals, and making multi-step decisions. This requires agents to possess the ability to **analyze large projects, understand dynamic runtime behavior, call domain-specific tools, and perform sequential decision-making**, which are blind spots for existing "read issue description → generate patch" static evaluations. Furthermore, most existing benchmarks are **not designed for agentic systems** and lack dynamic execution environments with tool-calling interfaces.

**Goal**: Construct a benchmark to evaluate an agent's **full-cycle DevOps capabilities** on real software projects. This includes covering un-evaluated operations stages like build and monitoring, presenting tasks in an agentic format requiring tool use and multi-step planning, and providing a rigorous, scalable execution environment for scoring.

**Core Idea**: Distill the DevOps cycle into four Minimal Viable Product (MVP) stages (Build & Configuration → Monitoring → Issue Resolution → Test Generation). Semi-automatically collect tasks from real Java/Go projects and construct end-to-end pipeline tasks that chain these four stages. Encapsulate everything into a terminal-bench formatted tool-use environment, allowing agents to "work on a live system" like real operations engineers.

## Method

### Overall Architecture
DevOps-Gym is a **benchmark dataset + dynamic evaluation environment**. It abstracts the software DevOps cycle into four core stages, providing agents with a real project and a set of corresponding command-line tools for each stage. Agents must complete tasks through multi-step tool calls:

- **Build & Configuration**: Tools such as `maven`, `gradle`, `npm`. Inputs are projects with build failures or new build requirements; outputs are fix patches or complete config files.
- **Monitoring**: Tools such as `top`, `iostat`, `pprof`, `ps`, `free`, `netstat`. Inputs are containerized applications with hidden performance/resource anomalies; outputs are structured diagnostic reports (anomaly type + quantitative evidence).
- **Issue Resolving**: Tools such as `sed`, `grep`, `awk`. Inputs are buggy projects + issue descriptions; outputs are fix patches.
- **Test Generation**: Tools such as JUnit, Maven, Go test. Inputs are issue descriptions; outputs are regression tests (assertions, coverage).

Above these stages are **end-to-end pipeline tasks**: chaining Build → Monitoring → Fix → Test into a single workflow. Any single step failure results in the failure of the entire pipeline. The benchmark contains 704 tasks (54 Build & Config + 34 Monitoring + 310 Issue Resolving + 310 Test Gen) plus 18 end-to-end tasks, spanning 30+ real Java/Go repositories. All are converted to terminal-bench standard formats for unified tool calling and scoring.

Data collection utilizes **semi-automatic processes with heavy expert intervention**: CI logs are mined for "fail-succeed" build pairs, and Multi-SWE-bench/SWT-bench flows are used for fix/test tasks. Experts analyze thousands of real GitHub issues to categorize anomaly types, inject synthetic faults, reproduce environments, and design scoring metrics. The paper notes that reproducing a single monitoring/build task often takes an expert 10+ hours (rebuilding environments, dependencies, configurations, and trigger inputs).

### Key Designs

**1. Four-stage MVP DevOps Pipeline: Including "post-development" operations in evaluation**

To address the blind spot where existing benchmarks only evaluate development, the authors distill DevOps into four stages: Build & Configuration, Monitoring, Issue Resolution, and Test Generation. The first two are new tasks that have rarely been evaluated. This selection forms a minimal closed loop: "making it runnable → finding issues → fixing issues → verifying fixes." **Java and Go** are specifically chosen over Python because they represent large enterprise projects with standardized, non-trivial build systems and mature monitoring stacks, reflecting real-world complexity. This also exposes the poor cross-language capabilities of Python-centric agents.

**2. Monitoring tasks focused on "Progressive Anomalies" rather than crashes: Testing continuous observation and temporal reasoning**

Monitoring is the most difficult stage. The authors deliberately **evaluate performance/resource anomalies instead of crashes**. Crashes are easily identified from console logs and lack diagnostic value. Six types of anomalies—memory leaks, disk leaks, handle exhaustion, CPU spikes (resource-related), IO bottlenecks, and inefficient SQL queries (performance-related)—exhibit **slow degradation patterns**. Early symptoms are subtle, requiring agents to collect system behavior across multiple requests and compare marginal differences. Tasks also include "healthy system" samples to test false positive control. Scoring uses binary accuracy: agents write conclusions to a file, and a pytest script verifies the type against the ground truth.

**3. Rigorous Decontamination and Reproducibility: Ensuring trustworthy results**

The authors use **prefix-completion analysis** to filter repositories that might have entered training corpora and **delete git metadata** to prevent agents from finding answers in version history. For each task, three senior DevOps engineers independently perform "observability validation" to ensure anomalies are detectable within 5–15 minutes using standard tools. For build tasks requiring both dynamic execution and static analysis, multi-dimensional scoring metrics are manually designed per task.

**4. End-to-End Pipeline Tasks: Testing cross-stage context transfer and long-horizon planning**

Isolated tasks fail to measure an agent's ability to **maintain context across tools, pass diagnostic information between stages, and verify end-to-end correctness**. Chain tasks are constructed: Stage 1 requires fixing a build error; Stage 2 requires monitoring the deployed system to detect a latent anomaly; Stage 3 requires a code-level fix; Stage 4 requires writing a regression test specifically for the performance characteristic identified in Stage 2. All four stages must pass sequentially for success, mirroring real-world constraints where partial solutions have no value.

### A Complete Example
Using a memory leak end-to-end task: ① The agent receives a file server repository with "missing dependencies/wrong paths," uses `maven`/`gradle` to diagnose logs, fixes configs, and compiles (Stage 1 Build). ② Once running, the agent uses `top`/`free`/`ps` for continuous observation, noticing memory increases during large file downloads. It must maintain attention over a monitoring token stream and compare large vs. small file requests to conclude "memory leak" (Stage 2 Monitoring). ③ It locates the unreleased cache code and generates a patch (Stage 3 Fix). ④ It rebuilds and writes a regression test to verify no further leaks occur (Stage 4 Test). Agents often fail at Stage 2 because the context is overwhelmed by monitoring logs.

## Key Experimental Results

### Main Results
Evaluation of three agent frameworks (OpenHands, mini-SWE-agent, Claude Code) with 5 LLMs (Claude-4-Sonnet, o4-mini, Gemini-2.5-Pro, DeepSeek-V3.1, Qwen3-Coder-30B). Success rates (%):

| Agent + Model | Build & Config | Monitoring | Issue Resolving | Test Generation |
|--------------|---------|------|---------|---------|
| Claude Code + Claude-4-Sonnet | **51.85** | **20.56** | **23.87** | **13.87** |
| OpenHands + Claude-4-Sonnet | 42.59 | 14.70 | 23.87 | 11.61 |
| OpenHands + o4-mini | 24.07 | 8.82 | 10.32 | 8.70 |
| OpenHands + Gemini-2.5-Pro | 16.66 | 11.76 | 10.96 | 2.90 |
| OpenHands + Qwen3-Coder-30B | 20.37 | 5.89 | 13.22 | 6.13 |
| OpenHands + DeepSeek-V3.1 | 11.11 | 0.00 | 14.20 | 3.22 |
| mini-SWE-Agent + Claude-4-Sonnet | 29.62 | 2.91 | 5.16 | 0.98 |
| Aider + Claude-4-Sonnet | 5.55 | 0.00 | 9.67 | 2.25 |

The strongest combination, Claude Code + Claude-4-Sonnet, only achieves 13.87%~51.85% across stages. **The success rate for end-to-end pipeline tasks for all agents is 0%.**

### Cross-language Comparison & Error Analysis

| Comparison Dimension | Data | Description |
|---------|------|------|
| Cross-language Performance Drop | SWE-bench-Verified 70.4% → DevOps-Gym(Java/Go) 23.87% | Using OpenHands+Claude-4-Sonnet, performance drops ~47 points when moving to Java/Go |
| Build & Config Error Types | Limited toolchain detection 33% / Planning failure 23% / Domain knowledge gap 37% | 17% belong to "inherently difficult" tasks |
| Framework Variance | Claude Code significantly outperforms mini-SWE-Agent | Agent architecture is critical for SE tasks |

### Key Findings
- **Monitoring is the largest weakness**: DeepSeek-V3.1 and Aider hit 0% in monitoring. Reasons include: ① Monitoring generates evolving state streams that exceed token limits before an anomaly manifests. ② Agents fail to focus on real-time monitoring and get distracted by early observations. ③ Poor baseline discrimination, misreporting normal fluctuations as anomalies.
- **Test generation is harder than issue resolution**: Success rates for generating high-quality tests are lower than fixing rates for the same issues. Tests requires both static understanding and dynamic reasoning of bug triggers, whereas patches can sometimes be guessed from issue descriptions.
- **Cross-language gap is real**: Compilation, dependency management, and build configurations in Java/Go are difficult for Python-centric agents; this persists even with Claude-4.
- **Build implementation tasks (migration/release) are particularly poor**: Agents struggle with the internal mechanisms of tools like Maven and goreleaser beyond just parsing error logs.

## Highlights & Insights
- **Standardizing "post-development" operations** into an evaluatable benchmark is the primary contribution.
- **Focusing on "progressive anomalies" rather than crashes** is a clever design choice that effectively identifies gaps in temporal reasoning.
- **The 0% end-to-end result** is a powerful signal demonstrating the fundamental lack of cross-stage context transfer and long-horizon planning in current agents.
- **Strict decontamination (prefix-completion + removing git metadata)** makes the "cross-language performance drop" attribution credible.
- **The "Tools are OOD" (Out-of-Distribution) hypothesis**: Base LLMs are rarely trained on DevOps tool-calling trajectories, suggesting the need for specialized models trained on agentic workflow data.

## Limitations & Future Work
- **Limited scale and coverage**: Monitoring only has 30–34 tasks, and end-to-end has 14–18. It currently only covers Java/Go and four DevOps stages.
- **High evaluation cost**: Experts spend 10+ hours reproducing single monitoring/build tasks, limiting rapid scaling.
- **Performance upper bound focus**: Currently mainly tests the strongest agents/models; hasn't systematically evaluated weaker models to map the "difficulty gradient."
- **Binary monitoring metrics**: Scoring only checks if the anomaly type is correct, which may not distinguish between a "lucky guess" and a "solid diagnosis."

## Related Work & Insights
- **vs SWE-bench / Multi-SWE-bench**: DevOps-Gym extends issue resolution to Java/Go with stricter decontamination; performance drops from 70.4% to 23.87% confirm the cross-language gap.
- **vs SWT-bench**: SWT-bench evaluates Python test generation with code coverage; DevOps-Gym uses stricter "reproduction of failure" metrics for compiled languages.
- **vs IT-bench / AIOpsLab**: While these also evaluate monitoring/ops, IT-bench uses simulation and AIOpsLab focuses on microservice cloud efficiency; DevOps-Gym focuses on task correctness across end-to-end stages on real repositories.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First end-to-end full-cycle DevOps agent benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 frameworks × 5 models with detailed error analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and task construction.
- Value: ⭐⭐⭐⭐⭐ Targets the "agents can code but can't operate" blind spot, providing direction for SE agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](innogym_benchmarking_the_innovation_potential_of_ai_agents.md)
- [\[ICML 2025\] Training Software Engineering Agents and Verifiers with SWE-Gym](../../ICML2025/code_intelligence/training_software_engineering_agents_and_verifiers_with_swe-gym.md)
- [\[ICLR 2026\] The Matthew Effect of AI Programming Assistants: A Hidden Bias in Software Evolution](the_matthew_effect_of_ai_programming_assistants_a_hidden_bias_in_software_evolut.md)
- [\[ICLR 2026\] SWE-RM: Execution-Free Feedback for Software Engineering Agents](swe-rm_execution-free_feedback_for_software_engineering_agents.md)
- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)

</div>

<!-- RELATED:END -->
