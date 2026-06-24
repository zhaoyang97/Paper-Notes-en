---
title: >-
  [Paper Note] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement
description: >-
  [ACL 2025][LLM Agent][Self-Referential Agent] Introduces Gödel Agent, a self-referential agent framework inspired by the Gödel Machine, which can read and modify its own code (including modifying its own modification logic) at runtime via Python monkey patching to achieve recursive self-improvement. It outperforms hand-crafted and meta-learning-optimized agents on DROP, MGSM, MMLU, and GPQA.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Self-Referential Agent"
  - "Recursive Self-Improvement"
  - "Monkey Patching"
  - "Meta-Learning"
  - "Agent Design Space Search"
date: 2026-05-08
content_hash: f7c65e2e4e6d7b17
---

# Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement

**Conference**: ACL 2025  
**arXiv**: [2410.04444](https://arxiv.org/abs/2410.04444)  
**Code**: [https://github.com/Arvid-pku/Godel_Agent](https://github.com/Arvid-pku/Godel_Agent)  
**Area**: LLM Agent  
**Keywords**: Self-Referential Agent, Recursive Self-Improvement, Monkey Patching, Meta-Learning, Agent Design Space Search

## TL;DR
Introduces Gödel Agent, a self-referential agent framework inspired by the Gödel Machine, which can read and modify its own code (including modifying its own modification logic) at runtime via Python monkey patching to achieve recursive self-improvement. It outperforms hand-crafted and meta-learning-optimized agents on DROP, MGSM, MMLU, and GPQA.

## Background & Motivation

**Background**: LLM agent systems are divided into two categories—hand-crafted agents with fixed workflows (e.g., CoT, Self-Refine, LLM Debate) and meta-learning optimized agents (e.g., Meta Agent Search, DSPy). The former relies entirely on human priors, while the latter can automatically optimize policies but employs a fixed meta-learning algorithm.

**Limitations of Prior Work**: (1) Hand-crafted agents cannot evolve once deployed; (2) Meta-learning agents are constrained by fixed meta-algorithms, leaving search spaces artificially restricted—for instance, only optimizing prompts or only collecting demonstrations; (3) Neither can search the complete agent design space.

**Key Challenge**: The agent's "optimization method" is itself an object that can be optimized. If the optimization algorithm is fixed, it remains impossible to discover better optimization algorithms.

**Goal**: Can an agent autonomously determine its own operational logic, module design, and even modify its own optimization methods?

**Key Insight**: Inspired by Schmidhuber’s Gödel Machine theory—a system that can modify any part of itself (including the modification logic itself) can theoretically find the global optimum.

**Core Idea**: Implementing complete self-reference—the agent can read all of its own code, modify any part (including decision functions and execution functions), and then recursively continue to improve itself.

## Method

### Overall Architecture
Gödel Agent = initial policy $\pi_0$ + self-referential learning algorithm $I_0$. Formalized as $\pi_{t+1}, I_{t+1} = I_t(\pi_t, I_t, r_t, g)$, meaning the optimization algorithm $I$ can modify not only $\pi$ but also itself. The main logic is implemented using recursive functions (rather than loops) and dynamically modifies code at runtime via monkey patching.

### Key Designs

1. **Self-Awareness**:

    - Function: The agent reads all of its own code by inspecting the Python runtime memory (local/global variables)
    - Mechanism: The `SELF_INSPECT()` function extracts all variables, functions, and class definitions of the agent
    - Design Motivation: The prerequisite of self-reference is knowing one's current state, including the code logic

2. **Self-Modification**:

    - Function: The agent generates new code using LLMs and writes it into runtime memory via monkey patching
    - Mechanism: The `self_update` action prompts the LLM to analyze the current policy and feedback, generate modified code, and dynamically replace running functions/classes
    - Design Motivation: Monkey patching allows replacing any function without restarting the program, achieving "modifying oneself while running"

3. **Recursive Self-Improvement**:

    - Function: The main function is implemented using recursion instead of loops, with each deeper recursion applying the latest modifications
    - Mechanism: The `SELF_IMPROVE` function recursively calls itself; within each recursion layer, the agent can modify the `SELF_IMPROVE` function itself, which will be utilized by the next recursion layer
    - Design Motivation: The recursive structure allows modifications of the modification logic to take effect in the subsequent call, which is key to "recursive self-improvement"

4. **Four Initial Actions**:

    - `self_inspect`: Inspects its own code
    - `interact`: Interacts with the environment to obtain feedback $r = U(\mathcal{E}, \pi)$
    - `self_update`: Modifies its own code
    - `continue_improve`: Recursively calls self-improvement
    - The action set itself can also be modified and expanded by the agent

### Auxiliary Designs
- **Think Before Acting**: Allows the agent to reason before taking action, enhancing decision quality
- **Error Handling**: Instead of terminating when a code modification fails, the execution proceeds to the next iteration along with the error traceback
- **Extra Tools**: Executing Python/Bash code, calling LLM APIs

### Loss & Training
- No traditional training—the agent iteratively self-improves on a validation set
- 6 independent cycles per task, with a maximum of 30 iterations per cycle
- The initial policy is CoT, and GPT-3.5 is used for all tests

## Key Experimental Results

### Main Results

| Method | DROP (F1) | MGSM (Acc) | MMLU (Acc) | GPQA (Acc) |
|------|----------|-----------|-----------|-----------|
| CoT | 64.2 | 28.0 | 65.4 | 29.2 |
| CoT-SC | 64.4 | 28.2 | 65.9 | 30.5 |
| Self-Refine | 59.2 | 27.5 | 63.5 | 31.6 |
| LLM Debate | 60.6 | 39.0 | 65.6 | 31.4 |
| Meta Agent Search | 79.4 | 53.4 | 69.6 | 34.6 |
| **Gödel-base (GPT-3.5)** | **80.9** | **64.2** | **70.9** | **34.9** |
| Gödel-free (Unconstrained) | 90.5 | 90.6 | 87.9 | 55.7 |

### Ablation Study (MGSM)

| Configuration | Accuracy |
|------|---------|
| Full | 64.2 |
| w/o think | 50.8 (-13.4) |
| w/o error handling | 49.4 (-14.8) |
| w/o code running | 57.1 (-7.1) |
| w/o LLM calling | 60.4 (-3.8) |

### Key Findings
- **Outperforms Meta Agent Search by 11 percentage points on MGSM** (64.2 vs 53.4), suggesting mathematical reasoning tasks offer larger room for self-improvement
- **Performance explodes in unconstrained mode**: The agent spontaneously requests GPT-4o assistance, elevating GPQA score from 34.9 to 55.7
- **Error Handling is extremely critical**: Performance drops by 14.8 points when removed—since LLM-generated code often contains errors, fault-tolerance mechanisms are essential for sustained optimization
- **Only 14% of the optimization trials ultimately failed**: Although temporary performance degradation occurred in 92% of the trials, they eventually outperformed the initial policy
- **Game of 24 Case Study**: The agent autonomously switched from LLM reasoning to search algorithms, achieving 100% accuracy—completely breaking free from the initial methodology

## Highlights & Insights
- **"The capacity to modify the modification of itself"**—True Meta-Recursion: Unlike meta-learning which only optimizes policies, Gödel Agent can modify the optimizer itself, theoretically approaching the global optimum infinitely
- **Clever Utilization of Monkey Patching**: Elegantly implements the abstract concept of "self-modification" using Python runtime features, rendering it simple and viable from an engineering perspective
- **Insights from the Unconstrained Mode**: The agent autonomously deciding to call stronger models is a "smart" strategy, hinting that future agents need to autonomously manage computing resources

## Limitations & Future Work
- Constrained by the code generation capability of current LLMs—the agent struggles to invent entirely new algorithms that surpass SOTA (e.g., initialized from CoT, it cannot independently invent ToT)
- 4% of the trials terminated unexpectedly (usually because modifying the recursive self-improvement module itself prevented continuation)
- Performance gains in the unconstrained mode mainly stem from calling stronger LLMs rather than algorithmic innovation
- Safety Issues: Self-modifying agents may introduce unpredictable behaviors
- Tested only on GPT-3.5; the self-improvement headroom on stronger base models remains unknown

## Related Work & Insights
- **vs Meta Agent Search**: MAS optimizes agent modules using a fixed meta-search algorithm. Gödel Agent can self-modify even the meta-search algorithm itself, yielding a larger search space
- **vs Self-Refine/Reflexion**: They can only modify outputs, but not their own reasoning logic
- **vs Gödel Machine (Schmidhuber 2003)**: A theoretical pioneer, but the original version requires formal mathematical proofs to carry out self-modification. Gödel Agent substitutes formal proofs with the heuristic capabilities of LLMs

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first fully self-referential LLM agent framework, featuring a highly breakthrough concept
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 benchmarks + detailed ablation studies + Game of 24 case analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical formalization and accurate analogies to the Gödel Machine
- Value: ⭐⭐⭐⭐⭐ Opens up a completely new direction for agent self-improvement

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Agentic Knowledgeable Self-Awareness](agentic_knowledgeable_self-awareness.md)
- [\[ACL 2025\] Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)
- [\[ICLR 2026\] Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine](../../ICLR2026/llm_agent/huxley-godel_machine_human-level_coding_agent_development_by_an_approximation_of.md)
- [\[ICLR 2026\] Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations](../../ICLR2026/llm_agent/presenting_a_paper_is_an_art_self-improvement_aesthetic_agents_for_academic_pres.md)
- [\[ICLR 2026\] Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents](../../ICLR2026/llm_agent/darwin_gödel_machine_open-ended_evolution_of_self-improving_agents.md)

</div>

<!-- RELATED:END -->
