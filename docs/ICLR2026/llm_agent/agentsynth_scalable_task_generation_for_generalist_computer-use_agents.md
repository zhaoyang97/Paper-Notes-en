---
title: >-
  [Paper Note] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents
description: >-
  [ICLR 2026][LLM Agent][synthetic data generation] This paper proposes AgentSynth, a pipeline that leverages information asymmetry (forward stepwise generation is easy; backward holistic solving is hard) to chain simple subtasks into complex long-horizon computer-use tasks. It automatically generates 6,000+ diverse tasks and trajectories at \$0.60 per trajectory, with SOTA agents achieving only 4% success rate at the highest difficulty level.
tags:
  - ICLR 2026
  - LLM Agent
  - synthetic data generation
  - computer-use agents
  - information asymmetry
  - task chaining
  - long-horizon task benchmark
date: 2026-05-08
content_hash: ccd19d360d2de997
---

# AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents

**Conference**: ICLR 2026
**arXiv**: [2506.14205](https://arxiv.org/abs/2506.14205)
**Code**: [https://github.com/sunblaze-ucb/AgentSynth](https://github.com/sunblaze-ucb/AgentSynth)
**Area**: LLM Agent
**Keywords**: synthetic data generation, computer-use agents, information asymmetry, task chaining, long-horizon task benchmark

## TL;DR
This paper proposes AgentSynth, a pipeline that leverages information asymmetry (forward stepwise generation is easy; backward holistic solving is hard) to chain simple subtasks into complex long-horizon computer-use tasks. It automatically generates 6,000+ diverse tasks and trajectories at \$0.60 per trajectory, with SOTA agents achieving only 4% success rate at the highest difficulty level.

## Background & Motivation

**State of the Field**: LLM agents are advancing rapidly in computer-use tasks (web navigation, desktop operations), yet high-quality training and evaluation data remain heavily dependent on human annotation.

**Limitations of Prior Work**: (a) Human annotation is prohibitively expensive (e.g., TheAgentCompany requires 17 hours/\$34–425 per task); (b) Human-annotated data has limited diversity and struggles to cover the full complexity of real-world computer-use scenarios; (c) Synthetic data pipelines face two core challenges—current LLM agents cannot reliably generate trajectories for complex tasks, and naive generation strategies yield insufficient diversity.

**Root Cause**: High-quality agent data requires tasks that are both complex and diverse, yet LLM agents can only reliably complete simple tasks. How can this tension be resolved?

**Paper Goals**: Design a low-cost, high-diversity, fully automated pipeline that generates realistic computer-use tasks of controllable difficulty along with corresponding trajectories.

**Starting Point**: Exploit **information asymmetry**—solving tasks forward step by step (each step requiring only one simple subtask) is far easier than reasoning through the entire solution from scratch. The pipeline has an agent generate subtasks in the forward direction while collecting trajectories, then uses a summarizer to merge them into a single high-level composite task.

**Core Idea**: Decompose complex tasks into a sequence of simple subtasks for forward generation, then merge them backward into an apparently unified long-horizon task—easy to generate, hard to solve.

## Method

### Overall Architecture
AgentSynth operates within the OSWorld virtual desktop environment and consists of six LLM-based agents working collaboratively: Task Proposer (initial task) → Task Executor (execution) → Task Verifier (verification) → Task Reviser (correction on failure) → Follow-up Task Proposer (subsequent subtasks) → Task Summarizer (merging into a high-level task). After iteratively generating $n$ subtasks, the summarizer aggregates the first $1$ through $n$ subtasks separately, naturally yielding task variants at difficulty levels 1 through $n$.

### Key Designs

1. **Information-Asymmetry-Driven Task Construction**

    - Function: Exploits the asymmetry between "easy forward stepwise execution" and "hard backward holistic solving" to construct challenging tasks.
    - Mechanism: Each subtask is a simple operation in the current desktop state (completable in a few atomic actions), allowing the agent to execute it reliably and collect trajectories. However, the high-level task description obtained by merging multiple subtasks omits intermediate step information, requiring the evaluation-time agent to reason through the entire solution path from scratch.
    - Design Motivation: Resolves the contradiction between "agents cannot reliably complete complex tasks" and "data requires complex tasks"—the generation side uses simple tasks to ensure trajectory quality, while the evaluation side uses composite tasks to ensure challenge.

2. **Six-Agent Collaborative Pipeline**

    - Function: End-to-end automated task generation, execution, verification, and iteration.
    - Mechanism:
        - **Task Proposer**: Generates an initial task based on a random persona and a desktop screenshot.
        - **Task Executor**: Uses GPT-4.1 for planning and computer-use-preview for precise coordinate-level operations (a two-stage architecture separating high-level reasoning from low-level execution).
        - **Task Verifier**: Follows a WebJudge-style architecture—extracts key requirements from the task description, selects keyframes from the screenshot sequence, and determines success/failure and completion percentage.
        - **Task Reviser**: When a task is partially completed, revises the task description to match what was actually accomplished.
        - **Follow-up Proposer**: Generates the next logically coherent subtask based on the history of prior subtasks and the current screenshot.
        - **Task Summarizer**: Abstracts the subtask sequence into a single high-level task description; varying the number of subtasks controls difficulty.
    - Design Motivation: The two-stage executor design (GPT-4.1 for planning + computer-use model for execution) leverages the complementary strengths of each model for high-level reasoning and low-level operation respectively.

3. **Controllable Task Difficulty**

    - Function: Generates tasks at varying difficulty levels by varying the number of merged subtasks (1 to 6).
    - Mechanism: Level $k$ corresponds to merging the first $k$ subtasks. Level 1 averages 5 steps/1.2 apps/2-step memory span; Level 6 averages 45 steps/3.3 apps/18-step memory span/4.3 app switches.
    - Design Motivation: Existing benchmarks lack systematic difficulty control, making it difficult to precisely identify agent capability boundaries.

### Safety Measures
- Operations involving login credentials and email sending are prohibited.
- All tasks are executed within virtual machines.

## Key Experimental Results

### Main Results
Multiple SOTA agents are evaluated on the generated benchmark (success rate %):

| Agent Model | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | Level 6 |
|---|---|---|---|---|---|---|
| SOTA Range | ~18% | ~14% | ~10% | ~8% | ~6% | ~4% |

Success rate drops sharply from ~18% at Level 1 to ~4% at Level 6, demonstrating the benchmark's discriminative power and challenge.

### Quality Evaluation (100 Human-Annotated Samples)

| Quality Metric | Pass Rate |
|---|---|
| Feasibility and Realism | 91% |
| Subtask Coherence | 90% |
| Persona Relevance | 94% |
| Verifier Accuracy | 88% |

### Cost Comparison

| Framework | Typical Steps | Human Hours/Task | Cost/Task |
|---|---|---|---|
| τ-bench | 20–30 | 2h | \$4–50 |
| OSWorld | 10–15 | 4.4h | \$8.8–110 |
| TheAgentCompany | 30–40 | 17h | \$34–425 |
| **AgentSynth** | **40–60** | **N/A** | **\$0.60** |

### Key Findings
- The information asymmetry principle is validated: the same subtask sequence yields high forward generation success rates (subtask verification passes), but the merged composite task is extremely difficult to solve in reverse (only 4% at Level 6).
- Over 60% of trajectories involve 2+ applications, and over 40% involve 3+ applications, authentically reflecting the complexity of cross-application coordination.
- The verifier performs well under adversarial testing: near-miss false acceptance rate is only 12%, while the benign correct acceptance rate is 94%.
- Task diversity spans office work, information retrieval, entertainment, programming, research, and other domains.

## Highlights & Insights
- **Information asymmetry** as a core design principle for synthetic data is an elegant insight: from a cognitive psychology perspective, "sequential execution" and "planning from scratch" genuinely impose fundamentally different cognitive loads—this paper systematizes that intuition into a data generation methodology.
- The cost contrast of **\$0.60 vs. \$34–425** is striking, representing a genuine solution for scalable agent data production.
- The **controllable difficulty** design makes AgentSynth not only a benchmark but also a training data source—data at specific difficulty levels can be generated on demand for curriculum learning.
- The two-stage executor design (GPT-4.1 planner + computer-use executor) is worth referencing in future work.

## Limitations & Future Work
- Task generation currently relies on GPT-4.1; different models may introduce systematic biases in complexity and realism—the authors acknowledge this as an open question.
- The verifier still has a 12% near-miss false acceptance rate, which may introduce noise into training data.
- Validation is limited to OSWorld (Ubuntu desktop); transferability to Windows/macOS environments is unknown.
- Logical coherence between subtasks is ensured by the LLM and may occasionally yield unnatural task combinations.
- The paper lacks downstream performance evaluation from training agents on the generated data—the ultimate value of the pipeline needs to be validated through training outcomes.

## Related Work & Insights
- **vs. OS-Genesis/Learn-by-interact**: Those approaches retroactively define tasks after executing trajectories; AgentSynth first defines subtasks and then composes them into composite tasks, affording stronger control over task quality.
- **vs. Evol-Instruct**: Generates trajectories only for final instructions, with no subtask chaining mechanism.
- **vs. WorkArena compositional**: Uses predefined atomic task combinations; AgentSynth's subtasks are dynamically generated by an LLM, yielding higher diversity.
- **vs. Human-annotated benchmarks (OSWorld/TheAgentCompany)**: Comparable quality at 50–700× lower cost.

## Rating
- Novelty: ⭐⭐⭐⭐ — The application of information asymmetry to agent data synthesis is the core innovation; the six-agent pipeline design is conceptually clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Human quality validation, adversarial testing, cost comparison, and difficulty gradient analysis are all rigorous, but downstream training evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-structured, with detailed explanation of each pipeline component and highly informative figures and tables.
- Value: ⭐⭐⭐⭐⭐ — Represents an infrastructure-level contribution to the agent community—a genuinely scalable solution for high-quality data generation.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Efficient Agent Training for Computer Use](efficient_agent_training_for_computer_use.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](../../CVPR2026/llm_agent/carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [\[ICLR 2026\] PerfGuard: A Performance-Aware Agent for Visual Content Generation](radiometrically_consistent_gaussian_surfels_for_inverse_rendering.md)

<!-- RELATED:END -->
