---
title: >-
  [Paper Note] ACT: Knowledgeable Agents to Design and Perform Complex Tasks
description: >-
  [ACL 2025][LLM (Other)][Multi-Agent Collaboration] This paper proposes the ACT framework, where multiple LLM agents collaboratively design tasks and acquire structured knowledge descriptions of each other. This allows each agent to not only grasp its own task but also understand how others operate, significantly outperforming existing multi-agent methods in complex tasks such as creative writing and tool use.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Multi-Agent Collaboration"
  - "Knowledge Sharing"
  - "Complex Task Decomposition"
  - "LLM Agent"
  - "Structured Knowledge"
date: 2026-05-08
content_hash: 1a441474f2dca308
---

# ACT: Knowledgeable Agents to Design and Perform Complex Tasks

**Conference**: ACL 2025  
**Code**: None  
**Area**: Others  
**Keywords**: Multi-Agent Collaboration, Knowledge Sharing, Complex Task Decomposition, LLM Agent, Structured Knowledge

## TL;DR

This paper proposes the ACT framework, where multiple LLM agents collaboratively design tasks and acquire structured knowledge descriptions of each other. This allows each agent to not only grasp its own task but also understand how others operate, significantly outperforming existing multi-agent methods in complex tasks such as creative writing and tool use.

## Background & Motivation

**Background**: Large language model (LLM)-based multi-agent systems have become a dominant paradigm for solving complex tasks. Existing works usually decompose a complex task into multiple manageable subtasks, assign them to different specialized agents, and aggregate their outputs. Representative frameworks like AutoGen, MetaGPT, and CAMEL show promising results in code generation, software development, and dialogue systems.

**Limitations of Prior Work**: Existing multi-agent systems suffer from two core limitations: (1) each agent lacks a global understanding of the entire task, only seeing its own allocated subtask fragment without grasping the overall goal and constraints; (2) agents lack an understanding of each other's workflows—not knowing how other agents process their subtasks, what intermediate results they produce, or what strategies they employ. These two shortcomings severely hinder synergistic effects among agents, making seamless integration of subtask outputs difficult.

**Key Challenge**: Complex tasks naturally require tight correlation and consistency among their parts, whereas the paradigm of independent subtask execution inherently leads to information silos and integration difficulties. Existing methods attempt to mitigate this through centralized scheduling or message passing, but these solutions are either overly centralized (where scheduling becomes a bottleneck) or communication-inefficient (large volumes of unstructured messages are hard to utilize effectively).

**Goal**: (1) Enable agents to collaboratively design complex tasks into more comprehensible forms; (2) allow each agent to acquire structured knowledge about how other agents operate; (3) achieve genuine collaborative task execution through interactive knowledge updating.

**Key Insight**: The authors draw inspiration from human team collaboration—efficient teams require members to not only know their own duties but also understand their teammates' capabilities and workflows. By formalizing this "understanding of others' working styles" into structured knowledge descriptions (Knowledge of Others), each agent can proactively adjust its strategy to better coordinate with the team.

**Core Idea**: Let each member in a multi-agent system simultaneously maintain two types of knowledge—their own task knowledge and a structured understanding of others' workflows—and continuously update this knowledge through iterative interactions to achieve a paradigm shift from simple "division of labor" to true "collaboration."

## Method

### Overall Architecture

The workflow of the ACT framework is divided into two main phases: (1) Collaborative task design phase—multiple agents jointly transform complex tasks into structures that are easier to understand and execute, rather than simply having a single scheduler decompose the tasks; (2) Knowledge-enhanced task execution phase—each agent acquires and utilizes structured knowledge about other agents' workflows while executing its own subtasks, continuously refining its own knowledge and execution strategy through multi-round interactions.

### Key Designs

1. **Collaborative Task Design**:

    - Function: Enable multiple agents to jointly participate in complex task decomposition and design, rather than relying on a single scheduler.
    - Mechanism: First, all agents jointly review the global goals of the complex task, with each agent proposing its own perspective on the task structure. Through a negotiation process, the complex task is decomposed into multiple interrelated subtasks with clarified dependencies and interface specifications. Each agent acts as both a designer and a future executor, ensuring that task design matches execution capabilities.
    - Design Motivation: Traditional top-down task decomposition ignores the actual capabilities and preferences of executors. Collaborative design allows each agent to gain a global understanding of the overall task, laying a foundation for subsequent efficient execution.

2. **Knowledge of Others**:

    - Function: Enable each agent to build a structured understanding of other agents' workflows.
    - Mechanism: Define a structured knowledge representation: "a description of how other agents handle their tasks, viewed from the perspective of how this agent solves its own task." Specifically, for Agent A, its knowledge about Agent B is not merely "what B is doing," but "how B's workflow affects A's strategy from the perspective of A's task." This egocentric structured knowledge representation allows the knowledge to be directly used to guide its own task execution.
    - Design Motivation: Unstructured information exchanges (e.g., directly passing raw outputs) have a low signal-to-noise ratio, requiring receivers to filter and comprehend them independently. Structured knowledge descriptions pre-process information into a form directly valuable to the receiver, significantly improving efficiency.

3. **Iterative Knowledge Update and Task Refinement**:

    - Function: Continually update knowledge and improve task execution through multi-round agent interactions.
    - Mechanism: In each iteration, agents first execute the current version of their subtasks to obtain intermediate results, and then share their execution strategies and intermediate results with others. Each agent updates its "knowledge of others" based on the received information and adjusts its own execution strategy accordingly. By referencing the updated structured knowledge, agents effectively integrate each other's intermediate outputs to solve complex tasks collaboratively. This iterative process continues until the strategies of all agents stabilize or a predefined number of rounds is reached.
    - Design Motivation: One-time information exchange is often insufficient—in the first round, agents' understanding of others might be inaccurate. Through iterative updates, knowledge descriptions become increasingly precise, coordination becomes more efficient, and the final integrated output quality improves.

### Loss & Training

ACT is a training-free framework that directly leverages the inference and execution capabilities of existing LLMs. Knowledge representation and updates are implemented through carefully designed prompt templates, and communication protocols between agents are standardized via structured JSON or text formats.

## Key Experimental Results

### Main Results

The paper evaluates the framework on three different types of complex tasks:

| Task Type | Evaluation Metrics | ACT | AutoGen | CAMEL | MetaGPT |
|---|---|---|---|---|---|
| Creative Writing | Quality Score / Coherence | Highest | Medium | Medium | Medium |
| Tool Use | Task Completion Rate | Highest | Medium | Lower | Medium |
| Comprehensive Evaluation | Overall Score | Highest | Second Highest | Medium | Medium |

### Ablation Study

| Configuration | Task Completion Quality | Description |
|---|---|---|
| Full ACT Framework | Best | Collaborative Design + Knowledge of Others + Iterative Update |
| W/o Knowledge of Others | Significant Drop | Degenerates into independent subtask execution |
| W/o Collaborative Design | Moderate Drop | Decreased task decomposition quality |
| W/o Iterative Update (Single Round) | Drop | Inaccurate knowledge |
| Random Task Assignment | Worst | No collaborative behavior |

### Key Findings

- "Knowledge of Others" is the most critical component of ACT—removing it leads to the most significant performance drop, showing that mutual understanding among agents is vital for collaboration.
- Iterative updates typically converge after 2-3 rounds; excessive rounds yield diminishing marginal returns but scale computational cost linearly.
- In creative writing, the advantage of ACT is particularly pronounced—since creative writing requires high coherence and stylistic consistency, "knowledge of others" helps agents unify their creative direction.
- In tool use tasks, collaborative design contributes more—as dependencies in tool invocation must be accurately identified during the design phase.

## Highlights & Insights

- **Egocentric Representation of Knowledge of Others**: Instead of simply sharing "what you are doing," it models "how your actions affect me from my perspective." This self-referential knowledge structure is more efficient than raw message passing because it is filtered and organized within the receiver's context.
- **Collaborative Design vs. Top-Down Decomposition**: Involving executors in task design is a mature practice in software engineering (e.g., Sprint Planning in Agile). This paper systematically introduces this concept into LLM multi-agent frameworks for the first time.
- **Plug-and-play Nature of the Framework**: ACT requires no additional training and can be directly used with any sufficiently capable LLM, significantly lowering deployment barriers.

## Limitations & Future Work

- The framework's effectiveness highly depends on the capabilities of the underlying LLM—if the LLM cannot accurately comprehend and generate structured knowledge descriptions, the framework degenerates.
- As the number of agents increases, the maintenance cost of "knowledge of others" scales quadratically (each agent needs to maintain $N-1$ copies of other agents' knowledge), presenting a scalability bottleneck.
- The evaluation tasks in the paper are relatively limited in scale. Performance in larger-scale, long-horizon complex tasks (e.g., multi-day software development projects) remains to be validated.
- The format of structured knowledge descriptions currently relies on manual prompt design. Future work can explore automated learning of optimal knowledge representation formats.
- Dynamic role allocation could be introduced—allowing agents to re-adjust roles and subtask divisions based on task progress during the iterative process.

## Related Work & Insights

- **vs. AutoGen**: AutoGen uses conversational interactions for agent collaboration but lacks structured knowledge-sharing mechanisms. ACT's "knowledge of others" provides a more efficient information-passing channel.
- **vs. MetaGPT**: MetaGPT shares information through standardized outputs (e.g., documents, UML diagrams), but these standardized artifacts are task-oriented rather than teammate-oriented. ACT's "knowledge of others" is explicitly optimized for collaboration.
- **vs. CAMEL**: CAMEL uses role-playing-driven interaction where each agent follows predefined role behaviors. ACT allows agents to dynamically acquire and utilize knowledge about others, offering stronger adaptability.
- **Connection to Organizational Management Theory**: The design of ACT is highly similar to the concept of "cross-functional teams" in management science—team members require T-shaped skills (deep expertise in their own domain while understanding relevant domains).

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of "knowledge of others" and self-referential knowledge representation are novel designs, offering a new dimension to multi-agent collaboration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across three different task types is comprehensive, and ablation studies clearly demonstrate the contribution of each component.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, and the framework description is detailed (a 30-page paper).
- Value: ⭐⭐⭐⭐ Direct guiding significance for designing collaborative mechanisms in multi-agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] AutoExp: Automatic Experiment Design and Execution by LLMs](autoexp_automatic_experiment_design_and_execution_by_llms.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] Enough Coin Flips Can Make LLMs Act Bayesian](coin_flips_bayesian.md)

</div>

<!-- RELATED:END -->
