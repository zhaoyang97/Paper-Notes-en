---
title: >-
  [Paper Note] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning
description: >-
  [AAAI 2026][LLM Agent][Tool-augmented LLM] This paper proposes a Planner-centric Plan-Execute framework that transforms complex queries into DAG-based execution plans. Through two-stage SFT+GRPO training of a dedicated P…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "Tool-augmented LLM"
  - "DAG planning"
  - "Plan-Execute paradigm"
  - "GRPO reinforcement learning"
  - "multi-tool orchestration"
date: 2026-05-08
content_hash: 4f0b3a844d10d869
---

# Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning

**Conference**: AAAI 2026
**arXiv**: [2511.10037](https://arxiv.org/abs/2511.10037)
**Code**: [https://github.com/weixiaolong94-hub/Beyond-React](https://github.com/weixiaolong94-hub/Beyond-React)
**Area**: Agent / LLM
**Keywords**: Tool-augmented LLM, DAG planning, Plan-Execute paradigm, GRPO reinforcement learning, multi-tool orchestration

## TL;DR
This paper proposes a Planner-centric Plan-Execute framework that transforms complex queries into DAG-based execution plans. Through two-stage SFT+GRPO training of a dedicated Planner model, the approach surpasses reactive methods such as ReAct on ComplexTool-Plan and StableToolBench, achieving higher success rates with fewer inference steps.

## Background & Motivation
Current tool-augmented LLMs rely primarily on **reactive** frameworks such as ReAct, which make decisions and execute actions incrementally. While viable for simple queries, this paradigm exhibits a fundamental deficiency on complex multi-tool composition tasks: the **local optimum trap**. Each decision attends only to the current state, lacking global planning capacity and failing to exploit inherent parallelism in task execution. Search-based methods such as Tree-of-Thought offer partial improvements but fundamentally still seek optimal sequential paths, incurring high computational overhead while ignoring parallelism.

Core limitations:
1. ReAct's incremental decision-making cannot model complex inter-tool dependencies.
2. Existing methods lack large-scale, structured training data for complex planning.
3. Evaluating planning quality is itself a non-trivial problem.

## Core Problem
How can an LLM **generate a globally optimal execution plan in a single pass** for queries requiring multi-tool collaboration, rather than proceeding through trial and error? This paper casts the problem as: given query $Q$ and tool set $T$, learn a policy $\pi$ that maps them to a DAG-structured execution plan $G=(V,E)$.

## Method

### Overall Architecture
The framework consists of two decoupled phases—**Planning** and **Execution**:
1. The **Planner** receives the user query and available tool set, then generates a DAG execution plan (nodes = tools, edges = data dependencies).
2. The **Executor** (e.g., GPT-4o) executes tool calls in topological order of the DAG, supporting parallel execution of independent nodes.

Training pipeline: construct the ComplexTool-Plan dataset → SFT cold start → GRPO reinforcement fine-tuning.

### Key Designs

1. **ComplexTool-Plan Dataset Construction (three-stage automated pipeline)**:

    - **Workflow generation**: DeepSeek-V3 samples subsets from 4,535 ModelScope tool APIs and generates structurally complex DAG execution plans.
    - **Query reverse engineering**: DeepSeek-V3 back-generates natural language queries from DAGs, effectively "writing problems from answers."
    - **Intent analysis and re-planning**: A teacher model re-plans the DAG from the generated query alone, ensuring high fidelity of $(Q, G)$ pairs—samples where the query is too ambiguous to recover the original DAG are filtered out.

   The final dataset contains 3,000 SFT instances across Easy/Medium/Hard difficulty levels. Higher difficulty implies a larger tool pool and more tools required per task.

2. **DAG as a structured execution plan representation**:

    - Nodes $V \subseteq T$ represent selected tools.
    - Directed edges $E$ represent data dependencies.
    - Parallel execution is supported: nodes without dependencies can be invoked simultaneously.
    - More expressive than linear sequences, enabling modeling of complex branching and merging logic.

3. **RL training set refinement**:

    - The SFT model is used to filter training data: instances the model already solves reliably (no learning signal) and those it completely fails on (too difficult) are removed.
    - The 787 retained high-variance instances focus training on the model's capability frontier, preventing policy degradation.

### Loss & Training

**Two-stage training**:

**Stage 1: SFT Cold Start**
NLL loss minimization on the Qwen3 series (0.6B/1.7B/4B/8B):
$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(Q,G_{gt})\sim D_{\text{train}}}[\log P(G_{gt}|Q,T;\theta)]$$

**Stage 2: GRPO Reinforcement Learning**
A hierarchical reward function $R(y)$ evaluates plan quality in strict priority order:

| Level | Check | Reward/Penalty |
|-------|-------|----------------|
| Level 1 | Syntax error (non-JSON format) | −10.0 |
| Level 2 | Cycle present (non-DAG) | −10.0 |
| Level 3 | Connectivity defect (isolated node) | −2.0 |
| Level 4 | Edge F1 score | $5 \times$ Edge F1 |
| Level 5 | Perfect match bonus | +5.0 |

The reward range is $[-10.0,\ +10.0]$ with a fail-fast design: high-level penalties immediately terminate evaluation. The hierarchical design elegantly distinguishes **structural errors** (fatal) from **policy errors** (improvable), providing multi-dimensional gradient signals to the model.

## Key Experimental Results

### ComplexTool-Plan Planning Quality (Easy Set)

| Method | Node F1 | Edge F1 | DAG EM |
|--------|---------|---------|--------|
| GPT-4o | 0.929 | 0.779 | 0.635 |
| Claude-3.7 | 0.949 | 0.815 | 0.644 |
| DeepSeek-V3 | 0.770 | 0.643 | 0.511 |
| Qwen3-0.6B (SFT) | 0.968 | 0.848 | 0.671 |
| Qwen3-1.7B (SFT+RL) | 0.979 | 0.879 | 0.756 |
| Qwen3-8B (SFT+RL) | **0.984** | **0.906** | **0.803** |

### ComplexTool-Plan Planning Quality (Hard Set)

| Method | Node F1 | Edge F1 | DAG EM |
|--------|---------|---------|--------|
| GPT-4o | 0.856 | 0.464 | 0.098 |
| Claude-3.7 | 0.897 | 0.491 | 0.106 |
| Qwen3-8B (SFT) | 0.910 | 0.657 | 0.295 |
| Qwen3-8B (SFT+RL) | **0.904** | **0.659** | **0.319** |

**Key finding**: DAG EM collapses across all models on the Hard set—GPT-4o reaches only 0.098, whereas Qwen3-8B (SFT+RL) achieves 0.319—demonstrating that a specially trained small model substantially outperforms general-purpose large models.

### StableToolBench End-to-End Execution

| Method | Avg. SoPR | Avg. SoWR |
|--------|-----------|-----------|
| GPT-3.5 (ReAct) | 47.9 | — |
| GPT-4 (ReAct) | 48.2 | 58.7 |
| GPT-4 (DFSDT) | 70.3 | 64.2 |
| ToolLLaMA (DFSDT) | 54.2 | 47.1 |
| LLMCompiler | 36.2 | 37.9 |
| Qwen3-8B (RL) + GPT-4o | **59.8** | **55.0** |

Average inference steps: the proposed method requires only **2.29 steps**, significantly fewer than DTA-Llama (2.48) and GPT-4 ReAct (3.27–4.23).

### Ablation Study
- **SFT → SFT+RL**: DAG EM for Qwen3-8B improves from 0.781 → 0.803 on the Easy set and from 0.295 → 0.319 on the Hard set (+8.1% relative gain). RL primarily improves edge prediction (dependency modeling) rather than node selection (tool selection).
- **Model scale effect**: From Easy to Hard, the 1.7B model suffers a 71.2% accuracy drop (0.756 → 0.218), while the 8B model drops only 60.3% (0.803 → 0.319), indicating stronger robustness to complexity at larger scale.
- **RL training instability at 0.6B**: Insufficient model capacity leads to reward hacking—the model learns simple strategies to avoid penalties rather than genuinely solving tasks, revealing a minimum capacity requirement for stable RL training.
- **SoPR vs. iterative methods**: Iterative methods such as DTA-Llama achieve higher SoPR by leveraging multi-round error correction; the proposed single-pass planning paradigm is more efficient (fewest steps) but lacks a correction mechanism.

## Highlights & Insights
- The **DAG-as-plan** formulation is intuitive and effective: decomposing complex tasks into tool nodes and dependency edges naturally supports parallel execution and is more expressive than linear chains.
- The **reverse-engineering data construction pipeline** is elegant: workflows are generated first, queries are back-synthesized, and re-planning serves as a quality filter—solving the bootstrapping problem of obtaining ground-truth planning data.
- The **hierarchical reward function** is well-designed: fail-fast evaluation, separation of structural vs. policy errors, and continuous F1 scoring provide rich gradient signals for RL.
- The **Plan-Execute decoupled architecture** allows the Planner and Executor to be upgraded independently, offering engineering flexibility.
- Filtering the RL training set using the SFT model draws on self-play principles, avoiding wasted training resources on trivially easy or intractably hard instances.

## Limitations & Future Work
- **No error correction in single-pass planning**: The plan-then-execute paradigm offers no opportunity to revise a flawed plan, representing a core disadvantage compared to iterative methods (e.g., DTA-Llama, Reflexion). Real-world queries may be ambiguous, making the single-pass planning assumption overly strong.
- **Hard set DAG EM remains low**: Even the best model achieves only 0.319, indicating that complex planning is far from solved.
- **Dependence on external Executor quality**: End-to-end performance is highly sensitive to the Executor (GPT-4o), and the Planner's planning quality alone cannot guarantee end-to-end success.
- **Dataset bias in ComplexTool-Plan**: Training data generated by DeepSeek-V3 may inherit its planning preferences and blind spots, raising generalization concerns.
- **Evaluation limited to StableToolBench**: Despite being a mainstream benchmark, its API simulator and caching mechanism differ from real API environments.
- **Absence of comparisons with recent Agent frameworks**: e.g., AutoGen, CrewAI, and OpenAI's native function calling support.
- Potential improvement: introducing a **lightweight re-planning** mechanism (local re-planning upon tool call failure during execution) to balance efficiency and robustness.

## Related Work & Insights
1. **vs. ReAct**: ReAct is a step-by-step reactive framework operating in a think-act-observe cycle. This paper argues that such incremental decision-making inherently leads to local optima, and that complex tasks require global planning. Experimentally, GPT-4 (ReAct) achieves only 48.2% SoPR, far below the proposed method's 59.8%.
2. **vs. LLMCompiler**: LLMCompiler also supports parallel tool invocation but performs localized parallelization within the ReAct framework. The fundamental distinction here is elevating planning to a dedicated stage with a specially trained model for global DAG generation.
3. **vs. DTA-Llama / iterative methods**: Iterative methods may achieve higher SoPR through multi-round execute-reflect-retry cycles. The proposed method is non-iterative, performing planning only once; its advantages are efficiency (fewest inference steps) and predictability, at the cost of lacking error correction.

## Highlights & Insights (Transfer Value)
- The **DAG planning + RL training** paradigm is transferable to other domains requiring complex workflow orchestration (e.g., multimodal tasks, scientific experiment automation).
- The hierarchical reward design pattern (structural check → semantic check → quality scoring) applies to any RL setting requiring evaluation of structured outputs.
- The "forward generation + backward validation" data construction pipeline is reusable for other tasks requiring bootstrapped training data.
- **A worthwhile open question**: can a hybrid paradigm be designed—applying single-pass planning for simple subtasks while retaining reflection opportunities for uncertain ones?

## Rating
- Novelty: ⭐⭐⭐⭐ The DAG planning + GRPO combination is novel, though plan-execute decoupling per se is not new (cf. PAL, PoT); the core contribution leans toward engineering implementation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both planning quality and end-to-end dimensions with detailed ablations; comparisons with recent Agent frameworks are missing, and Hard set results indicate the problem is far from solved.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, precise problem formulation, and well-designed figures; the Related Work section is citation-dense, slightly reducing readability.
- Value: ⭐⭐⭐⭐ Provides practical guidance for Agent planning research; the data construction pipeline and hierarchical reward design are reusable; practical value is somewhat reduced by the single-pass planning limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[ICLR 2026\] PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement](../../ICLR2026/llm_agent/physcensis_physics-augmented_llm_agents_for_complex_physical_scene_arrangement.md)
- [\[AAAI 2026\] Automating Complex Document Workflows via Stepwise and Rollback-Enabled Operations](automating_complex_document_workflows_via_stepwise_and_rollback-enabled_operatio.md)
- [\[ACL 2026\] Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Identification](../../ACL2026/llm_agent/spec-o3_a_tool-augmented_vision-language_agent_for_rare_celestial_object_candida.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)

</div>

<!-- RELATED:END -->
