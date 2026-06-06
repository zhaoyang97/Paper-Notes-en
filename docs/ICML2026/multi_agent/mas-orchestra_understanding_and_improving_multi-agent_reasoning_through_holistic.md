---
title: >-
  [Paper Note] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks
description: >-
  [ICML 2026][Multi-Agent][Multi-Agent Systems] This work reformulates "automated multi-agent system design" as a one-shot function-calling RL problem that outputs a complete MAS description. It also introduces MASBench to…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Multi-Agent Systems"
  - "Holistic Orchestration"
  - "Function Calling"
  - "GRPO"
  - "MASBench"
date: 2026-05-08
content_hash: c793352471a6a87d
---

# MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks

**Conference**: ICML 2026  
**arXiv**: [2601.14652](https://arxiv.org/abs/2601.14652)  
**Code**: https://github.com/SalesforceAIResearch/MAS-Orchestra (Available)  
**Area**: LLM Agent / Multi-Agent Systems / Reinforcement Learning  
**Keywords**: Multi-Agent Systems, Holistic Orchestration, Function Calling, GRPO, MASBench  

## TL;DR
This work reformulates "automated multi-agent system design" as a one-shot function-calling RL problem that outputs a complete MAS description. It also introduces MASBench to quantify "when MAS truly outperforms single agents" across five dimensions: Depth, Horizon, Breadth, Parallel, and Robustness.

## Background & Motivation

**Background**: Automated multi-agent system (MAS) design has evolved from manual wiring (e.g., fixed topologies like Debate or CoT-SC) toward training-time orchestration. This involves an orchestrator LLM automatically generating sub-agent roles, connectivity, and execution sequences based on the task.

**Limitations of Prior Work**: The authors categorize existing approaches into three issues. First, at the formalization level, most works use "executable code" to describe orchestration (MAS-Zero, AFlow, W4S). The orchestrator must understand or even replicate the internal code of sub-agents, leading to high orchestration costs for complex sub-agents (e.g., multi-turn search agents) and forcing sub-agents to degrade into simple forms like CoT. Second, at the training level, methods either rely on unstable inference-time heuristic searches or use multi-step RL to assemble components incrementally, which suffers from poor long-range credit assignment and error accumulation. Third, the decision of "when to use MAS" remains purely empirical without a quantitative framework.

**Key Challenge**: Sequential multi-step orchestration forces the orchestrator to make locally optimal decisions at each step, which inherently conflicts with the global coordination benefits of MAS. Meanwhile, defining sub-agents at the level of "changing a prompt line" or "switching a backbone" ignores the critical dimensions of tools and workflows that truly distinguish sub-agent capabilities.

**Goal**: (1) Propose a MAS formalization that enables global reasoning for the orchestrator while accommodating complex sub-agents. (2) Provide a controlled benchmark to disentangle the effects of task structure, verification protocols, and orchestrator/sub-agent capabilities on MAS gains.

**Key Insight**: The authors observe that the orchestrator's essential capability is "high-level system design" rather than "replicating sub-agent internal behaviors." By abstracting sub-agents as black-box callable functions (exposing only signatures), the orchestrator can generate a complete system structure in one shot, bypassing the long-range credit assignment issues of sequential RL.

**Core Idea**: MAS is represented as a "one-shot function-calling program" using two primitives: `create_agent` and `create_flow`. The orchestrator is trained via GRPO on end-to-end task rewards to generate the entire system, with an explicit Degree of MAS (DoM) introduced as a user-controllable complexity knob.

## Method

### Overall Architecture
Given a dataset $\mathcal{D}=\{(x_i,y_i)\}$ and a user-specified DoM level $m\in\{\text{LOW},\text{HIGH}\}$, the orchestrator policy samples a complete orchestration $a\sim\pi_\theta(\cdot\mid x,m)$. A deterministic rule parser $f$ سپس translates $a$ into an executable sub-agent calling graph, outputting the final prediction $\hat{y}=f(x,a)$. The key difference from sequential approaches is that the orchestrator only observes task $x$ at step 0 and makes no further incremental decisions; the quality of orchestration is backpropagated solely through the final answer's correctness.

### Key Designs

1. **Holistic Orchestration + Function-Calling Formalization**:
    - **Function**: Enables the orchestrator to output a complete MAS description, including multiple sub-agents and their connectivity topology, within a single forward pass.
    - **Mechanism**: The orchestration space is constrained to two primitives: `create_agent(role, goal, tools, workflow)` to instantiate a goal-oriented sub-agent, and `create_flow(from, to, payload)` to describe information flow. Sub-agents are black-box functions where the orchestrator only sees signatures.
    - **Design Motivation**: Sequential RL suffers from error accumulation and poor credit assignment. Holistic generation aligns RL signals directly with "system-level final returns," leading to stable training and allowing sub-agents to be arbitrarily complex (e.g., DeepResearch).

2. **DoM (Degree of MAS) Explicit Constraints**:
    - **Function**: Users can specify $m$ to enforce constraints on the orchestration space.
    - **Mechanism**: LOW level allows at most one instantiated sub-agent and prohibits explicit inter-agent topologies; HIGH level imposes no constraints.
    - **Design Motivation**: Empirical results show MAS does not benefit all tasks (e.g., sequential math problems). Leaving the "to MAS or not to MAS" decision to user/task priors saves significant ineffective orchestration overhead.

3. **GRPO + Task-level Sparse Rewards**:
    - **Function**: Uses final answer correctness $R(x,y,\hat{y})=\mathbb{1}[\hat{y}=y]$ as the sole signal for end-to-end MAS optimization.
    - **Mechanism**: For each $x$, $K$ candidate orchestrations $\{a_i\}_{i=1}^K\sim\pi_\theta(\cdot\mid x,m)$ are sampled. Group Relative Policy Optimization (GRPO, Shao et al. 2024) is used to construct a clipped policy gradient based on relative advantages within the group.
    - **Design Motivation**: Holistic orchestration only receives feedback at the end. GRPO replaces value models with group comparisons, which naturally fits the "one prompt, multiple candidate systems" training paradigm.

### Loss & Training
The training phase utilizes both controllable synthetic data from MASBench (generated by iGSM across Depth/Horizon/Breadth/Parallel/Robustness axes) and training sets from public benchmarks (DeepScaleR for AIME/GPQA, HotpotQA, and BrowseComp+). The sub-agent pool is fixed to five types: CoT, CoT-SC, Debate, Self-refine, and DeepResearch, all using the same LLM backend but differing in tools and workflows.

## Key Experimental Results

### Main Results

Using Qwen2.5-7B-Instruct as the orchestrator and GPT-OSS-120B (low) as the sub-agent backend, Ours is compared against standard standalone agents, SoTA inference-time orchestrators (AFlow, MaAS), and training-time orchestrators (ToolOrchestra):

| Benchmark | Task Type | Best Standalone Agent | SoTA Orchestration Baseline | MAS-Orchestra | Notes |
|-----------|-----------|-----------------------|-----------------------------|---------------|-------|
| AIME24 | Math (IID) | DebateAgent 62.08 | AFlow 62.50 | **66.25** | Low DoM |
| AIME25 | Math (IID) | DebateAgent 57.50 | AFlow 53.33 | **61.25** | Low DoM |
| HotpotQA | Multi-hop QA | DeepResearch 46.44 | ToolOrchestra 37.44 | **49.00** | High DoM |
| BrowseComp+ | Search QA | DeepResearch 8.56 | ToolOrchestra 1.38 | **11.00** | High DoM |
| GPQA | Reasoning (OOD) | DebateAgent 64.14 | AFlow 65.43 | **65.21** | Low DoM |

In terms of efficiency, MAS-Orchestra lies on the Pareto frontier, achieving over 10× inference cost savings compared to strong baselines.

### Ablation Study: MASBench Five-Axis Analysis

| Configuration | Conclusion | Explanation |
|---------------|------------|-------------|
| Sub-agent = Qwen-7B (Weak) | MAS beats SAS on Breadth/Parallel/Robustness; SAS beats MAS on Depth | Sequential CoT saves coordination overhead on strong dependency chains. |
| Sub-agent = GPT-120B (Strong)| MAS gain near zero on Depth/Horizon/Breadth/Parallel; MAS leads on Robustness | Strong sub-agents internalize structure; coordination costs outweigh benefits. |
| Orchestrator = RLM (e.g., GPT-OSS-20B) | Underperforms Instruction-tuned LLM | RLMs tend to solve tasks themselves instead of designing systems. |
| Robustness (Adversarial notes) | SAS accuracy near 0; MAS significantly leads | MAS actively adds final answer/moderator agents for cross-verification. |

### Key Findings
- **"Edge Capability" Hypothesis**: Maximum MAS gains occur when sub-agents are "competent but not strong enough to internalize the entire structure." If too weak, sub-agents fail sub-tasks; if too strong, coordination costs negate benefits.
- **Holistic vs. Sequential**: Ours outperforms ToolOrchestra across all benchmarks. On BrowseComp+, the jump from 1.38 to 11.00 demonstrates that sequential RL is particularly disadvantaged when sub-agents are complex.
- **Counter-intuitive Orchestrator Choice**: Zero-shot GPT-5/Claude-4.5 as orchestrators are outperformed by a trained 7B Qwen, indicating that orchestration capability must be explicitly shaped via RL rather than emerging from general reasoning.
- **DoM Configuration Strategy**: Tasks with strong sequential dependencies (Math/GPQA) favor LOW DoM, while tasks with parallel search (HotpotQA) favor HIGH DoM.

## Highlights & Insights
- **"Function is the correct abstraction for MAS"**: By abstracting to signatures, the orchestrator is no longer hindered by sub-agent complexity. This allows complex agents like DeepResearch to be naturally integrated.
- **Falsifiable MAS Research**: The five-axis decomposition of MASBench allows researchers to report exactly where MAS helps (e.g., "improves Robustness by $X\%$") rather than just aggregate numbers.
- **RLMs are not natural orchestrators**: RLM's end-to-end training target makes it prefer direct task-solving over delegation. This serves as a warning for works attempting to use reasoning models like o1 as orchestrators without fine-tuning.

## Limitations & Future Work
- The sub-agent pool is still restricted to fixed workflows; the capability to "invent" entirely new sub-agent types was not verified.
- Capability binding: There is an implicit coupling between the orchestrator and the sub-agent reasoning effort (context length management).
- The Comparison with GPT-5/Claude-4.5 is somewhat unfair as they were not fine-tuned for this specific orchestration task.
- MASBench relies heavily on synthetic iGSM data; lack of diverse use cases like code generation or long-document analysis.

## Related Work & Insights
- **vs. AFlow / MAS-Zero**: They use inference-time heuristic search. MAS-Orchestra uses GRPO for explicit training, achieving 10× better efficiency and better performance via DoM parameterization.
- **vs. ToolOrchestra**: Uses sequential RL. Sequential credit assignment is a significant bottleneck in MAS design for complex tasks.
- **vs. MAS-GPT**: SFT-based orchestration. RL with task rewards better captures system-level coordination patterns and demonstrates more stable OOD generalization.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](../../ACL2026/multi_agent/towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](../../ACL2026/multi_agent/towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[ICML 2026\] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)

</div>

<!-- RELATED:END -->
