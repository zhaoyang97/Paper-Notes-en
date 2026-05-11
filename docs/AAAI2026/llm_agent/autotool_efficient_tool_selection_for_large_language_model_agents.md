---
title: >-
  [Paper Note] AutoTool: Efficient Tool Selection for Large Language Model Agents
description: >-
  [AAAI 2026][LLM Agent][tool selection] This paper proposes AutoTool, a graph-based tool selection framework that exploits *tool usage inertia* to construct a Tool Inertia Graph (TIG). By leveraging statistical structure…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "tool selection"
  - "LLM agent efficiency"
  - "tool usage inertia"
  - "graph-based planning"
  - "inference cost reduction"
date: 2026-05-08
content_hash: d9bbf70cb1edd2d1
---

# AutoTool: Efficient Tool Selection for Large Language Model Agents

**Conference**: AAAI 2026
**arXiv**: [2511.14650](https://arxiv.org/abs/2511.14650)
**Code**: [GitHub](https://github.com/jiajingyyyyyy/AutoTool)
**Area**: Agent
**Keywords**: tool selection, LLM agent efficiency, tool usage inertia, graph-based planning, inference cost reduction

## TL;DR

This paper proposes AutoTool, a graph-based tool selection framework that exploits *tool usage inertia* to construct a Tool Inertia Graph (TIG). By leveraging statistical structure, AutoTool bypasses redundant LLM inference for tool selection and parameter filling, reducing inference overhead by up to 30% while maintaining task completion rates.

## Background & Motivation

1. **Background**: LLM agents have become powerful tools for automating complex tasks; frameworks such as ReAct drive multi-step decision-making through think–act–observe cycles.
2. **Limitations of Prior Work**: Existing frameworks rely on LLM inference at every step for tool selection, incurring high computational overhead and latency, particularly in multi-step tasks with numerous LLM calls.
3. **Key Challenge**: Not every decision step requires the full reasoning capacity of an LLM; many tool invocations occur in highly patterned contexts, making current approaches an over-reliance on LLM inference.
4. **Key Observation**: The authors identify the phenomenon of **tool usage inertia**—tool invocations follow predictable sequential patterns. For example, in ScienceWorld, `go_to` is followed by `look_around` in 88.7% of cases.
5. **Theoretical Validation**: A $k$-th order Markov chain analysis shows that conditional entropy decreases from 3.50 bits (0th order) to 2.52 bits (1st order) to 1.93 bits (2nd order), with likelihood ratio tests yielding $p < .001$, confirming sequential dependency.
6. **Core Idea**: A graph structure captures statistical regularities in tool invocations; tools are selected directly under high-confidence conditions, with fallback to LLM inference when uncertainty is high.

## Method

### Overall Architecture

AutoTool attempts an *inertia call* before each standard LLM invocation, comprising two stages: ① **Inertia Sensing** predicts the next tool; ② **Parameter Filling** automates parameter assignment. The LLM is bypassed only when both stages succeed; otherwise, the system falls back to standard inference.

### Module 1: Tool Inertia Graph (TIG) Construction

- **Hierarchical Node Structure**: Tool Nodes encode functional descriptions and execution states; each Tool Node embeds a sub-graph of Parameter Nodes.
- **Two Types of Directed Edges**:
  - *Tool Sequence Edges*: connect Tool Nodes and encode sequential dependencies.
  - *Parameter Dependency Edges*: connect Parameter Nodes and model inter-tool data flow.
- **Online Incremental Construction**: the graph is learned dynamically from historical execution trajectories; edge weights are updated via positive/negative reinforcement based on success/failure feedback.
- Edge weights are reinforced only from high-confidence LLM-generated sequences, preventing error propagation from inertia calls.

### Module 2: Inertia Sensing (CIPS Score)

The Comprehensive Inertia Potential Score is defined as:

$$\text{CIPS} = (1-\alpha) \cdot \text{Score}_{\text{freq}} + \alpha \cdot \text{Score}_{\text{ctx}}$$

- **Frequency Score**: historical usage frequency derived from TIG edge weights.
- **Contextual Score**: semantic similarity between the current agent rationale and candidate tool descriptions, computed via SimCSE.
- The parameter filling stage is entered only when $\text{CIPS}(v^*) > \theta_{\text{inertial}}$.

### Module 3: Hierarchical Parameter Filling

A strictly prioritized, LLM-free parameter filling strategy:
1. **Dependency Backtracking**: traverses Parameter Dependency Edges in the TIG to retrieve parameter values from the outputs of preceding tools.
2. **Environment State Matching**: extracts values from key states maintained by the agent (e.g., current location).
3. **Heuristic Filling**: infers values from the current state or task goal.
- If any parameter cannot be resolved, the inertia call is abandoned and the LLM is invoked.

### Safety Constraints

- Inertia calls are capped at 30% of total operations.
- Consecutive inertia calls are prohibited.
- A fault-tolerance mechanism triggers a recovery path upon detection of consecutive tool failures.

## Key Experimental Results

### Main Results: Efficiency Gains (ReAct + AutoTool)

| Dataset | Progress Rate | Token-In Speedup | Token-Out Speedup | LLM Call Speedup |
|--------|--------------|------------------|-------------------|-----------------|
| AlfWorld | 0.531 (↑ vs 0.394) | 1.60× | 2.87× | 1.18× |
| ScienceWorld | 0.708 (≈ 0.716) | 1.30× | 1.41× | 1.31× |
| ToolQuery-Academic | 0.895 (≈ 0.901) | 1.15× | 0.92× | 1.20× |

### Main Results: Efficiency Gains (Reflexion + AutoTool)

| Dataset | Progress Rate | Token-In Speedup | Token-Out Speedup | LLM Call Speedup |
|--------|--------------|------------------|-------------------|-----------------|
| AlfWorld | 0.453 (≈ 0.481) | 1.33× | 1.20× | 1.29× |
| ScienceWorld | 0.712 (≈ 0.730) | 0.93× | 1.20× | 1.28× |
| ToolQuery-Academic | 0.923 (↑ vs 0.917) | 1.33× | 1.19× | 1.26× |

### Overhead Analysis

| Module | Time Overhead |
|--------|--------------|
| Semantic computation (SimCSE embedding + similarity) | 2.7% ± 1.5% of total task time |
| Non-semantic modules (graph construction/search/parsing) | On the order of seconds; negligible |

## Highlights & Insights

- **Discovery and Quantification of Tool Usage Inertia**: This work is the first to systematically validate sequential dependencies in tool invocations using information theory (conditional entropy) and statistical hypothesis testing, providing a rigorous theoretical foundation for the proposed method.
- **Novel LLM Offloading Perspective**: Rather than optimizing the LLM itself, the paper identifies which decisions do not require LLM reasoning, embodying the key insight that not every step demands full inference.
- **Parameter-Level Data Flow Modeling**: Beyond modeling tool sequences, the framework tracks parameter flow across tools to enable automatic parameter filling.
- **Plug-and-Play Compatibility**: AutoTool can directly augment existing frameworks such as ReAct and Reflexion without any fine-tuning.

## Limitations & Future Work

- **Cold Start**: Online graph construction requires accumulated trajectories; inertia predictions are less reliable in early stages.
- The inertia window is fixed at 2; dynamic adjustment is not explored.
- Evaluation is limited to 3 datasets and does not cover more complex real-world API calling scenarios.
- The semantic similarity module (SimCSE) introduces an additional model dependency.
- The 30% inertia call cap is conservative and may limit further efficiency gains.

## Related Work & Insights

- Compared with tool selection methods such as ToolNet and AnyTool, AutoTool is the only approach that simultaneously achieves LLM offloading, inertia sensing, and parameter flow modeling.
- The approach shares conceptual similarity with Agent Workflow Memory (learning from historical interactions), but AutoTool focuses on statistical patterns at the tool level.
- The inertia graph concept is generalizable to other agent scenarios requiring repetitive decision-making, such as code generation and data analysis pipelines.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Tool usage inertia is a novel and well-supported empirical finding; the graph-based design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets with multi-model validation and detailed overhead/sensitivity analyses, though dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, theoretical analysis is rigorous, and figures are informative.
- **Value**: ⭐⭐⭐⭐ Offers a new direction for agent efficiency optimization with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Time, Identity and Consciousness in Language Model Agents](time_identity_and_consciousness_in_language_model_agents.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](../../NeurIPS2025/llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)

</div>

<!-- RELATED:END -->
