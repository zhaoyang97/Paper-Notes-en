---
title: >-
  [Paper Note] What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities
description: >-
  [ICML 2025 (Oral)][Multimodal VLM][virtual agent] This paper proposes OmniBench, a graph-based, scalable virtual agent benchmark. By synthesizing tasks of controllable complexity through an automated pipeline, combined with the OmniEval multi-dimensional evaluation framework, it generates 36K tasks across 20 application scenarios, systematically revealing the weaknesses of virtual agents across different capability dimensions.
tags:
  - "ICML 2025 (Oral)"
  - "Multimodal VLM"
  - "virtual agent"
  - "benchmark"
  - "graph-based tasks"
  - "multi-dimensional evaluation"
  - "automated pipeline"
date: 2026-05-08
content_hash: a3fec8b61c4ef095
---

# What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities

**Conference**: ICML 2025 (Oral)  
**arXiv**: [2506.08933](https://arxiv.org/abs/2506.08933)  
**Code**: [https://omni-bench.github.io/](https://omni-bench.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: virtual agent, benchmark, graph-based tasks, multi-dimensional evaluation, automated pipeline

## TL;DR
This paper proposes OmniBench, a graph-based, scalable virtual agent benchmark. By synthesizing tasks of controllable complexity through an automated pipeline, combined with the OmniEval multi-dimensional evaluation framework, it generates 36K tasks across 20 application scenarios, systematically revealing the weaknesses of virtual agents across different capability dimensions.

## Background & Motivation

**Background**: Multimodal Large Language Model (MLLM)-based virtual agents have made significant progress in recent years, capable of autonomously completing user interface execution tasks on platforms like mobile phones and computers. Benchmarks for evaluating these agents (e.g., AITW, OSWorld) are increasingly growing in number.

**Limitations of Prior Work**: Existing benchmarks face three major limitations:
   - **Uncontrollable task complexity**: Manually annotated tasks have highly uneven complexity levels, making it difficult to systematically analyze the impact of complexity on performance.
   - **Limited scale and diversity**: High manual annotation costs lead to insufficient scenario coverage (typically containing only a few hundred tasks).
   - **Lack of multi-dimensional evaluation**: They focus solely on overall task success rates, failing to diagnose the agent's shortfalls in specific capability dimensions (such as spatial reasoning or long-term planning).

**Key Challenge**: How can large-scale, multi-scenario, and multi-dimensional agent evaluation be achieved while maintaining task quality? Manual annotation offers high quality but lacks scalability, whereas automatic generation scales well but struggles with quality control.

**Goal**: To construct a benchmark that is both scalable and capable of multi-dimensional evaluation of virtual agent capabilities.

**Key Insight**: Representing tasks using graph structures—where each node represents an atomic operation (subtask) and edges represent dependencies. Composite tasks of controllable complexity are automatically synthesized through combinations of subtasks.

**Core Idea**: Modeling agent tasks as Directed Acyclic Graphs (DAGs), where task complexity is automatically controlled via the graph's topology, and evaluations of various capability dimensions are naturally introduced through the graph's node types.

## Method

### Overall Architecture
Input: An atomic operation library of 20 predefined scenarios  
Output: 36K graph-structured tasks + multi-dimensional evaluation results  

Pipeline:
1. **Atomic Operation Definition**: Define basic operations (such as "click button", "input text", "scroll screen") for each scenario.
2. **Graph Synthesis**: Automatically combine atomic operations into graph-structured tasks, controlling complexity through the graph's depth, width, and branching factor.
3. **Environment Instantiation**: Instantiate the abstract graph tasks into specific application scenarios.
4. **Multi-Dimensional Evaluation**: Evaluate the agent across multiple dimensions using OmniEval.

### Key Designs

1. **Graph-Based Task Synthesis**:

    - **Function**: Automatically generates tasks with controllable complexity.
    - **Mechanism**: Task = DAG $G = (V, E)$, where $V$ is the set of atomic operations and $E$ represents dependencies. The topological features of the graph determine task complexity:
        - **Depth**: The longest sequence of operations, reflecting long-term planning demands.
        - **Width**: The maximum number of parallel operations, reflecting multi-goal management capability.
        - **Branching**: Conditional branching, reflecting decision-making and reasoning capabilities.
    - Key formula: Task complexity $C(G) = f(\text{depth}, \text{width}, \text{branching})$
    - **Design Motivation**: The graph structure provides a flexible and interpretable task representation, allowing precise control over each dimension of complexity.

2. **Cross-Platform Automated Pipeline**:

    - **Function**: Automatically translates abstract graph tasks into concrete tasks on different platforms (Android, Web, Desktop).
    - **Mechanism**: Each atomic operation corresponds to a platform-specific UI interaction template. Executable tasks are generated through template instantiation and environment snapshots.
    - **Design Motivation**: Cross-platform capability ensures that the benchmark is not confined to a specific application ecosystem.

3. **Multi-Dimensional Evaluation Framework**:

    - **Function**: Evaluates agents across 10 capability dimensions.
    - Evaluation dimensions include:
        - **Basic Actions**: Accuracy of clicks/inputs/swipes.
        - **Spatial Perception**: UI element localization and layout understanding.
        - **Logical Reasoning**: Conditional judgments and branch selection.
        - **Long-Term Planning**: Path planning for multi-step tasks.
        - **Error Recovery**: Correction capability after execution failures.
        - **Multimodal Understanding**: Semantic understanding of icons/images.
        - and others, totaling 10 dimensions.
    - **Mechanism**: Each subtask is labeled with its corresponding capability dimension, which is then aggregated into dimension-level scores through subtask-level evaluations.
    - **Design Motivation**: Relying solely on end-to-end success rates fails to diagnose the specific weaknesses of agents.

### Loss & Training
Fine-tuning agents on the graph-structured data of OmniBench (for training experiments). Standard behavior cloning loss is used.

## Key Experimental Results

### Main Results

| Model | Overall Success Rate | Subtask Accuracy | Logical Reasoning | Long-Term Planning | Error Recovery |
|------|-----------|-----------|---------|---------|---------|
| GPT-4o | **61.2%** | **73.8%** | **68.5%** | 51.3% | 42.1% |
| Claude 3.5 | 58.7% | 71.2% | 65.3% | **53.8%** | 39.6% |
| Qwen2-VL-72B | 47.3% | 62.4% | 54.1% | 41.2% | 35.8% |
| CogAgent | 42.8% | 58.1% | 48.7% | 36.5% | **43.2%** |
| Human | 95.2% | 97.8% | 98.1% | 93.4% | 91.7% |

### Ablation Study

| Configuration | Training Data Efficiency | Explanation |
|------|-----------|------|
| Graph-Structured Data Training | **+15.3%** Success Rate | Significant improvement compared to flat data |
| Manually Annotated Data Training | +12.1% Success Rate | High quality but limited quantity |
| Flat Synthetic Data Training | +8.7% Success Rate | Lacks structural information |
| No Training (zero-shot) | Baseline | — |

| Task Complexity | GPT-4o Success Rate | Qwen2-VL Success Rate | Explanation |
|-----------|-------------|----------------|------|
| Depth 1-2 (Easy) | 82.5% | 68.3% | Small gap on simple tasks |
| Depth 3-5 (Medium) | 61.2% | 45.7% | Gap widens as complexity increases |
| Depth 6-10 (Hard) | 38.1% | 22.4% | Long-term planning is the biggest bottleneck |

### Key Findings
- Synthetic data achieves a 91% human acceptance rate, with quality close to manual annotation.
- Graph-structured training data improves agent performance more efficiently than flat data (+15.3% vs. +8.7%).
- **Long-term planning and error recovery** are the biggest weaknesses for all models—even GPT-4o achieves only around 50% and 42%.
- Performance drops sharply as task complexity (graph depth) increases, indicating that current agents are severely lacking in planning capabilities.
- The gap between open-source and closed-source models is narrow on basic operations, but is significant in reasoning and planning.

## Highlights & Insights
- **Oral paper with excellent benchmark design**: Graph-structured task synthesis is an elegant solution.
- **Unprecedented scale**: 36K tasks covering 20 scenarios, vastly exceeding existing benchmarks.
- **Multi-dimensional evaluation fills a gap**: First to systematically diagnose agents across 10 capability dimensions.
- **Value for training**: Graph-structured data is used not only for evaluation but also for efficiently training agents.

## Limitations & Future Work
- Automatically synthesized tasks might lack the "naturalness" of real-world user tasks.
- Whether the classification of the 10 capability dimensions is fully comprehensive remains open to discussion.
- Currently focused on UI operating tasks, with API-based agents yet to be covered.
- Evaluation results may be influenced by specific prompt templates.

## Related Work & Insights
- OSWorld (Xie et al., 2024): An OS-based environment benchmark for agents.
- AndroidWorld (Rawles et al., 2024): Android-platform agent benchmark.
- The graph-structured synthesis idea in this work can be extended to other agent evaluation scenarios that require controllable complexity.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Graph-structured task synthesis and the multi-dimensional evaluation framework are significant innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale evaluations, comparisons of multiple models, and training validation.
- **Writing Quality**: ⭐⭐⭐⭐ Systematic and clear, with abundant charts and tables.
- **Value**: ⭐⭐⭐⭐⭐ As an Oral paper, it significantly advances the field of agent evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Evolution via Multi-Agent Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-evolution_via_multi-agent_self-play.md)
- [\[ICML 2025\] Context is Key: A Benchmark for Forecasting with Essential Textual Information](context_is_key_a_benchmark_for_forecasting_with_essential_textual_information.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](../../ACL2025/multimodal_vlm/agent_rewardbench.md)
- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](../../CVPR2026/multimodal_vlm/hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)

</div>

<!-- RELATED:END -->
