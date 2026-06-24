---
title: >-
  [Paper Note] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning
description: >-
  [ACL 2025][LLM (Other)][Graph Retrieval-Augmented Generation] Graph Counselor proposes a multi-agent collaborative GraphRAG reasoning framework. It adaptively extracts graph structural information through three agents (Planning/Thought/Execution) and introduces a multi-perspective self-reflection mechanism to correct reasoning biases, outperforming existing methods on multiple graph reasoning tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Graph Retrieval-Augmented Generation"
  - "Multi-Agent Collaboration"
  - "Knowledge Graph Reasoning"
  - "Self-Reflection"
  - "Graph Structure Reasoning"
date: 2026-05-08
content_hash: bb2b15c55754765d
---

# Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning

**Conference**: ACL 2025  
**arXiv**: [2506.03939](https://arxiv.org/abs/2506.03939)  
**Code**: [https://github.com/Graph-Counselor](https://github.com/Graph-Counselor)  
**Area**: LLM/NLP  
**Keywords**: Graph Retrieval-Augmented Generation, Multi-Agent Collaboration, Knowledge Graph Reasoning, Self-Reflection, Graph Structure Reasoning  

## TL;DR
Graph Counselor proposes a multi-agent collaborative GraphRAG reasoning framework. It adaptively extracts graph structural information through three agents (Planning/Thought/Execution) and introduces a multi-perspective self-reflection mechanism to correct reasoning biases, outperforming existing methods on multiple graph reasoning tasks.

## Background & Motivation
**Background**: GraphRAG enhances the factual accuracy of LLMs by modeling knowledge relations, but existing methods suffer from two major limitations.

**Limitations of Prior Work**:
   - **Inefficient Information Aggregation**: Relying on a single agent and fixed iteration patterns, prior works fail to adaptively capture multi-level information (text, structure, degree) in graph data.
   - **Rigid Reasoning Mechanism**: Preset reasoning schemes cannot dynamically adjust reasoning depth based on task complexity, and they lack semantic correction capabilities.

**Key Challenge**: A natural gap exists between the non-linear nature of graph structures and the linear text understanding of LLMs, leading to semantic comprehension biases.

**Goal**: Devise a flexible framework for graph information extraction and self-correcting reasoning.

**Key Insight**: Decompose graph reasoning into three steps—planning, thinking, and executing—allowing different agents to perform dedicated duties.

**Core Idea**: Three specialized agents adaptively extract graph information, combined with multi-perspective self-reflection to correct reasoning biases.

## Method

### Overall Architecture
A two-layer architecture: the inner layer is AGIEM (three-agent iterative reasoning), and the outer layer is SR (self-reflection error correction). Input question $\rightarrow$ Planning Agent analyzes the reasoning path $\rightarrow$ Thought Agent determines what graph information is needed $\rightarrow$ Execution Agent invokes graph operations to extract information $\rightarrow$ Iterate until reasoning is complete $\rightarrow$ SR validates and reflects $\rightarrow$ If there is an error, send back corrected context to re-reason.

### Key Designs
1. **Adaptive Graph Information Extraction Module (AGIEM)**:

    - **Planning Agent**: Analyzes question semantics, formulates subsequent reasoning paths, or determines whether existing information is sufficient to infer the answer.
    - **Thought Agent**: Determines the specific graph information required for the current reasoning step based on the planning results.
    - **Execution Agent**: Invokes four types of graph operation components (Retrieve, Feature, Neighbor, Degree), supporting serial combination and parallel execution. For example, $\text{Retrieve}(t) \circ \text{Feature}(I_v, \mathcal{T}_v)$ retrieves first and then extracts features.
    - **Design Motivation**: The three agents perform dedicated functions to avoid role confusion of a single agent among planning, identifying information needs, and actual operations.

2. **Self-Reflection with Multiple Perspectives (SR)**:

    - **Function**: Checks the logical consistency of reasoning results and corrects misalignment between semantics and graph structures.
    - **Mechanism**: Three-stage reflection—(1) Recap & Understanding (recalling reasoning goals), (2) Analysis & Adjustment (identifying omissions/redundancies/inconsistencies), (3) Refinement & Update (optimizing reasoning strategies). This combines backward reasoning with multi-perspective analysis.
    - **Design Motivation**: Combats semantic drift issues of LLMs in graph reasoning by using divergent thinking to correct errors from multiple angles.

3. **LLM State Transition Mechanism**:

    - A single LLM transitions between the three agent roles via context switching to achieve multi-agent collaboration.
    - An outer judgment module evaluates the correctness of reasoning; if incorrect, it triggers the SR reflection loop.

### Loss & Training
No training or fine-tuning required. A purely prompt-based method utilized during inference.

## Key Experimental Results

### Main Results (GRBENCH Dataset, QwenScore)

| Model | Base | TextRAG | GraphRAG-1hop | Graph-CoT | Graph Counselor |
|------|------|---------|---------------|-----------|----------------|
| Llama-3.1-70B (Academic) | 10.82 | 14.00 | 34.94 | ~30 | **38+** |
| Llama-3.1-70B (Legal) | 16.11 | 28.89 | 37.22 | ~35 | **42+** |

### Ablation Study

| Configuration | Description |
|------|------|
| w/o AGIEM (Single Agent) | Significant performance drop, validating the necessity of multi-agent collaboration |
| w/o SR (No Reflection) | Drop in reasoning accuracy, especially for complex multi-hop questions |
| w/o Degree Component | Failure in degree-related queries |

### Key Findings
- The framework shows the greatest advantage in multi-hop reasoning questions that require traversing multiple graph relationships.
- The SR reflection mechanism contributes significantly to complex questions but has less impact on simple ones.
- The composition mechanism of the Execution Agent's components is key to flexibility.
- Performance improvements are observed across different domains (academic, e-commerce, medical, legal), demonstrating strong generalization.

## Highlights & Insights
- The three-agent division of labor is clearer and more effective than a single agent doing everything—separating planning from execution is a best practice in agent design.
- The serial and parallel combination of graph operation components provides programming-like flexibility, adapting better to complex queries than fixed-hop GraphRAG.
- Multi-perspective reflection in SR is more effective than simple self-checking, as it introduces divergent thinking to avoid repeating mistakes.
- Realizing multi-agent synergy with a single LLM through role switching reduces deployment complexity.

## Limitations & Future Work
- Multi-turn iteration and reflection incur high computational overhead and result in high inference latency.
- The approach is verified only on KGQA tasks, without exploring other graph tasks (e.g., link prediction, graph classification).
- Graph operation components are manually defined, lacking the capability to dynamically discover new operations.
- The correctness flag in SR's judgment module might be unreliable, affecting the accuracy of triggering reflection.

## Related Work & Insights
- **vs Graph-CoT**: Graph-CoT utilizes a fixed iteration pattern, whereas Graph Counselor's three agents adaptively adjust reasoning depth.
- **vs KG-Agent**: Both involve agent-based reasoning on graphs, but Graph Counselor incorporates a more comprehensive self-reflection mechanism.
- **vs ToG (Think-on-Graph)**: ToG relies on a single agent to traverse the graph, whereas Graph Counselor's multi-agent division of labor is more efficient.

## Rating
- Novelty: ⭐⭐⭐ The combination of multi-agent division, graph operation composition, and self-reflection is novel, although individual components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets, multiple baseline models, ablation studies, and various domains.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagram with well-defined agent roles.
- Value: ⭐⭐⭐⭐ Provides a practical multi-agent reasoning paradigm for GraphRAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)
- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
