---
title: >-
  [Paper Note] ControlLLM: Augment Language Models with Tools by Searching on Graphs
description: >-
  [ECCV2024][Audio & Speech][tool-augmented LLM] This paper proposes the ControlLLM framework, which plans multimodal tool execution by performing graph search (Thoughts-on-Graph) on a pre-built Tool Graph. This significantly improves the accuracy of tool selection and parameter assignment in complex tasks.
tags:
  - "ECCV2024"
  - "Audio & Speech"
  - "tool-augmented LLM"
  - "multi-modal interaction"
  - "graph search"
  - "task planning"
  - "Thoughts-on-Graph"
date: 2026-05-08
content_hash: 3d45cf05c14be56d
---

# ControlLLM: Augment Language Models with Tools by Searching on Graphs

**Conference**: ECCV2024  
**arXiv**: [2310.17796](https://arxiv.org/abs/2310.17796)  
**Code**: [OpenGVLab/ControlLLM](https://github.com/OpenGVLab/ControlLLM)  
**Area**: LLM/NLP  
**Keywords**: tool-augmented LLM, multi-modal interaction, graph search, task planning, Thoughts-on-Graph

## TL;DR

This paper proposes the ControlLLM framework, which plans multimodal tool execution by performing graph search (Thoughts-on-Graph) on a pre-built Tool Graph. This significantly improves the accuracy of tool selection and parameter assignment in complex tasks.

## Background & Motivation

Large language models (LLMs) have demonstrated exceptional capabilities in language understanding and generation, leading researchers to explore using LLMs as a "brain" to schedule external tools (e.g., image generation, audio processing, video editing) for multimodal interaction. Representative works include HuggingGPT, Visual ChatGPT, InternGPT, etc.

However, existing methods face three major challenges:

1. **Ambiguous Task Decomposition**: User inputs in natural language are often vague, making it difficult for existing methods to accurately decompose them into executable subtasks.
2. **Inaccurate Tool Selection and Parameter Assignment**: LLMs rely on Chain-of-Thought (CoT) or Tree-of-Thoughts (ToT) for tool scheduling, which are prone to hallucination, resulting in incorrect tool selection or mismatched parameter types.
3. **Inefficient Tool Scheduling**: Complex tasks involve intricate topological dependencies among multiple tools, which chain-like or tree-like thinking paradigms cannot effectively represent.

Key Observation: A natural topological relationship exists between the inputs and outputs of tools—the output type of one tool can serve as the input for another. This relationship can be modeled using a graph, thereby transforming task planning into a path-search problem on a graph.

## Core Problem

How can LLMs overcome hallucination issues in multimodal tool-use scenarios to accurately complete tool selection, parameter assignment, and execution scheduling?

## Method

ControlLLM consists of three phases: Task Decomposition → Task Planning (ToG) → Solution Execution.

### 1. Task Decomposition

A language model $\mathcal{M}$ (ChatGPT or fine-tuned LLaMA) is utilized to decompose a user request $r$ into several parallel subtasks:

$$\{s_0, \ldots, s_n\} = \mathcal{M}(r)$$

Each subtask contains structured attributes: task description, domain, input arguments (args), and return types, outputted in JSON format. This phase **does not involve** tool selection; instead, it solely focuses on splitting the request and inferring input/output types to determine the starting and ending points for subsequent graph search.

### 2. Task Planning: Thoughts-on-Graph (ToG)

**Tool Graph Construction**: A tool graph $G$ is pre-built, containing two types of nodes:

- **Resource Nodes**: Represent resource types (image, mask, video, audio, etc.), defined as $\langle\text{type}\rangle$
- **Tool Nodes**: Represent tools, defined as $\langle\text{desc, args, ret}\rangle$, which denote function descriptions, input argument lists, and return types, respectively.

The graph contains two types of edges:

- **Tool → Resource Edges**: Connect a tool to its output resource type.
- **Resource → Tool Edges**: Connect a resource type to a tool that takes it as input.

**Graph Search**: Based on the Depth-First Search (DFS) algorithm, the search starts from the input resource nodes of the subtask and explores all feasible paths along the graph until reaching the target output node or exceeding the maximum path length limit (default $m=10$). During searching, the language model is used to score and filter candidate tools. Four search strategies are provided:

| Strategy | Characteristics |
|------|------|
| Greedy | Selects the highest-rated tool at each step; fast but may not find the optimal solution. |
| Beam ($k=3$) | Retains the top-$k$ rated tools to expand the search space. |
| Adaptive | Dynamically adjusts the beam size, selecting tools that exceed a threshold to balance exploration and efficiency. |
| Exhaustive | Traverses all paths to guarantee optimality, but with extremely high time consumption. |

**Post-Processing**: Upon search completion, a Solution Expert selects the optimal solution from all candidate plans, and a Resource Expert infers the remaining parameters for the tools.

### 3. Solution Execution

The execution engine parses the solution into a sequence of Actions and schedules multiple independent subtasks in parallel according to the topological structure. It supports local deployment, remote cloud services, or hybrid endpoints. A state memory is maintained to store all intermediate results, supporting automatic parameter correction at runtime. Finally, the LLM aggregates the execution results to generate a user-friendly response.

### LLM Selection

The framework provides three variants:

- **ControlLLM-ChatGPT**: Uses ChatGPT-3.5 throughout, zero training cost but limited performance.
- **ControlLLM-LLaMA**: Fine-tunes LLaMA-7B to execute all modules; high performance but requires GPU training.
- **ControlLLM-Mix** (Default): Fine-tunes LLaMA-7B for task decomposition and uses ChatGPT for other modules, balancing performance and cost.

## Key Experimental Results

### Evaluation Metrics

- **IR** (Irrelevant Tool Inclusion Rate) ↓: Rate of introducing irrelevant tools.
- **NR** (Necessary Tool Inclusion Rate) ↑: Rate of including necessary tools.
- **HR** (Resource Hallucination Rate) ↓: Resource/parameter hallucination rate.
- **CR** (Resource Type Consistency Rate) ↑: Rate of resource type consistency.
- **SE** (Solution Evaluation) ↑: Overall solution success rate.

### Main Results (ControlLLM-Mix vs. Strongest Baseline HuggingGPT)

| Metric | ControlLLM-Mix | HuggingGPT |
|------|---------------|------------|
| IR ↓ | **0.03** | 0.45 |
| NR ↑ | 0.93 | 0.64 |
| HR ↓ | **0.02** | 0.16 |
| CR ↑ | **0.98** | 0.69 |
| SE (All) ↑ | **0.93** | 0.59 |
| SE (Hard) ↑ | **0.81** | 0.33 |

On hard tasks (>3 APIs), ControlLLM achieves an 81% success rate, whereas the strongest baseline only achieves 33%.

### Ablation Study: Search Strategies

The Adaptive strategy achieves the best balance between performance and efficiency: it accesses an average of 236 tool nodes with an SE of 0.93, whereas the Exhaustive strategy requires accessing 3444 nodes to achieve an SE of 0.97. The Greedy strategy is the fastest but yields an SE of only 0.49.

### Ablation Study: Language Models

After incorporating Prior Knowledge (PK), the performance of all models is significantly enhanced. GPT-4 + PK achieves an SE of 0.98, and even LLaMA2-13B + PK reaches an SE of 0.82.

## Highlights & Insights

1. **Novelty of the ToG Paradigm**: It shifts tool-use planning from "letting the LLM generate a solution" to "searching for a solution on a pre-built graph," which fundamentally avoids the hallucination issues of LLMs.
2. **High Scalability**: Adding new tools only requires updating the tool graph, without needing to re-train the LLM or modify prompts.
3. **Multiple Solution Outputs**: Graph search naturally yields multiple feasible paths, providing alternative plans for users.
4. **Breaking Token Limitations**: Solution search is executed on the graph, independent of academic context window constraints in LLMs.
5. **Full Modality Support**: Standardizes the processing and interaction of text, image, audio, and video modalities.

## Limitations & Future Work

1. **Uncontrolled Tool Output Quality**: The framework guarantees that the solution is theoretically feasible, but cannot guarantee that the physical output of the tools meets user expectations.
2. **Natural Language Ambiguity**: The inherent vagueness in user intent can cause the selected "optimal" solution to differ from the user's actual intention.
3. **Search Overhead**: The Adaptive strategy still of necessity inspects approximately 236 nodes, which introduces higher latency in real-world interactive scenarios.
4. **Tool Graph Maintenance Cost**: As the quantity of tools grows, the complexity of constructing and maintaining the graph rises.
5. **Evaluation Limitations**: The benchmark contains only about 100 instructions, which represents a relatively small scale; the evaluation relies on human voting, which lacks reproducibility.

## Related Work & Insights

| Method | Planning Paradigm | Multimodal Coverage | Multiple Solutions | SE (Hard) |
|------|---------|-----------|--------|-----------|
| HuggingGPT | CoT | Image/Video/Audio | ✗ | 0.33 |
| Visual ChatGPT | CoT | Image Only | ✗ | 0.10 |
| InternGPT | CoT | Image/Video | ✗ | 0.00 |
| GPT4Tools | CoT / Instruction Tuning | Image Only | ✗ | 0.00 |
| **ControlLLM** | **ToG (Graph Search)** | **Image/Video/Audio** | **✓** | **0.81** |

The core difference is that CoT/ToT relies on LLMs to dynamically generate chains/trees of thought at runtime, which is prone to hallucinations. In contrast, ToG searches on a pre-built tool dependency graph, avoiding runtime hallucinations and handling solutions with complex topological structures.

### Insights & Connections

1. The idea of **modeling tool relationships using graph structures** can be extended to agent system design: representing agent capability maps as graphs, to automatically search for multi-agent cooperation solutions.
2. The design philosophy of **decoupling task decomposition and planning** is highly referenceable—employing LLMs first for semantic understanding and decomposition, followed by deterministic algorithm search, thereby leveraging the strengths of both.
3. Comparisons with ToolLLM (DFSDT) demonstrate that **explicitly modeling** tool relationships as a graph is superior to forcing the LLM to **implicitly reason** about tool dependencies.
4. Future work could consider integrating ToG with RAG (Retrieval-Augmented Generation) to dynamically expand the tool graph.
5. The evaluation metric framework established in this paper (IR/NR/HR/CR/SE) can serve as a reference for benchmarking tool usage.

## Rating

- Novelty: ⭐⭐⭐⭐ (The ToG paradigm shifts task planning from LLM generation to graph search, offering a highly novel approach.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Establishes a complete evaluation framework with multi-dimensional ablation studies, though the benchmark scale is relatively small.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, intuitive diagrams, and well-articulated motivations.)
- Value: ⭐⭐⭐⭐ (Substantially advances the field of tool-augmented LLMs, offering high reference value for agent system design.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Continuous Audio Language Models](../../ICLR2026/audio_speech/continuous_audio_language_models.md)
- [\[ICML 2025\] Long-Form Speech Generation with Spoken Language Models](../../ICML2025/audio_speech/long-form_speech_generation_with_spoken_language_models.md)
- [\[ICLR 2026\] JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models](../../ICLR2026/audio_speech/jalmbench_benchmarking_jailbreak_vulnerabilities_in_audio_language_models.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](../../ACL2026/audio_speech/closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[AAAI 2026\] DiffA: Large Language Diffusion Models Can Listen and Understand](../../AAAI2026/audio_speech/diffa_large_language_diffusion_models_can_listen_and_understand.md)

</div>

<!-- RELATED:END -->
