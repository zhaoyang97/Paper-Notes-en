---
title: >-
  [Paper Note] Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation
description: >-
  [ACL 2025][LLM (Other)][Tool learning] Proposed DTA-Llama, which transforms sequential tool invocation paths of traditional tree search into Directed Acyclic Graph (DAG) structures to achieve parallel invocation. A Process/Thread inference framework is designed to enable the LLM to decompose tasks and execute multiple tools in parallel in each turn, allowing Llama2-7B to match the performance of GPT-3.5 Parallel Function Calling on StableToolBench.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Tool learning"
  - "Parallel tool invocation"
  - "DAG structure"
  - "Process/Thread inference"
  - "LLM Agent"
date: 2026-05-08
content_hash: f1098d70617b42b6
---

# Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation

**Conference**: ACL 2025  
**arXiv**: [2501.12432](https://arxiv.org/abs/2501.12432)  
**Code**: [https://corn0205.github.io/](https://corn0205.github.io/)  
**Area**: Others  
**Keywords**: Tool learning, Parallel tool invocation, DAG structure, Process/Thread inference, LLM Agent

## TL;DR
Proposed DTA-Llama, which transforms sequential tool invocation paths of traditional tree search into Directed Acyclic Graph (DAG) structures to achieve parallel invocation. A Process/Thread inference framework is designed to enable the LLM to decompose tasks and execute multiple tools in parallel in each turn, allowing Llama2-7B to match the performance of GPT-3.5 Parallel Function Calling on StableToolBench.

## Background & Motivation

**Background**: Tool learning enables LLMs to invoke external APIs to accomplish real-world tasks. Current approaches are divided into pipeline-based (such as CoT/ReAct, invoking one tool per turn) and tree-search-based (such as DFSDT, improving fault tolerance through depth-first search backtracking).

**Limitations of Prior Work**:
   - CoT/ReAct: Invokes only a single tool per turn, resulting in a narrow perception range and requiring more turns.
   - DFSDT: The backtracking mechanism leads to longer tool invocation sequences, significantly increasing token consumption and inference time.
   - Both types of approaches fail to invoke multiple tools in parallel within a single turn.

**Key Challenge**: How to reduce the number of tool invocation turns and computational overhead while maintaining the task completion rate.

**Key Insight**: Analogous to the Process/Thread mechanism in operating systems—each turn of the "Process" decomposes the task into parallelizable subtasks, where multiple "Threads" execute tool calls in parallel, and results are aggregated afterwards.

**Core Idea**: Transforming sequential paths of tree search into DAG parallel structure training data, combined with a Process/Thread inference framework, to achieve multi-tool parallel invocation in each turn.

## Method

### Overall Architecture
**Data Construction**: DFSDT search tree $\rightarrow$ extract successful paths $\rightarrow$ GPT-4 determines which tools can be parallelized $\rightarrow$ convert into DAG structure $\rightarrow$ level-order traversal to generate parallel training data DTA-Tool (~20K samples)  
$\rightarrow$ **Model Training**: Fine-tune the Llama series models on DTA-Tool  
$\rightarrow$ **Inference**: Process (LLM analyzes task state + decomposes parallel tool invocation plans) $\rightarrow$ Thread (executes tool APIs in parallel) $\rightarrow$ Intermediate State Lock (aggregates results) $\rightarrow$ loop until completion.

### Key Designs

1. **Sequential-to-Parallel Data Conversion**:

    - Extract successful paths $\mathcal{P}$ from tree search trajectories.
    - Use GPT-4 to judge which tool invocations in the path can be parallelized (no input/output dependency + no causal relationship).
    - Construct a DAG $\mathcal{G}$, where level-order traversal allows tools in the same layer to be executed in parallel.
    - Data Filtering: Filter out loop calls, incomplete calls, and un-aggregatable structures.

2. **Process/Thread Inference Framework**:

    - **Process**: LLM evaluates task state $\rightarrow$ analyzes current step needs $\rightarrow$ generates multiple parallelizable tool invocation plans (name + parameters).
    - **Thread**: Executes all tool calls proposed by the Process in parallel.
    - **Intermediate State Lock**: Waits for all Threads to complete and aggregates the results, which then serve as the input for the next Process turn.

3. **Loss & Training**:

    - Simplified from the Thought-Action-Observation framework to Thought-Observation (where Action is integrated into Thought as the tool invocation plan).
    - Loss Function: $$\mathcal{L}(\theta) = -\log \sum_{i=1}^n p_\theta(y^i | q, y^{[1:i-1]}, o^{[1:i-1]})$$

## Key Experimental Results

### Main Results (StableToolBench Average SoPR/SoWR)

| Method | SoPR ↑ | SoWR ↑ |
|------|--------|--------|
| GPT-3.5 (ReAct) | 47.9 | - |
| GPT-3.5 (DFSDT) | 66.7 | 65.5 |
| GPT-3.5 (Parallel) | 61.9 | 53.0 |
| ToolLLaMA (DFSDT) | 54.2 | 47.1 |
| **DTA-Llama2-7B** | **60.7** | **53.5** |

The performance of DTA-Llama2-7B (an open-source 7B model) is comparable to GPT-3.5 Parallel Function Calling.

### Efficiency Comparison (Token Consumption)

| Method | Avg. Token Consumption | Inference Time |
|------|----------------|---------|
| DFSDT | Highest (massive backtracking) | Slowest |
| ReAct | Medium | Medium |
| **DTA (Parallel)** | **Lowest** | **Fastest** |

### Key Findings
- **Parallel invocation significantly reduces invocation turns**: DTA requires only 2.46 tool invocation turns per data sample on average.
- 99.1% of the training data contains parallel tool calls.
- The generalization of the method was validated across multiple models, including Llama2-7B and Llama3-8B.
- DAG data quality is higher than the original sequential data—structural optimization eliminates redundant pathways.

## Highlights & Insights
- **Sequential-to-parallel data conversion is the core contribution**: Using GPT-4 to automatically identify dependencies between tool calls and reorganizing tree search paths into DAGs is a data engineering idea that can be transferred to data construction for other multi-step tasks.
- **The Process/Thread design, analogous to operating systems**, is intuitive and effective: the aggregation mechanism of the Intermediate State Lock is crucial for ensuring parallel reliability.
- Open-source 7B models closely approach the parallel function calling capability of GPT-3.5, demonstrating the power of data structure optimization.

## Limitations & Future Work
- Reliance on GPT-4 for sequential-to-parallel conversion introduces costs and potential biases.
- Parallel invocation assumes that APIs support concurrency and will not be overloaded; practical deployment may face rate-limiting issues.
- Evaluated only within the ToolBench ecosystem; its effectiveness on other tool invocation benchmarks (e.g., API-Bank) remains unknown.
- Dynamic adjustment of parallelism was not explored—in some cases, parallel invocation might perform worse than sequential execution (e.g., when information dependency chains are long).

## Related Work & Insights
- **vs DFSDT (ToolLLM)**: DFSDT improves fault tolerance through backtracking but at a huge cost; DTA shortens the path through parallelization while maintaining effectiveness.
- **vs GPT-3.5 Parallel FC**: GPT-3.5's parallel function calling is a black-box implementation; DTA provides an open-source, reproducible solution that achieves comparable performance even with a 7B model.
- **vs LLMCompiler**: LLMCompiler parallelizes tool calls using a compiler-like approach; DTA's DAG data conversion and Process/Thread framework provide a more systematic solution.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of DAG data conversion and the Process/Thread inference framework is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on StableToolBench, efficiency analysis, and multi-model generalization.
- Writing Quality: ⭐⭐⭐⭐ Clear diagrams, intuitive analogies, and systematic description of the method.
- Value: ⭐⭐⭐⭐ Provides a practical parallelization scheme and a high-quality dataset for tool learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision](codetool_enhancing_programmatic_tool_invocation_of_llms_via_process_supervision.md)
- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)

</div>

<!-- RELATED:END -->
