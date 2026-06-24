---
title: >-
  [Paper Note] Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research
description: >-
  [ACL 2025][Reasoning][Language Agent Framework] This paper proposes the AGORA framework, which unifies 10 mainstream Agent reasoning algorithms (such as CoT, ReAct, ToT, and RAP) into pluggable Operator modules using a DAG-based graph orchestration engine. Systematic self-controlled comparisons on mathematical reasoning and multimodal tasks reveal that simple CoT methods often outperform complex algorithms in both accuracy and cost-effectiveness…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Language Agent Framework"
  - "Graph-based Orchestration Engine"
  - "Agent Reasoning Algorithms"
  - "Standardized Evaluation"
  - "Modular Architecture"
date: 2026-05-08
content_hash: ca3c1f48d81e41fa
---

# Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research

**Conference**: ACL 2025  
**arXiv**: [2505.24354](https://arxiv.org/abs/2505.24354)  
**Code**: Yes ([https://github.com/om-ai-lab/OmAgent](https://github.com/om-ai-lab/OmAgent))  
**Area**: Others  
**Keywords**: Language Agent Framework, Graph-based Orchestration Engine, Agent Reasoning Algorithms, Standardized Evaluation, Modular Architecture

## TL;DR

This paper proposes the AGORA framework, which unifies 10 mainstream Agent reasoning algorithms (such as CoT, ReAct, ToT, and RAP) into pluggable Operator modules using a DAG-based graph orchestration engine. Systematic self-controlled comparisons on mathematical reasoning and multimodal tasks reveal that simple CoT methods often outperform complex algorithms in both accuracy and cost-effectiveness, and a single prompt modification can lead to a 90% performance leap.

## Background & Motivation

**Background**: LLM-driven language agents are rapidly permeating various application fields, with Gartner predicting that 33% of organizations will deploy LLM applications by 2025. Universal frameworks like LangChain, AutoGPT, and AgentVerse, as well as domain-specific frameworks like ChemCrow (chemistry) and OS-Copilot (operating systems), have emerged around Agent development. In terms of evaluation, benchmarks such as AgentBench and WebArena have been established.

**Limitations of Prior Work**: Despite the abundance of frameworks, each Agent reasoning algorithm (CoT, ReAct, ToT, RAP, etc.) has its own independent implementation and evaluation setup, leading to three serious issues: (1) integrating each new algorithm requires extensive custom engineering, making modules non-reusable; (2) component interfaces of different frameworks are incompatible, preventing fair comparison of algorithms within the same environment; (3) existing Agent Leaderboards (e.g., Galileo) primarily evaluate LLM's tool calling and API interaction capabilities without simultaneously measuring the effectiveness of the reasoning algorithms themselves.

**Key Challenge**: Agent algorithms are becoming increasingly complex (transitioning from CoT to ToT, RAP, and multimodal search). However, there is a lack of a unified evaluation platform with controlled variables, making it impossible to answer the critical question: "Are complex algorithms truly better than simple ones?" Researchers lack data support when selecting reasoning strategies and often rely on intuition to choose more complex methods.

**Goal**: (1) How to implement and manage diverse heterogeneous Agent algorithms using a unified architecture? (2) How to fairly compare the performance of these algorithms across different LLMs under controlled variable conditions? (3) What kind of reasoning strategies should be paired with models of different scales?

**Key Insight**: The authors observe that all Agent algorithms can essentially be abstracted into a "node $\rightarrow$ edge $\rightarrow$ graph" workflow pattern: each reasoning step is a node, the dependencies between steps are edges, and the entire algorithm is a Directed Acyclic Graph (DAG). This implies that a universal graph orchestration engine can unify the expression of all algorithms.

**Core Idea**: To unify 10 Agent reasoning algorithms into pluggable modules using a DAG graph orchestration engine, systematically revealing the practical rule of "simplicity is efficiency" under the same evaluation framework.

## Method

### Overall Architecture

AGORA (Agent Graph-based Orchestration for Reasoning and Assessment) is built upon the OmAgent framework, dividing Agent development into three layers: the bottom layer is the DAG-based graph orchestration engine responsible for task scheduling and execution; the middle layer is the modular algorithm Operator library containing standardized implementations of 10 reasoning algorithms; the top layer consists of multi-purpose client interfaces supporting real-time interaction, batch evaluation, and command-line debugging. The input is a user query or evaluation dataset, which, after algorithm selection and workflow orchestration, outputs reasoning results and evaluation scores.

### Key Designs

1. **DAG Graph Orchestration Engine**:

    - **Function**: Unifies the expression of all Agent workflows as Directed Acyclic Graphs (DAGs), supporting automatic task scheduling, asynchronous execution, and visual debugging.
    - **Mechanism**: Models each Agent workflow as a DAG, where nodes are divided into two categories: **Simple Task Nodes** (developer-defined custom logic, e.g., LLM calls, tool executions) and **Logical Task Nodes** (built-in control flow primitives, e.g., if-else branches and while loops). It is implemented based on the Netflix Conductor library to automatically handle data flow transfer and dependency resolution between nodes. It supports asynchronous distributed execution, allowing independent task nodes to run in parallel, while providing a visual workflow interface to track the input and output states of each node.
    - **Design Motivation**: The control flows of different Agent algorithms vary significantly (CoT is a linear chain, ToT is a tree search, GoT is subgraph aggregation), which traditional hard-coded pipelines cannot accommodate. The DAG abstraction represents arbitrary topologies in a unified manner, enabling different algorithms to share the same execution engine and thereby greatly reducing redundant engineering.

2. **Modular Operator Algorithm Library**:

    - **Function**: Encapsulates 10 mainstream reasoning algorithms into standardized, pluggable Operator modules.
    - **Mechanism**: Each Operator encompasses clear input/output interface specifications, allowing internal calls to public services like LLM reasoning, tool execution, and memory read/write. Currently implemented are CoT (zero-shot/few-shot), SC-CoT (multi-path voting), ToT (BFS/DFS tree search), ReAct (think-action-observation loop), PoT (program generation & execution), DnC (divide-and-conquer recursion), GoT (subtask graph aggregation), RAP (MCTS planning search), and two multimodal algorithms: V* (LLM-guided visual search) and ZoomEye (tree-structured zoom exploration). In addition, ReAct is enhanced into ReAct-Pro by splitting Think and Action into two independent model calls inspired by Reflexion. For PoT, short-answer and multiple-choice workflows are merged into a two-phase pipeline of "program executor + answer extractor". GoT is also extended from task-specific scenarios (sorting, keyword counting) to general task processing.
    - **Design Motivation**: Previously, each algorithm possessed its own independent codebase with non-reusable components, hindering fair comparison under identical conditions. Standardized Operator interfaces allow switching reasoning strategies by simply changing configurations rather than rewriting code, while ensuring all variables other than the algorithm itself remain strictly controlled during evaluation.

3. **Multi-Scenario Client Evaluation Interface**:

    - **Function**: Provides three plug-and-play interactive/evaluation clients, coverage spanning from qualitative research to large-scale quantitative assessment.
    - **Mechanism**: **WebPageClient** provides a web chat interface for real-time qualitative testing; **ProgrammaticClient** reads predefined JSON test files to conduct batch automated evaluation, supporting output logs and score statistics; **DefaultClient** provides a lightweight command-line interface for development and debugging. The three clients are switched via a unified configuration file, sharing the same Agent workflow and algorithm instances. The evaluation framework defines four core metrics: Accuracy, Cost (in USD), Token Consumption, and Pass Rate (proportion of valid predictions).
    - **Design Motivation**: Agent research simultaneously demands both qualitative analysis (to observe behavior patterns) and quantitative evaluation (to run benchmarks), whereas existing frameworks usually support only one of these. The unified client system allows researchers to switch between different evaluation modes without modifying the Agent code.

### Experimental Settings & Hyperparameters

The framework does not involve model training. The default LLM temperature is 0. Key algorithmic parameters: SC-CoT uses a temperature of 1 and 5 sampled paths; ToT employs BFS search, $b=1$, a maximum depth and maximum steps of 6, and 3 evaluations; ReAct-Pro has a maximum step limit of 10. For multimodal aspects, ZoomEye sets the minimum patch to 384, depth limit to 5, and confidence threshold to $0 \sim 0.4$; V* constrains the maximum search steps to 10 (to avoid excessive search time on high-resolution images).

## Key Experimental Results

### Main Results — Mathematical Reasoning (GSM8K / MATH-500 / AQuA)

| Model | Agent Algorithm | GSM8K (Acc) | AQuA (Acc) | Characteristics |
|------|---------|-------------|------------|------|
| Doubao-lite-32k | CoT | 89.31% | — | Cost of only $0.0558 |
| GPT-4o | CoT | Best | Best | But limited improvement from the Agent framework |
| Qwen2.5-72B | CoT | ≈GPT-4o | — | Open-source model surpasses GPT-4o |
| Llama-3.3-70B | CoT | ≈GPT-4o | — | 70B open-source model rivals commercial models |
| deepseek-r1-1.5B | CoT | — | — | Only 1.5B parameters, yet surpasses InternLM-7B |
| GPT-3.5 | ReAct | 38.13% | 34.25% | Baseline |
| GPT-3.5 | ReAct-Pro | 74.91% | 64.57% | +96% / +88.5% |

### Ablation Study — Breaking Down ReAct vs ReAct-Pro Improvements

| Modification | AQuA Acc | Relative Gain over Baseline | Description |
|------|----------|-------------|------|
| ReAct Baseline | 34.25% | — | Think + Action merged into a single call |
| + Split Think/Action | 40.16% | +17.3% | Two independent calls, each with distinct duties |
| + Add a prompt sentence | 64.57% | +88.5% | "You can take as many steps as needed" |

### Main Results — Multimodal Reasoning (MME-RealWorld 2K-4K)

| Agent Algorithm | VLM | Score | Pass Rate | Total Tokens |
|------------|-----|-------|-----------|--------------|
| ZoomEye | Qwen2.5-VL-72B | 51.56 | 99.81% | 78.1M |
| ZoomEye | Qwen2.5-VL-7B | 48.06 | 96.50% | 95.9M |
| IO (Direct Inference) | Qwen2.5-VL-72B | 44.47 | 100% | 6.2M |
| ZoomEye | InternVL2.5-8B | 43.42 | 99.34% | 155.9M |
| IO | Qwen2.5-VL-7B | 42.86 | 100% | 6.2M |
| ZoomEye | Llava-v1.5-7B | 31.60 | 98.86% | 114.4M |
| V* | seal_vqa + vsm | 15.14 | 72.37% | — |

### Key Findings

- **Simple algorithms are more robust**: CoT yields the highest accuracy and the fewest tokens on most models. The thought generation and state evaluation in ToT do not alleviate reasoning difficulty, but rather increase token consumption by 5x-10x while accuracy improvements do not exceed 1-2%. PoT depends heavily on the quality of code generation, which actually degrades performance on smaller models. The root cause is that simpler methods minimize "error propagation"—each step of multi-step reasoning can introduce errors, whereas single-chain reasoning significantly reduces the risk of error propagation.
- **Prompts serve as the greatest leverage**: By merely adding the prompt "You can take as many steps as needed", ReAct-Pro boosts the accuracy of AQuA from 34.25% to 64.57%, and GSM8K from 38.13% to 74.91%. This single instruction fundamentally alters the model's behavioral pattern, indicating that the marginal utility of prompt engineering far outweighs increases in algorithmic complexity.
- **Open-source models catch up with commercial ones**: 70B-scale open-source models (Llama-3.3-70B, Qwen2.5-72B) outperform GPT-4o on mathematical reasoning; deepseek-r1-1.5B beats InternLM-7B with only 1.5B parameters, demonstrating a marked advantage of reasoning models at smaller scales.
- **Multimodal Agents substantially boost small models**: ZoomEye enables Qwen2.5-VL-7B (48.06) to exceed the performance of the 72B model under direct inference (44.47), demonstrating that Agent workflows can effectively compensate for the limitations of smaller models, albeit at the cost of approximately 15x higher token consumption.
- **Insufficient generalization of V***: V* secures a pass rate of only 72.37%. Because it relies on the specifically trained seal_vqa/vsm models and searches in high-resolution images for extremely long periods, steps must be restricted, leading to frequent target localization failures.

## Highlights & Insights

- **Systemic value of a unified benchmark**: Comparing combinations of 10 algorithms $\times$ multiple LLMs under controlled variables within the same framework is the primary contribution. This horizontal alignment in experimental design breaks down the siloed evaluation dilemma of various Agent algorithms, providing a methodological reference for future research.
- **Practical rule of "simplicity is efficacy"**: The Score vs. Cost analysis clearly indicates that the optimal cost-benefit ratio is concentrated on CoT configurations. This conclusion offers direct guidance for industrial deployment: complexity should not be blindly stacked; rather, one should start with CoT and introduce complexity only as required.
- **Leverage effect of prompt engineering**: The ReAct-Pro case is a textbook demonstration: by splitting calls and adding a single-sentence instruction, performance is nearly doubled at near-zero cost. This approach can be transferred to prompt optimization in any Agent system.

## Limitations & Future Work

- The evaluation only covers mathematical reasoning and multimodal question-answering tasks, lacking more complex real-world scenarios such as tool use, web interaction, and code generation; the generalizability of the findings remains to be verified.
- GoT, RAP, and DnC are excluded from the cost analysis due to excessively high token consumption, failing to fully evaluate the potential advantages of these complex methods when resources are abundant.
- There is a lack of systematic measurement of reasoning latency/time; the cost analysis only assesses tokens and USD without considering the time dimension.
- Methods for adaptively selecting reasoning strategies based on task characteristics are not explored; future work could incorporate meta-learning or router mechanisms to dynamically select optimal algorithms.
- The evaluation of V* relies on custom-trained models rather than general VLMs, which does not constitute an entirely fair comparison with other algorithms.

## Related Work & Insights

- **vs. LangChain/AutoGPT**: These provide general Agent development infrastructure but do not focus on the comparison and optimization of reasoning algorithms. AGORA's core differentiator lies in treating the algorithms themselves as comparable, modular objects, rather than just providing development tools.
- **vs. AgentBench/WebArena**: These benchmarks focus on assessing end-to-end Agent performance in specific environments; AGORA focuses on comparing the reasoning algorithms themselves, making them complementary.
- **vs. Agent Leaderboard (Galileo)**: This evaluates the tool-calling capability of LLMs, whereas AGORA evaluates the interactive effects of both model capability and reasoning strategy, providing a more comprehensive scope.

## Rating

- **Novelty**: ⭐⭐⭐ — The component algorithms already exist; the contribution lies in the unified integration and systematic comparative experimental design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 10 algorithms $\times$ multiple models $\times$ 3 mathematical benchmarks + 1 multimodal benchmark; the findings are highly insightful.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, precise refinement of key findings, and strong practical guidance.
- **Value**: ⭐⭐⭐⭐ — Provides Agent developers with a selection guideline of "CoT first, complexity later" alongside a reusable evaluation framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](../../CVPR2025/llm_reasoning/enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](../../ICLR2026/llm_reasoning/agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[ICLR 2026\] TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture](../../ICLR2026/llm_reasoning/tumix_multi-agent_test-time_scaling_with_tool-use_mixture.md)
- [\[ACL 2025\] Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation](beyond_the_answer_advancing_multi-hop_qa_with_fine-grained_graph_reasoning_and_e.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](../../NeurIPS2025/llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)

</div>

<!-- RELATED:END -->
