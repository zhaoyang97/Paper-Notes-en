---
title: >-
  [Paper Note] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution
description: >-
  [ACL 2026][LLM Agent][Tool Learning] This paper proposes ToolOmni, a unified agent framework that integrates proactive tool retrieval and retrieval-grounded tool execution within the same reasoning loop. By utilizing col…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Tool Learning"
  - "Proactive Retrieval"
  - "Open-World"
  - "GRPO"
  - "End-to-End"
date: 2026-05-08
content_hash: b42a63f75604222c
---

# ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution

**Conference**: ACL 2026  
**arXiv**: [2604.13787](https://arxiv.org/abs/2604.13787)  
**Code**: [GitHub](https://github.com/Huangsz2021/ToolOmni)  
**Area**: LLM Agent  
**Keywords**: Tool Learning, Proactive Retrieval, Open-World, GRPO, End-to-End

## TL;DR

This paper proposes ToolOmni, a unified agent framework that integrates proactive tool retrieval and retrieval-grounded tool execution within the same reasoning loop. By utilizing cold-start SFT and decoupled multi-objective GRPO to jointly optimize retrieval and execution capabilities, it surpasses strong baselines by +10.8% in end-to-end execution success rate on ToolBench.

## Background & Motivation

**Background**: LLMs enhance problem-solving capabilities by invoking external tools. In open-world scenarios, tool libraries are massive (>10,000 APIs) and dynamically updated. Models must not only use tools but also proactively search for and select the correct ones.

**Limitations of Prior Work**: (1) **Embedding-based retrieval methods** rely on semantic similarity for passive retrieval, decoupling retrieval from agent reasoning and failing to proactively participate in tool selection or query refinement; (2) **Parameter-based memory methods** internalize tool documentation into model weights, requiring expensive retraining for tool updates and showing poor generalization; (3) Existing agent RL frameworks limit LLMs to a few tools like search engines or code executors, failing to scale to diverse open-world scenarios.

**Key Challenge**: Open-world tool use requires solving two interconnected problems simultaneously: "finding the right tools" (retrieval) and "using the tools correctly" (execution). Existing methods either treat them as independent pipelines or optimize only one.

**Goal**: To build an end-to-end agent framework that unifies proactive tool discovery and tool execution into a single reasoning loop.

**Key Insight**: Treat retrieval and execution as interconnected yet independent sub-tasks, optimizing both synchronously via decoupled multi-objective GRPO to avoid mutual interference.

**Core Idea**: A two-stage training approach: first, cold-start SFT to provide basic capabilities; then, decoupled GRPO to calculate rewards and advantages for retrieval and execution separately for independent updates, optimizing proactive retrieval and retrieval-grounded execution end-to-end in an online environment.

## Method

### Overall Architecture

Given a user query $Q$, ToolOmni alternately executes two phases within a unified reasoning loop: (1) **Proactive Retrieval Phase**: The agent autonomously decides if retrieval is needed, generates search queries, calls an embedding retrieval server to fetch candidate tools, and iterates for multiple rounds until an adequate toolset $\mathcal{T}_{sub}$ is collected; (2) **Retrieval-Grounded Execution Phase**: Based on the retrieved tool documentation, the agent generates the final answer through multiple rounds of reasoning and tool calls.

### Key Designs

1. **Proactive Tool Retrieval**:

    - **Function**: The agent autonomously formulates search queries, performs multi-round iterative retrieval, and selects the final toolset.
    - **Mechanism**: Differing from passive one-time retrieval, ToolOmni generates search queries within `<search>` tags and invokes a pre-trained embedding model to retrieve top-k candidate tools. The agent can decide whether to continue searching or adjust queries based on previous results. Once complete, it outputs a ranked subset of tools in `<tool_call>` tags.
    - **Design Motivation**: Proactive retrieval allows the agent to participate in tool selection decisions, dynamically adjusting search strategies based on task complexity and intermediate results.

2. **Decoupled Multi-objective GRPO**:

    - **Function**: Jointly optimizes retrieval accuracy and execution performance while avoiding mutual interference.
    - **Mechanism**: Retrieval and execution are treated as two sub-tasks with task-specific rewards and advantages. The retrieval reward $R_{ret}$ includes three weighted components: format correctness, recall, and conversion rate. The execution reward $R_{exec}$ includes format correctness and answer correctness. Advantage estimation uses group-relative normalization but is calculated independently per sub-task. Gradient updates are performed sequentially (Separated Update) for retrieval and execution to prevent one objective's gradient from over-powering the other.
    - **Design Motivation**: Coupling both sub-tasks into a single reward leads to signal confusion; the sparsity of execution rewards might interfere with retrieval learning.

3. **Selective Rollout Filtering**:

    - **Function**: Ensures the execution policy is trained only within high-quality contexts.
    - **Mechanism**: Only trajectories that successfully recall all ground-truth tools during the retrieval phase ($\mathcal{T}_{gold} \subseteq \mathcal{T}_{sub}$) are retained. Execution phase generation is initiated only after filtering out invalid retrieval instances.
    - **Design Motivation**: Training the execution policy on incorrect toolsets introduces noisy gradients, reducing training stability.

### Loss & Training

Two-stage training: (1) SFT Cold-start: Training with cross-entropy loss on approximately 28K retrieval trajectories + 33K execution trajectories; (2) Decoupled GRPO: Advantages are calculated separately for retrieval and execution followed by sequential updates, with a group size $G=5$ and temperature $T=1.0$.

## Key Experimental Results

### Retrieval Performance (Average NDCG, Multi-Domain)

| Method | Avg NDCG |
|------|-----------|
| BM25 | 18.29 |
| EmbSim | 37.13 |
| ToolGen | 68.64 |
| ToolRetriever | 76.44 |
| **Ours** | **78.29** |

### Ablation Study

| Configuration | Effect |
|------|-----------|
| Full ToolOmni | Best |
| Single-round Retrieval | Performance drops; lacks iterative refinement |
| Coupled GRPO (Single Reward) | Performance drops; signal interference |
| Without Selective Rollout | Performance drops; noisy training data |

### Key Findings
- ToolOmni achieves SOTA in both retrieval and execution, exceeding the baseline end-to-end execution success rate by +10.8%.
- It significantly outperforms baselines on NDCG@1 and @3, indicating the advantage of proactive retrieval in precisely locating "gold tools."
- It demonstrates strong generalization to unseen instructions and tools, learning general tool-use mechanisms rather than rote memorization.

## Highlights & Insights
- The **"Proactive Retrieval" paradigm shift** is crucial: Transitioning from "passively accepting retrieval results" to "the agent autonomously deciding what to search and when to stop" tightly couples retrieval with reasoning.
- **Decoupled multi-objective GRPO** provides a general solution for avoiding signal interference in multi-subtask RL, transferable to other multi-stage agent tasks.
- **Selective Rollout** cleverly addresses the issue of training data quality.

## Limitations & Future Work
- Based on the relatively small Qwen3-4B model; the performance of larger models remains unknown.
- Relies on pre-trained embedding models for low-level retrieval; the quality of these models defines the performance ceiling.
- Uses an LLM simulator in ToolBench instead of real API calls, which may deviate from real-world environments.
- The retrieval phase is limited to a maximum of 4 rounds, which may be insufficient for extremely complex multi-tool collaboration scenarios.

## Related Work & Insights
- **vs ToolGen**: ToolGen uses a generative approach for tool identifiers but requires retraining for new tools; ToolOmni generalizes naturally via proactive retrieval.
- **vs Meta-Tool**: Meta-Tool treats retrieval and execution as pipeline stages rather than jointly optimizing them.
- **vs Search-R1**: These agent RL frameworks inspired ToolOmni, but the latter extends the scope to diverse open-world toolsets.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework design of proactive retrieval + decoupled GRPO is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across both retrieval and execution dimensions on ToolBench.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and complete technical details.
- Value: ⭐⭐⭐⭐ Provides a practical end-to-end solution for open-world tool use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)
- [\[ACL 2026\] IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review](intragent_an_llm_agent_for_content-grounded_information_retrieval_through_litera.md)
- [\[NeurIPS 2025\] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions](../../NeurIPS2025/llm_agent/contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)

</div>

<!-- RELATED:END -->
