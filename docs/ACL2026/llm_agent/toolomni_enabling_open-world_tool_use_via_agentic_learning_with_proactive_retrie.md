---
title: >-
  [Paper Note] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution
description: >-
  [ACL 2026][LLM Agent][Tool Learning] This paper proposes ToolOmni, a unified agentic framework that integrates proactive tool retrieval and retrieval-grounded tool execution within a single reasoning loop. Through cold-s…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Tool Learning"
  - "Proactive Retrieval"
  - "Open-World"
  - "GRPO"
  - "End-to-End"
date: 2026-05-08
content_hash: 6b2352840d9e929b
---

# ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution

**Conference**: ACL 2026
**arXiv**: [2604.13787](https://arxiv.org/abs/2604.13787)  
**Code**: [GitHub](https://github.com/Huangsz2021/ToolOmni)  
**Area**: LLM Agent
**Keywords**: Tool Learning, Proactive Retrieval, Open-World, GRPO, End-to-End

## TL;DR

This paper proposes ToolOmni, a unified agentic framework that integrates proactive tool retrieval and retrieval-grounded tool execution within a single reasoning loop. Through cold-start SFT followed by decoupled multi-objective GRPO, the framework jointly optimizes retrieval and execution capabilities, achieving an end-to-end execution success rate that surpasses strong baselines by +10.8% on ToolBench.

## Background & Motivation

**Background**: LLMs enhance problem-solving by invoking external tools. In open-world settings, tool libraries are large-scale (>10,000 APIs) and dynamically updated, requiring models not only to use tools correctly but also to proactively search for and select the appropriate ones.

**Limitations of Prior Work**: (1) **Embedding-based retrieval methods** rely on semantic similarity for passive retrieval, decoupling retrieval from agentic reasoning and thus unable to actively participate in tool selection or refine searches based on task demands; (2) **Parametric memory methods** internalize tool documentation into model parameters, requiring expensive retraining upon every tool set update, with poor generalization; (3) Existing agentic RL training frameworks confine LLMs to a small number of tools such as search engines or code executors, and cannot scale to diverse open-world scenarios.

**Key Challenge**: Open-world tool use requires simultaneously solving two problems—"finding the right tool" (retrieval) and "using the tool correctly" (execution)—yet existing methods either treat them as independent pipelines or optimize only one of them.

**Goal**: To construct an end-to-end agentic framework that unifies proactive tool discovery and tool execution within a single reasoning loop.

**Key Insight**: Retrieval and execution are treated as interrelated yet distinct subtasks, with decoupled multi-objective GRPO simultaneously optimizing both to avoid mutual interference.

**Core Idea**: A two-stage training paradigm is adopted—cold-start SFT first establishes foundational capabilities, followed by decoupled GRPO that computes rewards and advantages independently for retrieval and execution and updates each separately, enabling end-to-end optimization of proactive retrieval and retrieval-grounded execution in an online environment.

## Method

### Overall Architecture

Given a user query $Q$, ToolOmni alternates between two phases within a unified reasoning loop: (1) **Proactive Retrieval Phase**: the agent autonomously decides whether retrieval is needed, generates search queries, calls an embedding retrieval server to obtain candidate tools, and iterates over multiple rounds until a sufficient tool subset $\mathcal{T}_{sub}$ is assembled; (2) **Retrieval-Grounded Execution Phase**: based on the retrieved tool documentation, the agent produces a final answer through multi-turn reasoning and tool invocations.

### Key Designs

1. **Proactive Tool Retrieval**:

    - **Function**: The agent autonomously formulates search queries, retrieves iteratively across multiple rounds, and selects the final tool set.
    - **Mechanism**: Unlike passive one-shot retrieval, ToolOmni generates search queries within `<search>` tags and calls a pretrained embedding model to retrieve top-k candidate tools. Based on prior retrieval results, the agent autonomously decides whether further search is needed and adjusts its queries accordingly. Upon completion, a ranked tool subset is output within `<tool_call>` tags.
    - **Design Motivation**: Proactive retrieval involves the agent in tool selection decisions, enabling dynamic adjustment of search strategies based on task complexity and intermediate results.

2. **Decoupled Multi-Objective GRPO**:

    - **Function**: Simultaneously optimizes retrieval accuracy and execution performance while preventing mutual interference.
    - **Mechanism**: Retrieval and execution are treated as two subtasks with task-specific rewards and advantages computed independently. The retrieval reward $R_{ret}$ comprises three weighted components: format correctness, recall rate, and conversion rate; the execution reward $R_{exec}$ comprises format correctness and answer correctness. Advantage estimation uses group-level normalization computed independently per subtask. During gradient updates, retrieval and execution are updated sequentially (Separated Update) to prevent one objective's gradients from dominating the other.
    - **Design Motivation**: Coupling both subtasks into a single reward causes signal confusion—the sparsity of execution rewards may interfere with retrieval learning.

3. **Selective Rollout Filtering**:

    - **Function**: Ensures that the execution policy is trained only on high-quality contexts.
    - **Mechanism**: Only trajectories in which the retrieval phase successfully recalls all ground-truth tools ($\mathcal{T}_{gold} \subseteq \mathcal{T}_{sub}$) are retained; the execution phase is initiated only after excluding failed retrieval instances.
    - **Design Motivation**: Training the execution policy on incorrect tool sets introduces noisy gradients and reduces training stability.

### Loss & Training

A two-stage training procedure is employed: (1) **SFT Cold Start**: cross-entropy loss training on approximately 28K retrieval trajectories and 33K execution trajectories; (2) **Decoupled GRPO**: advantages for retrieval and execution are computed separately and updated sequentially, with sampling group size $G=5$ and temperature $T=1.0$.

## Key Experimental Results

### Retrieval Performance (Average NDCG, Multi-Domain)

| Method | Avg. NDCG |
|--------|-----------|
| BM25 | 18.29 |
| EmbSim | 37.13 |
| ToolGen | 68.64 |
| ToolRetriever | 76.44 |
| **ToolOmni** | **78.29** |

### Ablation Study

| Configuration | Performance |
|---------------|-------------|
| Full ToolOmni | Best |
| Single-round retrieval | Degraded; lacks iterative refinement |
| Coupled GRPO (single reward) | Degraded; signal interference |
| Without selective rollout | Degraded; noisy training data |

### Key Findings
- ToolOmni achieves state-of-the-art performance on both retrieval and execution, with end-to-end execution success rate exceeding baselines by +10.8%.
- Significant improvements on NDCG@1 and @3 demonstrate the advantage of proactive retrieval in precisely locating gold tools.
- Strong generalization is observed on unseen instructions and tools, indicating that the model learns a general tool-use mechanism rather than memorizing specific instances.

## Highlights & Insights
- The paradigm shift to **"proactive retrieval"** is significant: moving from "passively accepting retrieval results" to "the agent autonomously deciding what to search for and when to stop" tightly couples retrieval with reasoning.
- **Decoupled multi-objective GRPO** offers a generalizable solution for avoiding signal interference in multi-subtask RL, transferable to other multi-stage agentic tasks.
- **Selective Rollout** elegantly addresses training data quality issues.

## Limitations & Future Work
- Experiments are based on the relatively small Qwen3-4B model; the effect of scaling to larger models remains unknown.
- The framework depends on a pretrained embedding model for underlying retrieval, whose quality imposes an upper bound on overall performance.
- ToolBench uses an LLM simulator to replace real API calls, which may introduce a gap with real-world environments.
- The retrieval phase is capped at 4 rounds, which may be insufficient for extremely complex multi-tool collaboration scenarios.

## Related Work & Insights
- **vs. ToolGen**: ToolGen employs a generative approach to directly produce tool identifiers but requires retraining to accommodate new tools; ToolOmni generalizes naturally through proactive retrieval.
- **vs. Meta-Tool**: Meta-Tool treats retrieval and execution as pipeline stages rather than jointly optimizing them.
- **vs. Search-R1**: These agentic RL frameworks inspired ToolOmni, which extends their scope to diverse tool sets in open-world settings.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework design combining proactive retrieval and decoupled GRPO is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of both retrieval and execution dimensions on ToolBench.
- Writing Quality: ⭐⭐⭐⭐ Clear framework presentation with complete technical details.
- Value: ⭐⭐⭐⭐ Provides a practical end-to-end solution for open-world tool use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)
- [\[NeurIPS 2025\] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions](../../NeurIPS2025/llm_agent/contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](your_llm_agents_are_temporally_blind_the_misalignment_between_tool_use_decisions.md)

</div>

<!-- RELATED:END -->
