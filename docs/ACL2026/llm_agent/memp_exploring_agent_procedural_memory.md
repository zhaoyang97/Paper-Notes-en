---
title: >-
  [Paper Note] Mem^p: Exploring Agent Procedural Memory
description: >-
  [ACL 2026][LLM Agent][Procedural Memory] The Mem^p framework is proposed to systematically investigate how to construct learnable, updatable…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Procedural Memory"
  - "Agent Learning"
  - "Trajectory Distillation"
  - "Memory Update"
  - "Lifelong Learning"
date: 2026-05-08
content_hash: d5e15d25d451cab2
---

# Mem^p: Exploring Agent Procedural Memory

**Conference**: ACL 2026  
**arXiv**: [2508.06433](https://arxiv.org/abs/2508.06433)  
**Code**: [GitHub](https://github.com/zjunlp/MemP)  
**Area**: Model Compression  
**Keywords**: Procedural Memory, Agent Learning, Trajectory Distillation, Memory Update, Lifelong Learning

## TL;DR

The Mem^p framework is proposed to systematically investigate how to construct learnable, updatable, and lifelong evolving procedural memory for LLM Agents. By distilling past task trajectories into fine-grained step-by-step instructions and high-level script abstractions, combined with a dynamic update mechanism (add/verify/reflect/discard), Mem^p achieves continuous improvement in success rates and significant reductions in execution steps on TravelPlanner and ALFWorld.

## Background & Motivation

**Background**: LLM Agents are capable of handling complex multi-step tasks (e.g., Deep Research, GUI operations, long-range tool calls), but the execution process requires dozens of operations and high token consumption. Currently, Agents start from scratch for each new task and cannot reuse previously accumulated experience.

**Limitations of Prior Work**: (1) Procedural knowledge in existing Agents is either manually designed prompt templates or implicitly embedded in model parameters, making it difficult to update; (2) Frameworks like LangGraph and AutoGPT provide coarse-grained memory abstractions (buffers, rule blocks) but lack systematic optimization for the memory lifecycle, including construction, retrieval, patching, and removal; (3) While works like Voyager and AWM utilize procedural memory, they lack a systematic analysis of different construction, retrieval, and update strategies.

**Key Challenge**: Many complex tasks share deep structural similarity. Agents acquire partial procedural knowledge in early tasks but fail to effectively transfer it to subsequent tasks, leading to repetitive exploration and token waste.

**Goal**: (1) Treat procedural memory as a first-class optimization object to systematically explore construction, retrieval, and update strategies; (2) Enable Agents to distill reusable experience from past trajectories and evolve continuously in new tasks.

**Key Insight**: Inspired by human procedural memory (e.g., riding a bicycle, typing), humans avoid re-learning by compiling skills into automated subroutines. Analogously, Agents should transform successful trajectories into reusable reasoning patterns and tool sequences.

**Core Idea**: Procedural memory is treated as an optimizable knowledge base. Through a tripartite strategy of "trajectory distillation + vector retrieval + dynamic update," the Agent accumulates and refines experience during continuous task execution.

## Method

### Overall Architecture

Mem^p models the Agent interaction as an MDP, where the policy is extended from $\pi(a_t|s_t)$ to $\pi_{m^p}(a_t|s_t)$ (introducing procedural memory $m^p$). The framework consists of three modules: Build (constructing memory from trajectories), Retrieve (recalling relevant memory), and Update (dynamically maintaining the memory base). The input is a task sequence, and the output is a continuously evolving procedural memory base Mem.

### Key Designs

1.  **Memory Build Policy**:

    - **Function**: Transforms completed task trajectories into reusable procedural knowledge.
    - **Mechanism**: Three granularities are designed: Trajectory (preserving complete step-by-step interaction trajectories), Script (using LLMs to analyze and summarize golden trajectories into abstract procedural knowledge), and Proceduralization (combining full trajectories with high-level scripts to provide both concrete examples and abstract guidance). This is formalized as $m^{p_t} = B(\tau_t, r_t)$.
    - **Design Motivation**: Trajectories provide precise execution context but have poor generalization; scripts provide abstract guidance but lack detail. Proceduralization combines both—scripts generalize better on new test sets, while trajectories are more precise for similar tasks.

2.  **Memory Retrieve Policy**:

    - **Function**: Recalls the most relevant procedural knowledge when facing new tasks.
    - **Mechanism**: Different key construction methods are designed: Random Sample, Query (using query descriptions as keys for semantic similarity retrieval), and AveFact (using LLMs to extract task keywords and calculating average keyword similarity). Retrieval uses cosine similarity: $m_{retrieved} = \arg\max_{m^{p_i} \in Mem} \frac{\phi(t_{new}) \cdot \phi(t_i)}{\|\phi(t_{new})\| \|\phi(t_i)\|}$.
    - **Design Motivation**: The Query method captures more accurate matches using semantic context, while AveFact improves efficiency by focusing on core task elements. Both significantly outperform random sampling.

3.  **Memory Update Policy**:

    - **Function**: Dynamically maintains and optimizes the memory base during testing.
    - **Mechanism**: Four update mechanisms are designed: Vanilla (directly appending new memory every t tasks), Validation (retaining only successful trajectory memories, filtering failures and redundancy), and Adjustment (in-place correction by combining the error trajectory with the original memory when retrieved memory leads to failure). This is formalized as $M(t+1) = U(M(t), E(t), \tau_t)$, where $U = Add(M_{new}) \ominus Del(M_{obs}) \oplus Update(M_{est})$.
    - **Design Motivation**: Simple appending leads to memory bloat and quality degradation. Reflection-based adjustment is the most effective—continually refining memory through error correction allows the Agent to achieve near-linear mastery across continuous tasks.

### Loss & Training

Mem^p is an inference-time framework and does not involve model training. During the build phase, GPT-4o, Claude-3.5-sonnet, and Qwen2.5-72B are used as backbone models, with text embeddings used for vector retrieval.

## Key Experimental Results

### Main Results

**Comparison of Build Policies (TravelPlanner #CS / ALFWorld Test)**

| Model | Policy | TravelPlanner #CS↑ | ALFWorld Test↑ | Steps↓ |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | No Memory | 71.93 | 42.14 | 23.76 |
| GPT-4o | Script | 72.08 | 56.43 | 18.52 |
| GPT-4o | Trajectory | 76.02 | 74.29 | 16.49 |
| GPT-4o | Proceduralization | **79.94** | **77.86** | **15.01** |
| Qwen2.5-72B | No Memory | 56.57 | 41.25 | 21.38 |
| Qwen2.5-72B | Proceduralization | **63.82** | **77.19** | **15.32** |

### Ablation Study

**Comparison of Retrieve Policies (TravelPlanner, GPT-4o)**

| Retrieval Policy | #CS↑ | #HC↑ | Steps↓ |
| :--- | :--- | :--- | :--- |
| No Memory | 71.93 | 12.88 | 17.84 |
| Random Sample | 74.59 | 6.72 | 15.12 |
| Key=Query | 73.38 | 8.95 | 15.44 |
| Key=AveFact | **76.02** | 8.25 | **14.64** |

### Key Findings

- Proceduralization (trajectory + script) is the optimal policy across all models and datasets. On ALFWorld, GPT-4o improved from 42.14% to 77.86% (+35.72%).
- The reflective update mechanism outperformed the second-best policy by +0.7 points and reduced steps by 14 in the final task set, proving that error correction during continuous updates is crucial.
- Procedural memory constructed by strong models (GPT-4o) remains effective when transferred to weaker models (Qwen2.5-14B), improving task completion by 5% and reducing steps by 1.6, indicating cross-model transferability.
- Performance improves as the number of retrieved memories increases, but excessive retrieval introduces imprecise memories that can degrade performance.

## Highlights & Insights

- Treating procedural memory as a first-class optimization object is highly valuable. Unlike fragmented memory augmentation works, Mem^p systematically decomposes construction, retrieval, and update dimensions.
- Cross-model transferability is a significant finding, implying experience can be accumulated by strong models and "taught" to weaker ones, similar to knowledge distillation but performed at inference time.
- Reflective update strategies yield the best results, echoing self-refinement research. Agents refining memory by learning from failure is more effective than simple accumulation.

## Limitations & Future Work

- Currently verified only on two benchmarks, TravelPlanner and ALFWorld, with limited task diversity.
- Memory retrieval relies on vector similarity, which may fail for tasks that are inherently similar but have large structural differences.
- The update policy only uses standard benchmark reward signals and lacks more granular feedback mechanisms.
- Long-term forgetting and memory capacity limits have not yet been explored.

## Related Work & Insights

- **vs Voyager/AWM**: These works leverage procedural memory to enhance Agent capabilities but lack systematic analysis of construction/retrieval/update strategies. Mem^p fills this gap.
- **vs ExpeL**: ExpeL focuses on learning from experience, while Mem^p focuses more on structured management and lifecycle optimization of memory.
- **vs AutoManual**: AutoManual generates operation manuals automatically; Mem^p introduces dynamic updates and cross-model transfer capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic analysis of the three dimensions of procedural memory is valuable, though individual component designs are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three backbones + two datasets + multi-dimensional ablations, though task types are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and systematic experimental design, though some descriptions are slightly verbose.
- Value: ⭐⭐⭐⭐ Provides a practical design guide and empirical reference for Agent memory system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents](hela-mem_hebbian_learning_and_associative_memory_for_llm_agents.md)
- [\[ACL 2026\] Exploring Reasoning Reward Model for Agents](exploring_reasoning_reward_model_for_agents.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](../../NeurIPS2025/llm_agent/a-mem_agentic_memory_for_llm_agents.md)
- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)
- [\[ACL 2026\] Grounding Agent Memory in Contextual Intent](grounding_agent_memory_in_contextual_intent.md)

</div>

<!-- RELATED:END -->
