---
title: >-
  [Paper Note] Mem^p: Exploring Agent Procedural Memory
description: >-
  [ACL 2026][Model Compression][Procedural Memory] This paper proposes the Mem^p framework, which systematically investigates how to construct learnable, updatable…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Procedural Memory"
  - "Agent Learning"
  - "Trajectory Distillation"
  - "Memory Update"
  - "Lifelong Learning"
date: 2026-05-08
content_hash: b1a1e5f8e4a4e78e
---

# Mem^p: Exploring Agent Procedural Memory

**Conference**: ACL 2026
**arXiv**: [2508.06433](https://arxiv.org/abs/2508.06433)
**Code**: [GitHub](https://github.com/zjunlp/MemP)
**Area**: Model Compression
**Keywords**: Procedural Memory, Agent Learning, Trajectory Distillation, Memory Update, Lifelong Learning

## TL;DR

This paper proposes the Mem^p framework, which systematically investigates how to construct learnable, updatable, and lifelong-evolving procedural memory for LLM agents. By distilling past task trajectories into fine-grained step-by-step instructions and high-level script abstractions, coupled with a dynamic update mechanism (addition / validation / reflection / retirement), Mem^p achieves consistent improvements in success rate and substantial reductions in execution steps on TravelPlanner and ALFWorld.

## Background & Motivation

**Background**: LLM agents are capable of handling complex multi-step tasks (e.g., Deep Research, GUI manipulation, long-horizon tool invocation), yet execution often requires dozens of steps and substantial token consumption. Current agents start from scratch for every new task, unable to reuse previously accumulated experience.

**Limitations of Prior Work**: (1) Existing agents' procedural knowledge is either hand-crafted prompt templates or implicitly encoded in model parameters and thus difficult to update. (2) Frameworks such as LangGraph and AutoGPT provide coarse-grained memory abstractions (buffers, rule blocks) but lack systematic optimization over the full memory lifecycle—construction, retrieval, repair, and retirement. (3) Works such as Voyager and AWM exploit procedural memory but offer no systematic analysis of different construction, retrieval, and update strategies.

**Key Challenge**: Many complex tasks share deep structural similarities. Agents acquire partial procedural knowledge from earlier tasks yet cannot effectively transfer it to subsequent ones, resulting in redundant exploration and wasted tokens.

**Goal**: (1) Treat procedural memory as a first-class optimization target and systematically explore construction, retrieval, and update strategies. (2) Enable agents to distill reusable experience from past trajectories and continuously evolve across new tasks.

**Key Insight**: Inspired by human procedural memory (e.g., riding a bicycle, typing)—humans avoid relearning by compiling skills into automated subroutines—agents should similarly transform successful trajectories into reusable reasoning patterns and tool sequences.

**Core Idea**: Treat procedural memory as an optimizable knowledge base and adopt a unified strategy of "trajectory distillation + vector retrieval + dynamic update" to enable agents to accumulate and refine experience across a continuous stream of tasks.

## Method

### Overall Architecture

Mem^p models agent interaction as an MDP, extending the policy from $\pi(a_t|s_t)$ to $\pi_{m^p}(a_t|s_t)$ by incorporating procedural memory $m^p$. The framework consists of three modules: **Build** (constructing memory from trajectories), **Retrieve** (retrieving relevant memory), and **Update** (dynamically maintaining the memory store). The input is a sequence of tasks; the output is a continuously evolving procedural memory store Mem.

### Key Designs

1. **Memory Construction Strategy (Build)**

    - **Function**: Transform completed task trajectories into reusable procedural knowledge.
    - **Mechanism**: Three construction granularities are designed—*Trajectory* (retaining the full turn-by-turn interaction as memory), *Script* (using an LLM to analyze and summarize gold trajectories into abstract procedural knowledge), and *Proceduralization* (combining full trajectories with high-level scripts to provide both concrete examples and abstract guidance). Formally: $m^{p_t} = B(\tau_t, r_t)$.
    - **Design Motivation**: Trajectories provide precise execution context but generalize poorly; scripts offer abstract guidance but lack detail. Proceduralization combines the strengths of both—scripts generalize better to unseen test sets, while trajectories are more precise for similar tasks.

2. **Memory Retrieval Strategy (Retrieve)**

    - **Function**: Recall the most relevant procedural knowledge from the memory store when facing a new task.
    - **Mechanism**: Different key construction methods are designed—*Random Sample* (random selection), *Query* (using the query description as the key for semantic similarity retrieval), and *AveFact* (using an LLM to extract task keywords and computing average keyword similarity for retrieval). Retrieval uses cosine similarity: $m_{retrieved} = \arg\max_{m^{p_i} \in Mem} \frac{\phi(t_{new}) \cdot \phi(t_i)}{\|\phi(t_{new})\| \|\phi(t_i)\|}$.
    - **Design Motivation**: The Query method leverages semantic context for more accurate matching; AveFact improves retrieval efficiency by focusing on core task elements. Both significantly outperform random sampling.

3. **Memory Update Strategy (Update)**

    - **Function**: Dynamically maintain and refine the memory store during test time.
    - **Mechanism**: Four update mechanisms are designed—*Vanilla* (directly appending new memories after every $t$ completed tasks), *Validation* (retaining only memories from successful trajectories, filtering failures and redundancies), and *Adjustment* (when a retrieved memory leads to execution failure, combining the erroneous trajectory with the original memory for in-place correction). Formally: $M(t+1) = U(M(t), E(t), \tau_t)$, where $U = Add(M_{new}) \ominus Del(M_{obs}) \oplus Update(M_{est})$.
    - **Design Motivation**: Naive appending leads to memory bloat and quality degradation. Reflection-based correction is the most effective mechanism—by continuously refining memory through error correction, agents achieve near-linear mastery improvement over successive tasks.

### Loss & Training

Mem^p is an inference-time framework that involves no model training. GPT-4o, Claude-3.5-sonnet, and Qwen2.5-72B serve as backbone models during the construction phase; text embeddings are used for vector retrieval.

## Key Experimental Results

### Main Results

**Build Strategy Comparison (TravelPlanner #CS / ALFWorld Test)**

| Model | Strategy | TravelPlanner #CS↑ | ALFWorld Test↑ | Steps↓ |
|------|------|-------------------|----------------|--------|
| GPT-4o | No Memory | 71.93 | 42.14 | 23.76 |
| GPT-4o | Script | 72.08 | 56.43 | 18.52 |
| GPT-4o | Trajectory | 76.02 | 74.29 | 16.49 |
| GPT-4o | Proceduralization | **79.94** | **77.86** | **15.01** |
| Qwen2.5-72B | No Memory | 56.57 | 41.25 | 21.38 |
| Qwen2.5-72B | Proceduralization | **63.82** | **77.19** | **15.32** |

### Ablation Study

**Retrieve Strategy Comparison (TravelPlanner, GPT-4o)**

| Retrieval Strategy | #CS↑ | #HC↑ | Steps↓ |
|---------|------|------|--------|
| No Memory | 71.93 | 12.88 | 17.84 |
| Random Sample | 74.59 | 6.72 | 15.12 |
| Key=Query | 73.38 | 8.95 | 15.44 |
| Key=AveFact | **76.02** | 8.25 | **14.64** |

### Key Findings

- Proceduralization (trajectory + script) is the optimal strategy across all models and datasets; on ALFWorld, GPT-4o improves from 42.14% to 77.86% (+35.72%).
- The reflection-based update mechanism surpasses the second-best strategy by +0.7 points and reduces execution steps by 14 in the final task group, demonstrating that error correction during continuous updates is critical.
- Procedural memory constructed by a strong model (GPT-4o) transferred to a weaker model (Qwen2.5-14B) still improves task completion rate by 5% and reduces steps by 1.6, confirming cross-model transferability of memory.
- Performance improves as the number of retrieved memories increases, but excessive retrieval introduces imprecise memories and degrades performance.

## Highlights & Insights

- Treating procedural memory as a first-class optimization target is a valuable framing—unlike prior scattered memory-augmentation works, Mem^p systematically decomposes the problem along three dimensions (build / retrieve / update) and ablates each independently.
- Cross-model transferability of memory is an important finding: strong models can accumulate experience and "transfer" it to weaker models, analogous to knowledge distillation but performed at inference time.
- The reflection-based update strategy achieves the best results, echoing the self-refinement research direction—agents that learn from failures to refine memory are more effective than those that simply accumulate it.

## Limitations & Future Work

- Validation is conducted on only two benchmarks (TravelPlanner and ALFWorld), limiting task diversity.
- Memory retrieval relies on vector similarity and may fail for tasks that are structurally dissimilar but semantically equivalent.
- Update strategies rely solely on standard benchmark reward signals, lacking more fine-grained feedback mechanisms.
- Long-term memory forgetting and capacity limits remain unexplored.

## Related Work & Insights

- **vs. Voyager / AWM**: These works leverage procedural memory to enhance agent capabilities but lack systematic analysis of construction, retrieval, and update strategies. Mem^p fills this gap.
- **vs. ExpeL**: ExpeL focuses on learning from experience; Mem^p places greater emphasis on structured memory management and lifecycle optimization.
- **vs. AutoManual**: AutoManual automatically generates operation manuals; Mem^p additionally introduces dynamic updating and cross-model transfer capabilities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematic analysis along three dimensions of procedural memory is valuable, though individual component designs are relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three backbone models, two datasets, and multi-dimensional ablations; task type diversity remains limited.
- **Writing Quality**: ⭐⭐⭐⭐ Framework is clearly presented and experiments are systematically designed, though some passages are slightly verbose.
- **Value**: ⭐⭐⭐⭐ Provides practical design guidelines and empirical reference for agent memory system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ACL 2026\] MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)
- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](../../ICLR2026/model_compression/lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[ICLR 2026\] QKV Projections Require a Fraction of Their Memory](../../ICLR2026/model_compression/qkv_projections_require_a_fraction_of_their_memory.md)

</div>

<!-- RELATED:END -->
