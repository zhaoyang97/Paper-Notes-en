---
title: >-
  [Paper Note] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation
description: >-
  [ACL 2026][LLM Agent][Self-evolving Agent] This paper proposes Mem²Evolve, a self-evolving Agent framework that achieves co-evolution of capability expansion and experience distillation through a dual-memory mechanism (A…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Self-evolving Agent"
  - "Dual-memory Mechanism"
  - "Capability Expansion"
  - "Experience Distillation"
  - "Co-evolution"
date: 2026-05-08
content_hash: 40e48351d5a3268d
---

# Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation

**Conference**: ACL 2026  
**arXiv**: [2604.10923](https://arxiv.org/abs/2604.10923)  
**Code**: [https://buaa-irip-llm.github.io/Mem2Evolve](https://buaa-irip-llm.github.io/Mem2Evolve)  
**Area**: Model Compression  
**Keywords**: Self-evolving Agent, Dual-memory Mechanism, Capability Expansion, Experience Distillation, Co-evolution

## TL;DR
This paper proposes Mem²Evolve, a self-evolving Agent framework that achieves co-evolution of capability expansion and experience distillation through a dual-memory mechanism (Asset Memory + Experience Memory). It achieves an average Pass@1 of 70.24% across 8 benchmarks in 6 task categories, outperforming the strongest baselines for experience-only and capability-only evolution by 11.80% and 6.46%, respectively.

## Background & Motivation

**Background**: LLM Agents are evolving from static task-specific systems toward self-evolving systems capable of utilizing past experiences and autonomously expanding their capabilities. Current self-evolving frameworks follow two primary paradigms: experience-centric evolution (optimizing execution strategies, prompts, or building experience pools) and capability-centric evolution (dynamically creating new tools or expert Agents to expand capability boundaries).

**Limitations of Prior Work**: Existing frameworks treat these two evolutionary processes in isolation. Experience-centric evolution is limited by pre-defined static toolsets, failing to handle tasks beyond existing capability boundaries. Capability-centric evolution creates new assets from scratch without experience guidance, failing to leverage verified strategies or avoid known pitfalls, leading to non-reproducible successes and repetitive errors.

**Key Challenge**: Capability expansion and experience accumulation are inherently interdependent—new capabilities allow Agents to complete more tasks and gain more experience, while experience guides better capability expansion—yet existing methods ignore this intrinsic collaborative relationship.

**Goal**: Design a self-evolving Agent framework that unifies capability expansion and experience distillation within the same evolutionary cycle to achieve co-evolution.

**Key Insight**: Inspired by Piaget's equilibration theory—where intelligence evolves through the interaction of assimilation (integrating new experiences) and accommodation (adjusting internal structures)—Agent evolution is conceptualized as a cognitive development process.

**Core Idea**: Through a dual-memory mechanism (Asset Memory for reusable capabilities and Experience Memory for strategic experience), co-evolution of capability and experience is realized in a loop of forward reasoning and backward evolution.

## Method

### Overall Architecture
The core of Mem²Evolve is a dual-stage task cycle: "Forward Reasoning + Backward Evolution." Forward Reasoning Stage: Task Planning → Asset Recruitment (Priority on reuse, creation on demand) → Execution. Backward Evolution Stage: Trajectory Evaluation → Asset Memory Evolution (retaining and improving high-quality newly created assets) → Experience Memory Evolution (distilling strategic experience from successes and failures). Both memory banks are updated after each task execution, forming a stable self-evolving loop.

### Key Designs

1.  **Dual-Memory Mechanism**:
    *   **Function**: Stores the Agent's reusable capabilities and strategic experience separately.
    *   **Mechanism**: Asset Memory $\mathcal{M}_A = \mathcal{B}_{agt} \cup \mathcal{B}_{tool}$ includes an Agent Bank (storing expert Agent roles, expertise, behavioral strategies, and available tools) and a Tool Bank (storing executable tools complying with the MCP protocol, including names, function descriptions, implementation code, and documentation). Experience Memory $\mathcal{M}_E = \mathcal{E}_{agt} \cup \mathcal{E}_{tool}$ stores strategic experience distilled from past successes and failures, where each entry contains a title, description, applicable scenarios, and core knowledge.
    *   **Design Motivation**: Asset memory provides boundary expansion, while experience memory provides instructional knowledge. The two are complementary—capability expansion without experience is blind, and experience accumulation without capability expansion is limited by fixed toolsets.

2.  **Experience-Guided Asset Creation**:
    *   **Function**: Utilizes past experience to guide the creation of high-quality assets when new capabilities are required.
    *   **Mechanism**: When the similarity between a sub-task and Asset Memory is below a threshold $\delta$, creation is triggered instead of reuse. Tool creation is achieved via experience-augmented generation: $m_{tool}^{new} \sim \pi_\theta(s_i | \text{Retrieve}(s_i, \mathcal{E}_{tool}), \text{Web}(s_i))$, combining retrieved relevant experiences and web search information. After creation, validation is performed through a Self-Correction Loop: the LLM synthesizes test cases from review feedback, and only assets passing all tests are retained.
    *   **Design Motivation**: Experience guidance improves the first-pass validation rate from 53.1% to 72.4% (a relative increase of 36.3%) and reduces the average debugging iterations from 1.01 to 0.48 (a 52.5% reduction).

3.  **Backward Experience Distillation**:
    *   **Function**: Extracts transferable strategic knowledge from each task execution trajectory.
    *   **Mechanism**: After task completion, an LLM-as-a-Judge evaluates execution quality, providing success/failure labels and review comments. Success triggers Success Generalization (abstracting high-level strategy guides), while failure triggers Failure Diagnosis (encoding anti-patterns and failure-fix pairs). Distilled experiences are merged into Experience Memory: $\mathcal{M}_E \leftarrow \mathcal{M}_E \cup \{e_{\text{new}}\}$.
    *   **Design Motivation**: Learning from both success and failure helps replicate effective strategies and avoid known pitfalls.

### Loss & Training
Ours is an inference-time framework and does not involve model parameter training. Asset recruitment uses embedding similarity retrieval (threshold $\delta$), and task evaluation uses LLM-as-a-Judge. All baselines and Mem²Evolve uniformly use GPT-5-chat as the LLM backbone.

## Key Experimental Results

### Main Results

| Method | GAIA Total | ALFWorld | HotpotQA | AIME24 | AIME25 | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5 (ReAct) | 18.47 | 86.87 | 41.40 | 66.67 | 60.00 | 48.27 |
| AFLOW (Experience-centric) | 19.75 | 93.40 | 60.80 | 66.67 | 63.33 | 58.44 |
| Alita (Capability-centric) | 72.73 | 86.13 | 58.80 | 70.00 | 66.67 | 63.78 |
| Mem²Evolve | **76.31** | **94.31** | **60.80** | **76.70** | **73.33** | **70.24** |

### Ablation Study

| Configuration | Average Pass@1 | Decrease |
| :--- | :--- | :--- |
| Full Mem²Evolve | 70.24 | – |
| w/o Tool Creation | 59.96 | ↓10.28 |
| w/o Agent Memory | 65.51 | ↓4.73 |
| w/o Tool Memory | 67.11 | ↓3.13 |
| w/o Expert Agent Creation | 68.52 | ↓1.72 |

### Key Findings
*   Dynamic tool creation is the most critical component (10.28% drop if removed), indicating that expanding the toolset is essential for handling complex tasks.
*   Experience guidance increases the first-pass rate for tool creation from 53.1% to 72.4%, reducing debugging iterations by more than half.
*   Cross-task initialization (initializing other tasks using memory from GAIA) consistently yields performance gains comparable to 25% same-task initialization, demonstrating the excellent transferability of the memory.
*   On GAIA, Mem²Evolve achieves a 76.31% Pass@1, second only to OpenAI DeepResearch's 67.36% (the latter being a proprietary system), showing the strong potential of the framework.

## Highlights & Insights
*   The co-evolutionary paradigm of dual-memory is the primary contribution—inspired by Piaget's cognitive development theory, it unifies "assimilation" (experience accumulation) and "accommodation" (capability adjustment) in one framework. This analogy provides both a theoretical foundation and practical value, making the framework's design logic highly intuitive.
*   The "Reuse first, Create on demand" forward reasoning strategy is highly practical. Using the similarity threshold $\delta$ to automatically determine if a task exceeds current capability boundaries avoids unnecessary asset creation overhead while allowing immediate expansion when needed.
*   The cross-task memory transfer results are impressive: memory accumulated from GAIA data provides gains even on completely different tasks like HotpotQA and AIME without negative transfer, suggesting the distilled experiences possess high abstraction and generality.

## Limitations & Future Work
*   The framework relies on sandboxed environments to execute auto-generated code, which limits deployment in open-world environments requiring direct local file system or unrestricted network access.
*   Continuous growth of asset and experience memories may lead to retrieval efficiency and noise issues; memory management strategies (e.g., forgetting, compression) for long-term deployment were not discussed.
*   The reliability of LLM-as-a-Judge evaluations without ground-truth labels may affect the quality of backward evolution.
*   Tool creation quality is limited by the LLM's code generation capability; complex tools may require multiple iterations to reach usable quality.

## Related Work & Insights
*   **vs Alita (Qiu et al., 2025)**: Alita supports dynamic tool creation but lacks experience guidance. Mem²Evolve adds experience-guided creation and distillation mechanisms, achieving a 6.46% average performance gain.
*   **vs AFLOW (Zhang et al., 2025)**: AFLOW optimizes module combinations through search algorithms but is limited by a fixed toolset. Mem²Evolve accumulates experience while dynamically expanding the toolset, resulting in an 11.80% average performance gain.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ First to propose a co-evolutionary paradigm for capability expansion and experience distillation with clear theoretical motivation.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 8 benchmarks in 6 task categories with extensive ablation, single-task, and cross-task analyses.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear framework diagrams and an insightful analogy with Piaget's theory.
*   **Value**: ⭐⭐⭐⭐ Provides a practical framework foundation for building general-purpose self-evolving Agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](expseek_self-triggered_experience_seeking_for_web_agents.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](../../ICML2026/llm_agent/evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](../../ICLR2026/llm_agent/your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)
- [\[ACL 2026\] SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents](searl_joint_optimization_of_policy_and_tool_graph_memory_for_self-evolving_agent.md)
- [\[ICML 2026\] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](../../ICML2026/llm_agent/towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)

</div>

<!-- RELATED:END -->
