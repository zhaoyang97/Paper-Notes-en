---
title: >-
  [Paper Note] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation
description: >-
  [ACL 2026][Model Compression][Self-Evolving Agent] This paper proposes Mem²Evolve, a self-evolving agent framework that achieves co-evolutionary capability expansion and experience distillation via a dual-memory mechanis…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Self-Evolving Agent"
  - "Dual-Memory Mechanism"
  - "Capability Expansion"
  - "Experience Distillation"
  - "Co-Evolution"
date: 2026-05-08
content_hash: e1190f188982d1fb
---

# Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation

**Conference**: ACL 2026
**arXiv**: [2604.10923](https://arxiv.org/abs/2604.10923)
**Code**: [https://buaa-irip-llm.github.io/Mem2Evolve](https://buaa-irip-llm.github.io/Mem2Evolve)
**Area**: Model Compression
**Keywords**: Self-Evolving Agent, Dual-Memory Mechanism, Capability Expansion, Experience Distillation, Co-Evolution

## TL;DR
This paper proposes Mem²Evolve, a self-evolving agent framework that achieves co-evolutionary capability expansion and experience distillation via a dual-memory mechanism (Asset Memory + Experience Memory). The framework attains an average Pass@1 of 70.24% across 8 benchmarks spanning 6 task categories, outperforming the strongest experience-centric and capability-centric baselines by 11.80% and 6.46%, respectively.

## Background & Motivation

**Background**: LLM agents are evolving from static, task-specific systems toward self-evolving systems that can leverage past experiences and autonomously expand their capabilities. Existing self-evolving frameworks follow two main paradigms: experience-centric evolution (optimizing execution strategies, prompts, or building experience repositories by accumulating experience) and capability-centric evolution (expanding capability boundaries by dynamically creating new tools or expert agents).

**Limitations of Prior Work**: These two evolutionary processes are treated in isolation by existing frameworks. Experience-centric evolution is constrained by a predefined static toolset and cannot handle tasks beyond existing capability boundaries. Capability-centric evolution creates new assets from scratch without experiential guidance, failing to leverage validated strategies or avoid known pitfalls, which leads to irreproducible successes and repeated errors.

**Key Challenge**: Capability expansion and experience accumulation are inherently interdependent — new capabilities enable agents to complete more tasks and thereby acquire more experience, while experience in turn guides better capability expansion — yet existing methods overlook this intrinsic synergy.

**Goal**: To design a self-evolving agent framework that unifies capability expansion and experience distillation within the same evolutionary loop, enabling their co-evolution.

**Key Insight**: Inspired by Piaget's equilibration theory — wherein intelligence evolves through the interplay of assimilation (integrating new experiences) and accommodation (adapting internal structures) — agent evolution is analogized to cognitive development.

**Core Idea**: Through a dual-memory mechanism (Asset Memory storing reusable capabilities, Experience Memory storing strategic experiences), co-evolution of capabilities and experiences is realized within a forward-inference and backward-evolution loop.

## Method

### Overall Architecture
The core of Mem²Evolve is a two-phase task cycle of *forward inference* and *backward evolution*. Forward inference: task planning → asset recruitment (reuse first, create on demand) → execution. Backward evolution: trajectory evaluation → asset memory evolution (retaining and refining newly created high-quality assets) → experience memory evolution (distilling strategic experience from successes and failures). Both memory stores are updated after each task execution, forming a stable self-evolving loop.

### Key Designs

1. **Dual-Memory Mechanism**:

    - **Function**: Separately stores the agent's reusable capabilities and strategic experiences.
    - **Mechanism**: Asset Memory $\mathcal{M}_A = \mathcal{B}_{agt} \cup \mathcal{B}_{tool}$ comprises an Agent Bank (storing expert agents' roles, expertise, behavioral strategies, and available tools) and a Tool Bank (storing executable tools compliant with the MCP protocol, including names, functional descriptions, implementation code, and documentation). Experience Memory $\mathcal{M}_E = \mathcal{E}_{agt} \cup \mathcal{E}_{tool}$ stores strategic experiences distilled from past successes and failures; each experience entry contains a title, description, applicable scenarios, and core knowledge.
    - **Design Motivation**: Asset Memory extends capability boundaries while Experience Memory provides guiding knowledge — the two are complementary, as capability expansion without experience is blind, and experience accumulation without capability expansion is constrained by a fixed toolset.

2. **Experience-Guided Asset Creation**:

    - **Function**: When new capabilities are required, leverages past experiences to guide the creation of high-quality assets.
    - **Mechanism**: When the similarity between a subtask and the asset memory falls below threshold $\delta$, creation is triggered instead of reuse. Tool creation is realized through experience-augmented generation: $m_{tool}^{new} \sim \pi_\theta(s_i | \text{Retrieve}(s_i, \mathcal{E}_{tool}), \text{Web}(s_i))$, jointly conditioning on retrieved relevant experiences and web-search information. After creation, a Self-Correction Loop performs validation: the LLM synthesizes test cases from review feedback, and only assets passing all tests are retained.
    - **Design Motivation**: Experience guidance improves the first-pass validation rate from 53.1% to 72.4% (a relative gain of 36.3%) and reduces average debugging iterations from 1.01 to 0.48 (a reduction of 52.5%).

3. **Backward Experience Distillation**:

    - **Function**: Extracts transferable strategic knowledge from execution trajectories of each task.
    - **Mechanism**: Upon task completion, LLM-as-a-Judge evaluates execution quality and assigns success/failure labels along with review comments. Successes trigger Success Generalization (abstracting high-level strategy guidelines), while failures trigger Failure Diagnosis (encoding anti-patterns and failure–repair pairs). Distilled experiences are merged into the experience memory: $\mathcal{M}_E \leftarrow \mathcal{M}_E \cup \{e_{\text{new}}\}$.
    - **Design Motivation**: Learning simultaneously from both successes and failures allows the agent to reproduce effective strategies and avoid known pitfalls.

### Loss & Training
Mem²Evolve is an inference-time framework that does not involve model parameter training. Asset recruitment uses embedding similarity retrieval (threshold $\delta$), and task evaluation employs LLM-as-a-Judge. All baselines and Mem²Evolve uniformly use GPT-5-chat as the LLM backbone.

## Key Experimental Results

### Main Results

| Method | GAIA Total | ALFWorld | HotpotQA | AIME24 | AIME25 | Average |
|--------|-----------|----------|----------|--------|--------|---------|
| GPT-5 (ReAct) | 18.47 | 86.87 | 41.40 | 66.67 | 60.00 | 48.27 |
| AFLOW (experience-centric) | 19.75 | 93.40 | 60.80 | 66.67 | 63.33 | 58.44 |
| Alita (capability-centric) | 72.73 | 86.13 | 58.80 | 70.00 | 66.67 | 63.78 |
| Mem²Evolve | **76.31** | **94.31** | **60.80** | **76.70** | **73.33** | **70.24** |

### Ablation Study

| Configuration | Avg. Pass@1 | Drop |
|---------------|------------|------|
| Full Mem²Evolve | 70.24 | – |
| w/o Tool Creation | 59.96 | ↓10.28 |
| w/o Agent Memory | 65.51 | ↓4.73 |
| w/o Tool Memory | 67.11 | ↓3.13 |
| w/o Expert Agent Creation | 68.52 | ↓1.72 |

### Key Findings
- Dynamic tool creation is the most critical component (removal causes a 10.28% drop), indicating that expanding the toolset is essential for handling complex tasks.
- Experience guidance improves the first-pass tool creation validation rate from 53.1% to 72.4%, reducing debugging iterations by more than half.
- Cross-task initialization (using memories accumulated from GAIA to initialize other tasks) consistently improves performance, approaching the gains of 25-sample same-task initialization, demonstrating good transferability of the memory.
- On GAIA, Mem²Evolve achieves 76.31% Pass@1, second only to OpenAI DeepResearch's 67.36% (a proprietary system), demonstrating the strong potential of the framework.

## Highlights & Insights
- The co-evolutionary paradigm of dual memory is the paper's most significant contribution. Inspired by Piaget's theory of cognitive development, it unifies "assimilation" (experience accumulation) and "accommodation" (capability adaptation) within a single framework. This analogy is both theoretically grounded and practically valuable, lending the framework's design logic exceptional clarity.
- The "Reuse first, Create on demand" forward inference strategy is highly practical. The similarity threshold $\delta$ automatically determines whether a current task exceeds capability boundaries, avoiding unnecessary asset creation overhead while enabling on-demand capability expansion.
- The cross-task memory transfer results are impressive: memories accumulated from GAIA data consistently yield improvements on entirely different tasks such as HotpotQA and AIME without negative transfer, indicating that distilled experiences possess strong abstraction and generality.

## Limitations & Future Work
- The framework relies on a sandbox environment to execute auto-generated code, limiting deployment in open-world settings that require direct access to local file systems or unrestricted network access.
- The continuous growth of asset and experience memories may introduce retrieval efficiency issues and noise; long-term memory management strategies (e.g., forgetting, compression) are not discussed.
- The reliability of LLM-as-a-Judge evaluation in the absence of ground-truth labels may affect the quality of backward evolution.
- The quality of tool creation is bounded by the LLM's code generation capability; complex tools may require multiple iterations to reach usable quality.

## Related Work & Insights
- **vs. Alita (Qiu et al., 2025)**: Alita supports dynamic tool creation but lacks experiential guidance. Mem²Evolve augments this with experience-guided creation and distillation mechanisms, achieving an average performance gain of 6.46%.
- **vs. AFLOW (Zhang et al., 2025)**: AFLOW optimizes module composition via search algorithms but is constrained by a fixed toolset and cannot extend capability boundaries. Mem²Evolve dynamically expands the toolset while accumulating experience, achieving an average performance gain of 11.80%.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to propose a co-evolutionary paradigm for capability expansion and experience distillation, with clear theoretical motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 8 benchmarks across 6 task categories with comprehensive ablation, single-task, and cross-task analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Framework diagrams are clear; the analogy to Piaget's theory is thought-provoking.
- **Overall Recommendation**: ⭐⭐⭐⭐ Provides a practical framework foundation for building general-purpose self-evolving agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mem^p: Exploring Agent Procedural Memory](memp_exploring_agent_procedural_memory.md)
- [\[AAAI 2026\] Condensed Data Expansion Using Model Inversion for Knowledge Distillation](../../AAAI2026/model_compression/condensed_data_expansion_using_model_inversion_for_knowledge_distillation.md)
- [\[ACL 2026\] Enabling Agents to Communicate Entirely in Latent Space](enabling_agents_to_communicate_entirely_in_latent_space.md)
- [\[ACL 2026\] ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)
- [\[ACL 2026\] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)

</div>

<!-- RELATED:END -->
