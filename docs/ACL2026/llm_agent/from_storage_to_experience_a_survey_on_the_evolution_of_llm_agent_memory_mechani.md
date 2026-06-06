---
title: >-
  [Paper Note] From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms
description: >-
  [ACL 2026][LLM Agent][Agent Memory] This paper provides a systematic survey of LLM Agent memory mechanisms using an evolutionary framework of "Storage → Reflection → Experience." It translates these three stages into fun…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Agent Memory"
  - "Storage"
  - "Reflection"
  - "Experience"
  - "MDL"
  - "Cross-Trajectory Abstraction"
date: 2026-05-08
content_hash: 3ac434f029521540
---

# From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms

**Conference**: ACL 2026  
**arXiv**: [2605.06716](https://arxiv.org/abs/2605.06716)  
**Code**: https://github.com/FeishuLuo/Evolving-LLM-Agent-Memory-Survey  
**Area**: LLM Agent / Memory Mechanism / Survey / Continual Learning  
**Keywords**: Agent Memory, Storage, Reflection, Experience, MDL, Cross-Trajectory Abstraction

## TL;DR
This paper provides a systematic survey of LLM Agent memory mechanisms using an evolutionary framework of "Storage → Reflection → Experience." It translates these three stages into functional signatures: "Trajectory Retention → Trajectory Refinement → Cross-Trajectory Abstraction" and structures the storyline around "Why-How-What" research questions, specifically emphasizing Active Exploration and Cross-Trajectory Abstraction in the Experience phase.

## Background & Motivation

**Background**: LLM agents (ReAct, AutoGPT, AutoGen, etc.) have become a primary focus in AI over the past two years. However, LLMs are inherently stateless—accumulating errors during multi-step reasoning, experiencing persona drift, and losing memory across sessions—necessitating "external memory modules." While many schemes have emerged (MemGPT, Reflexion, Generative Agents, Voyager, CLIN, Mem0...), the field lacks a unified perspective as most works define their own concepts.

**Limitations of Prior Work**: The authors identify two fundamental obstacles: (i) **Paradigmatic Fragmentation**: Existing works either follow an "OS engineering" route (treating LLM context as virtual memory, e.g., MemGPT) or a "cognitive science" route (mimicking hippocampal consolidation), with minimal cross-citation; (ii) **Absence of Technological Synthesis**: Existing surveys (Zhang 2024, Hu 2025, Du 2025, Wu 2025, etc.) focus on static classification without revealing the underlying drivers of evolutionary transitions.

**Key Challenge**: Current surveys provide "cross-sectional snapshots" (modularity by function) but fail to answer "where to go next" because they do not reveal **evolutionary drivers**. Consequently, new researchers lack an actionable roadmap.

**Goal**: To organize literature through an "evolutionary" lens, formally define the three stages, identify the "selection pressures" driving each transition, and provide an in-depth exploration of the frontier Experience stage.

**Key Insight**: The authors use the **level of information abstraction** as the classification axis—Storage preserves raw trajectories (prioritizing fidelity), Reflection performs semantic transformation within intra-trajectories (prioritizing quality), and Experience performs cross-trajectory induction (prioritizing universality). This axis is orthogonal to the "OS vs. Cognitive Science" lines and unifies them.

**Core Idea**: Conceptualize LLM agent memory evolution as an **MDL (Minimum Description Length) driven process**—moving from redundant storage to single-trajectory refinement, and finally to cross-trajectory compression into a general rule set $\mathcal{K}$.

## Method

> Note: As a survey, this paper does not propose a new algorithm but instead structures a framework and core definitions.

### Overall Architecture

The storyline is organized around three layers of RQs:

- **RQ1 (Why)** §3 Three drivers: Long-term consistency, dynamic environments, and continual learning.
- **RQ2 (How)** §4 Three-stage evolutionary path: Storage → Reflection → Experience.
- **RQ3 (What)** §5 Transformations brought by Experience: Active Exploration + Cross-Trajectory Abstraction.

Each stage is formally defined—at time $t$, the agent samples action $a_t \sim \pi_\theta(a_t | \mathcal{I}, o_t, m_t)$, where $m_t = \text{Retrieve}(\mathcal{M}, o_t)$ is the local memory retrieved from global memory $\mathcal{M}$. The difference between stages lies in how $\mathcal{M}$ is constructed.

### Key Designs

1.  **Three-stage Formalized Classification**:
    - **Function**: Precisely characterizes stage differences using "functional signatures" to avoid terminological confusion.
    - **Mechanism**: (i) **Storage**—Trajectory $\tau = \langle(o_1,a_1),...,(o_T,a_T)\rangle$, $\mathcal{M}_{raw} = \{\tau_i\}_{i=1}^N$, retaining raw records; subdivided into Linear (FIFO + context extension), Vector (high-dimensional embedding + semantic retrieval), and Structured (relational tables / hierarchies / graphs). (ii) **Reflection**—Intra-trajectory semantic transformation $\mathcal{F}_{ref}: \mathcal{T}\to\mathcal{S}$, $m_i' = \mathcal{F}_{ref}(\tau_i | \phi)$, where $\phi$ is the evaluation criterion; subdivided into Introspection (self-criticism), Environment (feedback as anchor), and Coordination (multi-agent reflection). (iii) **Experience**—Cross-trajectory induction $\mathcal{F}_{exp}: \mathcal{T}_{batch} \to \mathcal{K}$, requiring $|\mathcal{K}| \ll \sum_{\tau\in\mathcal{T}_{batch}} |\tau|$ (MDL constraint); subdivided into Explicit (NL policy / executable skill), Implicit (fine-tuned into weights / latent variables), and Hybrid (explicit cache + periodic compression to weights).
    - **Design Motivation**: Previous surveys categorized by "long/short term" (cognitive style) or "memory hierarchy" (OS style), failing to explain the generational gap between Reflexion and Voyager. Using the **level of information abstraction** as an axis clarifies that Voyager performs cross-trajectory abstraction (Experience stage), while Reflexion remains at single-trajectory refinement (Reflection stage).

2.  **Three Evolutionary Drivers (Why)**:
    - **Function**: Decomposes the transition from Storage to Experience into three selection pressures.
    - **Mechanism**: (i) **Long-term Consistency**—Agents must maintain state consistency (avoiding contradictory reasoning) and goal consistency (avoiding local optima), leading to early memory modules (MemGPT, Sumers 2023); (ii) **Dynamic Environments**—Knowledge has expiration (Lazaridou 2021) and causality has delayed effects (Joshi 2024), necessitating active management, temporal decay, and causal graphs; (iii) **Continual Learning**—Limitations of episodic memory and error propagation in infinite expansion (Xiong 2025) drive the jump from "recording" to "abstraction" (Experience).
    - **Design Motivation**: Each driver corresponds to a concentrated burst of literature, mapped to a timeline to show which selection pressure birthed which generation.

3.  **Two Transformative Mechanisms of the Experience Stage (What)**:
    - **Function**: Distinguishes frontier work (since late 2025) as the primary contribution of the survey.
    - **Mechanism**: (a) **Active Exploration**—Agents transition from passive recorders to goal-driven experience collectors. Mechanisms are categorized by driver: reward-driven (instant returns), curriculum-driven (dynamic task sequences), and reuse-driven (abstracting historical trajectories); and by dimension: breadth (filling cognitive gaps), depth (high-order skill extraction), and strategy (optimizing long-horizon decision paths). (b) **Cross-Trajectory Abstraction**—Mechanisms include contrastive induction (comparing success vs. failure), multi-granularity chunking (extracting thought patterns), code encapsulation (compositional reuse), and fine-tuning internalization.
    - **Design Motivation**: Experience is the **highest-value differentiator** of this survey. Other surveys conflate it with Reflection or fail to distinguish explicit vs. implicit. The paper clearly defines Reflection as intra-trajectory $\mathcal{F}_{ref}(\tau_i|\phi)=m_i'$ and Experience as inter-trajectory $\mathcal{F}_{exp}(\mathcal{T}_{batch})=\mathcal{K}$, providing clean terminological boundaries.

### Loss & Training
N/A (Survey). References over 200 papers covering 2022 to early 2026.

## Key Experimental Results

### Structural Comparison: Reflection vs. Experience

| Dimension | Reflection | Experience |
|------|-----------|-----------|
| Functional signature | Intra-trajectory: $\mathcal{F}_{ref}(\tau_i\|\phi) = m_i'$ | Inter-trajectory: $\mathcal{F}_{exp}(\mathcal{T}_{batch}) = \mathcal{K}$ |
| Output Form | Refined memory unit $m_i'$, bound to task context | Universal rules/skills $\mathcal{K}$, context-independent |
| Retrieval Reliance | Retrieves semantically similar past tasks during inference | Used as a policy prior in unseen scenarios; no matching needed |
| Representative Work | Reflexion (Shinn 2023), CLIN (Majumder 2023), AgentFold (Ye 2025) | FLEX (Cai 2025), MemSkill (Zhang 2026), SkillRL (Xia 2026) |

### Sub-categorization of Storage Stage

| Sub-category | Core Mechanism | Representative Work | Key Limitation |
|------|---------|---------|---------|
| Linear | FIFO + context window / attention modification | StreamingLLM (Xiao 2023), Mistral sliding window | Capacity limited by attention complexity |
| Vector | embedding + semantic/temporal decay retrieval | Generative Agents (Park 2023), Larimar (Das 2024) | Retrieval ambiguity |
| Structured | Relational tables / hierarchies / graphs | MemGPT (Packer 2023), AriGraph (Anokhin 2024) | High design complexity |

### Abstraction Granularity in Experience Stage

| Level | Abstraction Product | Interpretability | Generalization Boundary |
|------|---------|--------|----------|
| Shallow | NL Rules / Heuristics | High | Intra-domain cross-task |
| Intermediate | Executable modular skeletons (code / skill) | Medium | Task family |
| Deep | Model weight internalization (intuition) | Low | Global but non-editable |

### Key Findings

- **Evolution is "Compression"**: The transition from raw $\mathcal{M}_{raw}$ to refined $\{m_i'\}$ to universal $\mathcal{K}$ follows an MDL process; higher compression rates correlate with proximity to "General Intelligence."
- **Experience is a Qualitative Shift**: Reflection units remain bound to original trajectories, whereas Experience $\mathcal{K}$ acts as a policy prior—a paradigm shift from "case libraries" to "rule bases."
- **Next-generation directions**: Multi-agent, Multimodal, and Distributed Shared Memory are the consensus frontiers.
- **Implicit Experience overlaps with RL/Meta-Learning**: The framework notes that fine-tuning experience into weights overlaps with RL/meta-learning but emphasizes its role as a mediator between interaction trajectories and parameter updates in memory-centric architectures.

## Highlights & Insights

- **"Evolutionary" Narrative is a Major Unlock**: Unlike prior static taxonomies, this survey uses an evolutionary narrative to provide an **oriented roadmap** rather than just a map.
- **Formalized Stages + Functional Signatures**: Binding the three stages to mathematical objects ($\mathcal{M}_{raw}/m_i'/\mathcal{K}$) and differentiating $\mathcal{F}_{ref}$ vs. $\mathcal{F}_{exp}$ provides the community with "hard currency"—precise, citeable terminology.
- **MDL Perspective as a Key Insight**: Constraining the Experience stage to $|\mathcal{K}|\ll \sum|\tau|$ suggests the ultimate form of agent memory is an information-theoretically optimal generative policy prior, linking statistical learning theory to agent engineering.
- **Reflection vs. Experience Comparison (Table 1)**: This clarity is highly citeable, providing the first clean boundary between works like Reflexion, Voyager, CLIN, and FLEX.
- **Living Survey**: The commitment to a continuously updated GitHub repo provides lasting value for the rapidly evolving field of agent memory.

## Limitations & Future Work

- **Author Acknowledgements**: (1) Absence of quantitative horizontal comparison—lack of unified benchmarks makes direct comparison misleading; (2) Technical overlap between Implicit Experience and RL/meta-learning; (3) Recency bias for the Experience stage (some cited works are not yet peer-reviewed).
- **Hidden Issues**: (1) Idealized boundaries—real-world works often blur stages, which the evolutionary narrative might mask; (2) Lacks a "failure mode evolution" perspective—missing a systematic view of how each generation introduces new failure modes (e.g., over-abstraction in Experience); (3) Future directions like Distributed Shared Memory lack actionable research task breakdowns.
- **Improvement Ideas**: (1) Develop unified benchmarks to compare across stages using proxy metrics (e.g., success rate vs. memory size); (2) Identify "counter-evolutionary" phenomena where simpler Storage methods outperform Experience on specific tasks; (3) Explicitly discuss how reasoning models (e.g., o1, DeepSeek-R1) with long internal CoT change the necessity of external memory.

## Related Work & Insights

- **vs. Zhang et al. 2024 (Engineering-focused)**: Zhang focuses on modular engineering; this paper supplements it with evolutionary drivers.
- **vs. Hu et al. 2025 (Dynamic memory)**: Hu identifies dynamics but remains at functional categories; this paper identifies MDL as the underlying driver.
- **vs. Du et al. 2025 (Taxonomy)**: Du provides operation taxonomy; this paper complements it by explaining *why* those operations are needed.
- **vs. MemGPT (Packer 2023)**: Representative of Storage > Structured > Hierarchical sub-category.
- **vs. Reflexion (Shinn 2023)**: Representative of Reflection > Introspection, bound to single trajectories.
- **vs. Voyager (Wang 2023) / FLEX (Cai 2025)**: Representatives of Experience (Explicit skill library vs. Implicit continuous evolution).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "evolutionary + formalization" approach is innovative for a survey.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers over 200 papers but lacks unified benchmark comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure with RQs, formal definitions, and high-value tables.
- **Value**: ⭐⭐⭐⭐⭐ Provides a much-needed standardized terminology for the LLM agent memory field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)
- [\[ICML 2026\] SE-GA: Memory-Augmented Self-Evolution for GUI Agents](../../ICML2026/llm_agent/se-ga_memory-augmented_self-evolution_for_gui_agents.md)
- [\[ACL 2026\] Grounding Agent Memory in Contextual Intent](grounding_agent_memory_in_contextual_intent.md)

</div>

<!-- RELATED:END -->
