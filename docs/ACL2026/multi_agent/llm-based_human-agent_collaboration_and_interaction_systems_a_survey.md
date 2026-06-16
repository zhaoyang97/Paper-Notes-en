---
title: >-
  [Paper Note] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey
description: >-
  [ACL 2026][Multi-Agent][human-in-the-loop] This paper provides the first systematic review of "LLM-based Human-Agent Collaboration Systems (LLM-HAS)"—reintegrating humans into the agent loop. It establishes a unified taxonomy across five dimensions (Environment/Profiling, Human Feedback, Interaction Type, Orchestration, and Communication) and introduces a Human
tags:
  - ACL 2026
  - Multi-Agent
  - human-in-the-loop
  - agent orchestration
  - human feedback
  - human agency scale
  - LLM-HAS
date: 2026-05-08
content_hash: 92858cf3ce78b143
---
# LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey

**Conference**: ACL 2026 Findings  
**arXiv**: [2505.00753](https://arxiv.org/abs/2505.00753)  
**Code**: https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems  
**Area**: Human-Agent Collaboration / LLM Agent / Survey  
**Keywords**: human-in-the-loop, agent orchestration, human feedback, human agency scale, LLM-HAS

## TL;DR
This paper provides the first systematic review of "LLM-based Human-Agent Collaboration Systems (LLM-HAS)"—reintegrating humans into the agent loop. It establishes a unified taxonomy across five dimensions (Environment/Profiling, Human Feedback, Interaction Type, Orchestration, and Communication) and introduces a Human Agency Scale ($A1$–$A5$) to quantify the required depth of human involvement.

## Background & Motivation
**Background**: Recent LLM agent research has largely focused on "full autonomy," where systems like AutoGPT, MetaGPT, and SWE-Agent aim to minimize human intervention.

**Limitations of Prior Work**: The pursuit of full autonomy faces three major hurdles: (1) Reliability: Hallucinations are amplified during multi-step chains of action; (2) Complexity: Tasks in science, medicine, or those requiring long-context coherence exceed the capabilities of standalone LLMs; (3) Safety/Ethics: Risk of irreversible actions increases sharply in financial and medical scenarios. Existing surveys do not specifically address how to effectively involve humans.

**Key Challenge**: The community currently treats "autonomy" as a single progress bar to be maximized. However, for many real-world tasks, the optimal point lies in augmentation rather than automation. There is a lack of a unified framework to describe when, how, and at what granularity humans should interact with agents.

**Goal**: (a) Define LLM-HAS and distinguish it from single/multi-agent systems; (b) Categorize existing work across five dimensions; (c) Systematize types, granularity, and timing of human feedback; (d) Provide a Human Agency Scale to quantify the degree of "autonomy vs. augmentation"; (e) Summarize prompting, SFT, and RL implementation routes and benchmarks; (f) Identify five open challenges.

**Key Insight**: This work explicitly models the "human" as a first-class component of LLM-HAS (distinguishing between Lazy vs. Informative Users) and extends communication/orchestration concepts from multi-agent systems to human-agent scenarios.

**Core Idea**: An LLM-HAS is defined by Environment & Profiling + Human Feedback + Interaction Type + Orchestration + Communication, with the Human Agency Scale calibrating the depth of participation.

## Method

### Overall Architecture
The authors decompose LLM-HAS into five orthogonal core dimensions and one cross-dimensional scale:

- **Environment & Profiling**: Physical world vs. virtual simulation; four topologies (single/multi-human × single/multi-agent); human profiles as Lazy/Informative; agent profiles based on roles (general assistant, math expert, robot, etc.).
- **Human Feedback**: Type (Evaluative / Corrective / Guidance / Implicit) × Granularity (Coarse / Fine) × Phase (Initial / During / Post).
- **Interaction Type**: Collaboration (further divided into Delegation, Supervision, Cooperation, Coordination), Competition, and Coopetition.
- **Orchestration**: Task Strategy (One-by-One vs. Simultaneous) × Temporal Synchronization (Synchronous vs. Asynchronous).
- **Communication**: Structure (Centralized / Decentralized / Hierarchical) × Mode (Conversation / Observation / Shared Message Pool).
- **Human Agency Scale ($A1$–$A5$)**: $A1$ Full Automation $\rightarrow$ $A2$ Minimal Human Input $\rightarrow$ $A3$ Equal Partnership $\rightarrow$ $A4$ Agent-Assisted $\rightarrow$ $A5$ Human-Driven. $A1$–$A2$ fall under Automation, while $A3$–$A5$ represent Augmentation.

### Key Designs

**1. Three-Dimensional Taxonomy of Human Feedback (Type × Granularity × Phase): Mapping "how humans give feedback" into a locatable coordinate system.**

Implicitly, human feedback is often reduced to a single "score." This paper axes it into: Type (Evaluative, Corrective, Guidance, Implicit); Granularity (Holistic vs. Segment-level); and Phase (Initial, During, Post). This creates 24 analytical cells, allowing any system's feedback mechanism to be encoded as a triple (e.g., (Corrective, Fine, During)). This allows designers to weigh signal quality against user cost.

**2. Human Agency Scale ($A1$–$A5$): Quantifying "how deep humans should participate" across five levels.**

This provides a metric for tasks where augmentation is superior to automation. Level $A1$ is fully autonomous, while $A3$ represents equal partnership where both parties outperform their individual efforts. $A1$–$A2$ are classified as Automation, and $A3$–$A5$ as Augmentation. This scale helps benchmark designers move beyond "can the agent do it alone?" to "should the agent do it alone?".

**3. Four Sub-types of Collaboration (Delegation / Supervision / Cooperation / Coordination): Refinement of the term "collaboration."**

The paper divides Collaboration by "who leads" and "dynamic nature":
- **Delegation**: Initial instruction followed by autonomous execution.
- **Supervision**: Real-time monitoring and intervention.
- **Cooperation**: Voluntary alliance for a common goal.
- **Coordination**: Division of labor focusing on conflict avoidance.

### Loss & Training
While this is a survey, it compares three implementation paradigms:
- **Prompting-based**: Flexible and zero-cost but brittle and lacks cross-session accumulation.
- **SFT-based**: Converts interaction trajectories into behavioral improvements; more stable but expensive.
- **RL-based**: Optimizes long-term rewards; faces challenges in reward design and transparency, often shifting toward prompting/SFT-guided hybrid pipelines.

## Key Experimental Results

### Main Results
The authors summarized representative datasets/benchmarks (Table 4 selection):

| Area | Representative Benchmark | Representative Work |
| :--- | :--- | :--- |
| Embodied AI | PARTNR / MINT / IGLU Multi-Turn / TaPA | PARTNR (Chang 2024), TaPA (Wu 2023) |
| Conversational | WEBLINX / Ask-before-Plan / HOTPOTQA | Co-STORM, ReHAC, WebLINX |
| Software Dev | ConvCodeWorld / ColBench / RECODE-H | SWEET-RL, ConvCodeWorld, RECODE-H |
| Gaming | CuisineWorld / MineWorld | MindAgent, MineWorld |
| Healthcare | EmoEval / GenoTEX | EmoAgent, GenoMAS |
| Retail / Travel | $\tau$-Bench / $\tau^2$-Bench / UserBench | $\tau$-Bench (Yao 2025), UserBench (Qian 2025) |
| Finance | FinArena-Low-Cost | FineArena |
| Web / Computer Use | InterruptBench | InterruptBench (Zou 2026) |

Comparison of representative LLM-HAS frameworks:

| Framework | Interaction Type | Key Features |
| :--- | :--- | :--- |
| Collaborative Gym (Shao 2024) | Async + Collab | Evaluates outcome + interaction quality |
| COWPILOT (Huq 2025) | Sync + Suggest-then-Execute | Chrome extension; human-supervised web navigation |
| DPT-Agent (Zhang 2025) | Real-time Sync | Dual Process Theory; fast/slow dual systems |

### Ablation Study (Ability comparison via Feedback dimension)

| Feedback Type | Collection Difficulty | Signal Precision | Representative Work |
| :--- | :--- | :--- | :--- |
| Evaluative | Low (Rating/Preference) | Weak; lacks credit assignment | MINT, EmoAgent, SOTOPIA |
| Corrective | Medium (Editing/Modifying) | Strong; direct policy learning | SymbioticRAG, SWEET-RL |
| Guidance | Medium-High (Demo/Instruction) | Strong; supports bootstrapping | Hierarchical Agent, Ask-before-Plan |
| Implicit | Low (Behavior Observation) | Weak + Ambiguous | MTOM, Attentive Support |

### Key Findings
- Current research is **heavily agent-centered**: most treat humans as passive evaluators; agents actively observing or teaching humans remains white space.
- The gap between LLM-simulated users and real humans is unquantified; simulators lack the grammatical errors and ambiguity typical of humans.
- Evaluation is biased toward task accuracy; there is no standardized measure for "human workload / cognitive load / coordination cost."
- Safety (prompt injection, data exfiltration, interrupt safety) is largely ignored by existing frameworks.

## Highlights & Insights
- The "5-dimension taxonomy + Human Agency Scale" provides a paradigmatic contribution, moving the field from scattered work to a structured coordinate system.
- The 3D classification of human feedback is practical for橫向comparison and design space exploration.
- Emphasizing "Augmentation over Automation" acts as a necessary corrective to the current obsession with full autonomy in the agent community.
- Identification of four open challenges (Human Flexibility, Agent-Centered Bias, Inadequate Evaluation, Safety) serves as a roadmap for future benchmark construction.

## Limitations & Future Work
- The survey is primarily viewed from an NLP/Agent conference perspective, potentially overlooking cross-disciplinary work in cognitive science.
- Slight redundancy exists between dimensions (e.g., Communication Observation vs. Implicit Feedback).
- Lack of prescriptive "recommendation tables" (e.g., which setup is best for medical diagnosis).

## Related Work & Insights
- **vs. LLM Multi-Agent Surveys**: Those focus on agent-agent communication; this work treats humans as first-class agents.
- **vs. LLM Agent Surveys**: Those focus on single-agent modules (memory/planning); this work uses collaboration dimensions as its backbone.
- **vs. Human-in-the-Loop ML**: Traditional HITL focuses on data labeling; LLM-HAS focuses on the agent decision loop with higher dynamic complexity.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First survey specifically covering LLM-HAS with a new analytical framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of 50+ works across frameworks, datasets, and benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and consistent terminology.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the critical but neglected question of how humans stay in the LLM agent loop.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/multi_agent/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ACL 2026\] To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)

</div>

<!-- RELATED:END -->
