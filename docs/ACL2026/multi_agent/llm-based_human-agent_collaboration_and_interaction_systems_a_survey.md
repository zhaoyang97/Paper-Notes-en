---
title: >-
  [Paper Note] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey
description: >-
  [ACL 2026 Findings][Multi-Agent][human-in-the-loop] This paper provides the first systematic review of "LLM-based Human-Agent Collaboration and Interaction Systems (LLM-HAS)"—reintegrating humans into the agent loop. It establishes a unified taxonomy across five dimensions (Environment/Profiling, Human Feedback, Interaction Type, Orchestration, and Communication) and introduces a Human Agency Scale (A1–A5) to quantify the necessary depth of human involvement in tasks.
tags:
  - "ACL 2026 Findings"
  - "Multi-Agent"
  - "human-in-the-loop"
  - "agent orchestration"
  - "human feedback"
  - "human agency scale"
  - "LLM-HAS"
date: 2026-05-08
content_hash: afdf262fafd96c6e
---

# LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey

**Conference**: ACL 2026 Findings  
**arXiv**: [2505.00753](https://arxiv.org/abs/2505.00753)  
**Code**: https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems  
**Area**: Human-Agent Collaboration / LLM Agent / Survey  
**Keywords**: human-in-the-loop, agent orchestration, human feedback, human agency scale, LLM-HAS

## TL;DR
This paper provides the first systematic review of "LLM-based Human-Agent Collaboration and Interaction Systems (LLM-HAS)"—reintegrating humans into the agent loop. It establishes a unified taxonomy across five dimensions (Environment/Profiling, Human Feedback, Interaction Type, Orchestration, and Communication) and introduces a Human Agency Scale (A1–A5) to quantify the necessary depth of human involvement in tasks.

## Background & Motivation
**Background**: Recent LLM agent research has primarily focused on "full autonomy." Frameworks like single-agent (AutoGPT), multi-agent (MetaGPT), and long-horizon task executors (SWE-Agent) often treat "reducing human intervention" as the primary goal.

**Limitations of Prior Work**: The pursuit of full autonomy faces three major hurdles: (1) Reliability: hallucinations are amplified in multi-step action chains; (2) Complexity: tasks involving scientific research, healthcare, or long-context coherence exceed current standalone LLM capabilities; (3) Safety/Ethics: irreversible action risks increase sharply in finance, medical, and security scenarios. Existing surveys on LLM agents do not specifically investigate how humans can intervene effectively.

**Key Challenge**: The community currently treats "autonomy" as a linear progress bar to be maximized, yet the optimal point for many real-world tasks lies in augmentation rather than automation. There is a lack of a unified framework to describe "when, how, and at what granularity" humans should interact with agents.

**Goal**: (a) Define LLM-HAS and distinguish it from single/multi-agent setups; (b) Categorize existing work across five dimensions; (c) Systematize human feedback types, granularity, and timing; (d) Provide a Human Agency Scale to quantify the "autonomy vs. augmentation" spectrum; (e) Summarize prompting, SFT, and RL implementation routes and benchmarks; (f) Propose five open challenges.

**Key Insight**: Explicitly model the "human" as a first-class component of the LLM-HAS (e.g., Lazy User vs. Informative User) and extend communication/orchestration concepts from multi-agent systems to human-agent scenarios.

**Core Idea**: An LLM-HAS is defined by Environment & Profiling + Human Feedback + Interaction Type + Orchestration + Communication, guided by the Human Agency Scale to calibrate engagement depth.

## Method

### Overall Architecture
The authors decompose LLM-HAS into five orthogonal core dimensions plus one cross-dimensional scale:

- **Environment & Profiling**: Physical world vs. virtual simulation; 4 topologies (single/multi-human × single/multi-agent); Human profiles categorized as Lazy vs. Informative; Agent profiles based on roles (General Assistant, Math Expert, Robot, etc.).
- **Human Feedback**: Type (Evaluative / Corrective / Guidance / Implicit) × Granularity (Coarse / Fine) × Timing (Initial / During / Post).
- **Interaction Type**: Collaboration (sub-categorized into Delegation / Supervision / Cooperation / Coordination), Competition, and Coopetition.
- **Orchestration**: Task Strategy (One-by-One vs. Simultaneous) × Temporal Synchronization (Synchronous vs. Asynchronous).
- **Communication**: Structure (Centralized / Decentralized / Hierarchical) × Mode (Conversation / Observation / Shared Message Pool).
- **Human Agency Scale (A1–A5)**: A1 Full Automation → A2 Minimal Human Input → A3 Equal Partnership → A4 Agent-Assisted → A5 Human-Driven. A1–A2 fall under Automation, while A3–A5 fall under Augmentation.

### Key Designs

**1. Three-Dimensional Classification of Human Feedback (Type × Granularity × Phase)**
Past discussions on human feedback often focused solely on "scoring." This work maps feedback into a 3D coordinate system: Types include Evaluative (like RLHF preference), Corrective (like user edits), Guidance (like demonstrations), and Implicit (observing user behavior). Granularity levels are Holistic vs. Segment-level. Timing is split into Initial, During, and Post phases. This allows any feedback mechanism to be precisely encoded as a triplet (e.g., (Corrective, Fine, During)), facilitating rigorous comparison between systems.

**2. Human Agency Scale (A1–A5): Quantifying Human Involvement**
The Human Agency Scale provides five levels to evaluate where a task falls on the spectrum of automation vs. augmentation. A1 is fully automated, while A5 is human-driven with minimal agent assistance. This scale allows benchmark designers to move beyond the assumption that "full autonomy is always better," which is critical for high-stakes domains like medical diagnosis where A3–A5 partnership is naturally required.

**3. Four Sub-types of Collaboration (Delegation / Supervision / Cooperation / Coordination)**
To avoid treating "collaboration" as an atomic term, the paper further classifies it based on dominance and dynamism. Delegation involves high autonomy after initial instruction; Supervision involves real-time monitoring and intervention; Cooperation describes entities working toward a shared goal; and Coordination focuses on task synchronization and conflict avoidance. This taxonomy helps align interaction types with appropriate communication structures.

### Loss & Training
While this is a survey, it compares three implementation paradigms:
- **Prompting-based**: Flexible and zero-cost but brittle and lacks cross-session learning.
- **SFT-based**: Converts interaction trajectories into behavior improvements; more stable but expensive.
- **RL-based**: Optimizes for long-horizon rounds; faces substantial challenges in reward design and stability. Recent trends move toward hybrid pipelines using prompting/SFT for initialization followed by RL fine-tuning.

## Key Experimental Results

### Main Results
Representative datasets and benchmarks (selected from Table 4):

| Area | Representative Benchmark | Representative Work |
|------|--------------------------|---------------------|
| Embodied AI | PARTNR / MINT / IGLU / TaPA | PARTNR (Chang 2024), TaPA (Wu 2023) |
| Conversational | WEBLINX / Ask-before-Plan / WildSeek | Co-STORM, ReHAC, WebLINX |
| Software Dev | ConvCodeWorld / ColBench / RECODE-H | SWEET-RL, ConvCodeWorld, RECODE-H |
| Gaming | CuisineWorld / MineWorld | MindAgent, MineWorld |
| Healthcare | EmoEval / GenoTEX | EmoAgent, GenoMAS |
| Retail / Travel | τ-Bench / τ2-Bench / UserBench | τ-Bench (Yao 2025), UserBench (Qian 2025) |
| Finance | FinArena-Low-Cost | FineArena |
| Web / Computer Use | InterruptBench | InterruptBench (Zou 2026) |

Comparison of representative LLM-HAS frameworks:

| Framework | Interaction Type | Key Features |
|-----------|------------------|--------------|
| Collaborative Gym | Async + Collab | Evaluates both outcome and interaction quality |
| COWPILOT | Sync + Suggest-then-Execute | Web navigation with human supervision |
| DPT-Agent | Real-time Sync | Based on Dual Process Theory (fast/slow systems) |

### Ablation Study (Ability comparison by Human Feedback dimensions)

| Feedback Type | Collection Difficulty | Signal Precision | Representative Work |
|---------------|-----------------------|------------------|---------------------|
| Evaluative | Low (Scoring/Preference) | Weak; lacks credit assignment | MINT, EmoAgent |
| Corrective | Medium (Edits/Review) | Strong; supports direct learning | SymbioticRAG, AI Chains |
| Guidance | Med-High (Demos/Instr) | Strong; enables bootstrapping | Ask-before-Plan |
| Implicit | Low (Behavior Observation) | Weak and ambiguous | MTOM, MineWorld |

### Key Findings
- Current LLM-HAS research is heavily **agent-centered**. Humans are mostly perceived as passive evaluators. Scenarios where agents actively observe or teach humans are nearly non-existent.
- The gap between "simulated users" (using LLMs) and real humans remains unquantified. Simulated users rarely exhibit the grammatical errors or ambiguity common in real human interaction.
- Evaluation is biased toward task accuracy. Hardly any benchmarks measure "human workload" or "coordination cost," potentially masking systems that improve accuracy by over-burdening the human.
- Safety is frequently neglected. Most works do not address risks like prompt injection or interrupt safety within the collaboration loop.

## Highlights & Insights
- The "5-dimension taxonomy + Human Agency Scale" provides a paradigmatic contribution by shifting human-agent interaction from a collection of scattered works to a structured coordinate system.
- The 3D classification of human feedback is highly actionable for design space exploration, allowing researchers to precisely characterize their feedback mechanisms.
- The emphasis on "Augmentation over Automation" provides a timely correction to the current trend in the LLM agent community toward total autonomy.
- The proposed open challenges (Human Flexibility, Agent-Centered Bias, Evaluation, Safety) serve as a roadmap for next-generation benchmark construction.

## Limitations & Future Work
- The survey primarily adopts an NLP/Agent-centric perspective, potentially overlooking cross-disciplinary nuances from cognitive science.
- Some redundancy exists between dimensions (e.g., Implicit Feedback vs. Communication via Observation).
- It does not offer a "configuration prescription" (e.g., suggesting specific feedback/agency levels for healthcare vs. gaming).
- Future work could develop a regression model that inputs task attributes to automatically recommend the optimal Human Agency level.

## Related Work & Insights
- **vs. LLM Multi-Agent Surveys**: Those focus on agent-agent communication; this work elevates the "human" to a first-class agent within the same conceptual framework.
- **vs. Core LLM Agent Surveys**: Those focus on internal modules (memory/planning); this work focuses on interaction dimensions.
- **vs. Traditional HITL ML**: HITL is usually about data labeling for supervised learning; LLM-HAS focuses on the decision loop and dynamic reasoning.
- **vs. Human-AI Teaming (HCI)**: This work provides a system-centric perspective that complements user-centric HCI views.

## Rating
- Novelty: ⭐⭐⭐⭐ (First survey to comprehensively cover LLM-HAS with a structured analysis framework).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive mapping of datasets/frameworks across 50+ works).
- Writing Quality: ⭐⭐⭐⭐ (Clear hierarchy and unified terminology).
- Value: ⭐⭐⭐⭐⭐ (Establishes the foundational landscape for the critical but neglected area of human-agent collaboration).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards](../../ICLR2026/multi_agent/comas_co-evolving_multi-agent_systems_via_interaction_rewards.md)
- [\[ICML 2026\] Searching for Synergy in Shared Workspace Human-AI Collaboration](../../ICML2026/multi_agent/searching_for_synergy_in_shared_workspace_human-ai_collaboration.md)
- [\[ICML 2026\] Representational Similarity and Model Behavior in Multi-Agent Interaction](../../ICML2026/multi_agent/representational_similarity_and_model_behavior_in_multi-agent_interaction.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/multi_agent/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
