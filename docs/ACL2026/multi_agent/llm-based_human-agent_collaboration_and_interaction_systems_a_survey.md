---
title: >-
  [Paper Note] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey
description: >-
  [ACL 2026][Multi-Agent][human-in-the-loop] This paper provides the first systematic review of "LLM-based Human-Agent Collaboration Systems (LLM-HAS)"—reinstituting humans into the agent loop. It establishes a unified tax…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "human-in-the-loop"
  - "agent orchestration"
  - "human feedback"
  - "human agency scale"
  - "LLM-HAS"
date: 2026-05-08
content_hash: 5bdad20e3996b616
---

# LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey

**Conference**: ACL 2026  
**arXiv**: [2505.00753](https://arxiv.org/abs/2505.00753)  
**Code**: https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems  
**Area**: Human-Agent Collaboration / LLM Agent / Survey  
**Keywords**: human-in-the-loop, agent orchestration, human feedback, human agency scale, LLM-HAS

## TL;DR
This paper provides the first systematic review of "LLM-based Human-Agent Collaboration Systems (LLM-HAS)"—reinstituting humans into the agent loop. It establishes a unified taxonomy across five dimensions: Environment/Profiling, Human Feedback, Interaction Type, Orchestration, and Communication. Additionally, it introduces a Human Agency Scale (A1–A5) to quantify the depth of human involvement required in tasks.

## Background & Motivation
**Background**: Recent LLM agent research has primarily focused on "full autonomy." Single-agent (AutoGPT), multi-agent (MetaGPT), and long-horizon task execution (SWE-Agent) paradigms generally treat "reducing human intervention" as the ultimate goal.

**Limitations of Prior Work**: The full autonomy trajectory faces three major hurdles: (1) Reliability: Hallucinations are amplified in multi-step action chains; (2) Complexity: Tasks in science, medicine, and long-context coherence exceed the solo capabilities of current LLMs; (3) Safety/Ethics: Risk of irreversible actions increases sharply in financial, medical, and security scenarios. Existing surveys on LLM agents or multi-agent systems do not specifically address effective human intervention.

**Key Challenge**: The current community views "degree of autonomy" as a linear progress bar toward completion. However, the optimal point for many real-world tasks lies in augmentation rather than automation. There is a lack of a unified framework to describe when, how, and at what granularity humans interact with agents.

**Goal**: (a) Define LLM-HAS and distinguish it from single/multi-agent systems; (b) Categorize existing work along five dimensions; (c) Systematize the types, granularities, and timing of human feedback; (d) Provide a Human Agency Scale to quantify "autonomy vs. augmentation"; (e) Summarize implementation routes (prompting, SFT, RL) and representative benchmarks; (f) Identify five major open challenges.

**Key Insight**: Explicitly model the "human" as a first-class component of the LLM-HAS (Lazy User vs. Informative User), extending communication and orchestration concepts from multi-agent systems to human-agent scenarios.

**Core Idea**: An LLM-HAS consists of Environment & Profiling + Human Feedback + Interaction Type + Orchestration + Communication, guided by a Human Agency Scale to calibrate the depth of participation.

## Method

### Overall Architecture
The authors decompose LLM-HAS into five orthogonal core dimensions and one cross-dimensional scale:

- **Environment & Profiling**: Physical world vs. virtual simulation; four topologies based on single/multi-human × single/multi-agent. Human profiles range from Lazy to Informative, while agent profiles are based on roles (general assistant, math expert, robot, etc.).
- **Human Feedback**: Type (Evaluative / Corrective / Guidance / Implicit) × Granularity (Coarse / Fine) × Timing (Initial / During / Post).
- **Interaction Type**: Collaboration (subdivided into Delegation / Supervision / Cooperation / Coordination), Competition, and Coopetition.
- **Orchestration**: Task Strategy (One-by-One vs. Simultaneous) × Temporal Synchronization (Synchronous vs. Asynchronous).
- **Communication**: Structure (Centralized / Decentralized / Hierarchical) × Mode (Conversation / Observation / Shared Message Pool).
- **Human Agency Scale (A1–A5)**: Range from A1 (Full Automation) to A5 (Human-Driven). A1–A2 represent Automation, while A3–A5 represent Augmentation.

### Key Designs

1.  **3D Classification of Human Feedback (Type × Granularity × Phase)**:
    - **Function**: Expands human feedback from simple "scoring" into a 24-cell analytic coordinate system.
    - **Mechanism**: Evaluative mimics preference scoring in RLHF; Corrective involves learning from user edits (e.g., PRELUDE); Guidance uses demos (e.g., InteractGen); Implicit observes user behavior like slider movement (e.g., VeriPlan). Granularity distinguishes between holistic and segment-level feedback.
    - **Design Motivation**: This makes "feedback complexity" a comparable design choice. Coarse evaluations are easy to collect but suffer from weak credit assignment, while fine-grained feedback is precise but increases user burden.

2.  **Human Agency Scale (A1–A5)**:
    - **Function**: Quantifies the depth of human involvement to transform the debate over "agent autonomy" into a categorizable research problem.
    - **Mechanism**: A1 (Agent autonomous); A2 (Critical-point spot-checks); A3 (Equal partnership); A4 (Heavy human input required); A5 (Human-driven, agent as assistant). 
    - **Design Motivation**: Current benchmarks prioritize A1 scenarios, neglecting real-world tasks (e.g., medical diagnosis) that naturally reside in A3–A5. This scale provides a metric for benchmark consistency.

3.  **Four Sub-classes of Collaboration**:
    - **Function**: Avoids treating "collaboration" as an atomic term, subdividing it based on leading role and dynamics.
    - **Mechanism**: Delegation (full command, agent executes); Supervision (real-time monitoring and intervention); Cooperation (voluntary union for a shared goal); Coordination (division of labor to avoid conflict).
    - **Design Motivation**: Different sub-classes require distinct feedback mechanisms and autonomy levels.

### Loss & Training
While this is a survey without a specific training objective, it systematically compares three implementation routes:
- **Prompting-based** (MToM, Collaborative Gym): Flexible with zero training cost, but brittle and lacks cross-session accumulation.
- **SFT-based** (XtraGPT, Ask-before-Plan): Converts interaction trajectories into behavioral improvements; stable but expensive.
- **RL-based** (UserRL, ReHAC): Optimizes long-term rounds, though reward design and sample efficiency remain challenging. Hybrid pipelines (Prompt/SFT + RL fine-tuning) are becoming common.

## Key Experimental Results

### Main Results
Representative datasets and benchmarks (selected from Table 4):

| Area | Representative Benchmark | Key Work |
|------|--------------------------|----------|
| Embodied AI | PARTNR / MINT / IGLU | PARTNR (Chang 2024) |
| Conversational | WebLINX / Ask-before-Plan | Co-STORM, ReHAC |
| Software Dev | ConvCodeWorld / RECODE-H | SWEET-RL, ConvCodeWorld |
| Gaming | CuisineWorld / MineWorld | MindAgent, MineWorld |
| Healthcare | EmoEval / GenoTEX | EmoAgent, GenoMAS |
| Retail / Travel | τ-Bench / UserBench | τ-Bench (Yao 2025) |

Framework Comparison:

| Framework | Interaction Type | Key Characteristics |
|-----------|------------------|---------------------|
| Collaborative Gym | Async + Collab | Evaluates outcome + interaction quality |
| COWPILOT | Sync + Suggest-then-Execute | Human supervision for web navigation |
| DPT-Agent | Real-time Sync | Dual Process Theory; fast/slow systems |

### Ablation Study (Capability Comparison by Human Feedback Dimension)

| Feedback Type | Collection Difficulty | Signal Precision | Representative Work |
|---------------|-----------------------|------------------|---------------------|
| Evaluative    | Low (Scoring)         | Weak             | MINT, SOTOPIA       |
| Corrective    | Medium (Editing)      | Strong           | SymbioticRAG        |
| Guidance      | Mid-High (Demo)       | Strong           | Ask-before-Plan     |
| Implicit      | Low (Observation)     | Weak/Ambiguous   | MTOM, MineWorld     |

### Key Findings
- **Agent-centered Bias**: Most research treats humans as passive evaluators. Scenarios where agents actively observe or coach humans are nearly non-existent.
- **Simulation Gap**: The gap between LLM-simulated users and real humans is unquantified. Simulated users rarely exhibit the grammatical errors or ambiguity of real humans, potentially biasing benchmarks.
- **Metric Neglect**: Benchmarks focus on task accuracy but fail to measure "human workload" or "coordination cost."
- **Safety Void**: Security (prompt injection, interrupt safety) is largely ignored in current LLM-HAS work, despite being critical for high-stakes domains.

## Highlights & Insights
- The "5-dimension taxonomy + Human Agency Scale" shifts the field from scattered works to a structured coordinate system.
- The 3D classification of feedback precisely encodes feedback mechanisms (e.g., Corrective, Fine, During), enabling better design space exploration.
- Emphasizing that the optimal point of many tasks is "Augmentation" rather than "Automation" provides a necessary counterbalance to the autonomy-focused zeitgeist.

## Limitations & Future Work
- The survey leans toward an NLP/Agent perspective, potentially omitting relevant preprints from cognitive science.
- Slight redundancy exists between dimensions (e.g., Observation vs. Implicit Feedback).
- Future work could provide prescriptive "recipes" (e.g., recommending A3 + Fine-grained Corrective feedback for medical diagnosis).
- A potential regression model ($task \to agency level$) could automatically recommend the optimal collaboration depth based on task attributes.

## Related Work & Insights
- **vs. LLM Multi-Agent Surveys**: Previous works focused on agent-agent communication; this survey redefines those systems with humans as first-class agents.
- **vs. LLM Agent Surveys**: While others focus on single-agent modules (planning/tool use), this uses "collaboration dimensions" as the backbone.
- **vs. HITL ML**: Traditional HITL focuses on data labeling; LLM-HAS focuses on the decision loop, involving higher complexity and dynamics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First survey specifically covering LLM-HAS with a new analytic framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of benchmarks and frameworks (50+ works).
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and consistent terminology.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a foundation for the critical yet overlooked problem of human-in-the-loop agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](../../ICML2026/multi_agent/omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ACL 2026\] To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)

</div>

<!-- RELATED:END -->
