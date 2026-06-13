---
title: >-
  [Paper Note] Evaluating LLMs in Open-Source Games
description: >-
  [NeurIPS 2025][Interpretability][Game Theory] This work introduces a novel paradigm of open-source games—where agents submit programs rather than raw actions—to systematically evaluate LLMs on strategic reasoning…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Game Theory"
  - "Program Equilibrium"
  - "Open-Source Games"
  - "Multi-Agent Cooperation"
  - "Code Transparency"
date: 2026-05-08
content_hash: 37a8f4d4948462cc
---

# Evaluating LLMs in Open-Source Games

**Conference**: NeurIPS 2025
**arXiv**: [2512.00371](https://arxiv.org/abs/2512.00371)  
**Code**: [https://github.com/swadeshs/llm-osgt](https://github.com/swadeshs/llm-osgt)  
**Area**: Interpretability
**Keywords**: Game Theory, Program Equilibrium, Open-Source Games, Multi-Agent Cooperation, Code Transparency

## TL;DR
This work introduces a novel paradigm of open-source games—where agents submit programs rather than raw actions—to systematically evaluate LLMs on strategic reasoning, mutual learning, and cooperative gameplay, finding that LLMs can automatically discover approximate program equilibria.

## Background & Motivation

**Background**: Multi-agent LLM research has largely focused on communication and task decomposition, rarely addressing strategic reasoning and cooperation; traditional game theory has primarily targeted human or conventional RL agents.

**Limitations of Prior Work**:
   - LLMs' strategic reasoning capabilities in complex multi-agent environments remain poorly understood.
   - Existing evaluations mostly rely on natural language or black-box actions, making interpretation and verification difficult.
   - Whether LLMs can spontaneously reach cooperative equilibria in cooperative games is unknown.

**Key Challenge**: How to evaluate LLMs' ability to both safeguard self-interest and achieve cooperation in multi-agent strategic environments.

**Goal**: Leverage code transparency to design an evaluation framework that investigates strategic reasoning and the emergence of cooperation in LLMs.

**Key Insight**: Open-source games—overturning the "black-box" constraint by having agents exchange source code, enabling reasoning over known opponent strategies.

**Core Idea**: A three-tier progressive investigation of LLM strategic reasoning via the SPARC benchmark (code comprehension) + open-source games (dynamic strategy) + evolutionary analysis (long-term stability).

## Method

### Overall Architecture
A three-stage evaluation architecture: Tier 1 — SPARC benchmark assesses code comprehension ability → Tier 2 — open-source games (two-player matches) examine emergent strategic mechanisms → Tier 3 — evolutionary dynamics analyze the stability of program equilibria.

### Key Designs

1. **SPARC Benchmark**:

    - Function: Evaluates LLMs' ability to understand opponent strategy code.
    - Mechanism: 239 IPD strategies (from the Axelrod library); given an opponent's code, the model predicts whether the strategy will always cooperate with a pure cooperator within 10 rounds. Three difficulty levels: unmasked, masked (semantic information removed), and obfuscated (all identifiers randomly replaced).
    - Design Motivation: Code transparency is a prerequisite for open-source games; it must first be verified that LLMs can understand strategy code.

2. **Open-Source Game Experiments**:

    - Function: Two-player matches where agents submit Python programs rather than direct actions.
    - Mechanism: Three agent objectives — PM (purely self-interested), CPM (cooperation-first), and DPM (deception-prone). 10 meta-rounds; after each round, agents exchange code and execute it, then revise their strategies based on outcomes.
    - Strategic Feature Evaluation: GPT-4o is used as a judge to assess five features (independent development, exploitation, counter-strategy, imitation, and deception).
    - Design Motivation: Investigates LLM strategic behavior when opponent strategies are known.

3. **Evolutionary Dynamics Analysis**:

    - Function: Analyzes the long-term stability of different strategy types.
    - Mechanism: Replicator dynamics equation $\dot{x}_i = x_i[(Ax)_i - x^TAx]$; CPM/DPM/PM populations are initialized uniformly and their evolutionary trajectories are observed.
    - Design Motivation: Single-round games reveal only local behavior; evolutionary analysis exposes system-level equilibria.

## Key Experimental Results

### Main Results: SPARC Benchmark

| Model | Unmasked Zero-Shot | Unmasked CoT | Masked CoT | Obfuscated Zero-Shot | Obfuscated CoT |
|------|----------------|-----------|---------|-------------|---------|
| Qwen2.5 (7B) | 56.4% | 75.1% | 75.1% | 43.6% | 65.6% |
| Qwen2.5 (72B) | 59.8% | 83.8% | 83.8% | 51.9% | 78.8% |
| DeepSeek-V3 | **81.7%** | 86.3% | **87.6%** | 72.2% | 81.7% |
| Kimi-K2 | 80.1% | **86.7%** | 85.9% | **77.2%** | **83.0%** |
| DeepSeek-R1 | 82.6% | - | 84.2% | 83.4% | - |
| **o4-mini** | **87.6%** | - | **88.0%** | **84.2%** | - |

### Evolutionary Dynamics Analysis

| Game | Long-Term Stable Type | PM Attraction | Notes |
|------|-----------|----------|------|
| IPD | CPM + DPM coexistence | No | Tit-for-Tat-style cooperative strategies are stable; PM is eliminated. |
| Coin Game | Pure PM dominance | **Yes** | Spatial reasoning is more complex; defense is ineffective and active occupation is required. |

### Key Findings
- CoT prompting significantly improves non-reasoning models (average +20%) but has little effect on reasoning models.
- Obfuscation only marginally reduces performance (72–84%), suggesting LLMs rely primarily on algorithmic structure rather than semantic information.
- Although DPM agents have deceptive intent, deception is largely ineffective in a code-transparent environment.
- The same set of agents produces completely opposite evolutionary trajectories across different games, demonstrating that environmental characteristics determine strategy viability.

## Highlights & Insights
- **Strategic Advantage of Code Transparency**: LLMs can understand and reason over opponent code logic, maintaining 72–84% accuracy even after obfuscation, demonstrating deep algorithmic comprehension.
- **Effectiveness of Goal Instructions**: PM/CPM/DPM prompts successfully induce markedly different strategic patterns, showing that LLM behavioral objectives can be substantially shaped through prompt engineering.
- **Conditional Stability of Cooperation**: CPM strategies can persist stably in IPD, indicating that cooperation can be self-sustaining in structurally repeated games — an important insight for multi-agent safety.
- **Three-Tier Progressive Design**: The progression from code comprehension → dynamic games → evolutionary stability yields an elegant and rigorous experimental design.

## Limitations & Future Work
- Only two-player games are studied; more complex scenarios such as multi-player coalitions are not addressed.
- The assumption of full code transparency may not hold in practice, where partial concealment is possible.
- IPD is limited to 10 rounds and the Coin Game uses a small grid.
- Formal verification is not integrated, so it is impossible to guarantee that generated code satisfies safety properties.

## Related Work & Insights
- **vs. Traditional Game-Theoretic LLM Research**: Several prior works study LLM behavior in payoff-matrix games; this paper is the first to systematically investigate code-level strategic reasoning.
- **vs. Cooperative AI**: Frameworks by Hammond/Dafoe are primarily theoretical; this paper provides empirical evaluation tools.
- **Open-Source Game Theory**: Rubinstein's theoretical work is empirically validated on LLMs for the first time.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — An entirely new perspective on empirical LLM research through open-source games.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three-tier progression from SPARC → dyadic games → evolutionary dynamics.
- Writing Quality: ⭐⭐⭐⭐ — Concepts are clear, though some game-theoretic details could be made more accessible.
- Value: ⭐⭐⭐⭐⭐ — Offers deep insights into multi-agent safety, cooperation mechanisms, and strategic reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Saying the Unsaid: Revealing the Hidden Language of Multimodal Systems Through Telephone Games](saying_the_unsaid_revealing_the_hidden_language_of_multimodal_systems_through_te.md)
- [\[ICCV 2025\] Minerva: Evaluating Complex Video Reasoning](../../ICCV2025/interpretability/minerva_evaluating_complex_video_reasoning.md)
- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](../../AAAI2026/interpretability/pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[CVPR 2026\] Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing](../../CVPR2026/interpretability/edit-as-act_goal-regressive_planning_for_open-vocabulary_3d_indoor_scene_editing.md)
- [\[ICLR 2026\] When Machine Learning Gets Personal: Evaluating Prediction and Explanation](../../ICLR2026/interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)

</div>

<!-- RELATED:END -->
