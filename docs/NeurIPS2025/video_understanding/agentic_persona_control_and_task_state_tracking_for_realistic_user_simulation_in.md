---
title: >-
  [Paper Note] Agentic Persona Control and Task State Tracking for Realistic User Simulation
description: >-
  [NeurIPS 2025 (Workshop on SEA)][Video Understanding][Multi-agent framework] A three-agent collaborative framework for realistic user simulation is proposed, comprising a User Agent (coordination)…
tags:
  - "NeurIPS 2025 (Workshop on SEA)"
  - "Video Understanding"
  - "Multi-agent framework"
  - "user simulation"
  - "persona control"
  - "state tracking"
  - "restaurant ordering"
date: 2026-05-08
content_hash: 62f1709e13be07b1
---

# Agentic Persona Control and Task State Tracking for Realistic User Simulation

**Conference**: NeurIPS 2025 (Workshop on SEA)
**arXiv**: [2601.15290](https://arxiv.org/abs/2601.15290)
**Code**: None
**Area**: Video Understanding
**Keywords**: Multi-agent framework, user simulation, persona control, state tracking, restaurant ordering

## TL;DR
A three-agent collaborative framework for realistic user simulation is proposed, comprising a User Agent (coordination), a State Tracking Agent (structured task state), and a Message Attributes Generation Agent (behavior attribute control conditioned on persona and state). On a restaurant ordering scenario, the framework achieves a 102.6% improvement in composite realism score (CRRS), +19.9% in persona adherence, and +284.5% in behavioral variability. A core finding is that behavior control without state awareness yields BVS = 0 (completely rigid behavior).

## Background & Motivation

**Background**: Conversational AI systems require large-scale testing. LLM-driven user simulation is a viable alternative to human testing, yet single-model approaches struggle to balance persona consistency, task accuracy, and behavioral naturalness simultaneously.

**Limitations of Prior Work**: (a) A single LLM bears the concurrent responsibilities of state tracking, behavior modeling, and response generation, causing persona drift due to role overload; (b) static test sets fail to capture the dynamics of multi-turn dialogue; (c) decision traces lack interpretability.

**Key Challenge**: Simulation must simultaneously satisfy three mutually constraining objectives — strict persona adherence (consistency), accurate task completion (state tracking), and natural behavioral variation (diversity).

**Goal**: By decomposing responsibilities across specialized agents, each agent is made responsible for one core dimension, enabling high-quality user simulation through collaboration.

**Key Insight**: An analogy to human cognition — working memory (State Tracking Agent monitors progress), behavioral planning (MAG Agent determines *how* to respond based on persona and state), and language generation (User Agent synthesizes the final response).

**Core Idea**: Three specialized agents are each responsible for "progress tracking / behavior decision / response generation," collaborating through structured protocols to realize interpretable, controllable, and realistic user simulation.

## Method

### Overall Architecture
In each dialogue turn, the User Agent receives the system message, sequentially invokes the State Tracking Agent and the MAG Agent, and then synthesizes both outputs to generate a response. The termination condition is: $\mathcal{T}_{current} \supseteq \mathcal{T}_{target}$.

### Key Designs

1. **State Tracking Agent**:

   - **Function**: Maintains a structured task state — confirmed items vs. target state.
   - **Mechanism**: Operates on a structured list via add/remove/clear tools; extracts confirmed task items from the input message each turn and updates the state accordingly.
   - **Design Motivation**: Decouples state tracking from natural language reasoning; uses structured data representation to prevent forgetting or hallucination.

2. **Message Attributes Generation Agent**:

   - **Function**: Dynamically determines behavioral attributes (mood\_tone / task\_execution\_style / exploration\_style / task\_completion\_status) conditioned on persona and current progress.
   - **Mechanism**: Conditioned on persona biography and current state: $a_t = f_{msgAttrGen}(p_{bio}, s_t)$.
   - **Design Motivation**: Makes the decision of "how to respond" explicit, rendering every response traceable. Core finding: behavior control without state awareness (Config4) yields BVS = 0 — completely rigid behavior.

3. **Protocol Constraints**:

   - Strict sequential invocation: State Tracking → MAG.
   - Monotonic state updates (only explicit additions/deletions, no implicit modifications).
   - Persona boundary constraints.

### Loss & Training
No training is required — the system is a ready-to-use agent built on GPT-4o (implemented with Pydantic AI).

## Key Experimental Results

### Main Results (60 test cases × 5 configurations)

| Configuration | PAS | BVS | TRA | DEI | CRRS |
|---|---|---|---|---|---|
| Config1 (Single LLM baseline) | 0.589 | 0.218 | 0.608 | 0.000 | 0.404 |
| Config2 (User Agent only) | 0.585 | 0.485 | 0.582 | 0.200 | 0.487 |
| Config3 (+ ST Agent) | 0.554 | 0.689 | **0.785** | 0.498 | 0.651 |
| Config4 (+ MAG Agent) | **0.661** | 0.000 | 0.602 | 0.432 | 0.462 |
| **Config5 (Full system)** | 0.706 | **0.839** | **0.785** | **0.994** | **0.818** |

### Statistical Significance

| Metric | p-value | Gain |
|---|---|---|
| PAS | 0.0037** | +19.9% |
| BVS | 0.0000*** | +284.5% |
| TRA | 0.0047** | +29.1% |
| CRRS | 0.0000*** | **+102.6%** |

### Key Findings
- **Behavioral diversity requires state awareness**: Config4 (MAG without ST) yields BVS = 0 — without task progress variation, persona degenerates into a fixed template.
- **ST independently determines task accuracy**: TRA is identical between Config3 and Config5 (0.785).
- **Synergistic effect exceeds the sum of individual components**: Config5 > Config3 + Config4.
- Token overhead doubles (14,789 vs. 6,618) but remains acceptable.

## Highlights & Insights
- **"Behavioral diversity requires state awareness"** is a profound insight from cognitive science — human behavioral variation is a response to situational progress, not a random process.
- **Near-perfect interpretability** (DEI = 0.994): every decision is traceable to persona and state.
- The evaluation metric system (5 complementary metrics) is a noteworthy methodological contribution.

## Limitations & Future Work
- Only a single restaurant ordering scenario is evaluated; cross-domain generalization is unknown.
- The scale is limited — 20 personas and 60 test cases.
- Computational overhead doubles (+124% tokens, +356% latency).
- No direct comparison against human simulators is provided.
- The behavioral attribute space is fixed (4 dimensions × limited values), insufficient to capture more complex human behaviors.
- Only English is supported.

## Related Work & Insights
- **vs. Park et al. (Generative Agents)**: Uses memory and reflection to simulate an open world. This paper focuses more narrowly on task-oriented dialogue, replacing implicit memory with explicit state tracking.
- **vs. Single LLM simulation**: A single LLM struggles to simultaneously maintain persona and track state. This paper resolves this via responsibility decomposition.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-agent user simulation architecture is novel; the finding that "behavior requires state awareness" is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablation across 5 configurations with statistical significance validation.
- Writing Quality: ⭐⭐⭐⭐ Method and evaluation design are detailed and rigorous.
- Value: ⭐⭐⭐⭐ Offers a practical contribution to the methodology of conversational AI testing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)
- [\[NeurIPS 2025\] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)
- [\[NeurIPS 2025\] KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills](kungfubot_physics-based_humanoid_whole-body_control_for_learning_highly-dynamic_.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)

</div>

<!-- RELATED:END -->
