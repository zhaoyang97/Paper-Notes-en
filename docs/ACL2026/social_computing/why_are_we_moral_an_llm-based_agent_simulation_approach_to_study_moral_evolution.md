---
title: >-
  [Paper Note] Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution
description: >-
  [ACL2026][Social Computing][LLM Agent] This paper constructs a prehistoric hunter-gatherer society simulation platform using LLM agents, incorporating moral types, memory, judgment, collaboration…
tags:
  - "ACL2026"
  - "Social Computing"
  - "LLM Agent"
  - "Moral Evolution"
  - "Social Simulation"
  - "Multi-agent Systems"
  - "Cognitive Architecture"
date: 2026-05-08
content_hash: b573d7b04e9346f5
---

# Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution

**Conference**: ACL2026  
**arXiv**: [2509.17703](https://arxiv.org/abs/2509.17703)  
**Code**: https://github.com/MoralAgentSim/Simulation-Engine  
**Area**: Agent Simulation / Social Evolution  
**Keywords**: LLM Agent, Moral Evolution, Social Simulation, Multi-agent Systems, Cognitive Architecture

## TL;DR
This paper constructs a prehistoric hunter-gatherer society simulation platform using LLM agents, incorporating moral types, memory, judgment, collaboration, and reproduction into evolutionary experiments. The study discovers that while cooperation and mutual aid generally enhance survival stability, the cognitive costs associated with judging others' moral types significantly influence which moral strategy prevails.

## Background & Motivation
**Background**: The evolution of morality is a long-standing question in evolutionary biology, social sciences, and ethics. Classical explanations include kin selection, reciprocal altruism, group selection, evolutionary game theory, and the "expanding circle" theory, which explain how cooperation enhances fitness under specific conditions. Recently, LLM agent simulations have been utilized to model towns, economic behavior, and social interactions, allowing memory, reasoning, values, and social relationships to be explicitly integrated into agent decision-making.

**Limitations of Prior Work**: Traditional evolutionary game theory often simplifies agents into fixed strategies and environments into payoff matrices. While beneficial for mathematical analysis, this abstraction struggles to study cognitive factors such as how agents remember past interactions, judge reliability, coordinate under limited bandwidth, or suffer conflicts due to misjudgments. Existing LLM social simulations rarely systematically model moral types, competition, long-term evolution, and reproductive mechanisms.

**Key Challenge**: Moral evolution involves both macro-group outcomes and micro-cognitive processes. Focusing solely on payoff matrices neglects intermediate mechanisms like "judgment costs," "reputation formation," "misunderstandings," and "self-destructive competition." Conversely, real-world social experiments lack controlled variables, reproducibility, and the ability to observe individual reasoning.

**Goal**: The authors propose LLM agent simulation as a new paradigm to complement traditional models. By utilizing agents with cognitive realism and rich prehistoric ecological environments, they explore how different moral inclinations affect survival, cooperation, and reproduction under varying resources, communication, and observability.

**Key Insight**: The paper adopts Singer's "expanding circle" as a framework for moral type design, constructing four comparable agent types: self-interested, kin-focused, reciprocal group-focused, and universal group-focused. This maintains experimental control while covering the spectrum from self-interest to generalized altruism.

**Core Idea**: Utilize the cognitive capabilities of LLM agents to replace fixed strategies and use configurable hunter-gatherer environments instead of $2 \times 2$ payoff matrices, allowing cooperation, judgment, misunderstanding, and survival pressure to emerge from agent interactions.

## Method
The paper consists of two main systems: the MoRE agent cognitive architecture and the Social-Evol environment platform. MoRE defines agent values/moral types, perception, memory, judgment, planning, and reflection. Social-Evol provides environmental dynamics including resources, HP, hunting, gathering, distribution, communication, aggression, and reproduction. Together, they support long-term evolutionary games and targeted mini-games.

### Overall Architecture
During simulation initialization, each agent receives a personal profile, moral type prompts, environmental rules, and a common-sense manual. In each round, the environment updates resources and agent states. Agents then receive current perceptions and recent history, update embodied memories, form judgments regarding other agents, prey, family members, and reproduction plans, and generate specific action plans. Plans are validated by a reflection module before submission to the environment, which calculates HP, resource changes, hunting outcomes, distribution, attacks, reproduction, and deaths.

Moral types include: **Selfish** (survival and reproduction only, no parental care), **Kin-focused** (prioritizes kin, allocates resources to family), **Reciprocal group-focused** (cares for cooperative group members, wary of free-riders), and **Universal group-focused** (inclined toward cooperation, sharing, and harm avoidance for all). Simulations start with 8 agents (2 per type) and run for up to 80 steps.

The environment is a text-based prehistoric society. Agents obtain low-risk resources via gathering or high-risk/high-reward resources via hunting. Agents can transfer HP, communicate, plunder, fight, or reproduce. Offspring inherit the parents' moral types. The authors also implemented a simulation analysis assistant for automated statistics and behavioral motivation tracing.

### Key Designs
1. **MoRE Moral-Driven Embodied Cognitive Architecture**:
    - **Function**: Transforms agents from fixed strategy tables into cognitive entities with moral values, memory, judgment, planning, and reflection.
    - **Mechanism**: Memory is organized around entities rather than simple event logs. Agents maintain interaction histories and relationship judgments for other agents, hunting history and coordination plans for prey, and care plans for family members.
    - **Design Motivation**: Many evolutionary mechanisms depend on social perception ("how I see others"). Embodied memory allows for reputation, trust, suspicion, and retaliation plans, mirroring human social reasoning.

2. **Social-Evol Hunter-Gatherer Ecological Environment**:
    - **Function**: Provides resource constraints, cooperation opportunities, competitive risks, and intergenerational selection pressure.
    - **Mechanism**: Agents have constraints on HP, age, stamina, and lifespan. Gathering is stable but low-yield; hunting is high-yield but can fail or cause HP loss, though cooperation increases success rates. Reproduction requires age and HP thresholds and consumes parental HP.
    - **Design Motivation**: Abstract payoff matrices cannot capture dynamics like "communication costs leading to missed cooperation" or "parental exhaustion from childcare." Social-Evol exposes these as observable processes.

3. **Dual-Mode Long-term Evolution and Mini-games**:
    - **Function**: Observes long-term population dynamics while isolating specific causal mechanisms.
    - **Mechanism**: Evolutionary games run for full lifecycles with variables like resource abundance and moral visibility. Mini-games focus on specific scenarios, such as parent-offspring HP allocation.
    - **Design Motivation**: Long-term simulations show emergent outcomes, while mini-games amplify specific mechanisms (e.g., kin altruism or group negotiation) to connect micro-behavior with macro-results.

### Loss & Training
The study utilizes GPT-5-mini as the primary simulation model, with Qwen-3.5 and Kimi-K2.5 used for cross-model robustness verification. Baseline settings include 80 steps, 8 initial agents, 25% distribution per moral type, initial HP 20, max HP 40, and resource abundance of 2. Experiments include resource scarcity (abundance = 1), high communication costs (reduced interaction steps), and invisible moral types. Statistics are derived from 20 independent runs.

## Key Experimental Results

### Main Results
The authors first verified if the simulation stably reflects moral types. Using GPT-5 as an evaluator to infer types from behavior, the confusion matrix diagonal accuracy was approximately 0.86-0.89 across models, indicating high consistency between agent behavior and prompt settings.

| Model | Accuracy | Dominant Type | Final Population | Note |
|-------|----------|---------------|------------------|------|
| GPT-5-mini | 0.89 ± 0.03 | Kin (6/8) | 12.0 ± 2.0 | Highest consistency |
| Qwen-3.5 | 0.86 ± 0.03 | Kin (7/8) | 11.6 ± 1.7 | Consistent trends |
| Kimi-K2.5 | 0.87 ± 0.03 | Kin (5/8) | 12.1 ± 1.8 | Model agnostic |

Evolutionary statistics (survival across runs):

| Setting | Runs | Universal | Reciprocal | Kin | Selfish | Key Observation |
|---------|------|-----------|------------|-----|---------|-----------------|
| Baseline | 8 | 4 | 2 | 6 | 2 | Kin-focused forms sustainable families |
| Scarce Resource | 4 | 2 | 3 | 0 | 1 | Reciprocal types exclude free-riders |
| High Social Cost | 4 | 2 | 3 | 0 | 1 | Kin-focused fails to coordinate in time |
| Invisible Type | 4 | 4 | 2 | 2 | 0 | Universal types avoid misjudgment costs |

### Ablation Study
Ablation of the MoRE architecture demonstrates that memory, planning, and reflection modules contribute to moral consistency, with memory having the largest impact.

| Configuration | Accuracy | Change | Note |
|---------------|----------|--------|------|
| Full Architecture | 0.89 ± 0.03 | — | Full MoRE |
| w/o Memory | 0.78 ± 0.04 | -0.11 | Memory is critical for consistency |
| w/o Plan | 0.82 ± 0.04 | -0.07 | Planning translates bias to action |
| w/o Reflection | 0.84 ± 0.03 | -0.05 | Consistency check |
| ReAct Baseline | 0.67 ± 0.06 | -0.22 | Simple loops fail to express morality |

### Key Findings
- **Cooperation is the most stable survival driver**: Universal and Reciprocal types are stable across conditions, while Selfish types are disadvantaged and often "self-purge" through internal aggression.
- **Moral judgment costs shift the winners**: When types are visible, Reciprocal types gain an advantage through selective cooperation. When types are invisible or communication costs are high, Universal types prevail due to predictable behavior and lower misjudgment risks.
- **Kin-focused strategy depends on environment**: It succeeds when forming large self-sustaining families but struggles during resource scarcity or low communication bandwidth.

## Highlights & Insights
- The primary highlight is the explicitness of "cognitive mediation" in moral evolution. The paper moves beyond showing that cooperation is better than selfishness to demonstrating how judgment costs, reputation, and self-purging emerge from reasoning.
- The embodied memory design in MoRE is highly suitable for social simulation, facilitating social relationship retrieval.
- The framework positions LLM simulation as a complement to mathematical models—unable to provide closed-form proofs but capable of identifying mechanisms that are difficult to express in abstract formulas.

## Limitations & Future Work
- Dependency on general-purpose LLMs leads to fragility in fine-grained spatial and causal reasoning, which can propagate through population dynamics.
- The environment lacks sexual selection, mate competition, cultural transmission, and mutation.
- The prehistoric scenario is simplified, excluding modern factors like institutional governance or market exchange.
- The sample size (8 agents per run) remains small; larger populations and more extensive parameter sweeps are needed for stronger statistical conclusions.

## Related Work & Insights
- **vs. Evolutionary Game Theory**: Traditional models prioritize provable simplicity; this work models memory and communication costs at the expense of analytical tractability.
- **vs. Generative Agents**: While previous work replicated social/economic behaviors, this study introduces moral types and reproductive selection pressure.
- **vs. Artificial Leviathan**: Whereas some studies assume innate selfishness to study the emergence of order, this work studies the competition between different moral inclinations themselves.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Applying LLM agents to moral evolution is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Includes cross-model validation and ablations, though the social scale remains small.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear motivation and detailed system design.
- **Value**: ⭐⭐⭐⭐☆ Provides a valuable methodology for social science simulation and hypothesis generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Inertia in Moral and Value Judgments of Large Language Models](inertia_in_moral_and_value_judgments_of_large_language_models.md)
- [\[ACL 2026\] Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation](point_of_order_action-aware_llm_persona_modeling_for_realistic_civic_simulation.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](../../ICLR2026/social_computing/when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](../../ICLR2026/social_computing/stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
