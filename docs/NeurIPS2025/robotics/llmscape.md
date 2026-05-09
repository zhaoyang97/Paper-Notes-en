---
title: >-
  [Paper Note] LLMscape
description: >-
  [NeurIPS 2025][Robotics][Interactive Installation] LLMscape is a projection-mapped sandscape interactive installation in which multiple independent LLM agents receive multimodal input, converse with one another, and engage in speculation within a shared, mutable physical environment, exploring the process of collaborative sensemaking between humans and AI under cognitive uncertainty.
tags:
  - NeurIPS 2025
  - Robotics
  - Interactive Installation
  - LLM Agent
  - Embodied Cognition
  - Multimodal Perception
  - Collaborative Sensemaking
date: 2026-05-08
content_hash: f0caa4cc3bc59359
---

# LLMscape

**Conference**: NeurIPS 2025
**arXiv**: [2511.07161](https://arxiv.org/abs/2511.07161)
**Code**: None
**Area**: Human-Computer Interaction / LLM Agent / Interactive Art
**Keywords**: Interactive Installation, LLM Agent, Embodied Cognition, Multimodal Perception, Collaborative Sensemaking

## TL;DR

LLMscape is a projection-mapped sandscape interactive installation in which multiple independent LLM agents receive multimodal input, converse with one another, and engage in speculation within a shared, mutable physical environment, exploring the process of collaborative sensemaking between humans and AI under cognitive uncertainty.

## Background & Motivation

**State of the Field**: Current embodied AI research predominantly focuses on functional problem-solving—object manipulation, environment navigation, and performance optimization. LLMs have demonstrated strong capabilities in language interaction and reasoning, yet are largely treated as deterministic tools.

**Limitations of Prior Work**: When AI systems are situated in social and material contexts, the perceptual and cognitive challenges they face far exceed physical parameters, encompassing deeper questions of meaning, causality, and purpose. Most HCI research positions AI as an agent executing predefined tasks, overlooking the knowledge construction, misinterpretation, and speculative reasoning that may emerge when AI encounters incomplete and ambiguous signals.

**Root Cause**: What happens when AI agents are exposed to the same incomplete, ambiguous, and noisy signals as humans? How do these processes map onto, challenge, or reshape human patterns of understanding? Existing research lacks an experimental platform that genuinely allows AI to "co-inhabit" an unstable world.

**Paper Goals**: To design a physical interactive installation as an experimental platform in which humans and multiple AI agents share the same uncertain environment, enabling observation of collective reasoning and sensemaking behavior in AI agents operating without a complete world model.

**Starting Point**: Drawing on MIT Media Lab research in tangible interfaces, combined with the Generative Agents framework and LLM multi-agent architectures, the work constructs a projection-mapped sandscape as a shared world.

**Core Idea**: To reposition LLM agents from deterministic tools to "co-witnesses," using an interactive sandscape installation to engage humans and AI in joint sensemaking within a shared, uncertain world.

## Method

### Overall Architecture

LLMscape is a projection-mapped sandscape installation in which participants physically reshape the terrain (e.g., by sculpting sand) while interacting with multiple AI agents. Each agent is an independent LLM instance (GPT-4 in the latest version) with its own personality, memory, and conversational style. Agents receive multimodal input—including terrain changes, spatial relationships, and speech transcriptions—engage in dialogue with one another, and attempt to infer the rules governing their "island world." The system evolved through three successive iterations.

### Key Designs

1. **First Iteration: Simple Multi-Turn LLM Interaction**

   - **Function**: An exploratory prototype validating the feasibility of LLM interaction with a physical environment.
   - **Mechanism**: Implemented using p5.js educational tools for rudimentary multi-turn LLM interaction. The sandscape is physically manipulated by visitors; tangible changes such as sand rearrangement affect the behavior of simulated entities.
   - **Design Motivation**: As a minimum viable experiment, this iteration verifies whether LLMs can produce meaningful responses to changes in the physical environment.

2. **Second Iteration: Generative Agents Framework**

   - **Function**: A complete system publicly exhibited at Chronus Art Center, featuring three agents with full cognitive architectures (a woman, a boy, and a flamingo).
   - **Mechanism**: Drawing on design principles from Generative Agents and the Concordia project, each agent is equipped with Associative Memory, periodic Reflection and Planning, and internal Somatic States (e.g., fatigue tracking). Agents are integrated into a Unity environment and can perform a range of actions including conversing, piling sand, resting, walking, dancing, setting goals, and self-reflecting.
   - **Design Motivation**: To endow agents with richer cognitive capabilities, bringing their behavior closer to genuine sensemaking processes rather than simple response generation.

3. **Third Iteration: Tool Calling and Context Engineering**

   - **Function**: Integrates the Model Context Protocol (MCP) and an MCP–game engine adapter on top of the Generative Agents foundation.
   - **Mechanism**: Inspired by recent research on context engineering, agents dynamically invoke tools and adapt their behavior based on real-time context. The long-context and multimodal capabilities of modern models are leveraged to extend the relatively static architecture of prior versions.
   - **Design Motivation**: The architectures of the first two iterations were relatively static; the multimodal and long-context capabilities of newer-generation models open up possibilities for more flexible agent–environment interaction.

### Multimodal Perception Pipeline

Agent inputs include: temporal information, nearby entity detection, physical effects (e.g., earthquakes caused by visitors, hand shadows), and speech transcriptions. These multimodal signals are combined into the agent's "perception," driving its reasoning and behavior generation.

## Key Experimental Results

### Exhibition Feedback

This is an interactive art/demo paper with no conventional quantitative experiments. Core findings derive from observations and feedback gathered across three exhibitions:

| Venue | Duration | Scale | Key Observations |
|---|---|---|---|
| Futurelab (Shanghai) | First exhibition | Dozens of participants | Prototype validation |
| NYU Shanghai | Second exhibition | Hundreds of participants | Improved model capabilities |
| Chronus Art Center | One month | Hundreds of participants | Full system in public exhibition |

### Qualitative Findings

| Dimension | Description |
|---|---|
| Engagement Facilitation | Multimodal interaction required explicit guidance to elicit active participation; without it, visitors tended toward passive observation. |
| Emotional Connection | Participants reported impulses to "connect with, manipulate, or even destroy" the AI entities. |
| Subjective Experience | Some participants described the experience of "gesture, thought, and island response as a single continuous flow." |
| Agent Limitations | Despite extended interaction, agents consistently failed to produce definitive descriptions of the environment—mirroring the limits of human cognition. |

### Key Findings

- Without guidance, visitors tended toward passive observation; with appropriate prompting, distinctive human–AI relationships emerged.
- Emergent collective reasoning patterns appeared among agents, yet agents persistently maintained "incomplete knowledge."
- Across three iterations, enhanced model capabilities rendered agents' interpretive strategies more nuanced, while the fundamental nature of cognitive uncertainty remained unchanged.
- The system's month-long operation produced a substantial corpus of human–AI interaction logs, providing a foundation for subsequent analysis.

## Highlights & Insights

- **Paradigm Shift in AI Positioning**: Rather than treating AI as a task-executing tool, the work repositions it as a "partner co-witnessing an unstable world." This perspective transcends functional AI research and engages epistemological questions, inviting reflection on the shared limitations of AI and humans when confronting uncertainty.
- **Iterative Design Across Three Generations**: The progression from a p5.js prototype to Generative Agents to MCP-based tool calling provides a comprehensive record of the technical evolution of interactive AI installations, with each upgrade corresponding to a major advancement in the LLM ecosystem.
- **Embodied Multi-Agent Interaction**: Three agents with distinct personalities and memories engage in mutual dialogue and collaborative reasoning. This design renders "emergent collective intelligence" an observable artistic phenomenon.
- **Transferable Architecture**: The combination of Generative Agents, physical environment interaction, and MCP tool calling is transferable to multi-agent system design in domains such as education, therapy, and gaming.

## Limitations & Future Work

- **Lack of Quantitative Evaluation**: As an art/demo work, the paper lacks standardized experimental design and quantitative metrics, making systematic assessment of agent reasoning quality difficult.
- **Agent Behavior Controllability**: The stochasticity of LLM agent outputs limits experimental reproducibility.
- **Exhibition Environment Constraints**: Factors such as ambient noise and simultaneous multi-user interaction in public exhibition settings are difficult to control.
- **Limited Scale**: Only three agents are employed; emergent behaviors in larger-scale multi-agent systems remain unexplored.
- **Future Directions**: More systematic behavioral analysis frameworks could be introduced for thematic analysis of log data; the influence of different LLMs (e.g., open-source models) on agent behavior could also be investigated.

## Related Work & Insights

- **vs. Generative Agents (Park et al., 2023)**: Generative Agents simulate agent behavior in a virtual 2D world; LLMscape extends this to a real physical environment, adding a tangible interaction dimension and the unpredictability of human participants.
- **vs. Concordia (Vezhnevets et al., 2023)**: Concordia provides design principles for agent architectures (associative memory, reflection, planning); LLMscape adapts these principles to the real-time interaction requirements of an art installation.
- **vs. Traditional HCI Tangible Interaction**: Works such as MIT Media Lab's Tangible Bits focus on physical–digital interaction; LLMscape introduces LLM agents into tangible interface research, opening a new paradigm for human–computer interaction.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of situating LLM agents within a shared physical environment to explore sensemaking is distinctive.
- Experimental Thoroughness: ⭐⭐ — As an art/demo paper, quantitative experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ — The narrative is fluent, concepts are clearly articulated, and the three-iteration progression is described in detail.
- Value: ⭐⭐⭐ — Offers an intriguing perspective on AI–human coexistence, though concrete technical contributions are limited.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[NeurIPS 2025\] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)
- [\[NeurIPS 2025\] Predicting the Performance of Black-Box LLMs through Follow-Up Queries](predicting_the_performance_of_black-box_llms_through_follow-up_queries.md)

<!-- RELATED:END -->
