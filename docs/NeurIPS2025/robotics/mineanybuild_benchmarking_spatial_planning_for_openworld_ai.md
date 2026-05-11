---
title: >-
  [Paper Note] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks)][Robotics][spatial planning] MineAnyBuild is a spatial planning benchmark built upon Minecraft, requiring AI agents to generate executable blueprint matrices from multimodal instruct…
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks)"
  - "Robotics"
  - "spatial planning"
  - "Minecraft"
  - "open-world agent"
  - "MLLM benchmark"
  - "spatial intelligence"
date: 2026-05-08
content_hash: a01d46c29d64bbe8
---

# MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents

**Conference**: NeurIPS 2025 (Datasets & Benchmarks)  
**arXiv**: [2505.20148](https://arxiv.org/abs/2505.20148)  
**Code**: [Project Page](https://mineanybuild.github.io/)  
**Area**: Robotics  
**Keywords**: spatial planning, Minecraft, open-world agent, MLLM benchmark, spatial intelligence

## TL;DR

MineAnyBuild is a spatial planning benchmark built upon Minecraft, requiring AI agents to generate executable blueprint matrices from multimodal instructions. The benchmark comprises 4,000 tasks and 500+ architectural/decorative assets, and systematically evaluates MLLM spatial planning capabilities across four dimensions: spatial understanding, spatial reasoning, creativity, and spatial commonsense. Results reveal that even GPT-4o achieves only 41.02/100 overall, with open-source models performing substantially worse.

## Background & Motivation

1. **Spatial planning as a core capacity of spatial intelligence**: It requires understanding and planning the arrangement of objects in 3D space, with broad applications in robotic manipulation, automated assembly, and urban planning.
2. **Critical limitations of existing benchmarks**: Benchmarks such as VSI-Bench, SpatialVLM, and Lego-Puzzles predominantly evaluate spatial understanding and reasoning in a VQA format (e.g., "Is this object on the left or the right?"). However, **a substantial gap exists between abstract spatial understanding and concrete task execution**—the ability to answer spatial relation questions does not imply the ability to perform spatial planning.
3. **Unique value of Minecraft building tasks**: The discrete block world simplifies evaluation (via precise coordinate alignment) while preserving the core challenges of 3D spatial planning. Architectural creation is central to Minecraft's appeal to millions of players, naturally affording openness and creative freedom.

## Core Problem

Can MLLM-based agents translate spatial understanding capabilities into concrete, executable spatial plans? How large is the gap between "knowing spatial relations" and "generating 3D architectural blueprints"?

## Method

### Overall Architecture

- **Platform**: Minecraft 3D block world
- **Input**: Multimodal human instructions (textual descriptions + reference images)
- **Output**: Executable blueprint matrices, automatically instantiated as in-game structures via the mineflayer simulator
- **Scale**: 4,000 curated tasks + 500+ architectural/interior decoration assets + ~2,000 VQA pairs
- **Extensibility**: Provides an infinitely scalable data collection paradigm leveraging the vast volume of player-generated UGC content available online

### Five Task Categories

1. **Executable Spatial Plan Generation**: Given abstract architectural instructions (e.g., "build an apple with a stem made of black_terracotta"), the agent must reason about sub-structure decomposition and compositional connections to generate an executable 3D blueprint matrix. This is analogous to the real-world process of an architect translating textual requirements into construction drawings.
2. **Spatial Understanding**: Given step-by-step instructions that include relative coordinate mappings for each block (e.g., Layer 2: "red_wool": [(0,0),(1,0)]), the agent must convert relative coordinates into a complete blueprint matrix, simulating the cognitive mapping between egocentric and allocentric spatial perspectives.
3. **Creativity**: Given an instruction, the agent must brainstorm block combination schemes that satisfy structural constraints while maximizing aesthetic creativity (e.g., designing Chinese-style or castle-style roofs using various stair and slab blocks). Evaluation is conducted via an MLLM critic model combined with human assessment.
4. **Spatial Reasoning**: Inspired by the classic psychological mental rotation paradigm, 48 geometric stimuli composed of blocks are constructed to generate 1,900 tasks. The agent must determine whether a rotated geometric object matches a reference stimulus, with distractors created via mirror flipping and additional rotations.
5. **Spatial Commonsense**: Evaluates the agent's intuitive understanding of everyday spatial conventions, such as "refrigerators should not be placed in bathrooms" or the appropriate orientation of a bedroom, assessing the reasonableness of object placement.

### Data Construction Pipeline

1. **Data Collection**: Approximately 7,000 buildings are collected from GrabCraft and the Minecraft Official Wiki; ~500 interior decoration assets are collected from creator-sharing platforms; spatial reasoning stimuli are designed following the Shepard & Metzler mental rotation experiment.
2. **Quality Control**: Automated code-based filtering combined with human review to remove problematic data.
3. **Data Annotation**: Instructions are annotated manually or with MLLM assistance to guide agents toward reasoning about sub-structure decomposition and spatial planning; spatial commonsense VQA pairs are manually designed.

### Evaluation Metrics

- **Executable Spatial Plan Generation / Creativity / Spatial Commonsense**: GPT-4.1 serves as the critic model, scoring outputs across multiple sub-dimensions with weighted aggregation into a composite score (maximum 10 points); non-executable outputs receive a score of 0.
- **Spatial Understanding**: Scored by the critic model analogously.
- **Spatial Reasoning**: Evaluated via direct VQA accuracy (%).
- **Overall**: Weighted sum across the five dimensions (maximum 100).

## Key Experimental Results

### Main Results

Thirteen MLLMs (7 closed-source + 6 open-source) are evaluated in a zero-shot setting:

| Model | Exec. Planning | Spatial Underst. | Spatial Reason. | Creativity | Spatial Comm. | Overall |
|-------|---------------|-----------------|----------------|-----------|--------------|---------|
| GPT-4o | 3.27 | 4.75 | 24.4% | 2.73 | 7.32 | **41.02** |
| Claude-3.7-Sonnet | 3.48 | 5.07 | 17.6% | 3.10 | 6.94 | 40.70 |
| Gemini-1.5-Pro | 3.53 | 4.80 | 16.9% | 2.73 | 7.52 | 40.54 |
| GPT-4o-mini | 2.08 | 2.52 | **26.7%** | 2.38 | 7.14 | 33.58 |
| Qwen2.5VL-7B | 1.29 | 1.12 | 16.0% | 1.34 | 6.30 | 23.30 |
| InternVL2.5-8B | 0.68 | 0.62 | 20.4% | 0.66 | 5.62 | 19.24 |

### Key Findings

1. **Uniformly poor overall performance**: The strongest model, GPT-4o, achieves only 41.02/100—less than half the maximum—demonstrating that spatial planning remains a major challenge for current MLLMs.
2. **Substantial gap between open-source and closed-source models**: Closed-source models achieve an average output success rate (OSR) of 93.40%, whereas open-source models fall far below this threshold; many open-source models entirely fail to generate valid 3D blueprint matrices.
3. **Counterintuitive spatial reasoning results**: GPT-4o-mini (26.7%) outperforms GPT-4o (24.4%), indicating that stronger general capability does not imply superior spatial reasoning; most models score below 25%, approaching chance level.
4. **Spatial commonsense is relatively strongest**: Closed-source models score 6.8–7.5 out of 10, suggesting that MLLMs possess a degree of everyday spatial commonsense, yet this does not transfer to executable spatial planning.

### Three Core Failure Modes

1. **Spatial Misunderstanding**: Agents frequently misinterpret 3D positional relations—for example, Claude-3.5-Sonnet stacks the Olympic rings vertically rather than arranging them coplanar—reflecting a lack of spatial grounding capability.
2. **Implementation Gap**: Agents can articulate reasonable planning strategies in natural language but fail to translate them into precise blueprint matrices, with block index errors, orientation errors, and spatial logic inconsistencies leading to parsing and execution failures.
3. **Structural Degeneration**: When tasks require non-cubic, asymmetric, or creative designs, agents tend to degenerate toward simple box-shaped outputs or incoherent results, revealing limited ability to scale from basic patterns to complex architectural concepts.

## Highlights & Insights

- **The transition from "understanding" to "execution" is the core bottleneck in agent spatial intelligence**: There is virtually no positive correlation between VQA accuracy and spatial planning capability, which serves as an important warning for the broader spatial intelligence research community.
- **The incorporation of mental rotation experiments into AI evaluation is an elegant design**: Leveraging a classical cognitive psychology paradigm to construct spatial reasoning tests provides theoretical grounding while enabling controlled difficulty.
- **The infinitely extensible data collection paradigm is particularly well-designed**: By automatically extracting block information from player-built structures via mineflayer, filtering air blocks, and generating a standardized data format, the benchmark supports continuous expansion.
- **Introducing creativity as an evaluation dimension**: Although inherently subjective, creativity is an unavoidable dimension in AGI assessment; the dual-mechanism of MLLM critic combined with human evaluation mitigates evaluation bias.
- **The concept of Implementation Gap merits broader adoption**: It can be transferred to other agent task settings—many agents can "describe" but cannot "execute."

## Limitations & Future Work

- The discrete Minecraft block world substantially simplifies the complexity of continuous spatial planning, as issues such as continuous coordinate handling, collision detection, and physical constraints are absent.
- Creativity evaluation relies on GPT-4.1 as the critic model, whose own spatial understanding is limited, raising concerns about the reliability of creativity scores.
- Only zero-shot settings are evaluated; the effects of few-shot prompting, chain-of-thought reasoning, and domain-specific fine-tuning on spatial planning performance are not explored.
- No systematic comparison with RL-based agents (e.g., VPT, Voyager) is conducted; the benchmark focuses exclusively on MLLM-based approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first benchmark to systematically evaluate MLLM spatial *planning* (rather than spatial understanding) capability, filling the evaluation gap between comprehension and execution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 4,000 tasks × 13 models × 5-dimensional evaluation + detailed failure case analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rigorous task definitions, and rich visualizations.
- **Value**: ⭐⭐⭐ An important reference for evaluating agent spatial planning capability; the Implementation Gap concept is transferable to other agent research contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](../../AAAI2026/robotics/when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](coopera_continual_open-ended_human-robot_assistance.md)
- [\[NeurIPS 2025\] Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling](rethinking_the_simulation_vs_rendering_dichotomy_no_free_lunch_in_spatial_world_.md)
- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)

</div>

<!-- RELATED:END -->
