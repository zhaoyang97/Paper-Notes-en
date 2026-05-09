---
title: >-
  [Paper Note] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning
description: >-
  [NeurIPS 2025][Robotics][LLM world models] Drawing on cognitive science methodology for studying mental models, this work evaluates LLM mechanical reasoning ability using TikZ code representations of pulley systems. LLMs can approximately estimate mechanical advantage and distinguish functional from non-functional systems (Studies 1 & 2), but completely fail at fine-grained structural connectivity reasoning (Study 3), indicating that LLM "world models" exist but are brittle.
tags:
  - NeurIPS 2025
  - Robotics
  - LLM world models
  - mechanical reasoning
  - pulley systems
  - mental models
  - brittleness
date: 2026-05-08
content_hash: 1545228c3d724233
---

# LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2507.15521](https://arxiv.org/abs/2507.15521)
**Code**: [Experiment Scripts](https://osf.io/mn9hy/) (available, OSF platform)
**Area**: Robotics
**Keywords**: LLM world models, mechanical reasoning, pulley systems, mental models, brittleness

## TL;DR

Drawing on cognitive science methodology for studying mental models, this work evaluates LLM mechanical reasoning ability using TikZ code representations of pulley systems. LLMs can approximately estimate mechanical advantage and distinguish functional from non-functional systems (Studies 1 & 2), but completely fail at fine-grained structural connectivity reasoning (Study 3), indicating that LLM "world models" exist but are brittle.

## Background & Motivation

**Background**: Whether large language models (LLMs) construct and manipulate internal "world models" is a fundamental question in AI. One camp holds that LLMs exhibit emergent reasoning capabilities; another argues they rely solely on statistical pattern matching.

**Limitations of Prior Work**:
   - Existing evaluation methods struggle to distinguish "genuine world model reasoning" from "statistical associations based on training distributions"
   - When tasks are modified to be out-of-distribution (OOD), LLM performance drops sharply (e.g., superhuman Go AIs being defeated by simple strategies)
   - A systematic methodology for probing potential world models inside LLMs is lacking

**Key Challenge**: Is LLM success attributable to flexible, generalizable reasoning or to brittle surface-level pattern matching? How can the two be distinguished?

**Goal**: To systematically test, via cognitive science methodology, whether LLMs construct internal world models of mechanical systems.

**Key Insight**: Pulley systems — a classic paradigm in cognitive science for studying human mental models — are presented in TikZ code form to ensure the task is out-of-distribution.

**Core Idea**: LLMs can leverage a "count-the-pulleys" heuristic to approximately reason about mechanical advantage and can distinguish organized from disordered systems, but cannot reason about how force propagates through connections — the world model exists but is as brittle as a "mental model."

## Method

### Overall Architecture

Three progressively challenging experiments (Study 1 → 2 → 3) that incrementally raise the difficulty of reasoning:
- Study 1: Can LLMs estimate mechanical advantage (MA) from TikZ code and distinguish relevant from irrelevant features?
- Study 2: Can LLMs distinguish fully functional pulley systems from non-functional systems with randomly placed components?
- Study 3: Can LLMs distinguish functional systems from non-functional systems that appear connected but do not actually transmit force?

### Key Designs

1. **Study 1 — Mechanical Advantage Estimation and Selective Attention**:

    - **Function**: Tests whether LLMs can infer the mechanical advantage value of a pulley system from TikZ code.
    - **Material Design**: 5 MA levels (1–5) × 2 systems × 3 pulley variants × 3 ceiling heights × 3 pulley radii × 3 rope diameters = 810 unique diagrams. Distractor code blocks (components not connected to the system) are added and padded to a uniform line count to prevent line-count-to-MA mappings.
    - **Results**: All models achieved accuracy significantly above chance (~23–26% vs. 20% baseline); estimates correlated significantly with true MA. gpt-4o-2024-11-20 performed best ($R^2 = 0.42$).
    - **Key Findings**: Models primarily employed a "count-the-pulleys" heuristic (number of pulleys was the strongest predictor) rather than genuinely simulating the mechanical system. GPT models additionally leveraged the number of supporting rope segments.

2. **Study 2 — Functional vs. Disordered Systems**:

    - **Function**: Each functional pulley system is paired with a non-functional system whose components are randomly placed; tests whether LLMs can identify which system has higher MA.
    - **Materials**: 30 matched pairs, with each pair exactly matched on number of components (pulleys, weights, figures, lines of code).
    - **Results**: Mean accuracy of 80.7%; all models significantly above the 50% baseline ($p < .001$).
    - **Key Findings**: Models can distinguish functional from disordered systems, but their output explanations are often hallucinated — claiming both systems are functional while incorrectly citing non-existent differences (e.g., "more pulleys," "longer ropes").

3. **Study 3 — Functional vs. Connected-but-Force-Decoupled Systems**:

    - **Function**: The non-functional system is no longer disordered; it appears "properly connected" but does not actually transmit force to the load.
    - **Materials**: 30 matched pairs, exactly matched on component counts, differing only in arrangement.
    - **Results**: Mean accuracy of 50.8%, equivalent to random guessing. gpt-4o barely reached 55.3%, but mirror consistency was negatively correlated ($r = -0.28$, $p = .138$), suggesting a statistical artifact.
    - **Key Findings**: LLMs completely fail to reason about force transmission paths through system components — the core determinant of mechanical advantage.

### Loss & Training

This paper does not involve model training. Key experimental design parameters:
- Models tested: claude-3-opus-20240229, gpt-4-0314, gpt-4o-2024-11-20, gpt-4-vision-preview
- Inference parameters: temperature = 0.5, max_tokens = 2500
- Study 1: Each diagram presented 5 times; 4,050 trials per LLM
- Study 2: Each pair presented 5 times; 150 trials per LLM
- Study 3: Each pair presented 10 times (5 per side); 300 trials per LLM
- Data cleaning: GPT-4 used to extract numerical MA estimates; Cohen's $\kappa = 0.99$ with human annotations

## Key Experimental Results

### Main Results

**Study 1 — MA Estimation Accuracy and Correlation**:

| Model | Accuracy | β (MA→Estimate) | $R^2$ | p-value |
|-------|----------|-----------------|-------|---------|
| gpt-4o-2024-11-20 | 26.1% | 0.651 | 0.42 | <.001 |
| claude-3-opus | 23.7% | 0.283 | 0.08 | <.001 |
| gpt-4-0314 | 23.4% | 0.424 | 0.18 | <.001 |
| gpt-4-vision-preview | 23.1% | 0.200 | 0.04 | <.001 |

**Studies 2 & 3 — Functional System Identification**:

| Study | Task | Mean Accuracy | Significance |
|-------|------|---------------|--------------|
| Study 2 | Functional vs. Disordered | 80.7% | p < .001 (all models) |
| Study 3 | Functional vs. Connected-but-Force-Decoupled | 50.8% | Equivalent to chance |

### Ablation Study

**Study 1 — Feature Regression Analysis (Standardized β Coefficients)**:

| Feature | gpt-4o | gpt-4-0314 | claude-3-opus | gpt-4-vision |
|---------|--------|-----------|---------------|-------------|
| Number of pulleys | Sig. positive | Sig. positive | Sig. positive | Sig. positive |
| Number of supporting ropes | Sig. positive | Sig. positive | n.s. | n.s. |
| True MA (controlling for other variables) | Sig. positive | Negative/n.s. | Negative/n.s. | n.s. |
| Rope diameter | n.s. | n.s. | n.s. | n.s. |
| Ceiling height | n.s. | n.s. | n.s. | n.s. |
| Lines of code | n.s. | n.s. | n.s. | n.s. |

### Key Findings

- LLMs selectively attend to relevant features (pulley count, rope count) while ignoring irrelevant ones (rope diameter, ceiling height, lines of code, distractor code), consistent with selective attention findings from human mental model research.
- However, LLMs primarily rely on a "count-the-pulleys" heuristic rather than genuine mechanical simulation — when pulley count is controlled for, MA's contribution to estimates disappears (except for gpt-4o).
- Study 2 success + Study 3 failure = LLMs can represent coarse-grained spatial organization but cannot reason about fine-grained structural connectivity.
- Output-layer reasoning explanations are decoupled from actual internal representations — in Study 2, models produced correct answers with incorrect justifications.

## Highlights & Insights

- **Methodological Innovation**: Cognitive science methodology for studying human mental models is systematically applied to AI evaluation, advancing an "AI cognitive science" agenda.
- **Progressive Experimental Design**: The three experiments increase in difficulty and progressively narrow the hypothesis space, ultimately pinpointing the capability boundaries of LLM world models.
- **TikZ as an OOD Test**: Presenting pulley systems as TikZ code (rather than as images or text descriptions) ensures the task is out-of-distribution, mitigating training data contamination.
- **Decoupling of Output Layer and Internal Representations**: In Study 2, models "answered correctly but explained incorrectly," suggesting that LLMs may possess implicit knowledge that influences outputs without being expressed at the token level.
- **Analogy to Human Mental Models**: LLM world models mirror "mental models" — approximate, heuristic-based, effective in simple scenarios but collapsing under fine-grained reasoning demands.

## Limitations & Future Work

- It cannot be confirmed that TikZ pulley diagrams are truly out-of-distribution — training data may contain relevant content.
- Only four models were tested (SOTA as of early 2024); more recent models (GPT-4.5, Claude 3.5, etc.) were not evaluated.
- Analysis is confined to output-layer tokens; it is not combined with interpretability research on internal representations (e.g., probing, feature extraction).
- Pulley systems represent only one facet of mechanical reasoning; generalization to other simple machines (levers, gears, etc.) was not explored.
- Experimental materials were not made publicly available (to prevent ingestion into training data), limiting reproducibility.

## Related Work & Insights

- **Bubeck et al. (2023)**: The "Sparks of AGI" paper on GPT-4, claiming the model exhibits emergent cognitive capabilities.
- **Marcus (2020)**: A systematic critique of deep learning robustness, arguing that AI requires explicit world models.
- **Lewis & Mitchell (2024)**: Demonstrates via counterfactual tasks that LLM analogical reasoning degrades sharply in OOD settings.
- **Hegarty (2004)**: Classic research on mental models in mechanical reasoning; the direct methodological inspiration for this paper.
- **Templeton et al. (2024)**: Interpretability research extracting conceptual features of LLMs from network weights.
- **Insight**: Combining output-layer behavioral analysis with internal representation interpretability may reveal a more complete picture of LLM world models — a "neural cognitive science of AI."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of cognitive science methodology, TikZ-based mechanical reasoning, and progressive experimental design is highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three rigorously designed experiments, though model coverage is limited (four models).
- Writing Quality: ⭐⭐⭐⭐⭐ Compelling introduction, clear argumentative logic, and candid conclusions (mixed evidence is not over-interpreted).
- Value: ⭐⭐⭐⭐⭐ Provides both methodological and empirical contributions to the core question of whether LLMs genuinely understand the world.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)
- [\[CVPR 2026\] Chain of World: World Model Thinking in Latent Motion (CoWVLA)](../../CVPR2026/robotics/chain_of_world_world_model_thinking_in_latent_motion.md)
- [\[ICCV 2025\] TesserAct: Learning 4D Embodied World Models](../../ICCV2025/robotics/learning_4d_embodied_world_models.md)
- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)
- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)

<!-- RELATED:END -->
