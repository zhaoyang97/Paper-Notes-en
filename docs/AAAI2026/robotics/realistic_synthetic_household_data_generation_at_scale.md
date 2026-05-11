---
title: >-
  [Paper Note] Realistic Synthetic Household Data Generation at Scale
description: >-
  [AAAI 2026][Robotics][synthetic data generation] This paper proposes an LLM-driven bidirectional coupling generation framework that iteratively generates large-scale synthetic datasets — encompassing household environmen…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "synthetic data generation"
  - "household environment modeling"
  - "bidirectional coupling"
  - "LLM-driven"
  - "embodied AI"
date: 2026-05-08
content_hash: e4b84d604767a953
---

# Realistic Synthetic Household Data Generation at Scale

**Conference**: AAAI 2026
**arXiv**: [2602.07243](https://arxiv.org/abs/2602.07243)
**Code**: None
**Area**: Robotics
**Keywords**: synthetic data generation, household environment modeling, bidirectional coupling, LLM-driven, embodied AI

## TL;DR

This paper proposes an LLM-driven bidirectional coupling generation framework that iteratively generates large-scale synthetic datasets — encompassing household environment configurations, human activities, and human-robot interactions (HRI) — through a cycle in which persona profiles drive environment generation and environment semantics in turn guide activity generation, targeting the training of home robots.

## Background & Motivation

Training home robots faces a fundamental technical challenge: **modeling and understanding the bidirectional relationship between human behavioral patterns and environment configurations**. To operate safely and effectively across diverse household environments, robots must simultaneously understand:

- **Static environment properties**: object affordances, spatial relationships, semantic labels
- **Temporal human–environment interactions**: daily routines, object manipulation sequences, long-term spatial usage patterns

**Three core limitations of existing approaches**:

**Missing spatiotemporal dependencies**: Existing methods cannot capture how human activities influence object placement and room layout.

**Unidirectional rather than bidirectional coupling**: Environment generation and behavior synthesis are treated as independent processes.

**Broken semantic relationships**: There is no coherent semantic connection among persona characteristics, environment affordances, and behavioral patterns.

**Specific issues**:
- Algorithmic approaches such as ProcGen offer limited semantic diversity.
- LLM-based approaches such as Holodeck lack fine-grained behavior modeling.
- Wang et al.'s Dynamic Scene Generation does not account for persona-driven scene generation.
- All existing methods decouple the generation of environments and behaviors.

## Method

### Overall Architecture

The framework comprises four main modules operating within an iterative refinement loop:

1. **Environment Schematic Generator**
2. **Human Activity and HRI Generator**
3. **Bidirectional Influence Controller**
4. **Universal Simulator Adapter**

Core Idea: Persona → drives environment generation → environment semantics guide activity generation → generated activities modify the environment → iterative loop until convergence.

### Key Designs

#### 1. **Human Activity and HRI Generation**

A three-stage hierarchical decomposition strategy is adopted:

**Stage 1**: Activity generation — structured activity sequences are generated based on household member personas, environment constraints, temporal parameters, and robot capabilities, maintaining spatiotemporal consistency.

**Stage 2**: Interaction synthesis — activity sequences from Stage 1 are enriched with contextually appropriate dialogues, accounting for social dynamics and cultural factors.

**Stage 3**: Simulator adaptation — intermediate representations are converted into formats compatible with various simulation environments.

Key techniques:
- **Least-to-most prompt tuning**: progressive prompt engineering
- **Rolling-window context mechanism**: maintains event consistency and ensures logical activity progression
- **Contextual memory management**: at each step, the model is informed of (1) the task being executed, (2) the pipeline step, (3) work completed in the previous step, and (4) current step requirements, reducing hallucination

#### 2. **Environment Schematic Generation**

Several key improvements are introduced over prior work:

- **Asset database flexibility**: not bound to a specific asset library (e.g., Objaverse); only metadata (description, dimensions, pivot points, images) is required
- **Room layout error correction**: post-processing handles nested rooms and disconnected configurations
- **Realistic door connection generation**: the LLM recommends door types based on connected rooms (e.g., removing walls between open-plan kitchens and living rooms)

Persona integration:
- Each household member is assigned a personal bedroom
- Home office spaces are generated based on work patterns
- Assets are selected to match activity behaviors

#### 3. **Iterative Bidirectional Influence Controller**

This is the paper's central innovation. The environment generation module produces an object inventory, spatial layout, and affordance maps that constrain activity generation; the generated activity sequences in turn influence object placement, room usage, and environment modifications.

The convergence criterion is a weighted combination:

$$\text{Score} = w_2 \rho_{\text{env}}(i+1) + w_3 \gamma_{\text{act}}(i+1) + w_4 \sigma_{\text{sem}}(i+1)$$

where:
- $\rho_{\text{env}}$: environment object density = |Objects| / |Rooms|
- $\gamma_{\text{act}}$: activity schedule granularity = Σ duration / |Activities|
- $\sigma_{\text{sem}}$: semantic similarity = cosine similarity between environment and activity descriptions in SBERT embedding space

Iteration terminates when the maximum number of iterations is reached or a user-specified convergence threshold is met.

### Loss & Training

This framework does not involve neural network training. The core "loss" is reflected in the optimization of the convergence criterion — generation quality is improved through iterative refinement rather than gradient descent. Key parameters include LLM temperature, top_p, and top_k for controlling variability.

## Key Experimental Results

### Main Results: Semantic Alignment Analysis

Pairwise cosine similarities are computed using multimodal embeddings (SBERT + CLIP):

| Embedding Pair | SBERT Similarity |
|---|---|
| Persona–Environment | 0.68 ± 0.09 |
| Environment–Activity | 0.72 ± 0.07 |
| Persona–Activity | 0.61 ± 0.12 |
| House Image vs. Family Description (CLIP) | 0.74 ± 0.08 |

The highest correlation for Environment–Activity (0.72) indicates that the bidirectional influence mechanism effectively links environment and behavior.

### Real-World Alignment Validation

| Dataset Comparison | Cosine Similarity |
|---|---|
| HOMER (real-world data) vs. Ours | 0.60 |
| Wang et al. (synthetic data) vs. Ours | 0.27 |

The high alignment with the real-world dataset HOMER (self-reported activities from 21 participants) at 0.60 — exceeding the moderate threshold of 0.5 — validates that the behavioral patterns generated by the framework approximate real human behavior.

### Ablation Study

**Iterative improvement validation**:

| Iteration | Mutual Information MI(P,E)+MI(E,B) | Cosine Similarity |
|---|---|---|
| 1 | 0.45 ± 0.09 | 0.58 ± 0.12 |
| 2 | 0.62 ± 0.08 | 0.65 ± 0.10 |
| 3 | 0.74 ± 0.06 | 0.71 ± 0.08 |
| 4 | 0.81 ± 0.05 | 0.76 ± 0.07 |
| 5 | 0.85 ± 0.04 | 0.79 ± 0.06 |

MI increases from 0.45 to 0.85 across iterations 1–5, validating the effectiveness of the bidirectional refinement mechanism.

**Intervention analysis** (causal validation):

| Intervention Type | p-value | Cohen's d | Effect Size |
|---|---|---|---|
| Age: teenager | p<0.001 | 0.89 | Large |
| Age: retiree | p<0.001 | 1.12 | Large |
| Tidiness: messy | p=0.003 | 0.64 | Medium |
| Tidiness: tidy | p=0.001 | 0.73 | Medium |
| Sleep: early riser | p=0.012 | 0.51 | Medium |
| Sleep: night owl | p=0.008 | 0.58 | Medium |

All interventions reach statistical significance with Cohen's d = 0.51–1.12, confirming that the bidirectional coupling mechanism successfully translates persona characteristic differences into measurable differences in environment configuration and behavioral patterns.

### Key Findings

1. **Bidirectional coupling satisfies the mediation criterion**: MI(persona,env) + MI(env,beh) > MI(persona,beh), demonstrating that the environment serves as an effective mediator between persona and behavior.
2. **Mutual information nearly doubles after 5 iterations** (0.45→0.85), indicating that iterative refinement genuinely improves semantic coherence.
3. **Alignment with real-world data (0.60) substantially exceeds alignment with other synthetic data (0.27)**.
4. **Generation for a three-member/three-room/one-day scenario** requires approximately 150 seconds and 22 LLM calls, demonstrating practical feasibility.

## Highlights & Insights

1. **Bidirectional coupling architecture** is the core innovation — breaking the conventional paradigm of independently generating environments and behaviors.
2. **Contextual memory mechanism** reduces LLM hallucination by providing task context, completed work, and current requirements at each step.
3. **Structured input outperforms free-form text** — delivering information to the LLM in a step-wise structured manner yields better results than end-to-end free-form text.
4. **Pragmatic design for industrial applications** — supports natural language configuration, variation generation, and simulator agnosticism.

## Limitations & Future Work

1. **Absence of visualized 3D environment outputs**: the quality of actually generated 3D scenes is not demonstrated.
2. **LLM hallucination**: the paper acknowledges the generation of "impossible activities" (e.g., playing loud music while sleeping).
3. **Incomplete interaction conflict detection**: incompatible simultaneous activities (e.g., loud music vs. sleeping) are not fully resolved.
4. **Evaluation relies primarily on statistical metrics**: cosine similarity and mutual information do not fully reflect the practical usability of the generated data.
5. **Computational cost scales with scene complexity**: multi-member/multi-room/multi-day scenarios may lead to substantially increased generation time.
6. **Downstream task performance is not validated**: whether the generated data actually improves robot training outcomes remains unverified.

## Related Work & Insights

- **Relationship to Holodeck**: the proposed framework extends Holodeck's LLM-driven environment generation by adding bidirectional activity–environment coupling.
- **Comparison with Dynamic Scene Generation (Wang et al.)**: the latter does not consider persona-driven factors.
- **Broad practical application scenarios**: robotic vacuums understanding daily routines, assistive robots anticipating human needs, and smart home systems adapting to household dynamics.
- **Inspiration for future work**: the bidirectional coupling idea could be applied to other data generation domains (e.g., urban traffic simulation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The bidirectional coupling mechanism is novel, though individual modules (LLM-driven generation) are relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐ — Statistical validation is thorough, but downstream task validation and 3D scene quality assessment are absent.
- **Writing Quality**: ⭐⭐⭐ — Structure is clear but some sections are verbose; algorithmic pseudocode aids comprehension.
- **Value**: ⭐⭐⭐⭐ — The framework holds significant practical importance for scalable synthetic training data generation for home robots.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](../../NeurIPS2025/robotics/dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [\[AAAI 2026\] Sketch-HARP: Hierarchical Autoregressive Sketch Generation for Flexible Stroke-Level Drawing Manipulation](generating_sketches_in_a_hierarchical_auto-regressive_proces.md)
- [\[AAAI 2026\] LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward](laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)

</div>

<!-- RELATED:END -->
