---
title: >-
  [Paper Note] ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction
description: >-
  [ICLR 2026][Robotics][embodied cognition] ENACT formalizes embodied cognition evaluation as world-modeling VQA based on first-person interaction—revealing significant gaps and anthropomorphic biases in current top-tier VLMs compared to humans through forward/inverse sequence reshuffling tasks.
tags:
  - "ICLR 2026"
  - "Robotics"
  - "embodied cognition"
  - "world model"
  - "VLM evaluation"
  - "egocentric perception"
  - "POMDP"
date: 2026-05-08
content_hash: ebc3d6acc8878645
---

# ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction

**Conference**: ICLR 2026  
**arXiv**: https://enact-embodied-cognition.github.io  
**Code**: https://github.com/enact-embodied-cognition  
**Area**: robotics  
**Keywords**: embodied cognition, world model, VLM evaluation, egocentric perception, POMDP

## TL;DR

ENACT formalizes embodied cognition evaluation as world-modeling VQA based on first-person interaction—revealing significant gaps and anthropomorphic biases in current top-tier VLMs compared to humans through forward/inverse sequence reshuffling tasks.

## Background & Motivation

**Background**: Embodied cognition theory posits that intelligence originates from sensorimotor interaction rather than passive observation. Recently, VLMs (GPT-5, Gemini 2.5, Claude, etc.) have demonstrated impressive interactive capabilities through large-scale non-embodied training, making the question of whether "VLMs possess embodied cognition" a critical scientific inquiry.

**Limitations of Prior Work**: Existing efforts either focus on spatial perception in static scenes, evaluate only language planning capabilities, or examine simple object-to-object interactions. They lack a unified evaluation framework that tightly couples first-person perception with long-horizon embodied interaction. Subjective classification systems (e.g., Yang et al., 2025) struggle to provide reproducible objective measurements.

**Key Challenge**: While VLMs excel at static visual understanding, their multi-step, causal, and partially observable embodied world-modeling capabilities have not been rigorously quantified. The lack of evaluation tools limits the understanding of the boundaries of VLM embodied capabilities.

**Goal**: Construct a scalable, objective, and image-generation-decoupled embodied cognition benchmark to systematically measure forward/inverse world-modeling capabilities under a unified framework.

**Core Idea**: Transform embodied cognition evaluation into POMDP-based sequence reshuffling VQA—Forward World Modeling (reordering shuffled observations given a sequence of actions) and Inverse World Modeling (reordering shuffled actions given a sequence of observations). Actions are represented as scene graph differences, which strips interference from low-level image synthesis while implicitly requiring the model to demonstrate affordance recognition, action-effect reasoning, embodied perception, and long-term memory.

## Method

### Overall Architecture

```mermaid
flowchart TD
    A[Robot Manipulation Trajectory\nBEHAVIOR Simulator] --> B[Keyframe Extraction\nNon-empty Scene Graph Diffs]
    B --> C[Keyframe Trajectory Sampling\nLength L∈3..10\nCombinatorial C(M,L) Expansion]
    C --> D1[Forward World Modeling QA\nGiven Action Seq + Initial Obs\nReorder Shuffled Obs Images]
    C --> D2[Inverse World Modeling QA\nGiven Ordered Obs Seq\nReorder Shuffled Actions]
    D1 --> E[ENACT Benchmark\n8972 QA Pairs\n29 Home Activities]
    D2 --> E
    E --> F[VLM Evaluation\nOnline Validator\nTask Acc + Pairwise Acc]
```

### Key Designs

**1. Sequence Reshuffling VQA Formalized by POMDP: Decoupled from Image Generation**

ENACT defines world modeling on a POMDP $(S, O, A)$, where state space $S$ is a scene graph, observation space $O \subset \mathbb{R}^{H \times W \times 3}$ is the robot's first-person RGB view, and action space $A$ is the scene graph difference $a_t = \delta(s_t, s_{t-1})$. Evaluation is formalized as two permutation inference tasks:

- **Forward**: Given $o_0$, an ordered action sequence $(a_0,\ldots,a_{L-2})$, and a shuffled set of observations $O'$, the model outputs a permutation $\sigma \in \text{Sym}([L-1])$ such that $(o'_{\sigma(1)}, \ldots, o'_{\sigma(L-1)}) = (o_1, \ldots, o_{L-1})$.
- **Inverse**: Given $o_0$ and an ordered observation sequence $(o_1,\ldots,o_{L-1})$, along with a shuffled set of actions $A'$, the model outputs a permutation $\tau$ to align actions with observational progress.

This design decouples long-horizon interactive visual reasoning from high-fidelity video prediction, ensuring evaluation signals are clean and reproducible while implicitly examining affordance recognition, contact reasoning, and spatial memory under partial observability.

**2. Scalable Keyframe Trajectory Synthesis: Combinatorial Data Explosion**

In raw robot trajectories (30Hz), many moments lack semantic changes. ENACT extracts keyframes $K = \{t_1 < \cdots < t_M\}$ by detecting non-empty scene graph differences and filters near-duplicate frames using cosine similarity targeting predicate-level change signatures $c_j$. Trajectories of length $L$ are sampled from $M$ keyframes. Since $L \ll M$ (practically $L \leq 10, M \gtrsim 30$), a single trajectory can generate up to $\binom{M}{L}$ different candidates. This moves data scaling from "number of trajectories" to "number of combinations," theoretically enabling millions of QA pairs from a single trajectory to achieve true scalability.

**3. Multi-Granularity Metrics and Online Validator**

ENACT designs two layers of metrics: Task Accuracy (TA), requiring exact matches—$\text{TA} = \frac{1}{|D|}\sum_{x \in D} \mathbf{1}[\text{accepted}(x)]$; and Pairwise Accuracy (PA), providing partial credit for local correctness—$\text{PA} = \frac{\sum_x \#\text{Correct Adjacent Pairs}_x}{\sum_x L_x}$. As multiple valid permutations may satisfy constraints, an online validator accepts any permutation consistent with input constraints, preventing multi-solution problems from being misjudged as errors and accurately reflecting causal reasoning.

**4. Fine-grained Error Analysis Framework: Five Category Classification**

By converting predicted permutations into corresponding action sequences and performing Venn diagram analysis against ground truth scene graph differences, ENACT classifies atomic state changes into five categories: Correct, Omission, Hallucination, Entity Substitution, Polarity Inversion, and Predicate Substitution. This semantic-level classification is more diagnostic than permutation-level comparisons, directly revealing the root causes of cognitive failure.

## Key Experimental Results

### Main Results (Pairwise Accuracy, Select Horizons)

| Model | Forward L=3 | Forward L=6 | Forward L=10 | Inverse L=3 | Inverse L=6 | Inverse L=10 |
|------|----------|----------|-----------|----------|----------|-----------|
| Human | 93.62 | 93.87 | 95.13 | 92.05 | 94.25 | 96.29 |
| GPT-5 | 84.62 | 64.18 | 46.93 | 86.28 | 68.78 | 55.33 |
| GPT-5 mini | 87.50 | 63.41 | 44.11 | 85.05 | 67.67 | 50.02 |
| Gemini 2.5 Pro | 86.10 | 60.80 | 36.98 | 87.94 | 70.03 | 56.62 |
| InternVL3.5-241B | 75.79 | 45.85 | 25.24 | 82.26 | 53.38 | 30.56 |
| Qwen2.5-VL-72B | 78.15 | 41.92 | 25.07 | 77.80 | 48.19 | 36.27 |
| Claude Sonnet 4 | 65.65 | 30.52 | 20.16 | 73.25 | 43.07 | 28.49 |

### Ablation Study (Image Fidelity vs Camera Config, GPT-5 mini, Pairwise Acc Change Δ)

| Configuration Variant | Significance (p) | Impact Δ | Description |
|----------|-----------|--------|------|
| Path Tracing | p≥0.2 | Small | Rendering fidelity does not affect performance |
| Real Images (GPT-image-1 conversion) | p≥0.2 | Small | Minimal sim-to-real gap |
| FOV 60/80/Fisheye | p≤0.01 | Significant Drop | VLM biased toward human eye intrinsics |
| Camera Height +0.5m (Forward) | p<0.05 | Δ=−0.13 | Non-standard height significantly impairs performance |
| Right vs Left Hand (Confusion Rate) | — | Right 4.67% vs Left 9.38% | Right hand significantly outperforms left |

### Key Findings

- Inverse tasks consistently outperform forward tasks across all models and horizons, suggesting linguistic retrospective reasoning is stronger than visual proactive simulation.
- Accuracy scales monotonically downward as trajectory length increases. At $L=8-10$, most models' Task Accuracy approaches zero while humans remain stable at $>93\%$.
- GPT-5 and Gemini 2.5 Pro only approach human-level performance at $L=3$; the gap widens rapidly in long horizons.
- Primary error types: Forward tasks involve hallucinations (43.9%) + omissions (37.1%) ≈ 81%; Inverse tasks share these roughly equally at 41.8%.
- Cosmos-Reason1-7B (trained on embodied data) is more stable than same-sized models at $L>5$.
- Sim-to-real results are highly consistent, validating minimal simulation gaps.

## Highlights & Insights

- Elegant yet comprehensive task design: The "narrow" form of sequence reshuffling VQA implicitly demands core embodied capabilities like affordance recognition, action-effect reasoning, and partially observable long-term memory while avoiding video synthesis noise.
- Truly scalable data pipeline: Combinatorial keyframe sampling allows a single trajectory to generate millions of QA pairs, providing a foundation for large-scale embodied cognition research.
- Stark human contrast: Humans maintain ~94% accuracy across all horizons, while the strongest VLM (GPT-5) drops to 47% at $L=10$—revealing a gap much larger than previously assumed.
- Quantified anthropomorphic bias: Reveals a deep-seated bias in training data. VLMs' default world-view is tightly coupled with human perspectives, making it difficult to generalize to non-human robotic viewpoints.
- Semantic error framework: Directs future improvements by identifying that the main issue isn't misidentifying specific changes, but rather omitting or hallucinating state changes that do not exist.

## Limitations & Future Work

- Relying solely on simulation (BEHAVIOR); despite small sim-to-real gaps, the diversity of real-world trajectories remains limited.
- Action representation via scene graph differences depends on ground truth states from the simulator, which is difficult to replicate on physical robot platforms.
- Evaluation scale (8972 QA) is smaller than common LLM benchmarks and covers only 29 home activities; scene diversity needs expansion.
- Does not yet address cross-modal execution (language → action) or actual robot control, measuring only "understanding."

## Related Work & Insights

- **vs EmbodiedScan / ScanQA (Static Scene VQA)**: ENACT introduces temporal action chains and partial observability, upgrading from static spatial understanding to dynamic causal reasoning.
- **vs Aurora-Bench**: Aurora-Bench focuses on short-horizon general video world modeling; ENACT focuses on long-horizon robot manipulation with explicit action semantics.
- **vs BEHAVIOR Challenge**: ENACT reuses BEHAVIOR trajectory data but transforms it into an evaluation-oriented rather than training-oriented benchmark.
- **vs Cosmos-Reason1 (Embodied VLMs)**: Results highlight the value of embodied data training for long-horizon stability, providing quantitative evidence for future training data design.

## Rating

- Novelty: ⭐⭐⭐⭐ Fusion of world modeling, POMDP, and sequence reshuffling into a unified framework with clear concepts and formalization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 30 models, 8972 QAs, multi-dimensional bias analysis (perspective/FOV/handedness), and sim-to-real comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly structured, balancing mathematical and intuitive explanations with excellent readability.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in long-horizon embodied cognition evaluation with scalable data, providing essential infrastructure for VLM research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] World-In-World: World Models in a Closed-Loop World](world-in-world_world_models_in_a_closed-loop_world.md)
- [\[ICLR 2026\] Ctrl-World: A Controllable Generative World Model for Robot Manipulation](ctrl-world_a_controllable_generative_world_model_for_robot_manipulation.md)
- [\[ICLR 2026\] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)
- [\[CVPR 2025\] SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters](../../CVPR2025/robotics/solami_social_vision-language-action_modeling_for_immersive_interaction_with_3d_.md)
- [\[CVPR 2026\] TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos](../../CVPR2026/robotics/tracegen_world_modeling_in_3d_trace_space_enables_learning_from_cross-embodiment.md)

</div>

<!-- RELATED:END -->
