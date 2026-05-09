---
title: >-
  [Paper Note] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion
description: >-
  [ICLR 2026][Image Generation][Hierarchical Reinforcement Learning] This paper proposes HECRL, a hierarchical entity-centric offline goal-conditioned RL framework that combines a value-based GCRL agent with a factored subgoal diffusion model, achieving 150%+ success rate improvements on multi-entity long-horizon tasks.
tags:
  - ICLR 2026
  - Image Generation
  - Hierarchical Reinforcement Learning
  - Goal-Conditioned RL
  - Diffusion Models
  - Entity-Centric
  - Subgoal Generation
date: 2026-05-08
content_hash: 47782615fb313989
---

# Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion

**Conference**: ICLR 2026
**arXiv**: [2602.02722](https://arxiv.org/abs/2602.02722)
**Code**: [GitHub](https://github.com/DanHrmti/HECRL)
**Area**: Diffusion Models
**Keywords**: Hierarchical Reinforcement Learning, Goal-Conditioned RL, Diffusion Models, Entity-Centric, Subgoal Generation

## TL;DR
This paper proposes HECRL, a hierarchical entity-centric offline goal-conditioned RL framework that combines a value-based GCRL agent with a factored subgoal diffusion model, achieving 150%+ success rate improvements on multi-entity long-horizon tasks.

## Background & Motivation
Achieving long-horizon goals in multi-entity environments is a core challenge in RL. Goal-conditioned RL (GCRL) facilitates generalization across goals, but remains limited in high-dimensional observation and combinatorial state spaces, especially under sparse rewards. Existing methods such as HIQL extract hierarchical policies from a single value function, yet still struggle with combinatorial state spaces and image observations on the OGBench benchmark. The root cause lies in the fact that approximation errors in temporal difference (TD) learning accumulate over long horizons—the farther from the goal, the lower the signal-to-noise ratio of the value signal—defining a "policy competence radius" $R_\pi^V$. HECRL's starting point is to exploit entity-factored structure to generate sparsely modified subgoals, decomposing long-horizon tasks into multiple short-horizon subtasks that fall within the policy competence radius.

## Method

### Overall Architecture
A two-level architecture: the lower level is a value-based entity-centric GCRL agent (providing policy and value functions), and the upper level is a conditional diffusion model subgoal generator. The two levels are trained independently and composed at test time via value-function-based subgoal selection, enabling modularity and compatibility with arbitrary value-based GCRL algorithms.

### Key Designs
1. **Subgoal Diffuser**:

    - Function: Generates intermediate subgoals reachable within at most $K$ steps from the current state.
    - Mechanism: A conditional diffusion denoiser models the distribution $p(\tilde{g}|s, g)$—given the current state $s$ and the final goal $g$, it generates reachable subgoals. Training samples are drawn uniformly from offline data, without assuming that the data contains goal-directed behavior.
    - Design Motivation: The distribution $p(\tilde{g}|s,g)$ is highly multimodal; diffusion models can capture multiple subgoal modes present in the data.

2. **Value-based Subgoal Selection (Algorithm 1)**:

    - Function: Selects the optimal subgoal from candidates at test time.
    - Mechanism: $N$ candidate subgoals are sampled; reachability is filtered via a value threshold $\hat{R}$ (retaining those with $V(s,\tilde{g}) > \hat{R}$); the subgoal closest to the final goal (highest $V(\tilde{g}, g)$) is selected. If the goal is closer than the selected subgoal, it is pursued directly.
    - Design Motivation: The subgoal diffuser only fits behavioral data and does not capture the notion of optimality; value function guidance is therefore necessary.

3. **Entity-factored Subgoals**:

    - Function: Encourages the generation of sparse subgoals that modify only a small number of entities.
    - Mechanism: Given state and goal entity sets $s=\{s_m\}_{m=1}^M$ and $g=\{g_m\}_{m=1}^M$, the diffusion model iteratively denoises the subgoal entity set. The Transformer denoiser can selectively copy input tokens to the output via attention, naturally producing entity-level sparsity.
    - Design Motivation: Subgoals that modify only a few state factors are easier to achieve when those factors are independently controllable.

### Loss & Training
- The lower-level GCRL agent is trained with IQL.
- The upper-level diffusion model is trained with the standard diffusion denoising objective, using only 10 denoising steps.
- Both levels use the same offline dataset but are trained independently, requiring no joint optimization.

## Key Experimental Results

### Main Results (Long-Horizon Manipulation Success Rate)

| Environment | EC-SGIQL (Ours) | EC-IQL | EC-Diffuser | HIQL | IQL |
|---|---|---|---|---|---|
| PPP-Cube (State) | **82.5±3.1** | 51.5±4.4 | 44.8±6.7 | 48.3±7.3 | 34.3±4.9 |
| PPP-Cube (Image) | **64.3±4.9** | 25.0±5.7 | 0.3±0.5 | 0.0±0.0 | 0.0±0.0 |
| Scene (Image) | **61.5±5.9** | 53.0±5.5 | 3.3±2.5 | 8.3±1.3 | 17.5±2.7 |
| Push-Tetris (Image) | **61.4±3.3** | 31.6±1.3 | 7.9±0.5 | 5.2±0.8 | 3.4±0.8 |

### Ablation Study (Subgoal Quality — Average Number of Modified Entities)

| Method | PPP-Cube | Stack-Cube | Note |
|---|---|---|---|
| EC-Diffusion (Ours) | **1.36** | **1.04** | Close to modifying only 1 entity |
| EC-AWR | 2.96 | 2.82 | Nearly all 3 entities modified |
| AWR | 2.98 | 2.98 | All entities modified |

### Key Findings
- Achieves 150%+ success rate improvement on the most challenging PPP-Cube (Image) task (25.0 → 64.3).
- Diffusion model subgoals are substantially sparser than those of the deterministic AWR model (1.36 vs. 2.96 modified entities).
- AWR-generated subgoals contain weighted averages across multiple entities, providing ambiguous targets to the lower-level policy.
- Zero-shot compositional generalization: models trained on 3 objects partially generalize to 4–5 objects.

## Highlights & Insights
- The modular design is particularly elegant: the two levels are trained independently and flexibly composed at test time via the value function.
- The inductive bias of entity-centric diffusion naturally yields sparse subgoals without explicit sparsity constraints.
- The insight into the Transformer's mechanism of selectively copying input entities to the output is particularly illuminating.

## Limitations & Future Work
- The value threshold $\hat{R}$ requires manual tuning; adaptive schemes remain to be explored.
- The DLP representation occasionally duplicates the same entity within a subgoal.
- Generalization degrades as the number of objects increases; curriculum learning or online fine-tuning may offer a remedy.

## Related Work & Insights
- **vs. HIQL**: HIQL extracts deterministic subgoals from the value function and fails to produce effective sparse subgoals; this work uses a diffusion model to capture multimodal distributions.
- **vs. EC-Diffuser**: Behavior cloning diffusion directly predicts actions conditioned on goals, but lacks hierarchical subgoal reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ — Novel combination of hierarchy, entity-centricity, and diffusion
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive across environments, ablations, generalization, and visualization
- Writing Quality: ⭐⭐⭐⭐ — Motivation and methodology are clearly articulated
- Value: ⭐⭐⭐⭐ — Significant reference value for multi-entity offline RL

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation](hierloc_hyperbolic_entity_embeddings_for_hierarchical_visual_geolocation.md)
- [\[ICLR 2026\] Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)](improved_object-centric_diffusion_learning_with_registers_and_contrastive_alignm.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
