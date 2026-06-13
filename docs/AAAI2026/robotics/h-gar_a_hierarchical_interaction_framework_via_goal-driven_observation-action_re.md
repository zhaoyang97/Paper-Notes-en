---
title: >-
  [Paper Note] H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation
description: >-
  [AAAI2026][Robotics][robotic manipulation] This paper proposes H-GAR, a hierarchical goal-driven framework that first predicts a goal observation and then synthesizes intermediate observations…
tags:
  - "AAAI2026"
  - "Robotics"
  - "robotic manipulation"
  - "goal-conditioned planning"
  - "observation-action interaction"
  - "diffusion policy"
  - "coarse-to-fine refinement"
date: 2026-05-08
content_hash: 11860b061d3e15cf
---

# H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation

**Conference**: AAAI2026 Oral  
**arXiv**: [2511.17079](https://arxiv.org/abs/2511.17079)  
**Code**: To be confirmed  
**Area**: Robotics
**Keywords**: robotic manipulation, goal-conditioned planning, observation-action interaction, diffusion policy, coarse-to-fine refinement

## TL;DR

This paper proposes H-GAR, a hierarchical goal-driven framework that first predicts a goal observation and then synthesizes intermediate observations, while refining coarse-grained actions via a historical action memory bank. This design enables explicit bidirectional interaction between observations and actions, achieving state-of-the-art performance on both simulated and real-robot manipulation tasks.

## Background & Motivation

Unified video-and-action prediction models have demonstrated significant potential in robotic manipulation: future observations provide contextual cues for planning, while action sequences reveal how interactions affect the environment. However, existing approaches (e.g., UVA, PAD, UniPi) suffer from two fundamental limitations:

1. **Goal-agnostic observation generation**: Models predicting future observation sequences lack explicit goal-semantic guidance, producing sequences that are visually plausible but semantically inconsistent with task objectives, thereby degrading downstream planning accuracy.
2. **Implicit observation-action modeling**: Observations and actions are typically generated in parallel or only loosely coupled, without explicitly modeling their causal relationship, which weakens temporal consistency and adaptability.

Both issues are particularly pronounced in long-horizon, multi-stage manipulation tasks (e.g., drawer opening/closing combined with object placement), leading to failures at critical steps.

## Core Problem

How to introduce **explicit goal anchoring** and **structured bidirectional observation-action interaction** into a unified video-action prediction framework, so that generated actions are both semantically aligned with task goals and temporally coherent?

## Method

### Overall Architecture

H-GAR adopts a coarse-to-fine hierarchical design consisting of four stages:

1. **Encoding stage**: Historical observations $\{O_{t-h+1},\dots,O_t\}$ are encoded into latent tokens via a pretrained VAE, then combined with CLIP-encoded text instructions $T_I$ and masked future observations, and fed into a Transformer encoder to obtain the joint representation $\mathbf{Z}_{t+1:t+h'}$.
2. **Goal prediction stage**: A video diffusion decoder generates the goal observation $O_{t+h'}$ (i.e., the final visual state after task completion) from the terminal latent $\mathbf{Z}_{t+h'}$, along with a coarse-grained action sequence.
3. **Goal-Conditioned Observation Synthesizer (GOS)**: Conditioned on the goal observation latent and the coarse-grained action latent, this module synthesizes intermediate observation features.
4. **Interaction-Aware Action Refiner (IAAR)**: Using feedback from intermediate observations and a historical action memory bank, this module refines coarse-grained actions into precise, temporally consistent actions.

### Key Designs

#### GOS Module

GOS introduces learnable queries $\mathbf{Q}_{\text{Inter}} \in \mathbb{R}^{(h'-1)\times N\times D}$ to represent the latents of intermediate frames:

- **Self-attention to aggregate goal information**: The queries are concatenated with the goal observation latent $\mathbf{Z}_{t+h'}$ and processed via Self-Attention, injecting goal semantics into the queries.
- **Cross-attention to inject action context**: The updated queries use the coarse action latent $\mathbf{Z}_{t+1:t+h'}$ as Key/Value in Cross-Attention, followed by an FFN to produce intermediate observation features $\mathbf{Z}_{\text{Inter}}$.

This design enables intermediate observations to jointly reflect "where to go" (goal) and "how to get there" (actions).

#### IAAR Module

IAAR refines coarse actions in two steps:

1. **Historical action interaction**: Coarse action latents serve as Query, while the Historical Action Memory Bank $\mathcal{H}_t$ serves as Key/Value in an attention operation, injecting temporal behavioral priors.
2. **Observation-feedback refinement**: The output of the previous step serves as Query, with intermediate observation features $\mathbf{Z}_{\text{Inter}}$ as Key/Value in Cross-Attention, yielding the final refined actions.

#### Historical Action Memory Bank

The memory bank stores historical refined action latents. When the capacity threshold is exceeded, a **similarity-based eviction strategy** is applied: cosine similarities between adjacent action latents are computed, and the most similar pair is merged (by averaging), preserving memory diversity. This outperforms both FIFO and random eviction strategies.

### Loss & Training

The total loss consists of four components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{goal}} + \mathcal{L}_{\text{coarse}} + \mathcal{L}_{\text{inter}} + \mathcal{L}_{\text{fine}}$$

Each term is a diffusion denoising loss supervising the goal observation, coarse actions, intermediate observations, and refined actions, respectively. During training, position-consistent random masking is applied to future observations to prevent information leakage; at inference, generation begins from blank images.

## Key Experimental Results

### Main Results

#### Simulation Experiments (Success Rate)

| Method | PushT | PushT-M | Libero-10 |
|--------|-------|---------|-----------|
| Diffusion Policy-C | 0.91 | 0.68 | 0.53 |
| UVA | 0.96 | 0.85 | 0.89 |
| PD-VLA | 0.82 | 0.71 | 0.92 |
| **H-GAR** | **0.99** | **0.90** | **0.94** |

H-GAR ranks first on all three benchmarks, achieving 0.99 on PushT.

#### Real-Robot Experiments (ALOHA Platform)

| Task | H-GAR | UVA | PD-VLA |
|------|-------|-----|--------|
| Object Placement (2-stage) | 9/10 → 8/10 | 7/10 → 6/10 | 8/10 → 7/10 |
| Drawer Manipulation (3-stage) | 7/10 → 6/10 → 6/10 | 6/10 → 5/10 → 3/10 | 6/10 → 6/10 → 4/10 |
| Towel Folding | 8/10 | 5/10 | 6/10 |
| Mouse Arrangement | 6/10 | 3/10 | 4/10 |

H-GAR shows a marked advantage on long-horizon multi-stage tasks; e.g., in the final stage of Drawer Manipulation, H-GAR achieves 6/10 vs. UVA's 3/10.

#### Observation Generation Quality (FVD ↓)

Under 8-step generation, H-GAR achieves FVD 49.01 on Libero-10 (vs. UVA 51.10) and FVD 28.43 on Mouse Arrangement (vs. UVA 32.78). FVD exhibits a strong negative correlation with task success rate.

### Ablation Study

- The full model with both GOS and IAAR (including the memory bank) performs best; removing either module leads to significant performance degradation.
- Using goal-frame conditioning outperforms strategies based on random or uniformly sampled frames.
- A memory bank size of 32 achieves the best trade-off; similarity-based eviction outperforms FIFO and random eviction.

## Highlights & Insights

1. The **hierarchical coarse-to-fine paradigm** is elegantly designed: anchor the goal → generate coarse actions → synthesize intermediate observations → refine actions, forming a clear logical chain.
2. The **bidirectional interaction of GOS and IAAR** explicitly couples observations and actions, overcoming the loose-coupling bottleneck prevalent in existing methods.
3. The **Historical Action Memory Bank** with similarity-based eviction concisely and effectively encodes temporal behavioral priors.
4. **Real-robot validation is thorough**, covering four task categories spanning short-horizon, long-horizon, and fine-grained manipulation, with pronounced advantages on long-horizon tasks.
5. The negative-correlation analysis between FVD and success rate provides direct evidence linking observation generation quality to manipulation performance.

## Limitations & Future Work

1. **Misclassification of research area**: The core contribution belongs to robotic manipulation/policy learning rather than object detection; the current categorization may need to be reassigned to robotics.
2. **Computational overhead**: The multi-stage hierarchical design increases inference complexity; the paper does not discuss inference latency, which may pose a bottleneck for real-time control scenarios.
3. **Error propagation from goal observation prediction**: In the cascaded coarse-to-fine design, large deviations in goal observation prediction may be amplified by subsequent GOS and IAAR modules.
4. **Fixed memory capacity threshold**: The simple threshold-plus-merging strategy may be inadequate for extremely long-horizon tasks; adaptive memory management warrants further exploration.
5. **Limited generalization validation**: Experiments are conducted solely on the ALOHA platform, without evaluation on more complex settings such as dexterous hands or mobile manipulation.

## Related Work & Insights

| Method | Goal Anchoring | Obs.-Action Interaction | Coarse-to-Fine | Memory Mechanism |
|--------|---------------|------------------------|----------------|-----------------|
| Diffusion Policy | ✗ | ✗ | ✗ | ✗ |
| UniPi | ✗ | Implicit | ✗ | ✗ |
| UVA | ✗ | Joint optimization | ✗ | ✗ |
| PAD | ✗ | Joint denoising | ✗ | ✗ |
| LBP | Implicit latent goal | ✗ | ✗ | ✗ |
| **H-GAR** | **Explicit goal observation** | **GOS+IAAR bidirectional explicit** | **✓** | **Historical Memory Bank** |

H-GAR is the first method to unify explicit goal observation anchoring, bidirectional observation-action interaction, and historical action memory within a single hierarchical framework.

**Insights and connections:**

1. The **coarse-to-fine cascading approach** is transferable to other sequential decision-making problems (e.g., autonomous driving trajectory planning): predict the terminal state as an anchor, then progressively refine.
2. The **similarity-based eviction memory management strategy** is simple yet effective and applicable to any online learning scenario requiring a finite-size historical buffer.
3. The methodology of analyzing the correlation between observation generation quality and downstream task performance can be used to evaluate the role of video generation in other embodied tasks (e.g., navigation).
4. In connection with related research directions: replacing GOS with a 3D scene representation generator could potentially yield stronger spatial reasoning capabilities for 3D manipulation planning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The hierarchical goal-driven design with bidirectional interaction is novel, though individual components (e.g., diffusion decoder, cross-attention) are standard building blocks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Simulation, real-robot experiments, ablation studies, visualizations, and correlation analysis are all highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, and well-articulated motivation.
- Value: ⭐⭐⭐⭐ — Provides an effective hierarchical solution for joint video-action modeling in robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Human-assisted Robotic Policy Refinement via Action Preference Optimization](../../NeurIPS2025/robotics/human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)
- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](../../ICLR2026/robotics/memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[AAAI 2026\] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)

</div>

<!-- RELATED:END -->
