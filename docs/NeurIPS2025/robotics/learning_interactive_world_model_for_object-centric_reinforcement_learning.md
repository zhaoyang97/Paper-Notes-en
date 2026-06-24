---
title: >-
  [Paper Note] Learning Interactive World Model for Object-Centric Reinforcement Learning
description: >-
  [NeurIPS 2025][Robotics][world model] This paper proposes FIOC-WM, which learns the interaction structure among objects in a world model via a two-level factorization at the object and attribute levels. It trains a hierarchical policy grounded in interaction primitives, achieving more efficient policy learning and compositional generalization across multiple robot control tasks.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "world model"
  - "Object-Centric RL"
  - "Interaction Learning"
  - "Hierarchical Policy"
  - "Compositional Generalization"
date: 2026-05-08
content_hash: a774f79322fd405a
---

# Learning Interactive World Model for Object-Centric Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2511.02225](https://arxiv.org/abs/2511.02225)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: world model, Object-Centric RL, Interaction Learning, Hierarchical Policy, Compositional Generalization

## TL;DR

This paper proposes FIOC-WM, which learns the interaction structure among objects in a world model via a two-level factorization at the object and attribute levels. It trains a hierarchical policy grounded in interaction primitives, achieving more efficient policy learning and compositional generalization across multiple robot control tasks.

## Background & Motivation

**Background**: Existing world models for reinforcement learning learn state abstractions and dynamics from high-dimensional observations. However, most encode entire scenes into a monolithic latent vector, lacking structured modeling of objects and their interactions. Although Object-Centric RL decomposes states into individual object representations, interaction relations remain implicit—the model has no explicit knowledge of which objects interact and when.

**Limitations of Prior Work**: In realistic environments, physical interactions such as collision, stacking, and friction are the primary drivers of dynamics evolution. Without explicit interaction modeling, prediction accuracy is limited, and long-horizon tasks cannot be decomposed into structured interaction primitives for efficient planning and control. Moreover, each object's state can be further factored into static attributes (color, shape) and dynamic variables (position, velocity), a decomposition that reduces redundancy and focuses on the minimal sufficient information needed for control.

**Root Cause / Paper Goals**: The central question addressed is: **what type and degree of factorization structure makes latent representations most effective for efficient and generalizable policy learning?** The answer requires simultaneously modeling object-level interactions and attribute-level decomposition.

## Method

### Overall Architecture

FIOC-WM consists of two stages:
1. **Offline model learning**: Learns object-centric latent representations from pre-trained visual features, disentangles static/dynamic attributes, constructs interaction graphs, and trains low-level interaction policies.
2. **Online hierarchical policy learning**: A high-level policy selects sequences of interaction graphs; a low-level policy executes the corresponding interactions.

### Key Designs

1. **Factored Interactive Object-Centric POMDP (FIOC-POMDP)**:

    - Each object $i$ is factored into dynamic variables $\mathbf{d}_t^i$ (position, velocity) and static attributes $\mathbf{c}^i$ (color, mass).
    - A time-varying interaction graph $G_t$ models pairwise interactions; each edge indicates that two objects are interacting at that timestep.
    - State transitions are decomposed into self-transition $f_{\text{self}}$ (individual object evolution) and interaction transition $f_{\text{inter}}$ (inter-object influence), where interactions only affect dynamic variables.
    - Transition formula: $\mathbf{d}_{t+1}^i = f_{\text{self}}(\mathbf{d}_t^i, \mathbf{c}^i, \mathbf{a}_t) + \sum_{j \in \mathcal{N}_t(i)} f_{\text{inter}}(\mathbf{d}_t^i, \mathbf{d}_t^j, \mathbf{c}^j)$

2. **Two-level factorized latent representation learning**:

    - Pre-trained visual encoders (DINO-v2/R3M) extract features; Slot Attention clusters them into object-level representations.
    - A VAE maps slot representations to latent states via two separate encoders for static features $f_c$ and dynamic features $f_d$.
    - Static features are regularized by a temporal consistency loss $\mathcal{L}_{\text{static}}$ enforcing invariance across timesteps, and a contrastive loss $\mathcal{L}_{\text{con}}$ ensuring discriminability across objects.
    - Dynamic features are modeled with a GRU for temporal evolution, with dynamics conditioned on the interaction graph $G_t$ through a structured prior distribution.

3. **Interaction graph learning**:

    - A surrogate latent variable $\mathbf{u}_t$ parameterizes the distribution over interaction graphs.
    - For each object pair $(i,j)$, a GRU encodes their latent states to produce a pairwise embedding $\mathbf{u}_t^{ij}$.
    - Interaction structure is learned either via variational masking (differentiable categorical distribution sampling) or conditional independence testing.

### Loss & Training

The total offline-phase loss comprises:
- **Reconstruction loss** $\mathcal{L}_{\text{recon}}$: reconstructs current observations.
- **Prediction loss** $\mathcal{L}_{\text{pred}}$: predicts next-step observations.
- **KL divergence** $\mathcal{L}_{\text{KL}}$: aligns posterior with prior.
- **Reward loss** $\mathcal{L}_{\text{rew}}$: predicts rewards.
- **Static loss** $\mathcal{L}_{\text{static}}$ + **Contrastive loss** $\mathcal{L}_{\text{con}}$: attribute disentanglement.

For the hierarchical policy, the high-level policy selects a target interaction graph $G_t^g$, and the low-level policy $\pi^l(\mathbf{a}_t | \mathbf{s}_t, G_t^g)$ executes interactions via MPC or PPO. The high-level policy is jointly optimized with task reward and a diversity bonus $r_{\text{div}} = 1/\sqrt{|G_{\text{visited}}|}$.

## Key Experimental Results

### Main Results

| Environment | Generalization Type | FIOC | DreamerV3 | EIT | TD-MPC2 |
|-------------|---------------------|------|-----------|-----|---------|
| i-Gibson | Attribute | **0.79** | 0.62 | 0.70 | 0.65 |
| Libero | Attribute | **0.76** | 0.59 | 0.73 | 0.69 |
| Push&Switch | Compositional | **0.86** | 0.81 | 0.83 | 0.79 |
| Libero | Compositional | **0.70** | 0.58 | 0.65 | 0.63 |
| Franka Kitchen | Skill | **0.73** | 0.59 | 0.65 | 0.62 |

### Ablation Study

| Configuration | Single-task SR | Compositional Gen. | Notes |
|---------------|---------------|-------------------|-------|
| FIOC (full) | 0.81 | 0.70 | Baseline |
| w/o attribute decomposition | 0.77 (↓0.04) | 0.64 (↓0.06) | No static/dynamic split |
| w/o interaction modeling | 0.63 (↓0.18) | 0.52 (↓0.18) | Fully connected graph |
| w/o hierarchical policy | 0.58 (↓0.23) | 0.42 (↓0.28) | **Largest impact** |
| w/o pre-trained $\pi^l$ | 0.69 (↓0.12) | 0.59 (↓0.11) | Low-level learned online from scratch |
| w/o diversity reward | 0.62 (↓0.19) | 0.50 (↓0.20) | No exploration incentive |

### Key Findings

- Interaction modeling and the hierarchical policy are the two most critical components; removing either causes the largest performance degradation.
- FIOC consistently outperforms all baselines on generalization tasks with the smallest generalization gap.
- Variational masking (categorical distribution) performs best when the number of objects is large.
- Attribute decomposition improves static feature representation quality (lowest linear-probe MSE).

## Highlights & Insights

- **The two-level factorization is elegant**: object-level decomposition addresses "who interacts with whom," while attribute-level decomposition addresses "what changes"—the two are complementary.
- **The notion of interactions as skills** is inspiring: decomposing long-horizon tasks into sequences of interaction primitives naturally yields compositional task decomposition.
- Using pre-trained visual features as observation proxies leverages rich semantics while preserving structured modeling.

## Limitations & Future Work

- Relies on a pre-trained object discovery model (Slot Attention), which cannot automatically handle objects of unseen categories.
- The interaction model generalizes primarily to seen object categories; generalization to entirely novel categories remains unverified.
- Validation is limited to simulated environments; real-robot scenarios are not addressed.
- The action space of the high-level policy (interaction graph selection) may face scalability challenges as the number of objects grows.

## Related Work & Insights

- FIOC-WM inherits the strengths of both Factored RL and Object-Centric RL, unifying representation learning, interaction modeling, and policy learning within a single world model framework.
- It shares a "perception → reflection → planning" paradigm with Generative Agents, but realizes it in continuous control settings through a structured latent space.
- For long-horizon robotic manipulation tasks involving multiple objects (e.g., kitchen, warehouse picking), the compositional combination of interaction primitives warrants further exploration.

## Rating

- Novelty: ⭐⭐⭐⭐ — The joint framework of two-level factorization and interaction-graph world model is a relatively novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple environments, comprehensive ablations, and generalization evaluations.
- Writing Quality: ⭐⭐⭐⭐ — Clear formalization and intuitive architectural diagrams.
- Value: ⭐⭐⭐⭐ — Represents a substantive advance for Object-Centric RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](../../AAAI2026/robotics/object-centric_latent_action_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)
- [\[NeurIPS 2025\] Learning Parameterized Skills from Demonstrations](learning_parameterized_skills_from_demonstrations.md)

</div>

<!-- RELATED:END -->
