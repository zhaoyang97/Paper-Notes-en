---
title: >-
  [Paper Note] SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models
description: >-
  [ICML2025][Robotics][World Models] The SENSEI framework is proposed: it leverages a VLM to perform pairwise comparisons of environmental images based on how "interesting" they are, distill a semantic intrinsic reward, and combine it with novelty rewards driven by ensemble uncertainty to achieve task-free semantic exploration via a world model, significantly accelerating downstream task learning.
tags:
  - "ICML2025"
  - "Robotics"
  - "World Models"
  - "Intrinsic Motivation"
  - "VLM-Guided Exploration"
  - "Semantic Reward Distillation"
  - "Model-based RL"
  - "DreamerV3"
date: 2026-05-08
content_hash: 9dec95db8ea773b5
---

# SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models

**Conference**: ICML2025  
**arXiv**: [2503.01584](https://arxiv.org/abs/2503.01584)  
**Code**: [Project Page](https://sites.google.com/view/sensei-paper)  
**Area**: World Models / Intrinsic Exploration  
**Keywords**: World Models, Intrinsic Motivation, VLM-Guided Exploration, Semantic Reward Distillation, Model-based RL, DreamerV3

## TL;DR

The SENSEI framework is proposed: it leverages a VLM to perform pairwise comparisons of environmental images based on how "interesting" they are, distill a semantic intrinsic reward, and combine it with novelty rewards driven by ensemble uncertainty to achieve task-free semantic exploration via a world model, significantly accelerating downstream task learning.

## Background & Motivation

Intrinsic motivation is a core research direction in reinforcement learning (RL) for achieving autonomous exploration. Although traditional intrinsic reward designs (e.g., information gain, prediction error, state coverage) are domain-agnostic, they often only lead to low-level interactions and fail to efficiently unlock semantically meaningful behaviors. For example, when a robotic arm faces objects on a desk, traditional methods might only result in random arm stretching rather than actively opening drawers or grasping objects.

Recently, LLMs/VLMs have provided new possibilities for injecting "human priors" into exploration. However, existing methods have significant limitations:

- **Motif** depends on language-embedded environments and requires event captions.
- **ELLM** requires state representations in textual formats.
- **OMNI** requires high-level action spaces.
- None of these methods **learn an internal "interestingness model"**, continually relying on external LLM guidance instead.

SENSEI's motivation: Can a Model-based RL agent based on **visual observations + low-level actions** learn an internalized model of "what is interesting" through VLM feedback, thereby enabling autonomous exploration of semantically meaningful behaviors in a task-free phase?

## Method

SENSEI consists of three core components: reward distillation, world model learning, and exploration policy optimization.

### 1. Semantic Reward Distillation (VLM-Motif)

**Pre-training phase**: An initial dataset $\mathcal{D}^{\text{init}}$ is collected using Plan2Explore, and a VLM (GPT-4) is then used to perform pairwise comparisons of images to judge which one is more "interesting".

The annotation function is defined as:

$$\text{VLM}: \mathcal{O} \times \mathcal{O} \to \mathcal{Y}, \quad \mathcal{Y} = \{1, 2, \emptyset\}$$

Based on the preference pairs, a reward model $R_\psi: \mathcal{O} \to \mathbb{R}$ is trained using cross-entropy loss, distilling the semantic preferences of the VLM into a differentiable reward function. During exploration, the agent receives a semantic reward:

$$r_t^{\text{sem}} \leftarrow R_\psi(\boldsymbol{o}_t)$$

### 2. World Model (RSSM + Semantic Reward Prediction Head)

Based on the RSSM architecture of DreamerV3, which contains stochastic states $\boldsymbol{z}_t$ and deterministic memory $\boldsymbol{h}_t$:

- **Posterior**: $\boldsymbol{z}_t \sim q_\phi(\boldsymbol{z}_t \mid \boldsymbol{h}_t, \boldsymbol{o}_t)$
- **Dynamics**: $\boldsymbol{h}_{t+1} = f_\phi(\boldsymbol{a}_t, \boldsymbol{h}_t, \boldsymbol{z}_t)$
- **Prior**: $\hat{\boldsymbol{z}}_{t+1} \sim p_\phi(\hat{\boldsymbol{z}}_{t+1} \mid \boldsymbol{h}_{t+1})$

**Key Extension**: A new semantic reward prediction head $\hat{r}_t^{\text{sem}}$ is added, allowing the world model to predict semantic interestingness within its imagination, without actual VLM queries.

In addition, an ensemble of predictors ($N$ models) is trained to predict the next latent state, and **ensemble disagreement** is used to measure epistemic uncertainty:

$$r_t^{\text{dis}} = \frac{1}{J} \sum_{j=1}^{J} \text{Var}(\hat{z}_{j,t}^n)$$

### 3. Go-and-Explore Exploration Policy

Core idea: In uninteresting states, the agent mainly pursues "interestingness" (go), and switches to pursuing "novelty" (explore) upon reaching an interesting state. This switch is achieved via an adaptive threshold:

$$r_t^{\text{expl}} = \hat{r}_t^{\text{sem}} + \begin{cases} \beta^{\text{explore}} \cdot r_t^{\text{dis}}, & \text{if } \hat{r}_t^{\text{sem}} \geq Q_k(\hat{r}^{\text{sem}}) \\ \beta^{\text{go}} \cdot r_t^{\text{dis}}, & \text{otherwise} \end{cases}$$

where $Q_k$ is the $k$-th percentile of $\hat{r}^{\text{sem}}$ (estimated via moving average), and $\beta^{\text{explore}} > \beta^{\text{go}}$. This design avoids local minima caused by purely relying on semantic rewards.

### Loss & Training

- Reward Distillation: Cross-entropy loss based on preference pairs.
- World Model: Joint optimization of ELBO (Evidence Lower Bound), including reconstruction loss (observation, reward, continuation, semantic reward).
- Policy: Actor-Critic, optimized based on $r_t^{\text{expl}}$ inside the world model's imagination.

## Key Experimental Results

### Experimental Setup

| Environment | Type | Observation | Action | Characteristics |
|------|------|------|------|------|
| MiniHack (KeyRoom/KeyChest) | Dungeon game | Pixel (egocentric view) | Discrete | Requires keys to open doors/chests |
| Robodesk | Robotic manipulation | Pixel | Continuous | Multi-object interaction |
| Pokémon Red | RPG game | Game screen | Discrete (Game Boy buttons) | Open world + combat system |

### Task-free Exploration Results

**MiniHack**: SENSEI achieves significantly better interaction counts for picking up keys and opening doors/chests in KeyRoom-S15 and KeyChest compared to Plan2Explore and pure VLM-Motif ($\beta=0$).

**Robodesk**: Within 1M exploration steps, the number of interactions of SENSEI with most objects (drawers, sliding doors, blocks, balls, etc.) beats Plan2Explore and RND.

**Ablation Study**: Pure VLM-Motif ($\beta=0$) easily falls into local optima—for example, failing to continue to explore and open doors after picking up a key in KeyRoom.

### Downstream Task Learning

| Method | KeyRoom-S15 | KeyChest |
|------|-------------|----------|
| SENSEI exploration $\to$ DreamerV3 | **Fastest convergence** | **Fastest convergence** |
| P2X exploration $\to$ DreamerV3 | Unstable | Slower |
| DreamerV3 from scratch | Slow | Slower |
| PPO from scratch | >20M steps to stabilize | Early success but unstable |

SENSEI is about **two orders of magnitude** faster than PPO on KeyRoom-S15.

### Pokémon Red Exploration

- SENSEI is the only method that reaches the first Gym (navigating 9 map segments).
- After 750k steps, SENSEI's Pokémon levels are consistently higher than those of Plan2Explore (average level-up count is $2\times$ that of P2X starting from episode 390).
- SENSEI successfully beats the first Gym to obtain the Boulder Badge after the second annotation iteration.

## Highlights & Insights

1. **Internalized Interestingness Model**: Unlike existing methods that continuously rely on external LLMs, SENSEI internalizes semantic interestingness through the reward prediction head of the world model. It can directly predict interestingness during imagination, substantially improving sample efficiency.
2. **Go-and-Explore Switching Mechanism**: Elegantly decouples the two stages of "moving towards interesting states" and "exploring novel behaviors starting from interesting states," avoiding the local optima associated with pure semantic rewards.
3. **Zero-prior SENSEI General**: Environmental descriptions can be automatically generated by the VLM instead of being provided by humans, with virtually zero performance loss, enhancing generalizability.
4. **Cross-domain Verification**: Consistent effectiveness is demonstrated across three heavily distinct environments: discrete-action dungeon games, continuous-action robotic manipulation, and complex RPGs.
5. **Iterative Annotation**: The Pokémon experiment demonstrates the potential of iteratively refining semantic rewards (resolving out-of-distribution issues through multiple rounds of annotation).

## Limitations & Future Work

- **Reliance on Full Observation**: The performance of VLM annotations drops under occlusions (Robodesk requires multi-view to alleviate this).
- **Quality of Initial Dataset**: Semantic reward distillation depends on the behavior richness of $\mathcal{D}^{\text{init}}$. If initial exploration is too poor, the VLM annotations will lack signal.
- **VLM Annotation Noise**: Current VLM judgments are not always accurate; ablation studies show that stronger VLMs yield greater performance gains.
- **Scalability**: Annotation costs (GPT-4 API calls) could become a bottleneck in large-scale applications.
- **Single-timestep Annotation**: Annotations are based only on single-frame image pairwise comparisons rather than video sequences, discarding temporal information.
- Validation has not been done on real physical robots or photorealistic environments.

## Related Work & Insights

- **Plan2Explore** (Sekar et al., 2020): An MBRL exploration baseline utilizing ensemble disagreement as an intrinsic reward; SENSEI adds a semantic bias on top of this.
- **Motif** (Klissarov et al., 2023): Distills rewards by using an LLM to annotate preferences on event captions; SENSEI extends this to a visual version (VLM-Motif).
- **DreamerV3** (Hafner et al., 2023): The underlying world model and policy optimization framework.
- **RL-VLM-F** (Wang et al., 2024): Similar VLM preference distillation but non-model-based, and requires explicit task descriptions.
- **Go-Explore** (Ecoffet et al., 2021): Inspired the go-then-explore switching strategy of SENSEI.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of VLM preference distillation, world model internalized prediction, and adaptive switching is both novel and reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Thoroughly validated across three types of environments with detailed ablation experiments (Go-Explore, zero-prior, annotation noise, data quality).
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and smooth transition between motivation and methodology.
- Value: ⭐⭐⭐⭐ — Provides a practical and scalable paradigm for VLM-guided autonomous exploration, whose value will only grow with advancing VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[ICLR 2026\] World-In-World: World Models in a Closed-Loop World](../../ICLR2026/robotics/world-in-world_world_models_in_a_closed-loop_world.md)
- [\[CVPR 2026\] Dexterous World Models](../../CVPR2026/robotics/dexterous_world_models.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](../../NeurIPS2025/robotics/self-improving_embodied_foundation_models.md)
- [\[ICLR 2026\] Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](../../ICLR2026/robotics/theory_of_space_can_foundation_models_construct_spatial_beliefs_through_active_e.md)

</div>

<!-- RELATED:END -->
