---
title: >-
  [Paper Note] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data
description: >-
  [NeurIPS 2025][Robotics][manipulation concepts] This paper proposes a self-supervised framework that learns hierarchical manipulation concepts from unlabeled multi-modal robot demonstrations. It organizes representations…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "manipulation concepts"
  - "hierarchical representation"
  - "cross-modal correlation"
  - "multi-timescale subgoals"
  - "self-supervised learning"
date: 2026-05-08
content_hash: 0181a27a76ed9176
---

# HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data

**Conference**: NeurIPS 2025
**arXiv**: [2510.11321](https://arxiv.org/abs/2510.11321)
**Code**: [HiMaCon](https://github.com/zrllrz/HiMaCon)
**Area**: Robot Manipulation / Representation Learning / Imitation Learning
**Keywords**: manipulation concepts, hierarchical representation, cross-modal correlation, multi-timescale subgoals, self-supervised learning

## TL;DR
This paper proposes a self-supervised framework that learns hierarchical manipulation concepts from unlabeled multi-modal robot demonstrations. It organizes representations via a cross-modal correlation network and a multi-horizon future predictor, enhancing the generalization of imitation learning policies to novel objects, unseen obstacles, and new environments.

## Background & Motivation

1. **Generalization bottleneck**: Current robot manipulation policies perform well within the training distribution but frequently fail when encountering unseen obstacles, novel object appearances, or new environments (e.g., trained without obstacles but tested with them).
2. **Limitations of representation learning**:
    - Unimodal methods (purely visual or purely proprioceptive) fail to capture cross-modal functional invariances.
    - Temporal representation learning methods overlook the hierarchical temporal structure inherent in manipulation tasks.
    - Cross-modal alignment methods perform only feature concatenation/alignment without modeling inter-modal correlations.
3. **Core assumption**: Manipulation concepts (e.g., "placing an object into a container") encode invariant relational patterns that persist across objects and environments. By jointly modeling cross-modal correlations and multi-timescale subgoals, transferable hierarchical manipulation concepts can be learned.

## Core Problem
How to automatically discover hierarchical manipulation concepts from unlabeled multi-modal robot demonstrations such that they encode cross-modal functional invariances and are organized as multi-timescale subgoals, thereby enhancing policy generalization?

## Method

### Overall Architecture (Two-Stage)
**Stage 1: Concept Discovery** — A concept encoder $\mathcal{E}$ maps multi-modal observations to concept latents, trained with two objectives:
- Cross-Modal Correlation Network (CMCN) $\mathcal{C}$
- Multi-Horizon Future Predictor (MHFP) $\mathcal{F}$

**Stage 2: Policy Enhancement** — The learned concepts are integrated into imitation learning via a joint prediction head.

### Concept Encoder

Given a trajectory $\tau_i = \{(\mathbf{o}_i^t, a_i^t)\}_{t=1}^{T_i}$ with multi-modal observations $\mathbf{o}_i^t = \{o_i^{1,t},...,o_i^{M,t}\}$, the encoder produces a concept sequence:

$$\mathbf{z}_i \leftarrow \mathcal{E}(\mathbf{o}_i;\Theta_\mathcal{E})$$

A Transformer captures temporal dependencies, yielding $z_i^t \in \mathbb{R}^Z$ at each timestep.

### Cross-Modal Correlation Learning

**Core Idea**: Manipulation concepts should capture cross-modal correlations (persistent patterns across vision, proprioception, and force feedback) rather than simply concatenating features. This is achieved by maximizing conditional mutual information:

$$\max_\mathbf{Z}\sum_{S\subsetneq[M], S\neq\emptyset} \mathbb{I}(\mathbf{O}_S : \mathbf{O}_{[M]\setminus S} \mid \mathbf{Z})$$

A mask-and-predict strategy is employed: a random subset of modalities is masked, and the remaining modalities together with the concept latent are used to reconstruct all observations:

$$\mathcal{L}_\text{mm}(t, \tau_i) = \mathbb{E}_S \|\mathcal{C}(o_i^{[M]\setminus S,t}, z_i^t;\Theta_c) - o_i^t\|$$

### Multi-Horizon Subgoal Representation

Concept similarity is quantified via spherical distance: $\text{dist}(z,u) = \frac{1}{\pi}\arccos\langle\frac{z}{\|z\|_2}, \frac{u}{\|u\|_2}\rangle$

A coherence threshold $\epsilon$ governs subprocess granularity: small $\epsilon$ yields short-horizon fine-grained subgoals; large $\epsilon$ yields long-horizon coarse-grained goals. Subprocesses are partitioned automatically based on concept latent consistency.

The multi-horizon predictor learns to predict the terminal observation of each subprocess:

$$\mathcal{L}_\text{mh}(t, \tau_i) = \mathbb{E}_\epsilon\|\mathcal{F}(\mathbf{o}_i^t, z_i^t, \epsilon;\Theta_f) - \mathbf{o}_i^{g(t;\mathbf{z}_i,\epsilon)}\|$$

### Overall Training Objective

$$\mathcal{L}_z(t, \tau_i) = \lambda_\text{mm}\mathcal{L}_\text{mm}(t, \tau_i) + \lambda_\text{mh}\mathcal{L}_\text{mh}(t, \tau_i)$$

### Policy Enhancement (Stage 2)

Concept prediction is integrated into imitation learning as a regularization term:

$$\mathcal{L}_\pi(t, \tau_i, \ell_i) = \|\hat{a}_i^t - a_i^t\| + \lambda_\text{mc}\|\hat{z}_i^t - z_i^t\|$$

The policy comprises a shared backbone $\pi_h$, a concept prediction head $\pi_z$, and an action decoding head $\pi_a$, and is compatible with both ACT and Diffusion Policy.

## Key Experimental Results

### LIBERO Benchmark (Concepts trained on L90 only)

| Setting | Policy | Plain | XSkill | RPT | **HiMaCon** |
|---------|--------|-------|--------|-----|-------------|
| L90 (original tasks) | ACT | 46.6 | 73.4 | 68.8 | **74.8** |
| L90 (original tasks) | DP | 75.1 | 87.7 | 84.3 | **89.6** |
| L-LONG (long-horizon transfer) | ACT | 54.0 | 55.0 | 59.0 | **63.0** |
| L-LONG (long-horizon transfer) | DP | 34.1 | 73.0 | 61.3 | **89.0** |
| L-GOAL (new environment generalization) | ACT | 57.0 | 77.0 | 75.0 | **81.0** |
| L-GOAL (new environment generalization) | DP | 90.7 | 93.0 | 91.5 | **95.7** |

Key findings:
- On long-horizon transfer (L-LONG + DP), HiMaCon improves over Plain by **54.9 percentage points** (34.1→89.0).
- HiMaCon consistently leads on new-environment generalization, demonstrating strong concept transferability.
- HiMaCon achieves the best or second-best performance against all 11 baseline methods.

### Real Robot Experiments
The paper validates deployment on a physical robot, where the concept-enhanced policy successfully adapts to unseen obstacles (e.g., navigating around an obstacle while placing a cup), whereas the policy without concepts fails directly.

## Highlights & Insights
- **Theoretically grounded motivation**: The framework draws inspiration from cognitive science (concept formation driven by cross-modal correlations) and motor control (hierarchical goal organization).
- **Elegant self-supervised design**: The mask-and-predict strategy simultaneously achieves cross-modal correlation learning and information compression.
- **$\epsilon$ controls hierarchy**: A single continuous parameter naturally generates subgoal hierarchies ranging from short to long horizons, without requiring a predefined number of levels.
- **Architecture-agnostic**: Concept enhancement is realized through a joint prediction head, making it compatible with diverse policy architectures including ACT and Diffusion Policy.
- **Interpretable concepts**: The learned concepts automatically cluster into manipulation primitives resembling human understanding (grasping, placing, aligning, etc.).

## Limitations & Future Work
- The concept encoder requires pretraining on demonstration data (Stage 1), adding complexity to the training pipeline.
- The subprocess segmentation based on spherical distance and threshold $\epsilon$ implicitly assumes a particular geometric structure of the concept latent space.
- Tasks in LIBERO are relatively simple; effectiveness on more complex bimanual or contact-rich manipulation has not been validated.
- Real robot experiments are limited in scale, with insufficient statistical significance.

## Rating
- Novelty: ⭐⭐⭐⭐ — Hierarchical concept discovery combining cross-modal correlation and multi-horizon subgoals is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 11 baselines, 3 evaluation settings, 2 policy types, and real robot experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with tight connections between motivation and design.
- Value: ⭐⭐⭐⭐⭐ — A significant contribution to representation learning for robot manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[NeurIPS 2025\] Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms](operation_veja_fixing_fundamental_concepts_missing_from_modern_roleplaying_train.md)
- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)
- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)

</div>

<!-- RELATED:END -->
