---
title: >-
  [Paper Note] GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration
description: >-
  [ICCV 2025][Remote Sensing][Active Geo-Localization] This paper proposes GeoExplorer, an active geo-localization (AGL) agent that integrates goal-directed extrinsic rewards with curiosity-driven intrinsic rewards. By jointly modeling action-state dynamics and curiosity-based exploration within a reinforcement learning framework, GeoExplorer achieves more robust UAV search strategies and demonstrates superior generalization to unseen targets and environments.
tags:
  - ICCV 2025
  - Remote Sensing
  - Active Geo-Localization
  - Curiosity-Driven Exploration
  - Reinforcement Learning
  - UAV Navigation
  - Multi-Modal Target
date: 2026-05-08
content_hash: 583acda1daddcff9
---

# GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration

**Conference**: ICCV 2025
**arXiv**: [2508.00152](https://arxiv.org/abs/2508.00152)
**Code**: [Project Page](https://limirs.github.io/GeoExplorer/)
**Area**: Remote Sensing
**Keywords**: Active Geo-Localization, Curiosity-Driven Exploration, Reinforcement Learning, UAV Navigation, Multi-Modal Target

## TL;DR

This paper proposes GeoExplorer, an active geo-localization (AGL) agent that integrates goal-directed extrinsic rewards with curiosity-driven intrinsic rewards. By jointly modeling action-state dynamics and curiosity-based exploration within a reinforcement learning framework, GeoExplorer achieves more robust UAV search strategies and demonstrates superior generalization to unseen targets and environments.

## Background & Motivation

Active Geo-Localization (AGL) refers to the task of navigating a UAV agent to a target location within a predefined search region. The target may be specified in multiple modalities (aerial image, ground-level image, or text), but its precise location remains unknown at inference time. This capability is critical for search-and-rescue operations.

Existing methods (e.g., GOMAA-Geo) rely on **extrinsic rewards** (distance-based rewards) and suffer from three key limitations:

**Unreliable distance estimation**: Since target location is unknown at inference time, distance-based rewards cannot be computed directly, reducing the reliability of learned policies.

**Insufficient environment modeling**: These methods only predict action sequences without modeling state transitions, and thus fail to capture how actions alter the environment.

**Weak generalization**: Distance estimation tailored to training environments struggles to transfer to unseen targets and novel environments.

**Core Insight**: Curiosity-driven intrinsic rewards (target-agnostic) are introduced to complement goal-directed extrinsic rewards. The curiosity reward is based on the discrepancy between predicted and actual states, requiring no knowledge of target location, thereby providing dense, target-agnostic, content-aware exploration guidance.

## Method

### Overall Architecture

GeoExplorer training proceeds in three stages:
1. **Feature Representation**: Aligned multi-modal encoders process targets of different modalities.
2. **Action-State Dynamics Modeling (DM)**: Supervised pre-training with a causal Transformer for joint action and state transition modeling.
3. **Curiosity-driven Exploration (CE)**: PPO-based Actor-Critic RL combining extrinsic and intrinsic rewards.

### Key Designs

1. **Multi-Modal Feature Representation**: Three aligned encoders are employed:

    - Aerial image encoder Sat2Cap (ViT, aligned with CLIP)
    - Ground-level image encoder CLIP_img
    - Text encoder CLIP_text
    - All encoders are frozen after pre-training, ensuring targets from different modalities are embedded in a shared feature space.

2. **Joint Action-State Dynamics Modeling**: A causal Transformer (Falcon-7B) simultaneously predicts:

    - Optimal action $\hat{a}_t = \text{CausalTrans}(s_t | x_{t-1}, s_{goal})$ (which action brings the agent closer to the target)
    - State representation $\hat{s}_t = \text{CausalTrans}(a_{t-1} | x_{t-1}, s_{goal})$ (how actions affect the environment)
    - No architectural modifications are required; a state modeling loss $\mathcal{L}_{State} = \sum_{t=1}^{N-1} \|\hat{s}_t - s_t\|^2_2$ is simply appended.
    - Total DM loss: $\mathcal{L}_{DM} = \mathcal{L}_{Action} + \alpha \mathcal{L}_{State}$

3. **Curiosity-Driven Intrinsic Reward**: The discrepancy between the predicted next state $\hat{s}_{t+1}$ and the actual next state $s_{t+1}$, learned during the DM stage, serves as a measure of "surprise":

    - MSE formulation: $r^{in}_t = \|\hat{s}_{t+1} - s_{t+1}\|^2_2$
    - Cosine similarity formulation: $r^{in}_t = -\cos(\hat{s}_{t+1}, s_{t+1})$
    - Final reward: $r^{CE}_t = r^{ex}_t + \beta r^{in}_t$, where $\beta = 0.25$
    - Key: intrinsic rewards are normalized to $[-1, 1]$ before weighting.

### Loss & Training

- DM stage: supervised pre-training on randomly generated {start, goal} trajectory pairs.
- CE stage: the causal Transformer is frozen; only the Actor-Critic action prediction head (MLP) is trained.
- Search grid: $5\times5$; search budget $B=10$; evaluation distances $C\in\{4,5,6,7,8\}$.
- Training is conducted exclusively on the Masa dataset; zero-shot transfer is evaluated on other datasets.

## Key Experimental Results

### Main Results (Tables)

**Masa Dataset Validation Set (Success Rate SR)**:

| Method | C=4 | C=5 | C=6 | C=7 | C=8 |
|------|-----|-----|-----|-----|-----|
| Random | 0.141 | 0.058 | 0.064 | 0.025 | 0.024 |
| DiT | 0.201 | 0.296 | 0.357 | 0.422 | 0.456 |
| GOMAA-Geo | 0.409 | 0.506 | 0.717 | 0.803 | 0.785 |
| **GeoExplorer** | **0.432** | **0.532** | **0.816** | **0.923** | **0.950** |

**Generalization to Unseen Targets (SwissViewMonuments)**:

| Method | C=4 | C=5 | C=6 | C=7 | C=8 |
|------|-----|-----|-----|-----|-----|
| GOMAA-Geo | 0.403 | 0.383 | 0.627 | 0.728 | 0.783 |
| **GeoExplorer (I)** | **0.413** | **0.533** | **0.770** | **0.889** | **0.883** |
| **GeoExplorer (G)** | **0.416** | **0.517** | **0.773** | **0.878** | 0.783 |

### Ablation Study (Table)

| Action Loss | State Loss | $r^{ex}$ | $r^{in}$ | C=4 | C=5 | C=6 | C=7 | C=8 |
|:-:|:-:|:-:|:-:|-----|-----|-----|-----|-----|
| ✓ | | ✓ | | 0.409 | 0.506 | 0.717 | 0.803 | 0.785 |
| ✓ | ✓ | ✓ | | 0.398 | 0.494 | 0.761 | 0.841 | 0.865 |
| ✓ | | ✓ | MSE | 0.389 | 0.494 | 0.741 | 0.862 | 0.914 |
| ✓ | ✓ | ✓ | COS | 0.396 | 0.539 | 0.813 | 0.905 | 0.935 |
| ✓ | ✓ | ✓ | **MSE** | **0.432** | **0.532** | **0.816** | **0.923** | **0.950** |

The full combination of state modeling and curiosity reward yields the best performance; even without explicit supervision of state prediction, the intrinsic reward alone provides measurable gains.

### Key Findings

- Improvements are most pronounced on long-horizon paths: at $C=8$, GeoExplorer outperforms GOMAA-Geo by 0.1643.
- Cross-domain transfer: an average improvement of 0.0556 is observed on xBD-disaster (post-disaster environment with pre-disaster targets).
- The curiosity reward is content-aware: transition patches from forest to urban scenes receive the highest intrinsic rewards.
- More comprehensive exploration: at $C=4$, GeoExplorer visits 30.79% of patches located inside the search region, compared to only 20.08% for GOMAA-Geo.
- The performance advantage scales with larger search budgets (greater exploration capacity).

## Highlights & Insights

- Curiosity-driven RL is introduced into the AGL task for the first time, integrating seamlessly into the sequence modeling framework without additional components.
- Joint action-state modeling requires no modification to the Transformer architecture — a single state loss suffices.
- A new SwissView benchmark is proposed, with the SwissViewMonuments subset specifically designed to evaluate generalization to unseen targets.
- Visualization analysis of curiosity rewards is highly intuitive: semantic transitions (e.g., forest → urban) receive high intrinsic rewards.

## Limitations & Future Work

- The search space is a discrete grid ($5\times5$); future work should extend to continuous state and action spaces.
- Real-world deployment challenges such as UAV ego-pose noise and observation distortion are not addressed.
- The optimal balance between intrinsic and extrinsic rewards warrants further investigation.
- The training set (Masa) covers relatively homogeneous scenes, which may limit performance in more diverse environments.

## Related Work & Insights

- Compared to GOMAA-Geo (action modeling only + extrinsic rewards), GeoExplorer introduces two additional dimensions: state modeling and intrinsic rewards.
- Curiosity-driven RL has been extensively studied in classical control and game-playing tasks; this work applies it to geo-localization for the first time.
- The "by-product" of state prediction is elegantly repurposed to construct intrinsic rewards.

## Rating

- Novelty: ⭐⭐⭐⭐ (first integration of curiosity-driven RL with AGL)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 benchmarks + new dataset + comprehensive ablations + rich visualizations)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, detailed supplementary material)
- Value: ⭐⭐⭐⭐ (practical significance for search-and-rescue UAV deployment)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Scaling Image Geo-Localization to Continent Level](../../NeurIPS2025/remote_sensing/scaling_image_geo-localization_to_continent_level.md)
- [\[ICCV 2025\] Information-Bottleneck Driven Binary Neural Network for Change Detection](information-bottleneck_driven_binary_neural_network_for_change_detection.md)
- [\[CVPR 2026\] RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](../../CVPR2026/remote_sensing/rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)
- [\[ICCV 2025\] GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization](geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo.md)
- [\[AAAI 2026\] UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization](../../AAAI2026/remote_sensing/uniabg_unified_adversarial_view_bridging_and_graph_correspondence_for_unsupervis.md)

<!-- RELATED:END -->
