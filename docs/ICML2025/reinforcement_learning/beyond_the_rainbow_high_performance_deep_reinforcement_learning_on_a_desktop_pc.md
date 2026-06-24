---
title: >-
  [Paper Note] Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC
description: >-
  [ICML 2025][Reinforcement Learning][Deep Reinforcement Learning] BTR (Beyond The Rainbow) is proposed—integrating 6 RL improvements into Rainbow DQN to train on Atari-60 to an IQM of 7.4 (compared to 1.9 for Rainbow) within 12 hours on a single desktop PC, and successfully training agents to play 3D games like Super Mario Galaxy, Mario Kart, and Mortal Kombat for the first time.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Deep Reinforcement Learning"
  - "Rainbow DQN"
  - "Atari"
  - "Computational Efficiency"
  - "3D Games"
date: 2026-05-08
content_hash: 066544cb17ab91e2
---

# Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC

**Conference**: ICML 2025  
**arXiv**: [2411.03820](https://arxiv.org/abs/2411.03820)  
**Code**: [https://github.com/VIPTankz/BTR](https://github.com/VIPTankz/BTR)  
**Area**: Reinforcement Learning  
**Keywords**: Deep Reinforcement Learning, Rainbow DQN, Atari, Computational Efficiency, 3D Games

## TL;DR
BTR (Beyond The Rainbow) is proposed—integrating 6 RL improvements into Rainbow DQN to train on Atari-60 to an IQM of 7.4 (compared to 1.9 for Rainbow) within 12 hours on a single desktop PC, and successfully training agents to play 3D games like Super Mario Galaxy, Mario Kart, and Mortal Kombat for the first time.

## Background & Motivation

**Background**: SOTA RL algorithms (e.g., MuZero, Agent57) require distributed computing clusters and weeks of training time, going far beyond the capabilities of small laboratories and hobbyists.

**Limitations of Prior Work**: Rainbow DQN requires 34,200 GPU hours (1,435 days), and newer algorithms demand even higher computation. Unlike NLP/CV, RL lacks foundation models for fine-tuning; every environment must be trained from scratch.

**Key Challenge**: The contradiction between high performance and computational accessibility.

**Goal**: To design a high-performance RL algorithm that can be trained rapidly on a desktop PC.

**Key Insight**: Continuing the methodology of Rainbow DQN—selecting and combining independent improvement components.

**Core Idea**: Selecting 6 improvements that balance performance and computational efficiency to assemble BTR.

## Method

### Overall Architecture
BTR replaces or adds 6 components based on Rainbow DQN:
1. Distributional RL (retained)
2. N-step returns (retained)
3. Adam optimizer replacing RMSProp
4. Periodic layer reset to address loss of plasticity
5. Large batch training (batch size 512)
6. High update-to-data (UTD) ratio (UTD ratio = 8)

### Key Designs

1. **Periodic Layer Reset**:

    - **Function**: Periodically reinitialize the last few layers of the network
    - **Mechanism**: During RL training, the network gradually loses plasticity (increase in dormant neurons); resetting restores the learning capacity
    - **Design Motivation**: Replacing methods that increase network size to maintain computational efficiency

2. **High UTD**:

    - **Function**: Executes 8 network updates for every 1 step of environment interaction collected
    - **Mechanism**: Improving sample efficiency to reduce the number of environment interactions
    - **Design Motivation**: On a desktop PC, GPU computation is cheaper than CPU environment simulation

3. **2-Hot Distributional RL**:

    - **Function**: Replaces the probability mass representation of C51 with a more stable 2-hot encoding
    - **Mechanism**: Each return value is assigned to the two nearest bins (similar to interpolation), using cross-entropy loss
    - **Design Motivation**: More stable than the KL-divergence loss used in C51

### Loss & Training
- Distributional RL + 2-hot encoding + cross-entropy loss
- Adam optimizer, learning rate 6.25e-5
- Environment frame skip of 4, standard preprocessing with grayscale + frame-stacking

## Key Experimental Results

### Main Results
Atari-60 benchmark (200M frames):

| Algorithm | IQM ↑ | Training Time | Hardware |
|------|-------|---------|------|
| DQN | 0.5 | ~35h | Desktop PC |
| Rainbow DQN | 1.9 | 35h | Desktop PC |
| Dreamer-v3 | 2.5 | - | GPU Cluster |
| **BTR** | **7.4** | **<12h** | **Desktop PC** |

Superhuman performance: 52/60 games

### Ablation Study

| Component Removed | Change in IQM | Description |
|---------|---------|------|
| -Layer Reset | -2.3 | Most critical component |
| -High UTD | -1.8 | Key to sample efficiency |
| -2-hot Distributional | -1.2 | Stability improvement |
| -Adam | -0.5 | Mild improvement |

### Key Findings
- BTR achieves first-of-its-kind success in 3D games: Super Mario Galaxy (final level), Mario Kart (Rainbow Road), and Mortal Kombat
- Periodic layer reset is the most critical single improvement—directly addressing the loss of plasticity in RL
- High UTD is more efficient than increasing environment interaction

## Highlights & Insights
- **The philosophy of "Democratizing RL"**—emphasizing computational accessibility, enabling small labs and hobbyists to conduct cutting-edge RL research
- The principles of component selection are highly referenceable: choosing simple and efficient designs over complex ones (e.g., world models or search)
- Success in 3D games demonstrates the generalization of the methodology, not being limited to Atari

## Limitations & Future Work
- Only applicable to discrete action spaces (due to the constraints of the DQN family)
- Continual learning and multi-task settings have not been tested
- 3D game results remain preliminary (single environment, single level)

## Related Work & Insights
- **vs Rainbow DQN**: Same design methodology of combination, but with more modern components and a 4× performance gain
- **vs Dreamer-v3**: A world-model-based method, requiring more computation but theoretically being more sample-efficient
- **Insights for RL Research**: Combining simple methods can outperform complex approaches

## Rating
- Novelty: ⭐⭐⭐ Combining existing methods, lacking fundamental innovation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 60 games + 3D games + detailed ablation studies
- Writing Quality: ⭐⭐⭐⭐ Clear and practical
- Value: ⭐⭐⭐⭐⭐ Making high-performance RL accessible to everyone

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification](../../NeurIPS2025/reinforcement_learning/kimina_lean_server_a_high-performance_lean_server_for_large-scale_verification.md)
- [\[ICML 2025\] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)
- [\[NeurIPS 2025\] Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mind_the_gap_the_challenges_of_scale_in_pixel-based_deep_reinforcement_learning.md)
- [\[ICML 2025\] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](../../NeurIPS2025/reinforcement_learning/confounding_robust_deep_reinforcement_learning_a_causal_approach.md)

</div>

<!-- RELATED:END -->
