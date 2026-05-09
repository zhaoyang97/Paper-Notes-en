---
title: >-
  [Paper Note] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents
description: >-
  [NeurIPS 2025][Reinforcement Learning][Vision-language pretraining] This paper proposes AcTOL, which learns ordered and continuous vision-language representations via a visual-language ordering loss and a Brownian bridge constraint, without relying on rigid goal-reaching assumptions, achieving significant improvements on downstream simulated and real-world robot manipulation tasks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Vision-language pretraining
  - embodied intelligence
  - imitation learning
  - temporal consistency
  - Brownian bridge
date: 2026-05-08
content_hash: a1665bf199007675
---

# Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents

**Conference**: NeurIPS 2025
**arXiv**: [2502.01218](https://arxiv.org/abs/2502.01218)
**Code**: [https://actol-pretrain.github.io/](https://actol-pretrain.github.io/)
**Area**: Reinforcement Learning
**Keywords**: Vision-language pretraining, embodied intelligence, imitation learning, temporal consistency, Brownian bridge

## TL;DR
This paper proposes AcTOL, which learns ordered and continuous vision-language representations via a visual-language ordering loss and a Brownian bridge constraint, without relying on rigid goal-reaching assumptions, achieving significant improvements on downstream simulated and real-world robot manipulation tasks.

## Background & Motivation

### State of the Field

**State of the Field**: Pretraining vision-language representations on human action videos to reduce dependence on expert robot demonstrations is a promising direction. Methods such as R3M, LIV, and DecisionNCE employ temporal contrastive learning.

**Limitations of Prior Work**: Existing methods rely on a "goal-reaching" assumption—that the semantic alignment between language instructions and video frames improves monotonically toward later frames. However, actions in real videos may terminate early or be followed by irrelevant content, leading to erroneous vision-language associations.

**Root Cause**: Real human action videos are coarsely annotated and noisy, rendering rigid assumptions invalid.

**Core Idea**: Exploiting the intrinsic temporal consistency of videos, the learned representations are required to satisfy **ordering** (frames closer in time exhibit smaller semantic divergence) and **continuity** (representations of adjacent frames transition smoothly).

## Method

### Key Designs

1. **Vision-Language Ordering (VLO) Loss**:

    - **Mechanism**: For an anchor frame $o_i$ and an arbitrary frame pair $(o_j, o_k)$, the semantic alignment differential is defined as $\mathfrak{R}(\mathbf{v}_i, \mathbf{v}_j, \mathbf{l}) = -\|\text{sim}(\mathbf{v}_i, \mathbf{l}) - \text{sim}(\mathbf{v}_j, \mathbf{l})\|_2$
    - The negative sample set $\mathcal{N}_{i,j}$ selects frames that are **temporally farther** away, and contrastive training is performed using an InfoNCE-style loss.
    - **Theoretical guarantee**: When $\mathcal{L}_{VLO}$ approaches its lower bound $\mathcal{L}^*$, the learned representations provably satisfy the VLO property.

2. **Brownian Bridge Constraint**:

    - Inter-frame intervals in videos are modeled as a Brownian bridge process: the mean follows linear interpolation and the variance is maximized at intermediate time steps.
    - Loss: $\mathcal{L}_{BB} = \frac{1}{T}\sum_{t} \frac{1}{2\text{Var}[\mathbf{B}(t)]}\|\mathbf{v}_t - \mathbb{E}[\mathbf{B}(t)]\|^2$
    - Enforces local smoothness in the visual representation space.

3. **Language Robustness**: It is theoretically proven that, under language perturbations satisfying $\|\mathbf{l} - \mathbf{l}'\| \leq \delta_l$, the change in semantic alignment is bounded by $2C\delta_l$.

## Key Experimental Results

### Main Results — Simulation Success Rate (15 demos)

| Method | Franka Kitchen | Metaworld |
|------|---------------|-----------|
| CLIP | 27.47 | 60.33 |
| R3M | 42.20 | 56.50 |
| LIV | 42.73 | 64.33 |
| DecisionNCE | 43.20 | 59.08 |
| AcTOL w/o BB | 54.20 | 70.83 |
| **AcTOL** | **61.80 (+43%)** | **74.13 (+15%)** |

### Real Robot — Unitree D1

| Method | Pick Cup | Open Drawer | Close Drawer |
|------|---------|------------|-------------|
| DecisionNCE | 20% | 40% | 60% |
| **AcTOL** | **50%** | **80%** | **90%** |

### Key Findings
- The Brownian bridge constraint contributes substantially (AcTOL vs. w/o BB: +7.6% on Franka Kitchen).
- AcTOL exhibits virtually no performance degradation under language perturbations, whereas LIV degrades by 11.9%.
- With as few as 5 demonstrations, AcTOL surpasses competing methods that use 15–25 demonstrations.

## Highlights & Insights
- **Not assuming the final frame as the goal** is the key innovation—only relative temporal distances between frames are used to constrain representations, yielding greater robustness.
- The idea of using a **Brownian bridge as a continuity regularizer** is elegant: it naturally introduces uncertainty modeling into temporal representations.

## Limitations & Future Work
- The approach may not generalize to cyclic or repetitive actions (e.g., stirring), where the temporal ordering assumption does not hold.
- Pretraining is conducted solely on EPIC-KITCHENS-100; performance on larger-scale datasets remains unvalidated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of VLO and Brownian bridge is novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers simulation, real-world, robustness, ablation, and fine-tuning.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical analysis is rigorous and experimental design is meticulous.
- Value: ⭐⭐⭐⭐⭐ Significantly advances the frontier of embodied pretraining.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Towards Provable Emergence of In-Context Reinforcement Learning](towards_provable_emergence_of_in-context_reinforcement_learning.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [\[ICCV 2025\] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](../../ICCV2025/reinforcement_learning/navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)

<!-- RELATED:END -->
