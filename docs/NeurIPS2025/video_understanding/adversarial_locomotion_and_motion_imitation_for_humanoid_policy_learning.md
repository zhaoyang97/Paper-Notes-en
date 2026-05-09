---
title: >-
  [Paper Note] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning
description: >-
  [NeurIPS 2025][Video Understanding][Humanoid Robot] ALMI proposes an upper-lower body adversarial training framework: the lower-body policy learns robust locomotion under upper-body motion perturbations, while the upper-body policy learns precise motion imitation under lower-body locomotion perturbations. Through iterative adversarial training converging to a Nash equilibrium, the framework enables stable whole-body coordinated control on the Unitree H1-2 real robot.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Humanoid Robot
  - Adversarial Learning
  - Motion Imitation
  - Whole-Body Control
  - Sim-to-Real
date: 2026-05-08
content_hash: 5c932862442cdadf
---

# Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning

**Conference**: NeurIPS 2025
**arXiv**: [2504.14305](https://arxiv.org/abs/2504.14305)
**Code**: [Project Page](https://almi-humanoid.github.io)
**Area**: Video Understanding
**Keywords**: Humanoid Robot, Adversarial Learning, Motion Imitation, Whole-Body Control, Sim-to-Real

## TL;DR
ALMI proposes an upper-lower body adversarial training framework: the lower-body policy learns robust locomotion under upper-body motion perturbations, while the upper-body policy learns precise motion imitation under lower-body locomotion perturbations. Through iterative adversarial training converging to a Nash equilibrium, the framework enables stable whole-body coordinated control on the Unitree H1-2 real robot.

## Background & Motivation

### State of the Field

**Background**: Existing methods employ monolithic RL policies to simultaneously control all joints, using motion tracking error as the reward signal.

**Limitations of Prior Work**: Monolithic policies neglect the functional distinction between upper and lower bodies; the high DoF count (21) makes training difficult; prioritizing tracking accuracy over balance leads to frequent falls on real hardware.

**Key Challenge**: Large upper-body motions destabilize balance, while rapid lower-body locomotion degrades upper-body tracking accuracy — a naturally adversarial relationship.

**Goal**: Enable the upper and lower bodies to independently learn their respective tasks while ensuring whole-body coordination.

**Key Insight**: Model the upper and lower bodies as two players in a zero-sum game.

**Core Idea**: Adversarial training encourages the lower body to learn to maintain balance regardless of upper-body motions, and the upper body to achieve precise tracking regardless of lower-body locomotion.

## Method

### Overall Architecture
Two coupled zero-sum Markov games trained in alternation: (1) when learning $\pi^l$, the upper body acts as the adversary; (2) when learning $\pi^u$, the lower body acts as the adversary.

### Key Designs

1. **Zero-Sum Game Formulation**:

   - Mechanism: $\max_{\pi^l} \min_{\pi^u} V_\rho^l$ and $\max_{\pi^u} \min_{\pi^l} V_\rho^u$
   - Theoretical guarantee: Theorem 3.1 proves convergence to an $\epsilon$-approximate Nash equilibrium

2. **Command-Space Adversary (Simplified Implementation)**:

   - Instead of directly optimizing opponent parameters, adversarial commands are sampled (more extreme motions / higher velocities)
   - An Arm Curriculum progressively increases adversarial difficulty

3. **PPO Training**: 3 rounds of adversarial iteration, 4096 parallel environments, approximately 17 hours

### Loss & Training
The model is trained end-to-end with an objective that jointly optimizes task loss and regularization terms.

## Key Experimental Results

### Main Results (CMU MoCap, 1122 motion clips)

| Method | $E_{vel}$↓ | $E_{jpe}^{upper}$↓ | Survival↑ |
|--------|-----------|-------------------|----------|
| Exbody (monolithic) | 0.238 | 0.356 | 89.1% |
| ALMI (monolithic) | 0.139 | 0.576 | 99.9% |
| **ALMI** | **0.114** | **0.193** | **100%** |

### Ablation Study

| Configuration | Survival (Hard)↑ |
|---------------|-----------------|
| ALMI (full) | 97.2% |
| w/o curriculum | 96.1% |
| w/o adv. (round 1) | 93.1% |

### Key Findings
- Each round of adversarial iteration improves robustness
- Successful real-robot deployment
- ALMI-X dataset: 80K+ whole-body control trajectories with language descriptions

## Highlights & Insights
- Modeling the functional distinction between upper and lower bodies as an adversarial game is the core innovation
- Command-space adversary is a practically essential simplification
- A rare combination of theoretical guarantees and real-world deployment

## Limitations & Future Work
- Upper-body control is limited to joint position tracking, constraining expressiveness
- Real-robot experiments are relatively limited, lacking quantitative evaluation

## Related Work & Insights
- **vs. Exbody/Exbody2**: Monolithic policies that do not distinguish upper and lower body functions
- **vs. Decoupled control methods**: Also decouple the two bodies but lack adversarial training to ensure coordination

## Rating
- Novelty: ⭐⭐⭐⭐ Insightful adversarial game formulation
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulation + real robot + ablation + dataset
- Writing Quality: ⭐⭐⭐⭐ Clear methodology presentation
- Value: ⭐⭐⭐⭐ Practical value for humanoid robot control

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy](../../AAAI2026/video_understanding/coordinated_humanoid_robot_locomotion_with_symmetry_equivariant_reinforcement_le.md)
- [\[NeurIPS 2025\] KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills](kungfubot_physics-based_humanoid_whole-body_control_for_learning_highly-dynamic_.md)
- [\[NeurIPS 2025\] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)

<!-- RELATED:END -->
