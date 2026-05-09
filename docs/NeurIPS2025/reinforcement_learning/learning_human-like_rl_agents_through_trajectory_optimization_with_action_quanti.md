---
title: >-
  [Paper Note] Learning Human-Like RL Agents through Trajectory Optimization with Action Quantization
description: >-
  [NeurIPS 2025][Reinforcement Learning][Human-like RL] This paper proposes MAQ (Motion-Action Quantization), a method that discretizes human actions into a finite set of motion primitives via VQ-VAE, then performs trajectory optimization within the quantized action space to train RL agents whose behavioral patterns more closely resemble those of humans.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Human-like RL
  - action quantization
  - trajectory optimization
  - VQ-VAE
  - behavior modeling
date: 2026-05-08
content_hash: e557c51c96d0436e
---

# Learning Human-Like RL Agents through Trajectory Optimization with Action Quantization

**Conference**: NeurIPS 2025
**arXiv**: [2511.15055](https://arxiv.org/abs/2511.15055)
**Code**: Available
**Area**: Reinforcement Learning
**Keywords**: Human-like RL, action quantization, trajectory optimization, VQ-VAE, behavior modeling

## TL;DR

This paper proposes MAQ (Motion-Action Quantization), a method that discretizes human actions into a finite set of motion primitives via VQ-VAE, then performs trajectory optimization within the quantized action space to train RL agents whose behavioral patterns more closely resemble those of humans.

## Background & Motivation

**Background**: Although RL agents can match or surpass human performance on many tasks, their behavioral patterns often differ markedly from human behavior—exhibiting jittery, unnatural, and incoherent actions.

**Limitations of Prior Work**: (1) Continuous action spaces allow agents to exploit physical singularities; (2) reward shaping can only indirectly guide behavioral patterns; (3) imitation learning requires large amounts of expert data and generalizes poorly.

**Key Challenge**: The tension between high task performance and human-like behavior—optimizing for reward tends to produce non-human behavioral strategies.

**Key Insight**: Human motion can be decomposed into a finite set of motion primitives (e.g., walking, reaching); constraining the agent's action space to combinations of these primitives naturally induces human-like behavior.

**Core Idea**: VQ-VAE learns human motion primitives → trajectory optimization is performed in the quantized space → the agent is compelled to adopt human-like action patterns.

## Method

### Overall Architecture

(1) A VQ-VAE learns a discrete motion-primitive codebook from human motion data; (2) the policy network makes decisions within the quantized action space; (3) a trajectory optimizer optimizes trajectories subject to codebook constraints.

### Key Designs

1. **Action Quantization (VQ-VAE)**

   - **Function**: Discretizes the continuous human action space into $K$ primitives.
   - **Mechanism**: An encoder maps action sequences to latent variables, which are quantized via nearest-neighbor lookup in the codebook; a decoder reconstructs the actions. $\mathcal{L}_{VQ} = \|sg[\hat{s}_i] - z_{q_i}\|_2^2 + \beta\|\hat{s}_i - sg[z_{q_i}]\|_2^2$
   - **Design Motivation**: The quantized action space automatically excludes non-human motion patterns.

2. **Policy Learning in the Quantized Space**

   - **Function**: Executes RL within the discrete codebook space.
   - **Mechanism**: The policy network outputs a probability distribution over codebook indices; after index selection, the decoder generates the corresponding continuous action.
   - **Design Motivation**: Transforms the continuous optimization problem into a discrete selection problem, naturally constraining behavioral patterns.

3. **Trajectory Optimization**

   - **Function**: Optimizes entire trajectories subject to quantization constraints.
   - **Mechanism**: Planning methods such as CEM/MPPI sample action sequences from the codebook, evaluate trajectory rewards, and iteratively refine the plan.
   - **Design Motivation**: Trajectory-level optimization provides stronger guarantees of action coherence than step-wise RL.

### Loss & Training

VQ-VAE training: reconstruction loss + commitment loss. RL training: PPO/SAC in the quantized space. Trajectory optimization: CEM sampling + reward evaluation.

## Key Experimental Results

### Main Results

| Method | Task Success Rate↑ | Human Similarity↑ | Action Smoothness↑ |
|--------|-------------------|------------------|--------------------|
| PPO (continuous) | 92% | 0.35 | 0.42 |
| SAC (continuous) | 95% | 0.38 | 0.45 |
| GAIL (imitation) | 78% | 0.72 | 0.81 |
| **MAQ (Ours)** | **91%** | **0.78** | **0.85** |

### Ablation Study

| Configuration | Task Success Rate | Human Similarity | Notes |
|--------------|-----------------|-----------------|-------|
| No quantization (continuous) | 95% | 0.38 | High performance but not human-like |
| Quantization + step-wise RL | 85% | 0.65 | Quantization effective but coherence limited |
| Quantization + trajectory optimization | 88% | 0.75 | Trajectory-level yields better coherence |
| **Full MAQ** | **91%** | **0.78** | **Optimal trade-off** |

### Key Findings

- MAQ substantially outperforms continuous RL on human similarity (0.78 vs. 0.38) with only a 4% drop in task performance.
- A codebook size of $K=256$ yields the optimal balance—too small limits expressiveness, too large weakens the behavioral constraint.
- Trajectory optimization improves coherence significantly over step-wise RL (0.75 vs. 0.65).
- Compared to GAIL, MAQ does not require dense expert demonstrations; only a small amount of motion data is needed to construct the codebook.

## Highlights & Insights

- **Constraint as Inductive Bias**: Guiding behavior by restricting the action space rather than shaping the reward is more direct and controllable. This principle transfers naturally to robot control, game AI, and related domains.
- **Interpretability of Motion Primitives**: Each entry in the codebook corresponds to an interpretable motion pattern, facilitating debugging and analysis.
- **Data Efficiency**: Compared to imitation learning, only a small amount of human motion data is required to train the VQ-VAE, substantially reducing data requirements.

## Limitations & Future Work

- Task performance is slightly below that of unconstrained RL (91% vs. 95%), which may be unacceptable for high-precision tasks.
- A fixed codebook struggles to accommodate entirely novel motion patterns after training.
- Evaluation metrics for human similarity lack a unified, standardized definition.

## Related Work & Insights

- **vs. GAIL**: GAIL requires dense expert trajectories, whereas MAQ only needs motion data for codebook construction.
- **vs. MotionVAE**: MotionVAE targets animation generation; MAQ is the first to apply quantized actions to RL policy optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The combination of action quantization and trajectory optimization is a genuinely novel approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task validation with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and well-organized presentation.
- **Value**: ⭐⭐⭐⭐ Human-like AI is an important research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering](enhancing_interpretability_in_deep_reinforcement_learning_through_semantic_clust.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[AAAI 2026\] Know your Trajectory -- Trustworthy Reinforcement Learning Deployment through Importance-Based Trajectory Analysis](../../AAAI2026/reinforcement_learning/know_your_trajectory_--_trustworthy_reinforcement_learning_deployment_through_im.md)
- [\[NeurIPS 2025\] Human-Inspired Multi-Level Reinforcement Learning](human-inspired_multi-level_reinforcement_learning.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)

</div>

<!-- RELATED:END -->
