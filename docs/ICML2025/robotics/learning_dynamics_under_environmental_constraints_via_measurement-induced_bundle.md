---
title: >-
  [Paper Note] Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures
description: >-
  [ICML 2025][Robotics][fiber bundle] A geometric framework is proposed to unify measurement uncertainty, system constraints, and dynamics learning by leveraging the fiber bundle structures naturally induced by the measurement process. By defining a measurement-aware control barrier function (mCBF) on the fiber bundle and combining it with Neural ODEs for learning continuous-time dynamics, the method achieves a 96.3% success rate and a 99.3% constraint satisfaction rate across…
tags:
  - "ICML 2025"
  - "Robotics"
  - "fiber bundle"
  - "measurement-aware CBF"
  - "Neural ODE"
  - "safe learning control"
  - "geometric constraints"
date: 2026-05-08
content_hash: 95d49995930e2c09
---

# Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures

**Conference**: ICML 2025  
**arXiv**: [2505.19521](https://arxiv.org/abs/2505.19521)  
**Code**: [GitHub](https://github.com/ContinuumCoder/Measurement-Induced-Bundle-for-Learning-Dynamics/)  
**Area**: Safe Learning Control / Geometric Deep Learning  
**Keywords**: fiber bundle, measurement-aware CBF, Neural ODE, safe learning control, geometric constraints

## TL;DR

A geometric framework is proposed to unify measurement uncertainty, system constraints, and dynamics learning by leveraging the fiber bundle structures naturally induced by the measurement process. By defining a measurement-aware control barrier function (mCBF) on the fiber bundle and combining it with Neural ODEs for learning continuous-time dynamics, the method achieves a 96.3% success rate and a 99.3% constraint satisfaction rate across three robotic control tasks.

## Background & Motivation

**Background**: In robotics and related fields, learning unknown dynamics under environmental constraints is a fundamental problem. Control Barrier Functions (CBFs) serve as a powerful tool to ensure constraint satisfaction; however, classical CBF frameworks treat measurements as external observations rather than intrinsic components of the system's geometric structure.

**Limitations of Prior Work**:

1. Classical CBFs require complete global knowledge of constraints, which is infeasible when only local sensor measurements are available.

2. Probabilistic filtering methods (e.g., Kalman filtering) treat measurement uncertainty as external disturbances rather than exploiting its intrinsic geometric information.

3. Existing learning methods (e.g., Neural CBF, SafeLearn) treat measurements as perfect observations, which limits their robustness in practical deployment.

4. Physics-informed and geometric methods preserve structure but require global geometric information, making them incapable of handling local measurement uncertainty.

**Key Insight**: Local sensor measurements (such as force sensors or distance detectors), despite being imperfect, contain sufficient geometric information regarding constraints and dynamics—the key lies in properly utilizing this local structure rather than requiring global knowledge.

## Method

### Overall Architecture

State space manifold $M$ + measurement space $Y \rightarrow$ projection map $\pi$ induces fiber bundle structure $E = M \times Y \rightarrow$ connection $\nabla$ defined on the fiber bundle (encoding the state-measurement geometric relationship) $\rightarrow$ safety certificate $\Phi$ defined based on the fiber bundle $\rightarrow$ measurement-aware control barrier function (mCBF) $\rightarrow$ Neural ODE learning dynamics on the fiber bundle $\rightarrow$ safety-constrained policy update.

### Key Designs

1. **Measurement-Induced Fiber Bundle Structure**

    - The relationship between the state $x \in M$ and measurement $y = h(x) + v$ naturally defines a projection $\pi: E \rightarrow M$.
    - The fiber $\pi^{-1}(x)$ for each state $x$ characterizes all possible measurements under that state (bounded by the noise constraint $\delta_v$).
    - The connection $\nabla$ on the fiber bundle couples the state and measurement dynamics through a feedback gain operator $K(x)$.
    - Symmetries are captured via compatible Lie group actions, allowing for dimensionality reduction.

2. **Measurement-Aware Control Barrier Function (mCBF)**

    - A smooth function $b: E \rightarrow \mathbb{R}$ is defined on the fiber bundle $E$, satisfying three conditions:
    - (1) $b(x,y) \ge 0$ implies $x \in S_0$ (the safe set).
    - (2) Lie derivative condition along admissible vector fields: $\inf_u [L_f b + (L_g b)w + \alpha(b)] \ge 0$.
    - (3) Lipschitz continuity: $|b(x, y_1) - b(x, y_2)| \le L_b \cdot d(y_1, y_2)$—ensuring bounded sensitivity of the safety certificate to measurement noise.
    - Safety guarantee: $P(x(t) \in S_0, \forall t \ge 0) \ge 1 - \exp(-c/\delta_v^2)$.

3. **Uncertainty-Weighted Learning**

    - A covariance matrix $\Sigma_i$ weighting is introduced in the Neural ODE learning objective to reduce the weight of data points with high uncertainty.
    - Learning convergence guarantee: $\|\hat{f} - f\| \le c_1 \exp(-\lambda_1 t) + c_2 \delta_v$ (exponential convergence with the bound determined by measurement noise).
    - The safety-constrained policy update is achieved by projecting onto the safe policy set.

### Loss & Training

- Underlying RL framework: Soft Actor-Critic (SAC)
- Network architecture: Three-layer MLP (128-64-32, ReLU), with a $\tanh$ activation in the output layer of the barrier function to ensure boundedness.
- Adam optimizer, mixed-precision training, NVIDIA RTX 3090.
- Physical simulation for the three experimental tasks is based on the Genesis physics engine.

## Key Experimental Results

### Main Results

**Comprehensive comparison across three robotic tasks (summary of 500 experiments for worm, 400 for robotic arm, and 300 for quadrotor)**

| Method | Success Rate ↑ | Path Length ↓ | Constraint Satisfaction Rate ↑ |
|------|---------|----------|------------|
| **Ours** | **96.3%** | **18.5±0.7m** | **99.3%** |
| Neural-CBF | 84.0% | 22.3±0.8m | 98.7% |
| SafetyNet | 86.7% | 22.1±0.7m | 98.7% |
| SafeLearn | 82.3% | 23.8±0.9m | 98.3% |
| GEM | 76.3% | 24.0±0.9m | 98.0% |
| GPMPC | 67.7% | 26.9±0.9m | **99.7%** |
| RobustSafe | 70.0% | 26.5±0.8m | 98.7% |
| SafeRL | 70.3% | 26.4±0.8m | 98.7% |

### Ablation Study

| Component | Success Rate ↑ | Description |
|------|---------|------|
| Full Method | 96.3% | Fiber bundle + mCBF + Neural ODE |
| Without Fiber Bundle Structure | ~88% | Degenerates to standard CBF + learning |
| Without Uncertainty Weighting | ~85% | Treats all data points with equal weight |
| Without mCBF Lipschitz Condition | ~90% | Safety certificate becomes overly sensitive to noise |

### Key Findings

- Compared with the strongest baseline SafetyNet (86.7%), the success rate increases by 9.6 percentage points, and the path length is shortened from 22.1m to 18.5m.
- Although GPMPC achieves the highest constraint satisfaction rate (99.7%), its success rate is only 67.7%—over-conservatism leads to excessively long paths.
- The fiber bundle structure allows the safety boundary to adapt based on measurement quality: it becomes more conservative in areas with high measurement uncertainty and more aggressive in areas with low uncertainty.
- The proposed method outperforms baselines across three distinct tasks (soft worm, rigid robotic arm, and aerial quadrotor), demonstrating strong generalization capability.

## Highlights & Insights

- Elevating measurement uncertainty from "noise to be handled" to "valuable geometric information"—this perspective shift is the core contribution.
- The fiber bundle framework provides a unified mathematical language to handle measurements, constraints, and dynamics, which is more elegant than piecemeal multi-component approaches.
- The Lipschitz condition of mCBF formalizes the graceful degradation of safety guarantees—rather than an all-or-nothing outcome, the probability of safety varies smoothly with noise.
- Both theoretical guarantees (exponential convergence and probabilistic safety bounds) and experimental performance are outstanding.

## Limitations & Future Work

- All experiments were conducted in simulation, lacking real-world hardware validation.
- The fiber bundle framework assumes a known functional form of the measurement map $h(x)$, whereas sensor characteristics may be unknown in practice.
- Theoretical analysis assumes bounded noise (sub-Gaussian), and robustness to heavy-tailed noise has not been verified.
- The computational overhead is higher compared to simple CBFs (requiring maintenance of the fiber bundle structure and connection computation).
- The three-layer MLP network is relatively shallow; more powerful function approximators may be required for more complex systems.

## Related Work & Insights

- **vs Neural-CBF/SafeLearn**: These methods learn safety certificates but treat measurements as perfect observations, whereas this work embeds measurement uncertainty directly into the geometric framework.
- **vs GPMPC**: GP-MPC methods offer strong uncertainty quantification but generate overly conservative trajectories (yielding only a 67.7% success rate).
- **vs GeoPath/GEM**: These geometric methods require global geometric information, whereas this work only relies on local sensor measurements.
- **Insight**: Combining classical control theory concepts (such as fiber bundles, connections, and symmetries) with deep learning is a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The perspective of unifying measurements, constraints, and learning via a fiber bundle framework is highly original.
- Experimental Thoroughness: ⭐⭐⭐ The evaluation across three simulation tasks and 12 baselines is comprehensive, but real-world validation is missing.
- Writing Quality: ⭐⭐⭐ The theoretical parts are mathematically dense, making readability challenging for readers without a geometric background.
- Value: ⭐⭐⭐⭐ Provides a novel theoretical framework and practical solution for safe learning control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation](../../ICLR2026/robotics/momagen_generating_demonstrations_under_soft_and_hard_constraints_for_multi-step.md)
- [\[ICML 2025\] Geometric Contact Flows: Contactomorphisms for Dynamics and Control](geometric_contact_flows_contactomorphisms_for_dynamics_and_control.md)
- [\[NeurIPS 2025\] Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics](../../NeurIPS2025/robotics/massively_parallel_imitation_learning_of_mouse_forelimb_musculoskeletal_reaching.md)
- [\[ICLR 2026\] Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining](../../ICLR2026/robotics/disentangled_robot_learning_via_separate_forward_and_inverse_dynamics_pretrainin.md)
- [\[ICML 2025\] Learning to Stop: Deep Learning for Mean Field Optimal Stopping](learning_to_stop_deep_learning_for_mean_field_optimal_stopping.md)

</div>

<!-- RELATED:END -->
