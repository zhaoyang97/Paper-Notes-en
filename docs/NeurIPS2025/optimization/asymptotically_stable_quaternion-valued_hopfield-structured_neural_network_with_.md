---
title: >-
  [Paper Note] Asymptotically Stable Quaternionic Hopfield Structured Neural Network with Supervised Projection-based Manifold Learning
description: >-
  [NeurIPS 2025][Optimization][quaternion neural network] This paper proposes a Quaternion-valued Supervised Hopfield-structured Neural Network (QSHNN) that employs a periodic projection strategy to maintain the quaternion…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "quaternion neural network"
  - "Hopfield network"
  - "asymptotic stability"
  - "manifold learning"
  - "robotic path planning"
date: 2026-05-08
content_hash: 0681dfb74a0909a8
---

# Asymptotically Stable Quaternionic Hopfield Structured Neural Network with Supervised Projection-based Manifold Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.16607](https://arxiv.org/abs/2510.16607)
**Code**: None
**Area**: Neural Networks / Robotic Control / Optimization
**Keywords**: quaternion neural network, Hopfield network, asymptotic stability, manifold learning, robotic path planning

## TL;DR
This paper proposes a Quaternion-valued Supervised Hopfield-structured Neural Network (QSHNN) that employs a periodic projection strategy to maintain the quaternionic structural consistency of the weight matrix. The existence and uniqueness of fixed points and their asymptotic stability are established via Lyapunov theory, while bounded trajectory curvature guarantees path smoothness for robotic path planning.

## Background & Motivation
**Background**: Hopfield Neural Networks (HNNs) are classical attractor models with symmetric topology and recurrent connections, whose states converge to discrete equilibria. The quaternion algebra $\mathbb{H}$ naturally encodes 3D rotations, offering inherent advantages for parameterizing robotic joint orientations.

**Limitations of Prior Work**:
   - Existing quaternionic Hopfield networks operate primarily in discrete-time or unsupervised paradigms, relying on Hebbian or outer-product formulas for direct weight encoding. These approaches suffer from limited storage capacity (spurious attractors emerge after a small number of target states) and do not support dynamic reconfiguration.
   - Direct encoding lacks an error-driven optimization mechanism, preventing task-adaptive behavioral adjustment.
   - Modern continuous-time HNNs (e.g., Ramsauer et al. 2021) remain confined to energy-minimization-driven unsupervised paradigms, lacking explicit target tracking and structural control.

**Key Challenge**: When applying supervised learning in the quaternionic domain, standard gradient descent disrupts the quaternion block structure of the weight matrix (each $4\times4$ block should correspond to a quaternion left-multiplication matrix), causing the network to lose the geometric advantages of quaternion algebra.

**Core Idea**: A periodic projection learning strategy is designed: during standard gradient descent, every $\mathcal{P}$ steps, each $4\times4$ block of the weight matrix is projected onto the nearest quaternion structure in the least-squares sense, while maintaining both convergence and quaternionic consistency.

## Method

### Overall Architecture
Input: target quaternion state $\boldsymbol{d} \in \mathbb{H}^N$ → QSHNN dynamical system evolution (continuous-time ODE) → convergence to equilibrium $\boldsymbol{q}^*$ → error-driven gradient descent with periodic projection weight updates → Output: trained weight matrix $W$ and smooth trajectory.

Evolution equation: $\dot{\boldsymbol{q}} = -\gamma\boldsymbol{q} + \mu\boldsymbol{W}\circ\boldsymbol{\varphi}(\boldsymbol{q}) + \mu\boldsymbol{b}$

where $\boldsymbol{W}$ is the quaternionic weight matrix, $\boldsymbol{\varphi}$ is the component-wise tanh activation, and $\gamma, \mu$ are network parameters.

### Key Designs

1. **Quaternionic Neuron Structure**

    - Function: Integrates 4 real-valued neurons into a single quaternionic neuron, whose internal connections are fully characterized by a single quaternion weight $\boldsymbol{\omega}$.
    - Mechanism: Via the quaternion left-multiplication matrix representation (Eq. 2.2), quaternion multiplication is equivalent to matrix-vector multiplication in $\mathbb{R}^4$. All left-multiplication matrices form a 4-dimensional embedded submanifold $\mathcal{L}$ of $\mathbb{R}^{4\times4}$, constituting a real matrix Lie group.
    - Design Motivation: Preserves the quaternion algebraic structure while maintaining compatibility with standard numerical methods (Runge-Kutta ODE solvers).

2. **GHR Calculus-Driven Supervised Learning Rule**

    - Function: Derives exact gradient descent update formulas in the quaternionic domain.
    - Mechanism: Due to the non-commutativity of quaternion algebra, standard differential calculus fails. The Generalized $\mathbb{HR}$ (GHR) calculus framework is adopted; partial derivatives with respect to an orthonormal basis $\{1, \boldsymbol{i}^\mu, \boldsymbol{j}^\mu, \boldsymbol{k}^\mu\}$ obtained via quaternionic rotation transformations yield a weight update rule involving the sensitivity matrix $S = \mathbb{I}_{4n} - \frac{\mu}{\gamma}W \cdot J_\varphi(\boldsymbol{q})$.
    - Design Motivation: Unlike Hebbian direct encoding, error-driven learning enables dynamic adaptation of network behavior to task objectives.

3. **Periodic Projection Strategy**

    - Function: Every $\mathcal{P}=10$ gradient descent steps, each $4\times4$ block of the weight matrix is projected onto the nearest quaternion structure.
    - Mechanism: The projection formula $\widetilde{W} = c_1 L(1) + c_i L(\boldsymbol{i}) + c_j L(\boldsymbol{j}) + c_k L(\boldsymbol{k})$ is applied, where $c_1, c_i, c_j, c_k$ are determined via least squares. Exploiting the linear structure of manifold $\mathcal{L}$, the projection admits a closed-form solution.
    - Design Motivation: Pure gradient descent causes weights to drift off the quaternion submanifold (as visualized in heatmaps showing loss of block structure), while explicit constraint enforcement (e.g., Lagrange multipliers) is computationally expensive. Periodic projection balances efficiency and structural preservation.

4. **Asymptotic Stability Proof**

    - Function: Establishes the existence, uniqueness, and global asymptotic stability of QSHNN fixed points via Lyapunov theory.
    - Mechanism: A Lyapunov energy function $V(\boldsymbol{q})$ is constructed and shown to have a strictly negative time derivative ($\dot{V} < 0$), guaranteeing convergence from any initial state to the unique equilibrium. Trajectory curvature is additionally shown to be bounded, ensuring path smoothness.
    - Design Motivation: Essential for robotic control — provably convergent and smooth trajectories prevent abrupt joint motion or oscillation.

### Loss & Training
- Loss: $\ell = \sum_n |\boldsymbol{q}^*_n - \boldsymbol{d}_n|^2$ (MSE between equilibrium and target)
- Learning rate $\eta = 0.001 \sim 0.2$ (adaptively adjusted), projection period $\mathcal{P}=10$
- Maximum 30000 training epochs, convergence threshold $\tau = 10^{-6}$
- ODE solver: Runge-Kutta numerical method

## Key Experimental Results

### Main Results
Experiments are conducted on a network of 4 quaternionic neurons (16 real-valued neurons), with targets randomly generated as $d_i \sim \mathcal{U}(-1, 1)$.

| Metric | SHNN (no projection) | QSHNN (with projection) |
|--------|----------------------|-------------------------|
| Convergence accuracy | High ($< 10^{-6}$) | High ($< 10^{-6}$) |
| Quaternion structure preservation | ✗ No block structure | ✓ Quaternion-symmetric blocks |
| Trajectory smoothness | Not guaranteed | Bounded curvature |
| Max iterations | 10000 | 30000 |

### Ablation Study

| Configuration | Outcome | Notes |
|---------------|---------|-------|
| SHNN (pure gradient descent) | Fast convergence but destroys quaternion structure | Heatmap shows no discernible block structure |
| QSHNN (periodic projection) | Convergence + structure preservation | Heatmap shows clear $4\times4$ blocks |
| Equal-component targets $q_s^i=q_x^i=q_y^i=q_z^i$ | Weights concentrate on main diagonal | Model adapts to target symmetry |

### Key Findings
- Projection-induced training curves exhibit more pronounced fluctuations (projection interrupts continuous gradient descent), yet final convergence accuracy is comparable.
- Weight matrix heatmaps visually demonstrate: without projection, blocks lack internal structure; with projection, quaternion-symmetric block patterns emerge clearly.
- Preliminary robotic simulation (PyBullet) confirms that QSHNN can drive a 4-DOF manipulator to smoothly converge from arbitrary initial joint configurations to a target end-effector orientation.

## Highlights & Insights
- **Theoretical Completeness**: The paper establishes a complete theoretical chain from existence and uniqueness → asymptotic stability → bounded curvature, which is rarely achieved in quaternionic neural network literature and provides mathematical guarantees for practical deployment.
- **Elegance of Periodic Projection**: Exploiting the linear structure of the quaternion left-multiplication manifold $\mathcal{L}$, the projection admits a closed-form solution at low implementation cost. The strategy of periodically projecting onto a structured subspace during training has potential for generalization to other algebraically constrained networks.
- **GHR Calculus Framework**: Provides a systematic tool for gradient descent on non-commutative algebras, extendable beyond quaternions to octonions and other hypercomplex number systems.

## Limitations & Future Work
- Experiments are conducted at extremely small scale (only 4 quaternionic neurons); scalability and performance on larger networks remain undemonstrated.
- The robotic application is limited to a preliminary simulation prototype (PyBullet), without quantitative comparison against established baselines (RRT, PRM, etc.).
- Target states are randomly generated; validation on real-world tasks (e.g., specific grasp pose sequences) is absent.
- The choice of projection period $\mathcal{P}$ lacks theoretical justification and is fixed at 10 throughout.

## Related Work & Insights
- **vs. Classical QHNN**: Quaternionic Hopfield networks by Isokawa, Kobayashi, and others adopt discrete-time Hebbian encoding with limited capacity and no trainability; QSHNN introduces a continuous-time supervised learning paradigm.
- **vs. Modern HNN (Ramsauer 2021)**: Modern Hopfield layers serve as associative memory in Transformers but remain unsupervised/energy-minimization-driven; QSHNN is goal-directed.
- **vs. Quaternion Supervised Neural Networks (QSNN)**: Standard QSNNs perform static input-output mappings without continuous-time stability guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of quaternion algebra, Hopfield networks, supervised learning, and periodic projection is genuinely novel, with rigorous theoretical derivations.
- Experimental Thoroughness: ⭐⭐ Experiments are conducted at too small a scale (4 neurons), with no quantitative comparison against baselines and only a prototype robotic application.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, though the prose tends toward verbosity.
- Value: ⭐⭐⭐ The theoretical foundation is solid, but practical value requires validation at larger scales.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Faster Algorithms for Structured John Ellipsoid Computation](faster_algorithm_for_structured_john_ellipsoid_computation.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](constrained_network_slice_assignment_via_llms.md)
- [\[NeurIPS 2025\] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes](stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)
- [\[NeurIPS 2025\] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [\[NeurIPS 2025\] Learning to Insert for Constructive Neural Vehicle Routing Solver](learning_to_insert_for_constructive_neural_vehicle_routing_solver.md)

</div>

<!-- RELATED:END -->
