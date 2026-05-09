---
title: >-
  [Paper Note] Rapid Training of Hamiltonian Graph Networks using Random Features
description: >-
  [ICLR 2026][Optimization][Hamiltonian Graph Networks] This paper proposes RF-HGN, which constructs dense layer parameters via random feature sampling (ELM/SWIM) and solves a linear least-squares problem to train Hamiltonian Graph Networks. By completely bypassing gradient-descent-based iterative optimization, RF-HGN achieves 150–600× speedup on N-body physical systems while maintaining comparable accuracy and strong zero-shot generalization.
tags:
  - ICLR 2026
  - Optimization
  - Hamiltonian Graph Networks
  - Random Features
  - N-body Simulation
  - Zero-shot Generalization
  - Gradient-free Training
date: 2026-05-08
content_hash: bc87771f4c863c5a
---

# Rapid Training of Hamiltonian Graph Networks using Random Features

**Conference**: ICLR 2026
**arXiv**: [2506.06558](https://arxiv.org/abs/2506.06558)
**Code**: [GitLab](https://gitlab.com/fd-research/swimhgn)
**Area**: Physics Simulation / Graph Neural Networks
**Keywords**: Hamiltonian Graph Networks, Random Features, N-body Simulation, Zero-shot Generalization, Gradient-free Training

## TL;DR

This paper proposes RF-HGN, which constructs dense layer parameters via random feature sampling (ELM/SWIM) and solves a linear least-squares problem to train Hamiltonian Graph Networks. By completely bypassing gradient-descent-based iterative optimization, RF-HGN achieves 150–600× speedup on N-body physical systems while maintaining comparable accuracy and strong zero-shot generalization.

## Background & Motivation

**State of the Field**: Data-driven modeling of physical systems is a central challenge. Combining physical priors (e.g., Hamiltonian mechanics) with graph neural networks (GNNs) represents the dominant paradigm. Hamiltonian Graph Networks (HGN) encode the topology of N-body systems via graph structures and, together with Hamiltonian equation constraints, produce accurate, permutation-invariant dynamical predictions.

**Limitations of Prior Work**: Training graph networks is extremely slow. Backpropagation through GNNs involves irregular memory access and load imbalance, and physical models are sensitive to hyperparameters. When the model architecture incorporates a numerical integrator (e.g., Störmer-Verlet), training difficulty is further compounded. Among 15 commonly used optimizers (Adam, LBFGS, etc.), training on 3D lattice systems requires 23–96 seconds, far from meeting the demand for rapid prototyping of large-scale systems.

**Root Cause**: Although the physical inductive biases of GNNs (graph structure + Hamiltonian constraints) improve model quality, these structural constraints render gradient-based iterative optimization more difficult and time-consuming. A fundamental tension exists between accuracy and training efficiency.

**Paper Goals**: (1) How to drastically accelerate HGN training without sacrificing accuracy? (2) How to incorporate random feature methods into graph networks while preserving physical invariances? (3) Can the trained model generalize zero-shot to systems far larger than those seen during training?

**Starting Point**: Random feature methods have recently shown promise for approximating physical systems, yet have not been applied to graph networks. The key observation is that the HGN architecture can be decomposed into two parts—nonlinear dense layers and a linear output layer. If dense layer parameters can be determined by random sampling, training reduces to a convex linear least-squares problem.

**Core Idea**: Construct dense layer parameters of HGN via random feature sampling, transforming non-convex network training into convex linear system solving, thereby enabling gradient-free ultra-fast training.

## Method

### Overall Architecture

The RF-HGN pipeline consists of three stages: (1) **Invariance Encoding**: transforming positions and momenta of the N-body system into translation- and rotation-invariant coordinate representations; (2) **Graph Network Forward Pass**: obtaining graph-level representations via node/edge encoding, message passing, and global pooling; (3) **Random Feature Training**: dense layer parameters are determined by random sampling (ELM or SWIM), and the linear output layer is solved via least squares. The input is a phase-space trajectory $(q, p) \in \mathbb{R}^{2d \cdot N}$, and the output is a scalar Hamiltonian $\hat{\mathcal{H}}$; dynamics are simulated at inference time using the Störmer-Verlet integrator.

### Key Designs

1. **Physical Invariance Encoding**:

    - **Function**: Transforms coordinates in an arbitrary reference frame into a translation- and rotation-invariant representation, ensuring the Hamiltonian is invariant under global translations or rotations of the system.
    - **Mechanism**: Translation invariance is achieved by subtracting the center of mass: $q_i \leftarrow q_i - \frac{1}{N}\sum_{i=1}^{N}q_i$. Rotation invariance is achieved by constructing a local orthonormal basis—the node closest to the center of mass is selected to define the first basis vector $e_1 = q_1/\|q_1\|$, a complete orthogonal matrix $\mathcal{B}$ is built via rotation (2D) or Gram-Schmidt orthogonalization (higher dimensions), and coordinates are projected onto this local frame as $\bar{q}_i = \mathcal{B}^T q_i$. Permutation invariance is naturally guaranteed by the graph structure and the summation aggregation in message passing.
    - **Design Motivation**: The energy of a physical system should not depend on the observer's choice of reference frame. Explicitly encoding invariances reduces redundant information the model must learn, thereby lowering data requirements.

2. **Random Feature Parameter Construction (ELM and SWIM)**:

    - **Function**: Determines the weights and biases of all dense layers (node encoder $\phi_V$, edge encoder $\phi_E$, message constructor $\phi_M$) without gradient descent.
    - **Mechanism**: The **ELM** method (data-independent) samples weights $W$ from a standard normal distribution and biases $b$ from a uniform distribution. The **SWIM** method (data-driven) randomly selects two data points $(x^{(1)}, x^{(2)})$ from the training set and constructs $w_i = s_1(x^{(2)}_i - x^{(1)}_i)\|x^{(2)}_i - x^{(1)}_i\|^{-2}$, $b_i = -\langle w_i, x^{(1)}_i \rangle - s_2$, where $(s_1, s_2)$ are constants related to the activation function. SWIM exploits information from the data distribution, placing hyperplanes precisely in the regions where the data requires discrimination.
    - **Design Motivation**: Converts the non-convex optimization problem into a single linear system solve, completely avoiding gradient vanishing/explosion and non-convex optimization pitfalls.

3. **Linear Layer Least-Squares Solving**:

    - **Function**: Once dense layer parameters are fixed, the linear output layer—the only component requiring optimization—is solved to global optimality via a convex optimization problem.
    - **Mechanism**: A linear system $Z \cdot \theta_L = u$ is constructed, where $Z$ contains the gradients $\nabla\Phi(y)$ of the global pooling layer outputs along with Hamiltonian equation constraints, and $u$ contains time derivative information $J^{-1}\dot{y}$. The system is solved via $l^2$-regularized least squares with time complexity $\mathcal{O}(K d_L^2)$, scaling linearly with the number of data points $M$, particles $N$, and spatial dimension $d$.
    - **Design Motivation**: Convex optimization guarantees a globally optimal solution, and training time scales linearly with problem size, making large-system training feasible.

### Loss & Training

The training objective is to minimize the $l^2$ norm of the Hamiltonian equation residual: $\min_{\theta_L}\|Z\theta_L - u\|^2$. Training data consists of phase-space trajectories and their time derivatives (or pure time-series data). Only a single known Hamiltonian value $\mathcal{H}(y_0)$ is required to fix the integration constant. The training procedure requires no hyperparameter tuning (learning rate, number of epochs, etc.); only the dense layer width and regularization constant need to be specified.

## Key Experimental Results

### Main Results: Optimizer Comparison

| Optimizer | Test MSE | Training Time (s) | Speedup |
|-----------|----------|-------------------|---------|
| **RF-HGN (SWIM)** | **8.95e-5** | **0.16** | **—** |
| LBFGS | 3.56e-5 | 23.85 | 149× |
| Adam | 2.90e-3 | 91.64 | 572× |
| AdamW | 2.91e-3 | 92.15 | 576× |
| Adafactor | 2.41e-3 | 96.36 | 602× |
| SGD | 2.36e-2 | 91.75 | 573× |

RF-HGN achieves a 148–602× speedup over 15 PyTorch optimizers on 3D lattice systems, with accuracy only slightly below the second-order optimizer LBFGS.

### Ablation Study and Generalization Experiments

| Setting | Position MSE (final) | Notes |
|---------|----------------------|-------|
| SWIM RF-HGN, train 3×3, test 100×100 | Low error | Zero-shot generalization successful |
| ELM RF-HGN, train 3×3, test 100×100 | Moderate error | SWIM outperforms ELM by ~one order of magnitude |
| Train 2×2, test 100×100 | High error | 2×2 is an edge case lacking 4-degree nodes |
| RF-HNN (non-graph), train 8, test 8 | Higher error | Graph architecture is 1–2 orders of magnitude more accurate |

| Potential | Adam HGN | ELM RF-HGN | SWIM RF-HGN |
|-----------|----------|------------|-------------|
| Spring $V(r)=\frac{1}{2}\beta r^2$ | 3.88e-3 | 2.33e-3 | **3.41e-5** |
| Anharmonic oscillator | 4.56e-2 | 4.32e-2 | **5.23e-4** |
| Morse potential | 8.89e-2 | **7.40e-4** | 1.22e-3 |

### Key Findings

- **SWIM substantially outperforms ELM**: By leveraging data distribution information to place hyperplanes, SWIM achieves one to two orders of magnitude higher accuracy than ELM in nearly all experiments.
- **Strong zero-shot generalization**: Training on only $2^3=8$ nodes enables accurate prediction of dynamics for systems with $2^{12}=4096$ nodes; models trained on 3×3 lattices generalize to 100×100 lattices.
- **Comparison with NeurIPS 2022 benchmark**: RF-HGN requires only 2–5 seconds of training time, whereas other physics-informed GNNs (FGNN, LGN, etc.) require 400–53,000 seconds.
- **Applicability to complex potentials**: Nonlinear force fields such as anharmonic oscillators and Morse potentials can be reasonably approximated by RF-HGN, still maintaining 200–300× speedups.

## Highlights & Insights

- **Paradigm shift toward gradient-free training**: Transforming neural network training from non-convex iterative optimization to convex linear solving represents a fundamental conceptual shift. For structured physical models, this approach may outperform conventional deep learning training, since physical constraints already delimit the solution space.
- **Elegance of SWIM's data-driven sampling**: SWIM does not sample blindly; instead, it constructs hyperplane parameters from pairs of data points, aligning the "switching regions" of the activation function precisely with the gradients of variation in the data. This sampling strategy encodes prior knowledge—the data distribution—directly into the stochastic process.
- **Practical value of zero-shot generalization**: The ability to train on small systems and deploy on large systems is highly valuable in molecular dynamics simulation, where generating training data for large systems is itself computationally expensive.

## Limitations & Future Work

- **Restricted graph types**: Models trained on chain graphs cannot generalize to lattice graphs (which have different node degrees); zero-shot generalization is limited to graphs of the same structural type.
- **Moderate performance on dynamic edge settings**: In molecular dynamics with edges defined by cutoff distances, relative errors of approximately 10% are observed across all optimizers.
- **Single-layer message passing only**: The current framework employs only one layer of message passing; future work should explore random feature boosting (RF boosting) to support deeper architectures.
- **Suboptimal for small graphs**: For very small systems, fully connected HNN architectures train faster; the overhead of graph structure becomes a liability rather than an asset.

## Related Work & Insights

- **vs. Adam-trained HGN**: RF-HGN trains 100–600× faster, with comparable accuracy on spring systems and moderate accuracy loss on complex potentials that remains within acceptable bounds.
- **vs. RF-HNN (Rahma et al., 2024)**: RF-HNN is limited to small systems and lacks graph structure; extending to a graph architecture in RF-HGN introduces permutation invariance and zero-shot generalization capability, improving accuracy by 1–2 orders of magnitude.
- **vs. Echo State Graph Networks**: A similar random-weight philosophy, but RF-HGN is specifically designed for physical systems, integrating Hamiltonian constraints and physical invariances.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of random feature methods to physics-informed graph networks; the paradigm shift is meaningful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparisons across 15 optimizers, multiple potential functions, zero-shot generalization, and reproduction of the NeurIPS benchmark—comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete theoretical derivations, and high-quality figures.
- Value: ⭐⭐⭐⭐ Highly valuable to the physics simulation community, though applicability is constrained by specific graph network architecture types.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](../../NeurIPS2025/optimization/training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)
- [\[ICLR 2026\] SCRAPL: Scattering Transform with Random Paths for Machine Learning](scrapl_scattering_transform_with_random_paths_for_machine_learning.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](deepafl_deep_analytic_federated_learning.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)

<!-- RELATED:END -->
