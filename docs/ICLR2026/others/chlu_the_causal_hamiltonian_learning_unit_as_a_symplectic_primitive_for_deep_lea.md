---
title: >-
  [Paper Note] CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning
description: >-
  [ICLR 2026][Hamiltonian Neural Networks] CHLU is a computational learning primitive grounded in relativistic Hamiltonian mechanics and symplectic integration. By enforcing phase-space volume conservation and introducing a causal velocity upper bound, it addresses gradient explosion/vanishing in LSTMs and information dissipation in Neural ODEs, achieving infinite-horizon stability and thermodynamic generative capability.
tags:
  - ICLR 2026
  - Hamiltonian Neural Networks
  - Symplectic Integration
  - Relativistic Kinetic Energy
  - Energy Conservation
  - Long-Term Stability
date: 2026-05-08
content_hash: 5ac10bd1ff14d9f1
---

# CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning

**Conference**: ICLR 2026
**arXiv**: [2603.01768](https://arxiv.org/abs/2603.01768)
**Code**: Available (software package)
**Area**: Physics-Inspired Deep Learning / Sequential Modeling
**Keywords**: Hamiltonian Neural Networks, Symplectic Integration, Relativistic Kinetic Energy, Energy Conservation, Long-Term Stability

## TL;DR
CHLU is a computational learning primitive grounded in relativistic Hamiltonian mechanics and symplectic integration. By enforcing phase-space volume conservation and introducing a causal velocity upper bound, it addresses gradient explosion/vanishing in LSTMs and information dissipation in Neural ODEs, achieving infinite-horizon stability and thermodynamic generative capability.

## Background & Motivation
**State of the Field**: Sequential modeling in deep learning faces a fundamental dichotomy — discrete units (LSTM/RNN) are expressive but unstable (gradient explosion/vanishing), while continuous models (Neural ODEs) are smooth but dissipatively destroy information.

**Limitations of Prior Work**:
- LSTMs accumulate numerical errors in forward passes during long-range reasoning, causing trajectory divergence.
- Neural ODEs, due to their dissipative nature, cause trajectories to contract toward the origin, losing long-term information.
- Hamiltonian Neural Networks (Greydanus 2019) are primarily designed for physics simulation and have not been extended to general reasoning and generative tasks.

**Root Cause**: Long-term memory (requiring information preservation) and stability (requiring constrained updates) are fundamentally at odds — conservation laws can resolve both simultaneously.

**Paper Goals**: Design a computational unit that treats energy conservation as a structural prior rather than a learning objective.

**Starting Point**: Use relativistic mechanics to provide a velocity upper bound preventing kinetic energy explosion, and symplectic integration to guarantee long-term energy conservation.

**Core Idea**: Reframe information propagation as Hamiltonian evolution of internal states over a learnable potential energy surface.

## Method

### Overall Architecture
The internal state of CHLU, $\mathbf{z} = (\mathbf{q}, \mathbf{p})$, represents generalized coordinates and momenta, evolving according to Hamilton's equations $\dot{q} = \partial\mathcal{H}/\partial p$, $\dot{p} = -\partial\mathcal{H}/\partial q$. The Hamiltonian consists of three components: relativistic kinetic energy + learnable potential energy + constraint potential energy.

### Key Designs

1. **Relativistic Kinetic Governor**:

    - Function: Replaces Newtonian kinetic energy with a relativistic formulation, bounding velocity from above.
    - Mechanism: $T(\mathbf{p}) = \sqrt{c^2 \mathbf{p}^T \mathbf{M}^{-1} \mathbf{p} + m_0^2 c^4}$; as momentum grows, the velocity $\dot{\mathbf{q}} = \nabla_p T$ saturates toward the speed limit $c$.
    - Design Motivation: Newtonian kinetic energy permits unbounded velocities, leading to kinetic energy explosion and gradient explosion. Relativistic kinetic energy provides a natural velocity ceiling, preventing catastrophic divergence from initialization perturbations.

2. **Symplectic Integration (Velocity Verlet)**:

    - Function: Directly embeds a dissipative Velocity Verlet integrator into the forward pass.
    - Mechanism: $\mathbf{p}_{t+0.5} = \mathbf{p}_t - \frac{\epsilon}{2}\nabla V_\theta(\mathbf{q}_t)$ → $\mathbf{q}_{t+1} = \mathbf{q}_t + \epsilon \nabla T(\mathbf{p}_{t+0.5})$ → $\mathbf{p}_{t+1} = (1-\gamma)\mathbf{p}^*_{t+1}$
    - Design Motivation: Symplectic integrators strictly preserve phase-space volume (Liouville's theorem), ensuring energy oscillates boundedly over infinite horizons rather than monotonically growing or decaying. The friction coefficient $\gamma$ is tunable: $\gamma=0$ yields a conservative system (inference), while $\gamma>0$ yields a dissipative system (convergence to attractor).

3. **Hamiltonian Contrastive Divergence Training (Wake-Sleep)**:

    - **Wake Phase**: Supervised; minimizes MSE between predictions and targets plus Lyapunov exponent regularization.
    - **Sleep Phase**: Unsupervised; freely evolves from a replay buffer, raising the energy of hallucinated states.
    - Weight update: $\Delta\theta \propto -\nabla_\theta \mathcal{H}(z_{\text{wake}}) + \nabla_\theta \mathcal{H}(z_{\text{sleep}})$

4. **Langevin Dynamics Generation**:

    - Function: Converts deterministic inference into generative modeling.
    - Mechanism: Stochastic noise is injected into the momentum update: $d\mathbf{p} = -\nabla V_\theta dt - \gamma\mathbf{p}dt + \sqrt{2\gamma k_B \mathcal{T}} d\mathbf{W}$
    - Temperature annealing $\mathcal{T} \to 0$ drives the system into low-energy modes of the potential surface, "crystallizing" data samples.

### Loss & Training
- Wake loss: MSE + $\lambda \mathcal{L}_{reg}$ (Lyapunov exponent regularization to suppress chaos)
- Sleep loss: Contrastive signal — lowers energy of data states, raises energy of hallucinated states
- Hyperparameters: learnable diagonal mass matrix $\mathbf{M}$, velocity limit $c$, rest mass $m_0$, constraint potential coefficient $\alpha$

## Key Experimental Results

### Experiment I — Long-Range Stability (Lemniscate Trajectory Tracking, trained for 3 cycles, inferred for 50 cycles)

| Model | Performance |
|-------|-------------|
| LSTM | Numerical errors accumulate in the forward pass; trajectory diverges to a high-energy limit cycle. |
| Neural ODE | Trajectory spirals inward and collapses to the origin (dissipation). |
| **CHLU** | **Trajectory remains closed and stable**; error is bounded and non-accumulating. |

### Experiment II — Kinetic Safety (Perturbed Sinusoidal Wave Prediction)

| Model | Response to Initial Perturbation |
|-------|----------------------------------|
| LSTM | Produces non-physical instantaneous velocity spikes (unbounded acceleration). |
| Neural ODE | Waveform collapses entirely to zero (trivial solution). |
| **CHLU** | **Velocity smoothly saturates at $c$**; perturbation manifests as a phase shift rather than amplitude divergence. |

### Experiment III — MNIST Thermodynamic Generation

- Starting from test-set centroids with added Gaussian noise, recognizable handwritten digit patterns are generated via Langevin dynamics annealing.
- Legible digit patterns are successfully produced, though certain digits (3, 5, 8, 9) appear with disproportionately high frequency.

### Key Findings
- **Symplectic constraints are necessary for long-term topological fidelity**: Neither LSTM nor NODE can maintain trajectory topology beyond 50 cycles.
- **The causal velocity bound is a robust defense against initialization instability**: Perturbations are converted into phase shifts rather than amplitude explosions.
- **Generative capability is a byproduct**: The potential energy surface naturally defines attraction basins of the data manifold; sampling is achieved through temperature annealing.

## Highlights & Insights
- **Conservation laws as structural priors rather than learning objectives**: A fundamentally distinct design philosophy — conservation is *guaranteed* by the architecture rather than *learned* by the network.
- **An elegant analogy from physics to computation**: Data points = potential energy minima; inference = Hamiltonian evolution; generation = Langevin annealing — the physical intuition underlying the entire framework is remarkably coherent.
- **The velocity limit $c$ as a tunable hyperparameter**: Controls the upper bound on information propagation speed, analogous to the attention window in Transformers but with a physical grounding.

## Limitations & Future Work
- The work remains at a proof-of-concept stage (MNIST) and has not been validated on large-scale or high-dimensional data.
- None of the three experiments involve hyperparameter tuning or comparison against state-of-the-art baselines.
- The Wake-Sleep training mechanism is an empirical design choice, lacking theoretical convergence guarantees.
- The capacity of a single CHLU unit is limited; architectural designs for multi-CHLU networks are left for future work.
- Whether the expressiveness of the learnable potential $V_\theta$ is constrained by the symplectic structure remains unaddressed.

## Related Work & Insights
- **vs. LSTM**: LSTM's gating mechanism learns when to forget or retain information; CHLU structurally guarantees information preservation via symplectic geometry.
- **vs. Neural ODE**: Neural ODEs model dissipative systems, while CHLU models conservative systems — the former is suited for convergence to steady states, the latter for trajectory preservation.
- **vs. HNN (Greydanus 2019)**: HNNs learn Hamiltonians for simulation purposes; CHLU uses a fixed Hamiltonian structure as a primitive for both inference and generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of relativistic Hamiltonians, symplectic integration, and thermodynamic generation constitutes an entirely novel deep learning primitive.
- Experimental Thoroughness: ⭐⭐⭐ Proof-of-concept only; three simple experiments with no comparison to SOTA.
- Writing Quality: ⭐⭐⭐⭐ Physical intuition is clearly articulated, though some derivation details are deferred to the appendix.
- Value: ⭐⭐⭐⭐ Offers a unique physics-inspired perspective and a principled solution to long-range memory and stability problems.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](../../AAAI2026/others/how_to_marginalize_in_causal_structure_learning.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[ICLR 2026\] cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning](cadrille_multi-modal_cad_reconstruction_with_reinforcement_learning.md)
- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)

<!-- RELATED:END -->
