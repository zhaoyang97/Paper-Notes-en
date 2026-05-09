---
title: >-
  [Paper Note] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs
description: >-
  [ICLR 2026][Scientific Computing][Neural PDE solvers] Grounded in Green's function theory, DGNet embeds the superposition principle into a physics-neural hybrid architecture, achieving state-of-the-art accuracy with only tens of training trajectories and demonstrating robust zero-shot generalization to unseen source terms.
tags:
  - ICLR 2026
  - Scientific Computing
  - Neural PDE solvers
  - Green's function
  - graph neural networks
  - data-efficient learning
  - spatiotemporal PDEs
date: 2026-05-08
content_hash: c4e1ffd4c6859db8
---

# DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs

**Conference**: ICLR 2026
**arXiv**: [2603.01762](https://arxiv.org/abs/2603.01762)
**Code**: [Available](https://github.com/tanyingjie01/DGNet)
**Area**: Scientific Computing
**Keywords**: Neural PDE solvers, Green's function, graph neural networks, data-efficient learning, spatiotemporal PDEs

## TL;DR

Grounded in Green's function theory, DGNet embeds the superposition principle into a physics-neural hybrid architecture, achieving state-of-the-art accuracy with only tens of training trajectories and demonstrating robust zero-shot generalization to unseen source terms.

## Background & Motivation

Spatiotemporal PDEs underpin modeling in fluid dynamics, weather forecasting, molecular dynamics, and beyond. Traditional numerical solvers are computationally prohibitive, motivating growing interest in neural PDE solvers as surrogates. However, existing approaches share a critical bottleneck: **poor data efficiency**—they typically require large numbers of training trajectories, yet high-fidelity PDE data are expensive to obtain.

The authors identify the root cause as the failure of existing neural architectures to explicitly encode the strong structural inductive biases inherent in PDE dynamics—locality, conservation laws, and the superposition principle—forcing models to relearn fundamental physical structure from data. This problem is especially acute in the presence of **source terms** $f(x,t)$: training trajectories cover only a limited range of source patterns, making generalization to unseen source configurations difficult.

The paper revisits Green's function theory as a source of structural inductive bias. The Green's function decomposes the PDE solution into a homogeneous evolution component and a forced response component, thereby encoding the superposition principle and reducing the amount of fundamental behavior that must be learned from data.

## Method

### Overall Architecture

DGNet converts the Green's function into a graph-based discrete form and embeds the superposition principle into a physics-neural hybrid architecture. The pipeline proceeds as follows:

1. **Spatial discretization**: The spatial domain $\Omega$ is discretized into $N$ nodes, with a graph $G=(V,E)$ constructed via Delaunay triangulation.
2. **Temporal integration**: An implicit Crank–Nicolson (midpoint) scheme is adopted.
3. **Discrete Green's function**: The propagation operator is derived as $\mathbf{G}(\Delta t) = (\mathbf{I} - \frac{\Delta t}{2}\mathbf{L})^{-1}$.
4. **Hybrid operator**: $\mathbf{L} = \mathbf{L}_{\text{physics}} + \mathbf{L}_{\text{neural}}$, combining physical priors with GNN-based corrections.
5. **Residual GNN**: Captures additional dynamics not explicitly covered by the physics operator.

### Key Designs

**1. Green's Function Representation and the Superposition Principle**

The Green's function $G(x,t;x',\tau)$ describes the influence observed at $(x,t)$ due to a unit point source applied at $(x',\tau)$. By the superposition principle, the complete PDE solution decomposes into:

- **Initial-state propagation term**: $\int G(x,t;x',0)\cdot u_0(x')\,dx'$ — describes the propagation of initial conditions.
- **Source response term**: $\int\!\!\int G(x,t;x',\tau)\cdot f(x',\tau)\,dx'\,d\tau$ — describes the accumulated response to source terms.

This decomposition naturally decouples system evolution from source response, providing a principled basis for zero-shot generalization to unseen source terms.

**2. Discrete Green's Formula**

After Crank–Nicolson discretization on the graph, the single-step update rule reads:

$$\mathbf{u}^{k+1} = \mathbf{G}(\Delta t)\!\left(\mathbf{I} + \frac{\Delta t}{2}\mathbf{L}\right)\mathbf{u}^k + \mathbf{G}(\Delta t)\frac{\Delta t}{2}\!\left(\mathbf{f}^k + \mathbf{f}^{k+1}\right)$$

where $\mathbf{G}(\Delta t) = \left(\mathbf{I} - \frac{\Delta t}{2}\mathbf{L}\right)^{-1}$ is the discrete Green's function. This discrete form fully preserves the superposition structure of the continuous setting: the next-step state is composed of the propagated current state and the accumulated source response over the time step.

To address the computational cost of matrix inversion in large-scale systems, a *factorize-once, solve-many* strategy is employed: the coefficient matrix $(\mathbf{I} - \frac{\Delta t}{2}\mathbf{L})$ depends only on the static mesh geometry and can be pre-factorized via sparse LU decomposition and cached for reuse.

**3. Physics-Neural Hybrid Operator**

$$\mathbf{L} = \mathbf{L}_{\text{physics}} + \mathbf{L}_{\text{neural}}$$

**$\mathbf{L}_{\text{physics}}$ (physics prior operator)**:
- **Gradient operator**: Constructed via the Green–Gauss theorem using control-volume areas, normal vector projections, and face lengths.
- **Laplacian operator**: Based on the discrete Laplace–Beltrami operator with cotangent weights $w_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$ and Voronoi areas.

Both fundamental spatial operators are built directly from mesh geometry, ensuring consistency with physical laws.

**$\mathbf{L}_{\text{neural}}$ (neural correction operator)**:
A GNN following the Encode–Process–Decode paradigm learns:
- **Encoder**: Node features (spatial coordinates and node type) and edge features (relative displacement and distance) are encoded into embedding vectors separately.
- **Processor**: $M$ layers of message-passing neural networks (MPNNs) aggregate information via Message and Update functions with residual connections.
- **Decoder**: Terminal node embeddings are concatenated and passed through an MLP to predict edge-level correction values.

### Loss & Training

- **Subsequence training**: Full trajectories are split into short subsequences of length $Q \ll T$, with rollout starting from $u^{s_0}$.
- **Pushforward trick**: Small noise is injected into initial states during training to mitigate error accumulation.
- **Loss function**: $L_2$ loss is computed only on the first and last frames of each subsequence: $\mathcal{L} = \|\hat{\mathbf{u}}^{s_1} - \mathbf{u}^{s_1}\|^2 + \|\hat{\mathbf{u}}^{s_{Q-1}} - \mathbf{u}^{s_{Q-1}}\|^2$
- **Optimizer**: Adam with learning rate decay.

## Key Experimental Results

### Main Results (Three PDE Systems, Tens of Training Trajectories)

**Classical PDEs** (Allen–Cahn / Fisher–KPP / FitzHugh–Nagumo):

| Scenario | Metric | DeepONet | MGN | MP-PDE | BENO | PhyMPGN | **DGNet** |
|---|---|---|---|---|---|---|---|
| Allen–Cahn | MSE | 2.60e-1 | 2.70e-1 | 8.52e-1 | 2.52e+0 | 5.16e-1 | **8.75e-3** |
| Allen–Cahn | RNE | 0.669 | 0.681 | 1.211 | 2.081 | 0.942 | **0.019** |
| Fisher–KPP | MSE | 3.05e-2 | 3.66e-3 | 9.90e-2 | 6.26e-2 | 1.50e-2 | **2.59e-4** |
| FitzHugh–Nagumo | MSE | 2.49e-6 | 3.75e-5 | 6.46e-6 | 2.14e-4 | 1.69e-3 | **1.18e-7** |

**Complex Geometry** (Contaminant transport in channel flow with varying obstacles):

| Scenario | Metric | DeepONet | MGN | MP-PDE | BENO | PhyMPGN | **DGNet** |
|---|---|---|---|---|---|---|---|
| Cylinder | MSE | 4.44e-2 | 6.38e-3 | 9.31e-2 | 6.76e-2 | 4.13e-1 | **1.00e-4** |
| Sediments | MSE | 3.61e-2 | 5.94e-3 | 7.10e-3 | 1.07e-1 | 2.00e-1 | **4.60e-4** |
| Complex Obstacles | MSE | 5.33e-2 | 7.79e-3 | 6.09e-3 | 7.66e-2 | 2.97e-1 | **6.69e-5** |

**Zero-Shot Generalization (Laser Heating, Unseen Source Terms)**:

| Scenario | Metric | DeepONet | MGN | MP-PDE | BENO | PhyMPGN | **DGNet** |
|---|---|---|---|---|---|---|---|
| Laser Heat | MSE | 2.48e+3 | 4.98e+3 | 3.88e+3 | 1.95e+3 | 6.78e+3 | **1.76e+1** |
| Laser Heat | RNE | 0.121 | 0.171 | 0.151 | 0.107 | 0.200 | **0.010** |

### Ablation Study

Four variants are compared on the Complex Obstacles scenario:
- **(A) w/o $\mathbf{L}_{\text{physics}}$**: Removing the physics prior operator leads to severe performance degradation, demonstrating that the physical prior provides critical structural knowledge.
- **(B) w/o $\mathbf{L}_{\text{neural}}$**: Removing the neural correction causes a performance drop, though smaller than (A), indicating that the correction primarily fine-tunes discretization errors.
- **(C) w/o Residual GNN**: Removing the residual GNN degrades performance, confirming its effectiveness in capturing additional dynamics.
- **(D) w/o Green**: Replacing the framework with a generic end-to-end GNN yields **the largest performance drop**, highlighting the discrete Green's solver as the most critical structural prior.

### Key Findings

1. DGNet achieves **1–2 orders of magnitude** lower MSE than baselines across all scenarios.
2. In the FitzHugh–Nagumo system, only DGNet successfully reproduces the long-term propagation of nonlinear spiral waves.
3. In the laser heating zero-shot generalization test, baseline errors increase by several orders of magnitude, whereas DGNet exhibits virtually no performance degradation.
4. The discrete Green's solver is the single most critical component, as evidenced by the largest performance drop upon its removal in the ablation study.

## Highlights & Insights

1. **Theory-driven inductive bias**: Translating the Green's function—a classical tool of PDE theory—into a computable discrete form on graphs exemplifies the paradigm of "physical knowledge → network design."
2. **Explicit encoding of the superposition principle**: Decomposing the solution into initial-state propagation and source response naturally enables generalization to unseen source terms without relying on data coverage.
3. **Practical optimization via sparse computation**: The *factorize-once, solve-many* strategy makes the discrete Green's solver tractable on large meshes.
4. **Complementary hybrid operator design**: The physics prior provides reliable low-level structure, while the neural correction compensates for discretization errors; the two components serve clearly distinct roles.

## Limitations & Future Work

1. The framework is applicable only to linear or weakly nonlinear PDEs; principled extensions are required for **quasi-linear PDEs** where the superposition principle does not hold.
2. Validation is currently limited to 2D domains; scaling to **large-scale 3D systems** poses significant engineering challenges.
3. The physics prior operator is built around gradient and Laplacian operators; more complex operator forms require additional design effort.
4. Hyperparameters such as the subsequence length $Q$ and the noise injection magnitude require manual tuning.

## Related Work & Insights

- **PINNs**: Learn solutions to specific instances rather than operator mappings, precluding generalization across domains, parameters, or source terms.
- **Neural operators (FNO/DeepONet)**: Learn mappings in function space but require large amounts of data and lack physical structure.
- **Graph-based PDE solvers (GNS/PhyMPGN/BENO)**: Incorporate partial physical biases but lack a principled theoretical foundation.
- **BENO**: Also draws inspiration from Green's functions but targets only time-independent elliptic PDEs.
- **Insight**: Classical PDE theory contains a rich body of structural priors—Fourier methods, variational principles, and others—that remain underutilized as guides for neural architecture design.

## Rating

- **Novelty**: ★★★★☆ — The combination of Green's function discretization and a hybrid operator is innovative.
- **Technical Depth**: ★★★★★ — The derivation from continuous theory to discrete implementation and the ablation design are both rigorous.
- **Experimental Rigor**: ★★★★★ — Covers three PDE system classes, multiple baselines, and a comprehensive ablation study.
- **Practical Value**: ★★★★☆ — Open-source code is available and directly applicable in scientific computing.
- **Clarity**: ★★★★☆ — Derivations are clear and figures are informative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](../../NeurIPS2025/scientific_computing/neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](../../AAAI2026/scientific_computing/pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)
- [\[ICLR 2026\] Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](astral_training_physics-informed_neural_networks_with_error_majorants.md)

</div>

<!-- RELATED:END -->
