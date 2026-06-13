---
title: >-
  [Paper Note] Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics
description: >-
  [ICLR 2026][Graph Learning][Graph Neural Simulators] This paper proposes Information-preserving Graph Neural Simulators (IGNS), which leverage port-Hamiltonian dynamical structure to prevent information dissipation on gr…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Graph Neural Simulators"
  - "Hamiltonian Dynamics"
  - "Long-Range Interactions"
  - "Port-Hamiltonian"
  - "Multi-Step Training"
date: 2026-05-08
content_hash: 34dc827d9307a8c1
---

# Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics

**Conference**: ICLR 2026
**arXiv**: [2511.08185](https://arxiv.org/abs/2511.08185)  
**Code**: [thobotics/neural_pde_matching](https://thobotics.github.io/neural_pde_matching)  
**Area**: 3D Vision
**Keywords**: Graph Neural Simulators, Hamiltonian Dynamics, Long-Range Interactions, Port-Hamiltonian, Multi-Step Training

## TL;DR
This paper proposes Information-preserving Graph Neural Simulators (IGNS), which leverage port-Hamiltonian dynamical structure to prevent information dissipation on graphs. Combined with warmup initialization, geometric encoding, and multi-step training objectives, IGNS consistently outperforms existing graph neural simulators across 6 physics simulation benchmarks.

## Background & Motivation
Physics system simulation is a core task in scientific computing. Traditional numerical solvers (e.g., finite element methods) incur prohibitive computational costs at high accuracy requirements. Graph Neural Simulators (GNS) can accelerate simulation by orders of magnitude by learning dynamics on graph-structured data. However, existing GNS face two fundamental challenges: (1) **Difficulty modeling long-range interactions** — message passing loses information between distant nodes due to over-smoothing and over-squashing after multiple layers; (2) **Error accumulation** — errors grow rapidly during autoregressive rollout after single-step training. Existing implicit/explicit denoising objectives can only mitigate local noise and fail to capture low-frequency drift that emerges over multiple steps. The core idea is to exploit the information-preserving property of Hamiltonian dynamics so that information is not dissipated on the graph, enabling effective long-range propagation and stable multi-step rollout.

## Method

### Overall Architecture
IGNS takes initial node states $\bar{\mathbf{X}}$ as input, first executing $l$ steps of a **warmup phase** (which propagates global context via message passing without advancing time) to produce enhanced initial states $\mathbf{X}(0) = \bar{\mathbf{X}}^{(l)}$. The system then enters the **simulation phase**, where it evolves according to port-Hamiltonian dynamics, with a multi-step loss $\mathcal{L}_{\text{multi-step}}$ supervising all intermediate predictions.

### Key Designs

1. **Port-Hamiltonian Formulation**: IGNS parameterizes the graph dynamics as a port-Hamiltonian system:
    $\dot{\mathbf{x}}_i = \mathbf{J} \nabla_{\mathbf{x}_i} H_\theta(t, \mathbf{X}) - \begin{bmatrix} 0 \\ \mathbf{D}_\theta \nabla_{\mathbf{p}_i} H_\theta \end{bmatrix} + \begin{bmatrix} 0 \\ \mathbf{r}_\theta(t, \mathbf{X}) \end{bmatrix}$
   where $\mathbf{J}$ is a skew-symmetric matrix and $H_\theta$ is a learnable Hamiltonian. The pure Hamiltonian component ensures energy conservation and information non-dissipation; the damping term $\mathbf{D}_\theta$ captures non-conservative effects (e.g., frictional dissipation); and the external force term $\mathbf{r}_\theta$ models external driving. The Hamiltonian is parameterized via message passing, incorporating state information from both the node and its neighbors. A symplectic Euler integrator is employed to preserve energy conservation properties.

   **Core Advantage**: In the absence of damping and external forces, the system is purely rotational (divergence-free), so information is fully preserved. Theoretical analysis proves that the sensitivity matrix norm $\|\partial \mathbf{x}(t) / \partial \mathbf{x}(s)\| \geq 1$, guaranteeing non-vanishing gradients and supporting effective long-range information propagation.

2. **Warmup Phase**: Each message-passing step propagates information only to direct neighbors, yet long-range interactions are often critical from the very first simulation step. IGNS executes $l$ additional rounds of message passing before formal simulation (without advancing time), allowing each node to aggregate information within radius $l$. Crucially, due to the energy-conserving nature of the Hamiltonian core, the global context gathered during warmup is preserved throughout the entire rollout, rather than being dissipated as in standard GNNs.

3. **Geometric Information Encoding**: Accurately encoding geometric structure on irregular meshes is essential for physics simulation. IGNS encodes relative displacement vectors $\mathbf{s}_{ij}$ and distances $\mathbf{d}_{ij}$ into edge features used to form the external force term. Unlike MeshGraphNets, IGNS does not update edge messages at each time step; instead, edge information serves as a static prior to weight neighbor messages, reducing overfitting to specific mesh configurations.

4. **Multi-Step Training Objective**: Given a window $(q^{(t)}, \ldots, q^{(t+K)})$, the model rolls out from $q^{(t)}$ and optimizes all intermediate predictions:
    $\mathcal{L}_{\text{multi-step}} = \sum_{\tau=1}^{K} \left( \|\hat{q}^{(t+\tau)} - q^{(t+\tau)}\|_2^2 + \|\hat{p}^{(t+\tau)} - p^{(t+\tau)}\|_2^2 \right)$
   The multi-step loss is naturally compatible with IGNS: the non-dissipative core ensures signals do not decay over time, gradients from distant future steps do not vanish, and the model can genuinely leverage supervision across the full trajectory.

### Theoretical Guarantees
- **Universality (Theorem 1)**: IGNS can approximate any mapping from initial conditions to the solution at time $\tau$, proven by reducing IGNS to the neural oscillator framework.
- **Long-Range Propagation (Theorem 2)**: The sensitivity matrix norm is lower-bounded by 1, guaranteeing non-vanishing gradients — in contrast to the exponential decay observed in GCNs.

## Key Experimental Results

### Main Results
The paper evaluates on 6 benchmarks spanning Lagrangian and Eulerian systems:

| Dataset | Type | IGNS MSE | MGN MSE | Improvement |
|--------|------|----------|---------|------|
| Plate Deformation | Long-range propagation | **Best** | 1.27 | Both IGNS and MGN perform well, but IGNS does not overfit |
| Impact Plate | Long-range propagation | **Best** | 3095.75 | IGNS leads by a large margin |
| Sphere Cloth | Complex dynamics | **Best** (~25×10⁻³) | 32.07×10⁻³ | Significant improvement |
| Wave Balls | Oscillatory dynamics | **Best** (~1.5×10⁻³) | 1.78×10⁻³ | Substantially outperforms all baselines |
| Cylinder Flow | Fluid dynamics | **Best** | 12.08×10⁻³ | Comparable to GraphCON |
| Kuramoto-Sivashinsky | Chaotic dynamics | 2.41×10⁻³ | 10.76×10⁻³ | Close to GraphCON |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Data efficiency (Plate Def.) | IGNS advantage largest at 100 samples | Port-Hamiltonian inductive bias reduces data dependence |
| Warmup steps $l$ | Consistent improvement from $l=1$ to $l=30$ | $l=5$ yields the largest gain; diminishing returns beyond $l=30$ |
| Time-varying weight matrix (IGNS vs IGNSti) | IGNS performs better over long horizons | Time-varying parameterization improves expressivity for non-stationary dynamics |
| Longer rollout ($T=100$) | IGNS remains stable | Validates long-horizon stability |

### Key Findings
- The port-Hamiltonian structure consistently outperforms standard GNS **across all tasks**.
- GraphCON is a special case of IGNS (with unit mass matrix) and is theoretically less expressive.
- Although MGN is competitive on Plate Deformation, analysis reveals this stems from its large parameter count due to non-shared processors and geometric overfitting.
- IGNS substantially outperforms baselines on Wave Balls, as port-Hamiltonian dynamics are a natural generalization of the wave equation.
- The three newly introduced benchmarks (Plate Deformation, Sphere Cloth, Wave Balls) effectively probe long-range and oscillatory dynamics.

## Highlights & Insights
- **A perfect integration of physical inductive bias and data-driven learning**: Hamiltonian dynamics do not hard-code physics but provide an information-preserving structural framework within which data-driven methods can learn.
- **Theoretical-empirical consistency**: The two theorems (universality + information preservation) are not merely decorative — they directly explain why IGNS succeeds on long-range and multi-step tasks.
- **Warmup design is simple yet effective**: The global context deficit in GNS is addressed in the most straightforward way — multiple rounds of message passing without advancing time.
- **Highly parameter-efficient**: IGNS uses approximately 216K parameters versus MGN's 1.8M, achieving superior performance with an order-of-magnitude reduction in model size.

## Limitations & Future Work
- Currently supports only open-loop forward simulation; closed-loop control is not supported.
- The warmup step count $l$ requires manual selection, and the optimal value varies across tasks.
- For systems with strong global coupling (e.g., periodic boundary conditions), the local information diffusion during warmup remains limited.
- The advantage on Eulerian systems (Cylinder Flow, KS) is less pronounced than on Lagrangian systems.
- Integration with hierarchical or rewiring-based approaches has not been explored.

## Related Work & Insights
- **Relation to MeshGraphNets**: MGN uses first-order explicit Euler integration with non-shared processors; IGNS uses second-order port-Hamiltonian dynamics with a symplectic integrator — a qualitative step forward from first- to second-order dynamics.
- **Relation to GraphCON**: GraphCON is a special case of IGNS ($\mathbf{M}=\mathbf{I}$); IGNS is more expressive through learnable mass, damping, and stiffness matrices.
- **Relation to Neural ODEs**: IGNS generalizes Neural ODEs to graph-based port-Hamiltonian systems, while enjoying formal theoretical guarantees.
- **Insight**: The oscillatory nature of second-order dynamics is naturally suited to wave propagation and elastic systems in physics simulation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[AAAI 2026\] Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs](../../AAAI2026/graph_learning/are_graph_transformers_necessary_efficient_long-range_messag.md)
- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)

</div>

<!-- RELATED:END -->
