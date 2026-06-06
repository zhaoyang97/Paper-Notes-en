---
title: >-
  [Paper Note] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization
description: >-
  [ICML 2026][Optimization][QAOA] A Generative-Evaluative Network (GEN) is proposed to jointly differentiate both "graph partitioning" and "quantum circuit parameter initialization" for QAOA². The evaluator learns a high-f…
tags:
  - "ICML 2026"
  - "Optimization"
  - "QAOA"
  - "divide-and-conquer"
  - "differentiable graph partitioning"
  - "parameter warm-start"
  - "zero-shot generalization"
date: 2026-05-08
content_hash: 4d849d577d6b27af
---

# Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.13072](https://arxiv.org/abs/2605.13072)  
**Code**: https://github.com/0SliverBullet/Neural-QAOA-Squared (Available)  
**Area**: Quantum Optimization / Differentiable Programming / Graph Partitioning  
**Keywords**: QAOA, divide-and-conquer, differentiable graph partitioning, parameter warm-start, zero-shot generalization

## TL;DR
A Generative-Evaluative Network (GEN) is proposed to jointly differentiate both "graph partitioning" and "quantum circuit parameter initialization" for QAOA². The evaluator learns a high-fidelity quantum performance surrogate, while the generator outputs discrete partitions and initial parameters guided by its gradients. Combined with a Straight-Through Estimator (STE) and an Orthogonal Complement Head (OCH), the framework is end-to-end trainable. It outperforms heuristic baselines across 183 QUBO/Ising/MaxCut instances (21-1000 variables) and ranks first in 101 instances.

## Background & Motivation

**Background**: QAOA is a flagship algorithm for solving QUBO/MaxCut in the NISQ era. However, real-world problems often involve thousands of variables, while quantum hardware only supports hundreds of qubits. The divide-and-conquer (D&C) paradigm, exemplified by QAOA², addresses scalability by partitioning large graphs into hardware-compatible subgraphs, solving them via QAOA, and merging local solutions using $\mathbb{Z}_2$ symmetry.

**Limitations of Prior Work**: Existing D&C frameworks suffer from two decoupling flaws. First, partitioning heuristics (modularity, boundary, KL) are designed for graph-theoretic metrics and lack a direct link to quantum solution quality—the authors observed a Pearson correlation of only 0.2859 between modularity and performance ratio on g05_100.1, indicating near-randomness. Second, subgraph QAOA parameters $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ typically use random initialization without considering subgraph topology, leading to "cold-start" issues where doubling optimization steps ($T=40$) fails to match a topology-aware "warm-start" ($T=20$).

**Key Challenge**: Partitioning and parameter initialization are both sub-tasks mapping graph topology to quantum performance, yet they are currently handled by heuristics or randomness in isolation. Enabling end-to-end learning requires overcoming engineering hurdles such as gradient backpropagation through discrete partitions and enforcing hard qubit capacity constraints.

**Goal**: Construct a differentiable generator capable of simultaneously outputting partitions and initial values, driven by training signals from the "final quantum performance" rather than intermediate proxy metrics.

**Key Insight**: Model QAOA² performance prediction as a differentiable surrogate (quantum evaluator) and perform gradient ascent on the generator. Use a Straight-Through Estimator (STE) and Greedy Capacity Discretization (GCD) to integrate hard-constrained discrete partitioning into the differentiable pipeline, while employing an Orthogonal Complement Head (OCH) to provide geometric inductive bias for cluster centers and prevent GNN over-smoothing.

**Core Idea**: By using a dual-network structure of evaluator and generator, the framework transforms "which graph to partition" and "which initial values to assign" into a differentiable joint strategy. The evaluator provides quantum-aware gradients, achieving D&C optimization truly targeted at quantum solution outcomes.

## Method

### Overall Architecture
GEN (Generative Evaluative Network) consists of two parts. First, the **Quantum evaluator** $f_\phi(G, \mathbf{S}, \mathbf{P}) \to \hat{\rho}$ is a multi-view GNN that encodes the graph $G$, partition $\mathbf{S}$, and parameters $\mathbf{P}$ into a unified latent space to predict the performance ratio $\rho \in [0.5, 1]$ (defined as $\rho = (\text{Cut} - \text{Neg}) / (\text{OPT} - \text{Neg})$ to ensure boundedness). It is pre-trained to convergence using supervised MSE on an offline dataset $\mathcal{D}_{\text{offline}} = \{(G_i, \mathbf{S}_i, \mathbf{P}_i, \rho_i)\}$. Second, the **Joint generator** $g_\theta(G) \to (\mathbf{S}, \mathbf{P})$ follows the factorization $P(\mathbf{S}, \mathbf{P} | G) = P(\mathbf{S} | G) P(\mathbf{P} | \mathbf{S}, G)$. With $f_\phi$ frozen, unsupervised gradient ascent is performed by maximizing $\mathbb{E}_G [f_\phi(G, g_\theta(G))]$.

During inference, a forward pass yields initial values $(\mathbf{S}_0, \mathbf{P}_0) = g_\theta(G_{\text{new}})$, followed by test-time adaptation—fine-tuning the generator parameters $\theta$ via gradient ascent on the specific instance to obtain $\theta^*$ and the final output $(\mathbf{S}^*, \mathbf{P}^*) = g_{\theta^*}(G_{\text{new}})$.

### Key Designs

1.  **Multi-view Quantum Evaluator $f_\phi$**:
    - **Function**: Predicts the QAOA² performance ratio for a given partition and initialization with high fidelity, serving as a differentiable surrogate to replace expensive quantum circuit simulations.
    - **Mechanism**: Three parallel encoders process heterogeneous inputs: a topology encoder for the full adjacency matrix $\mathbf{A}$; a partition encoder for subgraph adjacency $\mathbf{A}_{\text{sub}} = \mathbf{A} \odot (\mathbf{S} \mathbf{S}^T)$ which masks inter-partition edges; and a parameter encoder that broadcasts $\mathbf{P}$ to node-level via $\mathbf{X}_{\text{param}} = \mathbf{S} \mathbf{P}^T$, using sin/cos embeddings $\tilde{\mathbf{X}}_{\text{param}} = [\sin(\mathbf{X}_{\text{param}}), \cos(\mathbf{X}_{\text{param}})]$ to respect $2\pi$ periodicity. Outputs are aggregated via global mean pooling and an MLP to produce $\hat{\rho} = 0.5(\text{sigmoid}(\text{MLP}(\mathbf{H})) + 1)$.
    - **Design Motivation**: A differentiable proxy reduces the cost of obtaining gradients from $O(\text{Simulation})$ to $O(\text{GNN Forward})$. The multi-view design ensures that each input signal has a dedicated encoder, preventing information dilution.

2.  **Orthogonal Complement Head (OCH)**:
    - **Function**: Projects GNN node embeddings onto $k$ cluster centers to obtain a soft partition $\tilde{\mathbf{S}} \in [0,1]^{N \times k}$ while preventing partition probability degradation caused by GNN over-smoothing.
    - **Mechanism**: The cluster center matrix $\mathbf{C} \in \mathbb{R}^{k \times h}$ is constrained such that $\mathbf{C} \boldsymbol{g} = \mathbf{0}$ and $\mathbf{C} \mathbf{C}^T = \mathbf{I}$, where $\boldsymbol{g} = \text{GMP}(\mathbf{H}_{\text{topology}})$ is the global graph embedding. $\mathbf{C}$ is generated dynamically via QR decomposition of a random matrix relative to $\boldsymbol{g}$. Finally, $\tilde{\mathbf{S}} = \text{softmax}(\mathbf{H}_{\text{topology}} \mathbf{C}^T)$.
    - **Design Motivation**: Standard GNNs with softmax tend to produce nearly uniform partitions as node embeddings converge (over-smoothing). Pinning cluster centers in the orthogonal complement of the global context maximizes inter-cluster separability.

3.  **Greedy Capacity Discretization (GCD) + Straight-Through Estimator (STE)**:
    - **Function**: Converts soft $\tilde{\mathbf{S}}$ into a discrete $\mathbf{S} \in \{0,1\}^{N \times k}$ that strictly satisfies qubit capacity constraints ($\sum_i \mathbf{S}_{ij} \leq \text{max\_nodes}$) while maintaining gradient flow.
    - **Mechanism**: GCD greedily assigns nodes to clusters based on descending probabilities, skipping clusters that have reached capacity. The discrete $\mathbf{S}$ is used in the forward pass for the evaluator, while the backward pass uses STE $\nabla_{\tilde{\mathbf{S}}} f \approx \nabla_{\mathbf{S}} f$. A stop-gradient operator $\text{sg}(\mathbf{A}_{\text{sub}})$ is applied during parameter generation to prevent parameter optimization from perturbing the partitions.
    - **Design Motivation**: Continuous relaxations like Gumbel-Softmax cannot guarantee hard capacity constraints (physical hardware limits). GCD ensures strict feasibility, while STE serves as a viable differentiation method for discrete decisions in NISQ contexts.

### Loss & Training
The process involves two stages: (1) The Evaluator stage minimizes MSE $\mathbb{E}_{(G, \mathbf{S}, \mathbf{P}, \rho)} [(f_\phi - \rho)^2]$ using data from heuristic partitions, uniformly sampled parameters, and QAOA² simulation labels. (2) The Generator stage freezes $f_\phi$ and maximizes $\mathbb{E}_G [f_\phi(G, g_\theta(G))]$. The generator is trained at $p=1$; for deeper circuits ($p=2, 3$), the parameter expansion strategy from Zhou 2020 is utilized instead of retraining.

## Key Experimental Results

### Main Results
Evaluated on 50 held-out test instances (20% from B/BE/W datasets):

| Dataset | Random | Modularity | Boundary | KL | **Neural QAOA²** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| B (8 QUBO) | 0.8047 (rank 4.75) | 0.8351 (2.38) | 0.8246 (2.63) | 0.8092 (3.75) | **0.8417 (1.50, 5/3 wins)** |
| BE (16 QUBO) | 0.8626 (4.81) | 0.8692 (3.13) | 0.8722 (2.31) | 0.8672 (3.69) | **0.8824 (1.06, 15/1 wins)** |
| W (26 MaxCut) | 0.8962 (3.23) | 0.9137 (2.23) | 0.9114 (2.96) | 0.8934 (4.27) | **0.9153 (2.23, 8/18 wins)** |
| **Overall (50)** | 0.8708 (3.98) | 0.8869 (2.54) | 0.8850 (2.70) | 0.8716 (4.00) | **0.8930 (1.74, 28/22 wins)** |

Neural QAOA² achieves near total dominance on the BE dataset (15/16) because QUBOs often lack explicit community structures, causing graph-theoretic heuristics like modularity to fail. On MaxCut (W), where community structures exist, modularity ties with Neural QAOA² in rank (2.23).

### Ablation Study
Tested on 93 Out-of-Distribution (OOD) instances (GKA + L):

| Configuration | GKA (45 QUBO) | L (48 Ising) | Overall (93) |
| :--- | :--- | :--- | :--- |
| Random | 0.8478 (4.16) | 0.6984 (4.65) | rank 4.41 |
| Modularity | 0.8659 (2.40) | 0.7391 (3.06) | rank 2.73 |
| Boundary | 0.8601 (2.89) | 0.8205 (1.60) | rank 2.24 |
| KL | 0.8503 (4.04) | 0.7022 (4.27) | rank 4.16 |
| **Neural QAOA² (Ours)** | **0.8762 (1.51, 32/13)** | **0.8160 (1.42, 28/20)** | **rank 1.46, 60/33 wins** |

The model maintains SOTA performance during zero-shot transfer to OOD topologies (e.g., Ising instances not seen in training), suggesting that GEN learns a universal mapping between partitions and quantum performance.

### Key Findings
- Empirical evidence showing low correlation (Pearson 0.2859) between heuristic partitioning and final performance is the core motivation of this paper.
- Cold-start losses cannot be compensated by more iterations; random initialization with $T=40$ optimization steps still underperforms topology-aware initialization with $T=20$.
- Models trained on $p=1$ generalize to $p=2, 3$ and outperform advanced baselines like TQA/INTERP/FOURIER, indicating the learned topological mapping is "parameter schedule agnostic."
- OCH is critical: removing the orthogonal complement constraint causes GNN outputs to collapse into uniform probability distributions, resulting in near-random partitioning.

## Highlights & Insights
- Successfully transitioning from heuristic D&C to end-to-end differentiable D&C is a significant engineering feat. The authors simultaneously addressed (a) gradient backpropagation through discrete decisions, (b) hard capacity constraints, and (c) GNN over-smoothing via modular components (STE/GCD/OCH).
- Using an evaluator as a differentiable surrogate to provide gradient signals has precedence in Neural Architecture Search but is novel to quantum combinatorial optimization. It bridges the gap between expensive oracle evaluation and differentiable optimization.
- Dynamically generating cluster centers via QR decomposition for OCH is clever: unlike traditional clustering where centers are learnable (and prone to collapse), pinning them in the "orthogonal complement of the global context" provides a stable geometric anchor.
- Test-time adaptation acknowledges the reality that training distributions $\neq$ inference distributions, allowing the model to transition from distribution priors to instance-specific configurations.

## Limitations & Future Work
- The upper bound of $\rho=1.0$ is relative to the best-known cut, which may not be the true optimum for large problems, introducing ground-truth bias.
- Training and evaluation rely on QAOA² simulations and have not been validated on real hardware (considering noise, readout errors, and connectivity).
- The generator is only trained at $p=1$; scaling to deeper circuits relies on empirical schedules without theoretical guarantees.
- The `max_nodes=10` limit does not yet reflect the capacity of modern QPUs (~100 qubits); scalability to higher qubit counts remains unverified.
- While OOD evaluation was performed on GKA/L, testing on truly "wild" industrial graphs is still needed.

## Related Work & Insights
- **vs DC-QAOA / Original QAOA²**: These share the D&C paradigm but rely on heuristic partitioning and random initialization. Neural QAOA² replaces these manual components with end-to-end learnable networks.
- **vs INTERP / FOURIER / TQA / QIBPI**: These methods only address parameter initialization for a given partition. This paper jointly optimizes both, showing that the gain from joint training exceeds that of parameter optimization alone.
- **vs GNN Policies in CO**: Similar to GNN + RL/differentiable optimization approaches. The specificity of this work lies in the evaluator-generator architecture and the quantum surrogate, reflecting the high cost of quantum performance signals.
- **Insight**: The evaluator-generator architecture could be applied to other "expensive oracle + discrete decision" problems, such as chip floorplanning or hyperparameter search, provided a high-fidelity differentiable proxy can be learned.

## Rating
- Novelty: ⭐⭐⭐⭐ Differentiable D&C is a solid integrative innovation rather than a disruptive new mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐ 183 instances, multi-dataset evaluation, and depth generalization comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation and pipeline are exceptionally clear.
- Value: ⭐⭐⭐⭐ Significant for the deployment of quantum combinatorial optimization in the NISQ era.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning to Approximate Uniform Facility Location via Graph Neural Networks](learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)
- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](../../ICLR2026/optimization/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[ICML 2026\] URS: A Unified Neural Routing Solver](urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](../../NeurIPS2025/optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)

</div>

<!-- RELATED:END -->
