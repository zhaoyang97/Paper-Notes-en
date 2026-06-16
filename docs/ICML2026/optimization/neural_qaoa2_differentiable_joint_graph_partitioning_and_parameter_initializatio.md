---
title: >-
  [Paper Note] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization
description: >-
  [ICML 2026][Optimization & Theory][QAOA] A generative-evaluative neural network (GEN) is used to jointly differentiate "graph partitioning + quantum circuit parameter initialization" for QAOA². The evaluator learns a high-fidelity quantum performance surrogate, while the generator outputs discrete partitions and parameter initializations under the guidance of
tags:
  - ICML 2026
  - Optimization & Theory
  - QAOA
  - divide-and-conquer
date: 2026-05-08
content_hash: f5a3f5a0bc4a8e7a
---
# Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.13072](https://arxiv.org/abs/2605.13072)  
**Code**: https://github.com/0SliverBullet/Neural-QAOA-Squared (Available)  
**Area**: Quantum Optimization / Differentiable Programming / Graph Partitioning  
**Keywords**: QAOA, divide-and-conquer, differentiable graph partitioning, parameter warm-start, zero-shot generalization

## TL;DR
A generative-evaluative neural network (GEN) is used to jointly differentiate "graph partitioning + quantum circuit parameter initialization" for QAOA². The evaluator learns a high-fidelity quantum performance surrogate, while the generator outputs discrete partitions and parameter initializations under the guidance of its gradients. This approach incorporate a straight-through estimator and an orthogonal complement head to enable end-to-end training. It outperforms heuristic baselines across 183 QUBO/Ising/MaxCut instances (21–1000 variables), ranking first in 101 instances.

## Background & Motivation

**Background**: QAOA is a flagship algorithm for solving QUBO/MaxCut in the NISQ era. However, real-world problems often involve thousands of variables, while quantum hardware is limited to hundreds of qubits. The divide-and-conquer (D&C) paradigm (represented by QAOA²) addresses scalability by partitioning large graphs into subgraphs that fit on hardware, solving them separately with QAOA, and merging local solutions using $\mathbb{Z}_2$ symmetry.

**Limitations of Prior Work**: Existing D&C frameworks suffer from two decoupling flaws. First, graph partitioning heuristics (modularity, boundary, KL) are designed for "graph-theoretic metrics" and have no direct relation to final quantum solution quality—the authors observed a Pearson correlation of only 0.2859 between modularity and performance ratio on g05_100.1, which is nearly random. Second, QAOA parameters $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ on subgraphs are initialized randomly without considering subgraph topology, leading to "cold-start" issues—even doubling the optimization steps ($T=40$) fails to catch up with a topology-aware warm-start ($T=20$).

**Key Challenge**: Partitioning and parameter initialization are both subtasks of mapping graph topology to quantum performance, yet they are handled separately via heuristics or randomness, lacking synergy. To make them end-to-end learnable, engineering challenges regarding "gradient propagation through discrete partitions" and "hard qubit capacity constraints" must be resolved.

**Goal**: Construct a differentiable generator capable of simultaneously outputting partitions and initial values, with training signals derived from "final quantum performance" rather than intermediate proxy metrics.

**Key Insight**: Model QAOA² performance prediction as a differentiable surrogate (quantum evaluator) and perform gradient ascent on the generator based on its gradients; utilize a straight-through estimator (STE) and greedy capacity discretization (GCD) to "sandwich" hard-constrained discrete partitions into the differentiable pipeline; finally, apply an orthogonal complement head (OCH) to provide a geometric inductive bias for cluster centers to prevent GNN over-smoothing.

**Core Idea**: Utilize a dual-network structure (evaluator + generator) to formulate "which graph to partition and which initial values to provide" as a differentiable joint strategy. The evaluator provides quantum-aware gradients, achieving D&C optimization truly focused on the quantum solutions.

## Method

### Overall Architecture
GEN (Generative Evaluative Network) consists of two components. First, the **Quantum evaluator** $f_\phi(G, \mathbf{S}, \mathbf{P}) \to \hat{\rho}$ is a multi-view GNN that encodes the graph $G$, partition $\mathbf{S}$, and parameters $\mathbf{P}$ into a unified latent space to predict the performance ratio $\rho \in [0.5, 1]$ (the formula $\rho = (\text{Cut} - \text{Neg}) / (\text{OPT} - \text{Neg})$ ensures boundedness). It is first trained to convergence using supervised MSE on a labeled dataset $\mathcal{D}_{\text{offline}} = \{(G_i, \mathbf{S}_i, \mathbf{P}_i, \rho_i)\}$. Second, the **Joint generator** $g_\theta(G) \to (\mathbf{S}, \mathbf{P})$ follows $P(\mathbf{S}, \mathbf{P} | G) = P(\mathbf{S} | G) P(\mathbf{P} | \mathbf{S}, G)$ for sequential partitioning and parameterization. With $f_\phi$ frozen, unsupervised gradient ascent is performed via $\max_\theta \mathbb{E}_G [f_\phi(G, g_\theta(G))]$.

During inference, a single forward pass $(\mathbf{S}_0, \mathbf{P}_0) = g_\theta(G_{\text{new}})$ provides initial values, followed by test-time adaptation—fine-tuning the generator parameters $\theta$ on that single instance via several steps of gradient ascent to obtain $\theta^*$ and the final output $(\mathbf{S}^*, \mathbf{P}^*) = g_{\theta^*}(G_{\text{new}})$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    G["Input Graph G"] --> GEN
    subgraph GEN["Joint Generator gθ (Partition first, then Params)"]
        direction TB
        OCH["Orthogonal Complement Head (OCH)<br/>Topology coding + cluster center orthogonal constraints<br/>→ soft partition S̃"]
        OCH --> GCD["Greedy Capacity Discretization (GCD) + STE<br/>Forward: Discretization meets qubit capacity / Backward: STE backprop gradients<br/>→ Discrete partition S"]
        GCD --> PG["Parameter Generator<br/>sg(A_sub) + arctan → initial parameters P"]
    end
    GEN -->|Output (S, P)| EVAL["Multi-view Quantum Evaluator fφ<br/>Three-way GNN (Topology / Partition / Param)<br/>→ Performance ratio ρ̂"]
    EVAL -->|Gradient ascent guidance (fφ frozen)| GEN
    EVAL --> OUT["Inference: Forward pass for initialization + test-time adaptation to fine-tune θ"]
```

### Key Designs

**1. Multi-view Quantum Evaluator $f_\phi$: Learning a differentiable proxy to replace "running a quantum simulation" with "a GNN forward pass"**

The Achilles' heel of previous D&C approaches was the reliance on heuristics for partitioning and parameter selection because the actual feedback signal—quantum performance—was too costly to evaluate and non-differentiable. GEN addresses this by training a high-fidelity surrogate $f_\phi$ to predict the performance ratio, reducing the "gradient calculation" cost from $O(\text{quantum simulation})$ to $O(\text{GNN forward pass})$. It employs three parallel encoders for heterogeneous inputs: a topology encoder for the full graph adjacency $\mathbf{A}$; a partition encoder for subgraph adjacency $\mathbf{A}_{\text{sub}}=\mathbf{A}\odot(\mathbf{S}\mathbf{S}^T)$ where cross-partition edges are masked; and a parameter encoder that broadcasts parameters to the node level via $\mathbf{X}_{\text{param}}=\mathbf{S}\mathbf{P}^T$, using $\tilde{\mathbf{X}}_{\text{param}}=[\sin(\mathbf{X}_{\text{param}}),\cos(\mathbf{X}_{\text{param}})]$ embedding to respect $2\pi$ periodicity. The outputs after global mean pooling are concatenated and passed through an MLP, with $\hat{\rho}=0.5(\text{sigmoid}(\text{MLP}(\mathbf{H}))+1)$ to ensure results stay within the theoretical range $[0.5,1]$. The multi-view design ensures each input signal has a dedicated encoder, preventing signal dilution and ensuring surrogate fidelity.

**2. Orthogonal Complement Head (OCH): Providing stationary geometric anchors for cluster centers to block GNN over-smoothing**

The generator projects node embeddings onto $k$ cluster centers to obtain a soft partition $\tilde{\mathbf{S}}\in[0,1]^{N\times k}$. However, standard GNNs with softmax suffer from over-smoothing, where node embeddings converge, causing partition probabilities to degenerate into near-uniformity and diluting training gradients. OCH counters this by imposing two orthogonal constraints on the cluster center matrix $\mathbf{C}\in\mathbb{R}^{k\times h}$: $\mathbf{C}\boldsymbol{g}=\mathbf{0}$ and $\mathbf{C}\mathbf{C}^T=\mathbf{I}$, where $\boldsymbol{g}=\text{GMP}(\mathbf{H}_{\text{topology}})$ is the global graph embedding. $\mathbf{C}$ is dynamically generated via QR decomposition of a random matrix relative to $\boldsymbol{g}$, and the final soft partition is $\tilde{\mathbf{S}}=\text{softmax}(\mathbf{H}_{\text{topology}}\mathbf{C}^T)$. Fixing centers in the orthogonal complement of the global embedding is equivalent to "subtraction using global context," maximizing inter-cluster separability—this is far more stable than treating centers as learnable parameters which are prone to collapse.

**3. Greedy Capacity Discretization (GCD) + Straight-Through Estimator (STE): Ensuring feasibility under hard constraints while allowing gradient backpropagation**

Qubit capacity is a physical hardware limit; $\sum_i\mathbf{S}_{ij}\le\text{max\_nodes}$ must never be violated. Consequently, continuous relaxations like Gumbel-Softmax are inapplicable. GCD greedily assigns nodes to clusters based on descending probabilities and skips full clusters, ensuring 100% compliance with capacity constraints. Since discretization breaks the gradient chain, the discrete $\mathbf{S}$ is used in the forward pass for accurate evaluator scores, while the straight-through estimator $\nabla_{\tilde{\mathbf{S}}}f\approx\nabla_{\mathbf{S}}f$ is used in the backward pass. A stop-gradient $\text{sg}(\mathbf{A}_{\text{sub}})$ is added during parameter generation to prevent parameter optimization from perturbing the partition. This approach prioritizes strict feasibility over gradient precision—a necessary compromise in NISQ hardware scenarios.

### Loss & Training
Two stages: (1) Evaluator stage minimizes MSE $\mathbb{E}_{(G, \mathbf{S}, \mathbf{P}, \rho)} [(f_\phi - \rho)^2]$, using data from heuristic partitions, uniformly sampled parameters, and ground-truth QAOA² simulations. (2) Generator stage freezes $f_\phi$ and maximizes $\mathbb{E}_G [f_\phi(G, g_\theta(G))]$. The generator is trained only on $p=1$; for deeper circuits ($p=2, 3$), the parameter extension strategy from Zhou 2020 is applied instead of retraining.

## Key Experimental Results

### Main Results
On 50 held-out test instances (20% held out from B/BE/W datasets, scale matching the training distribution):

| Dataset | Random | Modularity | Boundary | KL | **Neural QAOA² (Ours)**|
|--------|--------|------------|----------|------|-----------------|
| B (8 QUBO) | 0.8047 (rank 4.75) | 0.8351 (2.38) | 0.8246 (2.63) | 0.8092 (3.75) | **0.8417 (1.50, 5/3 wins)** |
| BE (16 QUBO) | 0.8626 (4.81) | 0.8692 (3.13) | 0.8722 (2.31) | 0.8672 (3.69) | **0.8824 (1.06, 15/1 wins)** |
| W (26 MaxCut) | 0.8962 (3.23) | 0.9137 (2.23) | 0.9114 (2.96) | 0.8934 (4.27) | **0.9153 (2.23, 8/18 wins)** |
| **Overall (50)** | 0.8708 (3.98) | 0.8869 (2.54) | 0.8850 (2.70) | 0.8716 (4.00) | **0.8930 (1.74, 28/22 wins)** |

Neural QAOA² dominates the BE dataset (15/16) because QUBO typically lacks explicit community structures, causing graph-theoretic heuristics like modularity to fail. On W (MaxCut), modularity ties with Neural QAOA² in rank (2.23) due to inherent community structures.

### Ablation Study
93 OOD instances (GKA + L, out-of-distribution, comparable scale):

| Configuration | GKA (45 QUBO) | L (48 Ising) | Overall (93) |
|------|------------------|-----------------|--------------|
| Random | 0.8478 (4.16) | 0.6984 (4.65) | rank 4.41 |
| Modularity | 0.8659 (2.40) | 0.7391 (3.06) | rank 2.73 |
| Boundary | 0.8601 (2.89) | 0.8205 (1.60) | rank 2.24 |
| KL | 0.8503 (4.04) | 0.7022 (4.27) | rank 4.16 |
| **Neural QAOA² (Ours)** | **0.8762 (1.51, 32/13)** | **0.8160 (1.42, 28/20)** | **rank 1.46, 60/33 wins** |

Zero-shot transfer to OOD topologies (Ising was not in the training set) remains SOTA, indicating that GEN learns a general mapping of partition-to-quantum-performance rather than dataset-specific features.

### Key Findings
- Empirical evidence of the low correlation (Pearson 0.2859) between heuristic partitions and final performance is the core observation supporting the paper's motivation.
- Even with random initialization and $T=40$ optimization steps, performance fails to match a topology-aware initialization with $T=20$, showing that "cold-start" losses cannot be compensated for by more iterations.
- Models trained on $p=1$ still outperform advanced initialization baselines like TQA/INTERP/FOURIER/QIBPI when transferred to $p=2, 3$, suggesting the learned topological mapping is "parameter schedule independent."
- OCH is critical: removing the orthogonal complement constraint results in GNN outputs degenerating into uniform probability distributions, making partitions effectively random.

## Highlights & Insights
- Transitioning from "Heuristic D&C to End-to-end Differentiable D&C" is a significant engineering achievement: the authors simultaneously solved (a) gradient backpropagation for discrete decisions, (b) hard capacity constraints, and (c) GNN over-smoothing using clean, modular components (STE/GCD/OCH).
- Using an evaluator as a differentiable surrogate for gradient signals has precedents in neural architecture search, but its application to quantum combinatorial optimization is novel—it bridges the gap between expensive oracle evaluation and differentiable optimization.
- The use of QR decomposition in OCH to dynamically generate cluster centers is clever: while traditional clustering treats centers as learnable parameters prone to collapse, anchoring them in the "orthogonal complement of the global context" provides a stationary geometric anchor.
- Test-time adaptation acknowledges the reality that training distributions differ from inference distributions, allowing the model to transition from a distribution prior to instance-specific configurations, which is beneficial for industrial deployment.

## Limitations & Future Work
- The upper bound of $\rho = 1.0$ is relative to the "best-known cut," which may not be the true optimum—on large problems, this is provided by classical heuristics, introducing a ground-truth bias.
- Both the training set and evaluation rely on QAOA² simulations and have not been validated on real hardware (noise, readout errors, connectivity constraints).
- The generator is trained only on $p=1$, and expansion to deeper circuits relies on empirical schedules without theoretical guarantees.
- The hard limit of `max_nodes=10` does not correspond to the largest current QPUs, and scalability at higher qubit counts remains unverified.
- OOD evaluation was performed on GKA/L, which are still from benchmark libraries; the model remains untested on "wild" industrial graphs.

## Related Work & Insights
- **vs DC-QAOA / Original QAOA²**: All use the D&C paradigm, but others rely on heuristic partitioning and random initial values; Neural QAOA² replaces these with end-to-end learnable networks.
- **vs INTERP / FOURIER / TQA / QIBPI Parameter Initialization**: These methods only address "how to initialize parameters" while assuming heuristics for partitioning; this work jointly optimizes both, showing that the gain from joint training significantly exceeds parameter-only optimization.
- **vs Sampled MuZero / GNN policy in Neural CO**: These use GNNs with RL or differentiable optimization for combinatorial decisions; this work is unique for its evaluator-generator dual network and quantum surrogate, reflecting the constraint that quantum performance evaluation is much costlier than typical reward signals.
- Insight: Can the evaluator-generator architecture be applied to other "expensive oracle + discrete decision" problems like chip floorplanning, compiler tuning, or hyperparameter search? The principles are general as long as a high-fidelity differentiable proxy can be learned.

## Rating
- Novelty: ⭐⭐⭐⭐ Differentiability of D&C is an incremental innovation, but the specific combination of OCH + GCD + STE is a solid contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 183 instances, 50 IID, 93 OOD, multiple heuristic baselines, and varying $p$ depths provide a systematic evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation (Figure 1) is clear, the pipeline is well-described, and key designs are detailed in dedicated sections.
- Value: ⭐⭐⭐⭐ Directly valuable for quantum combinatorial optimization deployment in the NISQ era; open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](../../ICML2025/optimization/quantum_optimization_via_gradient-based_hamiltonian_descent.md)
- [\[ICML 2025\] BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization](../../ICML2025/optimization/bopo_neural_combinatorial_optimization_via_best-anchored_and_objective-guided_pr.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](../../NeurIPS2025/optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)

</div>

<!-- RELATED:END -->
