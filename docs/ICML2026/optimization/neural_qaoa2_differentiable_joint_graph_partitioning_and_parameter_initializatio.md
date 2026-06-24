---
title: >-
  [Paper Note] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization
description: >-
  [ICML 2026][Optimization][QAOA] A generative-evaluative neural network (GEN) is proposed to jointly differentiate "graph partitioning + quantum circuit parameter initialization" for QAOA². The evaluator learns a high-fidelity quantum performance surrogate, while the generator outputs discrete partitions and initial parameters guided by its gradients. Straight-Through Estimator (STE) and an Orthogonal Complement Head (OCH) enable end-to-end training. The method surpasses heuri…
tags:
  - "ICML 2026"
  - "Optimization"
  - "QAOA"
  - "divide-and-conquer"
  - "differentiable graph partitioning"
  - "parameter warm-start"
  - "zero-shot generalization"
date: 2026-05-08
content_hash: 3955d72b68ab82a9
---

# Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.13072](https://arxiv.org/abs/2605.13072)  
**Code**: https://github.com/0SliverBullet/Neural-QAOA-Squared (Available)  
**Area**: Quantum Optimization / Differentiable Programming / Graph Partitioning  
**Keywords**: QAOA, divide-and-conquer, differentiable graph partitioning, parameter warm-start, zero-shot generalization

## TL;DR
A generative-evaluative neural network (GEN) is proposed to jointly differentiate "graph partitioning + quantum circuit parameter initialization" for QAOA². The evaluator learns a high-fidelity quantum performance surrogate, while the generator outputs discrete partitions and initial parameters guided by its gradients. Straight-Through Estimator (STE) and an Orthogonal Complement Head (OCH) enable end-to-end training. The method surpasses heuristic baselines across 183 QUBO/Ising/MaxCut instances (21-1000 variables), ranking first in 101 instances.

## Background & Motivation

**Background**: QAOA is a flagship algorithm for solving QUBO/MaxCut in the NISQ era, but real-world problems often involve thousands of variables while quantum hardware only supports hundreds of qubits. The divide-and-conquer (D&C) paradigm, represented by QAOA², scales by partitioning large graphs into hardware-compatible subgraphs, solving them via QAOA, and merging local solutions using $\mathbb{Z}_2$ symmetry.

**Limitations of Prior Work**: Existing D&C frameworks suffer from two decoupling flaws. First, partitioning heuristics (modularity, boundary, KL) are designed for "graph-theoretic metrics" and lack a direct link to final quantum solution quality—the authors found that the Pearson correlation between modularity and performance ratio on g05_100.1 is only 0.2859 (almost random). Second, QAOA parameters $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ are typically randomly initialized regardless of subgraph topology, leading to "cold-start" issues where even doubling optimization steps ($T=40$) cannot match a topology-aware warm-start ($T=20$).

**Key Challenge**: Partitioning and parameter initialization are both subtasks of mapping graph topology to quantum performance, yet they are currently handled separately via heuristics or randomness. To enable end-to-end learning, engineering challenges such as "differentiating through discrete partitions" and "handling hard qubit capacity constraints" must be addressed.

**Goal**: Construct a differentiable generator capable of simultaneously outputting partitions and initial values, where training signals originate from the "final quantum performance" rather than intermediate proxy metrics.

**Key Insight**: Model QAOA² performance prediction as a differentiable surrogate (quantum evaluator) and perform gradient ascent on the generator. Use a Straight-Through Estimator (STE) combined with Greedy Capacity Discretization (GCD) to embed hard-constrained discrete partitioning into the differentiable pipeline. Apply an Orthogonal Complement Head (OCH) to provide a geometric inductive bias for cluster centers, preventing GNN over-smoothing.

**Core Idea**: Utilize a dual-network structure (evaluator + generator) to create a differentiable joint strategy for "what to partition and where to initialize," providing quantum-aware gradients to optimize specifically for the quantum solution outcome.

## Method

### Overall Architecture
GEN (Generative Evaluative Network) consists of two parts. First, the **Quantum evaluator** $f_\phi(G, \mathbf{S}, \mathbf{P}) \to \hat{\rho}$ is a multi-view GNN that encodes the graph $G$, partition $\mathbf{S}$, and parameters $\mathbf{P}$ into a unified latent space to predict the performance ratio $\rho \in [0.5, 1]$ (where $\rho = (\text{Cut} - \text{Neg}) / (\text{OPT} - \text{Neg})$). It is pre-trained using supervised MSE on an offline dataset $\mathcal{D}_{\text{offline}} = \{(G_i, \mathbf{S}_i, \mathbf{P}_i, \rho_i)\}$. Second, the **Joint generator** $g_\theta(G) \to (\mathbf{S}, \mathbf{P})$ follows $P(\mathbf{S}, \mathbf{P} | G) = P(\mathbf{S} | G) P(\mathbf{P} | \mathbf{S}, G)$. With $f_\phi$ frozen, unsupervised gradient ascent is performed via $\max_\theta \mathbb{E}_G [f_\phi(G, g_\theta(G))]$.

During inference, initial values $(\mathbf{S}_0, \mathbf{P}_0) = g_\theta(G_{\text{new}})$ are obtained via a forward pass, followed by test-time adaptation—fine-tuning the generator parameters $\theta$ on the specific instance to obtain $\theta^*$ and the final output $(\mathbf{S}^*, \mathbf{P}^*)$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    G["Input Graph G"] --> GEN
    subgraph GEN["Joint Generator gθ (Partition then Param)"]
        direction TB
        OCH["Orthogonal Complement Head (OCH)<br/>Topology encoding + Orthogonal constraint<br/>→ soft partition S̃"]
        OCH --> GCD["Greedy Capacity Discretization (GCD) + STE<br/>Forward discrete for capacity / Backward STE for gradient<br/>→ discrete partition S"]
        GCD --> PG["Parameter Generator<br/>sg(A_sub) + arctan → initial param P"]
    end
    GEN -->|Output (S, P)| EVAL["Multi-view Quantum Evaluator fφ<br/>Topology / Partition / Param GNNs<br/>→ performance ratio ρ̂"]
    EVAL -->|Gradient Ascent Guidance (fφ frozen)| GEN
    EVAL --> OUT["Inference: Forward init + Test-time adaptation"]
```

### Key Designs

**1. Multi-view Quantum Evaluator $f_\phi$: A differentiable proxy replacing quantum simulation**
The bottleneck of prior D&C methods was the inability to use the true feedback signal (quantum performance) due to high evaluation costs and non-differentiability. GEN trains a high-fidelity surrogate $f_\phi$ to predict the performance ratio, reducing the cost of gradient calculation from $O(\text{Quantum Simulation})$ to $O(\text{GNN Forward})$. It uses three parallel encoders: a topology encoder for the full graph adjacency $\mathbf{A}$; a partition encoder for the subgraph adjacency $\mathbf{A}_{\text{sub}}=\mathbf{A}\odot(\mathbf{S}\mathbf{S}^T)$ where cross-partition edges are masked; and a parameter encoder that broadcasts parameters to node-level via $\mathbf{X}_{\text{param}}=\mathbf{S}\mathbf{P}^T$ and uses embeddings $\tilde{\mathbf{X}}_{\text{param}}=[\sin(\mathbf{X}_{\text{param}}),\cos(\mathbf{X}_{\text{param}})]$ to respect $2\pi$ periodicity. After global mean pooling, features are concatenated into an MLP, and $\hat{\rho}=0.5(\text{sigmoid}(\text{MLP}(\mathbf{H}))+1)$ ensures the output stays within the theoretical range $[0.5, 1]$.

**2. Orthogonal Complement Head (OCH): Geometric anchors to prevent GNN over-smoothing**
Generators typically project node embeddings onto $k$ cluster centers to obtain soft partitions $\tilde{\mathbf{S}}\in[0,1]^{N\times k}$. However, standard GNN + softmax often suffers from over-smoothing, where node embeddings converge and partition probabilities become nearly uniform. OCH enforces two constraints on the cluster center matrix $\mathbf{C}\in\mathbb{R}^{k\times h}$: $\mathbf{C}\boldsymbol{g}=\mathbf{0}$ and $\mathbf{C}\mathbf{C}^T=\mathbf{I}$, where $\boldsymbol{g}=\text{GMP}(\mathbf{H}_{\text{topology}})$ is the global graph embedding. $\mathbf{C}$ is dynamically generated via QR decomposition relative to $\boldsymbol{g}$, and $\tilde{\mathbf{S}}=\text{softmax}(\mathbf{H}_{\text{topology}}\mathbf{C}^T)$. By anchoring centers in the orthogonal complement of the global embedding, the model maximizes inter-cluster separability, which is more stable than treating centers as learnable parameters.

**3. Greedy Capacity Discretization (GCD) + Straight-Through Estimator (STE): Discrete feasibility with gradient flow**
Qubit capacity is a hard physical limit: $\sum_i\mathbf{S}_{ij}\le\text{max\_nodes}$ must not be violated. Thus, continuous relaxations like Gumbel-Softmax are unsuitable. GCD greedily assigns nodes to clusters based on descending probability until capacity is reached. To handle the non-differentiable discretization, the forward pass uses the discrete $\mathbf{S}$ for the evaluator, while the backward pass uses STE: $\nabla_{\tilde{\mathbf{S}}}f\approx\nabla_{\mathbf{S}}f$. For parameter generation, a stop-gradient $\text{sg}(\mathbf{A}_{\text{sub}})$ is added to prevent parameter optimization from perturbing the partition decisions.

### Loss & Training
Two-stage training: (1) Evaluator stage minimizes MSE $\mathbb{E}_{(G, \mathbf{S}, \mathbf{P}, \rho)} [(f_\phi - \rho)^2]$ using data generated from heuristic partitions and sampled parameters. (2) Generator stage freezes $f_\phi$ and maximizes $\mathbb{E}_G [f_\phi(G, g_\theta(G))]$. The generator is trained at $p=1$; deeper circuits ($p=2, 3$) leverage parameter expansion strategies rather than retraining.

## Key Experimental Results

### Main Results
On 50 held-out test instances (20% of B/BE/W datasets):

| Dataset | Random | Modularity | Boundary | KL | **Neural QAOA²** |
|--------|--------|------------|----------|------|-----------------|
| B (8 QUBO) | 0.8047 (rank 4.75) | 0.8351 (2.38) | 0.8246 (2.63) | 0.8092 (3.75) | **0.8417 (1.50, 5/3 wins)** |
| BE (16 QUBO) | 0.8626 (4.81) | 0.8692 (3.13) | 0.8722 (2.31) | 0.8672 (3.69) | **0.8824 (1.06, 15/1 wins)** |
| W (26 MaxCut) | 0.8962 (3.23) | 0.9137 (2.23) | 0.9114 (2.96) | 0.8934 (4.27) | **0.9153 (2.23, 8/18 wins)** |
| **Overall (50)** | 0.8708 (3.98) | 0.8869 (2.54) | 0.8850 (2.70) | 0.8716 (4.00) | **0.8930 (1.74, 28/22 wins)** |

Neural QAOA² dominates the BE dataset (15/16) because QUBO typically lacks explicit community structure, causing modularity heuristics to fail.

### Ablation Study
On 93 OOD instances (GKA + L, out-of-distribution):

| Config | GKA (45 QUBO) | L (48 Ising) | Overall (93) |
|------|------------------|-----------------|--------------|
| Random | 0.8478 (4.16) | 0.6984 (4.65) | rank 4.41 |
| Modularity | 0.8659 (2.40) | 0.7391 (3.06) | rank 2.73 |
| Boundary | 0.8601 (2.89) | 0.8205 (1.60) | rank 2.24 |
| KL | 0.8503 (4.04) | 0.7022 (4.27) | rank 4.16 |
| **Neural QAOA² (Ours)** | **0.8762 (1.51, 32/13)** | **0.8160 (1.42, 28/20)** | **rank 1.46, 60/33 wins** |

The SOTA zero-shot performance on Ising topologies (not present in the training set) suggests that GEN learns a universal mapping from partitioning to quantum performance.

### Key Findings
- Empirical evidence showing low correlation (Pearson 0.2859) between heuristic partitioning and performance supports the central motivation.
- Random initialization with $T=40$ optimization steps cannot outperform topology-aware initialization with $T=20$, proving that cold-start losses cannot be compensated by extra iterations.
- Models trained at $p=1$ outperform advanced initialization baselines (TQA/INTERP/FOURIER) at $p=2, 3$, showing the learned topology mapping is "parameter schedule independent."
- OCH is critical: removing the orthogonal complement constraint causes the GNN outputs to collapse into uniform probability distributions.

## Highlights & Insights
- Transitioning from heuristic D&C to end-to-end differentiable D&C is a significant engineering feat: the authors solved (a) gradient flow for discrete decisions, (b) hard capacity constraints, and (c) GNN over-smoothing using modular components (STE/GCD/OCH).
- Using an evaluator as a differentiable surrogate to provide gradient signals mirrors strategies in neural architecture search but is novel for quantum combinatorial optimization, bridging the gap between expensive oracle evaluation and differentiable optimization.
- The use of QR decomposition for OCH centers is clever; anchoring centers in the orthogonal complement of the global context prevents the collapse often seen when treating centers as learnable parameters.
- Test-time adaptation acknowledges the gap between training and inference distributions, transitioning from a distribution prior to instance-specific configurations.

## Limitations & Future Work
- The upper bound for $\rho$ (1.0) is relative to the best-known cut, which may not be the true optimum for large instances, introducing ground-truth bias.
- Both training and evaluation rely on QAOA² simulations; validation on real hardware (considering noise, readout errors, and connectivity) is missing.
- The generator is only trained at $p=1$; scaling to deeper circuits relies on empirical schedules without theoretical guarantees.
- The hard limit of `max_nodes=10` does not reflect current hundred-qubit QPUs; scalability to higher qubit counts remains unverified.

## Related Work & Insights
- **vs. DC-QAOA / Original QAOA²**: These use heuristic partitioning and random initialization. Neural QAOA² replaces these manual components with learnable networks.
- **vs. INTERP / FOURIER / TQA / QIBPI**: These methods only address parameter initialization for a given partition. This work demonstrates that joint optimization yield gains strictly greater than optimizing parameters alone.
- **vs. Sampled MuZero / GNN policies in NCO**: While similar in using GNNs for combinatorial decisions, this work's specificity lies in the evaluator-generator architecture and quantum surrogates, reflecting the high cost of quantum reward signals.
- **Insight**: The evaluator-generator architecture could potentially be applied to other "expensive oracle + discrete decision" problems like chip floorplanning or compiler tuning.

## Rating
- Novelty: ⭐⭐⭐⭐ (Solid engineering contribution through the combination of STE/GCD/OCH)
- Experimental Thoroughness: ⭐⭐⭐⭐ (183 instances, IID/OOD testing, and comparisons across circuit depths)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation and well-described pipeline)
- Value: ⭐⭐⭐⭐ (Directly applicable to quantum combinatorial optimization deployment in the NISQ era)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bayesian Parameter Shift Rules in Variational Quantum Eigensolvers](../../ICLR2026/optimization/bayesian_parameter_shift_rules_in_variational_quantum_eigensolvers.md)
- [\[ICML 2026\] A Fully First-Order Layer for Differentiable Optimization](a_fully_first-order_layer_for_differentiable_optimization.md)
- [\[ICLR 2026\] Multi-Action Self-Improvement for Neural Combinatorial Optimization](../../ICLR2026/optimization/multi-action_self-improvement_for_neural_combinatorial_optimization.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[ICML 2026\] URS: Unified Neural Routing Solver](urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)

</div>

<!-- RELATED:END -->
