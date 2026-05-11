---
title: >-
  [Paper Note] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization
description: >-
  [ICLR 2026][Optimization][Neural Combinatorial Optimization] This paper proposes the RRNCO architecture, which introduces two key innovations — Adaptive Node Embedding (ANE) and Neural Adaptive Bias (NAB) — to jointly mo…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Neural Combinatorial Optimization"
  - "Vehicle Routing Problem"
  - "Asymmetric Routing"
  - "Sim-to-Real Gap"
  - "Attention-Free Module"
date: 2026-05-08
content_hash: 7c81a0c2212c34c1
---

# RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization

**Conference**: ICLR 2026
**arXiv**: [2503.16159](https://arxiv.org/abs/2503.16159)
**Code**: [https://github.com/ai4co/real-routing-nco](https://github.com/ai4co/real-routing-nco)
**Area**: Combinatorial Optimization / Neural Routing Planning
**Keywords**: Neural Combinatorial Optimization, Vehicle Routing Problem, Asymmetric Routing, Sim-to-Real Gap, Attention-Free Module

## TL;DR

This paper proposes the RRNCO architecture, which introduces two key innovations — Adaptive Node Embedding (ANE) and Neural Adaptive Bias (NAB) — to jointly model asymmetric distances, travel durations, and bearing angles within a deep routing framework for the first time. It also constructs a VRP benchmark dataset based on 100 real-world cities, significantly narrowing the sim-to-real gap for NCO methods.

## Background & Motivation

**The Vehicle Routing Problem (VRP) is central to logistics optimization**: VRP is a class of NP-hard combinatorial optimization problems with broad applications in last-mile delivery, disaster relief, and urban mobility. The global logistics market exceeded $10 trillion in 2025, making improvements in routing efficiency highly valuable for cost reduction and environmental sustainability.

**NCO methods perform well on synthetic data but are disconnected from reality**: Neural combinatorial optimization learns heuristic policies via reinforcement learning and has achieved impressive results on synthetic VRP instances, but largely relies on simplified symmetric Euclidean distance data that fails to reflect the asymmetric nature of real road networks (one-way streets, traffic patterns, turn restrictions, etc.).

**Two root causes of the Sim-to-Real Gap**:
   - **Data level**: Training and testing use synthetic datasets (e.g., TSPLIB, CVRPLIB) that assume symmetric distances $d_{ij}=d_{ji}$, which does not hold in practice.
   - **Architecture level**: Existing NCO architectures are based on node-level Transformers, which are fundamentally ill-suited for efficiently handling edge features (asymmetric distance/duration matrices).

**Limitations of existing real-world datasets**: The few prior works in this space rely on commercial APIs, use static non-regenerable instances, are slow to query, and are often not publicly available; travel duration information is also typically absent.

**Insufficiency of existing edge feature encoding methods**: While MatNet's row/column embeddings and GOAL's cross-attention partially incorporate edge information, they generally handle only a single cost matrix and cannot integrate multimodal asymmetric features such as distance, duration, and bearing angle.

## Method

### Overall Architecture

RRNCO adopts an encoder–decoder architecture. The encoder constructs comprehensive node representations, and the decoder autoregressively generates routing solutions. The core innovations reside in the encoder and consist of two modules:

- **Adaptive Node Embedding (ANE)**: Fuses coordinate features with distance matrix information.
- **Neural Adaptive Bias (NAB)**: Provides learned asymmetric bias matrices for the Adaptation Attention-Free Module (AAFM).

### Key Designs

**1. Adaptive Node Embedding (ANE)**

- **Function**: Integrates edge features from the distance matrix with node coordinate features into a unified node representation.
- **Mechanism**: Applies probability-weighted sampling over the distance matrix combined with Contextual Gating for fusion.
- **Design Motivation**: Processing the full $N \times N$ distance matrix is computationally prohibitive, while using coordinates alone discards real distance information.

Specific steps:
- For distance matrix $\mathbf{D} \in \mathbb{R}^{N \times N}$, sample $k$ neighbor nodes with probability proportional to inverse distance: $p_{ij} = \frac{1/d_{ij}}{\sum_{j} 1/d_{ij}}$
- Sampled distances are projected linearly to obtain $\mathbf{f}_{\text{dist}}$; coordinates are projected separately to obtain $\mathbf{f}_{\text{coord}}$
- A gating weight is learned via MLP: $\mathbf{g} = \sigma(\text{MLP}([\mathbf{f}_{\text{coord}}; \mathbf{f}_{\text{dist}}]))$
- Fusion: $\mathbf{h} = \mathbf{g} \odot \mathbf{f}_{\text{coord}} + (1 - \mathbf{g}) \odot \mathbf{f}_{\text{dist}}$
- Row embeddings $\mathbf{h}^r$ and column embeddings $\mathbf{h}^c$ are generated (inspired by MatNet) to encode both directions of asymmetric relationships.

**2. Neural Adaptive Bias (NAB)**

- **Function**: Replaces the hand-crafted bias $A = -\alpha \cdot \log(N) \cdot d_{ij}$ in AAFM with a learned asymmetric bias matrix that fuses distance, bearing angle, and travel duration.
- **Mechanism**: Embeds each of the three edge feature types separately, fuses them via softmax gating, and outputs a scalar bias matrix.
- **Design Motivation**: Hand-crafted biases can only encode distance and cannot model directionality (one-way street effects) or the nonlinear relationship between duration and distance.

$$\mathbf{D}_{emb} = \text{ReLU}(\mathbf{D}\mathbf{W}_D)\mathbf{W}'_D$$
$$\mathbf{\Phi}_{emb} = \text{ReLU}(\mathbf{\Phi}\mathbf{W}_\Phi)\mathbf{W}'_\Phi$$
$$\mathbf{T}_{emb} = \text{ReLU}(\mathbf{T}\mathbf{W}_T)\mathbf{W}'_T$$

where $\phi_{ij} = \text{arctan2}(y_j - y_i, x_j - x_i)$ encodes bearing angle. The three embeddings are fused via softmax gating with a learnable temperature $\tau$, and projected to a scalar bias: $\mathbf{A} = \mathbf{H}\mathbf{w}_{out} \in \mathbb{R}^{B \times N \times N}$.

**3. AAFM (Adaptation Attention-Free Module)**

- Based on the attention-free module of Zhou et al. (2024a).
- Operation defined as: $\text{AAFM}(Q,K,V,A) = \sigma(Q) \odot \frac{\exp(A) \cdot (\exp(K) \odot V)}{\exp(A) \cdot \exp(K)}$
- The bias $\mathbf{A}$ produced by NAB is injected into this module, enabling it to be aware of asymmetric routing constraints.

### Loss & Training

- Policy gradient with REINFORCE and POMO baseline.
- Training objective: $\max_\theta J(\theta) = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \mathbb{E}_{\mathbf{a} \sim \pi_\theta(\cdot|\mathbf{x})} [R(\mathbf{a}, \mathbf{x})]$
- Reward $R$ is defined as the negative route cost.
- Training instances are sampled online from 100 cities using the OSRM engine, supporting efficient online data generation.

## Key Experimental Results

### Main Results

Performance on real-world routing benchmarks (ATSP task, 50 nodes):

| Method | In-Dist Cost | Gap(%) | OOD-City Cost | Gap(%) | Time |
|--------|-------------|--------|---------------|--------|------|
| LKH3 | 38.387 | *(best) | 38.903 | *(best) | 1.6h |
| POMO | 51.512 | 34.19 | 50.594 | 30.05 | 10s |
| MatNet | 39.915 | 3.98 | 40.548 | 4.23 | 27s |
| GOAL | 41.976 | 9.35 | 42.590 | 9.48 | 91s |
| **RRNCO** | **39.078** | **1.80** | **39.785** | **2.27** | **23s** |

ACVRP task (with capacity constraints):

| Method | In-Dist Cost | Gap(%) | OOD-City Cost | Gap(%) | Time |
|--------|-------------|--------|---------------|--------|------|
| PyVRP | 69.739 | *(best) | 70.488 | *(best) | 7h |
| MatNet | 74.801 | 7.26 | 75.722 | 7.43 | 30s |
| AAFM | 76.663 | 9.93 | 77.811 | 10.39 | 11s |
| **RRNCO** | **72.145** | **3.45** | **73.010** | **3.58** | **26s** |

### Ablation Study

| Configuration | ATSP Gap(%) | ACVRP Gap(%) |
|---------------|------------|-------------|
| Full RRNCO | 1.80 | 3.45 |
| w/o NAB (hand-crafted bias) | ~9.35 | ~9.93 |
| w/o ANE (coordinates only) | ~34.19 | ~23.16 |
| w/o bearing angle $\Phi$ | significant degradation | - |
| w/o duration matrix $T$ | significant degradation | - |

### Key Findings

1. **RRNCO achieves state-of-the-art NCO performance across all real-world tasks**: It consistently leads across ATSP/ACVRP/ACVRPTW tasks and across in-distribution, OOD-City, and OOD-Cluster settings.
2. **Minimal gap to classical solvers**: RRNCO is only 1.8% behind LKH3 on ATSP while being approximately 250× faster (23s vs. 1.6h).
3. **NAB is the critical innovation**: GOAL and AAFM with hand-crafted biases yield gaps of 9.35% and 19.81% respectively, compared to only 1.80% for RRNCO.
4. **Joint modeling of distance, duration, and bearing angle yields substantial gains**: Removing any single modality leads to significant performance degradation.
5. **Strong generalization**: Performance degrades only marginally under OOD-City and OOD-Cluster settings.

## Highlights & Insights

1. **First NCO work to jointly model distance, duration, and bearing angle**: The NAB mechanism is not only technically novel but also reveals the critical role of the coupling among these three factors in real-world routing quality.
2. **Probability-weighted sampling is an elegant solution for efficient distance matrix processing**: It avoids $O(N^2)$ full-matrix processing while retaining key asymmetric neighborhood information.
3. **Value of the open-source dataset**: The real-world dataset covering 100 cities combined with an online sampling framework provides a standardized real-world benchmark for future NCO research.
4. **Generality of Contextual Gating**: The gated fusion mechanism in both ANE and NAB is broadly applicable to other scenarios requiring the integration of heterogeneous features.

## Limitations & Future Work

1. **Only static routing is considered**: Dynamic traffic flow and real-time road condition changes are not addressed.
2. **Limited problem scale**: Experiments are primarily conducted at 50–100 nodes; scalability to large-scale instances (1000+) in real-world settings remains unverified.
3. **Dataset coverage**: While 100 cities is a notable improvement, coverage of diverse road types (rural roads, highways) can be further expanded.
4. **Gap to classical solvers**: The performance gap relative to LKH3/PyVRP may widen on large-scale instances.
5. **Multi-objective optimization**: Real-world logistics requires simultaneously optimizing distance, time, fuel consumption, and other objectives, whereas the current approach optimizes only a single cost function.

## Related Work & Insights

- **MatNet (Kwon et al., 2021)**: First introduced row/column embeddings to handle asymmetry; RRNCO's ANE extends this with distance sampling and gated fusion.
- **AAFM (Zhou et al., 2024a)**: Provides an efficient attention-free framework, but relies on hand-crafted biases; RRNCO's NAB upgrades this to a data-driven learned bias.
- **GOAL (Drakulic et al., 2024)**: Encodes edge information via cross-attention but handles only a single cost matrix.
- Insight: The sim-to-real gap is not unique to routing — it is pervasive in robot control, autonomous driving, and related fields. The dual strategy of data generation and architectural innovation in RRNCO offers broadly transferable value.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The NAB mechanism is the first to jointly model three asymmetric feature types in a conceptually clean and elegant manner; the probability-weighted sampling in ANE also offers a degree of originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three tasks × three distribution settings × multiple baselines, with comprehensive ablation studies and convincing real-city data.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear, architecture diagrams are intuitive, and experimental tables are rich; mathematical notation is dense but well-organized.
- **Value**: ⭐⭐⭐⭐⭐ — The open-source dataset and code provide the NCO community with its first real-world standard benchmark, advancing the field from toy problems toward practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](../../NeurIPS2025/optimization/rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[AAAI 2026\] Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](../../AAAI2026/optimization/bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[NeurIPS 2025\] Learning to Insert for Constructive Neural Vehicle Routing Solver](../../NeurIPS2025/optimization/learning_to_insert_for_constructive_neural_vehicle_routing_solver.md)

</div>

<!-- RELATED:END -->
