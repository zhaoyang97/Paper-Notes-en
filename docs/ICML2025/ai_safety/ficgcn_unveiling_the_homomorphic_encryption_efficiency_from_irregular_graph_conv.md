---
title: >-
  [Paper Note] FicGCN: Unveiling the Homomorphic Encryption Efficiency from Irregular Graph Convolutional Networks
description: >-
  [ICML 2025][AI Safety][Homomorphic Encryption] The FicGCN framework is proposed to address the fundamental conflict between the irregular sparsity of GCNs and the SIMD computation pattern of homomorphic encryption. By introducing three innovations—latency-aware packing, Sparse Intra-Ciphertext Aggregation (SpIntra-CA), and region-based node ordering—FicGCN achieves up to $4.10\times$ end-to-end acceleration on large-scale graphs such as Corafull.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Homomorphic Encryption"
  - "Graph Convolutional Networks"
  - "CKKS"
  - "Sparse Aggregation"
  - "Privacy-Preserving Inference"
date: 2026-05-08
content_hash: 84d383849fb0dbc1
---

# FicGCN: Unveiling the Homomorphic Encryption Efficiency from Irregular Graph Convolutional Networks

**Conference**: ICML 2025  
**arXiv**: [2506.10399](https://arxiv.org/abs/2506.10399)  
**Code**: None  
**Area**: Privacy-Preserving Computation / Homomorphic Encryption Inference  
**Keywords**: Homomorphic Encryption, Graph Convolutional Networks, CKKS, Sparse Aggregation, Privacy-Preserving Inference

## TL;DR

The FicGCN framework is proposed to address the fundamental conflict between the irregular sparsity of GCNs and the SIMD computation pattern of homomorphic encryption. By introducing three innovations—latency-aware packing, Sparse Intra-Ciphertext Aggregation (SpIntra-CA), and region-based node ordering—FicGCN achieves up to $4.10\times$ end-to-end acceleration on large-scale graphs such as Corafull.

## Background & Motivation

**Background**: GCNs are widely applied in privacy-sensitive scenarios such as healthcare and finance. The CKKS homomorphic encryption scheme allows direct computation on ciphertexts to protect privacy, but introduces a computation overhead of over three orders of magnitude.

**Limitations of Prior Work**: (1) The core bottleneck of GCNs under homomorphic encryption lies in rotation and multiplication operations during the matrix multiplication $\hat{A}XW$; (2) The GCN adjacency matrix $A$ is naturally sparse, but this irregular sparsity severely conflicts with the SIMD parallel mode of CKKS—sparse multiplication requires distinct aggregation per node, whereas SIMD requires uniform operations across the entire ciphertext vector; (3) The state-of-the-art solution CryptoGCN exploits sparsity but suffers from low ciphertext slot utilization, while Penguin achieves high packing efficiency but fails to leverage sparsity.

**Key Challenge**: The irregular node-wise computation pattern in the GCN aggregation phase vs. the SIMD collective operation requirement of CKKS ciphertexts—how to efficiently execute irregular sparse aggregation on fully packed ciphertexts?

**Goal**: Minimize inference latency under CKKS homomorphic encryption by simultaneously exploiting GCN sparsity and high ciphertext slot utilization.

**Key Insight**: Analogy-wise, intra-ciphertext rotation is treated similarly to the internal summation trick, enabling neighbors aggregation for all nodes in $O(\log N)$ rotations using bit decomposition.

**Core Idea**: Utilize logarithmic rotations within a ciphertext to achieve sparse aggregation, combined with aggregation-efficiency-optimized node ordering, to overcome the conflict between sparsity and SIMD.

## Method

### Overall Architecture

The workflow of FicGCN consists of three core modules: (1) **Latency-Aware Packing**: Optimizes the column packing number $t$ for each ciphertext based on data dimensions and the latency ratio of HE operations to achieve a global optimal balance between the aggregation and combination stages; (2) **SpIntra-CA**: Performs sparse neighbor aggregation within a ciphertext via bit-decomposition rotations, compressing $O(N)$ rotations down to $O(\log^2 N)$; (3) **NOO Node Ordering**: Reduces rotation overhead further through BFS-based region partitioning, interspersed region ordering, and greedy conflict minimization. During the inference phase, the optimal aggregation pattern (SpIntra-CA or Inter-CA) is automatically selected for each layer.

### Key Designs

1. **Sparse Intra-Ciphertext Aggregation (SpIntra-CA)**:

    - **Function**: Efficiently constructs "neighbor ciphertexts" via rotation operations within a ciphertext to achieve parallel aggregation of all nodes.
    - **Mechanism**: Performs bit decomposition on the rotation distance from each node to its neighbor's position, sequentially executing $2^0, 2^1, \ldots, 2^{\lfloor\log M\rfloor}$ step rotations from the lowest to the highest bit. Each step uses cleartext masks to select nodes that require or do not require rotation. Iteration formula: $\{ct\} \leftarrow \{ct\} \otimes Mask_1 \oplus Rot(\{ct\}, 2^{m-1}) \otimes Mask_2$. Ideally, it only requires $\log(M)$ rotations.
    - **Design Motivation**: Naive methods require extracting nodes one-by-one, resulting in $O(N)$ rotations where each rotation only affects a single node. Borrowing the bit-decomposition technique from intra-ciphertext summation waves, each rotation can simultaneously serve multiple nodes, reducing the efficiency complexity from $O(N)$ to $O(\log^2 N)$.

2. **Node Order Optimization (NOO)**:

    - **Function**: Optimizes the arrangement of nodes within the ciphertext to minimize the rotation distance and conflicts in SpIntra-CA.
    - **Mechanism**: A three-step pipeline—(1) BFS-based region partitioning (with a threshold TH restricting region size); (2) Interspersed region ordering to make each region a sub-ring within the ciphertext, maintaining rotation consistency; (3) Intra-region greedy ordering: sequentially selecting positions with the fewest conflicts against already fixed nodes.
    - **Design Motivation**: The arrangement of nodes in the ciphertext directly determines the rotation distance and conflict frequency—closer neighbor nodes lead to shorter rotations, and fewer conflicts within regions lead to fewer extra ciphertexts.

3. **Latency-Aware Packing and Aggregation Pattern Scheduling**:

    - **Function**: Automatically selects the optimal ciphertext packing parameters and aggregation mode layer-by-layer.
    - **Mechanism**: When $M > N \cdot F'$, the optimization objective is $\mathcal{J}(t;F,n) = 2\lceil F \cdot n/t \rceil + 20\lceil\log(t)\rceil$. Inter-CA ($2\lceil Fn/t \rceil$ Rot) and SpIntra-CA ($10cn\log^2(N)$ Rot) are compared layer-by-layer to select the one with lower latency.
    - **Design Motivation**: The latency of a Rotation operation is more than 20 times that of multiplication/addition, and the packing scheme directly impacts the number of rotations—global optimization is much more efficient than a static strategy.

### Loss & Training

FicGCN focuses on acceleration during the inference phase and does not alter the model training pipeline. The GCN model is trained in plaintext using the Adam optimizer (batch=64, lr=0.01, 200 epochs), with feature dimensions reduced from 32 to 16. CKKS parameters are configured for 128-bit security ($\Delta=2^{30}, M=2^{12}$). Latency for all experiments is evaluated on a single-threaded Intel i7-9750H.

## Key Experimental Results

### Main Results

End-to-end inference latency comparison (speedup factor is benchmarked against Gazelle):

| Dataset | Node Count | Gazelle | Penguin | CryptoGCN | FicGCN+NOO | Gain |
|--------|:------:|--------:|--------:|----------:|-----------:|:----:|
| Cora | 2,708 | 1535.3s | 128.8s | 131.1s | **64.1s** | 21.6× |
| Citeseer | 3,327 | 2897.3s | 142.9s | 150.4s | **80.0s** | 36.2× |
| Corafull | 19,793 | / | 35565s | 31735s | **7733s** | >120× |
| NTU | 25 | - | - | 1731.1s | **1373.8s** | 1.26× |

### Ablation Study

Contribution of each component on the Cora dataset:

| Component | Rot Count | Latency(s) | Notes |
|------|:-----:|:------:|------|
| Inter-CA Baseline | 0 | 70.08 | No sparsity utilization |
| SpIntra-CA (w/o AOO) | 7.04K | 47.65 | Effective sparse aggregation |
| + AOO | 5.74K | 40.39 | 18.5% rotation reduction |
| + CPOO | - | ≈34 | Further pruning rotation by 14-46% |
| + NOO | 1.93K | **69.28** (full pipeline) | Rotations reduced by another 66% |

### Key Findings

1. **More Significant Advantage on Large-Scale Graphs**: On Corafull (19,793 nodes), FicGCN is 4.1× faster than CryptoGCN and 4.6× faster than Penguin.
2. **Ciphertext Slot Utilization is Always 100%**: In contrast to CryptoGCN's 66-90%, FicGCN completely eliminates slot waste.
3. **NOO Outperforms Standard Ordering (Rabbit)**: Rotations are reduced from 2.85K to 1.93K, demonstrating dedicated optimization for the HE ring structure.
4. **No Loss in Model Accuracy**: The encrypted inference accuracy on Cora is 0.792 vs. 0.815 for the plaintext backbone, which is an acceptable gap.

## Highlights & Insights

- **Directly Addressing the Fundamental Conflict**: Irregular sparsity vs. SIMD is the core challenge of HE+GCN. The three proposed modules address conflicts at the packing, aggregation, and ordering levels, respectively.
- **Insight from Bit-Decomposition Rotation**: Generalizing the intra-ciphertext summation trick to GCN sparse aggregation, allowing a single rotation to serve multiple nodes.
- **Scalability**: The advantage is particularly prominent on large-scale graphs (with 20k nodes), as the $O(\log^2 N)$ complexity of SpIntra-CA is highly favorable for larger graphs.
- **End-to-End Optimization**: Instead of optimizing a single module, FicGCN performs global scheduling across packing $\times$ aggregation $\times$ ordering.

## Limitations & Future Work

- The framework has only been validated on GAE/STGCN architectures; more complex GNN variants (e.g., GAT, GIN) have not been tested.
- Evaluation was conducted on a single-threaded CPU, leaving the potential of GPU/multi-threaded parallelization unexplored.
- The greedy search of NOO may incur high computational overhead on extremely large-scale graphs (millions of nodes).
- No fair comparison has been made with MPC (Multi-Party Computation) schemes.
- The impact of non-linear layers ($x^2$ approximation) on model accuracy has not been analyzed thoroughly.

## Related Work & Insights

- **vs CryptoGCN**: Exploits sparsity but achieves only 66-83% ciphertext utilization—FicGCN simultaneously achieves 100% utilization and sparse acceleration.
- **vs Penguin**: Efficient packing but no sparsity exploitation—FicGCN complements the sparsity dimension via SpIntra-CA.
- **vs Gazelle**: Generic HE matrix multiplication—FicGCN is dedicatedly optimized for GCN, running 120×+ faster on Corafull.
- **Inspiration**: The ring structure of CKKS ciphertexts can be "adapted" to graph topologies through elaborately designed node ordering—marking a deep integration of cryptographic properties with computational graph topologies.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of bit-decomposition rotation in SpIntra-CA is highly novel, and the design of NOO tailored to the CKKS ring structure is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on four datasets, compared against four SOTA methods, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem analysis and well-structured diagrams to aid comprehension.
- Value: ⭐⭐⭐⭐ Offers practical acceleration benefits for privacy-preserving GCN inference, especially in large-scale graph scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](../../NeurIPS2025/ai_safety/influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[ECCV 2024\] Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](../../ECCV2024/ai_safety/unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)
- [\[ACL 2025\] CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](../../ACL2025/ai_safety/centaur_bridging_the_impossible_trinity_of.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
