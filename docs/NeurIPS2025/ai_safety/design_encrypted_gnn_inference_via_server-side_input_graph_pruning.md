---
title: >-
  [Paper Note] DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning
description: >-
  [NeurIPS 2025][AI Safety][Homomorphic Encryption] This paper proposes DESIGN, a framework that accelerates FHE-based GNN inference by approximately $2\times$ over the SEAL baseline through two-stage server-side optimizat…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Homomorphic Encryption"
  - "GNN Inference"
  - "Graph Pruning"
  - "Adaptive Polynomial Activation"
  - "Privacy-Preserving Computing"
date: 2026-05-08
content_hash: 5e897553adaef71c
---

# DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning

**Conference**: NeurIPS 2025  
**arXiv**: [2507.05649](https://arxiv.org/abs/2507.05649)  
**Code**: [https://github.com/LabRAI/DESIGN](https://github.com/LabRAI/DESIGN)  
**Area**: AI Security  
**Keywords**: Homomorphic Encryption, GNN Inference, Graph Pruning, Adaptive Polynomial Activation, Privacy-Preserving Computing

## TL;DR
This paper proposes DESIGN, a framework that accelerates FHE-based GNN inference by approximately $2\times$ over the SEAL baseline through two-stage server-side optimization—input graph pruning and adaptive polynomial activation degree allocation—while maintaining competitive accuracy.

## Background & Motivation

**Background**: GNNs have demonstrated strong performance in recommendation systems, bioinformatics, social networks, and other domains. However, processing sensitive graph data (user profiles, financial transactions, social relationships) poses serious privacy risks. Fully Homomorphic Encryption (FHE) enables direct computation over encrypted data and is thus an ideal privacy-preserving solution.

**Limitations of Prior Work**: Fundamental operations under FHE—especially homomorphic multiplication—are orders of magnitude slower than their plaintext counterparts. Existing FHE GNN methods (CryptoGCN, LinGCN, Penguin) focus primarily on optimizing internal model structure (e.g., sparse adjacency matrices, simplified activation functions), while overlooking redundancy in the input graph itself: not all nodes and edges are equally important.

**Key Challenge**: Three fundamental challenges exist: (a) plaintext graph metrics are inaccessible during online pruning in the encrypted domain; (b) existing methods apply uniform-degree polynomial activations across all nodes, where high-degree nodes inflate overall cost while low-degree nodes waste precision; (c) any optimization must be architecture-agnostic and generalizable.

**Goal**: To enable adaptive server-side processing of input graphs under FHE constraints by (a) logically pruning unimportant nodes and edges, and (b) assigning polynomial activations of varying degrees to nodes of differing importance.

**Key Insight**: Node degree is used as an FHE-friendly importance metric (requiring only addition), and multi-level importance masks are generated via approximate homomorphic comparison to guide pruning and adaptive activation assignment.

**Core Idea**: Compute importance scores from node degrees on the encrypted graph → generate multi-level masks via homomorphic comparison → drive graph pruning and adaptive polynomial activation degree allocation via the masks.

## Method

### Overall Architecture
The client encrypts the graph $\tilde{\mathcal{G}} = (\text{Enc}(\mathcal{V}), \text{Enc}(\mathcal{E}), \text{Enc}(\mathbf{X}))$ and uploads it to the server. The server performs two-stage processing: **Stage 1** FHE-compatible importance scoring and partitioning → **Stage 2** FHE GNN inference with pruning and adaptive activation. The encrypted result $\tilde{\mathbf{Y}}$ is returned to the client for decryption.

### Key Designs

1. **Encrypted Node Degree Computation**:

    - Function: Computes each node's degree on the encrypted adjacency matrix as an importance score.
    - Mechanism: $\tilde{s}_v = \bigoplus \tilde{A}_{vu}$, relying solely on HE.Add (and possibly HE.Rotate), entirely avoiding expensive homomorphic multiplication.
    - Design Motivation: Richer metrics such as feature norms require HE.Mult, whose overhead may negate the benefits of subsequent pruning. Node degree represents the optimal trade-off between FHE efficiency and informativeness.

2. **Homomorphic Partitioning and Mask Generation**:

    - Function: Partitions nodes into $m+1$ levels—Level 0 to be pruned, Levels 1–$m$ retained with varying precision.
    - Mechanism: Using $m$ predefined thresholds $\tau_1 > \tau_2 > \cdots > \tau_m$, binary masks are generated via the approximate homomorphic comparison operator HE.AprxCmp.
    - Pruning mask: $\tilde{M}_{0,v} \approx \tilde{\mathbf{1}}_v \ominus \texttt{HE.AprxCmp}(\tilde{s}_v, \tilde{\tau}_m)$
    - Level mask: $\tilde{M}_{i,v} \approx \texttt{HE.AprxCmp}(\tilde{s}_v, \tilde{\tau}_i) \odot (\tilde{\mathbf{1}}_v \ominus \texttt{HE.AprxCmp}(\tilde{s}_v, \tilde{\tau}_{i-1}))$
    - Cost: HE.AprxCmp requires evaluating high-degree comparison polynomials and constitutes the most compute-intensive stage; however, this one-time overhead is amortized over subsequent inference.

3. **Homomorphic Graph Pruning**:

    - Function: Uses the keep mask $(1 - M_0)$ to logically zero out the features of unimportant nodes and their associated edges.
    - Mechanism: $\tilde{\mathbf{X}}'_v = (\tilde{\mathbf{1}}_v \ominus \tilde{M}_{0,v}) \odot \tilde{\mathbf{X}}_v$, $\tilde{A}'_{uv} = (\tilde{\mathbf{1}}_u \ominus \tilde{M}_{0,u}) \odot (\tilde{\mathbf{1}}_v \ominus \tilde{M}_{0,v}) \odot \tilde{A}_{uv}$
    - Only a constant multiplicative depth overhead is incurred at the start of inference; all subsequent layers operate on the pruned graph.

4. **Adaptive Polynomial Activation**:

    - Function: Assigns polynomial approximation activations of varying degrees based on node importance level.
    - Mechanism: The activation output at layer $l$ is $\tilde{H}_v^{(l+1)} = \bigoplus_{i=1}^{m}(\tilde{M}_{i,v} \odot \texttt{HE.PolyEval}(P_{d_i}, \tilde{Z}_v^{(l+1)}))$
    - High-importance nodes use high-degree polynomials (e.g., degree 7) to preserve accuracy; low-importance nodes use low-degree polynomials (e.g., degree 2) to reduce cost.
    - Efficiency source: Evaluating $P_{d_i}$ requires $O(d_i)$ homomorphic multiplications; low-degree polynomials substantially reduce the average computation.

### Layer-wise GNN Computation
The homomorphic computation of a standard GNN layer is: $\tilde{\mathbf{Z}}^{(l+1)} = (\tilde{\mathbf{A}}' \otimes \tilde{\mathbf{H}}^{(l)} \otimes \mathbf{W}_1^{(l)}) \oplus (\tilde{\mathbf{H}}^{(l)} \otimes \mathbf{W}_2^{(l)}) \oplus \mathbf{b}^{(l)}$, where weights are in plaintext.

## Key Experimental Results

### Main Results (FHE GNN Inference Latency, in Seconds)

| Category | Method | Cora | Citeseer | PubMed | Yelp | ogbn-proteins |
|---|---|---|---|---|---|---|
| Basic FHE | SEAL | 1656.6 | 4207.6 | 528.9 | 321.1 | 14.2 |
| Basic FHE | OpenFHE | 813.7 | 3659.4 | 244.5 | 158.4 | 7.7 |
| Optimized FHE | CryptoGCN | 1035.8 | 2017.1 | 284.3 | 169.9 | 10.4 |
| Optimized FHE | LinGCN | 951.1 | 1850.4 | 260.8 | 155.9 | 9.5 |
| Optimized FHE | Penguin | 892.7 | 1928.1 | 249.7 | 162.3 | 9.1 |
| **Ours** | **DESIGN** | **806.1** | **1760.0** | **239.3** | **146.1** | **8.5** |

### Accuracy (Node Classification, %)

| Method | Cora | Citeseer | PubMed | Yelp | ogbn-proteins |
|---|---|---|---|---|---|
| GNN (Plaintext) | 84.0 | 60.0 | 80.0 | 56.8 | 59.2 |
| LinGCN | 72.5 | 52.1 | 58.3 | 48.2 | 58.2 |
| Penguin | 76.5 | 44.6 | 61.5 | 49.5 | 58.1 |
| **DESIGN** | 74.0 | 45.0 | 55.0 | **51.3** | 56.1 |

### Ablation Study (Cora Dataset)

| Configuration | Latency (s) | Accuracy (%) |
|---|---|---|
| Baseline FHE GNN (BFG) | ~1650 | ~68 |
| Pruning Only (PO) | ~1000 | ~72 |
| Adaptive Activation Only (AAO) | ~1200 | ~73 |
| **Full Framework (FF)** | **~806** | **74** |

### Key Findings
- Pruning and adaptive activation each contribute significantly in isolation, and their combination yields cumulative gains.
- A pruning ratio of 40% achieves the optimal latency–accuracy trade-off across most datasets.
- The polynomial degree set PSet2(5,3,2) is optimal under moderate fidelity: it is 30%+ faster than PSet1(7,5,3) with only a 2–3% accuracy drop.
- DESIGN achieves accuracy superior to some FHE baselines on Yelp and ogbn-proteins, suggesting that assigning low-degree polynomials to low-degree nodes avoids noise accumulation introduced by high-degree approximations.

## Highlights & Insights
- The choice of **node degree as an FHE-friendly importance metric** is highly practical—only addition operations are required, completely avoiding the noise accumulation and depth consumption of multiplication.
- The **dual optimization strategy** (pruning to reduce data volume + adaptive activation to reduce computational complexity) is mutually orthogonal, offering a flexible efficiency–accuracy knob.
- Migrating graph pruning from the plaintext domain to the encrypted domain represents a promising research direction.
- The framework is architecture-agnostic and can be applied as a plug-in to various GNN variants including GCN and GraphSAGE.

## Limitations & Future Work
- The homomorphic comparison stage (HE.AprxCmp) is itself compute-intensive and may partially offset pruning gains on large graphs.
- Using only node degree as an importance metric may fail to adequately capture task-relevant node importance.
- On some datasets, accuracy falls below Penguin (e.g., Citeseer: 45% vs. 44.6%), indicating that aggressive pruning can be harmful for certain graph structures.
- Accuracy variance on PubMed is large (±12.5%), and stability requires improvement.
- The combination with model compression techniques (e.g., knowledge distillation into FHE-friendly architectures) remains unexplored.

## Related Work & Insights
- **vs. CryptoGCN**: Optimizes only internal model structure (sparse adjacency + simplified activation) without addressing input redundancy; DESIGN optimizes both data and computation jointly.
- **vs. LinGCN/Penguin**: These methods employ improved polynomial approximation strategies but apply uniform degrees; DESIGN differentiates degree allocation based on node importance.
- **vs. Plaintext Graph Pruning**: Plaintext methods can directly access features for fine-grained pruning; the encrypted domain is constrained by FHE computability, necessitating the use of "cheap" metrics such as node degree.

## Rating
- Novelty: ⭐⭐⭐⭐ First to combine input graph pruning and adaptive activation for FHE GNN inference, though the core components are relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets, multiple baselines, complete ablation and sensitivity analyses.
- Writing Quality: ⭐⭐⭐ Notation is comprehensive but heavy; the main text is somewhat verbose.
- Value: ⭐⭐⭐⭐ Highly practical; opens an input-side optimization direction for FHE GNN inference efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)
- [\[NeurIPS 2025\] Robust Graph Condensation via Classification Complexity Mitigation](robust_graph_condensation_via_classification_complexity_mitigation.md)
- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[NeurIPS 2025\] Environment Inference for Learning Generalizable Dynamical System](environment_inference_for_learning_generalizable_dynamical_system.md)

</div>

<!-- RELATED:END -->
