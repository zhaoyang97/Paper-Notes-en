---
title: >-
  [Paper Note] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning
description: >-
  [AAAI 2026][Graph Learning][Heterogeneous graph learning] Echoless-LP eliminates training label leakage (the echo effect) caused by multi-hop message passing in label pre-computation via Partition-Focused Echoless Propag…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Heterogeneous graph learning"
  - "label pre-computation"
  - "training label leakage"
  - "echo effect"
  - "memory efficiency"
date: 2026-05-08
content_hash: 347d13e6787f04c9
---

# EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning

**Conference**: AAAI 2026
**arXiv**: [2511.11081](https://arxiv.org/abs/2511.11081)
**Code**: [Available](https://github.com/CrawlScript/Echoless-LP)
**Area**: Graph Learning / Heterogeneous Graph Neural Networks
**Keywords**: Heterogeneous graph learning, label pre-computation, training label leakage, echo effect, memory efficiency

## TL;DR

Echoless-LP eliminates training label leakage (the echo effect) caused by multi-hop message passing in label pre-computation via Partition-Focused Echoless Propagation (PFEP). Combined with an Asymmetric Partition Scheme (APS) and a PostAdjust mechanism to address information loss and distribution shift introduced by partitioning, the method remains memory-efficient, is compatible with arbitrary message-passing operators, and achieves state-of-the-art performance on multiple heterogeneous graph benchmarks.

## Background & Motivation

**Background**: Heterogeneous Graph Neural Networks (HGNNs) are widely used for deep learning on heterogeneous graphs. End-to-end HGNNs (e.g., RGCN, HAN) require repeated message passing and are inefficient on large-scale graphs. Pre-computation methods perform message passing only once during preprocessing, generate regular tensors, and train an MLP in mini-batches, substantially improving efficiency.

**Limitations of Prior Work**: Label-based pre-computation methods suffer from training label leakage—during multi-hop message passing, a node's own label information propagates back to itself (the echo effect). At training time, the encoder relies on leaked self-label information, but test nodes carry no labels, leading to poor generalization.

**Root Cause of Existing Approaches**:
- **LastResidual-LP**: Emphasizes high-hop neighbor information to mitigate leakage, but echoes can still occur at high hops, providing only partial relief.
- **RemoveDiag-LP**: Removes the diagonal in the linear message-passing matrix to block echoes, but requires explicitly computing multi-hop propagation matrices, which can exceed TB-scale memory for $K > 2$, and is incompatible with complex message-passing methods such as RpHGNN.

## Method

### Overall Architecture

Echoless-LP consists of three operational stages:

1. **Partition**: Divide target nodes into multiple partitions.
2. **Partition-Focused Echoless Propagation (PFEP)**: Execute echoless propagation for each partition.
3. **Partition Output Merge**: Aggregate results from all partitions.

### Key Designs

**1. Partition-Focused Echoless Propagation (PFEP)**

**Core Idea**: When aggregating neighbor information for a given node, its input label vector is zeroed out, completely preventing self-label information from propagating back to itself.

**Implementation**: For each partition $P_i$, a partition mask $M_i$ is defined, and the message-passing input is replaced as follows:

$$H_i = F_{MP}(\text{diag}(1 - M_i) \cdot Y,\ G)$$

where $F_{MP}$ is an arbitrary message-passing operator. The mask $(1 - M_i)$ prevents the label information of nodes in the current partition from being used, achieving echoless propagation. A key advantage is that the message-passing operator itself is not modified, ensuring compatibility with any method.

**2. Asymmetric Partition Scheme (APS)**

Naive uniform random partitioning causes information loss, as each neighbor has a $1/M$ probability of being masked. APS is designed as follows:

- Training nodes are randomly and uniformly divided into $M$ partitions.
- Unlabeled nodes (validation/test) are placed into a separate $(M+1)$-th partition.
- When processing unlabeled nodes, only unlabeled nodes are masked (which carry no labels anyway), so no neighbor label information is lost.

**3. PostAdjust Mechanism**

The asymmetry of APS causes nodes in different partitions to accumulate neighbor information at different magnitudes. The solution is:

- Append a column of training-node indicators to the input.
- After message passing, extract the retention ratio $r_i$ and content $\bar{H}_i$.
- Normalize using the global maximum retention ratio to ensure all node representations are scaled to the same reference level.

**4. Integration with Feature Pre-Computation**

Echoless-LP operates in parallel with the feature pre-computation of backbone methods (NARS, SeHGNN, RpHGNN):

- Feature pre-computation collects $K_\text{feat}$ tensors.
- Label pre-computation collects $K$ tensors (via Echoless-LP).
- Both are concatenated and fed into an encoder (e.g., a hierarchical MLP) for classification.

**5. Complexity Analysis**

- **Space**: $O(F_{MP}) + O(NC)$, consistent with backbone methods.
- **Time**: $O(M \cdot F_{MP}) + O(NC)$; $M$ is typically 2–4, keeping overhead manageable.
- **Key comparison**: RemoveDiag-LP may run out of memory (>TB) for $K > 2$, whereas Echoless-LP maintains linear scaling.

### Loss & Training

The training strategy of the respective backbone method is used (cross-entropy loss with mini-batch training), with early stopping based on the validation set. Key hyperparameters include the number of label propagation hops $K \in [1, 8]$ and the number of partitions $M \in [2, 5]$.

## Key Experimental Results

### Main Results (Node Classification)

| Method | DBLP Macro-F1 | OGBN-MAG Acc | OAG-Venue NDCG | OAG-L1 NDCG |
|--------|--------------|-------------|----------------|-------------|
| RpHGNN (no label) | 95.23 | 52.07 | 53.31 | 87.80 |
| LastResidual(RpHGNN) | 95.27 | 45.88 | 49.65 | 87.48 |
| RemoveDiag(RpHGNN) | -- | -- | -- | -- |
| **Echoless(RpHGNN)** | **95.39** | **54.04** | **54.72** | **88.11** |

RemoveDiag-LP is incompatible with RpHGNN (marked as --), while Echoless-LP integrates seamlessly.

### Memory Efficiency Comparison (OAG-Venue)

| Hops $K$ | RemoveDiag-LP | Echoless-LP |
|----------|--------------|-------------|
| 2 | 60 GB | 60 GB |
| 3 | OOM (>4 TB) | 60 GB |
| 4 | OOM (>4 TB) | 60 GB |
| 5 | OOM (>4 TB) | 70 GB |

### Ablation Study

- **Removing PFEP**: Using labels directly in pre-computation leads to a notable performance drop due to the echo effect causing label leakage.
- **Removing APS**: Replacing the asymmetric partition with uniform random partitioning degrades performance due to information loss on unlabeled nodes.
- **Removing PostAdjust**: Skipping distribution shift correction leads to a performance drop.

### Key Findings

- $M = 2$ is optimal in most cases; increasing $M$ is not always beneficial, as a small degree of information loss may act as a regularizer.
- On large-scale graphs (millions of nodes), Echoless-LP efficiently supports high-hop propagation ($K > 2$), whereas RemoveDiag-LP runs out of memory.
- LastResidual-LP sometimes degrades backbone performance, as it does not fully resolve the echo problem and may exacerbate leakage.
- Echoless-LP achieves the most first- or second-place results across 6 datasets and 3 backbone methods.

## Highlights & Insights

- The partition masking idea is elegant and concise—by masking only at the input layer without modifying the message-passing operator itself, compatibility with arbitrary methods is achieved.
- APS processes training and unlabeled nodes separately, ensuring zero information loss for test nodes.
- The retention-ratio normalization in PostAdjust is a general technique applicable to other partitioning scenarios.
- The echo effect is completely eliminated rather than merely mitigated, fundamentally addressing the generalization problem.

## Limitations & Future Work

- Time complexity is $M$ times that of the backbone message passing; while $M$ is small (2–4), additional overhead remains on extremely large-scale graphs.
- The partition scheme does not account for graph structure (e.g., community structure); structure-aware intelligent partitioning could further reduce information loss.
- Validation is limited to node classification; performance on link prediction and graph classification tasks has not been explored.
- Training nodes still incur $1/M$ information loss under APS; while this may benefit regularization, it is not controllable.

## Related Work & Insights

- **SeHGNN**: A pre-computation method that collects neighbor information via meta-paths and the original proposer of RemoveDiag-LP.
- **RpHGNN**: A complex message-passing method using random projections and feature normalization; only Echoless-LP is compatible with it.
- **NARS**: A pre-computation method based on relational subgraph sampling; another backbone used by Echoless-LP.
- **SIGN**: A pioneering pre-computation method on homogeneous graphs that collects neighbor information separately by hop.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Practicality | 5 |
| Overall | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](posterior_label_smoothing_for_node_classification.md)
- [\[AAAI 2026\] Spiking Heterogeneous Graph Attention Networks](spiking_heterogeneous_graph_attention_networks.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[AAAI 2026\] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)

</div>

<!-- RELATED:END -->
