---
title: >-
  [Paper Note] GraphFaaS: Serverless GNN Inference for Burst-Resilient, Real-Time Intrusion Detection
description: >-
  [NeurIPS 2025][Graph Learning][GNN Inference] This paper proposes GraphFaaS, a serverless inference architecture for GNN-based intrusion detection. Through incremental provenance graph construction…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "GNN Inference"
  - "Serverless"
  - "Intrusion Detection"
  - "Burst Traffic"
  - "Graph Partitioning"
date: 2026-05-08
content_hash: cca77ab3458e92dd
---

# GraphFaaS: Serverless GNN Inference for Burst-Resilient, Real-Time Intrusion Detection

**Conference**: NeurIPS 2025
**arXiv**: [2511.10554](https://arxiv.org/abs/2511.10554)
**Code**: None
**Area**: Graph Neural Networks / Systems
**Keywords**: GNN Inference, Serverless, Intrusion Detection, Burst Traffic, Graph Partitioning

## TL;DR

This paper proposes GraphFaaS, a serverless inference architecture for GNN-based intrusion detection. Through incremental provenance graph construction, feature-length-aware parallel node embedding, and greedy best-fit subgraph partitioning, GraphFaaS reduces mean detection latency from 14.16 seconds to 2.1 seconds (6.7×) and the coefficient of variation from 1.46 to 0.52 (64% reduction), maintaining stable low latency under bursty workloads without sacrificing detection accuracy.

## Background & Motivation

**Background**: Provenance graph-based intrusion detection systems (PIDS) represent an important application of graph machine learning in cybersecurity. System audit logs are modeled as directed acyclic provenance graphs — nodes represent system entities (processes, files, network sockets) and edges represent events (file reads/writes, process creation). GNNs learn graph patterns of normal behavior and detect anomalous nodes or graphs that deviate from these patterns at inference time.

**Limitations of Prior Work**: GNN-based intrusion detection faces a fundamental tension between two critical requirements: (1) consistently low latency, since excessive detection delay misses the attack response window with potentially irreversible consequences; and (2) handling highly irregular and bursty workloads, which frequently exhibit order-of-magnitude sudden spikes. Traditional static resource pre-allocation architectures cannot simultaneously satisfy both: reserving resources for peak loads wastes capacity, while provisioning for average loads causes latency spikes at peak time.

**Key Challenge**: Static resource allocation versus dynamic bursty workloads. Intrusion detection workloads are inherently imbalanced — malicious activity constitutes only a tiny fraction of network behavior, resulting in highly intermittent and unpredictable peak load patterns. Graph sizes also fluctuate dramatically during attacks.

**Goal**: How to maintain low detection latency and low latency variance for GNN-based intrusion detection under highly bursty workloads?

**Key Insight**: Leverage the elastic scaling capability of serverless computing — on-demand resource allocation, pay-per-use billing, and automatic scale-out/scale-in. The two stages of GNN inference (node embedding and message passing) are decomposed into fine-grained parallelizable execution units, with parallelism automatically regulated by the serverless platform.

**Core Idea**: Adapt the GNN intrusion detection inference pipeline to a serverless architecture, achieving stable low latency under bursty workloads through incremental graph construction, parallel node embedding, and adaptive graph partitioning.

## Method

### Overall Architecture

GraphFaaS consists of three main components: (1) incremental graph construction — exploiting temporal locality to avoid redundant computation; (2) serverless node embedding — parallelized conversion of text attributes to vectors; and (3) serverless GNN inference — parallel GNN message passing after subgraph partitioning. All three components are deployed on the OpenFaaS serverless platform and scale automatically based on load.

### Key Designs

1. **Incremental Graph Construction and Log Filtering**:

    - Function: Avoids reprocessing the entire provenance graph at each detection cycle, substantially reducing unnecessary computation.
    - Mechanism: Exploits the temporal locality property that most provenance graph structure remains unchanged between consecutive detection intervals, processing only the changed portions. Two-stage filtering is applied: (1) structural neighborhood filtering — retaining only nodes within 2K-hop distance of active nodes (K = number of GNN layers), since only these nodes participate in message passing; (2) frequency filtering — removing edges and nodes that appear frequently in training data (common patterns are generally non-anomalous) while retaining rare, low-frequency patterns. The filtered subgraph is then split into parallel subtasks.
    - Design Motivation: Provenance graphs can be very large, but most structure remains static over short intervals. Incremental processing reduces computation from the full graph to only the changed portion, which is a prerequisite for effective serverless elastic scaling.

2. **Feature-Length-Aware Serverless Node Embedding**:

    - Function: Converts each node's text attributes (process names, file paths, IP addresses, etc.) into numerical vectors as initial representations for the GNN.
    - Mechanism: Node embedding is inherently parallelizable (each node is independent) and is implemented as serverless functions. The key innovation is grouping nodes by feature length — short strings (e.g., IP addresses) are batched together into the same execution unit to reduce parallelization overhead, while long strings (e.g., full command lines) are processed individually to avoid timeouts. This grouping strategy ensures each execution unit's processing time remains within a preset threshold while avoiding the network transmission and packet processing overhead of excessive fragmentation. The serverless platform automatically scales based on the number of execution units.
    - Design Motivation: The execution time of embedding methods such as word2vec/doc2vec is positively correlated with string length; uniform batching causes long strings to slow down the entire batch. Length-based grouping achieves load balancing.

3. **Greedy Best-Fit Subgraph Partitioning with Vertical Scale-Up Fallback**:

    - Function: Partitions large graphs into balanced subgraphs so that each serverless function instance can complete GNN inference within the latency threshold.
    - Mechanism: Inference latency is primarily determined by graph size (with a fixed model). The greedy best-fit algorithm, analogous to bin packing, sorts nodes' K-hop neighborhoods in descending order of edge count and greedily places them into the bin with the closest remaining capacity, merging overlapping regions to minimize the total number of partitions. When even the smallest subgraph (the K-hop neighborhood of a single center node) exceeds the preset threshold — due to the dependency explosion problem in provenance graphs — a vertical scale-up is triggered, allocating more CPU and memory to that serverless instance rather than continuing to split.
    - Design Motivation: In provenance graphs, "super-nodes" (e.g., system-level processes) may have extremely large neighborhoods that cannot be resolved by horizontal partitioning. Vertical scale-up serves as a fallback mechanism for these extreme cases.

### Loss & Training

GraphFaaS does not alter the training procedure of the underlying GNN — it is an inference architecture optimization. The underlying GNN model (e.g., Flash PIDS) follows its standard training pipeline; GraphFaaS only parallelizes and elastically schedules computation at inference time.

## Key Experimental Results

### Main Results

Evaluated on the DARPA TC Engagement 3 dataset (11 days of audit logs, 4 attacks):

| Metric | GraphFaaS | Flash (Baseline) | Improvement |
|--------|-----------|------------------|-------------|
| Mean Detection Latency | **2.10s** | 14.16s | 6.7× reduction |
| Standard Deviation | **1.09** | 4498.92 | 4128× reduction |
| Coefficient of Variation (CV) | **0.52** | 1.46 | 64% reduction |
| Detection Accuracy | Same as Flash | - | No loss |
| Maximum Latency Spike | <10s | Far exceeds 10s | Significant improvement |

### Ablation Study

| Component | Effect | Description |
|-----------|--------|-------------|
| Incremental graph construction | Avoids full-graph reprocessing | Reduces computation via temporal locality |
| Feature-length-aware grouping | Balances embedding latency | Prevents long strings from slowing entire batch |
| Best-fit partitioning | Minimizes partition count | Reduces resource waste while guaranteeing latency |
| Vertical scale-up fallback | Handles super-nodes | Prevents latency spikes from dependency explosion |

### Key Findings

- **Latency stability is the most dramatic improvement**: Standard deviation drops from 4498.92 to 1.09, indicating that serverless elastic scaling effectively eliminates latency variance caused by bursty workloads.
- **Detection accuracy is completely unchanged**: GraphFaaS only optimizes the inference architecture without modifying the underlying model, so detection results are identical to the original Flash system.
- **Occasional latency spikes remain but are controllable**: Minor spikes still occur (from vertical scale-up for super-nodes), but maximum latency does not exceed 10 seconds — far better than the thousands-of-seconds range observed in the baseline.

## Highlights & Insights

- **Bringing serverless computing to GNN inference is a natural yet overlooked direction**: The bursty nature of intrusion detection workloads is a natural fit for the elastic scaling of serverless platforms. This architectural innovation requires no changes to model design yet delivers order-of-magnitude latency improvements.
- **The feature-length-aware grouping strategy is elegant**: It strikes a balance between parallelization granularity and communication overhead — excessively fine-grained parallelism is overwhelmed by communication costs, while coarse-grained parallelism fails to exploit elastic scaling. Grouping by string length is a simple yet effective heuristic.
- **The hybrid strategy of greedy best-fit partitioning and vertical scale-up**: Horizontal partitioning handles normal cases while vertical scale-up covers extreme cases (dependency explosion). This layered strategy is transferable to other graph inference systems.

## Limitations & Future Work

- **Preliminary results only**: The paper explicitly marks results as "Preliminary Results"; experiments are validated on a single dataset (DARPA TC) and compared against only one baseline (Flash).
- **Dependency explosion is not fundamentally resolved**: Vertical scale-up for super-nodes is a stopgap — individual serverless instances have resource ceilings (e.g., AWS Lambda's 10 GB memory limit), and timeouts may still occur in extreme cases.
- **Fixed number of GNN layers**: Provenance graph scale fluctuates dramatically during attacks, but a fixed-depth GNN cannot dynamically adjust its receptive field. The paper mentions dynamic GNN layer adjustment as future work but does not implement it.
- **Cold-start latency of serverless functions is not discussed**: Cold starts (instance creation on first invocation) typically incur hundreds of milliseconds to several seconds of latency, which may be problematic for real-time detection.
- **No evaluation across different serverless platforms**: Experiments are conducted solely on OpenFaaS; applicability to commercial platforms such as AWS Lambda and Google Cloud Functions is not discussed.
- **Cost analysis is absent**: Serverless platforms charge per invocation; the cost implications of bursty workloads are not addressed.

## Related Work & Insights

- **vs. Flash / Kairos and other traditional PIDS**: These systems use statically provisioned GNN inference that performs well under stable load but suffers latency spikes under bursty conditions. GraphFaaS resolves the elastic scaling problem through serverless adaptation.
- **vs. λGrapher**: λGrapher also explores serverless GNN serving but focuses on exploiting request-level graph locality and fine-grained resource control, without specifically addressing the bursty load characteristics of intrusion detection.
- **vs. Dorylus**: Dorylus applies serverless computing to GNN training (not inference), focusing on training cost efficiency rather than inference latency.
- **vs. GNNAdvisor**: GNNAdvisor optimizes GPU utilization but assumes static resources and cannot handle elastic scaling requirements.
- Insight: Serverless architectures can be generalized to other ML inference scenarios that must handle bursty workloads (e.g., real-time recommendation, anomaly detection); the key is identifying the appropriate task decomposition granularity.

## Rating

- Novelty: ⭐⭐⭐⭐ Adapting serverless computing to GNN-based intrusion detection inference is a novel and practical idea; the feature-length-aware grouping and hybrid scaling strategy are creative.
- Experimental Thoroughness: ⭐⭐⭐ Preliminary results demonstrate feasibility on a single dataset, but comparisons against more baselines and platforms are lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, architectural description is detailed, and the design motivation of all three components is well articulated.
- Value: ⭐⭐⭐⭐ The architectural innovation provides direct practical guidance for deploying GNN-based intrusion detection; the engineering value of a 6.7× latency reduction is significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Practical Bayes-Optimal Membership Inference Attacks](practical_bayes-optimal_membership_inference_attacks.md)
- [\[NeurIPS 2025\] GnnXemplar: Exemplars to Explanations -- Natural Language Rules for Global GNN Interpretability](gnnxemplar_exemplars_to_explanations_--_natural_language_rules_for_global_gnn_in.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[NeurIPS 2025\] Spatio-Temporal Directed Graph Learning for Account Takeover Fraud Detection](spatio-temporal_directed_graph_learning_for_account_takeover_fraud_detection.md)
- [\[ICLR 2026\] A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](../../ICLR2026/graph_learning/a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)

</div>

<!-- RELATED:END -->
