---
title: >-
  [Paper Note] HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos
description: >-
  [ICCV 2025][LLM Evaluation][Egocentric video understanding] This paper proposes HiERO, a weakly supervised hierarchical graph architecture that learns the hierarchy of functional activity cues by aligning video segments…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "Egocentric video understanding"
  - "hierarchical behavior modeling"
  - "graph neural networks"
  - "procedure learning"
  - "zero-shot reasoning"
date: 2026-05-08
content_hash: bc508a13d36bd2f4
---

# HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos

**Conference**: ICCV 2025
**arXiv**: [2505.12911](https://arxiv.org/abs/2505.12911)  
**Code**: [github.com/sapeirone/HiERO](https://github.com/sapeirone/HiERO)  
**Area**: LLM Evaluation
**Keywords**: Egocentric video understanding, hierarchical behavior modeling, graph neural networks, procedure learning, zero-shot reasoning

## TL;DR

This paper proposes HiERO, a weakly supervised hierarchical graph architecture that learns the hierarchy of functional activity cues by aligning video segments with narration text. The resulting segment features encode multi-scale behavioral dependencies. HiERO substantially outperforms fully supervised methods in zero-shot evaluation on procedure learning tasks (F1 +12.5% on EgoProceL) and achieves state-of-the-art performance on video–text alignment benchmarks.

## Background & Motivation

### Problem Definition

Human activities exhibit a natural hierarchical structure: low-level individual actions (e.g., cutting onions, peeling carrots) can be aggregated into mid-level activity cues (e.g., preparing vegetables), which in turn form high-level daily procedures (e.g., preparing dinner). Most existing egocentric video understanding methods focus on action-level understanding and overlook this intrinsic hierarchy.

### Limitations of Prior Work

**Action-level video–language alignment**: Methods such as EgoVLP align short video segments with individual narrations but ignore the temporal context and functional dependencies between actions.

**Limitations of procedure learning methods**:
- Supervised methods require extensive frame-level step annotations, which are prohibitively costly.
- Self-supervised methods rely on multiple video demonstrations of the same task for cross-video alignment, an overly strong assumption that is unsuitable for unscripted videos.
- Weakly supervised methods depend on external knowledge graphs such as wikiHow to provide explicit procedural supervision.

**Single-level aggregation**: Existing procedure learning methods consider only one level of aggregation (action → key step), failing to capture multi-level abstraction.

### Core Motivation

**Key insights**:
1. Functional activity cues can emerge naturally from unscripted videos without explicit supervision — actions that frequently co-occur can be naturally grouped into high-level activities through feature clustering.
2. Different feature extractors determine the abstraction level that clustering can capture, progressing from visual similarity → semantic similarity → functional similarity.
3. Encoding multi-scale functional dependencies via a hierarchical graph representation provides a strong inductive bias for diverse video understanding tasks.

## Method

### Overall Architecture

HiERO adopts an encoder–decoder architecture inspired by Graph U-Net:
1. **Temporal encoder**: Aggregates information within local temporal neighborhoods layer by layer to build multi-scale temporal representations.
2. **Function-aware decoder**: Discovers functionally similar node regions via spectral graph clustering and performs temporal reasoning independently within each partition.
3. **Training objectives**: Video–narration alignment loss + functional cue loss.

### Key Designs

#### 1. **Video Graph Representation and Temporal Encoder**

- **Function**: Represents a video as a graph and aggregates information within local temporal neighborhoods layer by layer.
- **Mechanism**: The input video $\mathcal{V}$ is encoded as a graph $\mathcal{G} = (\mathbf{X}, \mathcal{E}, \mathbf{p})$, where:
    - $\mathbf{X} \in \mathbb{R}^{N \times D}$: node embedding matrix, each node corresponding to a fixed-length video segment.
    - $e_{ij} \in \mathcal{E}$: an edge is added when the temporal distance between two nodes is smaller than threshold $\tau$.
    - $\mathbf{p} \in \mathbb{R}^N$: temporal positions of nodes (timestamps in seconds).

  The encoder consists of $N_l$ TDGC-based modules implementing temporally aware graph convolution:

  $$\mathbf{x}_i^{l+1} = \mathbf{W}_r^T \mathbf{x}_i^l + \text{mean}_{j \in \mathcal{N}(i)} (s_{ij} (\mathbf{w}_{ij} \odot \mathbf{x}_j')) + \mathbf{b}_r$$

  where $s_{ij} = \text{sign}(p_i^l - p_j^l)$ encodes temporal direction and $\mathbf{w}_{ij} = \text{MLP}(|p_i^l - p_j^l|)$ modulates weights according to temporal distance. After each layer, nodes are temporally subsampled to halve the resolution.

- **Design Motivation**: The temporal encoder progressively expands each node's temporal context regardless of whether related actions are temporally contiguous, providing multi-scale temporal features for subsequent functional cue discovery.

#### 2. **Function-Aware Decoder and Cut & Match Module**

- **Function**: Discovers and exploits relationships between functionally similar nodes that may be temporally distant.
- **Mechanism**:
  1. Receives features from the corresponding encoder layer (via skip connections) and the output of the previous decoder layer.
  2. Applies the **Cut & Match module** for spectral graph clustering:
     - Constructs a similarity matrix based on cosine similarity between node embeddings: $S_{ij} = \exp(\mathbf{x}_i^T \mathbf{x}_j / (\kappa \|\mathbf{x}_i\|_2 \|\mathbf{x}_j\|_2))$
     - Computes the normalized Laplacian: $\tilde{\mathbf{L}}_S = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{S} \mathbf{D}^{-1/2}$
     - Performs eigendecomposition of $\tilde{\mathbf{L}}_S$ and retains the eigenvectors corresponding to the $K$ smallest eigenvalues.
     - Applies K-Means clustering in the eigenvector subspace to partition nodes into $K$ clusters.
  3. Performs TDGC temporal reasoning independently within each cluster, then maps the result back to the original graph.

- **Design Motivation**: Spectral graph clustering requires no distributional assumptions and groups nodes directly by graph connectivity. Performing temporal reasoning within each functional partition enables functionally related but temporally distant actions (e.g., interleaved activities) to exchange information.

#### 3. **Training Objectives: Video–Narration Alignment and Functional Cue Loss**

- **Function**: Learns a feature space in which co-occurring actions are close and actions from different activity cues are distant.

**Video–narration alignment loss $\mathcal{L}_{vna}$**:
- Unlike EgoVLP, which aligns individual segments to individual narrations, each node is aligned to all narrations within its temporal window (positive samples) while being repelled from narrations outside the window and from other videos (negative samples).
- Positive set: $\mathcal{P}_j = \{(n,t) \in \mathcal{T}_i : |p - t| \leq 2^\alpha\}$
- Negative set: $\mathcal{N}_j = \{(n,t) \in \mathcal{T}_i : 2^\alpha < |p - t| \leq 2^\beta\} \cup \mathcal{T}_{k, k \neq i}$

$$\mathcal{L}_{v2t} = \frac{1}{B} \sum_{\mathbf{v}_j} \frac{\sum_{n \in \mathcal{P}_j} \exp(h_v(\mathbf{v}_j)^T h_t(\mathcal{F}(n)) / \tau)}{\sum_{n \in \mathcal{P}_j \cup \mathcal{N}_j} \exp(h_v(\mathbf{v}_j)^T h_t(\mathcal{F}(n)) / \tau)}$$

**Functional cue loss $\mathcal{L}_{ft}$**:
- Leverages the cluster assignments from the Cut & Match module in the decoder to encourage nodes in the same cluster to be closer in feature space:

$$\mathcal{L}_{ft} = \sum_{k=1}^{K} \sum_{c_i=k} \sum_{c_j=c_i} \frac{\exp(h_v(\mathbf{v}_i)^T h_v(\mathbf{v}_j) / \tau)}{\sum_{j'} \exp(h_v(\mathbf{v}_i)^T h_v(\mathbf{v}_{j'}) / \tau)}$$

Total loss: $\mathcal{L} = \mathcal{L}_{vna} + \mathcal{L}_{ft}$

### Loss & Training

- Trained on EgoClip (3.8M segment–text pairs) for 15 epochs.
- Batch size 8, learning rate $1 \times 10^{-5}$, linear warm-up for 5 epochs followed by cosine annealing.
- Training requires fewer than 20 GPU hours.
- Compatible with multiple backbone networks: Omnivore, EgoVLP, LaViLa.
- Zero-shot inference: procedure steps are detected directly by clustering on decoder outputs.

## Key Experimental Results

### Main Results

**EgoProceL Procedure Learning Benchmark (Zero-Shot vs. Fully Supervised)**:

| Method | Supervision | Avg F1 | Avg IoU |
|--------|-------------|--------|---------|
| CnC | Self-supervised (cross-video alignment) | 22.0 | 10.7 |
| GPL | Self-supervised | 25.6 | 13.9 |
| OPEL | Self-supervised | 32.0 | 16.3 |
| EgoVLP feature clustering | Zero-shot | 40.0 | 21.9 |
| **HiERO (EgoVLP)** | **Zero-shot** | **44.5** | **25.3** |

**EgoMCQ Video–Text Alignment**:

| Method | Inter Acc (%) | Intra Acc (%) |
|--------|--------------|--------------|
| EgoVLP | 90.6 | 57.2 |
| LaViLa | 94.5 | 63.1 |
| **HiERO (LaViLa)** | **94.6** | **64.4** |

**Ego4D Goal-Step Step Grounding (Avg mAP)**:

| Method | Supervision | Avg mAP |
|--------|-------------|---------|
| EgoOnly | Supervised | 13.6 |
| EgoVLP | Zero-shot | 8.3 |
| **HiERO (EgoVLP)** | **Zero-shot** | **8.7** |

### Ablation Study

**Contribution of Each HiERO Component (EgoVLP backbone)**:

| Alignment Loss | Functional Cue Clustering | Functional Cue Loss | EgoMCQ Intra | EgoProceL F1 | Step-Grounding R@1 |
|---------------|--------------------------|--------------------|--------------|--------------|--------------------|
| ✗ | ✗ | ✗ | 57.2 | 40.0 | 10.73 |
| ✓ | ✗ | ✗ | 59.5 | 43.8 | 11.27 |
| ✓ | ✓ | ✗ | 59.6 | 43.3 | 11.44 |
| ✓ | ✓ | ✓ | **59.6** | **44.5** | **11.57** |

### Key Findings

1. **Zero-shot substantially outperforms fully supervised methods**: On EgoProceL, zero-shot HiERO achieves 44.5% F1, substantially surpassing the best self-supervised method OPEL at 32.0% (+12.5%), even though OPEL requires multiple video demonstrations of the same task.
2. **Functional patterns emerge naturally**: Procedure steps can be discovered from feature space via unsupervised clustering without any step annotations.
3. **Window-level alignment outperforms single-narration alignment**: The windowed alignment strategy of $\mathcal{L}_{vna}$ yields more context-aware features, significantly improving procedure learning F1 (+3.8%) and EgoMCQ Intra (+2.3%).
4. **Functional cue loss guides better clustering**: $\mathcal{L}_{ft}$ encourages intra-cluster feature cohesion, enabling clustering to more accurately capture high-level activity patterns.
5. **Consistent gains across backbones**: Improvements are observed across all three backbones (Omnivore, EgoVLP, LaViLa), demonstrating the generality of HiERO.

## Highlights & Insights

1. **Hierarchical structure emerges without explicit supervision**: This is the paper's central insight — functional cues at different scales are discovered through graph clustering without any procedural annotations.
2. **From short-term temporal alignment to long-range functional reasoning**: The encoder handles local temporal reasoning, while the decoder uses Cut & Match to connect temporally distant but functionally related segments, enabling genuine long-range reasoning.
3. **One model, multiple tasks**: HiERO achieves strong performance across video–text alignment (EgoMCQ/EgoNLQ), procedure learning (EgoProceL), and step grounding (Goal-Step).
4. **Elegant application of spectral graph clustering**: Modeling video understanding as a graph partitioning problem leverages the mathematical foundations of spectral graph theory to provide a strong inductive bias.
5. **Extremely lightweight training**: The full model can be trained in fewer than 20 GPU hours.

## Limitations & Future Work

1. **Cluster count K must be specified in advance**: The current design requires presetting the number of clusters at each layer; adaptive determination of K would further improve practicality.
2. **Fixed subsampling strategy**: The temporal subsampling rate in the encoder is fixed at 2×, which may be suboptimal for activities at different temporal scales.
3. **Reliance on pre-extracted features**: HiERO operates on pre-extracted video features rather than being trained end-to-end, which may limit its potential.
4. **Step granularity ambiguity**: Qualitative analysis indicates that some failure cases stem from granularity ambiguity in step annotations (confusion between steps and sub-steps).
5. **Limited to egocentric video**: Training and evaluation are conducted solely on Ego4D; transferability to third-person video remains unexplored.

## Related Work & Insights

- **vs. HierVL**: HierVL employs video-level summaries for hierarchical alignment, requiring additional high-level annotations; in HiERO, the hierarchical structure emerges entirely from the data.
- **vs. Paprika**: Paprika relies on external procedural knowledge graphs from wikiHow; HiERO requires no external knowledge sources.
- **vs. TW-FINCH**: TW-FINCH demonstrates that clustering is a strong baseline for action segmentation; HiERO further shows that hierarchical clustering can capture more complex functional patterns.
- Viewing videos through a graph-theoretic lens can inspire new method designs for a broader range of temporal reasoning tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of having hierarchical functional cues emerge naturally from unscripted videos is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across multiple benchmarks, backbones, and tasks with clear ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is presented vividly (kitchen example), technical descriptions are rigorous, and figures are informative.
- **Value**: ⭐⭐⭐⭐⭐ — Zero-shot results surpassing fully supervised methods are highly impressive, opening a new paradigm for procedure learning and video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](../../NeurIPS2025/llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](../../NeurIPS2025/llm_evaluation/ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)
- [\[ICLR 2026\] MOSIV: Multi-Object System Identification from Videos](../../ICLR2026/llm_evaluation/mosiv_multi-object_system_identification_from_videos.md)
- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)
- [\[ACL 2026\] Idiom Understanding as a Tool to Measure the Dialect Gap](../../ACL2026/llm_evaluation/idiom_understanding_as_a_tool_to_measure_the_dialect_gap.md)

</div>

<!-- RELATED:END -->
