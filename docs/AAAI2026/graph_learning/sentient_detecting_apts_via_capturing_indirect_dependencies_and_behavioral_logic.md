---
title: >-
  [Paper Note] Sentient: Detecting APTs Via Capturing Indirect Dependencies and Behavioral Logic
description: >-
  [AAAI 2026][Graph Learning][APT Detection] This paper proposes Sentient, an APT detection method combining **Graph Transformer pre-training** and **bidirectional Mamba2 intent analysis**. Trained exclusively on benign da…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "APT Detection"
  - "Provenance Graph"
  - "Graph Transformer"
  - "Mamba"
  - "Behavioral Intent Analysis"
date: 2026-05-08
content_hash: 0a42d2abd2e310a7
---

# Sentient: Detecting APTs Via Capturing Indirect Dependencies and Behavioral Logic

**Conference**: AAAI 2026
**arXiv**: [2502.06521](https://arxiv.org/abs/2502.06521)  
**Code**: None  
**Area**: Graph Learning / Cybersecurity
**Keywords**: APT Detection, Provenance Graph, Graph Transformer, Mamba, Behavioral Intent Analysis

## TL;DR

This paper proposes Sentient, an APT detection method combining **Graph Transformer pre-training** and **bidirectional Mamba2 intent analysis**. Trained exclusively on benign data, it captures indirect dependencies, removes contextual noise, and correlates behavioral logic, achieving an average 44% reduction in false positive rate across three standard benchmarks.

## Background & Motivation

1. **Background**: Advanced Persistent Threats (APTs) are notoriously difficult to detect due to their stealthiness and complexity. Provenance graph-based methods represent the current state of the art, leveraging entity relationships in system audit logs to uncover attack traces.
2. **Limitations of Prior Work**: (a) **Missing indirect dependencies** — GNN-based methods are constrained by the receptive field of neighborhood aggregation, failing to capture relationships between non-directly connected nodes; (b) **Noise in complex scenarios** — infected entities continue to perform numerous benign tasks, causing neighborhood aggregation to erroneously incorporate weakly related activities; (c) **Missing behavioral logic correlation** — isolated system behaviors exhibit contextual ambiguity (e.g., sshd writing a log appears normal in isolation), yet their combination reveals malicious intent.
3. **Key Challenge**: GNN local aggregation cannot reach indirect dependencies, introduces noise through indiscriminate neighbor aggregation, and is unable to establish logical correlations between distant behaviors.
4. **Goal**: Design a globally aware APT detection method capable of understanding behavioral logic.
5. **Key Insight**: Employ global attention in Graph Transformers to capture indirect dependencies, construct denoised behavior sequences via random walks, and leverage bidirectional Mamba2 to mine logical correlations among behaviors.
6. **Core Idea**: Graph Transformer for global node embeddings + bidirectional Mamba2 for intent logic over behavior sequences = addressing the three challenges of indirect dependencies, noise, and logical correlation.

## Method

### Overall Architecture

Five components: (1) **Graph Construction** — builds a provenance graph from system logs, initializing nodes with Word2Vec semantic encoding and Laplacian positional encoding; (2) **Pre-training** — a Graph Transformer reconstructs key node information to learn globally structured semantic embeddings; (3) **Intent Analysis Module (IAM)** — random walks construct behavior sequences, and bidirectional Mamba2 mines logical correlations; (4) **Threat Detection** — an MLP reconstructs behavioral actions, with behaviors whose reconstruction error exceeds a threshold flagged as malicious; (5) **Attack Investigation** — clusters behaviors with similar intent.

### Key Designs

1. **Graph Transformer Pre-training**

    - **Function**: Learns global node embeddings that capture indirect dependencies, circumventing the receptive field limitations of GNNs.
    - **Mechanism**: The initial embedding $h_i^0 = \sigma((A^0\alpha + a^0) + (B^0\beta + b^0))$ combines semantic encoding $\alpha$ (Word2Vec) and positional encoding $\beta$ (Laplacian eigenvectors). Multi-head attention allows each node to attend to all others in the graph ($w_{ij} = \text{softmax}(Q h_i \cdot K h_j / \sqrt{d_k})$), with residual connections and FFN producing final embeddings. The pre-training objective is node type reconstruction (weighted cross-entropy to handle class imbalance).
    - **Design Motivation**: Attack behaviors in provenance graphs involve multi-hop relationships (e.g., file read → execution → network transmission), which GNNs require multiple layers to reach — incurring over-smoothing in deep settings. The global attention of Graph Transformers resolves this in a single pass.

2. **Intent Analysis Module (IAM)**

    - **Function**: Mines logical correlations among behaviors in a denoised context to understand behavioral intent.
    - **Mechanism**: Using pre-trained embeddings $h$, random walks over the provenance graph construct behavior sequences $\lambda_i = \{e_1, ..., e_W\}$, where each behavior $e_t$ is represented as the concatenation of source and target node embeddings $[h_{\phi(e_t)}; h_{\psi(e_t)}]$. Random walks naturally build a target-node-centric local context, filtering irrelevant neighbors (denoising). **Bidirectional Mamba2** then processes the sequence: $\lambda^{\ell+1} = \mathbf{F}(\mathbf{E}(\lambda^\ell) + \mathcal{R}(\mathbf{E}(\mathcal{R}(\lambda^\ell))), \lambda^\ell)$, where $\mathcal{R}$ denotes sequence reversal and $\mathbf{E}$ denotes the Mamba2 state space model operation. Bidirectional processing ensures both forward and backward contextual logic are captured.
    - **Design Motivation**: Isolated behaviors appear benign but reveal malicious intent only in combination. Mamba2's long-sequence modeling capability surpasses RNNs, and its linear complexity suits large-scale log processing. Bidirectional modeling is necessary because attack behaviors may depend on both preceding and subsequent context.

3. **Threat Detection and Attack Investigation**

    - **Function**: Detects anomalies based on deviation from benign patterns and clusters attack behaviors into attack narratives.
    - **Mechanism**: During training, key behavioral information (read/write/execute) is masked to learn benign behavioral reconstruction patterns. During inference, behaviors whose reconstruction error $RE = \text{CrossEntropy}(\mathbf{P}(a_t), L(a_t))$ exceeds the threshold (mean + 1.5 standard deviations) are flagged as malicious. For attack investigation, the concatenation of behavioral intent embedding $h_e$ with source/target node embeddings is clustered as $C_k = \{e_i | \arg\min_k \|h_{behavior}^{(i)} - \mu_k\|^2\}$, merging alerts with similar intent to reduce analyst burden.
    - **Design Motivation**: Training solely on benign data avoids the scarcity of attack samples. Reconstruction error naturally quantifies the degree of behavioral anomaly.

### Loss & Training

The pre-training loss is weighted cross-entropy (node type reconstruction); the detection loss is cross-entropy (behavior type reconstruction). The anomaly threshold is set to mean + 1.5 standard deviations computed over the training period.

## Key Experimental Results

### Main Results

Results on Streamspot, Unicorn Wget, and DARPA E3 datasets:

| Dataset | Method | Precision | Recall | F-score | FPR |
|--------|------|-----------|--------|---------|-----|
| Streamspot | Threatrace | 98% | 99% | 98% | 0.4% |
| Streamspot | **Sentient** | **99%** | **99%** | **99%** | **0.2%** |
| Unicorn Wget | Threatrace | 93% | 98% | 95% | 7.4% |
| Unicorn Wget | **Sentient** | **96%** | **99%** | **97%** | **4.1%** |
| DARPA Cadets | Flash | 92% | 99% | 95% | 0.3% |
| DARPA Cadets | Slot | 94% | 96% | 95% | 0.2% |
| DARPA Cadets | **Sentient** | **96%** | **99%** | **97%** | **0.2%** |
| DARPA Theia | Flash | 91% | 99% | 95% | 0.8% |
| DARPA Theia | **Sentient** | **95%** | **99%** | **97%** | **0.4%** |
| DARPA Trace | Flash | 93% | 99% | 96% | 0.4% |
| DARPA Trace | **Sentient** | **97%** | **99%** | **98%** | **0.2%** |

### Ablation Study

| Configuration | Precision Change | Notes |
|------|---------------|------|
| w/o Pre-training (PT) | −20.75% | Loss of indirect dependency information |
| w/o Intent Analysis (IAM) | −31.59% | Loss of behavioral logic correlation; largest impact |
| w/o Laplacian PE | −8.2% | Loss of topological positional information |
| w/o Semantic Encoding | −12.3% | Loss of node attribute semantics |

### Key Findings

- IAM contributes the most — removing it causes a 31.59% drop in precision, underscoring the critical importance of behavioral logic correlation for APT detection.
- Advantages are most pronounced in complex scenarios (Unicorn Wget, DARPA Theia), where noise and indirect dependencies are more prevalent.
- Achieving state-of-the-art detection using only benign training data is a significant practical advantage for real-world deployment.
- Runtime overhead is acceptable: processing one day of logs requires only 63.6 seconds with a peak memory footprint of 2.01 GB.

## Highlights & Insights

- **Graph Transformer + Sequential SSM combination**: Using Graph Transformer for global representation and Mamba2 for sequential logic correlation addresses long-range dependencies at both the graph and sequence levels. This combination strategy is transferable to other graph-plus-sequence tasks.
- **Random walk as a denoising mechanism**: Random walks naturally construct a target-node-centric context that filters irrelevant neighbors — an elegant denoising design.
- **Clustering for attack investigation**: Beyond anomaly detection, clustering behaviors with similar intent into "attack stories" substantially reduces the workload of security analysts.

## Limitations & Future Work

- The anomaly threshold (mean + 1.5σ) is heuristically defined; an adaptive threshold may yield better results.
- The random walk sequence length $W$ is fixed; adaptive length selection could offer greater flexibility.
- Robustness under concept drift (i.e., evolving system behavior patterns over time) has not been evaluated.
- The clustering method for attack investigation is relatively simple (K-means); more sophisticated clustering approaches could generate higher-quality attack narratives.

## Related Work & Insights

- **vs. Flash/Threatrace**: Employ GNN (GraphSAGE) neighborhood aggregation, which fails to capture indirect dependencies and introduces noise. Sentient addresses this via global attention in Graph Transformers.
- **vs. Slot**: Uses graph reinforcement learning to adaptively select neighbors, but remains constrained by the GNN receptive field. Sentient bypasses the neighborhood aggregation paradigm entirely.
- **vs. Atlas**: Requires attack data for training; Sentient requires only benign data.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of Graph Transformer, bidirectional Mamba2, and random-walk-based denoising is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets covering real and simulated attacks; complete ablation study
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; challenges are illustrated intuitively with figures
- Value: ⭐⭐⭐⭐ Offers practical deployment value for real-world cybersecurity

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](../../ICLR2026/graph_learning/structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)

</div>

<!-- RELATED:END -->
