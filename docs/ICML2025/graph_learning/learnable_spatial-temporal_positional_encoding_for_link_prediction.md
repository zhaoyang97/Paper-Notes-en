---
title: >-
  [Paper Note] L-STEP: Learnable Spatial-Temporal Positional Encoding for Link Prediction
description: >-
  [ICML 2025][Graph Learning][Temporal Graphs] L-STEP is proposed as a lightweight temporal link prediction model based on learnable spatial-temporal positional encoding. It captures the temporal evolution of positional encoding through Discrete Fourier Transform, replacing Transformer attention mechanisms with MLPs to achieve SOTA performance with faster execution.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Temporal Graphs"
  - "Positional Encoding"
  - "Link Prediction"
  - "Discrete Fourier Transform"
  - "Lightweight Architecture"
date: 2026-05-08
content_hash: 7a65d127a46c981c
---

# L-STEP: Learnable Spatial-Temporal Positional Encoding for Link Prediction

**Conference**: ICML 2025  
**arXiv**: [2506.08309](https://arxiv.org/abs/2506.08309)  
**Code**: [https://github.com/kthrn22/L-STEP](https://github.com/kthrn22/L-STEP)  
**Area**: Graph Learning / Temporal Link Prediction  
**Keywords**: Temporal Graphs, Positional Encoding, Link Prediction, Discrete Fourier Transform, Lightweight Architecture

## TL;DR
L-STEP is proposed as a lightweight temporal link prediction model based on learnable spatial-temporal positional encoding. It captures the temporal evolution of positional encoding through Discrete Fourier Transform, replacing Transformer attention mechanisms with MLPs to achieve SOTA performance with faster execution.

## Background & Motivation

**Background**: Temporal link prediction (determining if a connection between two nodes exists at a specific time) is a core task in graph learning. Graph Transformers combined with positional encoding (PE) have become the SOTA approach, but they rely on $O(n^2)$ or $O(n^3)$ attention mechanisms.

**Limitations of Prior Work**: (1) Most PE methods use predefined, fixed functions (such as Laplacian eigenvectors) that cannot adapt to complex attributed graphs; (2) The only existing learnable PE (Dwivedi et al.) only considers static structural information and does not handle temporal evolution; (3) PE is coupled with Transformer attention, leading to extremely high computational costs on large-scale graphs.

**Key Challenge**: The expressiveness of PE is required to encode the structural position information of the graph, but the high complexity of Transformer attention must be avoided—existing solutions cannot achieve both.

**Goal**: To design a learnable spatial-temporal positional encoding that enables a simple MLP architecture to achieve or even exceed the temporal link prediction performance of Graph Transformers.

**Key Insight**: Treat the node positional encoding as a signal sequence evolving over time, leveraging the Discrete Fourier Transform (DFT) to capture its temporal dependencies in the frequency domain, thereby "predicting" the positional encoding at the current time step.

**Core Idea**: Infer the current positional encoding from historical positional encoding sequences using DFT and a learnable frequency-domain filter, and then aggregate node, edge, and positional information with an MLP for link prediction, completely removing dependency on attention mechanisms.

## Method

### Overall Architecture
L-STEP predicts whether a link exists between nodes $u$ and $v$ at each time step $t$. The pipeline consists of: (1) estimating the current positional encoding $\tilde{p}_u^t$ from historical positional encodings using the LPE module; (2) aggregating node features, edge features, and positional encodings using the Node-Link-Positional Encoder to obtain node representations; (3) passing the representations of both nodes into an MLP to predict link existence.

### Key Designs

1. **Learnable Positional Encoding Module (LPE)**:

    - **Function**: Infers the positional encoding of the current time step from the historical positional encoding sequence of the node.
    - **Mechanism**: Takes the node $u$'s positional encodings from the most recent $L$ time steps $\{p_u^{t'_1}, \ldots, p_u^{t'_L}\}$, applies DFT to transform them into the frequency domain, uses a learnable complex-valued filter $W_{filter} \in \mathbb{C}^{d_P \times L}$ for frequency domain filtering (denoising + feature extraction), applies IDFT to transform back to the time domain, and finally uses weighted sum pooling to obtain the estimated $\tilde{p}_u^t$. After prediction, the stored positional encoding is updated with the actual topology of the current time step.
    - **Design Motivation**: Avoids recomputing the Laplacian eigendecomposition ($O(|V|^3)$) at each time step, while preserving temporal evolution. It is theoretically proven from a spectral perspective that this scheme preserves the spatial-temporal topology properties of the graph.

2. **Node-Link-Positional Encoder**:

    - **Function**: Fuses node features, 1-hop neighbor edge features, and positional encodings into a compact node representation.
    - **Mechanism**: Separately aggregates (a) the mean of node features of 1-hop neighbors within the time window, $h_{u,N}^t$, (b) edge features + time encoding of the most recent $K$ interactions, $h_{u,E}^t$, and (c) neighbors' positional encodings, $h_{u,P}^t$. These three components are transformed via MLPs, concatenated, and fed into another MLP to obtain the final representation $h_u^t$.
    - **Design Motivation**: Replaces the attention mechanism of Transformers entirely with MLPs, validating that when positional encoding is sufficiently expressive, MLPs can achieve the same effect.

3. **Time Encoder**:

    - **Function**: Maps timestamps to vector representations.
    - **Mechanism**: Uses a cosine function $f_T(t) = \cos(t \cdot \omega)$, where $\omega_i = \alpha^{-(i-1)/\beta}$ provides multi-scale temporal awareness. The parameters $\alpha, \beta$ are fixed and not trained.
    - **Design Motivation**: Provides continuous temporal awareness to distinguish different time intervals.

### Loss & Training
Binary cross-entropy loss is used, where positive samples are actual links, and negative samples are obtained through random, historical, and inductive sampling strategies.

## Key Experimental Results

### Main Results

| Sampling Strategy | No. of Datasets | L-STEP Rank | No. of Baselines |
|---------|---------|------------|----------|
| Random Negative | 13 | Overall Best | 10 |
| Historical Negative | 13 | Overall Best | 10 |
| Inductive Negative | 13 | Overall Best | 10 |
| TGB Benchmark | Large-scale | Leading | Multiple SOTAs |

The proposed method achieves optimal performance in both transductive and inductive settings.

### Ablation Study

| Configuration | Key Findings | Details |
|------|---------|------|
| LPE + MLP vs Transformer | Comparable performance | Validates that MLPs can replace attention |
| Different PE initializations | Robust | Learns effective PE even with random initialization |
| Without LPE | Significant performance drop | Learnable PE is a critical component |
| Running time comparison | Faster than SOTA | Avoids $O(n^2)$ attention |

### Key Findings
- MLP combined with sufficiently expressive learnable positional encoding can completely replace the attention mechanism of Graph Transformers, with faster inference speed.
- DFT frequency filtering effectively captures the temporal patterns of positional encoding, which is theoretically proven to preserve spatial-temporal graph spectral properties.
- Leading performance is achieved on the large-scale TGB benchmark, verifying the practical scalability of the method.

## Highlights & Insights
- **Positional encoding as a signal**: The perspective is highly novel—analogizing the temporal evolution of PE to a signal processing problem and using DFT in the frequency domain to learn clean and predictive representations.
- **MLP vs Transformer**: The experiments are highly convincing—proving that the representational bottleneck lies not in the aggregation architecture, but in the quality of positional information.
- The value of learnable PE is verified both theoretically (spatial-temporal spectral preservation) and practically (comprehensive lead across 13 datasets).

## Limitations & Future Work
- Initializing positional encodings still requires computing the Laplacian eigendecomposition of the initial graph.
- The fixed hyperparameters $L$ (historical window length) and $K$ (neighbor sampling number) require tuning for each dataset.
- When the graph structure changes drastically (e.g., a large number of new nodes are added), the extrapolation capability of historical PE remains to be verified.
- Currently, it only processes undirected/directed interaction graphs, leaving heterogeneous graph settings unaddressed.

## Related Work & Insights
- **vs DyRep/TGAT/TGN**: These methods use RNNs or attention to capture temporal information, whereas L-STEP uses DFT+PE, which is more lightweight and achieves better results.
- **vs GraphGPS (Rampásek et al.)**: GraphGPS uses Laplacian PE + Transformer attention, while L-STEP demonstrates that learnable PE + MLP is sufficient.
- **vs Dwivedi et al. (2022)**: The only prior work on learnable PE, but restricted to static graphs; L-STEP extends this to temporal settings.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of PE as temporal signals processed by DFT is novel, and the validation of replacing Transformers with MLPs is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 13 datasets, 10 algorithms, 3 sampling strategies, and the TGB benchmark.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and experimental organization.
- Value: ⭐⭐⭐⭐ Provides a more efficient paradigm for temporal graph learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Positional Encoding meets Persistent Homology on Graphs](positional_encoding_meets_persistent_homology_on_graphs.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](../../NeurIPS2025/graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[ICML 2025\] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)
- [\[ICLR 2026\] LEAP: Local ECT-Based Learnable Positional Encodings for Graphs](../../ICLR2026/graph_learning/leap_local_ect-based_learnable_positional_encodings_for_graphs.md)
- [\[NeurIPS 2025\] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](../../NeurIPS2025/graph_learning/ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)

</div>

<!-- RELATED:END -->
