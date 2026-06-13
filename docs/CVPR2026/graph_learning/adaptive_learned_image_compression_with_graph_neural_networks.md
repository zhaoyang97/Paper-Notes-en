---
title: >-
  [Paper Note] Adaptive Learned Image Compression with Graph Neural Networks
description: >-
  [CVPR 2026][Graph Learning][Image Compression] GLIC reformulates the nonlinear transforms in learned image compression (LIC) from fixed convolutions or window-based attention into content-adaptive graph neural network op…
tags:
  - "CVPR 2026"
  - "Graph Learning"
  - "Image Compression"
  - "GNN"
  - "Dual-Scale Sampling"
  - "RMS Gradient"
  - "Content-Adaptive Connectivity"
date: 2026-05-08
content_hash: 1c16766c24039e65
---

# Adaptive Learned Image Compression with Graph Neural Networks

**Conference**: CVPR 2026
**arXiv**: [2603.25316](https://arxiv.org/abs/2603.25316)  
**Code**: [https://github.com/UnoC-727/GLIC](https://github.com/UnoC-727/GLIC)  
**Area**: Graph Learning / Learned Image Compression
**Keywords**: Image Compression, GNN, Dual-Scale Sampling, RMS Gradient, Content-Adaptive Connectivity

## TL;DR

GLIC reformulates the nonlinear transforms in learned image compression (LIC) from fixed convolutions or window-based attention into content-adaptive graph neural network operations. A dual-scale graph determines *where to connect*, while a complexity-aware mechanism determines *how much to connect*, enabling more effective modeling of both local and long-range redundancies. GLIC consistently outperforms traditional codecs and recent LIC baselines across three standard benchmarks.

## Background & Motivation

Learned image compression has evolved from early convolutional autoencoders to architectures based on CNNs, Transformers, and Mamba, with rate-distortion performance increasingly matching or surpassing traditional codecs. However, these approaches share a fundamental assumption: neighborhood structures are largely predefined. Convolutions bind each pixel to a fixed $k \times k$ local window, and window-based attention confines interactions within predefined blocks—even with shifted or deformable variants, the connectivity pattern is essentially fixed before aggregation.

The core issue is that image compression is fundamentally about redundancy, which is neither spatially uniform nor confined to local Euclidean neighborhoods. Smooth regions are highly redundant, while edges and textured regions are not; structurally similar but spatially distant regions can also be highly informative for compression. Fixed connectivity forces models to aggregate irrelevant neighbors while overlooking genuinely redundant long-range counterparts.

The authors distill this into two central problems:

- **Where to connect**: which spatial positions should exchange information.
- **How much to connect**: how many connections to allocate to each position.

CNNs and window attention are too rigid along both dimensions. The authors therefore turn to GNNs, leveraging their dynamic graph connectivity to let the compression model adaptively determine connection patterns based on content complexity and feature similarity. This is not a straightforward substitution of convolutions with graph networks, but a design explicitly structured around spatial redundancy modeling in compression, covering candidate neighborhood construction, degree allocation, and graph aggregation.

## Method

### Overall Architecture

GLIC builds upon the standard VAE-based compression framework, retaining the analysis transform $g_a$, synthesis transform $g_s$, and hyperprior entropy model. Rather than modifying the macro-level compression pipeline, the authors redesign the nonlinear transform blocks into a graph-driven **Graph-based Feature Aggregation (GFA)** module.

Lightweight convolutional blocks are retained for early-stage shallow feature extraction, as constructing graphs over high-resolution feature maps is computationally prohibitive. In the final two stages, where spatial resolution is reduced, conventional convolution and attention blocks are replaced with cascaded GFA-Local and GFA-Global modules. This design applies graph operations at the stages where they provide the greatest benefit, both in the encoder and decoder.

### Key Designs

1. **Dual-Scale Candidate Sampling**

    - **Function**: Provides each pixel node with a candidate neighbor set that encompasses both local and long-range information.
    - **Mechanism**: Two candidate sets are constructed per node. The local candidate set is drawn from a fixed local window to preserve fine-grained texture and boundary structure. The global candidate set is drawn from a strided mesh-grid spanning the entire feature map, providing a low-cost mechanism for long-range redundancy modeling. The final candidate set is the union of both.
    - **Design Motivation**: Using only a local graph discards long-range correlations; using only a global sparse graph fails to capture low-level structural details. The dual-scale design simultaneously supports fine local modeling and sparse global matching, with complexity far below full global self-attention.

2. **Complexity-Aware Adaptive Node Degree**

    - **Function**: Allocates different numbers of graph connections to different spatial positions rather than enforcing a uniform node degree.
    - **Mechanism**: Sobel operators are applied per channel to compute gradient magnitudes, which are then aggregated via RMS pooling to form a complexity score. Higher gradients indicate greater local structural complexity and lower redundancy, warranting more neighbors for effective modeling. The total edge budget $B = N \cdot \bar{d}$ is then distributed across nodes in proportion to their complexity scores, yielding a target degree $d_i^*$ per node.
    - **Design Motivation**: Unlike classification, compression does not require uniform modeling capacity at every position. Allocating fewer connections to smooth regions does not degrade reconstruction quality, while preserving budget for edges and textures that are harder to compress.

3. **Similarity-Threshold Graph Construction and GFA Aggregation**

    - **Function**: Selects the most informative neighbors from the candidate set and performs message-passing aggregation.
    - **Mechanism**: For each node, cosine similarities with all candidates are computed, and binary search is used to find a threshold that retains a number of neighbors as close as possible to the target degree $d_i^*$. Graph feature aggregation is then performed on the resulting directed graph, first via local graph aggregation and subsequently via global graph aggregation.
    - **Design Motivation**: Dual-scale sampling addresses the question of which nodes are potentially worth connecting to; threshold-based selection resolves the final connectivity. This two-stage design is more controllable than soft global attention and more naturally induces the sparse structure favorable for compression.

### Loss & Training

The training objective follows standard rate-distortion optimization, minimizing the weighted sum of rate and distortion. Separate models are trained under PSNR and MS-SSIM criteria, with evaluation conducted using BD-rate and BD-PSNR metrics. This ensures that observed gains stem from improved transform representations under the same compression objective, rather than from changes in evaluation protocol.

## Key Experimental Results

### Main Results

GLIC is evaluated against VTM-9.1 and a range of recent LIC baselines on the Kodak, Tecnick, and CLIC benchmarks. Key results are summarized below.

| Metric | Kodak | Tecnick | CLIC |
|---|---:|---:|---:|
| BD-rate vs. VTM-9.1 | **−19.29%** | **−21.69%** | **−18.71%** |
| BD-PSNR gain vs. FTIC | +0.26 dB | +0.38 dB | +0.37 dB |
| BD-PSNR gain vs. TCM-L | +0.39 dB | +0.56 dB | +0.46 dB |

GLIC achieves consistent improvements across all three datasets, including high-resolution Tecnick and 2K CLIC, with the 21.69% BD-rate reduction on Tecnick being particularly notable.

### Ablation Study

A detailed ablation over complexity scoring strategies and channel pooling methods provides strong empirical support for the Sobel-RMS design choice.

| Scoring Strategy | Channel Pooling | Kodak | CLIC | Tecnick |
|---|---|---:|---:|---:|
| None | None | −16.97 | −16.21 | −18.21 |
| Local Entropy | RMS | −17.05 | −17.01 | −18.97 |
| Rescaling Residual | RMS | −17.67 | −17.03 | −19.68 |
| Rescaling Residual | Mean | −18.23 | −17.82 | −20.39 |
| Sobel Gradient | Mean | −18.02 | −17.42 | −20.62 |
| **Sobel Gradient** | **RMS** | **−19.29** | **−18.71** | **−21.69** |

### Key Findings

- The dual-scale graph design is validated: using only a local or only a global graph leads to noticeable degradation, with the global-only variant performing worst, confirming that both local structural detail and long-range redundancy modeling are necessary for effective compression.
- Complexity-aware degree allocation is critical: removing it degrades the model toward a fixed-$k$NN GNN, with consistent drops across all three datasets.
- Sobel + RMS outperforms mean pooling, suggesting that emphasizing large-gradient regions is appropriate for compression. RMS pooling is more sensitive to strong edges, appropriately directing connectivity budget toward hard-to-compress regions.
- Efficiency comparisons with MambaIC demonstrate that GLIC achieves meaningful reductions in parameter count, FLOPs, decoding latency, and memory usage, confirming that the graph structure does not incur prohibitive computational overhead.

## Highlights & Insights

- The primary contribution of this work is not the headline of "first GNN for LIC," but rather the targeted application of GNNs to the two problems most critical for compression: connectivity scope and connectivity density. This problem decomposition is both intuitive and grounded in the physical structure of image redundancy.
- The decoupling of *where* and *how much* is worth noting as a general design principle. Many architectures conflate all adaptivity into a single attention module, whereas GLIC explicitly separates candidate sampling from budget allocation, resulting in clearer module responsibilities and more interpretable ablations.
- The hybrid architecture—retaining lightweight convolutions in early stages and applying GFA only in the final two stages—reflects strong engineering pragmatism, making the approach more practical than a fully graph-based pipeline.
- The effective receptive field (ERF) analysis provides compelling evidence that content-adaptive connectivity is functioning as intended: GLIC generates visibly different ERF patterns at positions with different content characteristics.

## Limitations & Future Work

- Graph construction still requires computing candidate similarities and performing binary search for threshold selection. Although substantially cheaper than full global attention, this may become a bottleneck at higher resolutions or in latency-sensitive deployment scenarios.
- The current work focuses on static image compression. Extension to video compression introduces a temporal dimension to graph nodes, making graph construction and synchronization considerably more complex.
- The connectivity budget relies on a hand-crafted complexity score. While Sobel-RMS proves effective, it remains a manually designed feature; a learned complexity estimator may offer further improvements.
- Comparisons are primarily against VTM and academic LIC baselines. End-to-end evaluation within full industrial encoding pipelines, including hardware constraints, remains an open direction.
- Incorporating GFA into the entropy model, rather than only the transform network, represents a natural extension that may yield additional rate-distortion gains.

## Related Work & Insights

- **vs. CNN-based LIC**: Convolutions are highly efficient but limited by fixed Euclidean neighborhoods. GLIC demonstrates that for redundancy-driven tasks such as compression, fixed local connectivity is not always the optimal inductive bias.
- **vs. Window Transformer LIC**: Window-based attention extends expressiveness beyond convolutions but remains fundamentally block-local. GLIC's global sparse sampling addresses the limitation of being unable to connect across window boundaries.
- **vs. Deformable Convolution-based Compression**: Deformable convolutions allow dynamic spatial offsets, but remain bounded in their range and cardinality; graph connectivity offers greater freedom over the neighbor set.
- A broader takeaway is that in low-level vision tasks, the most valuable property of graph networks is not abstract semantic aggregation, but the ability to handle spatially non-uniform relational structures. Tasks such as compression, denoising, and super-resolution are natural candidates for redesigning adjacency structures along this direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — GNNs are genuinely integrated into the rate-distortion modeling logic of LIC, rather than used as a drop-in module replacement.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three standard benchmarks, comprehensive main results, complexity analysis, and multiple ablation groups are all well-covered; downstream deployment evaluations could further strengthen the work.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem decomposition is clear, and the correspondence between methodology and experiments is strong.
- **Value**: ⭐⭐⭐⭐⭐ — Establishes a new modeling paradigm for learned image compression with potential influence extending beyond the compression domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](../../AAAI2026/graph_learning/beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](../../ICML2026/graph_learning/quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](../../NeurIPS2025/graph_learning/graph_neural_networks_for_interferometer_simulations.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](../../ICLR2026/graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)

</div>

<!-- RELATED:END -->
