---
title: >-
  [Paper Note] COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation
description: >-
  [ECCV2024][Self-Supervised Learning][Urban Layout Generation] This work proposes a city-scale 2.5D urban layout generation method based on Graph Masked Autoencoders (GMAE). By capturing multi-level semantic context across buildings, blocks, and communities through a canonical graph representation and combining it with priority-scheduled iterative sampling, the method achieves realistic, semantically consistent, and topologically correct large-scale urban layout generation acr…
tags:
  - "ECCV2024"
  - "Self-Supervised Learning"
  - "Urban Layout Generation"
  - "Graph Masked Autoencoder"
  - "Context Sensitivity"
  - "Vector Quantization"
  - "Scheduled Iterative Sampling"
date: 2026-05-08
content_hash: f9977629f9c1604a
---

# COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation

**Conference**: ECCV2024  
**arXiv**: [2407.11294](https://arxiv.org/abs/2407.11294)  
**Code**: [Arking1995/COHO](https://github.com/Arking1995/COHO)  
**Area**: Self-Supervised  
**Keywords**: Urban Layout Generation, Graph Masked Autoencoder, Context Sensitivity, Vector Quantization, Scheduled Iterative Sampling

## TL;DR

This work proposes a city-scale 2.5D urban layout generation method based on Graph Masked Autoencoders (GMAE). By capturing multi-level semantic context across buildings, blocks, and communities through a canonical graph representation and combining it with priority-scheduled iterative sampling, the method achieves realistic, semantically consistent, and topologically correct large-scale urban layout generation across 330 US cities.

## Background & Motivation

Large-scale urban layout generation is widely demanded in computer vision, urban planning, digital twins, and game design. A city typically contains 1,000 to 50,000 blocks. While block configurations vary significantly, adjacent blocks often share stylized patterns or deliberate multi-block arrangements.

Limitations of prior work:

- **Procedural Generation** (e.g., Parish & Müller) requires manual rule-crafting, lacking flexibility.
- **Pixel-based Deep Learning** methods (InfiniCity, CityDreamer, CityGEN) are trained on image patches, which fails to capture global features and inter-dependencies. This leads to limited semantics and realism, often requiring post-processing.
- **Graph-based Layout Generation** methods (LayoutTransformer, VTN, LayoutDM, GlobalMapper) treat each block as an independent unit, overlooking the context sensitivity of urban layouts—meaning style consistency and semantic relations between adjacent blocks are not modeled.

Key Observation: Urban layouts of buildings, blocks, and communities are not isolated; they exhibit multi-level semantic dependencies. Generation without considering context leads to unnatural over-diversity or over-similarity between neighboring blocks.

## Core Problem

How to achieve context-sensitive city-scale 2.5D layout generation while maintaining large-scale scalability? Specifically, three sub-problems need to be addressed:

1. **Representation**: How to compactly represent arbitrary-shaped buildings, blocks, communities, and roads in a city using a unified data structure?
2. **Context Sensitivity**: How to explicitly incorporate multi-level structural inter-dependencies during generation?
3. **Priority**: How to determine the generation order of different areas in large-scale city generation to improve overall quality?

## Method

### 1. Canonical Graph Representation

The entire city is represented as a graph $G = \{B, E\}$:

- **Nodes** $b_i$: Each urban block comprises two sets of features:
    - $s_i$ (4D): Shape and positional features of the block—aspect ratio, total area, the ratio of its area to its convex hull area, and the relative distance to the city center.
    - $q_i$ (512D): A discrete vector obtained from a pre-trained quantized codebook $C$, hierarchically encoding the layouts, shapes, and heights of all buildings within the block.
    - The two are concatenated into a 516D node feature $[s_i, q_i]$.
- **Edges** $e_{ij}$: Connect spatially adjacent blocks, with edge features $d_{ij}$ representing the distance between the centroids of neighboring blocks.
- **Subgraphs**: Any connected subgraph corresponds to a community.

### 2. Building Layout Quantization

The process of obtaining the 512D vector $q_i$:

1. Train a block-level Variational Autoencoder (BVAE) with a GAT backbone, using graph structures to represent the shape, position, and height of multiple buildings within a single block.
2. Use the pre-trained BVAE encoder to obtain a 512D latent vector for all building layouts.
3. Quantize the distribution of each dimension into $L=20$ equal-percentile bins.
4. Replace each latent value with its corresponding bin index to obtain the final quantized index vector $q_i = [c_1, ..., c_{512}]$, where $c \in \{1,2,...,20\}$.

Note: Unlike VQVAE/VQGAN, this codebook is defined after BVAE training and is not dynamically trainable.

### 3. Graph Masked Autoencoder (GMAE)

Self-supervised training pipeline:

- **Masking Strategy**: Randomly mask the building layout features $Q$, with the masking ratio $m$ sampled from a truncated Gaussian distribution $\mathcal{N}(0.55, 0.25)$ constrained to $[0.5, 1.0]$. Block shape features $S$ and edge features $E$ are preserved.
- **Encoder** $f_E$: GAT (3 layers, i.e., $D=3$-hop message passing), obtaining context-aware node features $F$ via neighborhood message passing.
- **Decoder** $f_D$: A simple MLP that reconstructs the quantized building layout features $Q'$ from $F$.
- **Loss Function**: Cross-entropy loss is applied to masked nodes: $L_{recon} = -\sum_{i=1}^{L} [Q_i \log(Q'_i)]^{mask}$.
- **Community Sampling**: During training, subgraphs with a radius of 500 meters are randomly sampled from the city graph as batches to cover a reasonable number of blocks.

### 4. Priority-based Scheduled Generation

Inference employs $T=12$ iterative sampling steps:

1. In each iteration, GMAE predicts the building layout features for all remaining masked nodes.
2. Nodes are sorted by prediction confidence, and those with the highest confidence are accepted.
3. The acceptance ratio is controlled by a cosine schedule function: $\beta(t) = 1 - \cos(t/T)$.
    - Initial iterations are conservative (accepting a small number of high-confidence nodes), with speedups in the later stages.
    - Effect: Important and representative blocks are generated first, steering the style of neighboring blocks.
4. Supports arbitrary-ratio prior constraints $[0, 100\%]$, enabling generation from scratch, area completion, and refinement.

## Key Experimental Results

### Dataset

- 330 US cities (population > 100,000), totaling 833,473 blocks and 17,663,607 buildings.
- Data sources: OpenStreetMap, Microsoft Building Footprints, and TIGER datasets.
- Splits: 70% training, 20% validation, 10% testing.

### Quantitative Comparison (100 communities, 3700 blocks)

| Method | CTS↓ | WD-5D↓ | WD-CO↓ | Overlap↓ | FID↓ | KID↓ | LPIPS↓ |
|------|-------|--------|--------|----------|------|------|--------|
| SDXL | - | - | - | - | 120.24 | 0.079 | 0.48 |
| VTN | -1.14 | 3.18 | 5.81 | 1.24 | 69.14 | 0.047 | 0.32 |
| LayoutDM | -2.20 | 2.92 | 12.50 | 4.56 | 66.77 | 0.040 | 0.39 |
| GlobalMapper | 0.62 | 4.77 | 4.14 | 2.52 | 49.55 | 0.024 | 0.34 |
| **COHO** | **0.21** | **2.28** | **1.91** | **1.27** | **23.63** | **0.005** | **0.20** |

- FID drops from the second-best 49.55 to 23.63 (52% gain), and KID from 0.024 to 0.005.
- CTS (Context Score) is closest to 0, indicating the best neighborhood style consistency.
- Training time: BVAE 12h + GMAE 15h (on a single A5000 GPU); inference is nearly instantaneous.

### Ablation Study

- **BVAE Backbone**: GAT >> GCN > GraphSAGE >> LayoutVAE; GAT achieves the best performance across all geometric error metrics.
- **Quantization Method**: Dimension-wise Quantization (DIM-Q) >> Trainable VQ >> KMeans-Q, with $L=20$ being the optimal trade-off.
- **GMAE Encoder Depth**: $D=3$ is optimal; $D=1$ provides insufficient context, while $D=4$ causes over-similarity and increased computational overhead.
- **Masking Strategy**: Dynamic high masking rate $[0.5, 1.0]$ is optimal; a fixed low masking rate of 0.15 leads to over-similarity.
- **Schedule Function**: Cosine function with $T=12$ is optimal; logarithmic function performs the worst, and single-step generation ($T=1$) yields extremely poor quality.

## Highlights & Insights

1. **Context-Sensitive Graph Representation**: For the first time, multi-level urban semantics (buildings $\rightarrow$ blocks $\rightarrow$ communities $\rightarrow$ cities) are unified into a canonical graph representation, supporting large-scale generation.
2. **Ingenious Self-Supervised Learning Paradigm**: Adapting the mask-and-reconstruct concept from BERT/MAE/MaskGIT to the urban graph generation task, requiring no paired data.
3. **Well-Founded Priority Scheduling**: Generating highly confident and representative blocks first before broadcasting matches the "anchor $\rightarrow$ infill" intuition in urban planning.
4. **Large-Scale and Systematic Evaluation**: Evaluated across 330 cities, 830k blocks, and 17.66 million buildings, backed by thorough and systematic ablation studies.
5. **Open-Source Dataset**: Releasing a large-scale urban layout dataset, which is of great value for future research in the community.

## Limitations & Future Work

- Cannot handle non-simple polygon buildings (e.g., buildings with inner courtyards) and highly irregular concave blocks (e.g., cul-de-sacs).
- The relationship between the road network and context structures is not explicitly modeled; the road network serves as a given input rather than a generation target.
- The quantization process may lose fine-grained building details (as $L=20$ offers only 20 quantization levels).
- Currently trained and evaluated only on US cities; generalizability to European and Asian cities remains unexplored.
- Future Work: Combining with large-scale photorealistic multi-view scene synthesis, city-scale 3D modeling, and synthetic data generation for autonomous driving.

## Related Work & Insights

| Dimension | COHO (Ours) | GlobalMapper | LayoutDM | VTN | SDXL |
|------|-------------|--------------|----------|-----|------|
| Representation | Graph (Nodes=Blocks) | Graph (Single Block) | Token Sequence | Token Sequence | Pixels |
| Context Awareness | Multi-level Graph MP | None | None | None | None |
| Priority Generation | Cosine-scheduled Iterative | Single-step | Single-step | Autoregressive | Diffusion |
| Scalability | City-scale | Single Block | Single Block | Single Block | Patch-level |
| Arbitrary Shape | Supported | Supported | Limited | Limited | Unsupported |
| FID | 23.63 | 49.55 | 66.77 | 69.14 | 120.24 |

Connection to MaskGIT/MAGE: COHO transitions the mask-then-iterative-decode paradigm from image generation to graph-based urban layouts. While the core philosophy remains aligned, targeted designs are implemented for node feature quantization, graph message passing, and scheduling policies.

## Insights & Connections

- **Graph Masked Autoencoders for Structural Generation**: The self-supervised paradigm of masking and reconstruction is applicable not only to image and text but can also be transferred to graph-structured large-scale scene generation. This is worth exploring in other structured generation tasks (e.g., indoor scene generation, molecular design).
- **Hierarchical Quantized Representation**: Utilizing a task-specific VAE to quantize low-level details into discrete tokens before generating at a higher level is an effective strategy for handling complex multi-level structures.
- **Priority Scheduling**: Generating high-confidence and high-importance elements first, followed by incremental infilling, is better suited for spatially dependent generation tasks than uniform parallel or simple autoregressive approaches.
- **Evaluation Metric CTS**: The proposed Context Score (CTS) can serve as a universal evaluation tool to measure neighborhood style consistency.

## Rating
- Novelty: 8/10 — Introduces context sensitivity to large-scale urban layout generation for the first time; the combination of GMAE and priority scheduling is highly novel.
- Experimental Thoroughness: 9/10 — Large-scale experiments across 330 cities; systematic ablation studies cover model variants, quantization strategies, masking policies, and scheduling functions.
- Writing Quality: 8/10 — Well-structured with a clear logical progression from three observations to three designs; high-quality figures.
- Value: 7/10 — Although large-scale urban layout generation is a relatively niche area, the open-source dataset and proposed method carry significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] In Context Semi-Supervised Learning](../../ICLR2026/self_supervised/in_context_semi-supervised_learning.md)
- [\[CVPR 2026\] Free-Grained Hierarchical Visual Recognition](../../CVPR2026/self_supervised/free-grained_hierarchical_visual_recognition.md)
- [\[AAAI 2026\] Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision](../../AAAI2026/self_supervised/improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)
- [\[CVPR 2025\] Few-Shot Implicit Function Generation via Equivariance](../../CVPR2025/self_supervised/few-shot_implicit_function_generation_via_equivariance.md)
- [\[AAAI 2026\] FineXtrol: Controllable Motion Generation via Fine-Grained Text](../../AAAI2026/self_supervised/finextrol_controllable_motion_generation_via_fine-grained_text.md)

</div>

<!-- RELATED:END -->
