---
title: >-
  [Paper Note] GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs
description: >-
  [CVPR 2025][Image Generation][multimodal attributed graph] This paper proposes GraphGPT-o, which injects structural information of multimodal attributed graphs (MMAGs, where nodes contain image+text and edges represent relations) into a Multimodal Large Language Model (MLLM). Through PPR sampling, a hierarchical Q-Former aligner, and flexible inference strategies, it achieves joint text-image generation conditioned on the graph context.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "multimodal attributed graph"
  - "MLLM"
  - "graph linearization"
  - "hierarchical aligner"
  - "Q-Former"
  - "DreamLLM"
  - "Stable Diffusion"
date: 2026-05-08
content_hash: 66bdb722f45a7819
---

# GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs

**Conference**: CVPR 2025  
**arXiv**: [2502.11925](https://arxiv.org/abs/2502.11925)  
**Code**: To be open-sourced  
**Area**: Image Generation  
**Keywords**: multimodal attributed graph, MLLM, graph linearization, hierarchical aligner, Q-Former, DreamLLM, Stable Diffusion

## TL;DR

This paper proposes GraphGPT-o, which injects structural information of multimodal attributed graphs (MMAGs, where nodes contain image+text and edges represent relations) into a Multimodal Large Language Model (MLLM). Through PPR sampling, a hierarchical Q-Former aligner, and flexible inference strategies, it achieves joint text-image generation conditioned on the graph context.

## Background & Motivation

**Background**: MLLMs (such as DreamLLM) are already capable of understanding and generating text-image content. However, in reality, text and images are often interconnected in graph structures (such as co-purchase graphs of e-commerce products or art genre association graphs), forming Multimodal Attributed Graphs (MMAGs).

**Limitations of Prior Work**:
1. **Graph scale explosion**: Directly inputting the complete local subgraph leads to exponentially growing sequence lengths.
2. **Non-Euclidean structure**: The complex topology of graphs is difficult to input directly into a serialized MLLM.
3. **Hierarchical modal dependencies**: Node-level (text+image complementarity) and subgraph-level (node semantics + structure) fusion require different levels of integration.
4. **Inference dependency**: The sequential order of text and image generation affects the output quality.

**Key Challenge**: MLLMs process linear sequences, whereas the value of MMAGs lies precisely in their non-linear graph structural relationships.

**Goal**: Enable MLLMs to utilize structural and semantic information within MMAGs to generate matching images and texts for target nodes.

**Key Insight**: Compress subgraphs with PPR sampling + encode graph structure with a two-level Q-Former + inject graph tokens into the MLLM.

**Core Idea**: Use Personalized PageRank to sample key neighbors, employ a hierarchical Q-Former to encode node modalities and graph structures, and inject graph tokens into the MLLM to achieve graph-conditioned multimodal generation.

## Method

### Overall Architecture

1. **PPR Neighbor Sampling**: Uses Personalized PageRank on the target node to select the top-K most relevant neighbors.
2. **Hierarchical Multimodal Aligner**: 
    - Node Feature Q-Former: Fuses text and image features of each neighbor.
    - Graph Structure Q-Former: Aggregates structural information among neighboring nodes.
3. **MLLM Inference**: Feeds the graph token $\mathbf{g}_{v_i}$ into DreamLLM along with text/image tokens.
4. **Stable Diffusion Decoding**: The generated image tokens are decoded into images via SD.

### Key Designs

#### 1. Personalized PageRank Neighbor Sampling

To resolve the issue of graph scale explosion, the PPR matrix $\mathbf{P} = \beta \hat{\mathbf{A}} \mathbf{P} + (1-\beta)\mathbf{I}$ calculates the relevance score of each node to the target, selecting the top-K neighbors:

$$N(v_i) = \arg\max_{|N(v_i)|=K} \sum_{v_j \in N(v_i)} P_{i,j}$$

Compared to fixed-hop neighbor sampling, PPR can select the most relevant nodes across multiple hops, avoiding the introduction of irrelevant noise.

#### 2. Hierarchical Q-Former Aligner

**Node Feature Q-Former** $\phi(\cdot)$:
- Concatenates the neighbor node's text token $\mathbf{w}_{v_j}$ and CLIP image token $\mathbf{I}_{v_j}$.
- Interchanges text-image modal information through an $L_1$-layer self-attention Transformer.
- Compresses into a fixed-length representation $\mathbf{H}_{v_j}$ via cross-attention with a learnable soft prompt $\mathbf{Q}_V$.

**Graph Structure Q-Former** $\psi(\cdot)$:
- Concatenates the node representations $\mathbf{H}_{v_j}$ of all neighbors as input.
- Fuses deep information among nodes through an $L_2$-layer self-attention.
- Aggregates into a graph token $\mathbf{g}_{v_i}$ via cross-attention with a learnable soft prompt $\mathbf{Q}_G$.

#### 3. Exploration of Inference Strategies

- **Sequential (Text-first)**: Generates text $d_{v_i}$ first, then generates the image $p_{v_i}$ conditioned on the text and graph tokens.
- **Sequential (Image-first)**: Generates the image first, then generates text conditioned on the image and graph tokens.
- **Parallel**: Generates text and image independently and simultaneously to avoid error propagation.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{MLLM}^{GraphGPT-o} + \mathcal{L}_{SD}^{GraphGPT-o}$$

- MLLM Loss: Autoregressive next-token prediction (interleaved sequences of text, image, and graph tokens).
- SD Loss: Standard diffusion denoising loss, conditioned on text, image, and graph tokens.

## Key Experimental Results

### Main Results

Three datasets: ART500K (artwork graph), Amazon-Baby, Amazon-Beauty (e-commerce product graphs).

Evaluation metrics: CLIP-I2 (generated image vs. GT image), Perplexity (text fluency), CLIP-IT (image-text alignment), KL-DV (consistency with neighbor distribution).

**Key findings from individual Graph Linearization experiments**:
- Using bi-modality (text+image) generally outperforms single modality.
- Modal ordering has inconsistent impacts on performance.
- Image-first inference improves image quality but may degrade text quality.
- Text-first inference achieves the lowest KL-DV (highest consistency with neighbor distribution).

**Hierarchical Aligner vs. Linearization (ART500K)**:

| Method | CLIP-I2 ↑ | Perp. ↓ | CLIP-IT ↑ | KL-DV ↓ |
|------|-----------|---------|-----------|---------|
| Best Linearization | 79.26 | 117.7 | 20.15 | 0.19 |
| **Hierarchical Aligner** | **82.15** | **98.3** | **23.41** | **0.15** |

The Hierarchical Q-Former fully outperforms simple linearization.

### Ablation Study

- Node Q-Former: Removing it leads to a significant drop in CLIP-I2 (node-level modal fusion is key).
- Graph Q-Former: Removing it increases KL-DV (graph structural information is important for distribution consistency).
- PPR vs. Random Sampling: PPR outperforms random sampling across all metrics.

### Key Findings

1. Graph structural information contributes significantly to generation quality and cannot rely solely on node attributes.
2. Hierarchical alignment (node-level fusion followed by graph-level aggregation) outperforms one-step flat fusion.
3. PPR sampling captures relevant neighbors across multiple hops better than BFS/DFS.
4. The optimal choice of inference strategy depends on dataset characteristics (artwork vs. e-commerce).
5. There is an optimal number of neighbors; too many will introduce noise.

## Highlights & Insights

1. **Novel Problem Definition**: Formulates the multimodal content generation task on MMAGs for the first time, generating text and images simultaneously.
2. **Reasonable Hierarchical Design**: The two-level design of Node Q-Former and Graph Q-Former elegantly handles information fusion at different granularities.
3. **Systematic Experiments**: Conducts comprehensive combinatorial experiments on modal selection, modal order, and inference strategies for linearization.
4. **Rich Application Scenarios**: E-commerce recommendation (product generation), virtual art creation, and social network content recommendation.
5. **PPR Sampling**: Introduces the mature PPR method from graph learning to MLLM subgraph selection, balancing efficiency and effectiveness.

## Limitations & Future Work

1. Built on DreamLLM + Stable Diffusion 1.x, the generated image quality is capped by the foundation models.
2. Validated only on three relatively small-scale datasets, without testing on large-scale graphs (million-node level).
3. PPR sampling requires precomputing the PPR matrix for the entire graph, which is unfriendly to dynamic graphs or streaming scenarios.
4. The selection of inference strategies requires tuning for different datasets, lacking an adaptive mechanism.
5. Edge attributes (such as relationship types) are not considered; edges are only used for connectivity.

## Related Work & Insights

1. **DreamLLM** (Dong et al., 2024): The foundational MLLM of GraphGPT-o, supporting interleaved text-image understanding and generation.
2. **BLIP-2 / InstructBLIP**: The architectural source of Q-Former.
3. **GraphGPT** (Tang et al., 2023): A pioneer in injecting graph information into LLMs, but only processes textual graphs.
4. **PPR / APPNP** (Klicpera et al., 2019): Application of Personalized PageRank in graph learning.

**Insights**: Graph structure is a universally existing relational form in the real world. Injecting graph tokens into various foundation models (LLMs, MLLMs, diffusion models) could be an important direction. The "two-stage compression" concept of Q-Former can be generalized to other hierarchical structured data.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — The problem definition is novel, but the individual components (Q-Former, PPR, DreamLLM) are combinations of existing technologies.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic studies on linearization variants are very detailed, but the dataset scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation of the problem, and complete description of the methodology.
- **Value**: ⭐⭐⭐ — Promising in e-commerce recommendation/art creation scenarios, but the actual demand for graph-based generation still needs verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation](divot_diffusion_powers_video_tokenizer_for_comprehension_and_generation.md)
- [\[ICCV 2025\] Video Motion Graphs](../../ICCV2025/image_generation/video_motion_graphs.md)
- [\[NeurIPS 2025\] Janus-Pro-R1: Advancing Collaborative Visual Comprehension and Generation via Reinforcement Learning](../../NeurIPS2025/image_generation/janus-pro-r1_advancing_collaborative_visual_comprehension_and_generation_via_rei.md)
- [\[CVPR 2025\] WeGen: A Unified Model for Interactive Multimodal Generation as We Chat](wegen_a_unified_model_for_interactive_multimodal_generation_as_we_chat.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
