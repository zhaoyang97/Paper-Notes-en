---
title: >-
  [Paper Note] Graph4MM: Weaving Multimodal Learning with Structural Information
description: >-
  [ICML2025][Multimodal VLM][Multimodal Graphs] Proposing the Graph4MM framework, which injects multi-hop graph structural information into the self-attention mechanism via Hop-Diffused Attention and designs MM-QFormer to achieve cross-modal fusion, achieving an average improvement of 6.93% on generative and discriminative tasks.
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "Multimodal Graphs"
  - "Structural Information Fusion"
  - "Hop-Diffused Attention"
  - "QFormer"
  - "Zero-Shot Classification"
date: 2026-05-08
content_hash: d0c2bc343638b9ee
---

# Graph4MM: Weaving Multimodal Learning with Structural Information

**Conference**: ICML2025  
**arXiv**: [2510.16990](https://arxiv.org/abs/2510.16990)  
**Code**: [GitHub](https://github.com/YennNing/Graph4MM)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Graphs, Structural Information Fusion, Hop-Diffused Attention, QFormer, Zero-Shot Classification

## TL;DR

Proposing the Graph4MM framework, which injects multi-hop graph structural information into the self-attention mechanism via Hop-Diffused Attention and designs MM-QFormer to achieve cross-modal fusion, achieving an average improvement of 6.93% on generative and discriminative tasks.

## Background & Motivation

Real-world multimodal data typically possesses complex structural relationships that extend beyond simple one-to-one mappings (such as image-text pairs). For instance, in an academic paper, an image is directly paired with its caption, but the relationships between the image and subsequent section content or page summaries are non-linear and multi-layered. Existing VLMs (such as BLIP2 and Qwen2-VL) remain limited to modeling one-to-one image-text relationships, failing to capture complex multimodal interactions.

Although the pioneering work MMGL models modal data as a graph, it suffers from two key limitations:

**Indiscriminate treatment of neighbors**: It simply concatenates the multimodal data of neighbors without differentiating the importance of nodes at different hop distances.

**Graph as an independent modality**: It injects the graph topology as an independent modality in parallel with text/vision. However, since the feature spaces of pre-trained language and vision models are already highly aligned, the graph embeddings introduce semantic gaps instead, leading to performance degradation.

The core insight of the authors is: **the graph structure should not serve as an independent modality, but rather as a structural prior guiding intra- and inter-modal interactions**.

## Method

### Multimodal Graph Modeling

Define a multimodal graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T}, \mathcal{P})$, where each node $v_i$ contains optional text attributes $t_{v_i}$ and visual attributes $p_{v_i}$. Edges are categorized into three types: text-text, image-image, and text-image. For a target node, a text subgraph $\mathcal{G}_t$ and a visual subgraph $\mathcal{G}_p$ are extracted based on its $\tau$-hop neighbors.

### Hop-Diffused Attention

This is the core innovation of this work, which integrates multi-hop graph structural information into the self-attention mechanism in three steps:

**Step 1: Self-Attention Calculation**. For visual embeddings $\mathbf{H}_P \in \mathbb{R}^{|\mathcal{V}_p| \times d}$, the standard attention matrix is computed as:

$$\mathbf{A}'_{i,j} = \text{Softmax}_j\left(\frac{\mathbf{q}_{v_i}^\top \mathbf{k}_{v_j}}{\sqrt{d}}\right)$$

**Step 2: Causal Masking**. Mask $\mathbf{M}_{i,j}$ is defined based on the graph edge set $\mathcal{E}_p$, allowing attention only between connected nodes to align attention with the graph topology:

$$\mathbf{A}_{i,j} = \text{Softmax}(\mathbf{M}_{i,j} \cdot \mathbf{A}'_{i,j})$$

**Step 3: Diffusion Mechanism**. Multi-hop structural information is captured by iteratively propagating attention:

$$\boldsymbol{\mathcal{A}} = \sum_{i=0}^{\infty} \theta_i \mathbf{A}^i, \quad \theta_i = \alpha(1-\alpha)^i, \quad \alpha \in (0,1)$$

where $\theta_i$ is an exponential decay coefficient, and $\alpha$ controls the influence of distant neighbors. The embeddings are finally updated via a residual connection:

$$\mathbf{H}_P \leftarrow \mathbf{H}_P + \boldsymbol{\mathcal{A}} \mathbf{H}_P$$

**Theoretical Guarantee**: The authors prove that Hop-Diffused Attention retains higher Dirichlet Energy than stacking $k$ GAT layers, i.e., $\mathcal{E}_{\text{Hop-Diffused}}(\mathbf{X}^{(1)}) > \mathcal{E}_{\text{GAT}}(\mathbf{X}^{(k)})$, effectively mitigating the oversmoothing problem.

**Lightweight Alternative: Hop-Aware Attention**. To reduce computational complexity (from $O(|\mathcal{V}_p| \cdot d^2)$ to $O(|\mathcal{V}_p| \cdot d)$), a learnable hop embedding $\mathbf{h}_{\text{hop}}^{(h)}$ is introduced and directly added to the node embeddings, letting the downstream model adaptively learn the importance of different hop information.

### MM-QFormer (Multimodal Query Transformer)

Inspired by BLIP2's Q-Former, a module designed for cross-modal fusion:

1. **Shared Self-Attention**: The learnable query tokens $\mathbf{Q}_v^{(0)}$ are concatenated with text embeddings $\mathbf{H}_T$, allowing the query tokens to perceive text context through shared self-attention.
2. **Cross-Modal Cross-Attention**: The updated query tokens serve as queries, while visual embeddings $\mathbf{H}_P$ serve as keys/values, extracting vision features relevant to the text.
3. **Feed-Forward Network**: A two-layer fully connected network further processes the query tokens.

After $L$ layers, the final query tokens serve as multimodal tokens and are inserted after the text attribute tokens, which are then fed into the frozen pre-trained language model to generate outputs.

### Loss & Training

The model adopts a standard autoregressive language modeling loss, freezing the visual encoder and the LLM, and training only the parameters in the Hop-Diffused Attention module and MM-QFormer.

## Key Experimental Results

### Datasets

- **WikiWeb2M** (generative task): Document section summary generation, containing multimodal webpage content such as page descriptions, section texts, images, and captions.
- **Ele-Fashion** (discriminative task): Zero-shot product classification, where nodes represent products and edges represent co-purchase relationships.

### Main Results (OPT-125M backbone)

| Method | BLEU-4 | ROUGE-L | CIDEr | Acc(%) |
|------|--------|---------|-------|--------|
| BLIP2 (Subgraph Text) | 0.0000 | 0.0530 | 0.0063 | 31.37 |
| Qwen2-VL (Subgraph Text) | 0.0000 | 0.1192 | 0.0084 | 12.33 |
| MMGL (Subgraph T&I) | 0.0778 | 0.4041 | 0.7712 | 99.85 |
| MMGL (Subgraph T&I+GNN) | 0.0633 | 0.3814 | 0.6326 | 70.89 |
| **Graph4MM Hop-Diffused** | **0.0800** | **0.4076** | **0.7831** | **100.00** |

### LLaMA-1B Backbone Results

| Method | BLEU-4 | ROUGE-L | CIDEr | Acc(%) |
|------|--------|---------|-------|--------|
| MMGL (Subgraph T&I) | 0.1157 | 0.4685 | 1.1072 | 98.07 |
| **Graph4MM Hop-Diffused** | **0.1177** | **0.4713** | **1.1221** | **100.00** |

### Ablation Study (OPT-125M, Generative Task)

| Variant | BLEU-4 | ROUGE-L | CIDEr |
|------|--------|---------|-------|
| Hop-Diffused MM-QFormer (Full) | 0.0800 | 0.4076 | 0.7831 |
| Remove text subgraph structure | 0.0786 | 0.4065 | 0.7765 |
| Remove image subgraph structure | 0.0769 | 0.4044 | 0.7684 |

Key Findings: Removing structural information of the image modality leads to a more significant performance drop, as text can retain some structural information through prompts (e.g., "context from 1-hop neighbors"), whereas images have no such route.

## Highlights & Insights

1. **Revisualizing the Role of Graphs in Multimodal Learning**: Theoretical and empirical evidence demonstrates that graph structures should not be injected as independent modalities (e.g., the GNN approach in MMGL leads to performance degradation) but should instead serve as structural priors guiding modal interactions.
2. **Theoretical Guarantees of Hop-Diffused Attention**: Dirichlet Energy analysis proves its ability to avoid oversmoothing, outperforming stacked multi-layer GCNs/GATs, and capturing multi-hop information using only a single layer.
3. **Small Models Outperforming Large Models**: By incorporating structural information, Graph4MM with smaller backbones like OPT-125M/LLaMA-1B outperforms much larger models such as BLIP2-OPT-2.7B and Qwen2-VL-7B.
4. **Hop-Aware as a Lightweight Alternative**: An alternative is provided that reduces computational complexity from $O(d^2)$ to $O(d)$, with performance closely matching or even partially exceeding Hop-Diffused.

## Limitations & Future Work

1. **Limited Scale of Datasets**: Validated only on two datasets (WikiWeb2M and Ele-Fashion), lacking testing in larger-scale or more diverse scenarios.
2. **Graph Construction Relies on Manual Definition**: The establishment of edges depends on predefined rules (such as chapter hierarchies, co-purchase relationships), without exploring automatic graph construction methods.
3. **Relatively Small Backbone Scales**: Only OPT-125M and LLaMA-1B were used, leaving the effectiveness and scalability on larger LLMs (such as 7B+) unverified.
4. **Selection of Diffusion Steps $K$**: The infinite series is truncated to finite steps, but the paper does not fully discuss the sensitivity of the choice of $K$ to different graph structures.
5. **Lack of Comparison with Recent Multimodal Graph Methods**: Mainly compared with MMGL, without comparisons with other recent multimodal graph learning methods (such as GraphAdapter, etc.).

## Related Work & Insights

- **MMGL (Yoon et al., 2023)**: The first work to model multimodal data as graphs, but it simply concatenates neighbors and treats the graph as an independent modality.
- **BLIP2 (Li et al., 2023)**: The design of Q-Former inspired MM-QFormer, but BLIP2 only processes individual image-text pairs.
- **Personalized PageRank / APPNP**: The theoretical foundation of the diffusion mechanism originates from PPR, generalized here to attention matrices.
- **Insight**: In multimodal learning, the value of structural information lies in "how to guide attention allocation" rather than being injected as an auxiliary feature.

## Rating

- Novelty: ⭐⭐⭐⭐ — The design of Hop-Diffused Attention, combining PPR diffusion with attention masking, is novel and redefines the role of graphs in multimodal learning.
- Experimental Thoroughness: ⭐⭐⭐ — Ablation studies are thorough, but with only two datasets and relatively small backbone scales.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, solid theoretical analysis, and consistent notation.
- Value: ⭐⭐⭐⭐ — Provides theoretical and practical guidance for incorporating structural information into multimodal learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Learning Optimal Multimodal Information Bottleneck Representations](learning_optimal_multimodal_information_bottleneck_representations.md)
- [\[CVPR 2026\] Information-Theoretic Decomposition for Multimodal Interaction Learning](../../CVPR2026/multimodal_vlm/information-theoretic_decomposition_for_multimodal_interaction_learning.md)
- [\[CVPR 2026\] SO-Bench: A Structural Output Evaluation of Multimodal LLM](../../CVPR2026/multimodal_vlm/so-bench_a_structural_output_evaluation_of_multimodal_llm.md)
- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](../../AAAI2026/multimodal_vlm/conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)
- [\[CVPR 2026\] StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](../../CVPR2026/multimodal_vlm/structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)

</div>

<!-- RELATED:END -->
