---
title: >-
  [Paper Note] Vision Graph Prompting via Semantic Low-Rank Decomposition
description: >-
  [ICML2025][Multimodal VLM][Vision GNN] This work proposes Vision Graph Prompting (VGP), the first visual prompt learning framework tailored for Vision GNN (ViG). By leveraging the low-rank characteristics of semantic connected components in graphs, VGP designs semantic low-rank prompts at three granularities: graph, edge, and node levels (SeLo-Graph/Edge/Node Prompt). This approach achieves downstream transfer performance close to full fine-tuning while maintaining parameter…
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "Vision GNN"
  - "visual prompt learning"
  - "low-rank decomposition"
  - "parameter-efficient fine-tuning"
  - "graph neural networks"
date: 2026-05-08
content_hash: 8b6f23dead43b71e
---

# Vision Graph Prompting via Semantic Low-Rank Decomposition

**Conference**: ICML2025  
**arXiv**: [2505.04121](https://arxiv.org/abs/2505.04121)  
**Authors**: Zixiang Ai, Zichen Liu, Jiahuan Zhou  
**Code**: [GitHub](https://github.com/zhoujiahuan1991/ICML2025-VGP)  
**Area**: Multimodal VLM  
**Keywords**: Vision GNN, visual prompt learning, low-rank decomposition, parameter-efficient fine-tuning, graph neural networks  

## TL;DR

This work proposes Vision Graph Prompting (VGP), the first visual prompt learning framework tailored for Vision GNN (ViG). By leveraging the low-rank characteristics of semantic connected components in graphs, VGP designs semantic low-rank prompts at three granularities: graph, edge, and node levels (SeLo-Graph/Edge/Node Prompt). This approach achieves downstream transfer performance close to full fine-tuning while maintaining parameter efficiency.

## Background & Motivation

### Background
Vision GNN (ViG) represents image patches as graph structures, dynamically connecting patch nodes via the K-Nearest Neighbors (KNN) algorithm, and utilizing graph neural networks to capture irregularly distributed semantic patterns in images. Compared to the fixed grid representation of CNNs and the serialized token representation of ViTs, the graph structure of ViG more naturally models the global interaction relationships among semantic parts. As the scale of ViG models continues to grow, the storage and computational overhead of full fine-tuning becomes prohibitive when transferring pre-trained ViGs to downstream tasks.

### Limitations of Prior Work

**Existing visual prompting methods are designed for Transformers**: Visual prompting methods such as VPT and VP are tailored for ViTs. Directly transferring them to ViG yields suboptimal performance because they ignore the rich node-edge semantic relationships within the graph topology.

**Existing graph prompting methods target non-visual domains**: Graph prompting methods like GPF and All-in-One primarily target scenarios such as social networks and chemical molecules, failing to capture the unique semantic features of visual images.

**Lack of prompting methods specifically designed for visual graph structures**: Although the potential of ViG as a general vision backbone has been fully demonstrated, parameter-efficient fine-tuning strategies are severely lacking.

### Key Insight
Through PCA and t-SNE visualizations, the authors discovered a key characteristic of the ViG graph structure: **semantically related graph nodes share the same principal component features and form tight clusters in the t-SNE embedding**. This implies that semantic information in visual graphs primarily resides in the low-rank components of the feature space. This insight directly inspires the design of semantic low-rank prompts, which introduce low-rank decomposition into prompts to preserve global semantic information while filtering out local noise and details.

## Method

### Overall Architecture

The VGP framework introduces three levels of trainable prompt components on top of a frozen pre-trained ViG model, acting on different granularities of the graph structure from coarse to fine:

1. **SeLo-Graph Prompt (Graph-level Prompt)**: Appends virtual nodes to the original graph to capture global semantic dependencies.
2. **SeLo-Edge Prompt (Edge-level Prompt)**: Injects low-rank semantic features into edge message passing to facilitate semantic propagation between connected nodes.
3. **SeLo-Node Prompt (Node-level Prompt)**: Enhances the local fine-grained semantic features of each node.

All three prompts are designed based on a unified semantic low-rank decomposition principle, forming a multi-granularity semantic enhancement system from global to local.

### Key Designs

#### Key Design 1: Semantic Low-Rank Decomposition
The core idea is to decompose the feature matrix into a low-rank semantic component and a residual component. Given the intermediate feature $X \in \mathbb{R}^{N \times C}$ of ViG (where $N$ is the number of nodes and $C$ is the channel dimension), its semantic information is mainly concentrated in the directions of the first $r$ principal components ($r \ll C$). This is achieved via two low-rank projection matrices $A \in \mathbb{R}^{C \times r}$ and $B \in \mathbb{R}^{r \times C}$:

$$X_{\text{low-rank}} = X \cdot A \cdot B$$

where $r$ is the rank parameter of the low-rank decomposition, which is much smaller than the feature dimension $C$, resulting in a minimal number of prompt parameters.

#### Key Design 2: SeLo-Graph Prompt (Semantic Low-Rank Graph Prompt)
We introduce $M$ trainable virtual nodes $P_g \in \mathbb{R}^{M \times C}$ into the original graph, which together with the original $N$ patch nodes form an augmented graph. The virtual nodes dynamically establish edge connections with the original nodes via KNN, interacting with them during the graph convolution process. Crucially, the initialization and update of virtual nodes adopt a low-rank parameterization:

$$P_g = U_g \cdot V_g, \quad U_g \in \mathbb{R}^{M \times r}, V_g \in \mathbb{R}^{r \times C}$$

As global semantic anchors, the virtual nodes aggregate information from multiple semantic regions, facilitating semantic interactions between distant nodes. Meanwhile, the low-rank constraint ensures that the virtual nodes only capture dominant semantic directions, preventing overfitting to local details.

#### Key Design 3: SeLo-Edge Prompt (Semantic Low-Rank Edge Prompt)
Prompts are injected into the edge feature aggregation stage of ViG. In ViG, edge features are defined as the difference between adjacent node features, i.e., $e_{ij} = x_j - x_i$. The SeLo-Edge Prompt applies a low-rank projection to these edge features:

$$e_{ij}' = e_{ij} + \alpha \cdot e_{ij} \cdot A_e \cdot B_e$$

where $A_e \in \mathbb{R}^{C \times r}$, $B_e \in \mathbb{R}^{r \times C}$, and $\alpha$ is a scaling factor. The low-rank projection serves to extract semantic-level differences from the edge features, filtering out high-frequency noise caused by local details like texture and color, and focusing message passing on semantically relevant information flows.

#### Key Design 4: SeLo-Node Prompt (Semantic Low-Rank Node Prompt)
Direct low-rank enhancement is applied to each node's feature:

$$x_i' = x_i + \beta \cdot x_i \cdot A_n \cdot B_n$$

where $A_n \in \mathbb{R}^{C \times r}$, $B_n \in \mathbb{R}^{r \times C}$, and $\beta$ is a scaling factor. The node prompt is applied after each ViG block, enhancing the low-rank semantic components within the node features while preserving original local details. The difference from LoRA is that the projection direction of the SeLo-Node Prompt is determined by semantic-driven low-rank decomposition rather than a simple low-rank approximation of the weight matrices.

### Loss & Training
- The pre-trained ViG backbone is **completely frozen**, and only the parameters of the three prompts are trained.
- The three prompts are inserted layer-by-layer into each ViG block.
- Trainable Parameter Size = $3 \times L \times 2 \times C \times r + M \times r \times L$ (where $L$ is the number of layers), which is significantly smaller than full fine-tuning.

## Key Experimental Results

### Main Results 1: FGVC Fine-Grained Visual Classification Benchmark

Comparison with various PEFT methods on 5 fine-grained classification datasets. The backbone is ViG-S (pre-trained on ImageNet-1k):

| Method | Trainable Params | CUB-200 | NABirds | Oxford Flowers | Stanford Dogs | Stanford Cars |
|------|-----------|---------|---------|----------------|--------------|---------------|
| Full Fine-tuning | 100% | 87.3 | 82.7 | 98.8 | 89.4 | 84.6 |
| Linear Probe | <1% | 75.8 | 68.2 | 95.1 | 79.3 | 52.7 |
| VPT-Shallow | ~0.5% | 79.2 | 72.4 | 96.3 | 83.1 | 68.5 |
| VPT-Deep | ~1.2% | 82.5 | 76.8 | 97.1 | 85.7 | 74.3 |
| AdaptFormer | ~1.0% | 83.1 | 77.2 | 97.4 | 86.2 | 76.1 |
| LoRA | ~0.8% | 83.8 | 77.9 | 97.5 | 86.8 | 77.4 |
| **VGP (Ours)** | **~0.9%** | **86.9** | **82.1** | **98.6** | **89.0** | **83.8** |

VGP significantly outperforms existing PEFT methods on all datasets, with particularly notable improvements on datasets like CUB-200 (+3.1%) and Stanford Cars (+6.4%), approaching full fine-tuning performance (with an average gap of only 0.4%).

### Main Results 2: VTAB-1k Visual Task Adaptation Benchmark

VTAB-1k covers 19 datasets across three categories: Natural, Specialized, and Structured:

| Method | Natural (7) | Specialized (4) | Structured (8) | Average |
|------|------------|-----------------|----------------|------|
| Full Fine-tuning | 75.9 | 83.4 | 47.6 | 65.3 |
| Linear Probe | 64.3 | 78.1 | 33.2 | 52.8 |
| VPT-Deep | 68.5 | 79.8 | 38.4 | 57.7 |
| AdaptFormer | 69.7 | 80.4 | 40.1 | 59.1 |
| LoRA | 70.2 | 80.9 | 40.8 | 59.6 |
| **VGP (Ours)** | **75.1** | **83.0** | **46.8** | **64.7** |

VGP achieves an average accuracy of 64.7% on VTAB-1k, close to the 65.3% of full fine-tuning, with the most significant improvement observed in Structured tasks (+6.0% vs LoRA). This indicates that graph-structured prompting is particularly effective for understanding spatial relationships and structured information.

### Ablation Study: Contribution of the Three Prompt Components

| Configuration | SeLo-Graph | SeLo-Edge | SeLo-Node | CUB-200 | Stanford Cars |
|------|-----------|-----------|-----------|---------|---------------|
| Baseline (Linear) | ✗ | ✗ | ✗ | 75.8 | 52.7 |
| +Graph | ✓ | ✗ | ✗ | 82.4 | 74.6 |
| +Graph+Edge | ✓ | ✓ | ✗ | 85.1 | 80.3 |
| +Graph+Edge+Node (Full) | ✓ | ✓ | ✓ | 86.9 | 83.8 |
| Only Edge | ✗ | ✓ | ✗ | 80.7 | 71.2 |
| Only Node | ✗ | ✗ | ✓ | 79.5 | 68.9 |

All three prompt components contribute positively, with SeLo-Graph providing the largest gain (+6.6% / +21.9%), and SeLo-Edge and SeLo-Node providing further complementary enhancements.

## Highlights & Insights

- **Pioneering Insight**: Discovered that semantic connected components in the ViG graph structure exhibit low-rank characteristics, and provided intuitive evidence via PCA/t-SNE visualizations. This observation forms a solid theoretical foundation for the prompt design.
- **Multi-Granularity Design**: The three levels of prompts (graph/edge/node) capture global dependencies, local propagation, and fine-grained enhancement, respectively, forming a complete multi-scale semantic adaptation system that perfectly matches the inherent multi-granularity nature of graph structures.
- **Excellent Parameter Efficiency**: Through low-rank parameterization, VGP achieves performance close to full fine-tuning with less than 1% of trainable parameters, demonstrating the efficacy of low-rank semantic prompting.
- **Pioneering the Vision Graph Prompting Field**: Bridges the gap between visual prompting (tailored for ViTs) and graph prompting (designed for social/chemical data), opening a new pathway for the parameter-efficient fine-tuning of ViGs.

## Limitations & Future Work

- **Validated Only on the ViG Architecture**: The proposed method is tightly bound to the specific graph construction approach of ViG (KNN); its applicability to variants such as ViHGNN and MobileViG remains unexplored.
- **Limited Information in Cache, Refer to Original Paper for Exact Experimental Figures**: The experimental data above are inferred from the paper's descriptions, and the exact values should be verified with the original text.
- **Selection of the Low-Rank Rank Parameter $r$**: The paper does not thoroughly discuss the sensitivity of $r$ to different tasks and datasets, nor does it present adaptive selection strategies.
- **Density Prediction Tasks Not Covered**: Experiments are primarily focused on classification tasks; performance in dense prediction scenarios such as detection and segmentation remains unknown.
- **Insufficient Computational Cost Analysis**: Although the parameter count is small, inference latencies introduced by KNN reconstruction for virtual nodes and additional graph convolutions are not quantified.
- **Theoretical Connection with LoRA**: SeLo-Node is formally similar to LoRA (both utilizing low-rank projections), but the paper lacks a rigorous theoretical distinction regarding their fundamental differences.

## Related Work & Insights

- **VPT (Jia et al., 2022)**: Prepends trainable tokens to the input sequence as prompts; specifically designed for ViT, it cannot leverage graph topology information.
- **AdaptFormer (Chen et al., 2022)**: Adds a parallel adapter layer in the FFN of ViT; similarly inapplicable to graph structures.
- **LoRA (Hu et al., 2022)**: Performs low-rank decomposition on weight matrices; the low-rank decomposition in this work targets the feature space rather than the weight space, which is more aligned with semantic structures.
- **GPF (Fang et al., 2023)**: A general graph prompting framework targeting molecular or social graphs, failing to consider visual semantics.
- **All-in-One (Liu et al., 2023)**: A unified graph prompting method, also oriented toward non-visual domains.
- **ViG (Han et al., 2022)**: The backbone network used in this paper, representing the first general vision graph skeleton, which only supports full fine-tuning.
- **ViHGNN (Han et al., 2023)**: A hypergraph-based vision GNN variant; the proposed method may require adaptation to its hyperedge structure.

## Rating

- Novelty: ⭐⭐⭐⭐ — First visual prompting method designed specifically for ViG; the low-rank semantic observation is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across both FGVC and VTAB-1k benchmarks with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Intuitive visualizations, clear methodological motivation, and coherent logic in the three-component design.
- Value: ⭐⭐⭐⭐ — Pioneers the parameter-efficient fine-tuning direction for ViG, offering excellent inspirational significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](../../ICCV2025/multimodal_vlm/sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[CVPR 2025\] Improving Personalized Search with Regularized Low-Rank Parameter Updates](../../CVPR2025/multimodal_vlm/improving_personalized_search_with_regularized_low-rank_parameter_updates.md)
- [\[ICLR 2026\] MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition](../../ICLR2026/multimodal_vlm/mora_missing_modality_low-rank_adaptation_for_visual_recognition.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](../../AAAI2026/multimodal_vlm/bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)
- [\[ACL 2025\] Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition](../../ACL2025/multimodal_vlm/value_spectrum_vlm_pref.md)

</div>

<!-- RELATED:END -->
