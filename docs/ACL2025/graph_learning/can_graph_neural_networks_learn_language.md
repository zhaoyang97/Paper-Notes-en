---
title: >-
  [Paper Note] Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?
description: >-
  [ACL 2025][Graph Learning][Graph Neural Networks] This paper proposes Morpher, a multimodal prompt learning paradigm. Under extremely weak text supervision (only a few tokens of label names), Morpher aligns a pre-trained GNN into the semantic space of an LLM by simultaneously learning graph prompts and text prompts, enabling cross-task and cross-domain graph classification transfer, as well as the first CLIP-style zero-shot GNN classification prototype.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Multimodal Prompt Learning"
  - "Graph-Text Alignment"
  - "Few-shot Learning"
  - "Zero-shot Classification"
date: 2026-05-08
content_hash: 0cc79bd345e92116
---

# Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?

**Conference**: ACL 2025  
**arXiv**: [2412.08174](https://arxiv.org/abs/2412.08174)  
**Code**: [Violet24K/Morpher](https://github.com/Violet24K/Morpher)  
**Area**: Graph Learning  
**Keywords**: Graph Neural Networks, Multimodal Prompt Learning, Graph-Text Alignment, Few-shot Learning, Zero-shot Classification  

## TL;DR

This paper proposes Morpher, a multimodal prompt learning paradigm. Under extremely weak text supervision (only a few tokens of label names), Morpher aligns a pre-trained GNN into the semantic space of an LLM by simultaneously learning graph prompts and text prompts, enabling cross-task and cross-domain graph classification transfer, as well as the first CLIP-style zero-shot GNN classification prototype.

---

## Background & Motivation

**Background:** CLIP constructs high-quality vision-language alignment models through joint vision-language pre-training, but extending this paradigm to graph data faces significant challenges. Graph data is naturally scarce, text supervision is extremely weak (with label names containing only a few tokens), tasks span three levels (nodes, edges, and graphs), and the same graph structure can express entirely different semantics across different domains.

**Limitations of Prior Work:** (1) Joint graph-text pre-training is only feasible in molecular domains and text-attribute graphs, which is impractical for general graph data due to data scarcity; (2) Existing GNN prompting methods (such as GPF) suffer from cross-connections overwhelming the original graph structure in practical applications, leading to training instability; (3) Prompting only a single modality limits the flexibility of adjusting the other modality.

**Design Motivation:** Leveraging the high-quality semantic space already established by the LLM encoder, dual-modal prompt learning is employed to align graph embeddings to this semantic space while freezing both GNN and LLM parameters.

---

## Method

### Overall Architecture

Morpher consists of three learnable components: graph prompts $\mathbf{P}_\theta^g$, text prompts $\mathbf{P}_\theta^t$, and a cross-modal projector $\text{Proj}_\theta$. The parameters of the GNN and the LLM are frozen entirely, and alignment is achieved solely through the prompts and the projector.

### Key Designs

**1. Improved Graph Prompt Design:** The fundamental issue of existing graph prompting (Sun et al., 2023) is analyzed: since prompt tokens are initialized close to zero vectors, their sigmoid values are close to 0.5, leading to excessively dense cross-connections where the prompt graph's features overwhelm the original graph. Solution: The number of cross-connections is restricted to not exceed the number of original graph edges $n_e$, with each node connecting to at most $\lfloor n_e/a \rfloor$ prompt tokens, and cosine similarity is used instead of sigmoid to calculate connection weights.

**2. Cross-Modal Projector:** A linear layer with tanh activation is used to map the graph embedding space to the text embedding space:
$$\widetilde{\mathbf{v}} = \text{Proj}_\theta(\mathbf{v}) := \tanh(\mathbf{W}\mathbf{v} + \mathbf{b}) \in \mathbb{R}^{1 \times d_t}$$

**3. Text Embedding Normalization:** Considering that a small number of label texts might be close in semantics, the mean $\mu$ is subtracted before applying L2 normalization to separate semantically similar category embeddings.

### Loss & Training

In-batch contrastive loss is adopted for graph-text alignment training:

$$\mathcal{L}_{G \rightarrow T} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathbf{z}_i^{\mathcal{G}} \cdot \mathbf{z}_i^t / \tau)}{\sum_{j=1}^{B} \exp(\mathbf{z}_i^{\mathcal{G}} \cdot \mathbf{z}_j^t / \tau)}$$

During inference, classification is performed by computing the cosine similarity between graph embeddings and text embeddings of each category.

---

## Key Experimental Results

### Main Results (Few-shot Graph Classification)

| Training Method | GNN Pre-training | MUTAG | ENZYMES | PROTEINS | MSRC_21C |
|----------|-----------|-------|---------|----------|----------|
| Supervised | N/A+GCN | 66.00 | 16.67 | 65.89 | 38.85 |
| Pre-train+FT | GraphCL+GCN | 70.00 | 17.91 | 65.89 | 40.00 |
| Graph Prompt | GPF+GCN | 64.67 | 17.02 | 63.50 | 43.46 |
| **Morpher** | **GCN+LLaMA** | **75.33** | **22.39** | **68.32** | **50.86** |

### Ablation Study

| Component | Effect |
|------|------|
| Without modified graph prompts | Unstable training, fail to converge on some datasets |
| Without text prompts | Performance drops by 2-5%, insufficient flexibility in single-modality adjustments |
| Without cross-modal projector | Dimension mismatch makes training impossible |
| Original GPF graph prompts | Overly dense cross-connections lead to performance degradation |

### Key Findings

- **Efficacy of Extremely Weak Supervision:** With only class names (a few tokens) as text supervision, Morpher significantly improves GNN classification performance.
- **Cross-Domain Transfer:** Under cross-domain settings (e.g., molecule $\rightarrow$ social network), Morpher remains highly competitive.
- **Zero-Shot Prototype:** The first CLIP-style zero-shot classification is realized on GNNs—after projecting graph embeddings into the text space, classification can directly utilize unseen class names.
- **Diagnosis of Graph Prompting Issues:** The root cause of excessively dense cross-connections in existing graph prompt designs is revealed, and an effective solution is proposed.

---

## Highlights & Insights

- The first graph-text multimodal prompt learning framework under extremely weak text supervision, where both GNN and LLM parameters are completely frozen.
- Deep analysis and correction of the core flaw of overly dense cross-connections in existing graph prompting designs.
- Implementation of the first GNN CLIP-style zero-shot classification prototype, demonstrating the viability of graph models learning language.
- Superb performance across few-shot, multi-task, and cross-domain settings.

## Limitations & Future Work

- Reliance on the quality of pre-trained GNNs and LLMs; pre-training domain bias of both models may affect performance.
- The cross-modal projector only uses a simple linear layer + tanh, which has limited expressiveness.
- Zero-shot classification was validated only on limited scenarios, and its generalization ability requires further evaluation.
- Experiments were primarily conducted on small-to-medium graph datasets, lacking validation on large-scale industrial graphs.

## Related Work & Insights

- **CLIP and Vision-Language Alignment:** The CLIP framework of Radford et al. (2021) is the core inspiration of this work.
- **Graph Prompt Learning:** GPF (Sun et al., 2023) pioneered the concept of graph prompts, and this work identifies and resolves its design flaws.
- **Graph Self-Supervised Pre-training:** GraphCL (You et al., 2020), GCC (Qiu et al., 2020), etc., provide pre-trained GNNs.
- **Multimodal Prompt Learning:** CoCoOp (Zhou et al., 2022) and MaPLe (Khattak et al., 2023) utilize dual-modality prompts in vision-language tasks.

---

## Rating

| Metric | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [\[ACL 2025\] GraphNarrator: Generating Textual Explanations for Graph Neural Networks](graphnarrator.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](../../NeurIPS2025/graph_learning/dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](../../NeurIPS2025/graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](../../NeurIPS2025/graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
