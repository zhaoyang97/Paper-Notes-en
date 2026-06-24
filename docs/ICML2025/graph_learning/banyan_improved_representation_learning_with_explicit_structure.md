---
title: >-
  [Paper Note] Banyan: Improved Representation Learning with Explicit Structure
description: >-
  [ICML 2025][Graph Learning][Recursive Neural Networks] Banyan introduces two innovations, **entangled hierarchical tree structures** and **diagonalised message passing**. With only 14 non-embedding parameters, it outperforms large-scale Transformer models on semantic textual similarity tasks, offering an efficient and viable alternative for semantic representation learning in low-resource languages.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Recursive Neural Networks"
  - "Hierarchical Structure Learning"
  - "Semantic Representation"
  - "Diagonalised Message Passing"
  - "Entangled Trees"
date: 2026-05-08
content_hash: 2f2817f0c34053b7
---

# Banyan: Improved Representation Learning with Explicit Structure

**Conference**: ICML 2025  
**arXiv**: [2407.17771](https://arxiv.org/abs/2407.17771)  
**Code**: [github.com/exlab-research/Banyan](https://github.com/exlab-research/Banyan)  
**Area**: Graph Learning  
**Keywords**: Recursive Neural Networks, Hierarchical Structure Learning, Semantic Representation, Diagonalised Message Passing, Entangled Trees

## TL;DR

Banyan introduces two innovations, **entangled hierarchical tree structures** and **diagonalised message passing**. With only 14 non-embedding parameters, it outperforms large-scale Transformer models on semantic textual similarity tasks, offering an efficient and viable alternative for semantic representation learning in low-resource languages.

## Background & Motivation

Semantic representation is fundamental to many NLP applications such as RAG, question answering, and summarization. Current mainstream methods rely on large-scale Transformers, requiring massive data and computational power for training. This poses severe challenges in low-resource language scenarios—insufficient data and limited computational resources make scaling approaches unfeasible.

Drawing from linguistics and science, the **principle of compositional semantics** posits that understanding the overall meaning equals understanding the meanings of the parts plus understanding the rules of composition. This principle is inherently efficient—processing new expressions through systematic composition rules rather than storing meanings individually.

Prior work Self-StrAE has initially demonstrated the potential of structured learning, but a gap remains compared to large-scale pre-trained Transformers. This paper proposes the Banyan model to address two core problems:

**Missing Context**: In Self-StrAE, each sentence independently constructs a tree, meaning identical substructures (e.g., "ate doughnuts") cannot share information across different contexts (e.g., "Lisa ate..." vs. "Homer ate...").

**Parameter Redundancy**: In the original linear layer message passing, the number of parameters is proportional to the square of the embedding dimension $U$, causing low efficiency.

## Method

### Overall Architecture

Banyan is essentially a **Recursive Neural Network (RvNN)** that simultaneously learns both structures and representations. The core process is as follows:

1. **Upward Compose**: Starting from leaf nodes (token embeddings), adjacent node pairs with the highest cosine similarity are greedily merged to build a tree structure bottom-up, layer by layer, until the root node embedding is generated.
2. **Entangling**: The independent tree structures of all sentences in a batch are merged into a shared graph, where nodes with identical substructures are unified into a single node to prevent duplication.
3. **Downward Decompose**: Decomposing top-down from the root node, the downward embedding of each node infuses global context by averaging multiple decomposition results from different contexts.
4. **Reconstruction/Prediction**: Reconstructing the original tokens at the leaf nodes, trained via cross-entropy loss.

### Key Designs

#### Entangled Trees

This is the core innovation distinguishing Banyan from Self-StrAE. Self-StrAE constructs an independent binary tree for each sentence without any inter-tree connections. In contrast, Banyan maintains a **global frontier**, placing leaf nodes from all sentences in a batch into a single frontier for merging.

Key steps (Algorithm 1):

- Initialize by placing tokens of all sentences into the global frontier $\mathcal{A}$.
- Iteratively locate the adjacent node pair $(e_{i^\star}, e_{i^\star+1})$ in the frontier with the highest cosine similarity.
- Generate the parent node $e_p$ using the composition function.
- **Crucial**: Check if the composition already exists in the graph using the node identity $s_n$ (span identifier); if it exists, reuse it instead of creating a new one.
- Simultaneously replace this node pair with the parent node at **all occurrences** in the frontier.
- Continue iterating until all sentences are merged.

Three advantages of this approach:

- **Context Aggregation**: Downward embeddings of the same span in different contexts are aggregated by averaging: $\underline{e} = \frac{1}{|\mathcal{Y}|}\sum_y \underline{e}^y$.
- **Eliminating False Negatives**: In contrastive learning, different instances of the same span in Self-StrAE are incorrectly treated as negative samples and pushed apart. Entangled trees naturally avoid this issue through node deduplication.
- **Memory Efficiency**: The number of nodes is heavily reduced after deduplication, significantly lowering memory consumption.

#### Diagonalised Message Passing

The second innovation is simplifying the composition/decomposition functions from linear layers to diagonal operations. The original Self-StrAE uses fully connected linear layers:

$$C_\Phi(\bar{e}_i, \bar{e}_{i+1}) = \text{hcat}(\bar{e}_i, \bar{e}_{i+1})\Phi + \phi, \quad \Phi \in \mathbb{R}^{2U \times U}$$

Banyan replaces this with **element-wise scaling**:

$$C(\bar{e}_i, \bar{e}_{i+1}) = (\bar{e}_i \cdot \sigma(\Phi_l) + \bar{e}_{i+1} \cdot \sigma(\Phi_r)) + \phi$$

$$D(\underline{e}_i) = (\underline{e}_i \cdot \sigma(\Theta_l) + \theta_l, \; \underline{e}_i \cdot \sigma(\Theta_r) + \theta_r)$$

where $\Phi_l, \Phi_r, \phi, \Theta_l, \Theta_r, \theta_l, \theta_r \in \mathbb{R}^U$, and $\sigma$ is the sigmoid function.

Design motivation and advantages:

- **Inspiration**: Drawing inspiration from diagonalisation techniques in the recent revival of linear RNNs (e.g., Mamba, RWKV), which introduce decaying memory along the depth of the structure.
- **Parameter Reduction**: The number of parameters drops from $O(U^2)$ to $O(U)$, a $U$-fold reduction.
- **Preservation of Contraction**: The sigmoid non-linearity ensures numerical stability and enforces memory decay—representation magnitude increases during upward composition (sum of two child nodes) and contracts during downward decomposition back to the leaf-node level.
- **Space Alignment**: The encoder and decoder embeddings are constrained to the same space, making amortised learning a prerequisite for successful reconstruction.

#### Embedding Dimension Configuration

Self-StrAE treats a 256-dimensional embedding as a $16 \times 16$ square matrix ($K=16, U=16$), where composition/decomposition functions act independently on $K$ channels. Banyan changes this configuration to $K=128, U=2$—**maximizing the number of independent channels** while reducing the per-channel dimension to an extreme minimum. This results in a total of only **14** non-embedding parameters ($7 \times U = 7 \times 2$).

### Loss & Training

**Training Objective**: Since diagonalised functions constrain encoding and decoding embeddings to the same space, Banyan can be trained directly using cross-entropy loss without requiring the contrastive loss in Self-StrAE:

$$\mathcal{L}_{\text{CE}}(\mathbf{w}, \hat{\mathbf{w}}) = -\frac{1}{N}\sum_{n=1}^N w_n \cdot \log \hat{w}_n$$

**Training Data**: It is trained on an English Wikipedia subset of only about 10 million tokens (simulating a low-resource setting), which is far less than the 100 million tokens used by RoBERTa.

**Batch Entanglement Estimation**: Ideally, entangled trees should be constructed over the entire corpus, but exponential growth in data makes this intractable. Banyan constructs entangled trees independently on each batch as an estimation, which is unbiased—the batch average is an unbiased estimator of the population average.

## Key Experimental Results

### Main Results

**English Sentence-Level Semantic Textual Similarity (Spearman's $\rho \times 100$):**

| Model | STS-12 | STS-13 | STS-14 | STS-15 | STS-B | SICK-R | Average Score |
|------|--------|--------|--------|--------|-------|--------|-----------|
| Self-StrAE | 31.98 | 53.88 | 37.73 | 55.23 | 39.53 | 51.78 | 46.59 |
| GloVe+stopword rm | 39.00 | 41.61 | 39.31 | 51.06 | 48.40 | 52.80 | 44.96 |
| Sent2Vec | 38.14 | 51.37 | 48.64 | 67.28 | 53.39 | 59.67 | 53.28 |
| RoBERTa+SimCSE | 50.63 | 62.23 | 54.17 | 68.77 | 53.53 | 56.87 | 59.02 |
| **Banyan** | **51.38** | **69.60** | **63.20** | **73.08** | **61.90** | **55.23** | **62.97** |

**English Retrieval and Classification Tasks**:

| Model | Quora NDCG@10 | Arguana NDCG@10 | SST-2 Acc | MRPC F1 |
|------|--------------|----------------|-----------|---------|
| Self-StrAE | 40.02 | 15.48 | 74.67 | 80.34 |
| RoBERTa+SimCSE | 59.30 | 21.84 | 75.97 | 80.83 |
| **Banyan** | **65.71** | **28.28** | 75.96 | 79.48 |

### Ablation Study

| Configuration | Description | Effect |
|------|------|------|
| Self-StrAE Baseline | Independent trees + fully connected layers + contrastive loss | Score 46.59 |
| + Entangled Trees | Replaced with global entangled graph structures | Significant improvement (eliminates false negatives, shares context) |
| + Diagonalised Message Passing | Diagonal scaling replaces linear layers | Further improvement (parameters reduced from 1,072 to 14) |
| + CE loss replacing contrastive loss | Switch objective using space alignment | More stable training |
| K=128, U=2 configuration | Maximizes channel count, minimizes channel dimension | Best performance |

**Model Scale Comparison:**

| Model | Non-embedding Parameters |
|------|------------|
| **Banyan** | **14** |
| Self-StrAE | 1,072 |
| RoBERTa (M) | ~10M |
| MiniLM-L12 | ~21M |
| XLM-R | ~85M |
| Llama 3.1 | ~8B |
| Mistral Nemo | ~12B |

### Key Findings

1. **Extreme Parameter Efficiency**: 14 non-embedding parameters outperform Transformers with millions or even billions of parameters. On STS tasks, Banyan (62.97) > RoBERTa+SimCSE (59.02), a parameter size difference of over 5 orders of magnitude.
2. **Cross-level Semantic Transfer**: Unlike GloVe/Sent2Vec, structured models can seamlessly transfer word-level semantics to sentence-level semantics—demonstrating the true advantage of compositional semantics.
3. **Significant Advantage in Retrieval Tasks**: On Quora retrieval, Banyan achieves an NDCG@10 of 65.71, surpassing SimCSE RoBERTa's 59.30, indicating that structured representations hold practical value in key RAG applications.
4. **Outstanding Performance in Multilingual Low-Resource Scenarios**: In the SemRel evaluation across 9 languages, Banyan achieves an average score of 60.01, surpassing large models such as XLM-R (46.61) and Llama 3.1 8B (53.06). Its advantage is particularly pronounced in extremely low-resource languages like Hausa (49.68 vs. 4.1).

## Highlights & Insights

- **Counter-intuitive Minimalism**: In an era where deep learning pursues increasingly larger scales, Banyan beats models with billions of parameters using only 14 parameters, proving that a correct inductive bias is more valuable than brute-force scaling.
- **Elegant Design of Entangled Trees**: Utilizing batch-level shared graph structures kills three birds with one stone—fusing context, eliminating false negatives, and reducing memory usage.
- **Profound Impact of Diagonalisation**: Adapting the concepts of linear RNNs to tree structures not only reduces parameters but also boosts performance, suggesting that "memory decay along structure" is a universally beneficial inductive bias.
- **Metaphorical Name**: The model is named Banyan because banyan trees have multiple roots (corresponding to the multi-root structure of the entangled graph) and nodes can be shared across branches—combining poetic intuition with technical suitability.

## Limitations & Future Work

1. **Weakness in Word-Level Similarity**: Performance on SimLex-999 is weak (14.65) because the model needs to simultaneously model similarity and relatedness, while SimLex strictly excludes relatedness.
2. **No Advantage in Classification Tasks**: Banyan performs on par with baselines on SST-2 and MRPC, suggesting that classification may depend more on parameterized top classification layers than on representation quality itself.
3. **Greedy Merging Strategy**: Greedy merging based on cosine similarity can yield sub-optimal tree structures; future work could explore more global structural optimization methods.
4. **Approximation of Batch Entanglement**: Although unbiased, the batch-level entangled tree is only an approximation of full-dataset entanglement. Nominally, larger batch sizes yield better results but are constrained by memory.
5. **NLP Only**: It is currently only validated on textual semantic similarity tasks. Its applicability to domains like vision or multi-modality has not been explored.

## Related Work & Insights

- **Self-StrAE** (Opper et al., 2023): The direct predecessor of Banyan, which provides an automatic structure induction method based on representation similarity.
- **The Revival of Linear RNNs** (Mamba, RWKV, etc.): The source of the diagonalised memory decay concept, proving that simplification does not entail performance degradation.
- **SimCSE** (Gao et al., 2021): An representative of contrastive learning to enhance representation quality; Banyan outperforms it without using contrastive learning.
- **SemRel Dataset** (Ousidhoum et al., 2024): Provides a standardized test suite for evaluating low-resource languages.
- **Insights**: This work reveals that in graph learning and structured representation domains, carefully designed inductive biases can be more effective than scaling. The concept of entangled graphs can also be extended to other scenarios requiring cross-instance structure sharing.

## Rating

| Dimension | Score (1-10) | Description |
|---|---|---|
| Novelty | 9 | Both entangled trees and diagonalised message passing are original and elegant designs |
| Technical Depth | 8 | Solid theoretical analysis with clear proof of unbiased estimation |
| Experimental Thoroughness | 8 | Multi-dimensional English evaluation + 9-language cross-lingual validation with comprehensive ablations |
| Practical Value | 8 | Directly applicable to low-resource language scenarios, extremely easy to deploy with 14 parameters |
| Writing Quality | 8 | Clear structure, intuitive diagrams, and clever naming of Banyan |
| **Overall** | **8.2** | A striking piece of work challenging the scaling law with an extremely minimalist solution |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] NN-Former: Rethinking Graph Structure in Neural Architecture Representation](../../CVPR2025/graph_learning/nn-former_rethinking_graph_structure_in_neural_architecture_representation.md)
- [\[ICML 2025\] TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration](toping_topologically_interpretable_graph_learning_via_persistent_rationale_filtr.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](../../ICLR2026/graph_learning/towards_improved_sentence_representations_using_token_graphs.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/graph_learning/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](../../ICML2026/graph_learning/t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)

</div>

<!-- RELATED:END -->
